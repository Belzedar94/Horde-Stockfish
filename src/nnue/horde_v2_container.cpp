/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "horde_v2_container.h"

#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

#include "../data/sha256.h"

namespace Stockfish::Eval::NNUE::HordeV2 {
namespace {

using Byte = std::uint8_t;

constexpr std::array<Byte, 8> Magic                 = {'H', 'S', 'V', '2', 'I', 'N', 'T', 0};
constexpr std::uint16_t       FormatVersion         = 1;
constexpr std::uint32_t       EndianTag             = 0x01020304;
constexpr std::size_t         DirectoryOffset       = 640;
constexpr std::size_t         DirectoryEntryBytes   = 64;
constexpr std::size_t         SectionCount          = 10;
constexpr std::size_t         MaximumContainerBytes = 16 * 1024 * 1024;

constexpr std::string_view ContainerSchemaName = "HORDE_V2_INTEGER_NETWORK_V1";
constexpr std::string_view RoyalSchemaName     = "V2_BASE_P0_64X192";
constexpr std::string_view AbsoluteSchemaName  = "V2_C1_ABS_NONKING_64X192";
constexpr std::string_view RoyalRank8SchemaName = "V2_C1_ROYAL_RANK8_64X192";
constexpr std::string_view RoyalStructureSha =
  "7456B4C43FCF9CCCCB57C465D92708975BC131058619EB58D7EB239BCA42F9D1";
constexpr std::string_view AbsoluteStructureSha =
  "4FC702F66B3B1BB12105B67C05FFFC73B9B17E9B33EF09EBF9BEE04570AFE21A";
constexpr std::string_view RoyalRank8StructureSha =
  "D99A6CBE80F94ED45DA5888193474C590FDDFBD2A458D9F3516CE1EFCC0B135F";
constexpr std::string_view RoyalTrainingStructureSha =
  "5360985A5B08E43A6F7C23E8601DF159BAEE38E9306C3652867AA93EEAD39862";
constexpr std::string_view AbsoluteTrainingStructureSha =
  "66538DB76A0248662A824545A2ECCD2AE84CD4805DDFCA206B5239FA3BDE45B6";
constexpr std::string_view RoyalRank8TrainingStructureSha =
  "53A82F734EBCAD97508AD91D54027ADD10772A1DC85A612F5D395AEE08567083";

enum class DType : Byte {
    I8  = 1,
    I16 = 2,
    I32 = 3,
};

struct SchemaSpec {
    ContainerSchema  schema;
    FirstDomain      firstDomain;
    std::size_t      firstRows;
    std::string_view name;
    std::string_view structureSha;
    std::string_view trainingStructureSha;
    std::size_t      parameterBytes;
};

constexpr SchemaSpec RoyalSpec{ContainerSchema::ROYAL_64X192,
                               FirstDomain::ROYAL,
                               RoyalPieceSquareDimensions,
                               RoyalSchemaName,
                               RoyalStructureSha,
                               RoyalTrainingStructureSha,
                               Width64x192::ParameterBytes};
constexpr SchemaSpec AbsoluteSpec{ContainerSchema::ABS_NONKING_64X192,
                                  FirstDomain::ABSOLUTE_NONKING,
                                  RoyalNonKingRoleCount* SQUARE_NB,
                                  AbsoluteSchemaName,
                                  AbsoluteStructureSha,
                                  AbsoluteTrainingStructureSha,
                                  362824};
constexpr SchemaSpec RoyalRank8Spec{ContainerSchema::ROYAL_RANK8_64X192,
                                    FirstDomain::ROYAL_RANK8,
                                    RoyalRankPieceSquareDimensions,
                                    RoyalRank8SchemaName,
                                    RoyalRank8StructureSha,
                                    RoyalRank8TrainingStructureSha,
                                    936264};

struct SectionSpec {
    std::uint16_t id;
    DType         dtype;
    Byte          rank;
    std::uint32_t first;
    std::uint32_t second;
    std::size_t   bytes;
};

constexpr std::array<SectionSpec, SectionCount> section_specs(const SchemaSpec& spec) {
    return {{{1, DType::I16, 2, std::uint32_t(spec.firstRows), 64, spec.firstRows * 64 * 2},
             {2, DType::I32, 1, 64, 0, 64 * 4},
             {3, DType::I16, 2, 704, 192, 704 * 192 * 2},
             {4, DType::I32, 1, 192, 0, 192 * 4},
             {5, DType::I8, 2, 32, 256, 32 * 256},
             {6, DType::I32, 1, 32, 0, 32 * 4},
             {7, DType::I8, 2, 32, 32, 32 * 32},
             {8, DType::I32, 1, 32, 0, 32 * 4},
             {9, DType::I8, 2, 2, 32, 2 * 32},
             {10, DType::I32, 1, 2, 0, 2 * 4}}};
}

const SchemaSpec* schema_spec(std::uint32_t value) noexcept {
    if (value == std::uint32_t(ContainerSchema::ROYAL_64X192))
        return &RoyalSpec;
    if (value == std::uint32_t(ContainerSchema::ABS_NONKING_64X192))
        return &AbsoluteSpec;
    if (value == std::uint32_t(ContainerSchema::ROYAL_RANK8_64X192))
        return &RoyalRank8Spec;
    return nullptr;
}

ContainerLoadResult failure(ContainerLoadError error, std::string message) {
    ContainerLoadResult result;
    result.error   = error;
    result.message = std::move(message);
    return result;
}

std::uint16_t read_u16(const Byte* data) noexcept {
    return std::uint16_t(data[0]) | (std::uint16_t(data[1]) << 8);
}

std::uint32_t read_u32(const Byte* data) noexcept {
    return std::uint32_t(data[0]) | (std::uint32_t(data[1]) << 8) | (std::uint32_t(data[2]) << 16)
         | (std::uint32_t(data[3]) << 24);
}

std::uint64_t read_u64(const Byte* data) noexcept {
    std::uint64_t value = 0;
    for (int index = 7; index >= 0; --index)
        value = (value << 8) | data[index];
    return value;
}

std::int16_t read_i16(const Byte* data) noexcept {
    const std::uint16_t raw   = read_u16(data);
    const std::int32_t  value = raw <= std::uint16_t(std::numeric_limits<std::int16_t>::max())
                                ? std::int32_t(raw)
                                : std::int32_t(raw) - 65536;
    return std::int16_t(value);
}

std::int32_t read_i32(const Byte* data) noexcept {
    const std::uint32_t raw   = read_u32(data);
    const std::int64_t  value = raw <= std::uint32_t(std::numeric_limits<std::int32_t>::max())
                                ? std::int64_t(raw)
                                : std::int64_t(raw) - (std::int64_t(1) << 32);
    return std::int32_t(value);
}

std::int8_t read_i8(Byte byte) noexcept {
    return std::int8_t(byte <= 127 ? int(byte) : int(byte) - 256);
}

bool all_zero(const Byte* begin, const Byte* end) noexcept {
    return std::all_of(begin, end, [](Byte byte) { return byte == 0; });
}

std::string digest_hex(const Byte* bytes, std::size_t size) {
    Data::Sha256 hash;
    hash.update(bytes, size);
    return Data::sha256_hex_upper(hash.digest());
}

std::string bytes_hex(const Byte* bytes, std::size_t size, bool uppercase = true) {
    constexpr char Upper[] = "0123456789ABCDEF";
    constexpr char Lower[] = "0123456789abcdef";
    const char*    digits  = uppercase ? Upper : Lower;
    std::string    result;
    result.reserve(size * 2);
    for (std::size_t index = 0; index < size; ++index)
    {
        result.push_back(digits[bytes[index] >> 4]);
        result.push_back(digits[bytes[index] & 15]);
    }
    return result;
}

bool matches_hex(const Byte* bytes, std::string_view expected) {
    return bytes_hex(bytes, expected.size() / 2) == expected;
}

bool nonzero_digest(const Byte* bytes) noexcept { return !all_zero(bytes, bytes + 32); }

constexpr bool
range_within(std::uint64_t offset, std::uint64_t bytes, std::uint64_t containerBytes) noexcept {
    return offset <= containerBytes && bytes <= containerBytes - offset;
}

std::string expected_provenance(const Byte* header) {
    std::ostringstream json;
    json << "{\"checkpoint_sha256\":\"" << bytes_hex(header + 272, 32)
         << "\",\"container_schema\":\"" << ContainerSchemaName << "\",\"source_commit\":\""
         << bytes_hex(header + 432, 20, false)
         << "\",\"source_dirty\":false,\"train_file_sha256\":\"" << bytes_hex(header + 336, 32)
         << "\",\"training_architecture_structural_sha256\":\"" << bytes_hex(header + 240, 32)
         << "\",\"training_receipt_sha256\":\"" << bytes_hex(header + 304, 32)
         << "\",\"validation_file_sha256\":\"" << bytes_hex(header + 368, 32)
         << "\",\"wdl_calibration_sha256\":\"" << bytes_hex(header + 400, 32) << "\"}";
    return json.str();
}

template<typename T, std::size_t Size>
void decode_i32_array(const Byte* source, std::array<T, Size>& destination) {
    static_assert(sizeof(T) == sizeof(std::int32_t));
    for (std::size_t index = 0; index < Size; ++index)
        destination[index] = T(read_i32(source + index * 4));
}

void decode_i16_buffer(const Byte* source, AlignedBuffer<FtWeight>& destination) {
    for (std::size_t index = 0; index < destination.size(); ++index)
        destination[index] = FtWeight(read_i16(source + index * 2));
}

void decode_i8_buffer(const Byte* source, AlignedBuffer<DenseWeight>& destination) {
    for (std::size_t index = 0; index < destination.size(); ++index)
        destination[index] = DenseWeight(read_i8(source[index]));
}

}  // namespace

const char* container_load_error_name(ContainerLoadError error) noexcept {
    switch (error)
    {
    case ContainerLoadError::NONE :
        return "NONE";
    case ContainerLoadError::OPEN_FAILED :
        return "OPEN_FAILED";
    case ContainerLoadError::READ_FAILED :
        return "READ_FAILED";
    case ContainerLoadError::TRUNCATED :
        return "TRUNCATED";
    case ContainerLoadError::MAGIC_MISMATCH :
        return "MAGIC_MISMATCH";
    case ContainerLoadError::HEADER_MISMATCH :
        return "HEADER_MISMATCH";
    case ContainerLoadError::SCHEMA_MISMATCH :
        return "SCHEMA_MISMATCH";
    case ContainerLoadError::STRUCTURE_MISMATCH :
        return "STRUCTURE_MISMATCH";
    case ContainerLoadError::PROVENANCE_MISMATCH :
        return "PROVENANCE_MISMATCH";
    case ContainerLoadError::DIRECTORY_MISMATCH :
        return "DIRECTORY_MISMATCH";
    case ContainerLoadError::PAYLOAD_MISMATCH :
        return "PAYLOAD_MISMATCH";
    case ContainerLoadError::PARAMETER_RANGE :
        return "PARAMETER_RANGE";
    }
    return "UNKNOWN";
}

ContainerLoadResult load_integer_container(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        return failure(ContainerLoadError::OPEN_FAILED, "cannot open Horde V2 network");
    const std::streamoff length = input.tellg();
    if (length < std::streamoff(V2ContainerHeaderBytes))
        return failure(ContainerLoadError::TRUNCATED, "network is shorter than the V2 header");
    if (length > std::streamoff(MaximumContainerBytes))
        return failure(ContainerLoadError::HEADER_MISMATCH, "network exceeds the V2 size ceiling");
    input.seekg(0);
    std::vector<Byte> file(static_cast<std::size_t>(length));
    input.read(reinterpret_cast<char*>(file.data()), static_cast<std::streamsize>(length));
    if (!input || input.gcount() != length)
        return failure(ContainerLoadError::READ_FAILED, "cannot read complete Horde V2 network");

    const Byte* header = file.data();
    if (!std::equal(Magic.begin(), Magic.end(), header))
        return failure(ContainerLoadError::MAGIC_MISMATCH, "network is not a Horde V2 container");
    if (read_u16(header + 8) != FormatVersion || read_u16(header + 10) != V2ContainerHeaderBytes
        || read_u32(header + 12) != EndianTag || read_u16(header + 20) != SectionCount
        || read_u16(header + 22) != 0 || read_u64(header + 24) != file.size())
        return failure(ContainerLoadError::HEADER_MISMATCH, "fixed V2 header fields differ");

    const SchemaSpec* spec = schema_spec(read_u32(header + 16));
    if (!spec)
        return failure(ContainerLoadError::SCHEMA_MISMATCH, "unregistered V2 schema id");
    const auto schemaEnd = std::find(header + 176, header + 240, Byte(0));
    if (schemaEnd == header + 240
        || std::string_view(reinterpret_cast<const char*>(header + 176),
                            std::size_t(schemaEnd - (header + 176)))
             != spec->name
        || !all_zero(schemaEnd, header + 240))
        return failure(ContainerLoadError::SCHEMA_MISMATCH, "V2 schema name or padding differs");

    const std::uint64_t structureOffset  = read_u64(header + 32);
    const std::uint64_t structureBytes   = read_u64(header + 40);
    const std::uint64_t provenanceOffset = read_u64(header + 80);
    const std::uint64_t provenanceBytes  = read_u64(header + 88);
    const std::uint64_t parameterOffset  = read_u64(header + 128);
    const std::uint64_t parameterBytes   = read_u64(header + 136);
    const std::uint64_t fileBytes        = file.size();
    if (structureOffset != V2ContainerHeaderBytes
        || !range_within(structureOffset, structureBytes, fileBytes)
        || provenanceOffset != structureOffset + structureBytes
        || !range_within(provenanceOffset, provenanceBytes, fileBytes)
        || parameterOffset != provenanceOffset + provenanceBytes
        || !range_within(parameterOffset, parameterBytes, fileBytes)
        || parameterOffset + parameterBytes != fileBytes || parameterBytes != spec->parameterBytes)
        return failure(ContainerLoadError::HEADER_MISMATCH, "V2 section offsets or lengths differ");

    if (!matches_hex(header + 48, spec->structureSha)
        || digest_hex(file.data() + structureOffset, std::size_t(structureBytes))
             != spec->structureSha)
        return failure(ContainerLoadError::STRUCTURE_MISMATCH,
                       "V2 structural descriptor identity differs");
    if (!matches_hex(header + 240, spec->trainingStructureSha))
        return failure(ContainerLoadError::STRUCTURE_MISMATCH,
                       "training architecture structural identity differs");

    for (const std::size_t offset : {272U, 304U, 336U, 368U, 400U})
        if (!nonzero_digest(header + offset))
            return failure(ContainerLoadError::PROVENANCE_MISMATCH,
                           "V2 provenance contains a zero identity");
    if (all_zero(header + 432, header + 452) || header[452] != 0)
        return failure(ContainerLoadError::PROVENANCE_MISMATCH,
                       "V2 source identity is zero or dirty");
    const std::string provenance(reinterpret_cast<const char*>(file.data() + provenanceOffset),
                                 std::size_t(provenanceBytes));
    if (digest_hex(file.data() + provenanceOffset, std::size_t(provenanceBytes))
          != bytes_hex(header + 96, 32)
        || provenance != expected_provenance(header))
        return failure(ContainerLoadError::PROVENANCE_MISMATCH,
                       "V2 provenance payload or identity differs");

    if (header[453] != 1 || header[454] != FtActivationShift || header[455] != DenseActivationShift
        || read_u32(header + 456) != std::uint32_t(spec->firstDomain)
        || read_u32(header + 460) != spec->firstRows || read_u32(header + 464) != 64
        || read_u32(header + 468) != FixedRolePieceSquareDimensions || read_u32(header + 472) != 192
        || read_u32(header + 476) != 32 || read_u32(header + 480) != 32
        || read_u32(header + 484) != 2 || read_u32(header + 488) != 1
        || read_u32(header + 492) != 8128 || read_u32(header + 496) != 64
        || read_i32(header + 500) != 0 || read_i32(header + 504) != ActivationMax
        || read_i32(header + 508) != ScalarOutputScale
        || read_i32(header + 512) != MaxSafeBiasMagnitude || read_u32(header + 516) != 1
        || read_u32(header + 520) != 1 || read_u32(header + 524) != DirectoryOffset
        || read_u32(header + 528) != DirectoryEntryBytes
        || !all_zero(header + 532, header + DirectoryOffset))
        return failure(ContainerLoadError::SCHEMA_MISMATCH,
                       "V2 topology or integer contract differs");

    const Byte* parameterPayload = file.data() + parameterOffset;
    if (digest_hex(parameterPayload, std::size_t(parameterBytes)) != bytes_hex(header + 144, 32))
        return failure(ContainerLoadError::PAYLOAD_MISMATCH,
                       "V2 parameter payload SHA-256 differs");

    const auto                            sectionSpecs = section_specs(*spec);
    std::array<const Byte*, SectionCount> sectionData{};
    std::size_t                           expectedOffset = 0;
    for (std::size_t index = 0; index < SectionCount; ++index)
    {
        const Byte*         entry    = header + DirectoryOffset + index * DirectoryEntryBytes;
        const auto&         expected = sectionSpecs[index];
        const std::uint64_t offset   = read_u64(entry + 12);
        const std::uint64_t bytes    = read_u64(entry + 20);
        if (read_u16(entry) != expected.id || entry[2] != Byte(expected.dtype)
            || entry[3] != expected.rank || read_u32(entry + 4) != expected.first
            || read_u32(entry + 8) != expected.second || offset != expectedOffset
            || bytes != expected.bytes || read_u32(entry + 60) != 0
            || !range_within(offset, bytes, parameterBytes))
            return failure(ContainerLoadError::DIRECTORY_MISMATCH,
                           "V2 parameter directory differs");
        sectionData[index] = parameterPayload + offset;
        if (digest_hex(sectionData[index], std::size_t(bytes)) != bytes_hex(entry + 28, 32))
            return failure(ContainerLoadError::PAYLOAD_MISMATCH,
                           "V2 parameter section SHA-256 differs");
        expectedOffset += std::size_t(bytes);
    }
    const std::size_t directoryEnd = DirectoryOffset + SectionCount * DirectoryEntryBytes;
    if (expectedOffset != parameterBytes
        || !all_zero(header + directoryEnd, header + V2ContainerHeaderBytes))
        return failure(ContainerLoadError::DIRECTORY_MISMATCH,
                       "V2 directory does not exactly cover the payload");

    ContainerParameters parameters(spec->firstRows);
    parameters.schema          = spec->schema;
    parameters.firstDomain     = spec->firstDomain;
    parameters.schemaName      = std::string(spec->name);
    parameters.parameterSha256 = bytes_hex(header + 144, 32);
    parameters.fileSha256      = digest_hex(file.data(), file.size());
    decode_i16_buffer(sectionData[0], parameters.firstWeights);
    decode_i32_array(sectionData[1], parameters.firstBias);
    decode_i16_buffer(sectionData[2], parameters.globalWeights);
    decode_i32_array(sectionData[3], parameters.globalBias);
    decode_i8_buffer(sectionData[4], parameters.hidden0Weights);
    decode_i32_array(sectionData[5], parameters.hidden0Bias);
    decode_i8_buffer(sectionData[6], parameters.hidden1Weights);
    decode_i32_array(sectionData[7], parameters.hidden1Bias);
    decode_i8_buffer(sectionData[8], parameters.outputWeights);
    decode_i32_array(sectionData[9], parameters.outputBias);
    if (!parameters.valid())
        return failure(ContainerLoadError::PARAMETER_RANGE,
                       "V2 decoded parameters violate their registered bounds");

    ContainerLoadResult result;
    result.parameters = std::move(parameters);
    return result;
}

static_assert(RoyalSpec.parameterBytes == 2902344);
static_assert(AbsoluteSpec.parameterBytes == 362824);
static_assert(RoyalRank8Spec.parameterBytes == 936264);

}  // namespace Stockfish::Eval::NNUE::HordeV2
