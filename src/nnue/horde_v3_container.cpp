/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "horde_v3_container.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include "../data/sha256.h"

namespace Stockfish::Eval::NNUE::HordeV3 {
namespace {

using Byte = std::uint8_t;

constexpr std::array<Byte, 8> Magic               = {'H', 'S', 'V', '2', 'I', 'N', 'T', 0};
constexpr std::uint16_t       FormatVersion       = 1;
constexpr std::uint32_t       EndianTag           = 0x01020304;
constexpr std::size_t         DirectoryOffset     = 640;
constexpr std::size_t         DirectoryEntryBytes = 64;
constexpr std::size_t         MaximumContainerBytes = 16 * 1024 * 1024;

constexpr std::string_view ContainerSchemaName = "HORDE_V2_INTEGER_NETWORK_V1";
constexpr std::string_view NetworkSchemaName   = "V3_G1024_PAWN_WPC8";

// SHA-256 of the canonical structural descriptor emitted by
// tools/horde_v3_container.py. Any byte of that descriptor changing changes
// this digest, so the reader is pinned to the registered V3 schema.
constexpr std::string_view StructureSha =
  "5FD8D6DE6B375312C1610597260528150A1D49A6C54C4A8FB1B1E8A6A78A854C";

// The V3 header extension block. Offsets 532 through 575 are zero in every V2
// container, so a V2 network can never be mistaken for a V3 one.
constexpr std::uint32_t OutputWeightScaleNumerator   = 9600;
constexpr std::uint32_t OutputWeightScaleDenominator = 127;
constexpr std::uint32_t OutputBiasScale              = 9600;
constexpr std::uint32_t PsqtScale                    = 9600;
constexpr std::uint32_t ActiveRowBound               = 100;
constexpr std::uint32_t NetworkToScoreValue          = 600;
constexpr std::uint32_t HiddenBiasScale              = 8128;
constexpr std::uint32_t FeatureScale                 = 8128;
constexpr std::uint32_t ContextualBlockCount         = 6;
constexpr std::uint32_t FirstDomainV3Stream          = 4;

enum class DType : Byte {
    I8  = 1,
    I16 = 2,
    I32 = 3,
};

struct SectionSpec {
    std::uint16_t id;
    DType         dtype;
    Byte          rank;
    std::uint32_t first;
    std::uint32_t second;
    std::size_t   bytes;
};

constexpr std::array<SectionSpec, V3ContainerSectionCount> Sections = {{
  {1, DType::I16, 2, 896, 1024, 896 * 1024 * 2},
  {2, DType::I32, 1, 1024, 0, 1024 * 4},
  {3, DType::I32, 2, 896, 16, 896 * 16 * 4},
  {4, DType::I8, 2, 128, 1024, 128 * 1024},
  {5, DType::I32, 1, 128, 0, 128 * 4},
  {6, DType::I8, 2, 256, 16, 256 * 16},
  {7, DType::I32, 1, 256, 0, 256 * 4},
  {8, DType::I8, 2, 16, 32, 16 * 32},
  {9, DType::I32, 1, 16, 0, 16 * 4},
  {10, DType::I8, 1, 37, 0, 37},
}};

constexpr std::size_t total_section_bytes() {
    std::size_t total = 0;
    for (const SectionSpec& section : Sections)
        total += section.bytes;
    return total;
}

static_assert(total_section_bytes() == V3ContainerParameterBytes);

V3ContainerLoadResult failure(V3ContainerLoadError error, std::string message) {
    V3ContainerLoadResult result;
    result.error   = error;
    result.message = std::move(message);
    return result;
}

std::uint16_t read_u16(const Byte* data) noexcept {
    return std::uint16_t(std::uint16_t(data[0]) | (std::uint16_t(data[1]) << 8));
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

// The canonical provenance document the header commits to, rebuilt field by
// field. Keys are sorted and the separators are compact, exactly as
// canonical_json() writes them.
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

}  // namespace

const char* v3_container_load_error_name(V3ContainerLoadError error) noexcept {
    switch (error)
    {
    case V3ContainerLoadError::NONE :
        return "NONE";
    case V3ContainerLoadError::OPEN_FAILED :
        return "OPEN_FAILED";
    case V3ContainerLoadError::READ_FAILED :
        return "READ_FAILED";
    case V3ContainerLoadError::TRUNCATED :
        return "TRUNCATED";
    case V3ContainerLoadError::MAGIC_MISMATCH :
        return "MAGIC_MISMATCH";
    case V3ContainerLoadError::HEADER_MISMATCH :
        return "HEADER_MISMATCH";
    case V3ContainerLoadError::SCHEMA_MISMATCH :
        return "SCHEMA_MISMATCH";
    case V3ContainerLoadError::STRUCTURE_MISMATCH :
        return "STRUCTURE_MISMATCH";
    case V3ContainerLoadError::PROVENANCE_MISMATCH :
        return "PROVENANCE_MISMATCH";
    case V3ContainerLoadError::DIRECTORY_MISMATCH :
        return "DIRECTORY_MISMATCH";
    case V3ContainerLoadError::PAYLOAD_MISMATCH :
        return "PAYLOAD_MISMATCH";
    case V3ContainerLoadError::PARAMETER_RANGE :
        return "PARAMETER_RANGE";
    case V3ContainerLoadError::PHASE_LOOKUP :
        return "PHASE_LOOKUP";
    }
    return "UNKNOWN";
}

V3ContainerLoadResult load_v3_container(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        return failure(V3ContainerLoadError::OPEN_FAILED, "cannot open Horde V3 network");
    const std::streamoff length = input.tellg();
    if (length < std::streamoff(V3ContainerHeaderBytes))
        return failure(V3ContainerLoadError::TRUNCATED, "network is shorter than the V3 header");
    if (length > std::streamoff(MaximumContainerBytes))
        return failure(V3ContainerLoadError::HEADER_MISMATCH,
                       "network exceeds the registered size ceiling");
    input.seekg(0);
    std::vector<Byte> file(static_cast<std::size_t>(length));
    input.read(reinterpret_cast<char*>(file.data()), static_cast<std::streamsize>(length));
    if (!input || input.gcount() != length)
        return failure(V3ContainerLoadError::READ_FAILED, "cannot read complete Horde V3 network");

    const Byte* header = file.data();
    if (!std::equal(Magic.begin(), Magic.end(), header))
        return failure(V3ContainerLoadError::MAGIC_MISMATCH, "network is not an HSV2INT container");
    if (read_u16(header + 8) != FormatVersion || read_u16(header + 10) != V3ContainerHeaderBytes
        || read_u32(header + 12) != EndianTag
        || read_u16(header + 20) != V3ContainerSectionCount || read_u16(header + 22) != 0
        || read_u64(header + 24) != file.size())
        return failure(V3ContainerLoadError::HEADER_MISMATCH, "fixed V3 header fields differ");

    if (read_u32(header + 16) != V3NetworkSchemaId)
        return failure(V3ContainerLoadError::SCHEMA_MISMATCH, "unregistered V3 network schema id");
    const auto schemaEnd = std::find(header + 176, header + 240, Byte(0));
    if (schemaEnd == header + 240
        || std::string_view(reinterpret_cast<const char*>(header + 176),
                            std::size_t(schemaEnd - (header + 176)))
             != NetworkSchemaName
        || !all_zero(schemaEnd, header + 240))
        return failure(V3ContainerLoadError::SCHEMA_MISMATCH, "V3 schema name or padding differs");

    const std::uint64_t structureOffset  = read_u64(header + 32);
    const std::uint64_t structureBytes   = read_u64(header + 40);
    const std::uint64_t provenanceOffset = read_u64(header + 80);
    const std::uint64_t provenanceBytes  = read_u64(header + 88);
    const std::uint64_t parameterOffset  = read_u64(header + 128);
    const std::uint64_t parameterBytes   = read_u64(header + 136);
    const std::uint64_t fileBytes        = file.size();
    if (structureOffset != V3ContainerHeaderBytes
        || !range_within(structureOffset, structureBytes, fileBytes)
        || provenanceOffset != structureOffset + structureBytes
        || !range_within(provenanceOffset, provenanceBytes, fileBytes)
        || parameterOffset != provenanceOffset + provenanceBytes
        || !range_within(parameterOffset, parameterBytes, fileBytes)
        || parameterOffset + parameterBytes != fileBytes
        || parameterBytes != V3ContainerParameterBytes)
        return failure(V3ContainerLoadError::HEADER_MISMATCH, "V3 section offsets or lengths differ");

    if (!matches_hex(header + 48, StructureSha)
        || digest_hex(file.data() + structureOffset, std::size_t(structureBytes)) != StructureSha)
        return failure(V3ContainerLoadError::STRUCTURE_MISMATCH,
                       "V3 structural descriptor identity differs");

    for (const std::size_t offset : {240U, 272U, 304U, 336U, 368U, 400U})
        if (!nonzero_digest(header + offset))
            return failure(V3ContainerLoadError::PROVENANCE_MISMATCH,
                           "V3 provenance contains a zero identity");
    if (all_zero(header + 432, header + 452) || header[452] != 0)
        return failure(V3ContainerLoadError::PROVENANCE_MISMATCH,
                       "V3 source identity is zero or dirty");

    const std::string provenance(reinterpret_cast<const char*>(file.data() + provenanceOffset),
                                 std::size_t(provenanceBytes));
    if (digest_hex(file.data() + provenanceOffset, std::size_t(provenanceBytes))
          != bytes_hex(header + 96, 32)
        || provenance != expected_provenance(header))
        return failure(V3ContainerLoadError::PROVENANCE_MISMATCH,
                       "V3 provenance payload or identity differs");

    if (header[453] != 1 || header[454] != FtActivationShift || header[455] != DenseActivationShift
        || read_u32(header + 456) != FirstDomainV3Stream || read_u32(header + 460) != V3StreamRows
        || read_u32(header + 464) != V3Lanes || read_u32(header + 468) != G0Rows
        || read_u32(header + 472) != ContextualRows || read_u32(header + 476) != V3Hidden0Lanes
        || read_u32(header + 480) != V3Hidden1Lanes || read_u32(header + 484) != V3OutputHeads
        || read_u32(header + 488) != V3PhaseBuckets || read_u32(header + 492) != FeatureScale
        || read_u32(header + 496) != 64 || read_i32(header + 500) != 0
        || read_i32(header + 504) != ActivationMax || read_i32(header + 508) != OutputDivisor
        || read_i32(header + 512) != MaxSafeBiasMagnitude || read_u32(header + 516) != 1
        || read_u32(header + 520) != 1 || read_u32(header + 524) != DirectoryOffset
        || read_u32(header + 528) != DirectoryEntryBytes)
        return failure(V3ContainerLoadError::SCHEMA_MISMATCH,
                       "V3 topology or integer contract differs");

    // The V3 header extension block, offsets 532 through 575.
    if (read_u32(header + 532) != OutputWeightScaleNumerator
        || read_u32(header + 536) != OutputWeightScaleDenominator
        || read_u32(header + 540) != OutputBiasScale || read_u32(header + 544) != PsqtScale
        || read_u32(header + 548) != V3PsqtColumns || read_i32(header + 552) != MaxPsqtMagnitude
        || read_u32(header + 556) != ActiveRowBound || read_u32(header + 560) != V3PhaseLookupSize
        || read_u32(header + 564) != NetworkToScoreValue
        || read_u32(header + 568) != HiddenBiasScale
        || read_u32(header + 572) != ContextualBlockCount
        || !all_zero(header + 576, header + DirectoryOffset))
        return failure(V3ContainerLoadError::SCHEMA_MISMATCH,
                       "V3 header extension block differs from the registered contract");

    static_assert(ActiveRowBound == MaxActiveRows);
    static_assert(NetworkToScoreValue == NetworkToScore);

    const Byte* parameterPayload = file.data() + parameterOffset;
    if (digest_hex(parameterPayload, std::size_t(parameterBytes)) != bytes_hex(header + 144, 32))
        return failure(V3ContainerLoadError::PAYLOAD_MISMATCH,
                       "V3 parameter payload SHA-256 differs");

    std::array<const Byte*, V3ContainerSectionCount> sectionData{};
    std::size_t                                      expectedOffset = 0;
    for (std::size_t index = 0; index < V3ContainerSectionCount; ++index)
    {
        const Byte*         entry    = header + DirectoryOffset + index * DirectoryEntryBytes;
        const SectionSpec&  expected = Sections[index];
        const std::uint64_t offset   = read_u64(entry + 12);
        const std::uint64_t bytes    = read_u64(entry + 20);
        if (read_u16(entry) != expected.id || entry[2] != Byte(expected.dtype)
            || entry[3] != expected.rank || read_u32(entry + 4) != expected.first
            || read_u32(entry + 8) != expected.second || offset != expectedOffset
            || bytes != expected.bytes || read_u32(entry + 60) != 0
            || !range_within(offset, bytes, parameterBytes))
            return failure(V3ContainerLoadError::DIRECTORY_MISMATCH,
                           "V3 parameter directory differs");
        sectionData[index] = parameterPayload + offset;
        if (digest_hex(sectionData[index], std::size_t(bytes)) != bytes_hex(entry + 28, 32))
            return failure(V3ContainerLoadError::PAYLOAD_MISMATCH,
                           "V3 parameter section SHA-256 differs");
        expectedOffset += std::size_t(bytes);
    }
    const std::size_t directoryEnd = DirectoryOffset + V3ContainerSectionCount * DirectoryEntryBytes;
    if (expectedOffset != parameterBytes
        || !all_zero(header + directoryEnd, header + V3ContainerHeaderBytes))
        return failure(V3ContainerLoadError::DIRECTORY_MISMATCH,
                       "V3 directory does not exactly cover the payload");

    V3Parameters parameters;
    parameters.schemaName      = std::string(NetworkSchemaName);
    parameters.parameterSha256 = bytes_hex(header + 144, 32);
    parameters.fileSha256      = digest_hex(file.data(), file.size());

    for (std::size_t index = 0; index < FtWeightCount; ++index)
        parameters.ftWeights[index] = FtWeight(read_i16(sectionData[0] + index * 2));
    for (std::size_t index = 0; index < V3Lanes; ++index)
        parameters.ftBias[index] = AffineBias(read_i32(sectionData[1] + index * 4));
    for (std::size_t index = 0; index < PsqtWeightCount; ++index)
        parameters.psqtWeights[index] = PsqtWeight(read_i32(sectionData[2] + index * 4));
    for (std::size_t index = 0; index < Hidden0WeightCount; ++index)
        parameters.hidden0Weights[index] = DenseWeight(read_i8(sectionData[3][index]));
    for (std::size_t index = 0; index < parameters.hidden0Bias.size(); ++index)
        parameters.hidden0Bias[index] = AffineBias(read_i32(sectionData[4] + index * 4));
    for (std::size_t index = 0; index < Hidden1WeightCount; ++index)
        parameters.hidden1Weights[index] = DenseWeight(read_i8(sectionData[5][index]));
    for (std::size_t index = 0; index < parameters.hidden1Bias.size(); ++index)
        parameters.hidden1Bias[index] = AffineBias(read_i32(sectionData[6] + index * 4));
    for (std::size_t index = 0; index < OutputWeightCount; ++index)
        parameters.outputWeights[index] = DenseWeight(read_i8(sectionData[7][index]));
    for (std::size_t index = 0; index < parameters.outputBias.size(); ++index)
        parameters.outputBias[index] = AffineBias(read_i32(sectionData[8] + index * 4));

    // The phase lookup is the serialized white_piece_count to bucket map. It
    // must stay inside the eight buckets, never decrease and reach every
    // bucket, or the output bucket selection is not the trained one.
    std::array<bool, V3PhaseBuckets> reached{};
    for (std::size_t index = 0; index < V3PhaseLookupSize; ++index)
    {
        const std::int8_t bucket = read_i8(sectionData[9][index]);
        if (bucket < 0 || std::size_t(bucket) >= V3PhaseBuckets)
            return failure(V3ContainerLoadError::PHASE_LOOKUP,
                           "phase lookup contains an out-of-range bucket");
        if (index && bucket < std::int8_t(parameters.phaseLookup[index - 1]))
            return failure(V3ContainerLoadError::PHASE_LOOKUP,
                           "phase lookup is not monotone nondecreasing in white_piece_count");
        parameters.phaseLookup[index] = std::uint8_t(bucket);
        reached[std::size_t(bucket)]  = true;
    }
    if (!std::all_of(reached.begin(), reached.end(), [](bool value) { return value; }))
        return failure(V3ContainerLoadError::PHASE_LOOKUP,
                       "phase lookup does not reach every bucket");

    const auto safeBias = [](AffineBias bias) {
        return bias >= -MaxSafeBiasMagnitude && bias <= MaxSafeBiasMagnitude;
    };
    if (!std::all_of(parameters.ftBias.begin(), parameters.ftBias.end(), safeBias)
        || !std::all_of(parameters.hidden0Bias.begin(), parameters.hidden0Bias.end(), safeBias)
        || !std::all_of(parameters.hidden1Bias.begin(), parameters.hidden1Bias.end(), safeBias)
        || !std::all_of(parameters.outputBias.begin(), parameters.outputBias.end(), safeBias))
        return failure(V3ContainerLoadError::PARAMETER_RANGE,
                       "a V3 bias exceeds the registered safe magnitude");
    for (std::size_t index = 0; index < PsqtWeightCount; ++index)
        if (parameters.psqtWeights[index] > MaxPsqtMagnitude
            || parameters.psqtWeights[index] < -MaxPsqtMagnitude)
            return failure(V3ContainerLoadError::PARAMETER_RANGE,
                           "a V3 PSQT weight exceeds the registered magnitude bound");

    if (!parameters.valid())
        return failure(V3ContainerLoadError::PARAMETER_RANGE,
                       "V3 decoded parameters violate their registered bounds");

    V3ContainerLoadResult result;
    result.provenance               = provenance;
    result.trainingStructuralSha256 = bytes_hex(header + 240, 32);
    result.parameters               = std::move(parameters);
    return result;
}

}  // namespace Stockfish::Eval::NNUE::HordeV3
