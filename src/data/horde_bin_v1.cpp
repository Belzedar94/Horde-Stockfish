/*
  Horde-Stockfish training-data generator
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "horde_bin_v1.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <limits>
#include <sstream>
#include <system_error>
#include <utility>

#include "movegen.h"
#include "position.h"

#ifdef _WIN32
    #include <fcntl.h>
    #include <io.h>
    #include <sys/stat.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
#endif

namespace Stockfish::Data {
namespace {

constexpr std::array<u8, 8> FileMagic     = {'H', 'O', 'R', 'D', 'E', 'B', 'I', 'N'};
constexpr u16               FormatVersion = 1;
constexpr std::string_view  CapabilityJson =
  R"({"schema":"HORDE_BIN_V1","schema_sha256":"B46ADE18AB8954A6AB232593484273E50C12B51550A938763A7A7D94DCCB63E4","write":true,"record_size":48,"header_size":2048})";

bool little_endian_host() {
    constexpr u16 value = 1;
    return *reinterpret_cast<const u8*>(&value) == 1;
}

std::string system_error_message(int error) {
    return std::error_code(error, std::generic_category()).message();
}

void write_u16_le(u8* destination, u16 value) {
    destination[0] = u8(value & 0xFF);
    destination[1] = u8(value >> 8);
}

void write_u32_le(u8* destination, u32 value) {
    destination[0] = u8(value & 0xFF);
    destination[1] = u8((value >> 8) & 0xFF);
    destination[2] = u8((value >> 16) & 0xFF);
    destination[3] = u8(value >> 24);
}

bool is_hex_commit(std::string_view text) {
    return text.size() == 40 && std::all_of(text.begin(), text.end(), [](unsigned char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
           });
}

std::string json_bool(bool value) { return value ? "true" : "false"; }

u8 piece_code(Piece piece) {
    switch (piece)
    {
    case NO_PIECE :
        return 0;
    case W_PAWN :
        return 1;
    case W_KNIGHT :
        return 2;
    case W_BISHOP :
        return 3;
    case W_ROOK :
        return 4;
    case W_QUEEN :
        return 5;
    case B_PAWN :
        return 6;
    case B_KNIGHT :
        return 7;
    case B_BISHOP :
        return 8;
    case B_ROOK :
        return 9;
    case B_QUEEN :
        return 10;
    case B_KING :
        return 11;
    default :
        return 0xFF;
    }
}

std::string
manifest_json(const HordeBinV1Manifest& manifest, u64 records, std::string_view payloadSha256) {
    std::ostringstream json;
    json << "{\"schema\":\"" << HordeBinV1SchemaName << "\",\"schema_sha256\":\""
         << HordeBinV1SchemaSha256
         << "\",\"format_version\":1,\"header_bytes\":" << HordeBinV1HeaderSize
         << ",\"record_bytes\":" << HordeBinV1RecordSize << ",\"record_count\":" << records
         << ",\"byte_order\":\"little\",\"source_commit\":\"" << manifest.sourceCommit
         << "\",\"source_dirty\":" << json_bool(manifest.sourceDirty)
         << ",\"network\":{\"schema\":\"HORDETEST_HP_LEGACY_V1\",\"sha256\":\""
         << manifest.networkSha256 << "\"},\"book_sha256\":\"" << manifest.bookSha256
         << "\",\"producer_sha256\":\"" << manifest.producerSha256 << "\",\"payload_sha256\":\""
         << payloadSha256
         << "\",\"generation\":{\"requested_records\":" << manifest.requestedRecords
         << ",\"seed\":\"" << manifest.seed << "\",\"threads\":" << manifest.threads
         << ",\"hash_mb\":" << manifest.hashMb << ",\"depth\":" << manifest.depth
         << ",\"nodes\":" << manifest.nodes
         << ",\"random_move_min_ply\":" << manifest.randomMoveMinPly
         << ",\"random_move_max_ply\":" << manifest.randomMoveMaxPly
         << ",\"random_move_count\":" << manifest.randomMoveCount
         << ",\"random_multi_pv\":" << manifest.randomMultiPv
         << ",\"random_multi_pv_diff\":" << manifest.randomMultiPvDiff
         << ",\"write_min_ply\":" << manifest.writeMinPly
         << ",\"write_max_ply\":" << manifest.writeMaxPly
         << ",\"max_game_ply\":" << manifest.maxGamePly
         << ",\"opening_count\":" << manifest.openingCount << "}}";
    return json.str();
}

}  // namespace

std::string_view horde_data_schema_json() noexcept { return CapabilityJson; }

DataResult validate_horde_bin_v1_manifest(const HordeBinV1Manifest& manifest) {
    if (!is_hex_commit(manifest.sourceCommit)
        || std::all_of(manifest.sourceCommit.begin(), manifest.sourceCommit.end(),
                       [](char c) { return c == '0'; }))
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "Source commit must be a nonzero full 40-digit Git object ID");
    if (manifest.sourceDirty)
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "HORDE_BIN_V1 generation requires a clean source tree");
    if (!is_upper_sha256(manifest.networkSha256)
        || manifest.networkSha256
             != "B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7")
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "Network SHA-256 must identify the registered Run 6B artifact");
    if (manifest.bookSha256 != "NONE" && !is_upper_sha256(manifest.bookSha256))
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "Book SHA-256 must be uppercase hexadecimal or NONE");
    if (!is_upper_sha256(manifest.producerSha256))
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "Producer SHA-256 must be uppercase hexadecimal");
    if (!manifest.requestedRecords || !manifest.seed || !manifest.threads || !manifest.hashMb
        || manifest.depth <= 0 || manifest.depth >= MAX_PLY || manifest.randomMoveMinPly < 0
        || manifest.randomMoveMaxPly < manifest.randomMoveMinPly || manifest.randomMoveCount < 0
        || manifest.randomMoveCount > manifest.maxGamePly || manifest.randomMultiPv < 0
        || manifest.randomMultiPv > int(MAX_MOVES) || manifest.randomMultiPvDiff < 0
        || manifest.writeMinPly < 0 || manifest.writeMaxPly <= manifest.writeMinPly
        || manifest.maxGamePly <= manifest.writeMaxPly || !manifest.openingCount)
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "Generation settings are outside the HORDE_BIN_V1 domain");
    return DataResult::success();
}

DataResult encode_horde_bin_v1(const TrainingDataSample& sample, HordeBinV1Record& record) {
    record.fill(0);
    if (!little_endian_host())
        return DataResult::failure(DataError::UNSUPPORTED_BYTE_ORDER,
                                   "HORDE_BIN_V1 generation requires a little-endian host");
    if (sample.score < std::numeric_limits<i16>::min()
        || sample.score > std::numeric_limits<i16>::max())
        return DataResult::failure(DataError::SCORE_OUT_OF_RANGE,
                                   "Training score does not fit HORDE_BIN_V1 int16");
    if (sample.result < -1 || sample.result > 1)
        return DataResult::failure(DataError::RESULT_OUT_OF_RANGE,
                                   "Training result must be -1, 0, or 1");
    const int reason = int(sample.outcomeReason);
    if (reason < int(HordeOutcomeReason::CHECKMATE)
        || reason > int(HordeOutcomeReason::FIVEFOLD_REPETITION))
        return DataResult::failure(DataError::OUTCOME_OUT_OF_RANGE,
                                   "Training outcome reason is not a Horde terminal reason");

    Position  position;
    StateInfo state{};
    if (const auto setError = position.set(sample.fen, false, &state))
        return DataResult::failure(DataError::UNSUPPORTED_POSITION,
                                   std::string("Cannot encode Horde FEN: ") + setError->what());
    if (position.is_chess960() || position.count<KING>(WHITE) != 0
        || position.count<KING>(BLACK) != 1 || popcount(position.pieces(WHITE)) > 36
        || popcount(position.pieces()) > 52 || position.can_castle(WHITE_OO)
        || position.can_castle(WHITE_OOO))
        return DataResult::failure(DataError::UNSUPPORTED_POSITION,
                                   "Position violates HORDE_BIN_V1 physical invariants");
    if (position.rule50_count() < 0 || position.rule50_count() > std::numeric_limits<u16>::max()
        || position.game_ply() < 0 || position.game_ply() > std::numeric_limits<u16>::max())
        return DataResult::failure(DataError::POSITION_CLOCK_OUT_OF_RANGE,
                                   "Position clocks do not fit HORDE_BIN_V1 uint16 fields");
    const MoveList<LEGAL> legalMoves(position);
    if (!sample.bestMove.is_ok() || !legalMoves.contains(sample.bestMove)
        || !sample.playedMove.is_ok() || !legalMoves.contains(sample.playedMove))
        return DataResult::failure(DataError::INVALID_MOVE,
                                   "Stored best and played moves must both be legal");

    for (int square = 0; square < SQUARE_NB; ++square)
    {
        const u8 code = piece_code(position.piece_on(Square(square)));
        if (code == 0xFF)
        {
            record.fill(0);
            return DataResult::failure(DataError::UNSUPPORTED_POSITION,
                                       "Position contains an unregistered physical piece code");
        }
        record[usize(square / 2)] |= u8(code << (4 * (square & 1)));
    }

    record[32] = u8(position.side_to_move());
    record[33] =
      u8((position.can_castle(BLACK_OO) ? 1 : 0) | (position.can_castle(BLACK_OOO) ? 2 : 0));
    record[34] = position.ep_square() == SQ_NONE ? 64 : u8(position.ep_square());

    u8         flags    = 0;
    const auto set_flag = [&flags](int bit, bool enabled) {
        if (enabled)
            flags |= u8(1U << bit);
    };
    set_flag(0, position.capture(sample.bestMove));
    set_flag(1, position.gives_check(sample.bestMove));
    set_flag(2, sample.bestMove.type_of() == PROMOTION);
    set_flag(3, sample.bestMove.type_of() == EN_PASSANT);
    set_flag(4, sample.bestMove.type_of() == CASTLING);
    set_flag(5, sample.playedMove != sample.bestMove);
    set_flag(6, position.side_to_move() == BLACK && position.checkers());
    record[35] = flags;

    write_u16_le(record.data() + 36, u16(position.rule50_count()));
    write_u16_le(record.data() + 38, u16(position.game_ply()));
    write_u16_le(record.data() + 40, u16(i16(sample.score)));
    write_u16_le(record.data() + 42, sample.bestMove.raw());
    write_u16_le(record.data() + 44, sample.playedMove.raw());
    record[46] = u8(i8(sample.result));
    record[47] = u8(sample.outcomeReason);
    return DataResult::success();
}

HordeBinV1Sink::HordeBinV1Sink(std::filesystem::path path, HordeBinV1Manifest manifest_) :
    outputPath(std::move(path)),
    manifest(std::move(manifest_)) {}

HordeBinV1Sink::~HordeBinV1Sink() {
    if (!finalized && !aborted)
    {
        int closeError  = 0;
        int removeError = 0;
        cleanup_partial(closeError, removeError);
    }
}

DataResult HordeBinV1Sink::open_exclusively() {
    if (fileDescriptor != -1)
        return DataResult::success();
    if (DataResult valid = validate_horde_bin_v1_manifest(manifest); !valid)
        return valid;

    errno = 0;
#ifdef _WIN32
    fileDescriptor =
      ::_wopen(outputPath.c_str(), _O_BINARY | _O_WRONLY | _O_CREAT | _O_EXCL | _O_NOINHERIT,
               _S_IREAD | _S_IWRITE);
#else
    int flags = O_WRONLY | O_CREAT | O_EXCL;
    #ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
    #endif
    fileDescriptor = ::open(outputPath.c_str(), flags, 0666);
#endif
    if (fileDescriptor == -1)
    {
        const int error = errno;
        return DataResult::failure(
          error == EEXIST ? DataError::OUTPUT_EXISTS : DataError::OPEN_FAILED,
          "Cannot create HORDE_BIN_V1 output exclusively: " + system_error_message(error));
    }

    created = true;
    std::array<u8, HordeBinV1HeaderSize> placeholder{};
    if (DataResult result =
          write_bytes(placeholder.data(), placeholder.size(), "header placeholder");
        !result)
        return result;
    return DataResult::success();
}

DataResult HordeBinV1Sink::write_bytes(const u8* data, std::size_t size, std::string_view label) {
    std::size_t written = 0;
    while (written < size)
    {
        errno = 0;
#ifdef _WIN32
        const int count = ::_write(fileDescriptor, data + written, unsigned(size - written));
#else
        const ssize_t count = ::write(fileDescriptor, data + written, size - written);
#endif
        if (count > 0)
        {
            written += std::size_t(count);
            continue;
        }
        if (count < 0 && errno == EINTR)
            continue;
        const int error = count == 0 ? EIO : errno;
        return DataResult::failure(DataError::WRITE_FAILED, "Cannot write HORDE_BIN_V1 "
                                                              + std::string(label) + ": "
                                                              + system_error_message(error));
    }
    return DataResult::success();
}

DataResult HordeBinV1Sink::append(const TrainingDataSample& sample) {
    if (!accepting)
        return DataResult::failure(DataError::SINK_CLOSED,
                                   "Cannot write to a closed HORDE_BIN_V1 sink");
    if (recordsWritten >= manifest.requestedRecords)
        return DataResult::failure(DataError::RECORD_COUNT_MISMATCH,
                                   "HORDE_BIN_V1 output exceeded its declared record count");

    HordeBinV1Record record{};
    if (DataResult encoded = encode_horde_bin_v1(sample, record); !encoded)
        return encoded;
    if (DataResult opened = open_exclusively(); !opened)
        return opened;
    if (DataResult written = write_bytes(record.data(), record.size(), "record"); !written)
    {
        accepting = false;
        return written;
    }

    payloadHasher.update(record.data(), record.size());
    ++recordsWritten;
    return DataResult::success();
}

DataResult HordeBinV1Sink::seek_to_start() {
    errno = 0;
#ifdef _WIN32
    const auto offset = ::_lseeki64(fileDescriptor, 0, SEEK_SET);
#else
    const auto offset = ::lseek(fileDescriptor, 0, SEEK_SET);
#endif
    if (offset == 0)
        return DataResult::success();
    return DataResult::failure(DataError::SEEK_FAILED, "Cannot seek to HORDE_BIN_V1 header: "
                                                         + system_error_message(errno));
}

DataResult HordeBinV1Sink::sync_and_close() {
    errno = 0;
#ifdef _WIN32
    if (::_commit(fileDescriptor) != 0)
#else
    if (::fsync(fileDescriptor) != 0)
#endif
        return DataResult::failure(DataError::CLOSE_FAILED,
                                   "Cannot synchronize HORDE_BIN_V1 output: "
                                     + system_error_message(errno));

    const int descriptor = fileDescriptor;
    fileDescriptor       = -1;
    errno                = 0;
#ifdef _WIN32
    const int result = ::_close(descriptor);
#else
    const int result = ::close(descriptor);
#endif
    if (result == 0)
    {
        finalized = true;
        return DataResult::success();
    }
    return DataResult::failure(DataError::CLOSE_FAILED,
                               "Cannot close HORDE_BIN_V1 output: " + system_error_message(errno));
}

DataResult HordeBinV1Sink::finalize() {
    if (finalized)
        return DataResult::success();
    if (aborted || !accepting)
        return DataResult::failure(DataError::SINK_CLOSED,
                                   "Cannot finalize an aborted or failed HORDE_BIN_V1 sink");
    accepting = false;
    if (!recordsWritten || fileDescriptor == -1)
        return DataResult::failure(DataError::EMPTY_DATASET,
                                   "HORDE_BIN_V1 forbids an empty dataset");
    if (recordsWritten != manifest.requestedRecords)
        return DataResult::failure(DataError::RECORD_COUNT_MISMATCH,
                                   "HORDE_BIN_V1 record count does not match the request");

    const std::string payloadSha = sha256_hex_upper(payloadHasher.digest());
    const std::string json       = manifest_json(manifest, recordsWritten, payloadSha);
    if (json.size() > HordeBinV1HeaderSize - 16)
        return DataResult::failure(DataError::INVALID_MANIFEST,
                                   "HORDE_BIN_V1 manifest exceeds the fixed header");

    std::array<u8, HordeBinV1HeaderSize> header{};
    std::copy(FileMagic.begin(), FileMagic.end(), header.begin());
    write_u16_le(header.data() + 8, FormatVersion);
    write_u16_le(header.data() + 10, u16(HordeBinV1HeaderSize));
    write_u32_le(header.data() + 12, u32(json.size()));
    std::copy(json.begin(), json.end(), header.begin() + 16);

    if (DataResult seek = seek_to_start(); !seek)
        return seek;
    if (DataResult written = write_bytes(header.data(), header.size(), "header"); !written)
        return written;
    return sync_and_close();
}

void HordeBinV1Sink::cleanup_partial(int& closeError, int& removeError) noexcept {
    closeError  = 0;
    removeError = 0;
    accepting   = false;
    if (fileDescriptor != -1)
    {
        const int descriptor = fileDescriptor;
        fileDescriptor       = -1;
        errno                = 0;
#ifdef _WIN32
        if (::_close(descriptor) != 0)
#else
        if (::close(descriptor) != 0)
#endif
            closeError = errno;
    }

    if (created)
    {
        errno = 0;
#ifdef _WIN32
        if (::_wunlink(outputPath.c_str()) != 0 && errno != ENOENT)
#else
        if (::unlink(outputPath.c_str()) != 0 && errno != ENOENT)
#endif
            removeError = errno;
        else
            created = false;
    }
    aborted = !created;
}

DataResult HordeBinV1Sink::abort() {
    if (aborted)
        return DataResult::success();
    if (finalized)
        return DataResult::failure(DataError::SINK_CLOSED,
                                   "Cannot abort a finalized HORDE_BIN_V1 sink");

    int closeError  = 0;
    int removeError = 0;
    cleanup_partial(closeError, removeError);
    if (closeError || removeError)
        return DataResult::failure(
          DataError::ABORT_FAILED,
          "Cannot fully remove partial HORDE_BIN_V1 output: "
            + system_error_message(removeError ? removeError : closeError));
    return DataResult::success();
}

}  // namespace Stockfish::Data
