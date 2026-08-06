/*
  Horde-Stockfish training-data generator
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef DATA_HORDE_BIN_V1_H_INCLUDED
#define DATA_HORDE_BIN_V1_H_INCLUDED

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>

#include "sha256.h"
#include "training_data.h"

namespace Stockfish::Data {

inline constexpr std::size_t      HordeBinV1HeaderSize = 2048;
inline constexpr std::size_t      HordeBinV1RecordSize = 48;
inline constexpr std::string_view HordeBinV1SchemaName = "HORDE_BIN_V1";
inline constexpr std::string_view HordeBinV1SchemaSha256 =
  "B46ADE18AB8954A6AB232593484273E50C12B51550A938763A7A7D94DCCB63E4";

using HordeBinV1Record = std::array<u8, HordeBinV1RecordSize>;

struct HordeBinV1Manifest {
    std::string sourceCommit;
    bool        sourceDirty = false;
    std::string networkSha256;
    std::string bookSha256;
    std::string producerSha256;

    u64   requestedRecords  = 0;
    u64   seed              = 0;
    usize threads           = 0;
    usize hashMb            = 0;
    int   depth             = 0;
    u64   nodes             = 0;
    int   randomMoveMinPly  = 0;
    int   randomMoveMaxPly  = 0;
    int   randomMoveCount   = 0;
    int   randomMultiPv     = 0;
    int   randomMultiPvDiff = 0;
    int   writeMinPly       = 0;
    int   writeMaxPly       = 0;
    int   maxGamePly        = 0;
    usize openingCount      = 0;
};

std::string_view horde_data_schema_json() noexcept;
DataResult       validate_horde_bin_v1_manifest(const HordeBinV1Manifest& manifest);
DataResult       encode_horde_bin_v1(const TrainingDataSample& sample, HordeBinV1Record& record);

class HordeBinV1Sink final: public DatasetSink {
   public:
    HordeBinV1Sink(std::filesystem::path path, HordeBinV1Manifest manifest);
    ~HordeBinV1Sink() override;

    HordeBinV1Sink(const HordeBinV1Sink&)            = delete;
    HordeBinV1Sink& operator=(const HordeBinV1Sink&) = delete;

    DataResult append(const TrainingDataSample& sample) override;
    DataResult finalize() override;
    DataResult abort() override;

    u64 records_written() const noexcept { return recordsWritten; }

   private:
    DataResult open_exclusively();
    DataResult write_bytes(const u8* data, std::size_t size, std::string_view label);
    DataResult seek_to_start();
    DataResult sync_and_close();
    void       cleanup_partial(int& closeError, int& removeError) noexcept;

    std::filesystem::path outputPath;
    HordeBinV1Manifest    manifest;
    Sha256                payloadHasher;
    u64                   recordsWritten = 0;
    int                   fileDescriptor = -1;
    bool                  created        = false;
    bool                  accepting      = true;
    bool                  finalized      = false;
    bool                  aborted        = false;
};

}  // namespace Stockfish::Data

#endif  // DATA_HORDE_BIN_V1_H_INCLUDED
