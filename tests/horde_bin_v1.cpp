/*
  Horde-Stockfish HORDE_BIN_V1 contract tests
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include <cstdlib>
#include <iostream>
#include <string>

#include "attacks.h"
#include "data/horde_bin_v1.h"
#include "data/sha256.h"
#include "position.h"
#include "types.h"

using namespace Stockfish;

namespace {

void require(bool condition, const char* message) {
    if (!condition)
    {
        std::cerr << "HORDE_BIN_V1 test failure: " << message << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main() {
    Attacks::init();
    Position::init();

    Data::Sha256 hasher;
    hasher.update("abc", 3);
    require(Data::sha256_hex_upper(hasher.digest())
              == "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD",
            "streaming SHA-256 does not match the standard abc vector");

    Data::TrainingDataSample sample{
      "8/8/8/8/8/8/8/kQ6 b - - 0 1",       -123, Move(SQ_A1, SQ_B1), Move(SQ_A1, SQ_B1), 1,
      Data::HordeOutcomeReason::EXTINCTION};
    Data::HordeBinV1Record record{};
    require(bool(Data::encode_horde_bin_v1(sample, record)), "forced extinction did not encode");
    require(record[0] == 0x5B, "physical Black king and White queen piece codes changed");
    require(record[32] == BLACK, "side-to-move byte changed");
    require(record[33] == 0 && record[34] == 64, "state metadata changed");
    require(record[35] == ((1U << 0) | (1U << 6)), "sample flag encoding changed");
    require(record[38] == 1 && record[39] == 0, "game-ply encoding changed");
    require(record[42] == 1 && record[43] == 0 && record[44] == 1 && record[45] == 0,
            "move encoding changed");
    require(record[46] == 1 && record[47] == 2, "result or terminal-reason encoding changed");

    sample.result = 0;
    require(!Data::encode_horde_bin_v1(sample, record),
            "decisive terminal reason accepted a draw result");
    sample.result        = 1;
    sample.outcomeReason = Data::HordeOutcomeReason::STALEMATE;
    require(!Data::encode_horde_bin_v1(sample, record),
            "draw terminal reason accepted a decisive result");
    sample.outcomeReason = Data::HordeOutcomeReason::EXTINCTION;

    sample.playedMove = Move(SQ_A1, SQ_A2);
    require(!Data::encode_horde_bin_v1(sample, record), "illegal played move was accepted");
    sample.playedMove = Move(SQ_A1, SQ_B1);
    sample.fen        = "8/8/8/8/8/8/8/K6k w - - 0 1";
    require(!Data::encode_horde_bin_v1(sample, record), "White king was accepted");

    Data::HordeBinV1Manifest manifest;
    manifest.sourceCommit      = std::string(40, 'a');
    manifest.networkSha256     = "B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7";
    manifest.bookSha256        = "NONE";
    manifest.producerSha256    = std::string(64, 'A');
    manifest.requestedRecords  = 1;
    manifest.seed              = 1;
    manifest.threads           = 1;
    manifest.hashMb            = 16;
    manifest.depth             = 1;
    manifest.randomMoveMinPly  = 0;
    manifest.randomMoveMaxPly  = 0;
    manifest.randomMoveCount   = 0;
    manifest.randomMultiPv     = 1;
    manifest.randomMultiPvDiff = 0;
    manifest.writeMinPly       = 0;
    manifest.writeMaxPly       = 2;
    manifest.maxGamePly        = 4;
    manifest.openingCount      = 1;
    require(bool(Data::validate_horde_bin_v1_manifest(manifest)), "valid manifest was rejected");
    require(Data::HordeLabelContractName == "HORDE_LABEL_CONTRACT_V1"
              && Data::HordeLabelContractSha256
                   == "C299BA9ECD96DEF24363F8F62A8C67B88241AA860FB0735D4558B8EFEA0DCC22",
            "label-contract identity changed");
    manifest.sourceDirty = true;
    require(!Data::validate_horde_bin_v1_manifest(manifest), "dirty source was accepted");
    manifest.sourceDirty   = false;
    manifest.networkSha256 = std::string(64, '0');
    require(!Data::validate_horde_bin_v1_manifest(manifest), "unregistered network was accepted");

    std::cout << "HORDE_BIN_V1 codec tests passed" << std::endl;
    return EXIT_SUCCESS;
}
