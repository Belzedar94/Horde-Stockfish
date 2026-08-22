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
#include "data/expansion_selection.h"
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

Data::ExpansionSelection
selection_for(Position& position, StateInfo& state, const char* fen, int promo, int check, int cap) {
    require(!position.set(fen, false, &state), "expansion fixture FEN was rejected");
    return Data::select_expansion_children(position, promo, check, cap);
}

int family_count(const Data::ExpansionSelection& selection, Data::HordeExpansionFamily family) {
    int total = 0;
    for (const auto& choice : selection.children)
        total += int(choice.family == family);
    return total;
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

    // HORDE_BIN_V1_R2, layout. A parent is byte-identical to what V1 wrote: the
    // family nibble is zero and the EXPANSION_CHILD flag is clear.
    require((record[35] & 0x80U) == 0, "a parent record set the EXPANSION_CHILD flag");
    require(record[47] == u8(Data::HordeOutcomeReason::EXTINCTION),
            "a parent record disturbed the outcome byte");

    Data::HordeBinV1DecodedRecord decoded;
    require(bool(Data::decode_horde_bin_v1(record, decoded)), "parent record did not decode");
    require(decoded.outcomeReason == Data::HordeOutcomeReason::EXTINCTION
              && decoded.family == Data::HordeExpansionFamily::NONE && !decoded.expansion_child(),
            "parent round trip lost its identity");
    require(decoded.result == 1 && decoded.score == -123 && decoded.sideToMove == BLACK
              && decoded.gamePly == 1 && decoded.bestMove == 1 && decoded.playedMove == 1,
            "parent round trip lost a field");

    // Families. Each one sets bit 7 and its own nibble, and the terminal reason
    // survives underneath.
    const Data::HordeExpansionFamily families[] = {Data::HordeExpansionFamily::BESTMOVE,
                                                   Data::HordeExpansionFamily::PROMOTION,
                                                   Data::HordeExpansionFamily::CHECK};
    for (const Data::HordeExpansionFamily family : families)
    {
        sample.family = family;
        require(bool(Data::encode_horde_bin_v1(sample, record)), "expansion child did not encode");
        require((record[35] & 0x80U) != 0, "expansion child did not set EXPANSION_CHILD");
        require(record[47]
                  == u8(u8(Data::HordeOutcomeReason::EXTINCTION) | u8(u8(family) << 3)),
                "expansion family nibble encoding changed");
        require(bool(Data::decode_horde_bin_v1(record, decoded)), "expansion child did not decode");
        require(decoded.family == family && decoded.expansion_child()
                  && decoded.outcomeReason == Data::HordeOutcomeReason::EXTINCTION,
                "expansion child round trip lost its family or its terminal reason");
    }

    // The two markings must agree. A record that claims to be a child in one
    // field and a parent in the other is rejected, in both directions.
    sample.family = Data::HordeExpansionFamily::PROMOTION;
    require(bool(Data::encode_horde_bin_v1(sample, record)), "promotion child did not encode");
    Data::HordeBinV1Record tampered = record;
    tampered[35] &= u8(~0x80U);
    require(!Data::decode_horde_bin_v1(tampered, decoded),
            "a child with a cleared EXPANSION_CHILD flag was accepted");
    tampered = record;
    tampered[47] = u8(Data::HordeOutcomeReason::EXTINCTION);
    require(!Data::decode_horde_bin_v1(tampered, decoded),
            "a flagged child with family NONE was accepted");
    tampered = record;
    tampered[47] = u8(tampered[47] | 0x40U);
    require(!Data::decode_horde_bin_v1(tampered, decoded),
            "reserved outcome-byte bit 6 was accepted");
    tampered = record;
    tampered[47] = u8(u8(Data::HordeOutcomeReason::EXTINCTION) | u8(4U << 3));
    require(!Data::decode_horde_bin_v1(tampered, decoded),
            "an unregistered expansion family was accepted by the reader");

    sample.family = Data::HordeExpansionFamily(4);
    require(!Data::encode_horde_bin_v1(sample, record),
            "an unregistered expansion family was accepted by the writer");
    sample.family = Data::HordeExpansionFamily::NONE;

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
    manifest.networkSha256 = "B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7";
    require(bool(Data::validate_horde_bin_v1_manifest(manifest)),
            "the restored manifest was rejected");

    // The three expansion settings must agree, so the manifest identity is a
    // reliable statement about whether the payload can contain children.
    require(!manifest.expansion_enabled() && manifest.expansion_ceiling() == 0,
            "the default manifest claims expansion");
    manifest.expandPromo = 2;
    require(!Data::validate_horde_bin_v1_manifest(manifest),
            "a family cap without a ceiling was accepted");
    manifest.expandMaxChildren = 5;
    manifest.expandCheck       = 2;
    require(bool(Data::validate_horde_bin_v1_manifest(manifest)),
            "the declared 2/2/5 expansion budget was rejected");
    require(manifest.expansion_enabled() && manifest.expansion_ceiling() == 4,
            "the realised ceiling is not min(promo + check, max_children)");
    manifest.expandPromo = 0;
    manifest.expandCheck = 0;
    require(!Data::validate_horde_bin_v1_manifest(manifest),
            "a ceiling without a family cap was accepted");
    manifest.expandMaxChildren = 0;
    manifest.expandPromo       = Data::HordeBinV1MaxExpansionCap + 1;
    require(!Data::validate_horde_bin_v1_manifest(manifest),
            "an out-of-domain family cap was accepted");

    // Selection rule. White pawns on b7 and d7 both promote with check, and the
    // rooks supply exactly two non-promoting checks, Ra8 and Rg8. The h2 pawn is
    // there to block Rg1-h1, which would otherwise be a third check.
    static constexpr auto CombinedFen = "7k/1P1P4/8/8/8/8/7P/R5R1 w - - 0 1";
    Position              board;
    StateInfo             boardState{};

    Data::ExpansionSelection selection = selection_for(board, boardState, CombinedFen, 2, 2, 5);
    // Eight legal promotion moves collapse to two candidates, one per push, so a
    // single pawn cannot spend the whole promotion budget on underpromotions.
    require(selection.promotionCandidates == 2, "promotion candidates are not counted per push");
    require(selection.checkCandidates == 2, "non-promoting check candidates changed");
    require(selection.children.size() == 4, "the 2/2 budget did not produce four children");
    require(family_count(selection, Data::HordeExpansionFamily::PROMOTION) == 2
              && family_count(selection, Data::HordeExpansionFamily::CHECK) == 2,
            "the selected children are not two per family");
    require(!selection.capped, "a selection inside its caps reported that it hit them");

    // The combined case: every promotion here also gives check, and every one of
    // them is labelled PROMOTION. No move is counted in both families.
    for (const auto& choice : selection.children)
        if (choice.family == Data::HordeExpansionFamily::PROMOTION)
            require(choice.move.type_of() == PROMOTION && board.gives_check(choice.move),
                    "the combined promotion-with-check case lost its family");
        else
            require(choice.move.type_of() != PROMOTION && board.gives_check(choice.move),
                    "a check child is a promotion or does not give check");
    require(selection.children[0].move.from_sq() != selection.children[1].move.from_sq()
              || selection.children[0].move.to_sq() != selection.children[1].move.to_sq(),
            "two promotion children share one push");

    // Caps bite, and a capped parent says so.
    selection = selection_for(board, boardState, CombinedFen, 1, 1, 5);
    require(selection.children.size() == 2 && selection.capped,
            "the 1/1 budget did not cap the selection");
    require(family_count(selection, Data::HordeExpansionFamily::PROMOTION) == 1
              && family_count(selection, Data::HordeExpansionFamily::CHECK) == 1,
            "the capped selection is not one per family");

    // The per-parent ceiling is applied after the family caps.
    selection = selection_for(board, boardState, CombinedFen, 2, 2, 3);
    require(selection.children.size() == 3, "the hard ceiling did not bound the child count");
    selection = selection_for(board, boardState, CombinedFen, 2, 2, 0);
    require(selection.children.empty(), "a zero ceiling still produced children");
    selection = selection_for(board, boardState, CombinedFen, 0, 0, 5);
    require(selection.children.empty(), "zero family caps still produced children");

    // Black to move, so no checking move exists at all: the Horde has no king.
    // Both black promotion pushes are candidates, the quiet one included.
    selection = selection_for(board, boardState, "k7/8/8/8/8/8/6p1/7R b - - 0 1", 2, 2, 5);
    require(selection.checkCandidates == 0, "the kingless Horde was reported to be in check");
    require(selection.promotionCandidates == 2, "the two black promotion pushes were not counted");
    require(selection.children.size() == 2
              && family_count(selection, Data::HordeExpansionFamily::PROMOTION) == 2,
            "black promotion children were not selected");
    require(!selection.capped, "a selection inside its caps reported that it hit them");
    selection = selection_for(board, boardState, "k7/8/8/8/8/8/6p1/7R b - - 0 1", 1, 2, 5);
    require(selection.children.size() == 1 && selection.capped,
            "the promotion cap did not bite without any check candidates");

    std::cout << "HORDE_BIN_V1_R2 codec and expansion selection tests passed" << std::endl;
    return EXIT_SUCCESS;
}
