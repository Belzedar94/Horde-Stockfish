/*
  Horde-Stockfish training-data generator
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef DATA_EXPANSION_SELECTION_H_INCLUDED
#define DATA_EXPANSION_SELECTION_H_INCLUDED

#include <vector>

#include "position.h"
#include "training_data.h"
#include "types.h"

namespace Stockfish::Data {

struct ExpansionChoice {
    Move                 move   = Move::none();
    HordeExpansionFamily family = HordeExpansionFamily::NONE;
};

struct ExpansionSelection {
    std::vector<ExpansionChoice> children;

    // Candidates are counted before the caps are applied, so a shard that hit a
    // cap is distinguishable from one that did not.
    int  promotionCandidates = 0;
    int  checkCandidates     = 0;
    int  promotionSelected   = 0;
    int  checkSelected       = 0;
    bool capped              = false;
};

// Which children of a written position the HORDE_BIN_V1_R2 tactical expansion
// emits. Pure, deterministic and independent of the search, so the selection rule
// is testable on its own.
//
// - Promotion candidates are the legal promotion moves, counted once per distinct
//   origin-destination push, so one pawn cannot spend the whole promotion budget
//   on its own underpromotions.
// - Check candidates are the legal checking moves that are not promotions.
// - A promotion that also gives check belongs to PROMOTION and spends only the
//   promotion budget. The check budget therefore stays reserved for non-promoting
//   checks and no move is counted twice.
// - promoCap and checkCap bound each family; ceiling is the hard per-parent
//   ceiling applied afterwards. Selection follows legal-move generation order.
ExpansionSelection
select_expansion_children(const Position& position, int promoCap, int checkCap, int ceiling);

}  // namespace Stockfish::Data

#endif  // DATA_EXPANSION_SELECTION_H_INCLUDED
