/*
  Horde-Stockfish training-data generator
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "expansion_selection.h"

#include <algorithm>

#include "movegen.h"

namespace Stockfish::Data {

ExpansionSelection
select_expansion_children(const Position& position, int promoCap, int checkCap, int ceiling) {
    ExpansionSelection selection;
    if (ceiling <= 0 || (promoCap <= 0 && checkCap <= 0))
        return selection;
    selection.children.reserve(usize(ceiling));

    std::vector<u16> promotionPushes;

    for (const Move move : MoveList<LEGAL>(position))
    {
        const bool isPromotion = move.type_of() == PROMOTION;
        const bool givesCheck  = position.gives_check(move);
        if (!isPromotion && !givesCheck)
            continue;

        if (isPromotion)
        {
            // PROMOTION wins the combined case. A promotion that also gives
            // check is a promotion child and spends only the promotion budget.
            const u16 push = u16((u16(move.from_sq()) << 6) | u16(move.to_sq()));
            if (std::find(promotionPushes.begin(), promotionPushes.end(), push)
                != promotionPushes.end())
                continue;
            promotionPushes.push_back(push);
            ++selection.promotionCandidates;
            if (selection.promotionSelected < promoCap && int(selection.children.size()) < ceiling)
            {
                ++selection.promotionSelected;
                selection.children.push_back({move, HordeExpansionFamily::PROMOTION});
            }
        }
        else
        {
            ++selection.checkCandidates;
            if (selection.checkSelected < checkCap && int(selection.children.size()) < ceiling)
            {
                ++selection.checkSelected;
                selection.children.push_back({move, HordeExpansionFamily::CHECK});
            }
        }
    }

    selection.capped = selection.promotionCandidates > selection.promotionSelected
                    || selection.checkCandidates > selection.checkSelected;
    return selection;
}

}  // namespace Stockfish::Data
