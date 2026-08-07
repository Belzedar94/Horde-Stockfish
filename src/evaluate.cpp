/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "evaluate.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

#include "misc.h"
#include "nnue/network.h"
#include "nnue/nnue_misc.h"
#include "position.h"
#include "types.h"
#include "uci.h"
#include "nnue/nnue_accumulator.h"

namespace Stockfish {

namespace {

// Fairy-Stockfish's frozen Hordetest representation stores every unpromoted
// White Horde pawn as the custom H piece. H has MG value 152 in c19b5f6c, so
// it contributes to non-pawn material and not to count<PAWN>(). Preserve those
// evaluator inputs at the legacy boundary without changing the physical board.
constexpr int HordeLegacyPawnMgValue = 152;

int horde_legacy_non_pawn_material(const Position& pos, Color color) {
    return int(pos.non_pawn_material(color))
         + (color == WHITE ? HordeLegacyPawnMgValue * pos.count<PAWN>(WHITE) : 0);
}

}  // namespace

// Evaluate is the evaluator for the outer world. It returns a static evaluation
// of the position from the point of view of the side to move.
Value Eval::evaluate(const Eval::NNUE::Network&     network,
                     const Position&                pos,
                     Eval::NNUE::AccumulatorStack&  accumulators,
                     Eval::NNUE::AccumulatorCaches& caches,
                     int                            optimism) {

    assert(!pos.checkers());

    // Modern Stockfish optimism is calibrated for its current orthodox net.
    // Run 6B instead keeps the Fairy-Stockfish legacy blend and scale.
    (void) optimism;
    const auto [rawPsqt, rawPositional] = network.evaluate_raw(pos, accumulators, caches);

    const int whiteNpm = horde_legacy_non_pawn_material(pos, WHITE);
    const int blackNpm = horde_legacy_non_pawn_material(pos, BLACK);
    const int deltaNpm = std::abs(whiteNpm - blackNpm);
    const int entertainment = deltaNpm <= BishopValue - KnightValue ? 7 : 0;
    const int blendedRaw =
      ((128 - entertainment) * rawPsqt + (128 + entertainment) * rawPositional) / 128;

    const int scale = 903 + 32 * pos.count<PAWN>(BLACK) + 32 * (whiteNpm + blackNpm) / 1024;
    int       v     = (blendedRaw / 16) * scale / 1024;

    // Match the frozen FSF Hordetest 50-move contract exactly. The terminal
    // draw is reached after 100 plies; this is only its linear eval damping.
    constexpr int MoveRulePlies = 100;
    v = v * (MoveRulePlies - std::min(pos.rule50_count(), MoveRulePlies)) / MoveRulePlies;

    // Guarantee evaluation does not hit the tablebase range
    v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);

    return v;
}

// Like evaluate(), but instead of returning a value, it returns
// a string (suitable for outputting to stdout) that contains the detailed
// descriptions and values of each evaluation term. Useful for debugging.
// Trace scores are from white's point of view
std::string Eval::trace(Position& pos, const Eval::NNUE::Network& network) {

    if (pos.checkers())
        return "Final evaluation: none (in check)";

    auto accumulators = std::make_unique<Eval::NNUE::AccumulatorStack>();
    auto caches       = std::make_unique<Eval::NNUE::AccumulatorCaches>(network);

    std::stringstream ss;
    ss << std::showpoint << std::noshowpos << std::fixed << std::setprecision(2);
    ss << '\n' << NNUE::trace(pos, network, *caches) << '\n';

    ss << std::showpoint << std::showpos << std::fixed << std::setprecision(2) << std::setw(15);

    const auto [rawPsqt, rawPositional] = network.evaluate_raw(pos, *accumulators, *caches);
    Value v                             = Value((rawPsqt + rawPositional) / 16);
    ss << "Horde legacy raw NNUE: psqt " << rawPsqt << ", positional " << rawPositional
       << ", total " << rawPsqt + rawPositional << '\n';
    ss << "NNUE evaluation          " << v << " (side to move, internal units)\n";
    v = pos.side_to_move() == WHITE ? v : -v;
    ss << "NNUE evaluation        " << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)\n";

    v = evaluate(network, pos, *accumulators, *caches, VALUE_ZERO);
    v = pos.side_to_move() == WHITE ? v : -v;

    ss << "Final evaluation      ";
    ss << 0.01 * UCIEngine::to_cp(v, pos) << " (white side)";
    ss << " [with scaled NNUE, ...]\n";

    return ss.str();
}

}  // namespace Stockfish
