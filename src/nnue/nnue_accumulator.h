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

// Class for difference calculation of NNUE evaluation function

#ifndef NNUE_ACCUMULATOR_H_INCLUDED
#define NNUE_ACCUMULATOR_H_INCLUDED

#include <array>
#include <type_traits>

#include "../misc.h"
#include "../types.h"
#include "horde_legacy_network.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE {

struct alignas(CacheLineSize) Accumulator;

// Run 6B has a fixed 512-lane, king-independent accumulator. Keeping the
// physical layout tied to that registered schema avoids carrying the unused
// standard-NNUE lanes through every search ply.
struct alignas(CacheLineSize) Accumulator {
    std::array<std::array<i16, HordeLegacyNetwork::AccumulatorDimensions>, COLOR_NB>
      accumulation;
    std::array<std::array<i32, HordeLegacyNetwork::PsqtBuckets>, COLOR_NB> psqtAccumulation;
    std::array<bool, COLOR_NB>                                              computed = {};
};

// The legacy schema is king-independent, so Finny tables cannot provide a
// useful refresh key. Preserve the public adapter used by Worker while making
// it a zero-state object.
struct AccumulatorCaches {
    template<typename Network>
    explicit AccumulatorCaches(const Network&) noexcept {}

    template<typename Network>
    void clear(const Network&) noexcept {}
};

struct AccumulatorState: public Accumulator {
    DirtyPiece dirtyPiece{};
};

class AccumulatorStack {
   public:
    static constexpr usize MaxSize = MAX_PLY + 1;

    [[nodiscard]] const AccumulatorState& latest() const noexcept;

    void        reset() noexcept;
    DirtyPiece& push() noexcept;
    void        pop() noexcept;

    // The legacy HordeTest schema is king-independent. The stack keeps one
    // exact 512-lane state per ply and updates it through make/undo.
    void evaluate_horde_legacy(const Position& pos, const HordeLegacyNetwork& network) noexcept;

   private:
    [[nodiscard]] AccumulatorState& mut_latest() noexcept;

    std::array<AccumulatorState, MaxSize> accumulators;
    usize                                 size = 1;
};

static_assert(HordeLegacyNetwork::AccumulatorDimensions == 512);
static_assert(HordeLegacyNetwork::NetworkInputs
              == COLOR_NB * HordeLegacyNetwork::AccumulatorDimensions);
static_assert(HordeLegacyNetwork::PsqtBuckets == HordeLegacyNetwork::LayerStacks);
static_assert(std::is_trivially_copyable_v<AccumulatorState>);

}  // namespace Stockfish::Eval::NNUE

#endif  // NNUE_ACCUMULATOR_H_INCLUDED
