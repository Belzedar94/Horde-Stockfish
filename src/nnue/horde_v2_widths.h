/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_V2_WIDTHS_H_INCLUDED
#define HORDE_V2_WIDTHS_H_INCLUDED

#include <cstddef>
#include <cstdint>

#include "horde_v2_features.h"

namespace Stockfish::Eval::NNUE::HordeV2 {

inline constexpr IndexType V2Hidden0Lanes = 32;
inline constexpr IndexType V2Hidden1Lanes = 32;
inline constexpr IndexType V2OutputHeads  = COLOR_NB;

// WidthPoint is the single source of truth for architecture footprint. It
// deliberately excludes container capacity, alignment padding, accumulator
// frames and caches: ParameterBytes is the serialized quantized payload only.
template<IndexType RoyalWidth, IndexType GlobalWidth>
struct WidthPoint {
    static_assert(RoyalWidth > 0 && GlobalWidth > 0);
    static_assert(RoyalWidth % 32 == 0 && GlobalWidth % 32 == 0);

    static constexpr IndexType RoyalLanes       = RoyalWidth;
    static constexpr IndexType GlobalLanes      = GlobalWidth;
    static constexpr IndexType TransformedLanes = RoyalLanes + GlobalLanes;
    static constexpr IndexType Hidden0Lanes     = V2Hidden0Lanes;
    static constexpr IndexType Hidden1Lanes     = V2Hidden1Lanes;
    static constexpr IndexType OutputHeads      = V2OutputHeads;

    static constexpr std::size_t RoyalWeightCount =
      std::size_t(RoyalPieceSquareDimensions) * RoyalLanes;
    static constexpr std::size_t GlobalWeightCount =
      std::size_t(FixedRolePieceSquareDimensions) * GlobalLanes;
    static constexpr std::size_t Hidden0WeightCount = std::size_t(Hidden0Lanes) * TransformedLanes;
    static constexpr std::size_t Hidden1WeightCount = std::size_t(Hidden1Lanes) * Hidden0Lanes;
    static constexpr std::size_t OutputWeightCount  = std::size_t(OutputHeads) * Hidden1Lanes;

    static constexpr std::size_t RoyalTableBytes  = RoyalWeightCount * sizeof(std::int16_t);
    static constexpr std::size_t GlobalTableBytes = GlobalWeightCount * sizeof(std::int16_t);
    static constexpr std::size_t TransformerBiasBytes =
      std::size_t(RoyalLanes + GlobalLanes) * sizeof(std::int32_t);
    static constexpr std::size_t DenseWeightBytes =
      (Hidden0WeightCount + Hidden1WeightCount + OutputWeightCount) * sizeof(std::int8_t);
    static constexpr std::size_t DenseBiasBytes =
      std::size_t(Hidden0Lanes + Hidden1Lanes + OutputHeads) * sizeof(std::int32_t);
    static constexpr std::size_t ParameterBytes =
      RoyalTableBytes + GlobalTableBytes + TransformerBiasBytes + DenseWeightBytes + DenseBiasBytes;
    static constexpr std::size_t AccumulatorBytes =
      std::size_t(RoyalLanes + GlobalLanes) * sizeof(std::int32_t);
    static constexpr std::size_t DenseMacs =
      Hidden0WeightCount + Hidden1WeightCount + OutputWeightCount;

    template<std::size_t RoyalRows, std::size_t GlobalRows>
    static constexpr std::size_t sparse_weight_bytes() noexcept {
        return RoyalRows * RoyalLanes * sizeof(std::int16_t)
             + GlobalRows * GlobalLanes * sizeof(std::int16_t);
    }
};

using Width256x256 = WidthPoint<256, 256>;
using Width128x256 = WidthPoint<128, 256>;
using Width128x128 = WidthPoint<128, 128>;
using Width64x192  = WidthPoint<64, 192>;

static_assert(Width256x256::ParameterBytes == 10865992);
static_assert(Width128x256::ParameterBytes == 5618504);
static_assert(Width128x128::ParameterBytes == 5433672);
static_assert(Width64x192::ParameterBytes == 2902344);

static_assert(Width256x256::DenseMacs == 17472);
static_assert(Width128x256::DenseMacs == 13376);
static_assert(Width128x128::DenseMacs == 9280);
static_assert(Width64x192::DenseMacs == 9280);

}  // namespace Stockfish::Eval::NNUE::HordeV2

#endif  // HORDE_V2_WIDTHS_H_INCLUDED
