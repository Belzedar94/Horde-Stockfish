/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_V2_PERFORMANCE_H_INCLUDED
#define HORDE_V2_PERFORMANCE_H_INCLUDED

#if !defined(HORDE_V2_PERF_ROYAL_WIDTH) || !defined(HORDE_V2_PERF_GLOBAL_WIDTH)
    #error "HORDE_V2 performance builds require explicit Royal and Global widths"
#endif

#include "horde_v2_stack.h"

namespace Stockfish::Eval::NNUE::HordeV2 {

inline constexpr IndexType PerformanceRoyalWidth  = HORDE_V2_PERF_ROYAL_WIDTH;
inline constexpr IndexType PerformanceGlobalWidth = HORDE_V2_PERF_GLOBAL_WIDTH;

inline constexpr bool ApprovedPerformanceWidth =
  (PerformanceRoyalWidth == 256 && PerformanceGlobalWidth == 256)
  || (PerformanceRoyalWidth == 128 && PerformanceGlobalWidth == 256)
  || (PerformanceRoyalWidth == 128 && PerformanceGlobalWidth == 128)
  || (PerformanceRoyalWidth == 64 && PerformanceGlobalWidth == 192);
static_assert(ApprovedPerformanceWidth, "unregistered Horde V2 performance width");

using PerformanceWidth   = WidthPoint<PerformanceRoyalWidth, PerformanceGlobalWidth>;
using PerformanceNetwork = LeanNetwork<PerformanceWidth>;
using PerformanceStack   = LeanAccumulatorStack<PerformanceWidth>;

inline constexpr u64 PerformanceFixtureSeed = 0x4856325F50455246ULL;

inline const PerformanceNetwork& performance_network() {
    static const PerformanceNetwork network(make_lean_parameters<PerformanceWidth>(
      PerformanceFixtureSeed, FixturePayload::PERF_COMMON_V1));
    return network;
}

}  // namespace Stockfish::Eval::NNUE::HordeV2

#endif  // HORDE_V2_PERFORMANCE_H_INCLUDED
