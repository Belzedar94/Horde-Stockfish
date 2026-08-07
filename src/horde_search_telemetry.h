/*
  Horde-Stockfish, a UCI Horde chess engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_SEARCH_TELEMETRY_H_INCLUDED
#define HORDE_SEARCH_TELEMETRY_H_INCLUDED

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <ostream>

#include "position.h"
#include "types.h"

namespace Stockfish::Search {

#define HORDE_SEARCH_COUNTERS(X)     \
    X(nodes)                         \
    X(legalMoves)                    \
    X(searchedMoves)                 \
    X(failHighs)                     \
    X(bestMoveSamples)               \
    X(bestMoveRankSum)               \
    X(nmpConsidered)                 \
    X(nmpTried)                      \
    X(nmpCutoffs)                    \
    X(nmpPawnOnlyBlocked)            \
    X(probCutTried)                  \
    X(probCutMoves)                  \
    X(probCutCutoffs)                \
    X(razorCuts)                     \
    X(nodeFutilityCuts)              \
    X(lmpTriggered)                  \
    X(lmrSearches)                   \
    X(lmrReductions)                 \
    X(lmrResearches)                 \
    X(pvResearches)                  \
    X(captureFutilityPrunes)         \
    X(captureSeePrunes)              \
    X(quietHistoryPrunes)            \
    X(quietFutilityPrunes)           \
    X(quietSeePrunes)                \
    X(qNodes)                        \
    X(qStandPatCutoffs)              \
    X(qMoveCountPrunes)              \
    X(qFutilityPrunes)               \
    X(qNonCapturePrunes)             \
    X(qSeePrunes)                    \
    X(extinctionCapturesSeen)        \
    X(extinctionCapturesSearched)    \
    X(quietPawnPrunes)               \
    X(quietPawnSkipCandidates)       \
    X(fortressSamples)               \
    X(fortressNanoseconds)

struct HordeSearchMetrics {
#define HORDE_DECLARE_COUNTER(name) u64 name = 0;
    HORDE_SEARCH_COUNTERS(HORDE_DECLARE_COUNTER)
#undef HORDE_DECLARE_COUNTER

    void merge(const HordeSearchMetrics& other) {
#define HORDE_MERGE_COUNTER(name) name += other.name;
        HORDE_SEARCH_COUNTERS(HORDE_MERGE_COUNTER)
#undef HORDE_MERGE_COUNTER
    }
};

enum HordeSearchExperiment : u64 {
    HordeDisableNmp             = 1ULL << 0,
    HordeDisableProbCut         = 1ULL << 1,
    HordeDisableLmp             = 1ULL << 2,
    HordeDisableNodeFutility    = 1ULL << 3,
    HordeDisableCaptureFutility = 1ULL << 4,
    HordeDisableCaptureSee      = 1ULL << 5,
    HordeDisableQuietHistory    = 1ULL << 6,
    HordeDisableQuietFutility   = 1ULL << 7,
    HordeDisableQuietSee        = 1ULL << 8,
    HordeDisableQsearchPruning  = 1ULL << 9,
    HordeDisableLmr             = 1ULL << 10,
    HordeDisableRazoring        = 1ULL << 11,
    HordeEnableWhitePawnNmp      = 1ULL << 12,
    HordeDisableWhitePawnPruning = 1ULL << 13,
    HordeDisableOneKingSingular  = 1ULL << 14,
    HordeExperimentMaskMax       = (1ULL << 15) - 1,
};

class HordeSearchTelemetry {
   public:
    static constexpr std::size_t DepthBuckets = 16;
    static constexpr std::size_t PieceBuckets = 9;

    void reset(bool enabled) {
        enabled_       = enabled;
        sampleSequence = 0;
        for (auto& byDepth : cells)
            for (auto& byPieces : byDepth)
                for (auto& metrics : byPieces)
                    metrics = {};
    }

    bool enabled() const { return enabled_; }

    HordeSearchMetrics* enter(const Position& pos, Depth depth) {
        if (!enabled_)
            return nullptr;

        auto& metrics = cells[pos.side_to_move()][depth_bucket(depth)]
                             [piece_bucket(popcount(pos.pieces(WHITE)))];
        ++metrics.nodes;

        // The exact fortress predicate is deliberately sampled because it can
        // generate moves for both sides. Instrumented builds trade speed for a
        // direct cost measurement without changing the searched value.
        if ((++sampleSequence & 1023) == 0)
        {
            const auto start = std::chrono::steady_clock::now();
            (void) pos.horde_is_fortress();
            const auto stop = std::chrono::steady_clock::now();
            ++metrics.fortressSamples;
            metrics.fortressNanoseconds +=
              std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count();
        }

        return &metrics;
    }

    void merge(const HordeSearchTelemetry& other) {
        if (!other.enabled_)
            return;
        enabled_ = true;
        for (std::size_t color = 0; color < COLOR_NB; ++color)
            for (std::size_t depth = 0; depth < DepthBuckets; ++depth)
                for (std::size_t pieces = 0; pieces < PieceBuckets; ++pieces)
                    cells[color][depth][pieces].merge(other.cells[color][depth][pieces]);
    }

    void write(std::ostream& output,
               std::size_t   threads,
               TimePoint     elapsedMs,
               u64           experimentMask) const {
        HordeSearchMetrics total;
        for (const auto& byDepth : cells)
            for (const auto& byPieces : byDepth)
                for (const auto& metrics : byPieces)
                    total.merge(metrics);

        output << "info string horde telemetry schema=1 threads=" << threads
               << " elapsed_ms=" << elapsedMs << " experiment_mask=" << experimentMask
               << " nodes=" << total.nodes << '\n';

        for (std::size_t color = 0; color < COLOR_NB; ++color)
            for (std::size_t depth = 0; depth < DepthBuckets; ++depth)
                for (std::size_t pieces = 0; pieces < PieceBuckets; ++pieces)
                {
                    const auto& metrics = cells[color][depth][pieces];
                    if (!metrics.nodes)
                        continue;
                    output << "info string horde telemetry side="
                           << (color == WHITE ? "white" : "black") << " depth="
                           << depth_label(depth) << " white_pieces=" << piece_label(pieces);
#define HORDE_WRITE_COUNTER(name) output << " " #name "=" << metrics.name;
                    HORDE_SEARCH_COUNTERS(HORDE_WRITE_COUNTER)
#undef HORDE_WRITE_COUNTER
                    output << '\n';
                }
    }

   private:
    static std::size_t depth_bucket(Depth depth) {
        return std::min<std::size_t>(std::max(int(depth), 0), DepthBuckets - 1);
    }

    static std::size_t piece_bucket(int pieces) {
        constexpr std::array<int, PieceBuckets - 1> UpperBounds = {0, 1, 2, 4, 8, 16, 24, 36};
        for (std::size_t bucket = 0; bucket < UpperBounds.size(); ++bucket)
            if (pieces <= UpperBounds[bucket])
                return bucket;
        return PieceBuckets - 1;
    }

    static const char* depth_label(std::size_t bucket) {
        static constexpr std::array<const char*, DepthBuckets> Labels = {
          "0", "1", "2", "3", "4", "5", "6", "7",
          "8", "9", "10", "11", "12", "13", "14", "15+"};
        return Labels[bucket];
    }

    static const char* piece_label(std::size_t bucket) {
        static constexpr std::array<const char*, PieceBuckets> Labels = {
          "0", "1", "2", "3-4", "5-8", "9-16", "17-24", "25-36", "37+"};
        return Labels[bucket];
    }

    bool enabled_ = false;
    u64  sampleSequence = 0;
    std::array<std::array<std::array<HordeSearchMetrics, PieceBuckets>, DepthBuckets>, COLOR_NB>
      cells{};
};

#undef HORDE_SEARCH_COUNTERS

}  // namespace Stockfish::Search

#endif  // HORDE_SEARCH_TELEMETRY_H_INCLUDED
