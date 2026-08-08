/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_V2_SCALAR_H_INCLUDED
#define HORDE_V2_SCALAR_H_INCLUDED

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "horde_v2_full_refresh.h"

namespace Stockfish::Eval::NNUE::HordeV2 {

// V2_BASE_P0 is an engineering-only integer reference. It deliberately has no
// production dispatch, file parser, incremental accumulator, or SIMD path.
// Those consumers must match this discrete full-refresh contract before a V2
// network can become eligible for strength testing.
inline constexpr IndexType RoyalLanes       = 256;
inline constexpr IndexType GlobalLanes      = 256;
inline constexpr IndexType TransformedLanes = RoyalLanes + GlobalLanes;
inline constexpr IndexType Hidden0Lanes     = 32;
inline constexpr IndexType Hidden1Lanes     = 32;
inline constexpr IndexType OutputHeads      = COLOR_NB;

inline constexpr int FtActivationShift    = 6;
inline constexpr int DenseActivationShift = 6;
inline constexpr int ActivationMax        = 127;
inline constexpr int ScalarOutputScale    = 16;
inline constexpr i32 MaxSafeBiasMagnitude = i32(1) << 30;

using FtWeight    = i16;
using DenseWeight = std::int8_t;
using AffineBias  = i32;
using Accumulator = i32;
using Activation  = u8;

inline constexpr std::size_t RoyalWeightCount =
  std::size_t(RoyalPieceSquareDimensions) * RoyalLanes;
inline constexpr std::size_t GlobalWeightCount =
  std::size_t(FixedRolePieceSquareDimensions) * GlobalLanes;
inline constexpr std::size_t Hidden0WeightCount = std::size_t(Hidden0Lanes) * TransformedLanes;
inline constexpr std::size_t Hidden1WeightCount = std::size_t(Hidden1Lanes) * Hidden0Lanes;
inline constexpr std::size_t OutputWeightCount  = std::size_t(OutputHeads) * Hidden1Lanes;
inline constexpr std::size_t ScalarParameterBytes =
  RoyalWeightCount * sizeof(FtWeight) + GlobalWeightCount * sizeof(FtWeight)
  + (RoyalLanes + GlobalLanes) * sizeof(AffineBias)
  + (Hidden0WeightCount + Hidden1WeightCount + OutputWeightCount) * sizeof(DenseWeight)
  + (Hidden0Lanes + Hidden1Lanes + OutputHeads) * sizeof(AffineBias);

constexpr Activation clipped_activation(Accumulator value, int shift) noexcept {
    if (value <= 0)
        return 0;

    const Accumulator shifted = value >> shift;
    return static_cast<Activation>(std::min<Accumulator>(shifted, ActivationMax));
}

constexpr Value apply_rule50_postprocessor(i32 value, int rule50Count) noexcept {
    const int r      = std::clamp(rule50Count, 0, 100);
    const i32 damped = static_cast<i32>((static_cast<i64>(value) * (100 - r)) / 100);
    return Value(
      std::clamp(damped, int(VALUE_TB_LOSS_IN_MAX_PLY) + 1, int(VALUE_TB_WIN_IN_MAX_PLY) - 1));
}

struct ScalarParameters {
    std::array<AffineBias, RoyalLanes>  royalBias{};
    std::vector<FtWeight>               royalWeights;
    std::array<AffineBias, GlobalLanes> globalBias{};
    std::vector<FtWeight>               globalWeights;

    std::array<AffineBias, Hidden0Lanes> hidden0Bias{};
    std::vector<DenseWeight>             hidden0Weights;
    std::array<AffineBias, Hidden1Lanes> hidden1Bias{};
    std::vector<DenseWeight>             hidden1Weights;
    std::array<AffineBias, OutputHeads>  outputBias{};
    std::vector<DenseWeight>             outputWeights;

    ScalarParameters() :
        royalWeights(RoyalWeightCount),
        globalWeights(GlobalWeightCount),
        hidden0Weights(Hidden0WeightCount),
        hidden1Weights(Hidden1WeightCount),
        outputWeights(OutputWeightCount) {}

    [[nodiscard]] bool valid() const noexcept {
        if (royalWeights.size() != RoyalWeightCount || globalWeights.size() != GlobalWeightCount
            || hidden0Weights.size() != Hidden0WeightCount
            || hidden1Weights.size() != Hidden1WeightCount
            || outputWeights.size() != OutputWeightCount)
            return false;

        const auto biasIsSafe = [](AffineBias bias) {
            return bias >= -MaxSafeBiasMagnitude && bias <= MaxSafeBiasMagnitude;
        };
        return std::all_of(royalBias.begin(), royalBias.end(), biasIsSafe)
            && std::all_of(globalBias.begin(), globalBias.end(), biasIsSafe)
            && std::all_of(hidden0Bias.begin(), hidden0Bias.end(), biasIsSafe)
            && std::all_of(hidden1Bias.begin(), hidden1Bias.end(), biasIsSafe)
            && std::all_of(outputBias.begin(), outputBias.end(), biasIsSafe);
    }
};

namespace Detail {

inline u64 splitmix64(u64& state) noexcept {
    u64 z = (state += 0x9E3779B97F4A7C15ULL);
    z     = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z     = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

template<typename T>
inline T bounded_signed(u64& state, int radius) noexcept {
    const u64 width = u64(2 * radius + 1);
    return static_cast<T>(int(splitmix64(state) % width) - radius);
}

template<typename Container, typename T>
inline void fill_bounded(Container& values, u64& state, int radius) noexcept {
    for (auto& value : values)
        value = bounded_signed<T>(state, radius);
}

}  // namespace Detail

// The deterministic payload is intentionally non-zero and bounded. It is a
// parity fixture, not a trained evaluator and not a candidate network file.
inline ScalarParameters make_deterministic_parameters(u64 seed) {
    ScalarParameters parameters;
    u64              state = seed;

    for (AffineBias& bias : parameters.royalBias)
        bias = Detail::bounded_signed<AffineBias>(state, 6144) + 4096;
    Detail::fill_bounded<decltype(parameters.royalWeights), FtWeight>(parameters.royalWeights,
                                                                      state, 31);

    for (AffineBias& bias : parameters.globalBias)
        bias = Detail::bounded_signed<AffineBias>(state, 6144) + 4096;
    Detail::fill_bounded<decltype(parameters.globalWeights), FtWeight>(parameters.globalWeights,
                                                                       state, 31);

    Detail::fill_bounded<decltype(parameters.hidden0Bias), AffineBias>(parameters.hidden0Bias,
                                                                       state, 4096);
    Detail::fill_bounded<decltype(parameters.hidden0Weights), DenseWeight>(
      parameters.hidden0Weights, state, 7);
    Detail::fill_bounded<decltype(parameters.hidden1Bias), AffineBias>(parameters.hidden1Bias,
                                                                       state, 4096);
    Detail::fill_bounded<decltype(parameters.hidden1Weights), DenseWeight>(
      parameters.hidden1Weights, state, 7);
    Detail::fill_bounded<decltype(parameters.outputBias), AffineBias>(parameters.outputBias, state,
                                                                      4096);
    Detail::fill_bounded<decltype(parameters.outputWeights), DenseWeight>(parameters.outputWeights,
                                                                          state, 7);

    return parameters;
}

enum class ScalarEvalError {
    NONE,
    INVALID_PARAMETERS,
    INVALID_POSITION,
    INVALID_SIDE_TO_MOVE,
    INVALID_SOURCE_TRACE,
    INVALID_DIRTY_PIECE,
    SOURCE_POSITION_MISMATCH,
    DIRTY_BOARD_MISMATCH,
    STACK_UNINITIALIZED,
    STACK_OVERFLOW,
};

struct ScalarTrace {
    std::array<Accumulator, RoyalLanes>      royalAccumulator{};
    std::array<Accumulator, GlobalLanes>     globalAccumulator{};
    std::array<Activation, TransformedLanes> transformed{};
    std::array<Accumulator, Hidden0Lanes>    hidden0Affine{};
    std::array<Activation, Hidden0Lanes>     hidden0{};
    std::array<Accumulator, Hidden1Lanes>    hidden1Affine{};
    std::array<Activation, Hidden1Lanes>     hidden1{};
    Accumulator                              outputAffine   = 0;
    i32                                      preRule50Value = 0;
    Value                                    value          = VALUE_ZERO;
    RoyalKey                                 royalKey{RoyalBucketCount, false};
    bool                                     royalRefreshed = false;
    FullRefreshError                         featureError   = FullRefreshError::NONE;
    ScalarEvalError                          error          = ScalarEvalError::NONE;

    [[nodiscard]] constexpr bool valid() const noexcept { return error == ScalarEvalError::NONE; }
};

class ScalarNetwork {
   public:
    explicit ScalarNetwork(ScalarParameters parameters) :
        parameters_(std::move(parameters)) {}

    [[nodiscard]] const ScalarParameters& parameters() const noexcept { return parameters_; }

    [[nodiscard]] ScalarTrace evaluate_full_refresh(const std::array<Piece, SQUARE_NB>& board,
                                                    Color                               sideToMove,
                                                    int rule50Count) const noexcept {
        ScalarTrace trace{};
        if (const ScalarEvalError error = common_error(sideToMove); error != ScalarEvalError::NONE)
        {
            trace.error = error;
            return trace;
        }

        const FullRefreshFeatures features = extract_full_refresh_features(board);
        if (!features.valid())
        {
            trace.error        = ScalarEvalError::INVALID_POSITION;
            trace.featureError = features.error;
            return trace;
        }

        trace.royalAccumulator  = parameters_.royalBias;
        trace.globalAccumulator = parameters_.globalBias;
        trace.royalKey          = features.royalKey;
        trace.royalRefreshed    = true;

        refresh_royal(features, trace.royalAccumulator);
        for (std::size_t active = 0; active < features.globalSize; ++active)
        {
            const std::size_t offset = std::size_t(features.global[active]) * GlobalLanes;
            for (IndexType lane = 0; lane < GlobalLanes; ++lane)
                trace.globalAccumulator[lane] += parameters_.globalWeights[offset + lane];
        }

        return propagate(std::move(trace), sideToMove, rule50Count);
    }

    [[nodiscard]] ScalarTrace evaluate_incremental(const DirtyPiece&                   dirty,
                                                   const std::array<Piece, SQUARE_NB>& targetBoard,
                                                   const ScalarTrace&                  source,
                                                   Color                               sideToMove,
                                                   int rule50Count) const noexcept {
        ScalarTrace trace{};
        if (const ScalarEvalError error = common_error(sideToMove); error != ScalarEvalError::NONE)
        {
            trace.error = error;
            return trace;
        }
        if (!source.valid() || !is_valid_royal_key(source.royalKey))
        {
            trace.error = ScalarEvalError::INVALID_SOURCE_TRACE;
            return trace;
        }

        const FullRefreshFeatures features = extract_full_refresh_features(targetBoard);
        if (!features.valid())
        {
            trace.error        = ScalarEvalError::INVALID_POSITION;
            trace.featureError = features.error;
            return trace;
        }
        if (!valid_dirty_piece(dirty))
        {
            trace.error = ScalarEvalError::INVALID_DIRTY_PIECE;
            return trace;
        }

        trace.globalAccumulator = source.globalAccumulator;
        trace.royalAccumulator  = source.royalAccumulator;
        trace.royalKey          = features.royalKey;

        if (!apply_global_delta(trace.globalAccumulator, dirty.pc, dirty.from, -1)
            || (dirty.to != SQ_NONE
                && !apply_global_delta(trace.globalAccumulator, dirty.pc, dirty.to, 1))
            || (dirty.remove_sq != SQ_NONE
                && !apply_global_delta(trace.globalAccumulator, dirty.remove_pc, dirty.remove_sq,
                                       -1))
            || (dirty.add_sq != SQ_NONE
                && !apply_global_delta(trace.globalAccumulator, dirty.add_pc, dirty.add_sq, 1)))
        {
            trace.error = ScalarEvalError::INVALID_DIRTY_PIECE;
            return trace;
        }

        if (trace.royalKey != source.royalKey)
        {
            trace.royalAccumulator = parameters_.royalBias;
            refresh_royal(features, trace.royalAccumulator);
            trace.royalRefreshed = true;
        }
        else
        {
            const auto applyRoyal = [&](Piece piece, Square square, int sign) {
                return piece == B_KING
                    || apply_royal_delta(trace.royalAccumulator, piece, square, trace.royalKey,
                                         sign);
            };
            if (!applyRoyal(dirty.pc, dirty.from, -1)
                || (dirty.to != SQ_NONE && !applyRoyal(dirty.pc, dirty.to, 1))
                || (dirty.remove_sq != SQ_NONE && !applyRoyal(dirty.remove_pc, dirty.remove_sq, -1))
                || (dirty.add_sq != SQ_NONE && !applyRoyal(dirty.add_pc, dirty.add_sq, 1)))
            {
                trace.error = ScalarEvalError::INVALID_DIRTY_PIECE;
                return trace;
            }
        }

        return propagate(std::move(trace), sideToMove, rule50Count);
    }

    [[nodiscard]] ScalarTrace evaluate_incremental(const DirtyPiece&  dirty,
                                                   const Position&    target,
                                                   const ScalarTrace& source) const noexcept {
        return evaluate_incremental(dirty, target.piece_array(), source, target.side_to_move(),
                                    target.rule50_count());
    }

    // Null moves do not push the engine's NNUE accumulator stack because the
    // physical board is unchanged. Re-run only the dense path so the selected
    // STM head and rule50 postprocessor still follow the live Position.
    [[nodiscard]] ScalarTrace evaluate_from_accumulators(const ScalarTrace& source,
                                                         Color              sideToMove,
                                                         int rule50Count) const noexcept {
        ScalarTrace trace{};
        if (const ScalarEvalError error = common_error(sideToMove); error != ScalarEvalError::NONE)
        {
            trace.error = error;
            return trace;
        }
        if (!source.valid() || !is_valid_royal_key(source.royalKey))
        {
            trace.error = ScalarEvalError::INVALID_SOURCE_TRACE;
            return trace;
        }

        trace.royalAccumulator  = source.royalAccumulator;
        trace.globalAccumulator = source.globalAccumulator;
        trace.royalKey          = source.royalKey;
        trace.royalRefreshed    = false;
        return propagate(std::move(trace), sideToMove, rule50Count);
    }

    [[nodiscard]] ScalarTrace evaluate_from_accumulators(const ScalarTrace& source,
                                                         const Position&    target) const noexcept {
        return evaluate_from_accumulators(source, target.side_to_move(), target.rule50_count());
    }

    [[nodiscard]] ScalarTrace evaluate_full_refresh(const Position& pos) const noexcept {
        return evaluate_full_refresh(pos.piece_array(), pos.side_to_move(), pos.rule50_count());
    }

   private:
    [[nodiscard]] ScalarEvalError common_error(Color sideToMove) const noexcept {
        if (!parameters_.valid())
            return ScalarEvalError::INVALID_PARAMETERS;
        if (sideToMove != WHITE && sideToMove != BLACK)
            return ScalarEvalError::INVALID_SIDE_TO_MOVE;
        return ScalarEvalError::NONE;
    }

    [[nodiscard]] static bool valid_dirty_piece(const DirtyPiece& dirty) noexcept {
        return is_registered_piece(dirty.pc) && is_ok(dirty.from)
            && (dirty.to == SQ_NONE || is_ok(dirty.to))
            && (dirty.remove_sq == SQ_NONE
                || (is_ok(dirty.remove_sq) && is_registered_piece(dirty.remove_pc)))
            && (dirty.add_sq == SQ_NONE
                || (is_ok(dirty.add_sq) && is_registered_piece(dirty.add_pc)));
    }

    [[nodiscard]] bool apply_global_delta(std::array<Accumulator, GlobalLanes>& accumulator,
                                          Piece                                 piece,
                                          Square                                square,
                                          int sign) const noexcept {
        const IndexType index = fixed_role_piece_square_index(piece, square);
        if (index == InvalidFeatureIndex)
            return false;

        const std::size_t offset = std::size_t(index) * GlobalLanes;
        for (IndexType lane = 0; lane < GlobalLanes; ++lane)
            accumulator[lane] += sign * parameters_.globalWeights[offset + lane];
        return true;
    }

    [[nodiscard]] bool apply_royal_delta(std::array<Accumulator, RoyalLanes>& accumulator,
                                         Piece                                piece,
                                         Square                               square,
                                         RoyalKey                             key,
                                         int                                  sign) const noexcept {
        const IndexType index = royal_piece_square_index(piece, square, key);
        if (index == InvalidRoyalFeatureIndex)
            return false;

        const std::size_t offset = std::size_t(index) * RoyalLanes;
        for (IndexType lane = 0; lane < RoyalLanes; ++lane)
            accumulator[lane] += sign * parameters_.royalWeights[offset + lane];
        return true;
    }

    void refresh_royal(const FullRefreshFeatures&           features,
                       std::array<Accumulator, RoyalLanes>& accumulator) const noexcept {
        for (std::size_t active = 0; active < features.royalSize; ++active)
        {
            const std::size_t offset = std::size_t(features.royal[active]) * RoyalLanes;
            for (IndexType lane = 0; lane < RoyalLanes; ++lane)
                accumulator[lane] += parameters_.royalWeights[offset + lane];
        }
    }

    [[nodiscard]] ScalarTrace
    propagate(ScalarTrace trace, Color sideToMove, int rule50Count) const noexcept {
        for (IndexType lane = 0; lane < RoyalLanes; ++lane)
            trace.transformed[lane] =
              clipped_activation(trace.royalAccumulator[lane], FtActivationShift);
        for (IndexType lane = 0; lane < GlobalLanes; ++lane)
            trace.transformed[RoyalLanes + lane] =
              clipped_activation(trace.globalAccumulator[lane], FtActivationShift);

        for (IndexType output = 0; output < Hidden0Lanes; ++output)
        {
            Accumulator       sum    = parameters_.hidden0Bias[output];
            const std::size_t offset = std::size_t(output) * TransformedLanes;
            for (IndexType input = 0; input < TransformedLanes; ++input)
                sum += Accumulator(parameters_.hidden0Weights[offset + input])
                     * trace.transformed[input];
            trace.hidden0Affine[output] = sum;
            trace.hidden0[output]       = clipped_activation(sum, DenseActivationShift);
        }

        for (IndexType output = 0; output < Hidden1Lanes; ++output)
        {
            Accumulator       sum    = parameters_.hidden1Bias[output];
            const std::size_t offset = std::size_t(output) * Hidden0Lanes;
            for (IndexType input = 0; input < Hidden0Lanes; ++input)
                sum +=
                  Accumulator(parameters_.hidden1Weights[offset + input]) * trace.hidden0[input];
            trace.hidden1Affine[output] = sum;
            trace.hidden1[output]       = clipped_activation(sum, DenseActivationShift);
        }

        const IndexType   head   = IndexType(sideToMove);
        const std::size_t offset = std::size_t(head) * Hidden1Lanes;
        trace.outputAffine       = parameters_.outputBias[head];
        for (IndexType input = 0; input < Hidden1Lanes; ++input)
            trace.outputAffine +=
              Accumulator(parameters_.outputWeights[offset + input]) * trace.hidden1[input];

        // C++ signed division truncates toward zero. The trainer parity path
        // must reproduce this operation exactly for negative values.
        trace.preRule50Value = trace.outputAffine / ScalarOutputScale;
        trace.value          = apply_rule50_postprocessor(trace.preRule50Value, rule50Count);
        return trace;
    }

    ScalarParameters parameters_;
};

static_assert(RoyalLanes == 256);
static_assert(GlobalLanes == 256);
static_assert(TransformedLanes == 512);
static_assert(Hidden0Lanes == 32 && Hidden1Lanes == 32);
static_assert(sizeof(FtWeight) == 2);
static_assert(sizeof(DenseWeight) == 1);
static_assert(sizeof(Accumulator) == 4);
static_assert(RoyalWeightCount == 5242880);
static_assert(GlobalWeightCount == 180224);
static_assert(Hidden0WeightCount == 16384);
static_assert(Hidden1WeightCount == 1024);
static_assert(OutputWeightCount == 64);
static_assert(ScalarParameterBytes == 10865992);
static_assert(clipped_activation(-1, FtActivationShift) == 0);
static_assert(clipped_activation(64, FtActivationShift) == 1);
static_assert(clipped_activation(64 * 127, FtActivationShift) == 127);
static_assert(clipped_activation(64 * 128, FtActivationShift) == 127);
static_assert(apply_rule50_postprocessor(101, 50) == Value(50));
static_assert(apply_rule50_postprocessor(-101, 50) == Value(-50));
static_assert(apply_rule50_postprocessor(101, 100) == VALUE_ZERO);

}  // namespace Stockfish::Eval::NNUE::HordeV2

#endif  // HORDE_V2_SCALAR_H_INCLUDED
