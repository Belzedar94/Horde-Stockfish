/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_V3_SCALAR_H_INCLUDED
#define HORDE_V3_SCALAR_H_INCLUDED

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>

#include "horde_v2_backend.h"
#include "horde_v3_features.h"

// The exact integer forward of one V3 network, and the frame layout every
// backend shares.
//
//   acc      = ft_bias + sum of ft_weights[row] over the active rows
//   act      = clip(max(acc, 0) >> 6, 0, 127)
//   bucket   = phase_lookup[white_piece_count]
//   h0_pre   = hidden0_bias[bucket*16+j] + sum_k hidden0_weights[bucket*16+j,k] * act[k]
//   h0       = clip(max(h0_pre, 0) >> 6, 0, 127)
//   h1_pre   = hidden1_bias[bucket*32+j] + sum_k hidden1_weights[bucket*32+j,k] * h0[k]
//   h1       = clip(max(h1_pre, 0) >> 6, 0, 127)
//   out_pre  = output_bias[bucket*2+stm] + sum_k output_weights[bucket*2+stm,k] * h1[k]
//   psqt_sum = sum over the SAME active rows of psqt_weights[row, bucket*2+stm]
//   value    = trunc_toward_zero(out_pre + psqt_sum, 16)
//   damped   = trunc_toward_zero(value * (100 - min(rule50, 100)), 100)
//   result   = clamp(damped, -31506, 31506)
//
// Every int32 sum is proved in docs/horde/nnue-v3-integer-bounds.md to keep a
// factor near two of headroom, so accumulation is non-saturating throughout.

namespace Stockfish::Eval::NNUE::HordeV3 {

using HordeV2::AffineBias;
using HordeV2::AlignedBuffer;
using HordeV2::apply_rule50_postprocessor;
using HordeV2::clipped_activation;
using HordeV2::DenseWeight;
using HordeV2::FtWeight;
using HordeV2::NormalizedDirty;
using HordeV2::normalize_dirty_piece;

using Accumulator = HordeV2::Accumulator;
using Activation  = HordeV2::Activation;
using PsqtWeight  = i32;

// ---------------------------------------------------------------------------
// Frozen topology and integer contract, mirroring tools/horde_v3_container.py
// ---------------------------------------------------------------------------

inline constexpr std::size_t V3Lanes            = 1024;
inline constexpr std::size_t V3Hidden0Lanes     = 16;
inline constexpr std::size_t V3Hidden1Lanes     = 32;
inline constexpr std::size_t V3OutputHeads      = 2;
inline constexpr std::size_t V3PhaseBuckets     = 8;
inline constexpr std::size_t V3PsqtColumns      = V3PhaseBuckets * V3OutputHeads;  // 16
inline constexpr std::size_t V3PhaseLookupSize  = 37;   // white_piece_count 0..36
inline constexpr std::size_t V3MaxWhitePieceCount = 36;

inline constexpr int FtActivationShift    = 6;
inline constexpr int DenseActivationShift = 6;
inline constexpr int ActivationMax        = 127;
inline constexpr int OutputDivisor        = 16;
inline constexpr int NetworkToScore       = 600;
inline constexpr i32 MaxSafeBiasMagnitude = i32(1) << 30;
inline constexpr i32 MaxPsqtMagnitude     = i32(1) << 20;

inline constexpr std::size_t FtWeightCount      = std::size_t(V3StreamRows) * V3Lanes;
inline constexpr std::size_t PsqtWeightCount    = std::size_t(V3StreamRows) * V3PsqtColumns;
inline constexpr std::size_t Hidden0WeightCount = V3PhaseBuckets * V3Hidden0Lanes * V3Lanes;
inline constexpr std::size_t Hidden1WeightCount = V3PhaseBuckets * V3Hidden1Lanes * V3Hidden0Lanes;
inline constexpr std::size_t OutputWeightCount  = V3PhaseBuckets * V3OutputHeads * V3Hidden1Lanes;

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

struct V3Parameters {
    std::string schemaName;
    std::string fileSha256;
    std::string parameterSha256;

    AlignedBuffer<FtWeight> ftWeights;
    alignas(64) std::array<AffineBias, V3Lanes> ftBias{};
    AlignedBuffer<PsqtWeight> psqtWeights;

    AlignedBuffer<DenseWeight> hidden0Weights;
    alignas(64) std::array<AffineBias, V3PhaseBuckets * V3Hidden0Lanes> hidden0Bias{};
    AlignedBuffer<DenseWeight> hidden1Weights;
    alignas(64) std::array<AffineBias, V3PhaseBuckets * V3Hidden1Lanes> hidden1Bias{};
    AlignedBuffer<DenseWeight> outputWeights;
    alignas(64) std::array<AffineBias, V3PhaseBuckets * V3OutputHeads> outputBias{};

    alignas(64) std::array<std::uint8_t, V3PhaseLookupSize> phaseLookup{};

    V3Parameters() :
        ftWeights(FtWeightCount),
        psqtWeights(PsqtWeightCount),
        hidden0Weights(Hidden0WeightCount),
        hidden1Weights(Hidden1WeightCount),
        outputWeights(OutputWeightCount) {}

    [[nodiscard]] bool valid() const noexcept {
        if (ftWeights.size() != FtWeightCount || psqtWeights.size() != PsqtWeightCount
            || hidden0Weights.size() != Hidden0WeightCount
            || hidden1Weights.size() != Hidden1WeightCount
            || outputWeights.size() != OutputWeightCount)
            return false;

        const auto safeBias = [](AffineBias bias) {
            return bias >= -MaxSafeBiasMagnitude && bias <= MaxSafeBiasMagnitude;
        };
        if (!std::all_of(ftBias.begin(), ftBias.end(), safeBias)
            || !std::all_of(hidden0Bias.begin(), hidden0Bias.end(), safeBias)
            || !std::all_of(hidden1Bias.begin(), hidden1Bias.end(), safeBias)
            || !std::all_of(outputBias.begin(), outputBias.end(), safeBias))
            return false;

        for (std::size_t index = 0; index < PsqtWeightCount; ++index)
        {
            const PsqtWeight weight = psqtWeights[index];
            if (weight > MaxPsqtMagnitude || weight < -MaxPsqtMagnitude)
                return false;
        }

        // The phase lookup must stay inside the eight buckets, be monotone
        // nondecreasing in white_piece_count and reach every bucket.
        std::array<bool, V3PhaseBuckets> reached{};
        for (std::size_t index = 0; index < V3PhaseLookupSize; ++index)
        {
            if (phaseLookup[index] >= V3PhaseBuckets)
                return false;
            if (index && phaseLookup[index] < phaseLookup[index - 1])
                return false;
            reached[phaseLookup[index]] = true;
        }
        return std::all_of(reached.begin(), reached.end(), [](bool value) { return value; });
    }
};

// ---------------------------------------------------------------------------
// Frame and scratch. One layout, shared by every backend.
// ---------------------------------------------------------------------------

struct alignas(64) V3AccumulatorFrame {
    alignas(64) std::array<Accumulator, V3Lanes> accumulator{};
    alignas(64) std::array<Accumulator, V3PsqtColumns> psqt{};
    V3PawnStructure pawns{};
    std::uint16_t   whitePieceCount = 0;
    std::uint16_t   blackPieceCount = 0;
    std::uint8_t    blackKings      = 0;
    bool            computed        = false;
};

struct alignas(64) V3DenseScratch {
    alignas(64) std::array<Activation, V3Lanes> transformed{};
    alignas(64) std::array<Accumulator, V3Hidden0Lanes> hidden0Affine{};
    alignas(64) std::array<Activation, V3Hidden0Lanes> hidden0{};
    alignas(64) std::array<Accumulator, V3Hidden1Lanes> hidden1Affine{};
    alignas(64) std::array<Activation, V3Hidden1Lanes> hidden1{};
};

struct V3EvalResult {
    std::size_t bucket         = 0;
    std::size_t outputRow      = 0;  // bucket * 2 + side_to_move
    Accumulator outputAffine   = 0;  // out_pre, without the PSQT skip
    Accumulator psqtSum        = 0;
    i32         preRule50Value = 0;
    Value       value          = VALUE_NONE;
};

// ---------------------------------------------------------------------------
// Scalar kernels
// ---------------------------------------------------------------------------

struct V3ScalarKernels {
    static constexpr const char* Name = "scalar";

    static void add_ft(Accumulator* accumulator, const FtWeight* row) noexcept {
        for (std::size_t lane = 0; lane < V3Lanes; ++lane)
            accumulator[lane] += Accumulator(row[lane]);
    }

    static void subtract_ft(Accumulator* accumulator, const FtWeight* row) noexcept {
        for (std::size_t lane = 0; lane < V3Lanes; ++lane)
            accumulator[lane] -= Accumulator(row[lane]);
    }

    static void add_psqt(Accumulator* psqt, const PsqtWeight* row) noexcept {
        for (std::size_t column = 0; column < V3PsqtColumns; ++column)
            psqt[column] += row[column];
    }

    static void subtract_psqt(Accumulator* psqt, const PsqtWeight* row) noexcept {
        for (std::size_t column = 0; column < V3PsqtColumns; ++column)
            psqt[column] -= row[column];
    }

    static Accumulator affine(const Activation*  input,
                              const DenseWeight* weights,
                              std::size_t        count,
                              Accumulator        bias) noexcept {
        Accumulator sum = bias;
        for (std::size_t index = 0; index < count; ++index)
            sum += Accumulator(weights[index]) * Accumulator(input[index]);
        return sum;
    }
};

// ---------------------------------------------------------------------------
// Dense propagation, shared by every backend
// ---------------------------------------------------------------------------

template<typename Kernels>
[[nodiscard]] inline V3EvalResult propagate_v3(const V3AccumulatorFrame& frame,
                                               const V3Parameters&       parameters,
                                               V3DenseScratch&           scratch,
                                               Color                     sideToMove,
                                               int                       rule50Count) noexcept {
    assert(sideToMove == WHITE || sideToMove == BLACK);
    assert(frame.whitePieceCount <= V3MaxWhitePieceCount);

    for (std::size_t lane = 0; lane < V3Lanes; ++lane)
        scratch.transformed[lane] = clipped_activation(frame.accumulator[lane], FtActivationShift);

    const std::size_t bucket = parameters.phaseLookup[frame.whitePieceCount];
    assert(bucket < V3PhaseBuckets);

    for (std::size_t output = 0; output < V3Hidden0Lanes; ++output)
    {
        const std::size_t row = bucket * V3Hidden0Lanes + output;
        const Accumulator sum =
          Kernels::affine(scratch.transformed.data(), parameters.hidden0Weights.data() + row * V3Lanes,
                          V3Lanes, parameters.hidden0Bias[row]);
        scratch.hidden0Affine[output] = sum;
        scratch.hidden0[output]       = clipped_activation(sum, DenseActivationShift);
    }

    for (std::size_t output = 0; output < V3Hidden1Lanes; ++output)
    {
        const std::size_t row = bucket * V3Hidden1Lanes + output;
        const Accumulator sum = Kernels::affine(scratch.hidden0.data(),
                                                parameters.hidden1Weights.data()
                                                  + row * V3Hidden0Lanes,
                                                V3Hidden0Lanes, parameters.hidden1Bias[row]);
        scratch.hidden1Affine[output] = sum;
        scratch.hidden1[output]       = clipped_activation(sum, DenseActivationShift);
    }

    V3EvalResult result{};
    result.bucket    = bucket;
    result.outputRow = bucket * V3OutputHeads + std::size_t(sideToMove);
    result.outputAffine =
      Kernels::affine(scratch.hidden1.data(),
                      parameters.outputWeights.data() + result.outputRow * V3Hidden1Lanes,
                      V3Hidden1Lanes, parameters.outputBias[result.outputRow]);
    result.psqtSum = frame.psqt[result.outputRow];

    // C++ integer division truncates toward zero, which is the frozen contract
    // for both conversions. The sum is proved to fit int32; the wider
    // intermediate only removes any chance of undefined behaviour.
    const i64 combined     = i64(result.outputAffine) + i64(result.psqtSum);
    result.preRule50Value  = i32(combined / OutputDivisor);
    result.value           = apply_rule50_postprocessor(result.preRule50Value, rule50Count);
    return result;
}

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

template<typename Kernels>
class V3Network {
   public:
    using Frame   = V3AccumulatorFrame;
    using Scratch = V3DenseScratch;

    explicit V3Network(const V3Parameters& parameters) :
        parameters_(parameters) {}
    V3Network(V3Parameters&&)       = delete;
    V3Network(const V3Parameters&&) = delete;

    [[nodiscard]] const V3Parameters& parameters() const noexcept { return parameters_; }
    [[nodiscard]] bool                valid() const noexcept { return parameters_.valid(); }
    [[nodiscard]] static constexpr const char* backend_name() noexcept { return Kernels::Name; }

    [[nodiscard]] std::size_t bucket_of(unsigned whitePieceCount) const noexcept {
        assert(whitePieceCount <= V3MaxWhitePieceCount);
        return parameters_.phaseLookup[whitePieceCount];
    }

    void add_row(Frame& frame, IndexType row) const noexcept {
        assert(row < V3StreamRows);
        Kernels::add_ft(frame.accumulator.data(), parameters_.ftWeights.data() + row * V3Lanes);
        Kernels::add_psqt(frame.psqt.data(), parameters_.psqtWeights.data() + row * V3PsqtColumns);
    }

    void subtract_row(Frame& frame, IndexType row) const noexcept {
        assert(row < V3StreamRows);
        Kernels::subtract_ft(frame.accumulator.data(), parameters_.ftWeights.data() + row * V3Lanes);
        Kernels::subtract_psqt(frame.psqt.data(),
                               parameters_.psqtWeights.data() + row * V3PsqtColumns);
    }

    // A complete rebuild from an enumerated stream.
    void full_refresh(Frame& frame, const V3Features& features) const noexcept {
        assert(features.valid());
        frame.accumulator = parameters_.ftBias;
        frame.psqt.fill(0);
        for (std::size_t active = 0; active < features.size; ++active)
            add_row(frame, features.rows[active]);
        frame.pawns           = features.pawns;
        frame.whitePieceCount = features.whitePieceCount;
        frame.blackPieceCount = features.blackPieceCount;
        frame.blackKings      = 1;
        frame.computed        = true;
    }

    [[nodiscard]] bool full_refresh(Frame& frame, const std::array<Piece, SQUARE_NB>& board) const
      noexcept {
        const V3Features features = extract_v3_features(board);
        if (!features.valid())
        {
            frame.computed = false;
            return false;
        }
        full_refresh(frame, features);
        return true;
    }

    [[nodiscard]] V3EvalResult propagate(const Frame& frame,
                                         Scratch&     scratch,
                                         Color        sideToMove,
                                         int          rule50Count) const noexcept {
        return propagate_v3<Kernels>(frame, parameters_, scratch, sideToMove, rule50Count);
    }

   private:
    const V3Parameters& parameters_;
};

using V3ScalarNetwork = V3Network<V3ScalarKernels>;

static_assert(V3Lanes == 1024);
static_assert(V3PsqtColumns == 16);
static_assert(FtWeightCount == 917504);
static_assert(PsqtWeightCount == 14336);
static_assert(Hidden0WeightCount == 131072);
static_assert(Hidden1WeightCount == 4096);
static_assert(OutputWeightCount == 512);
static_assert(sizeof(FtWeight) == 2 && sizeof(DenseWeight) == 1 && sizeof(PsqtWeight) == 4);
static_assert(clipped_activation(-1, FtActivationShift) == 0);
static_assert(clipped_activation(64, FtActivationShift) == 1);
static_assert(clipped_activation(64 * 128, FtActivationShift) == ActivationMax);
static_assert(apply_rule50_postprocessor(101, 50) == Value(50));
static_assert(apply_rule50_postprocessor(-101, 50) == Value(-50));
static_assert(apply_rule50_postprocessor(1 << 30, 0) == Value(31506));
static_assert(apply_rule50_postprocessor(-(1 << 30), 0) == Value(-31506));
static_assert(alignof(V3AccumulatorFrame) == 64);
static_assert(alignof(V3DenseScratch) == 64);

}  // namespace Stockfish::Eval::NNUE::HordeV3

#endif  // HORDE_V3_SCALAR_H_INCLUDED
