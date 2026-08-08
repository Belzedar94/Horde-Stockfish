/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_V2_CONTAINER_H_INCLUDED
#define HORDE_V2_CONTAINER_H_INCLUDED

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "horde_v2_stack.h"

namespace Stockfish::Eval::NNUE::HordeV2 {

inline constexpr std::size_t V2ContainerHeaderBytes = 2048;
inline constexpr std::size_t V2ContainerFirstLanes  = 64;
inline constexpr std::size_t V2ContainerGlobalLanes = 192;
inline constexpr std::size_t V2ContainerInputLanes  = 256;

enum class ContainerSchema : std::uint32_t {
    ROYAL_64X192       = 0x00010001,
    ABS_NONKING_64X192 = 0x00010002,
};

enum class FirstDomain : std::uint32_t {
    ROYAL            = 1,
    ABSOLUTE_NONKING = 2,
};

enum class ContainerLoadError {
    NONE,
    OPEN_FAILED,
    READ_FAILED,
    TRUNCATED,
    MAGIC_MISMATCH,
    HEADER_MISMATCH,
    SCHEMA_MISMATCH,
    STRUCTURE_MISMATCH,
    PROVENANCE_MISMATCH,
    DIRECTORY_MISMATCH,
    PAYLOAD_MISMATCH,
    PARAMETER_RANGE,
};

const char* container_load_error_name(ContainerLoadError error) noexcept;

struct ContainerParameters {
    ContainerSchema schema      = ContainerSchema::ROYAL_64X192;
    FirstDomain     firstDomain = FirstDomain::ROYAL;
    std::size_t     firstRows   = 0;
    std::string     schemaName;
    std::string     fileSha256;
    std::string     parameterSha256;

    alignas(64) std::array<AffineBias, V2ContainerFirstLanes> firstBias{};
    AlignedBuffer<FtWeight> firstWeights;
    alignas(64) std::array<AffineBias, V2ContainerGlobalLanes> globalBias{};
    AlignedBuffer<FtWeight> globalWeights;

    alignas(64) std::array<AffineBias, Width64x192::Hidden0Lanes> hidden0Bias{};
    AlignedBuffer<DenseWeight> hidden0Weights;
    alignas(64) std::array<AffineBias, Width64x192::Hidden1Lanes> hidden1Bias{};
    AlignedBuffer<DenseWeight> hidden1Weights;
    alignas(64) std::array<AffineBias, Width64x192::OutputHeads> outputBias{};
    AlignedBuffer<DenseWeight> outputWeights;

    explicit ContainerParameters(std::size_t firstRows_ = 0) :
        firstRows(firstRows_),
        firstWeights(firstRows_ * V2ContainerFirstLanes),
        globalWeights(FixedRolePieceSquareDimensions * V2ContainerGlobalLanes),
        hidden0Weights(Width64x192::Hidden0WeightCount),
        hidden1Weights(Width64x192::Hidden1WeightCount),
        outputWeights(Width64x192::OutputWeightCount) {}

    [[nodiscard]] bool valid() const noexcept {
        const bool schemaRows =
          (schema == ContainerSchema::ROYAL_64X192 && firstDomain == FirstDomain::ROYAL
           && firstRows == RoyalPieceSquareDimensions)
          || (schema == ContainerSchema::ABS_NONKING_64X192
              && firstDomain == FirstDomain::ABSOLUTE_NONKING
              && firstRows == RoyalNonKingRoleCount * SQUARE_NB);
        if (!schemaRows || firstWeights.size() != firstRows * V2ContainerFirstLanes
            || globalWeights.size()
                 != std::size_t(FixedRolePieceSquareDimensions) * V2ContainerGlobalLanes
            || hidden0Weights.size() != Width64x192::Hidden0WeightCount
            || hidden1Weights.size() != Width64x192::Hidden1WeightCount
            || outputWeights.size() != Width64x192::OutputWeightCount)
            return false;

        const auto safe = [](AffineBias bias) {
            return bias >= -MaxSafeBiasMagnitude && bias <= MaxSafeBiasMagnitude;
        };
        return std::all_of(firstBias.begin(), firstBias.end(), safe)
            && std::all_of(globalBias.begin(), globalBias.end(), safe)
            && std::all_of(hidden0Bias.begin(), hidden0Bias.end(), safe)
            && std::all_of(hidden1Bias.begin(), hidden1Bias.end(), safe)
            && std::all_of(outputBias.begin(), outputBias.end(), safe);
    }
};

struct ContainerLoadResult {
    ContainerLoadError  error = ContainerLoadError::NONE;
    std::string         message;
    ContainerParameters parameters;

    [[nodiscard]] explicit operator bool() const noexcept {
        return error == ContainerLoadError::NONE && parameters.valid();
    }
};

ContainerLoadResult load_integer_container(const std::filesystem::path& path);

enum class ContainerEvalError : std::uint8_t {
    NONE,
    INVALID_NETWORK,
    INVALID_POSITION,
    INVALID_SIDE_TO_MOVE,
};

inline constexpr RoyalKey NoContainerFirstDomainKey{RoyalBucketCount, false};

struct alignas(64) ContainerAccumulatorFrame {
    alignas(64) std::array<Accumulator, V2ContainerFirstLanes> first{};
    alignas(64) std::array<Accumulator, V2ContainerGlobalLanes> global{};
    RoyalKey key = NoContainerFirstDomainKey;
};

using ContainerDenseScratch = LeanDenseScratch<Width64x192>;

struct alignas(64) ContainerTrace {
    ContainerEvalError error        = ContainerEvalError::NONE;
    FullRefreshError  featureError = FullRefreshError::NONE;
    alignas(64) std::array<Accumulator, V2ContainerFirstLanes> firstAccumulator{};
    alignas(64) std::array<Accumulator, V2ContainerGlobalLanes> globalAccumulator{};
    alignas(64) std::array<Activation, V2ContainerInputLanes> transformed{};
    alignas(64) std::array<Accumulator, Width64x192::Hidden0Lanes> hidden0Affine{};
    alignas(64) std::array<Activation, Width64x192::Hidden0Lanes> hidden0{};
    alignas(64) std::array<Accumulator, Width64x192::Hidden1Lanes> hidden1Affine{};
    alignas(64) std::array<Activation, Width64x192::Hidden1Lanes> hidden1{};
    Accumulator outputAffine   = 0;
    i32         preRule50Value = 0;
    Value       value          = VALUE_NONE;

    [[nodiscard]] bool valid() const noexcept {
        return error == ContainerEvalError::NONE && featureError == FullRefreshError::NONE
            && value != VALUE_NONE;
    }
};

template<typename Kernels = DefaultLeanKernels>
class ContainerNetwork {
   public:
    using Frame   = ContainerAccumulatorFrame;
    using Scratch = ContainerDenseScratch;

    explicit ContainerNetwork(const ContainerParameters& parameters) :
        parameters_(parameters) {}
    ContainerNetwork(ContainerParameters&&) = delete;
    ContainerNetwork(const ContainerParameters&&) = delete;

    [[nodiscard]] bool valid() const noexcept { return parameters_.valid(); }

    [[nodiscard]] FirstDomain first_domain() const noexcept { return parameters_.firstDomain; }

    [[nodiscard]] RoyalKey first_domain_key(const FullRefreshFeatures& features) const noexcept {
        return parameters_.firstDomain == FirstDomain::ROYAL ? features.royalKey
                                                              : NoContainerFirstDomainKey;
    }

    [[nodiscard]] RoyalKey first_domain_key(const Position& pos) const noexcept {
        return parameters_.firstDomain == FirstDomain::ROYAL
               ? royal_key(pos.square<KING>(BLACK))
               : NoContainerFirstDomainKey;
    }

    [[nodiscard]] bool requires_refresh(RoyalKey sourceKey, RoyalKey targetKey) const noexcept {
        if (parameters_.firstDomain == FirstDomain::ABSOLUTE_NONKING)
            return sourceKey != NoContainerFirstDomainKey
                || targetKey != NoContainerFirstDomainKey;
        return !is_valid_royal_key(sourceKey) || !is_valid_royal_key(targetKey)
            || sourceKey != targetKey;
    }

    void full_refresh(Frame& frame, const FullRefreshFeatures& features) const noexcept {
        assert(valid());
        assert(features.valid());
        frame.first  = parameters_.firstBias;
        frame.global = parameters_.globalBias;
        frame.key    = first_domain_key(features);
        refresh_first(frame.first, features);
        refresh_global(frame.global, features);
    }

    [[nodiscard]] bool materialize_child_delta(Frame&                 child,
                                               const Frame&           source,
                                               const NormalizedDirty& dirty,
                                               RoyalKey               targetKey) const noexcept {
        if (requires_refresh(source.key, targetKey))
            return false;

        child.global = source.global;
        apply_global_deltas(child.global, dirty);
        child.first = source.first;
        apply_first_deltas(child.first, dirty, targetKey);
        child.key = targetKey;
        return true;
    }

    [[nodiscard]] bool materialize_child(Frame&                     child,
                                         const Frame&               source,
                                         const NormalizedDirty&     dirty,
                                         const FullRefreshFeatures& target) const noexcept {
        assert(&child != &source);
        assert(target.valid());

        const RoyalKey targetKey = first_domain_key(target);
        if (!requires_refresh(source.key, targetKey))
            return materialize_child_delta(child, source, dirty, targetKey);

        assert(parameters_.firstDomain == FirstDomain::ROYAL);
        child.global = source.global;
        apply_global_deltas(child.global, dirty);
        child.key   = targetKey;
        child.first = parameters_.firstBias;
        refresh_first(child.first, target);
        return true;
    }

    [[nodiscard]] LeanEvalResult propagate(const Frame& frame,
                                           Scratch&     scratch,
                                           Color        sideToMove,
                                           int          rule50Count) const noexcept {
        return propagate_dual_domain<Width64x192, Kernels>(
          frame.first, frame.global, parameters_, scratch, sideToMove, rule50Count);
    }

    [[nodiscard]] ContainerTrace evaluate_full_refresh(const std::array<Piece, SQUARE_NB>& board,
                                                       Color sideToMove,
                                                       int   rule50Count) const noexcept {
        ContainerTrace trace{};
        if (!valid())
        {
            trace.error = ContainerEvalError::INVALID_NETWORK;
            return trace;
        }
        if (sideToMove != WHITE && sideToMove != BLACK)
        {
            trace.error = ContainerEvalError::INVALID_SIDE_TO_MOVE;
            return trace;
        }

        const FullRefreshFeatures features = extract_full_refresh_features(board);
        if (!features.valid())
        {
            trace.error        = ContainerEvalError::INVALID_POSITION;
            trace.featureError = features.error;
            return trace;
        }

        Frame frame{};
        full_refresh(frame, features);
        Scratch              scratch{};
        const LeanEvalResult result = propagate(frame, scratch, sideToMove, rule50Count);

        trace.firstAccumulator  = frame.first;
        trace.globalAccumulator = frame.global;
        trace.transformed       = scratch.transformed;
        trace.hidden0Affine     = scratch.hidden0Affine;
        trace.hidden0           = scratch.hidden0;
        trace.hidden1Affine     = scratch.hidden1Affine;
        trace.hidden1           = scratch.hidden1;
        trace.outputAffine      = result.outputAffine;
        trace.preRule50Value    = result.preRule50Value;
        trace.value             = result.value;
        return trace;
    }

   private:
    void refresh_first(std::array<Accumulator, V2ContainerFirstLanes>& accumulator,
                       const FullRefreshFeatures&                      features) const noexcept {
        if (parameters_.firstDomain == FirstDomain::ROYAL)
        {
            for (std::size_t active = 0; active < features.royalSize; ++active)
            {
                const std::size_t row = features.royal[active];
                Kernels::add_row(accumulator,
                                 parameters_.firstWeights.data() + row * V2ContainerFirstLanes);
            }
            return;
        }

        for (std::size_t active = 0; active < features.globalSize; ++active)
        {
            const IndexType row = absolute_nonking_index_from_global(features.global[active]);
            if (row != InvalidFeatureIndex)
                Kernels::add_row(accumulator,
                                 parameters_.firstWeights.data()
                                   + std::size_t(row) * V2ContainerFirstLanes);
        }
    }

    void refresh_global(std::array<Accumulator, V2ContainerGlobalLanes>& accumulator,
                        const FullRefreshFeatures&                       features) const noexcept {
        for (std::size_t active = 0; active < features.globalSize; ++active)
        {
            const std::size_t row = features.global[active];
            Kernels::add_row(accumulator,
                             parameters_.globalWeights.data() + row * V2ContainerGlobalLanes);
        }
    }

    void add_global(std::array<Accumulator, V2ContainerGlobalLanes>& accumulator,
                    Piece                                            piece,
                    Square                                           square) const noexcept {
        const IndexType index = fixed_role_piece_square_index(piece, square);
        assert(index != InvalidFeatureIndex);
        Kernels::add_row(accumulator, parameters_.globalWeights.data()
                                        + std::size_t(index) * V2ContainerGlobalLanes);
    }

    void subtract_global(std::array<Accumulator, V2ContainerGlobalLanes>& accumulator,
                         Piece                                            piece,
                         Square                                           square) const noexcept {
        const IndexType index = fixed_role_piece_square_index(piece, square);
        assert(index != InvalidFeatureIndex);
        Kernels::subtract_row(accumulator, parameters_.globalWeights.data()
                                             + std::size_t(index) * V2ContainerGlobalLanes);
    }

    void add_first(std::array<Accumulator, V2ContainerFirstLanes>& accumulator,
                   Piece                                           piece,
                   Square                                          square,
                   RoyalKey                                       key) const noexcept {
        if (piece == B_KING)
            return;
        const IndexType index = parameters_.firstDomain == FirstDomain::ROYAL
                                ? royal_piece_square_index(piece, square, key)
                                : absolute_nonking_piece_square_index(piece, square);
        assert(index
               != (parameters_.firstDomain == FirstDomain::ROYAL ? InvalidRoyalFeatureIndex
                                                                  : InvalidFeatureIndex));
        Kernels::add_row(accumulator, parameters_.firstWeights.data()
                                        + std::size_t(index) * V2ContainerFirstLanes);
    }

    void subtract_first(std::array<Accumulator, V2ContainerFirstLanes>& accumulator,
                        Piece                                           piece,
                        Square                                          square,
                        RoyalKey                                       key) const noexcept {
        if (piece == B_KING)
            return;
        const IndexType index = parameters_.firstDomain == FirstDomain::ROYAL
                                ? royal_piece_square_index(piece, square, key)
                                : absolute_nonking_piece_square_index(piece, square);
        assert(index
               != (parameters_.firstDomain == FirstDomain::ROYAL ? InvalidRoyalFeatureIndex
                                                                  : InvalidFeatureIndex));
        Kernels::subtract_row(accumulator, parameters_.firstWeights.data()
                                             + std::size_t(index) * V2ContainerFirstLanes);
    }

    void apply_global_deltas(std::array<Accumulator, V2ContainerGlobalLanes>& accumulator,
                             const NormalizedDirty&                           dirty) const noexcept {
        subtract_global(accumulator, dirty.pc, dirty.from);
        if (dirty.remove_sq != SQ_NONE)
            subtract_global(accumulator, dirty.remove_pc, dirty.remove_sq);
        if (dirty.to != SQ_NONE)
            add_global(accumulator, dirty.pc, dirty.to);
        if (dirty.add_sq != SQ_NONE)
            add_global(accumulator, dirty.add_pc, dirty.add_sq);
    }

    void apply_first_deltas(std::array<Accumulator, V2ContainerFirstLanes>& accumulator,
                            const NormalizedDirty&                          dirty,
                            RoyalKey                                       key) const noexcept {
        subtract_first(accumulator, dirty.pc, dirty.from, key);
        if (dirty.remove_sq != SQ_NONE)
            subtract_first(accumulator, dirty.remove_pc, dirty.remove_sq, key);
        if (dirty.to != SQ_NONE)
            add_first(accumulator, dirty.pc, dirty.to, key);
        if (dirty.add_sq != SQ_NONE)
            add_first(accumulator, dirty.add_pc, dirty.add_sq, key);
    }

    const ContainerParameters& parameters_;
};

template<typename Kernels = DefaultLeanKernels>
using ContainerAccumulatorStack = LazyAccumulatorStack<ContainerNetwork<Kernels>>;

template<typename Kernels = DefaultLeanKernels>
using ValidatingContainerAccumulatorStack =
  LazyAccumulatorStack<ContainerNetwork<Kernels>, true>;

static_assert(V2ContainerFirstLanes + V2ContainerGlobalLanes == V2ContainerInputLanes);
static_assert(V2ContainerInputLanes == Width64x192::TransformedLanes);
static_assert(alignof(ContainerAccumulatorFrame) == 64);
static_assert(alignof(ContainerTrace) == 64);

}  // namespace Stockfish::Eval::NNUE::HordeV2

#endif  // HORDE_V2_CONTAINER_H_INCLUDED
