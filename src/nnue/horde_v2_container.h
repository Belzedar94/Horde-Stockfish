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

#include "horde_v2_backend.h"

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

struct alignas(64) ContainerTrace {
    FullRefreshError featureError = FullRefreshError::NONE;
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
        return featureError == FullRefreshError::NONE && value != VALUE_NONE;
    }
};

template<typename Kernels = DefaultLeanKernels>
class ContainerNetwork {
   public:
    explicit ContainerNetwork(const ContainerParameters& parameters) :
        parameters_(parameters) {}

    [[nodiscard]] ContainerTrace evaluate_full_refresh(const std::array<Piece, SQUARE_NB>& board,
                                                       Color sideToMove,
                                                       int   rule50Count) const noexcept {
        ContainerTrace trace{};
        const auto     features = extract_full_refresh_features(board);
        if (!parameters_.valid() || !features.valid())
        {
            trace.featureError =
              features.valid() ? FullRefreshError::INVALID_PIECE : features.error;
            return trace;
        }

        trace.firstAccumulator  = parameters_.firstBias;
        trace.globalAccumulator = parameters_.globalBias;
        if (parameters_.firstDomain == FirstDomain::ROYAL)
        {
            for (std::size_t active = 0; active < features.royalSize; ++active)
            {
                const std::size_t row = features.royal[active];
                Kernels::add_row(trace.firstAccumulator,
                                 parameters_.firstWeights.data() + row * V2ContainerFirstLanes);
            }
        }
        else
        {
            for (std::size_t active = 0; active < features.globalSize; ++active)
            {
                const std::size_t row = features.global[active];
                if (row < parameters_.firstRows)
                    Kernels::add_row(trace.firstAccumulator,
                                     parameters_.firstWeights.data() + row * V2ContainerFirstLanes);
            }
        }

        for (std::size_t active = 0; active < features.globalSize; ++active)
        {
            const std::size_t row = features.global[active];
            Kernels::add_row(trace.globalAccumulator,
                             parameters_.globalWeights.data() + row * V2ContainerGlobalLanes);
        }

        for (std::size_t lane = 0; lane < V2ContainerFirstLanes; ++lane)
            trace.transformed[lane] =
              clipped_activation(trace.firstAccumulator[lane], FtActivationShift);
        for (std::size_t lane = 0; lane < V2ContainerGlobalLanes; ++lane)
            trace.transformed[V2ContainerFirstLanes + lane] =
              clipped_activation(trace.globalAccumulator[lane], FtActivationShift);

        for (std::size_t output = 0; output < Width64x192::Hidden0Lanes; ++output)
        {
            const std::size_t offset = output * V2ContainerInputLanes;
            trace.hidden0Affine[output] =
              Kernels::affine(trace.transformed.data(), parameters_.hidden0Weights.data() + offset,
                              V2ContainerInputLanes, parameters_.hidden0Bias[output]);
            trace.hidden0[output] =
              clipped_activation(trace.hidden0Affine[output], DenseActivationShift);
        }

        for (std::size_t output = 0; output < Width64x192::Hidden1Lanes; ++output)
        {
            const std::size_t offset = output * Width64x192::Hidden0Lanes;
            trace.hidden1Affine[output] =
              Kernels::affine(trace.hidden0.data(), parameters_.hidden1Weights.data() + offset,
                              Width64x192::Hidden0Lanes, parameters_.hidden1Bias[output]);
            trace.hidden1[output] =
              clipped_activation(trace.hidden1Affine[output], DenseActivationShift);
        }

        if (sideToMove != WHITE && sideToMove != BLACK)
        {
            trace.featureError = FullRefreshError::INVALID_PIECE;
            return trace;
        }
        const std::size_t head   = std::size_t(sideToMove);
        const std::size_t offset = head * Width64x192::Hidden1Lanes;
        trace.outputAffine =
          Kernels::affine(trace.hidden1.data(), parameters_.outputWeights.data() + offset,
                          Width64x192::Hidden1Lanes, parameters_.outputBias[head]);
        trace.preRule50Value = trace.outputAffine / ScalarOutputScale;
        trace.value          = apply_rule50_postprocessor(trace.preRule50Value, rule50Count);
        return trace;
    }

   private:
    const ContainerParameters& parameters_;
};

static_assert(V2ContainerFirstLanes + V2ContainerGlobalLanes == V2ContainerInputLanes);
static_assert(V2ContainerInputLanes == Width64x192::TransformedLanes);
static_assert(alignof(ContainerTrace) == 64);

}  // namespace Stockfish::Eval::NNUE::HordeV2

#endif  // HORDE_V2_CONTAINER_H_INCLUDED
