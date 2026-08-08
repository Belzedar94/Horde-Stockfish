/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_V2_BACKEND_H_INCLUDED
#define HORDE_V2_BACKEND_H_INCLUDED

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

#if defined(USE_AVX2)
    #include <immintrin.h>
#endif

#include "horde_v2_scalar.h"

namespace Stockfish::Eval::NNUE::HordeV2 {

enum class FixturePayload : std::uint8_t {
    PARITY_FULL_V1,
    PERF_COMMON_V1,
};

template<typename T, std::size_t Alignment = 64>
class AlignedBuffer {
   public:
    explicit AlignedBuffer(std::size_t size) :
        size_(size),
        data_(static_cast<T*>(::operator new[](size * sizeof(T), std::align_val_t(Alignment)))) {}

    ~AlignedBuffer() { ::operator delete[](data_, std::align_val_t(Alignment)); }

    AlignedBuffer(const AlignedBuffer&)            = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept :
        size_(std::exchange(other.size_, 0)),
        data_(std::exchange(other.data_, nullptr)) {}

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {
        if (this != &other)
        {
            ::operator delete[](data_, std::align_val_t(Alignment));
            size_ = std::exchange(other.size_, 0);
            data_ = std::exchange(other.data_, nullptr);
        }
        return *this;
    }

    [[nodiscard]] T*          data() noexcept { return data_; }
    [[nodiscard]] const T*    data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] T*          begin() noexcept { return data_; }
    [[nodiscard]] const T*    begin() const noexcept { return data_; }
    [[nodiscard]] T*          end() noexcept { return data_ + size_; }
    [[nodiscard]] const T*    end() const noexcept { return data_ + size_; }

    T&       operator[](std::size_t index) noexcept { return data_[index]; }
    const T& operator[](std::size_t index) const noexcept { return data_[index]; }

   private:
    std::size_t size_ = 0;
    T*          data_ = nullptr;
};

template<typename Width>
struct LeanParameters {
    alignas(64) std::array<AffineBias, Width::RoyalLanes> royalBias{};
    AlignedBuffer<FtWeight> royalWeights;
    alignas(64) std::array<AffineBias, Width::GlobalLanes> globalBias{};
    AlignedBuffer<FtWeight> globalWeights;

    alignas(64) std::array<AffineBias, Width::Hidden0Lanes> hidden0Bias{};
    AlignedBuffer<DenseWeight> hidden0Weights;
    alignas(64) std::array<AffineBias, Width::Hidden1Lanes> hidden1Bias{};
    AlignedBuffer<DenseWeight> hidden1Weights;
    alignas(64) std::array<AffineBias, Width::OutputHeads> outputBias{};
    AlignedBuffer<DenseWeight> outputWeights;

    LeanParameters() :
        royalWeights(Width::RoyalWeightCount),
        globalWeights(Width::GlobalWeightCount),
        hidden0Weights(Width::Hidden0WeightCount),
        hidden1Weights(Width::Hidden1WeightCount),
        outputWeights(Width::OutputWeightCount) {}
};

namespace BackendDetail {

enum class ParameterBlock : u64 {
    ROYAL_BIAS = 1,
    ROYAL_FT,
    GLOBAL_BIAS,
    GLOBAL_FT,
    HIDDEN0_BIAS,
    HIDDEN0_ROYAL,
    HIDDEN0_GLOBAL,
    HIDDEN1_BIAS,
    HIDDEN1,
    OUTPUT_BIAS,
    OUTPUT,
};

constexpr u64 splitmix64(u64 value) noexcept {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

constexpr u64 coordinate_hash(u64            seed,
                              ParameterBlock block,
                              std::size_t    first,
                              std::size_t    second = 0) noexcept {
    u64 value = seed ^ (u64(block) * 0xD6E8FEB86659FD93ULL);
    value ^= u64(first) * 0xA0761D6478BD642FULL;
    value ^= u64(second) * 0xE7037ED1A0B428DBULL;
    return splitmix64(value);
}

template<typename T>
constexpr T bounded_signed(u64 hash, int radius) noexcept {
    return static_cast<T>(int(hash % u64(2 * radius + 1)) - radius);
}

template<typename T>
constexpr T bounded_nonzero(u64 hash, int radius) noexcept {
    const int magnitude = int(hash % u64(radius)) + 1;
    return static_cast<T>((hash >> 63) ? -magnitude : magnitude);
}

}  // namespace BackendDetail

struct ScalarLeanKernels {
    template<std::size_t Lanes>
    static void add_row(std::array<Accumulator, Lanes>& destination, const FtWeight* row) noexcept {
        for (std::size_t lane = 0; lane < Lanes; ++lane)
            destination[lane] += row[lane];
    }

    template<std::size_t Lanes>
    static void subtract_row(std::array<Accumulator, Lanes>& destination,
                             const FtWeight*                 row) noexcept {
        for (std::size_t lane = 0; lane < Lanes; ++lane)
            destination[lane] -= row[lane];
    }

    static Accumulator affine(const Activation*  input,
                              const DenseWeight* weights,
                              std::size_t        count,
                              Accumulator        bias) noexcept {
        Accumulator sum = bias;
        for (std::size_t index = 0; index < count; ++index)
            sum += Accumulator(weights[index]) * input[index];
        return sum;
    }
};

#if defined(USE_AVX2)

struct Avx2LeanKernels {
    template<std::size_t Lanes>
    static void add_row(std::array<Accumulator, Lanes>& destination, const FtWeight* row) noexcept {
        static_assert(Lanes % 8 == 0);
        for (std::size_t lane = 0; lane < Lanes; lane += 8)
        {
            const __m256i current =
              _mm256_load_si256(reinterpret_cast<const __m256i*>(destination.data() + lane));
            const __m128i packed  = _mm_load_si128(reinterpret_cast<const __m128i*>(row + lane));
            const __m256i widened = _mm256_cvtepi16_epi32(packed);
            _mm256_store_si256(reinterpret_cast<__m256i*>(destination.data() + lane),
                               _mm256_add_epi32(current, widened));
        }
    }

    template<std::size_t Lanes>
    static void subtract_row(std::array<Accumulator, Lanes>& destination,
                             const FtWeight*                 row) noexcept {
        static_assert(Lanes % 8 == 0);
        for (std::size_t lane = 0; lane < Lanes; lane += 8)
        {
            const __m256i current =
              _mm256_load_si256(reinterpret_cast<const __m256i*>(destination.data() + lane));
            const __m128i packed  = _mm_load_si128(reinterpret_cast<const __m128i*>(row + lane));
            const __m256i widened = _mm256_cvtepi16_epi32(packed);
            _mm256_store_si256(reinterpret_cast<__m256i*>(destination.data() + lane),
                               _mm256_sub_epi32(current, widened));
        }
    }

    static Accumulator affine(const Activation*  input,
                              const DenseWeight* weights,
                              std::size_t        count,
                              Accumulator        bias) noexcept {
        assert(count % 32 == 0);
        __m256i sum  = _mm256_setzero_si256();
        __m256i ones = _mm256_set1_epi16(1);
        for (std::size_t index = 0; index < count; index += 32)
        {
            const __m256i activations =
              _mm256_load_si256(reinterpret_cast<const __m256i*>(input + index));
            const __m256i denseWeights =
              _mm256_load_si256(reinterpret_cast<const __m256i*>(weights + index));
            const __m256i pairProducts = _mm256_maddubs_epi16(activations, denseWeights);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(pairProducts, ones));
        }

        alignas(32) std::array<Accumulator, 8> lanes{};
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes.data()), sum);
        for (const Accumulator lane : lanes)
            bias += lane;
        return bias;
    }
};

using DefaultLeanKernels                            = Avx2LeanKernels;
inline constexpr const char* DefaultLeanBackendName = "avx2";

#else

using DefaultLeanKernels                            = ScalarLeanKernels;
inline constexpr const char* DefaultLeanBackendName = "scalar";

#endif

template<typename Width>
LeanParameters<Width> make_lean_parameters(u64 seed, FixturePayload payload) {
    LeanParameters<Width> parameters;

    for (std::size_t lane = 0; lane < Width::RoyalLanes; ++lane)
        parameters.royalBias[lane] =
          BackendDetail::bounded_signed<AffineBias>(
            BackendDetail::coordinate_hash(seed, BackendDetail::ParameterBlock::ROYAL_BIAS, lane),
            6144)
          + 4096;
    for (std::size_t row = 0; row < RoyalPieceSquareDimensions; ++row)
        for (std::size_t lane = 0; lane < Width::RoyalLanes; ++lane)
            parameters.royalWeights[row * Width::RoyalLanes + lane] =
              BackendDetail::bounded_nonzero<FtWeight>(
                BackendDetail::coordinate_hash(seed, BackendDetail::ParameterBlock::ROYAL_FT, row,
                                               lane),
                31);

    for (std::size_t lane = 0; lane < Width::GlobalLanes; ++lane)
        parameters.globalBias[lane] =
          BackendDetail::bounded_signed<AffineBias>(
            BackendDetail::coordinate_hash(seed, BackendDetail::ParameterBlock::GLOBAL_BIAS, lane),
            6144)
          + 4096;
    for (std::size_t row = 0; row < FixedRolePieceSquareDimensions; ++row)
        for (std::size_t lane = 0; lane < Width::GlobalLanes; ++lane)
            parameters.globalWeights[row * Width::GlobalLanes + lane] =
              BackendDetail::bounded_nonzero<FtWeight>(
                BackendDetail::coordinate_hash(seed, BackendDetail::ParameterBlock::GLOBAL_FT, row,
                                               lane),
                31);

    for (std::size_t output = 0; output < Width::Hidden0Lanes; ++output)
    {
        parameters.hidden0Bias[output] = BackendDetail::bounded_signed<AffineBias>(
          BackendDetail::coordinate_hash(seed, BackendDetail::ParameterBlock::HIDDEN0_BIAS, output),
          4096);

        const std::size_t offset = output * Width::TransformedLanes;
        for (std::size_t lane = 0; lane < Width::RoyalLanes; ++lane)
            parameters.hidden0Weights[offset + lane] =
              payload == FixturePayload::PERF_COMMON_V1 && lane >= 64
                ? DenseWeight(0)
                : BackendDetail::bounded_nonzero<DenseWeight>(
                    BackendDetail::coordinate_hash(
                      seed, BackendDetail::ParameterBlock::HIDDEN0_ROYAL, output, lane),
                    7);
        for (std::size_t lane = 0; lane < Width::GlobalLanes; ++lane)
            parameters.hidden0Weights[offset + Width::RoyalLanes + lane] =
              payload == FixturePayload::PERF_COMMON_V1 && lane >= 128
                ? DenseWeight(0)
                : BackendDetail::bounded_nonzero<DenseWeight>(
                    BackendDetail::coordinate_hash(
                      seed, BackendDetail::ParameterBlock::HIDDEN0_GLOBAL, output, lane),
                    7);
    }

    for (std::size_t output = 0; output < Width::Hidden1Lanes; ++output)
    {
        parameters.hidden1Bias[output] = BackendDetail::bounded_signed<AffineBias>(
          BackendDetail::coordinate_hash(seed, BackendDetail::ParameterBlock::HIDDEN1_BIAS, output),
          4096);
        for (std::size_t input = 0; input < Width::Hidden0Lanes; ++input)
            parameters.hidden1Weights[output * Width::Hidden0Lanes + input] =
              BackendDetail::bounded_nonzero<DenseWeight>(
                BackendDetail::coordinate_hash(seed, BackendDetail::ParameterBlock::HIDDEN1, output,
                                               input),
                7);
    }

    for (std::size_t head = 0; head < Width::OutputHeads; ++head)
    {
        parameters.outputBias[head] = BackendDetail::bounded_signed<AffineBias>(
          BackendDetail::coordinate_hash(seed, BackendDetail::ParameterBlock::OUTPUT_BIAS, head),
          4096);
        for (std::size_t input = 0; input < Width::Hidden1Lanes; ++input)
            parameters.outputWeights[head * Width::Hidden1Lanes + input] =
              BackendDetail::bounded_nonzero<DenseWeight>(
                BackendDetail::coordinate_hash(seed, BackendDetail::ParameterBlock::OUTPUT, head,
                                               input),
                7);
    }

    return parameters;
}

template<typename Width>
struct alignas(64) LeanAccumulatorFrame {
    alignas(64) std::array<Accumulator, Width::RoyalLanes> royal{};
    alignas(64) std::array<Accumulator, Width::GlobalLanes> global{};
    RoyalKey key{RoyalBucketCount, false};
};

template<typename Width>
struct alignas(64) LeanDenseScratch {
    alignas(64) std::array<Activation, Width::TransformedLanes> transformed{};
    alignas(64) std::array<Accumulator, Width::Hidden0Lanes> hidden0Affine{};
    alignas(64) std::array<Activation, Width::Hidden0Lanes> hidden0{};
    alignas(64) std::array<Accumulator, Width::Hidden1Lanes> hidden1Affine{};
    alignas(64) std::array<Activation, Width::Hidden1Lanes> hidden1{};
};

struct LeanEvalResult {
    Accumulator outputAffine   = 0;
    i32         preRule50Value = 0;
    Value       value          = VALUE_NONE;
};

template<typename Width, typename Kernels = DefaultLeanKernels>
class LeanNetwork {
   public:
    using Frame      = LeanAccumulatorFrame<Width>;
    using Scratch    = LeanDenseScratch<Width>;
    using Parameters = LeanParameters<Width>;

    explicit LeanNetwork(Parameters parameters) :
        parameters_(std::move(parameters)) {}

    [[nodiscard]] const Parameters& parameters() const noexcept { return parameters_; }

    void full_refresh(Frame& frame, const FullRefreshFeatures& features) const noexcept {
        assert(features.valid());
        frame.royal  = parameters_.royalBias;
        frame.global = parameters_.globalBias;
        frame.key    = features.royalKey;
        refresh_royal(frame.royal, features);
        refresh_global(frame.global, features);
    }

    [[nodiscard]] bool materialize_child_same_key(Frame&            child,
                                                  const Frame&      source,
                                                  const DirtyPiece& dirty,
                                                  RoyalKey          targetKey) const noexcept {
        if (!is_valid_royal_key(targetKey) || targetKey != source.key)
            return false;

        child.global = source.global;
        apply_global_deltas(child.global, dirty);
        child.royal = source.royal;
        apply_royal_deltas(child.royal, dirty, targetKey);
        child.key = targetKey;
        return true;
    }

    void materialize_child(Frame&                     child,
                           const Frame&               source,
                           const DirtyPiece&          dirty,
                           const FullRefreshFeatures& target) const noexcept {
        assert(&child != &source);
        assert(target.valid());

        if (target.royalKey == source.key)
        {
            const bool materialized =
              materialize_child_same_key(child, source, dirty, target.royalKey);
            assert(materialized);
            return;
        }

        child.global = source.global;
        apply_global_deltas(child.global, dirty);
        child.key   = target.royalKey;
        child.royal = parameters_.royalBias;
        refresh_royal(child.royal, target);
    }

    // Production-position overload. Ordinary moves derive the Royal key in
    // O(1) and never enumerate the board. Only a Black-king bucket or mirror
    // transition extracts sparse Royal rows for a refresh; Global remains a
    // delta update in both cases.
    [[nodiscard]] bool materialize_child(Frame&            child,
                                         const Frame&      source,
                                         const DirtyPiece& dirty,
                                         const Position&   target) const noexcept {
        if (target.count<KING>(WHITE) != 0 || target.count<KING>(BLACK) != 1)
            return false;

        const RoyalKey targetKey = royal_key(target.square<KING>(BLACK));

        if (targetKey == source.key)
            return materialize_child_same_key(child, source, dirty, targetKey);

        const FullRefreshFeatures features = extract_full_refresh_features(target);
        if (!features.valid() || features.royalKey != targetKey)
            return false;

        child.global = source.global;
        apply_global_deltas(child.global, dirty);
        child.key   = targetKey;
        child.royal = parameters_.royalBias;
        refresh_royal(child.royal, features);
        return true;
    }

    [[nodiscard]] LeanEvalResult propagate(const Frame& frame,
                                           Scratch&     scratch,
                                           Color        sideToMove,
                                           int          rule50Count) const noexcept {
        assert(sideToMove == WHITE || sideToMove == BLACK);

        for (std::size_t lane = 0; lane < Width::RoyalLanes; ++lane)
            scratch.transformed[lane] = clipped_activation(frame.royal[lane], FtActivationShift);
        for (std::size_t lane = 0; lane < Width::GlobalLanes; ++lane)
            scratch.transformed[Width::RoyalLanes + lane] =
              clipped_activation(frame.global[lane], FtActivationShift);

        for (std::size_t output = 0; output < Width::Hidden0Lanes; ++output)
        {
            const std::size_t offset = output * Width::TransformedLanes;
            const Accumulator sum    = Kernels::affine(
              scratch.transformed.data(), parameters_.hidden0Weights.data() + offset,
              Width::TransformedLanes, parameters_.hidden0Bias[output]);
            scratch.hidden0Affine[output] = sum;
            scratch.hidden0[output]       = clipped_activation(sum, DenseActivationShift);
        }

        for (std::size_t output = 0; output < Width::Hidden1Lanes; ++output)
        {
            const std::size_t offset = output * Width::Hidden0Lanes;
            const Accumulator sum =
              Kernels::affine(scratch.hidden0.data(), parameters_.hidden1Weights.data() + offset,
                              Width::Hidden0Lanes, parameters_.hidden1Bias[output]);
            scratch.hidden1Affine[output] = sum;
            scratch.hidden1[output]       = clipped_activation(sum, DenseActivationShift);
        }

        const std::size_t head   = std::size_t(sideToMove);
        const std::size_t offset = head * Width::Hidden1Lanes;
        LeanEvalResult    result{};
        result.outputAffine =
          Kernels::affine(scratch.hidden1.data(), parameters_.outputWeights.data() + offset,
                          Width::Hidden1Lanes, parameters_.outputBias[head]);

        result.preRule50Value = result.outputAffine / ScalarOutputScale;
        result.value          = apply_rule50_postprocessor(result.preRule50Value, rule50Count);
        return result;
    }

   private:
    void refresh_royal(std::array<Accumulator, Width::RoyalLanes>& accumulator,
                       const FullRefreshFeatures&                  features) const noexcept {
        for (std::size_t active = 0; active < features.royalSize; ++active)
        {
            const FtWeight* row = parameters_.royalWeights.data()
                                + std::size_t(features.royal[active]) * Width::RoyalLanes;
            Kernels::add_row(accumulator, row);
        }
    }

    void refresh_global(std::array<Accumulator, Width::GlobalLanes>& accumulator,
                        const FullRefreshFeatures&                   features) const noexcept {
        for (std::size_t active = 0; active < features.globalSize; ++active)
        {
            const FtWeight* row = parameters_.globalWeights.data()
                                + std::size_t(features.global[active]) * Width::GlobalLanes;
            Kernels::add_row(accumulator, row);
        }
    }

    void add_global(std::array<Accumulator, Width::GlobalLanes>& accumulator,
                    Piece                                        piece,
                    Square                                       square) const noexcept {
        const IndexType index = fixed_role_piece_square_index(piece, square);
        assert(index != InvalidFeatureIndex);
        Kernels::add_row(accumulator, parameters_.globalWeights.data()
                                        + std::size_t(index) * Width::GlobalLanes);
    }

    void subtract_global(std::array<Accumulator, Width::GlobalLanes>& accumulator,
                         Piece                                        piece,
                         Square                                       square) const noexcept {
        const IndexType index = fixed_role_piece_square_index(piece, square);
        assert(index != InvalidFeatureIndex);
        Kernels::subtract_row(accumulator, parameters_.globalWeights.data()
                                             + std::size_t(index) * Width::GlobalLanes);
    }

    void add_royal(std::array<Accumulator, Width::RoyalLanes>& accumulator,
                   Piece                                       piece,
                   Square                                      square,
                   RoyalKey                                    key) const noexcept {
        if (piece == B_KING)
            return;
        const IndexType index = royal_piece_square_index(piece, square, key);
        assert(index != InvalidRoyalFeatureIndex);
        Kernels::add_row(accumulator,
                         parameters_.royalWeights.data() + std::size_t(index) * Width::RoyalLanes);
    }

    void subtract_royal(std::array<Accumulator, Width::RoyalLanes>& accumulator,
                        Piece                                       piece,
                        Square                                      square,
                        RoyalKey                                    key) const noexcept {
        if (piece == B_KING)
            return;
        const IndexType index = royal_piece_square_index(piece, square, key);
        assert(index != InvalidRoyalFeatureIndex);
        Kernels::subtract_row(accumulator, parameters_.royalWeights.data()
                                             + std::size_t(index) * Width::RoyalLanes);
    }

    void apply_global_deltas(std::array<Accumulator, Width::GlobalLanes>& accumulator,
                             const DirtyPiece&                            dirty) const noexcept {
        subtract_global(accumulator, dirty.pc, dirty.from);
        if (dirty.to != SQ_NONE)
            add_global(accumulator, dirty.pc, dirty.to);
        if (dirty.remove_sq != SQ_NONE)
            subtract_global(accumulator, dirty.remove_pc, dirty.remove_sq);
        if (dirty.add_sq != SQ_NONE)
            add_global(accumulator, dirty.add_pc, dirty.add_sq);
    }

    void apply_royal_deltas(std::array<Accumulator, Width::RoyalLanes>& accumulator,
                            const DirtyPiece&                           dirty,
                            RoyalKey                                    key) const noexcept {
        subtract_royal(accumulator, dirty.pc, dirty.from, key);
        if (dirty.to != SQ_NONE)
            add_royal(accumulator, dirty.pc, dirty.to, key);
        if (dirty.remove_sq != SQ_NONE)
            subtract_royal(accumulator, dirty.remove_pc, dirty.remove_sq, key);
        if (dirty.add_sq != SQ_NONE)
            add_royal(accumulator, dirty.add_pc, dirty.add_sq, key);
    }

    Parameters parameters_;
};

static_assert(ActivationMax * 128 * 2 <= 32767);
static_assert(alignof(LeanAccumulatorFrame<Width256x256>) == 64);
static_assert(alignof(LeanAccumulatorFrame<Width64x192>) == 64);

}  // namespace Stockfish::Eval::NNUE::HordeV2

#endif  // HORDE_V2_BACKEND_H_INCLUDED
