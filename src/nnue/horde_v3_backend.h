/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_V3_BACKEND_H_INCLUDED
#define HORDE_V3_BACKEND_H_INCLUDED

#include <array>
#include <cassert>
#include <cstddef>

#if defined(USE_AVX2)
    #include <immintrin.h>
#endif

#include "horde_v3_scalar.h"

// The AVX2 path for the V3 evaluator. It shares the frame layout, the
// parameters and the dense propagation of the scalar path; only the three
// kernels change. Every intermediate value must be bit-identical to the scalar
// path, which parity gate 2 checks layer by layer.

namespace Stockfish::Eval::NNUE::HordeV3 {

#if defined(USE_AVX2)

struct V3Avx2Kernels {
    static constexpr const char* Name = "avx2";

    static void add_ft(Accumulator* accumulator, const FtWeight* row) noexcept {
        for (std::size_t lane = 0; lane < V3Lanes; lane += 8)
        {
            const __m256i current =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulator + lane));
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + lane));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(accumulator + lane),
                                _mm256_add_epi32(current, _mm256_cvtepi16_epi32(packed)));
        }
    }

    static void subtract_ft(Accumulator* accumulator, const FtWeight* row) noexcept {
        for (std::size_t lane = 0; lane < V3Lanes; lane += 8)
        {
            const __m256i current =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulator + lane));
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + lane));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(accumulator + lane),
                                _mm256_sub_epi32(current, _mm256_cvtepi16_epi32(packed)));
        }
    }

    static void add_psqt(Accumulator* psqt, const PsqtWeight* row) noexcept {
        for (std::size_t column = 0; column < V3PsqtColumns; column += 8)
        {
            const __m256i current =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(psqt + column));
            const __m256i weights =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + column));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(psqt + column),
                                _mm256_add_epi32(current, weights));
        }
    }

    static void subtract_psqt(Accumulator* psqt, const PsqtWeight* row) noexcept {
        for (std::size_t column = 0; column < V3PsqtColumns; column += 8)
        {
            const __m256i current =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(psqt + column));
            const __m256i weights =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + column));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(psqt + column),
                                _mm256_sub_epi32(current, weights));
        }
    }

    // uint8 activations against int8 weights. _mm256_maddubs_epi16 saturates at
    // int16, so the pairwise sum has to stay inside it: the largest term pair is
    // 2 * 127 * 127 = 32258, below 32767, which is the same bound the V2 kernel
    // relies on.
    static Accumulator affine(const Activation*  input,
                              const DenseWeight* weights,
                              std::size_t        count,
                              Accumulator        bias) noexcept {
        assert(count % 16 == 0);

        std::size_t   index = 0;
        const __m256i ones  = _mm256_set1_epi16(1);
        __m256i       wide  = _mm256_setzero_si256();
        for (; index + 32 <= count; index += 32)
        {
            const __m256i activations =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + index));
            const __m256i denseWeights =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weights + index));
            const __m256i pairProducts = _mm256_maddubs_epi16(activations, denseWeights);
            wide = _mm256_add_epi32(wide, _mm256_madd_epi16(pairProducts, ones));
        }

        __m128i narrow =
          _mm_add_epi32(_mm256_castsi256_si128(wide), _mm256_extracti128_si256(wide, 1));
        for (; index + 16 <= count; index += 16)
        {
            const __m128i activations =
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + index));
            const __m128i denseWeights =
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights + index));
            const __m128i pairProducts = _mm_maddubs_epi16(activations, denseWeights);
            narrow = _mm_add_epi32(narrow, _mm_madd_epi16(pairProducts, _mm_set1_epi16(1)));
        }

        alignas(16) std::array<Accumulator, 4> lanes{};
        _mm_storeu_si128(reinterpret_cast<__m128i*>(lanes.data()), narrow);
        Accumulator sum = bias;
        for (const Accumulator lane : lanes)
            sum += lane;
        for (; index < count; ++index)
            sum += Accumulator(weights[index]) * Accumulator(input[index]);
        return sum;
    }
};

using V3DefaultKernels                            = V3Avx2Kernels;
inline constexpr const char* V3DefaultBackendName = "avx2";
inline constexpr bool        V3Avx2Available      = true;

#else

using V3DefaultKernels                            = V3ScalarKernels;
inline constexpr const char* V3DefaultBackendName = "scalar";
inline constexpr bool        V3Avx2Available      = false;

#endif

using V3DefaultNetwork = V3Network<V3DefaultKernels>;

static_assert(ActivationMax * 128 * 2 <= 32767);
static_assert(V3Lanes % 32 == 0);
static_assert(V3Hidden0Lanes % 16 == 0);
static_assert(V3Hidden1Lanes % 32 == 0);
static_assert(V3PsqtColumns % 8 == 0);

}  // namespace Stockfish::Eval::NNUE::HordeV3

#endif  // HORDE_V3_BACKEND_H_INCLUDED
