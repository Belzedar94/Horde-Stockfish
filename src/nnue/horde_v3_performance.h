/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_V3_PERFORMANCE_H_INCLUDED
#define HORDE_V3_PERFORMANCE_H_INCLUDED

#include <array>
#include <cstddef>
#include <cstdint>

#include "horde_v3_container.h"
#include "horde_v3_stack.h"

// V3 registers exactly one activated width, so unlike the V2 performance
// header this one selects nothing: it publishes the process-wide performance
// network and the Frame and Scratch types the engine branch needs. A build
// that tries to move the width is rejected here rather than silently
// evaluating an unregistered architecture.
//
// The parameters are the frozen V3 parity fixture, byte for byte the payload
// tests/horde_v3_parity.py::deterministic_sections() writes and
// tests/horde_v3_kernel_timing.cpp::build_fixture_parameters() mirrors: the
// same per-section salt derived from the registered schema id, the same
// modulus per dtype and the same phase lookup. No new fixture is invented, so
// a performance binary and the parity oracle evaluate identical weights.

namespace Stockfish::Eval::NNUE::HordeV3 {

#if defined(HORDE_V3_PERF_LANES)
static_assert(std::size_t(HORDE_V3_PERF_LANES) == V3Lanes,
              "unregistered Horde V3 performance width");
#endif
#if defined(HORDE_V3_PERF_ROYAL_WIDTH) || defined(HORDE_V3_PERF_GLOBAL_WIDTH)
    #error "Horde V3 registers one width; the V2 split-width knobs do not apply"
#endif

inline constexpr std::size_t PerformanceLanes        = V3Lanes;
inline constexpr std::size_t PerformanceHidden0Lanes = V3Hidden0Lanes;
inline constexpr std::size_t PerformanceHidden1Lanes = V3Hidden1Lanes;
inline constexpr std::size_t PerformanceRows         = V3StreamRows;

// The registered width. Reaching this header with anything else is a build
// error, not a slower or a different evaluator.
static_assert(PerformanceLanes == 1024, "unregistered Horde V3 performance width");
static_assert(PerformanceHidden0Lanes == 16 && PerformanceHidden1Lanes == 32,
              "unregistered Horde V3 performance dense trunk");
static_assert(PerformanceRows == 896, "unregistered Horde V3 performance row count");

using PerformanceNetwork = V3DefaultNetwork;
using PerformanceStack   = V3DefaultStack;
using PerformanceFrame   = PerformanceNetwork::Frame;
using PerformanceScratch = PerformanceNetwork::Scratch;

// The frozen fixture salt: schema id * 17 + section_id * 101, with the section
// ids the registered directory assigns, 1 through 9.
constexpr std::int64_t performance_fixture_salt(std::int64_t sectionId) noexcept {
    return std::int64_t(V3NetworkSchemaId) * 17 + sectionId * 101;
}

inline constexpr std::array<std::uint8_t, V3PhaseLookupSize> PerformancePhaseLookup = {
  0, 0, 0, 0, 0, 0,  //
  1, 1, 1, 1,        //
  2, 2, 2, 2,        //
  3, 3, 3, 3, 3,     //
  4, 4, 4, 4,        //
  5, 5, 5, 5,        //
  6, 6, 6,           //
  7, 7, 7, 7, 7, 7, 7};

inline V3Parameters make_performance_parameters() {
    V3Parameters parameters;
    parameters.schemaName      = "V3_G1024_PAWN_WPC8";
    parameters.fileSha256      = "";
    parameters.parameterSha256 = "";

    const auto denseWeight = [](std::int64_t index, std::int64_t salt) {
        return DenseWeight(((index * 37 + salt) % 15) - 7);
    };
    const auto bias = [](std::int64_t index, std::int64_t salt) {
        return AffineBias(((index * 193 + salt) % 8193) - 4096);
    };

    for (std::size_t index = 0; index < FtWeightCount; ++index)
        parameters.ftWeights[index] =
          FtWeight(((std::int64_t(index) * 97 + performance_fixture_salt(1)) % 63) - 31);
    for (std::size_t index = 0; index < V3Lanes; ++index)
        parameters.ftBias[index] = AffineBias(
          ((std::int64_t(index) * 193 + performance_fixture_salt(2)) % 12289) - 6144 + 4096);
    for (std::size_t index = 0; index < PsqtWeightCount; ++index)
        parameters.psqtWeights[index] = PsqtWeight(
          ((std::int64_t(index) * 8191 + performance_fixture_salt(3)) % 40001) - 20000);

    for (std::size_t index = 0; index < Hidden0WeightCount; ++index)
        parameters.hidden0Weights[index] =
          denseWeight(std::int64_t(index), performance_fixture_salt(4));
    for (std::size_t index = 0; index < parameters.hidden0Bias.size(); ++index)
        parameters.hidden0Bias[index] = bias(std::int64_t(index), performance_fixture_salt(5));
    for (std::size_t index = 0; index < Hidden1WeightCount; ++index)
        parameters.hidden1Weights[index] =
          denseWeight(std::int64_t(index), performance_fixture_salt(6));
    for (std::size_t index = 0; index < parameters.hidden1Bias.size(); ++index)
        parameters.hidden1Bias[index] = bias(std::int64_t(index), performance_fixture_salt(7));
    for (std::size_t index = 0; index < OutputWeightCount; ++index)
        parameters.outputWeights[index] =
          denseWeight(std::int64_t(index), performance_fixture_salt(8));
    for (std::size_t index = 0; index < parameters.outputBias.size(); ++index)
        parameters.outputBias[index] = bias(std::int64_t(index), performance_fixture_salt(9));

    parameters.phaseLookup = PerformancePhaseLookup;
    return parameters;
}

// The parameters outlive every worker, so the network holds a reference to
// them exactly as the candidate path does. Copy elision constructs the static
// in place, which matters because V3Parameters owns aligned buffers and is
// deliberately not copyable.
inline const V3Parameters& performance_parameters() {
    static const V3Parameters parameters = make_performance_parameters();
    return parameters;
}

inline const PerformanceNetwork& performance_network() {
    static const PerformanceNetwork network(performance_parameters());
    return network;
}

}  // namespace Stockfish::Eval::NNUE::HordeV3

#endif  // HORDE_V3_PERFORMANCE_H_INCLUDED
