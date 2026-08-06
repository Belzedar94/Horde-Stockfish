/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_LEGACY_NETWORK_H_INCLUDED
#define HORDE_LEGACY_NETWORK_H_INCLUDED

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>

#include "../types.h"

namespace Stockfish {

class Position;

namespace Eval::NNUE {

// The Run 6B file uses the Fairy-Stockfish HalfKAv2 variant format without
// king buckets. Its 14 piece-square planes are deliberately kept at the NNUE
// boundary; the board continues to contain ordinary PAWN pieces only.
class HordeLegacyNetwork {
   public:
    static constexpr const char* SchemaName = "HORDETEST_HP_LEGACY_V1";
    static constexpr const char* Sha256 =
      "B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7";

    static constexpr usize FileSize              = 1088416;
    static constexpr usize FeatureDimensions     = 896;
    static constexpr usize AccumulatorDimensions = 512;
    static constexpr usize NetworkInputs         = 1024;
    static constexpr usize PsqtBuckets           = 8;
    static constexpr usize LayerStacks           = 8;

    using RawOutput = std::tuple<i32, i32>;

    bool load(const unsigned char* data, usize size, std::string& description);

    [[nodiscard]] RawOutput evaluate_raw(const Position& pos, int bucket = -1) const;
    [[nodiscard]] bool      loaded() const { return loaded_; }
    [[nodiscard]] usize     content_hash() const;
    [[nodiscard]] int       bucket_for(const Position& pos) const;

   private:
    struct LayerStack {
        std::array<i32, 16>                         fc0Biases{};
        std::array<std::int8_t, 16 * NetworkInputs> fc0Weights{};
        std::array<i32, 32>                         fc1Biases{};
        // The serialized input is padded from 16 to 32 bytes.
        std::array<std::int8_t, 32 * 32> fc1Weights{};
        i32                              fc2Bias{};
        std::array<std::int8_t, 32>      fc2Weights{};
    };

    std::array<i16, AccumulatorDimensions>                     biases_{};
    std::array<i16, FeatureDimensions * AccumulatorDimensions> weights_{};
    std::array<i32, FeatureDimensions * PsqtBuckets>           psqtWeights_{};
    std::array<LayerStack, LayerStacks>                        layers_{};
    bool                                                       loaded_ = false;
};

static_assert(std::is_trivially_copyable_v<HordeLegacyNetwork>);

}  // namespace Eval::NNUE
}  // namespace Stockfish

#endif  // HORDE_LEGACY_NETWORK_H_INCLUDED
