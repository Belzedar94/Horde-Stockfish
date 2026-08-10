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
#include "horde_legacy_features.h"
#include "layers/affine_transform.h"

namespace Stockfish {

class Position;

namespace Eval::NNUE {

class AccumulatorStack;
struct AccumulatorState;

// The Run 6B file uses the Fairy-Stockfish HalfKAv2 variant format without
// king buckets. Its 14 piece-square planes are deliberately kept at the NNUE
// boundary; the board continues to contain ordinary PAWN pieces only.
class HordeLegacyNetwork {
   public:
    static constexpr const char* SchemaName = "HORDETEST_HP_LEGACY_V1";
    // The default production artifact remains Run 6B. Additional legacy
    // artifacts are accepted only when their full digest and size are present
    // in the compiled registry in horde_legacy_network.cpp.
    static constexpr const char* Sha256 =
      "B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7";

    static constexpr usize Run6BFileSize          = 1088416;
    static constexpr usize FeatureDimensions     = HordeLegacy::PieceSquareDimensions;
    static constexpr usize AccumulatorDimensions = 512;
    static constexpr usize NetworkInputs         = 1024;
    static constexpr usize PsqtBuckets           = 8;
    static constexpr usize LayerStacks           = 8;

    using RawOutput = std::tuple<i32, i32>;

    bool load(const unsigned char* data, usize size, std::string& description);

    [[nodiscard]] RawOutput evaluate_raw(const Position&   pos,
                                         AccumulatorStack& accumulatorStack,
                                         int               bucket = -1) const;
    [[nodiscard]] RawOutput evaluate_raw_full_refresh(const Position& pos,
                                                      int             bucket = -1) const;
    [[nodiscard]] bool      loaded() const { return loaded_; }
    [[nodiscard]] usize     content_hash() const;
    [[nodiscard]] int       bucket_for(const Position& pos) const;
    [[nodiscard]] const char* artifact_name() const;
    [[nodiscard]] const char* artifact_sha256() const;

   private:
    struct LayerStack {
        using Fc0 = Layers::AffineTransform<NetworkInputs, 16, false>;
        using Fc1 = Layers::AffineTransform<16, 32, false>;
        using Fc2 = Layers::AffineTransform<32, 1, false>;

        Fc0 fc0{};
        Fc1 fc1{};
        Fc2 fc2{};
    };

    std::array<i16, AccumulatorDimensions>                     biases_{};
    std::array<i16, FeatureDimensions * AccumulatorDimensions> weights_{};
    std::array<i32, FeatureDimensions * PsqtBuckets>           psqtWeights_{};
    std::array<LayerStack, LayerStacks>                        layers_{};
    bool                                                       loaded_ = false;
    u8                                                         manifestId_ = 0;

    void refresh_accumulator(const Position& pos, AccumulatorState& target) const;
    void update_accumulator(const DirtyPiece&       dirty,
                            const AccumulatorState& source,
                            AccumulatorState&       target) const;
    [[nodiscard]] RawOutput propagate(const Position&         pos,
                                      const AccumulatorState& accumulator,
                                      int                     bucket) const;

    friend class AccumulatorStack;
};

static_assert(std::is_trivially_copyable_v<HordeLegacyNetwork>);

}  // namespace Eval::NNUE
}  // namespace Stockfish

#endif  // HORDE_LEGACY_NETWORK_H_INCLUDED
