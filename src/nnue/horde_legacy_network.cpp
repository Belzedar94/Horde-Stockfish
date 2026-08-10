/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "horde_legacy_network.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <type_traits>

#include "../bitboard.h"
#include "../position.h"
#include "nnue_accumulator.h"

namespace Stockfish::Eval::NNUE {

namespace {

constexpr u32 FileVersion      = 0x7AF32F20u;
constexpr u32 NetworkHash      = 0x3C103E72u;
constexpr u32 TransformerHash  = 0x5F2348B8u;
constexpr u32 ArchitectureHash = 0x633376CAu;
constexpr int WeightScaleBits  = 6;
constexpr int OutputScale      = 16;
constexpr int StartPieceCount  = 52;

struct LegacyArtifactManifest {
    const char*                   name;
    const char*                   sha256;
    usize                         fileSize;
    std::array<unsigned char, 32> digest;
    usize                         contentHash;
};

constexpr std::array<LegacyArtifactManifest, 2> LegacyArtifactRegistry = {{
  {"Run 6B",
   "B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7",
   HordeLegacyNetwork::Run6BFileSize,
   {0xB7, 0x11, 0x08, 0x58, 0x79, 0x68, 0xAC, 0x54, 0x4E, 0xB2, 0xE6, 0x2C, 0x23, 0x33,
    0xFE, 0xCA, 0x88, 0x0D, 0xA5, 0xAC, 0xA5, 0x28, 0x66, 0x78, 0x7F, 0x14, 0x02, 0x16,
    0x34, 0x44, 0xAD, 0xF7},
   usize(0xB71108587968AC54ULL)},
  {"fresh legacy 250k seed 1",
   "3E518F19CCC381235399FD11C64E2C81BAF6983FB823FCE628C2768D5820DFEE",
   1088499,
   {0x3E, 0x51, 0x8F, 0x19, 0xCC, 0xC3, 0x81, 0x23, 0x53, 0x99, 0xFD, 0x11, 0xC6, 0x4E,
    0x2C, 0x81, 0xBA, 0xF6, 0x98, 0x3F, 0xB8, 0x23, 0xFC, 0xE6, 0x28, 0xC2, 0x76, 0x8D,
    0x58, 0x20, 0xDF, 0xEE},
   usize(0x3E518F19CCC38123ULL)},
}};

const LegacyArtifactManifest* find_manifest(const std::array<unsigned char, 32>& digest,
                                            usize                                size) {
    for (const LegacyArtifactManifest& manifest : LegacyArtifactRegistry)
        if (manifest.fileSize == size && manifest.digest == digest)
            return &manifest;
    return nullptr;
}

const LegacyArtifactManifest* manifest_by_id(u8 manifestId) {
    if (manifestId == 0 || manifestId > LegacyArtifactRegistry.size())
        return nullptr;
    return &LegacyArtifactRegistry[manifestId - 1];
}

constexpr std::array<u32, 64> Sha256Constants = {
  0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
  0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
  0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
  0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
  0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
  0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
  0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
  0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
  0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
  0xc67178f2u};

constexpr u32 rotate_right(u32 value, int bits) { return (value >> bits) | (value << (32 - bits)); }

void sha256_block(std::array<u32, 8>& state, const unsigned char* block) {
    std::array<u32, 64> words{};
    for (usize i = 0; i < 16; ++i)
        words[i] = (u32(block[4 * i]) << 24) | (u32(block[4 * i + 1]) << 16)
                 | (u32(block[4 * i + 2]) << 8) | u32(block[4 * i + 3]);

    for (usize i = 16; i < words.size(); ++i)
    {
        const u32 s0 =
          rotate_right(words[i - 15], 7) ^ rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3);
        const u32 s1 =
          rotate_right(words[i - 2], 17) ^ rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    u32 a = state[0];
    u32 b = state[1];
    u32 c = state[2];
    u32 d = state[3];
    u32 e = state[4];
    u32 f = state[5];
    u32 g = state[6];
    u32 h = state[7];

    for (usize i = 0; i < words.size(); ++i)
    {
        const u32 sum1     = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const u32 choose   = (e & f) ^ (~e & g);
        const u32 temp1    = h + sum1 + choose + Sha256Constants[i] + words[i];
        const u32 sum0     = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const u32 majority = (a & b) ^ (a & c) ^ (b & c);
        const u32 temp2    = sum0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

std::array<unsigned char, 32> sha256(const unsigned char* data, usize size) {
    std::array<u32, 8> state = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};

    const usize fullBlocks = size / 64;
    for (usize i = 0; i < fullBlocks; ++i)
        sha256_block(state, data + i * 64);

    std::array<unsigned char, 128> tail{};
    const usize                    remainder = size % 64;
    std::memcpy(tail.data(), data + fullBlocks * 64, remainder);
    tail[remainder] = 0x80;

    const usize paddingSize = remainder < 56 ? 64 : 128;
    const u64   bitLength   = u64(size) * 8;
    for (usize i = 0; i < 8; ++i)
        tail[paddingSize - 1 - i] = static_cast<unsigned char>(bitLength >> (8 * i));

    sha256_block(state, tail.data());
    if (paddingSize == 128)
        sha256_block(state, tail.data() + 64);

    std::array<unsigned char, 32> digest{};
    for (usize i = 0; i < state.size(); ++i)
        for (usize j = 0; j < 4; ++j)
            digest[4 * i + j] = static_cast<unsigned char>(state[i] >> (24 - 8 * j));
    return digest;
}

class Reader {
   public:
    Reader(const unsigned char* data, usize size) :
        data_(data),
        size_(size) {}

    template<typename T>
    bool read(T& value) {
        static_assert(std::is_integral_v<T>);
        using U = std::make_unsigned_t<T>;
        if (remaining() < sizeof(T))
            return false;

        U raw = 0;
        for (usize i = 0; i < sizeof(T); ++i)
            raw |= U(data_[offset_++]) << (8 * i);
        std::memcpy(&value, &raw, sizeof(T));
        return true;
    }

    template<typename T, usize N>
    bool read(std::array<T, N>& values) {
        for (T& value : values)
            if (!read(value))
                return false;
        return true;
    }

    bool read_string(std::string& value, usize length) {
        if (remaining() < length)
            return false;
        value.assign(reinterpret_cast<const char*>(data_ + offset_), length);
        offset_ += length;
        return true;
    }

    [[nodiscard]] usize remaining() const { return size_ - offset_; }
    [[nodiscard]] const unsigned char* current() const { return data_ + offset_; }

   private:
    const unsigned char* data_;
    usize                size_;
    usize                offset_ = 0;
};

i16 wrapping_add(i16 lhs, i16 rhs) {
    u16 left;
    u16 right;
    std::memcpy(&left, &lhs, sizeof(left));
    std::memcpy(&right, &rhs, sizeof(right));
    const u16 result = u16(left + right);
    i16       signedResult;
    std::memcpy(&signedResult, &result, sizeof(result));
    return signedResult;
}

i32 wrapping_add(i32 lhs, i32 rhs) {
    u32 left;
    u32 right;
    std::memcpy(&left, &lhs, sizeof(left));
    std::memcpy(&right, &rhs, sizeof(right));
    const u32 result = left + right;
    i32       signedResult;
    std::memcpy(&signedResult, &result, sizeof(result));
    return signedResult;
}

template<typename T>
T wrapping_sub(T lhs, T rhs) {
    static_assert(std::is_same_v<T, i16> || std::is_same_v<T, i32>);
    using U = std::make_unsigned_t<T>;
    U left;
    U right;
    std::memcpy(&left, &lhs, sizeof(left));
    std::memcpy(&right, &rhs, sizeof(right));
    const U result = U(left - right);
    T       signedResult;
    std::memcpy(&signedResult, &result, sizeof(result));
    return signedResult;
}

u8 activate(i32 value) { return static_cast<u8>(std::clamp(value >> WeightScaleBits, 0, 127)); }

}  // namespace

bool HordeLegacyNetwork::load(const unsigned char* data, usize size, std::string& description) {
    loaded_     = false;
    manifestId_ = 0;
    description.clear();

    if (!data || size < 1'000'000 || size > 2'000'000)
    {
        description = "file size is outside the registered legacy NNUE envelope";
        return false;
    }
    const auto digest = sha256(data, size);
    const auto* manifest = find_manifest(digest, size);
    if (!manifest)
    {
        constexpr char Hex[] = "0123456789ABCDEF";
        description          = "SHA-256 does not match a registered legacy NNUE artifact at this size (got ";
        for (const unsigned char byte : digest)
        {
            description += Hex[byte >> 4];
            description += Hex[byte & 0x0F];
        }
        description += ')';
        return false;
    }

    Reader reader(data, size);
    u32    version;
    u32    networkHash;
    u32    descriptionLength;
    u32    transformerHash;

    if (!reader.read(version) || !reader.read(networkHash) || !reader.read(descriptionLength)
        || descriptionLength > 1U << 20 || !reader.read_string(description, descriptionLength)
        || !reader.read(transformerHash) || version != FileVersion || networkHash != NetworkHash
        || transformerHash != TransformerHash)
    {
        description = "legacy NNUE header or feature-transformer hash is invalid";
        return false;
    }
    if (!reader.read(biases_) || !reader.read(weights_) || !reader.read(psqtWeights_))
    {
        description = "legacy feature-transformer parameters are truncated";
        return false;
    }

    std::string layerBytes(reinterpret_cast<const char*>(reader.current()), reader.remaining());
    std::istringstream layerStream(std::move(layerBytes), std::ios::in | std::ios::binary);
    for (LayerStack& layer : layers_)
    {
        const u32 architectureHash = read_little_endian<u32>(layerStream);
        if (layerStream.fail() || architectureHash != ArchitectureHash
            || !layer.fc0.read_parameters(layerStream) || !layer.fc1.read_parameters(layerStream)
            || !layer.fc2.read_parameters(layerStream))
        {
            description = "legacy layer-stack parameters or architecture hash are invalid";
            return false;
        }
    }

    loaded_ = layerStream.peek() == std::char_traits<char>::eof();
    if (!loaded_)
        description = "legacy network contains trailing bytes";
    else
        manifestId_ = static_cast<u8>(manifest - LegacyArtifactRegistry.data() + 1);
    return loaded_;
}

int HordeLegacyNetwork::bucket_for(const Position& pos) const {
    const int pieces = pos.count<ALL_PIECES>();
    return std::clamp((pieces - 1) * int(PsqtBuckets) / StartPieceCount, 0, int(PsqtBuckets) - 1);
}

void HordeLegacyNetwork::refresh_accumulator(const Position& pos, AccumulatorState& target) const {
    for (Color perspective : {WHITE, BLACK})
    {
        std::copy(biases_.begin(), biases_.end(), target.accumulation[perspective].begin());
        target.psqtAccumulation[perspective].fill(0);

        Bitboard occupied = pos.pieces();
        while (occupied)
        {
            const Square square = pop_lsb(occupied);
            const Piece  pc     = pos.piece_on(square);
            const usize  index  = HordeLegacy::feature_index(perspective, square, pc);
            assert(index < FeatureDimensions);

            const usize offset = index * AccumulatorDimensions;
            for (usize i = 0; i < AccumulatorDimensions; ++i)
                target.accumulation[perspective][i] =
                  wrapping_add(target.accumulation[perspective][i], weights_[offset + i]);

            for (usize i = 0; i < PsqtBuckets; ++i)
                target.psqtAccumulation[perspective][i] = wrapping_add(
                  target.psqtAccumulation[perspective][i], psqtWeights_[index * PsqtBuckets + i]);
        }
        target.computed[perspective] = true;
    }
}

void HordeLegacyNetwork::update_accumulator(const DirtyPiece&       dirty,
                                            const AccumulatorState& source,
                                            AccumulatorState&       target) const {
    for (Color perspective : {WHITE, BLACK})
    {
        std::copy_n(source.accumulation[perspective].begin(), AccumulatorDimensions,
                    target.accumulation[perspective].begin());
        target.psqtAccumulation[perspective] = source.psqtAccumulation[perspective];
    }

    const auto apply = [&](Piece pc, Square square, bool add) {
        if (square == SQ_NONE)
            return;
        assert(pc != NO_PIECE);

        for (Color perspective : {WHITE, BLACK})
        {
            const usize index  = HordeLegacy::feature_index(perspective, square, pc);
            const usize offset = index * AccumulatorDimensions;
            assert(index < FeatureDimensions);

            for (usize i = 0; i < AccumulatorDimensions; ++i)
                target.accumulation[perspective][i] =
                  add ? wrapping_add(target.accumulation[perspective][i], weights_[offset + i])
                      : wrapping_sub(target.accumulation[perspective][i], weights_[offset + i]);

            for (usize i = 0; i < PsqtBuckets; ++i)
            {
                const i32 weight = psqtWeights_[index * PsqtBuckets + i];
                target.psqtAccumulation[perspective][i] =
                  add ? wrapping_add(target.psqtAccumulation[perspective][i], weight)
                      : wrapping_sub(target.psqtAccumulation[perspective][i], weight);
            }
        }
    };

    apply(dirty.pc, dirty.from, false);
    apply(dirty.pc, dirty.to, true);
    if (dirty.remove_sq != SQ_NONE)
        apply(dirty.remove_pc, dirty.remove_sq, false);
    if (dirty.add_sq != SQ_NONE)
        apply(dirty.add_pc, dirty.add_sq, true);

    target.computed.fill(true);
}

HordeLegacyNetwork::RawOutput HordeLegacyNetwork::propagate(
  const Position& pos, const AccumulatorState& accumulator, int bucket) const {
    bucket = bucket < 0 ? bucket_for(pos) : std::clamp(bucket, 0, int(LayerStacks) - 1);
    const std::array<Color, COLOR_NB> perspectives = {pos.side_to_move(), ~pos.side_to_move()};
    alignas(CacheLineSize) std::array<u8, NetworkInputs> transformed{};
    for (usize p = 0; p < COLOR_NB; ++p)
        for (usize i = 0; i < AccumulatorDimensions; ++i)
            transformed[p * AccumulatorDimensions + i] =
              static_cast<u8>(
                std::clamp<int>(accumulator.accumulation[perspectives[p]][i], 0, 127));

    const i32 materialist = (accumulator.psqtAccumulation[perspectives[0]][bucket]
                             - accumulator.psqtAccumulation[perspectives[1]][bucket])
                          / 2;

    const LayerStack& layer = layers_[bucket];

    alignas(CacheLineSize) LayerStack::Fc0::OutputBuffer fc0Output{};
    alignas(CacheLineSize) std::array<u8, LayerStack::Fc0::PaddedOutputDimensions> hidden0{};
    layer.fc0.propagate(transformed.data(), fc0Output);
    for (usize output = 0; output < LayerStack::Fc0::OutputDimensions; ++output)
        hidden0[output] = activate(fc0Output[output]);

    alignas(CacheLineSize) LayerStack::Fc1::OutputBuffer fc1Output{};
    alignas(CacheLineSize) std::array<u8, LayerStack::Fc1::PaddedOutputDimensions> hidden1{};
    layer.fc1.propagate(hidden0.data(), fc1Output);
    for (usize output = 0; output < LayerStack::Fc1::OutputDimensions; ++output)
        hidden1[output] = activate(fc1Output[output]);

    alignas(CacheLineSize) LayerStack::Fc2::OutputBuffer positional{};
    layer.fc2.propagate(hidden1.data(), positional);

    return {materialist, positional[0]};
}

void AccumulatorStack::evaluate_horde_legacy(const Position&           pos,
                                             const HordeLegacyNetwork& network) noexcept {
    usize begin = size - 1;
    while (begin > 0
           && !(accumulators[begin].computed[WHITE] && accumulators[begin].computed[BLACK]))
        --begin;

    if (accumulators[begin].computed[WHITE] && accumulators[begin].computed[BLACK])
        for (usize next = begin + 1; next < size; ++next)
            network.update_accumulator(accumulators[next].dirtyPiece, accumulators[next - 1],
                                       accumulators[next]);
    else
        network.refresh_accumulator(pos, mut_latest());
}

HordeLegacyNetwork::RawOutput HordeLegacyNetwork::evaluate_raw(
  const Position& pos, AccumulatorStack& accumulatorStack, int bucket) const {
    assert(loaded_);
    accumulatorStack.evaluate_horde_legacy(pos, *this);
    const RawOutput result = propagate(pos, accumulatorStack.latest(), bucket);

#if !defined(NDEBUG)
    constexpr bool shadow = true;
#elif defined(HORDE_NNUE_SHADOW)
    static thread_local u64 shadowCounter = 0;
    const bool              shadow        = (++shadowCounter & 1023) == 0;
#endif
#if !defined(NDEBUG) || defined(HORDE_NNUE_SHADOW)
    if (shadow && result != evaluate_raw_full_refresh(pos, bucket))
        std::abort();
#endif

    return result;
}

HordeLegacyNetwork::RawOutput HordeLegacyNetwork::evaluate_raw_full_refresh(const Position& pos,
                                                                            int bucket) const {
    assert(loaded_);
    AccumulatorState accumulator{};
    refresh_accumulator(pos, accumulator);
    return propagate(pos, accumulator, bucket);
}

usize HordeLegacyNetwork::content_hash() const {
    const auto* manifest = loaded_ ? manifest_by_id(manifestId_) : nullptr;
    return manifest ? manifest->contentHash : 0;
}

const char* HordeLegacyNetwork::artifact_name() const {
    const auto* manifest = loaded_ ? manifest_by_id(manifestId_) : nullptr;
    return manifest ? manifest->name : "unloaded";
}

const char* HordeLegacyNetwork::artifact_sha256() const {
    const auto* manifest = loaded_ ? manifest_by_id(manifestId_) : nullptr;
    return manifest ? manifest->sha256 : "";
}

}  // namespace Stockfish::Eval::NNUE
