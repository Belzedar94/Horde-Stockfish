/*
  Horde-Stockfish, a UCI chess variant playing engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "sha256.h"

#include <algorithm>
#include <fstream>

namespace Stockfish::Data {
namespace {

constexpr std::array<std::uint32_t, 64> Constants = {
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

constexpr std::uint32_t rotate_right(std::uint32_t value, int bits) {
    return (value >> bits) | (value << (32 - bits));
}

}  // namespace

Sha256::Sha256() :
    state{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
          0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

void Sha256::transform(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t i = 0; i < 16; ++i)
        words[i] = (std::uint32_t(block[4 * i]) << 24) | (std::uint32_t(block[4 * i + 1]) << 16)
                 | (std::uint32_t(block[4 * i + 2]) << 8) | std::uint32_t(block[4 * i + 3]);

    for (std::size_t i = 16; i < words.size(); ++i)
    {
        const std::uint32_t s0 =
          rotate_right(words[i - 15], 7) ^ rotate_right(words[i - 15], 18) ^ (words[i - 15] >> 3);
        const std::uint32_t s1 =
          rotate_right(words[i - 2], 17) ^ rotate_right(words[i - 2], 19) ^ (words[i - 2] >> 10);
        words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    std::uint32_t a = state[0];
    std::uint32_t b = state[1];
    std::uint32_t c = state[2];
    std::uint32_t d = state[3];
    std::uint32_t e = state[4];
    std::uint32_t f = state[5];
    std::uint32_t g = state[6];
    std::uint32_t h = state[7];

    for (std::size_t i = 0; i < words.size(); ++i)
    {
        const std::uint32_t sum1   = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const std::uint32_t choose = (e & f) ^ (~e & g);
        const std::uint32_t temp1  = h + sum1 + choose + Constants[i] + words[i];
        const std::uint32_t sum0   = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const std::uint32_t temp2    = sum0 + majority;

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

void Sha256::update(const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    totalBytes += std::uint64_t(size);

    if (bufferSize)
    {
        const std::size_t copied = std::min(size, buffer.size() - bufferSize);
        std::copy_n(bytes, copied, buffer.begin() + std::ptrdiff_t(bufferSize));
        bufferSize += copied;
        bytes += copied;
        size -= copied;
        if (bufferSize == buffer.size())
        {
            transform(buffer.data());
            bufferSize = 0;
        }
    }

    while (size >= buffer.size())
    {
        transform(bytes);
        bytes += buffer.size();
        size -= buffer.size();
    }

    if (size)
    {
        std::copy_n(bytes, size, buffer.begin());
        bufferSize = size;
    }
}

Sha256Digest Sha256::digest() const {
    Sha256 final = *this;

    final.buffer[final.bufferSize++] = 0x80;
    if (final.bufferSize > 56)
    {
        std::fill(final.buffer.begin() + std::ptrdiff_t(final.bufferSize), final.buffer.end(), 0);
        final.transform(final.buffer.data());
        final.bufferSize = 0;
    }

    std::fill(final.buffer.begin() + std::ptrdiff_t(final.bufferSize), final.buffer.begin() + 56,
              0);
    const std::uint64_t bitLength = totalBytes * 8;
    for (std::size_t i = 0; i < 8; ++i)
        final.buffer[63 - i] = std::uint8_t(bitLength >> (8 * i));
    final.transform(final.buffer.data());

    Sha256Digest result{};
    for (std::size_t i = 0; i < final.state.size(); ++i)
        for (std::size_t j = 0; j < 4; ++j)
            result[4 * i + j] = std::uint8_t(final.state[i] >> (24 - 8 * j));
    return result;
}

std::string sha256_hex_upper(const Sha256Digest& digest) {
    constexpr char Hex[] = "0123456789ABCDEF";
    std::string    result;
    result.reserve(64);
    for (const std::uint8_t byte : digest)
    {
        result.push_back(Hex[byte >> 4]);
        result.push_back(Hex[byte & 0x0F]);
    }
    return result;
}

bool is_upper_sha256(std::string_view text) {
    return text.size() == 64 && std::all_of(text.begin(), text.end(), [](unsigned char c) {
               return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'F');
           });
}

bool sha256_file(const std::filesystem::path& path, std::string& digest, std::string& error) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        error = "Cannot open file for SHA-256: " + path.string();
        return false;
    }

    Sha256                      hasher;
    std::array<char, 64 * 1024> buffer{};
    while (input)
    {
        input.read(buffer.data(), std::streamsize(buffer.size()));
        const auto count = input.gcount();
        if (count > 0)
            hasher.update(buffer.data(), std::size_t(count));
    }
    if (!input.eof())
    {
        error = "Cannot read file for SHA-256: " + path.string();
        return false;
    }

    digest = sha256_hex_upper(hasher.digest());
    return true;
}

}  // namespace Stockfish::Data
