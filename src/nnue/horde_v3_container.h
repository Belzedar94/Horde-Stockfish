/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_V3_CONTAINER_H_INCLUDED
#define HORDE_V3_CONTAINER_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

#include "horde_v3_stack.h"

// Fail-closed reader for the V3 network schema 0x00020001, registered in
// tools/horde_v3_container.py. The envelope is the unchanged
// HORDE_V2_INTEGER_NETWORK_V1 container: the same magic, the same 2,048 byte
// little-endian header and the same authenticated section directory.
//
// Every framing field, both identity digests, all ten directory entries with
// their per-section digest, and the V3 header extension block are re-checked
// here. Any failure invalidates the parameters; a caller never falls back to a
// previously loaded network.

namespace Stockfish::Eval::NNUE::HordeV3 {

inline constexpr std::size_t   V3ContainerHeaderBytes    = 2048;
inline constexpr std::uint32_t V3NetworkSchemaId         = 0x00020001;
inline constexpr std::size_t   V3ContainerSectionCount   = 10;
inline constexpr std::size_t   V3ContainerParameterBytes = 2033765;

enum class V3ContainerLoadError : std::uint8_t {
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
    PHASE_LOOKUP,
};

const char* v3_container_load_error_name(V3ContainerLoadError error) noexcept;

struct V3ContainerLoadResult {
    V3ContainerLoadError error = V3ContainerLoadError::NONE;
    std::string          message;
    std::string          provenance;
    std::string          trainingStructuralSha256;
    V3Parameters         parameters;

    [[nodiscard]] explicit operator bool() const noexcept {
        return error == V3ContainerLoadError::NONE && parameters.valid();
    }
};

V3ContainerLoadResult load_v3_container(const std::filesystem::path& path);

}  // namespace Stockfish::Eval::NNUE::HordeV3

#endif  // HORDE_V3_CONTAINER_H_INCLUDED
