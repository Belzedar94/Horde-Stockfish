/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_V3_FEATURES_H_INCLUDED
#define HORDE_V3_FEATURES_H_INCLUDED

#include <array>
#include <cstddef>
#include <cstdint>

#include "../types.h"
#include "horde_v2_features.h"

// The V3 stream. One sparse transformer over a fixed Horde frame: 896 rows.
//
//   rows 0..703    G0, index = fixed_role * 64 + absolute square, one row per
//                  physical piece, at most 52 active.
//   rows 704..895  six contextual White-pawn blocks. Every block is keyed on
//                  the frontier pawn of a file, so each activates at most one
//                  row per file and the tail has at most 48 active rows.
//
// This header is the C++ mirror of ``v3_global_rows`` in
// ``tools/horde_training_decoder.py``. It must produce the same sorted row
// multiset for every legal Horde board; there is no Royal domain and no
// RoyalKey. Every violation of the physical Horde contract fails closed.

namespace Stockfish::Eval::NNUE::HordeV3 {

using HordeV2::fixed_role;
using HordeV2::fixed_role_piece_square_index;
using HordeV2::FixedRole;
using HordeV2::InvalidFeatureIndex;
using HordeV2::is_registered_piece;

// ---------------------------------------------------------------------------
// Frozen stream geometry. Mirrors V3_BLOCK_BASES / V3_BLOCK_WIDTHS.
// ---------------------------------------------------------------------------

inline constexpr IndexType G0Rows         = 704;
inline constexpr IndexType ContextualRows = 192;
inline constexpr IndexType V3StreamRows   = G0Rows + ContextualRows;  // 896

inline constexpr int FileNb          = 8;
inline constexpr int RankNb          = 8;
inline constexpr int PawnRanks       = 7;   // a White pawn never stands on rank 8
inline constexpr int FileCountStates = 7;   // a file holds 1 to 7 White pawns

enum ContextualBlock : std::uint8_t {
    FRONTIER_SQUARE,
    REARMOST_SQUARE,
    FILE_COUNT,
    FRONTIER_BLOCKED,
    FRONTIER_SUPPORTED,
    FRONTIER_PHALANX,
    CONTEXTUAL_BLOCK_NB
};

inline constexpr std::array<IndexType, CONTEXTUAL_BLOCK_NB> ContextualBlockBase = {
  704, 760, 816, 872, 880, 888};
inline constexpr std::array<IndexType, CONTEXTUAL_BLOCK_NB> ContextualBlockWidth = {
  56, 56, 56, 8, 8, 8};

inline constexpr std::size_t MaxG0Rows         = 52;
inline constexpr std::size_t MaxContextualRows = 48;
inline constexpr std::size_t MaxActiveRows     = MaxG0Rows + MaxContextualRows;  // 100
inline constexpr std::size_t MaxRowsPerFile    = CONTEXTUAL_BLOCK_NB;
inline constexpr std::size_t MaxWhitePieces    = 36;
inline constexpr std::size_t MaxBlackPieces    = 16;
inline constexpr std::size_t MaxHordePieces    = 52;

constexpr const char* contextual_block_name(ContextualBlock block) noexcept {
    switch (block)
    {
    case FRONTIER_SQUARE :
        return "frontier_square";
    case REARMOST_SQUARE :
        return "rearmost_square";
    case FILE_COUNT :
        return "file_count";
    case FRONTIER_BLOCKED :
        return "frontier_blocked";
    case FRONTIER_SUPPORTED :
        return "frontier_supported";
    case FRONTIER_PHALANX :
        return "frontier_phalanx";
    default :
        return "unknown";
    }
}

// Name the block a stream row belongs to. Used by the parity reports.
constexpr const char* v3_row_block_name(IndexType row) noexcept {
    if (row < G0Rows)
        return "g0";
    for (std::size_t block = 0; block < CONTEXTUAL_BLOCK_NB; ++block)
        if (row >= ContextualBlockBase[block]
            && row < ContextualBlockBase[block] + ContextualBlockWidth[block])
            return contextual_block_name(ContextualBlock(block));
    return "out-of-range";
}

// ---------------------------------------------------------------------------
// Physical pawn state the six contextual blocks are keyed on.
//
// The two bitboards are the minimum physical state that makes an incremental
// transition exact: a move can change the contextual role of a pawn that did
// not move, and (frontier, rearmost, count) alone cannot be updated from a
// DirtyPiece because removing the frontier pawn of a file exposes a pawn whose
// rank the triple never recorded.
// ---------------------------------------------------------------------------

struct V3PawnStructure {
    std::uint64_t                          whitePawns = 0;
    std::uint64_t                          occupancy  = 0;
    std::array<std::int8_t, FileNb>        frontier{{-1, -1, -1, -1, -1, -1, -1, -1}};
    std::array<std::int8_t, FileNb>        rearmost{{-1, -1, -1, -1, -1, -1, -1, -1}};
    std::array<std::uint8_t, FileNb>       count{};
    std::uint8_t                           blocked   = 0;  // one bit per file
    std::uint8_t                           supported = 0;
    std::uint8_t                           phalanx   = 0;
};

constexpr bool operator==(const V3PawnStructure& lhs, const V3PawnStructure& rhs) noexcept {
    return lhs.whitePawns == rhs.whitePawns && lhs.occupancy == rhs.occupancy
        && lhs.frontier == rhs.frontier && lhs.rearmost == rhs.rearmost && lhs.count == rhs.count
        && lhs.blocked == rhs.blocked && lhs.supported == rhs.supported
        && lhs.phalanx == rhs.phalanx;
}

constexpr bool operator!=(const V3PawnStructure& lhs, const V3PawnStructure& rhs) noexcept {
    return !(lhs == rhs);
}

constexpr std::uint64_t square_bit(int square) noexcept {
    return std::uint64_t(1) << square;
}

constexpr int square_of(int rank, int file) noexcept { return rank * 8 + file; }

constexpr bool file_bit(std::uint8_t mask, int file) noexcept {
    return (mask & std::uint8_t(1u << file)) != 0;
}

// Rebuild (count, frontier, rearmost) for one file out of the pawn bitboard.
// Returns false when the file holds a White pawn on rank 8, which is illegal.
[[nodiscard]] constexpr bool v3_recompute_file(V3PawnStructure& state, int file) noexcept {
    if (state.whitePawns & square_bit(square_of(7, file)))
        return false;

    int pawns    = 0;
    int frontier = -1;
    int rearmost = -1;
    for (int rank = 0; rank < PawnRanks; ++rank)
        if (state.whitePawns & square_bit(square_of(rank, file)))
        {
            ++pawns;
            if (rearmost < 0)
                rearmost = rank;
            frontier = rank;
        }

    // Only seven ranks can hold a White pawn, so the file-count block can never
    // overflow its seven states. The check stays as a contract assertion.
    if (pawns > FileCountStates)
        return false;

    state.count[file]    = std::uint8_t(pawns);
    state.frontier[file] = std::int8_t(frontier);
    state.rearmost[file] = std::int8_t(rearmost);
    return true;
}

// Rebuild the three frontier predicates of one file. They read the neighbouring
// files' White pawns, so a file must be recomputed whenever a neighbour changes.
constexpr void v3_recompute_predicates(V3PawnStructure& state, int file) noexcept {
    const std::uint8_t bit  = std::uint8_t(1u << file);
    const std::uint8_t keep = std::uint8_t(~bit);
    state.blocked           = std::uint8_t(state.blocked & keep);
    state.supported         = std::uint8_t(state.supported & keep);
    state.phalanx           = std::uint8_t(state.phalanx & keep);

    const int rank = state.frontier[file];
    if (rank < 0)
        return;

    if (state.occupancy & square_bit(square_of(rank + 1, file)))
        state.blocked = std::uint8_t(state.blocked | bit);

    for (int delta = -1; delta <= 1; delta += 2)
    {
        const int neighbour = file + delta;
        if (neighbour < 0 || neighbour >= FileNb)
            continue;
        if (rank >= 1 && (state.whitePawns & square_bit(square_of(rank - 1, neighbour))))
            state.supported = std::uint8_t(state.supported | bit);
        if (state.whitePawns & square_bit(square_of(rank, neighbour)))
            state.phalanx = std::uint8_t(state.phalanx | bit);
    }
}

[[nodiscard]] constexpr bool v3_rebuild_pawn_structure(V3PawnStructure& state) noexcept {
    for (int file = 0; file < FileNb; ++file)
        if (!v3_recompute_file(state, file))
            return false;
    for (int file = 0; file < FileNb; ++file)
        v3_recompute_predicates(state, file);
    return true;
}

// The contextual rows contributed by one file, emitted in block order. Returns
// the number of rows written; a file with no White pawn contributes none.
constexpr std::size_t
v3_file_contextual_rows(const V3PawnStructure& state, int file, IndexType* out) noexcept {
    if (state.count[file] == 0)
        return 0;

    const IndexType frontier = IndexType(state.frontier[file]);
    const IndexType rearmost = IndexType(state.rearmost[file]);
    std::size_t     written  = 0;

    out[written++] = ContextualBlockBase[FRONTIER_SQUARE] + frontier * 8 + IndexType(file);
    out[written++] = ContextualBlockBase[REARMOST_SQUARE] + rearmost * 8 + IndexType(file);
    out[written++] = ContextualBlockBase[FILE_COUNT] + IndexType(file) * FileCountStates
                   + IndexType(state.count[file]) - 1;
    if (file_bit(state.blocked, file))
        out[written++] = ContextualBlockBase[FRONTIER_BLOCKED] + IndexType(file);
    if (file_bit(state.supported, file))
        out[written++] = ContextualBlockBase[FRONTIER_SUPPORTED] + IndexType(file);
    if (file_bit(state.phalanx, file))
        out[written++] = ContextualBlockBase[FRONTIER_PHALANX] + IndexType(file);
    return written;
}

// The whole contextual tail, block by block and file by file in ascending
// order. This is the emission order of ``v3_contextual_rows``.
constexpr std::size_t v3_contextual_rows(const V3PawnStructure& state, IndexType* out) noexcept {
    std::size_t written = 0;
    for (int file = 0; file < FileNb; ++file)
        if (state.count[file] != 0)
            out[written++] =
              ContextualBlockBase[FRONTIER_SQUARE] + IndexType(state.frontier[file]) * 8
              + IndexType(file);
    for (int file = 0; file < FileNb; ++file)
        if (state.count[file] != 0)
            out[written++] =
              ContextualBlockBase[REARMOST_SQUARE] + IndexType(state.rearmost[file]) * 8
              + IndexType(file);
    for (int file = 0; file < FileNb; ++file)
        if (state.count[file] != 0)
            out[written++] = ContextualBlockBase[FILE_COUNT] + IndexType(file) * FileCountStates
                           + IndexType(state.count[file]) - 1;
    for (int file = 0; file < FileNb; ++file)
        if (file_bit(state.blocked, file))
            out[written++] = ContextualBlockBase[FRONTIER_BLOCKED] + IndexType(file);
    for (int file = 0; file < FileNb; ++file)
        if (file_bit(state.supported, file))
            out[written++] = ContextualBlockBase[FRONTIER_SUPPORTED] + IndexType(file);
    for (int file = 0; file < FileNb; ++file)
        if (file_bit(state.phalanx, file))
            out[written++] = ContextualBlockBase[FRONTIER_PHALANX] + IndexType(file);
    return written;
}

// ---------------------------------------------------------------------------
// Full enumeration
// ---------------------------------------------------------------------------

enum class V3FeatureError : std::uint8_t {
    NONE,
    INVALID_PIECE,
    WHITE_KING,
    BLACK_KING_COUNT,
    TOO_MANY_WHITE_PIECES,
    TOO_MANY_BLACK_PIECES,
    TOO_MANY_PIECES,
    WHITE_PAWN_ON_RANK_8,
    ROW_OUT_OF_RANGE,
    DUPLICATE_ROW,
    ACTIVE_ROW_OVERFLOW,
};

constexpr const char* v3_feature_error_name(V3FeatureError error) noexcept {
    switch (error)
    {
    case V3FeatureError::NONE :
        return "NONE";
    case V3FeatureError::INVALID_PIECE :
        return "INVALID_PIECE";
    case V3FeatureError::WHITE_KING :
        return "WHITE_KING";
    case V3FeatureError::BLACK_KING_COUNT :
        return "BLACK_KING_COUNT";
    case V3FeatureError::TOO_MANY_WHITE_PIECES :
        return "TOO_MANY_WHITE_PIECES";
    case V3FeatureError::TOO_MANY_BLACK_PIECES :
        return "TOO_MANY_BLACK_PIECES";
    case V3FeatureError::TOO_MANY_PIECES :
        return "TOO_MANY_PIECES";
    case V3FeatureError::WHITE_PAWN_ON_RANK_8 :
        return "WHITE_PAWN_ON_RANK_8";
    case V3FeatureError::ROW_OUT_OF_RANGE :
        return "ROW_OUT_OF_RANGE";
    case V3FeatureError::DUPLICATE_ROW :
        return "DUPLICATE_ROW";
    case V3FeatureError::ACTIVE_ROW_OVERFLOW :
        return "ACTIVE_ROW_OVERFLOW";
    }
    return "UNKNOWN";
}

struct V3Features {
    std::array<IndexType, MaxActiveRows> rows{};
    std::size_t                          size            = 0;
    std::size_t                          g0Size          = 0;
    V3PawnStructure                      pawns{};
    std::uint16_t                        whitePieceCount = 0;
    std::uint16_t                        blackPieceCount = 0;
    Square                               blackKingSquare = SQ_NONE;
    V3FeatureError                       error           = V3FeatureError::NONE;

    [[nodiscard]] constexpr bool valid() const noexcept { return error == V3FeatureError::NONE; }
};

// Enumerate the whole V3 stream for one 64-square board, in the exact emission
// order of the trainer: G0 in A1-to-H8 square order, then the contextual tail
// block by block and file by file.
[[nodiscard]] constexpr V3Features
extract_v3_features(const std::array<Piece, SQUARE_NB>& board) noexcept {
    V3Features features{};

    std::size_t whitePieces = 0;
    std::size_t blackPieces = 0;
    std::size_t blackKings  = 0;

    for (int rawSquare = 0; rawSquare < SQUARE_NB; ++rawSquare)
    {
        const Piece piece = board[rawSquare];
        if (piece == NO_PIECE)
            continue;
        if (piece == W_KING)
        {
            features.error = V3FeatureError::WHITE_KING;
            return features;
        }
        if (!is_registered_piece(piece))
        {
            features.error = V3FeatureError::INVALID_PIECE;
            return features;
        }

        features.pawns.occupancy |= square_bit(rawSquare);
        if (color_of(piece) == WHITE)
        {
            ++whitePieces;
            if (piece == W_PAWN)
                features.pawns.whitePawns |= square_bit(rawSquare);
        }
        else
        {
            ++blackPieces;
            if (piece == B_KING)
            {
                ++blackKings;
                features.blackKingSquare = Square(rawSquare);
            }
        }
    }

    if (blackKings != 1)
    {
        features.error = V3FeatureError::BLACK_KING_COUNT;
        return features;
    }
    if (whitePieces > MaxWhitePieces)
    {
        features.error = V3FeatureError::TOO_MANY_WHITE_PIECES;
        return features;
    }
    if (blackPieces > MaxBlackPieces)
    {
        features.error = V3FeatureError::TOO_MANY_BLACK_PIECES;
        return features;
    }
    if (whitePieces + blackPieces > MaxHordePieces)
    {
        features.error = V3FeatureError::TOO_MANY_PIECES;
        return features;
    }
    features.whitePieceCount = std::uint16_t(whitePieces);
    features.blackPieceCount = std::uint16_t(blackPieces);

    for (int rawSquare = 0; rawSquare < SQUARE_NB; ++rawSquare)
    {
        const Piece piece = board[rawSquare];
        if (piece == NO_PIECE)
            continue;
        const IndexType row = fixed_role_piece_square_index(piece, Square(rawSquare));
        if (row == InvalidFeatureIndex)
        {
            features.error = V3FeatureError::INVALID_PIECE;
            return features;
        }
        features.rows[features.size++] = row;
    }
    features.g0Size = features.size;

    if (!v3_rebuild_pawn_structure(features.pawns))
    {
        features.error = V3FeatureError::WHITE_PAWN_ON_RANK_8;
        return features;
    }

    std::array<IndexType, MaxContextualRows> contextual{};
    const std::size_t written = v3_contextual_rows(features.pawns, contextual.data());
    if (written > MaxContextualRows || features.size + written > MaxActiveRows)
    {
        features.error = V3FeatureError::ACTIVE_ROW_OVERFLOW;
        return features;
    }
    for (std::size_t index = 0; index < written; ++index)
        features.rows[features.size++] = contextual[index];

    for (std::size_t index = 0; index < features.size; ++index)
    {
        if (features.rows[index] >= V3StreamRows)
        {
            features.error = V3FeatureError::ROW_OUT_OF_RANGE;
            return features;
        }
        for (std::size_t other = index + 1; other < features.size; ++other)
            if (features.rows[index] == features.rows[other])
            {
                features.error = V3FeatureError::DUPLICATE_ROW;
                return features;
            }
    }

    return features;
}

static_assert(V3StreamRows == 896);
static_assert(MaxActiveRows == 100);
static_assert(ContextualBlockBase[FRONTIER_SQUARE] == 704);
static_assert(ContextualBlockBase[REARMOST_SQUARE] == 760);
static_assert(ContextualBlockBase[FILE_COUNT] == 816);
static_assert(ContextualBlockBase[FRONTIER_BLOCKED] == 872);
static_assert(ContextualBlockBase[FRONTIER_SUPPORTED] == 880);
static_assert(ContextualBlockBase[FRONTIER_PHALANX] == 888);
static_assert(ContextualBlockBase[FRONTIER_PHALANX] + ContextualBlockWidth[FRONTIER_PHALANX]
              == V3StreamRows);
static_assert(fixed_role_piece_square_index(W_PAWN, SQ_A1) == 0);
static_assert(fixed_role_piece_square_index(B_KING, SQ_H8) == G0Rows - 1);
static_assert(v3_row_block_name(0)[0] == 'g');
static_assert(v3_row_block_name(895)[0] == 'f');

}  // namespace Stockfish::Eval::NNUE::HordeV3

#endif  // HORDE_V3_FEATURES_H_INCLUDED
