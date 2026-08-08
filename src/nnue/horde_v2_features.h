/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_V2_FEATURES_H_INCLUDED
#define HORDE_V2_FEATURES_H_INCLUDED

#include "../types.h"
#include "nnue_common.h"

namespace Stockfish::Eval::NNUE::HordeV2 {

// FixedRole is an evaluation identity only. It never changes the physical
// Piece stored by Position. In particular, every pawn remains PAWN.
enum FixedRole : int {
    HORDE_PAWN,
    HORDE_KNIGHT,
    HORDE_BISHOP,
    HORDE_ROOK,
    HORDE_QUEEN,
    ROYAL_PAWN,
    ROYAL_KNIGHT,
    ROYAL_BISHOP,
    ROYAL_ROOK,
    ROYAL_QUEEN,
    ROYAL_KING,
    FIXED_ROLE_NB
};

inline constexpr IndexType FixedRolePieceSquareDimensions = FIXED_ROLE_NB * SQUARE_NB;
inline constexpr IndexType InvalidFeatureIndex            = FixedRolePieceSquareDimensions;

inline constexpr IndexType RoyalNonKingRoleCount = ROYAL_KING;
inline constexpr IndexType RoyalBucketCount      = RANK_NB * 4;
inline constexpr IndexType RoyalPieceSquareDimensions =
  RoyalBucketCount * RoyalNonKingRoleCount * SQUARE_NB;
inline constexpr IndexType InvalidRoyalFeatureIndex = RoyalPieceSquareDimensions;

struct RoyalKey {
    IndexType bucket;
    bool      mirror;
};

constexpr bool operator==(RoyalKey lhs, RoyalKey rhs) noexcept {
    return lhs.bucket == rhs.bucket && lhs.mirror == rhs.mirror;
}

constexpr bool operator!=(RoyalKey lhs, RoyalKey rhs) noexcept { return !(lhs == rhs); }

constexpr FixedRole fixed_role(Piece piece) noexcept {
    switch (piece)
    {
    case W_PAWN :
        return HORDE_PAWN;
    case W_KNIGHT :
        return HORDE_KNIGHT;
    case W_BISHOP :
        return HORDE_BISHOP;
    case W_ROOK :
        return HORDE_ROOK;
    case W_QUEEN :
        return HORDE_QUEEN;
    case B_PAWN :
        return ROYAL_PAWN;
    case B_KNIGHT :
        return ROYAL_KNIGHT;
    case B_BISHOP :
        return ROYAL_BISHOP;
    case B_ROOK :
        return ROYAL_ROOK;
    case B_QUEEN :
        return ROYAL_QUEEN;
    case B_KING :
        return ROYAL_KING;
    default :
        return FIXED_ROLE_NB;
    }
}

constexpr bool is_registered_piece(Piece piece) noexcept {
    return fixed_role(piece) != FIXED_ROLE_NB;
}

constexpr IndexType fixed_role_piece_square_index(Piece piece, Square square) noexcept {
    const FixedRole role = fixed_role(piece);
    return role == FIXED_ROLE_NB || !is_ok(square)
           ? InvalidFeatureIndex
           : IndexType(role) * SQUARE_NB + IndexType(square);
}

// The absolute first-domain control deliberately shares the first ten Global
// rows (all registered non-king roles) and excludes the final Black-king role.
constexpr IndexType absolute_nonking_piece_square_index(Piece piece, Square square) noexcept {
    const FixedRole role = fixed_role(piece);
    return role >= ROYAL_KING || !is_ok(square)
           ? InvalidFeatureIndex
           : IndexType(role) * SQUARE_NB + IndexType(square);
}

constexpr IndexType absolute_nonking_index_from_global(IndexType globalIndex) noexcept {
    return globalIndex < RoyalNonKingRoleCount * SQUARE_NB ? globalIndex : InvalidFeatureIndex;
}

constexpr Square horizontal_flip(Square square) noexcept {
    return is_ok(square) ? Square(IndexType(square) ^ 7) : SQ_NONE;
}

constexpr bool royal_mirror(Square blackKingSquare) noexcept {
    return is_ok(blackKingSquare) && file_of(blackKingSquare) <= FILE_D;
}

constexpr Square royal_orient(Square square, bool mirror) noexcept {
    return mirror ? horizontal_flip(square) : square;
}

constexpr IndexType royal_bucket(Square blackKingSquare) noexcept {
    if (!is_ok(blackKingSquare))
        return RoyalBucketCount;

    const Square canonicalKing = royal_orient(blackKingSquare, royal_mirror(blackKingSquare));
    return IndexType(rank_of(canonicalKing)) * 4 + IndexType(file_of(canonicalKing) - FILE_E);
}

constexpr RoyalKey royal_key(Square blackKingSquare) noexcept {
    return {royal_bucket(blackKingSquare), royal_mirror(blackKingSquare)};
}

constexpr bool is_valid_royal_key(RoyalKey key) noexcept { return key.bucket < RoyalBucketCount; }

constexpr IndexType royal_piece_square_index(Piece piece, Square square, RoyalKey key) noexcept {
    const FixedRole role = fixed_role(piece);
    return !is_valid_royal_key(key) || role >= ROYAL_KING || !is_ok(square)
           ? InvalidRoyalFeatureIndex
           : (key.bucket * RoyalNonKingRoleCount + IndexType(role)) * SQUARE_NB
               + IndexType(royal_orient(square, key.mirror));
}

constexpr IndexType
royal_piece_square_index(Piece piece, Square square, Square blackKingSquare) noexcept {
    return royal_piece_square_index(piece, square, royal_key(blackKingSquare));
}

static_assert(FixedRolePieceSquareDimensions == 704);
static_assert(RoyalNonKingRoleCount == 10);
static_assert(RoyalBucketCount == 32);
static_assert(RoyalPieceSquareDimensions == 20480);
static_assert(fixed_role_piece_square_index(W_PAWN, SQ_A1) == 0);
static_assert(fixed_role_piece_square_index(W_QUEEN, SQ_H8) == 4 * 64 + 63);
static_assert(fixed_role_piece_square_index(B_PAWN, SQ_A1) == 5 * 64);
static_assert(fixed_role_piece_square_index(B_KING, SQ_H8) == 10 * 64 + 63);
static_assert(absolute_nonking_piece_square_index(W_PAWN, SQ_A1) == 0);
static_assert(absolute_nonking_piece_square_index(B_QUEEN, SQ_H8) == 10 * 64 - 1);
static_assert(absolute_nonking_piece_square_index(B_KING, SQ_A1) == InvalidFeatureIndex);
static_assert(absolute_nonking_index_from_global(10 * 64 - 1) == 10 * 64 - 1);
static_assert(absolute_nonking_index_from_global(10 * 64) == InvalidFeatureIndex);
static_assert(fixed_role_piece_square_index(W_KING, SQ_E1) == InvalidFeatureIndex);
static_assert(fixed_role_piece_square_index(NO_PIECE, SQ_A1) == InvalidFeatureIndex);
static_assert(horizontal_flip(SQ_A1) == SQ_H1);
static_assert(horizontal_flip(SQ_D4) == SQ_E4);
static_assert(royal_key(SQ_D4).bucket == royal_key(SQ_E4).bucket);
static_assert(royal_key(SQ_D4).mirror != royal_key(SQ_E4).mirror);
static_assert(royal_piece_square_index(W_PAWN, SQ_A1, SQ_D4)
              == royal_piece_square_index(W_PAWN, SQ_H1, SQ_E4));
static_assert(royal_piece_square_index(B_KING, SQ_E4, SQ_E4) == InvalidRoyalFeatureIndex);
static_assert(royal_piece_square_index(W_KING, SQ_E1, SQ_E4) == InvalidRoyalFeatureIndex);

}  // namespace Stockfish::Eval::NNUE::HordeV2

#endif  // HORDE_V2_FEATURES_H_INCLUDED
