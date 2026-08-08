/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_LEGACY_FEATURES_H_INCLUDED
#define HORDE_LEGACY_FEATURES_H_INCLUDED

#include "../types.h"
#include "nnue_common.h"

namespace Stockfish::Eval::NNUE::HordeLegacy {

// Run 6B was trained with seven piece families and two relative-color planes.
// The Horde pawn family is an evaluator identity only: the physical board and
// HORDE_BIN_V1 both continue to store White pawns as ordinary PAWN pieces.
inline constexpr IndexType PieceFamilyCount     = 7;
inline constexpr IndexType PieceSquareDimensions = 2 * PieceFamilyCount * SQUARE_NB;
inline constexpr IndexType InvalidFeatureIndex   = PieceSquareDimensions;

constexpr int feature_family(Piece piece) noexcept {
    const PieceType type = type_of(piece);
    if (type == PAWN)
        return color_of(piece) == WHITE ? 5 : 0;

    switch (type)
    {
    case KNIGHT :
        return 1;
    case BISHOP :
        return 2;
    case ROOK :
        return 3;
    case QUEEN :
        return 4;
    case KING :
        return 6;
    default :
        return -1;
    }
}

constexpr IndexType feature_index(Color perspective, Square square, Piece piece) noexcept {
    const int family = feature_family(piece);
    if ((perspective != WHITE && perspective != BLACK) || !is_ok(square) || family < 0)
        return InvalidFeatureIndex;

    const IndexType plane = 2 * IndexType(family) + (color_of(piece) != perspective);
    return plane * SQUARE_NB + IndexType(relative_square(perspective, square));
}

static_assert(PieceSquareDimensions == 896);
static_assert(feature_index(WHITE, SQ_A1, W_PAWN) == 10 * 64);
static_assert(feature_index(BLACK, SQ_A1, W_PAWN) == 11 * 64 + SQ_A8);
static_assert(feature_index(WHITE, SQ_A7, B_PAWN) == 1 * 64 + SQ_A7);
static_assert(feature_index(BLACK, SQ_A7, B_PAWN) == SQ_A2);
static_assert(feature_index(BLACK, SQ_E8, B_KING) == 12 * 64 + SQ_E1);
static_assert(feature_index(WHITE, SQ_NONE, W_PAWN) == InvalidFeatureIndex);
static_assert(feature_index(WHITE, SQ_A1, NO_PIECE) == InvalidFeatureIndex);

}  // namespace Stockfish::Eval::NNUE::HordeLegacy

#endif  // HORDE_LEGACY_FEATURES_H_INCLUDED
