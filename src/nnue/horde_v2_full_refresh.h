/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_V2_FULL_REFRESH_H_INCLUDED
#define HORDE_V2_FULL_REFRESH_H_INCLUDED

#include <array>
#include <cstddef>

#include "../position.h"
#include "horde_v2_features.h"

namespace Stockfish::Eval::NNUE::HordeV2 {

inline constexpr std::size_t MaxHordePieces      = 52;
inline constexpr std::size_t MaxHordeSidePieces  = 36;
inline constexpr std::size_t MaxRoyalSidePieces  = 16;
inline constexpr std::size_t MaxRoyalInputPieces = MaxHordePieces - 1;

enum class FullRefreshError {
    NONE,
    INVALID_PIECE,
    WHITE_KING,
    BLACK_KING_COUNT,
    TOO_MANY_WHITE_PIECES,
    TOO_MANY_BLACK_PIECES,
};

struct FullRefreshFeatures {
    std::array<IndexType, MaxHordePieces>      global{};
    std::array<IndexType, MaxRoyalInputPieces> royal{};
    std::size_t                                globalSize = 0;
    std::size_t                                royalSize  = 0;
    RoyalKey                                   royalKey{RoyalBucketCount, false};
    FullRefreshError                           error = FullRefreshError::NONE;

    constexpr bool valid() const noexcept { return error == FullRefreshError::NONE; }
};

// Enumerate physical squares in A1-to-H8 order. The order is part of the
// trainer ABI even though sparse affine accumulation is commutative.
constexpr FullRefreshFeatures
extract_full_refresh_features(const std::array<Piece, SQUARE_NB>& board) noexcept {
    FullRefreshFeatures features{};
    Square              blackKingSquare = SQ_NONE;
    std::size_t         whitePieces     = 0;
    std::size_t         blackPieces     = 0;
    std::size_t         blackKings      = 0;

    for (int rawSquare = 0; rawSquare < SQUARE_NB; ++rawSquare)
    {
        const Piece piece = board[rawSquare];
        if (piece == NO_PIECE)
            continue;

        if (piece == W_KING)
        {
            features.error = FullRefreshError::WHITE_KING;
            return features;
        }

        if (!is_registered_piece(piece))
        {
            features.error = FullRefreshError::INVALID_PIECE;
            return features;
        }

        if (color_of(piece) == WHITE)
            ++whitePieces;
        else
            ++blackPieces;

        if (piece == B_KING)
        {
            ++blackKings;
            blackKingSquare = Square(rawSquare);
        }
    }

    if (blackKings != 1)
    {
        features.error = FullRefreshError::BLACK_KING_COUNT;
        return features;
    }
    if (whitePieces > MaxHordeSidePieces)
    {
        features.error = FullRefreshError::TOO_MANY_WHITE_PIECES;
        return features;
    }
    if (blackPieces > MaxRoyalSidePieces)
    {
        features.error = FullRefreshError::TOO_MANY_BLACK_PIECES;
        return features;
    }

    features.royalKey = royal_key(blackKingSquare);

    for (int rawSquare = 0; rawSquare < SQUARE_NB; ++rawSquare)
    {
        const Piece piece = board[rawSquare];
        if (piece == NO_PIECE)
            continue;

        const Square    square      = Square(rawSquare);
        const IndexType globalIndex = fixed_role_piece_square_index(piece, square);
        if (globalIndex == InvalidFeatureIndex)
        {
            features.error = FullRefreshError::INVALID_PIECE;
            return features;
        }
        features.global[features.globalSize++] = globalIndex;

        if (piece == B_KING)
            continue;

        const IndexType royalIndex = royal_piece_square_index(piece, square, features.royalKey);
        if (royalIndex == InvalidRoyalFeatureIndex)
        {
            features.error = FullRefreshError::INVALID_PIECE;
            return features;
        }
        features.royal[features.royalSize++] = royalIndex;
    }

    return features;
}

inline FullRefreshFeatures extract_full_refresh_features(const Position& pos) noexcept {
    FullRefreshFeatures features{};

    if (pos.count<KING>(WHITE) != 0)
    {
        features.error = FullRefreshError::WHITE_KING;
        return features;
    }
    if (pos.count<KING>(BLACK) != 1)
    {
        features.error = FullRefreshError::BLACK_KING_COUNT;
        return features;
    }
    if (pos.count<ALL_PIECES>(WHITE) > int(MaxHordeSidePieces))
    {
        features.error = FullRefreshError::TOO_MANY_WHITE_PIECES;
        return features;
    }
    if (pos.count<ALL_PIECES>(BLACK) > int(MaxRoyalSidePieces))
    {
        features.error = FullRefreshError::TOO_MANY_BLACK_PIECES;
        return features;
    }

    features.royalKey = royal_key(pos.square<KING>(BLACK));

    Bitboard occupied = pos.pieces();
    while (occupied)
    {
        const Square square = pop_lsb(occupied);
        const Piece  piece  = pos.piece_on(square);

        const IndexType globalIndex = fixed_role_piece_square_index(piece, square);
        if (globalIndex == InvalidFeatureIndex)
        {
            features.error = FullRefreshError::INVALID_PIECE;
            return features;
        }
        features.global[features.globalSize++] = globalIndex;

        if (piece == B_KING)
            continue;

        const IndexType royalIndex = royal_piece_square_index(piece, square, features.royalKey);
        if (royalIndex == InvalidRoyalFeatureIndex)
        {
            features.error = FullRefreshError::INVALID_PIECE;
            return features;
        }
        features.royal[features.royalSize++] = royalIndex;
    }

    return features;
}

static_assert(MaxHordePieces == 52);
static_assert(MaxRoyalInputPieces == 51);

}  // namespace Stockfish::Eval::NNUE::HordeV2

#endif  // HORDE_V2_FULL_REFRESH_H_INCLUDED
