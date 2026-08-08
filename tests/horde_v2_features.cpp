/*
  Exhaustive contract checks for the experimental fixed-role Horde V2 index.
*/

#include <array>
#include <cassert>
#include <iostream>

#include "../src/nnue/horde_v2_features.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE::HordeV2;

int main() {
    constexpr std::array<Piece, FIXED_ROLE_NB> FixedRolePieces = {
      W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN,
      B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN, B_KING};
    constexpr std::array<Piece, RoyalNonKingRoleCount> RoyalPieces = {
      W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN,
      B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN};

    std::array<bool, FixedRolePieceSquareDimensions> seen{};
    for (const Piece piece : FixedRolePieces)
        for (int rawSquare = 0; rawSquare < SQUARE_NB; ++rawSquare)
        {
            const auto index = fixed_role_piece_square_index(piece, Square(rawSquare));
            assert(index < FixedRolePieceSquareDimensions);
            assert(!seen[index]);
            seen[index] = true;
        }

    for (const bool present : seen)
        assert(present);

    assert(!is_registered_piece(W_KING));
    assert(!is_registered_piece(NO_PIECE));
    assert(fixed_role_piece_square_index(W_KING, SQ_E1) == InvalidFeatureIndex);
    assert(fixed_role_piece_square_index(B_KING, SQ_E8) < FixedRolePieceSquareDimensions);

    // The canonical half-board (files e-h) visits every Royal bucket exactly
    // once, so it must cover the complete table without collisions.
    std::array<bool, RoyalPieceSquareDimensions> royalSeen{};
    for (int rawKingSquare = 0; rawKingSquare < SQUARE_NB; ++rawKingSquare)
    {
        const Square kingSquare = Square(rawKingSquare);
        const auto   key        = royal_key(kingSquare);

        assert(is_valid_royal_key(key));
        assert(key.mirror == (file_of(kingSquare) <= FILE_D));
        assert(file_of(royal_orient(kingSquare, key.mirror)) >= FILE_E);

        if (file_of(kingSquare) < FILE_E)
            continue;

        for (const Piece piece : RoyalPieces)
            for (int rawSquare = 0; rawSquare < SQUARE_NB; ++rawSquare)
            {
                const auto index =
                  royal_piece_square_index(piece, Square(rawSquare), kingSquare);
                assert(index < RoyalPieceSquareDimensions);
                assert(!royalSeen[index]);
                royalSeen[index] = true;
            }
    }

    for (const bool present : royalSeen)
        assert(present);

    // Reflecting both the king and the piece preserves the Royal row.
    for (int rawKingSquare = 0; rawKingSquare < SQUARE_NB; ++rawKingSquare)
        for (const Piece piece : RoyalPieces)
            for (int rawSquare = 0; rawSquare < SQUARE_NB; ++rawSquare)
            {
                const Square kingSquare = Square(rawKingSquare);
                const Square square     = Square(rawSquare);
                assert(royal_piece_square_index(piece, square, kingSquare)
                       == royal_piece_square_index(
                         piece, horizontal_flip(square), horizontal_flip(kingSquare)));
            }

    const RoyalKey d4Key = royal_key(SQ_D4);
    const RoyalKey e4Key = royal_key(SQ_E4);
    assert(d4Key.bucket == e4Key.bucket);
    assert(d4Key != e4Key);
    assert(royal_piece_square_index(W_PAWN, SQ_A1, d4Key)
           == royal_piece_square_index(W_PAWN, SQ_H1, e4Key));

    assert(!is_valid_royal_key(royal_key(SQ_NONE)));
    assert(royal_piece_square_index(W_KING, SQ_E1, SQ_E8) == InvalidRoyalFeatureIndex);
    assert(royal_piece_square_index(B_KING, SQ_E8, SQ_E8) == InvalidRoyalFeatureIndex);
    assert(royal_piece_square_index(NO_PIECE, SQ_E4, SQ_E8) == InvalidRoyalFeatureIndex);
    assert(royal_piece_square_index(W_PAWN, SQ_NONE, SQ_E8) == InvalidRoyalFeatureIndex);
    assert(royal_piece_square_index(W_PAWN, SQ_E4, SQ_NONE) == InvalidRoyalFeatureIndex);

    std::cout << "Horde V2 feature contracts passed: "
              << FixedRolePieceSquareDimensions << " global and "
              << RoyalPieceSquareDimensions << " Royal indices\n";
}
