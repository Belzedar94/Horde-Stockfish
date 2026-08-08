/*
  Exhaustive contract checks for the experimental fixed-role Horde V2 index.
*/

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <random>

#include "../src/nnue/horde_v2_features.h"
#include "../src/nnue/horde_v2_full_refresh.h"
#include "../src/nnue/horde_v2_scalar.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE::HordeV2;

namespace {

std::array<Piece, SQUARE_NB> horde_start_board() {
    std::array<Piece, SQUARE_NB> board{};

    for (int rawSquare = SQ_A1; rawSquare <= SQ_H4; ++rawSquare)
        board[rawSquare] = W_PAWN;
    for (const Square square : {SQ_B5, SQ_C5, SQ_F5, SQ_G5})
        board[square] = W_PAWN;

    for (int rawSquare = SQ_A7; rawSquare <= SQ_H7; ++rawSquare)
        board[rawSquare] = B_PAWN;

    board[SQ_A8] = B_ROOK;
    board[SQ_B8] = B_KNIGHT;
    board[SQ_C8] = B_BISHOP;
    board[SQ_D8] = B_QUEEN;
    board[SQ_E8] = B_KING;
    board[SQ_F8] = B_BISHOP;
    board[SQ_G8] = B_KNIGHT;
    board[SQ_H8] = B_ROOK;
    return board;
}

std::array<Piece, SQUARE_NB>
horizontal_reflection(const std::array<Piece, SQUARE_NB>& board) {
    std::array<Piece, SQUARE_NB> reflected{};
    for (int rawSquare = 0; rawSquare < SQUARE_NB; ++rawSquare)
        reflected[horizontal_flip(Square(rawSquare))] = board[rawSquare];
    return reflected;
}

template<std::size_t Capacity>
std::array<Eval::NNUE::IndexType, Capacity>
sorted_prefix(std::array<Eval::NNUE::IndexType, Capacity> values, std::size_t size) {
    std::sort(values.begin(), values.begin() + size);
    return values;
}

}  // namespace

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

    const auto startBoard    = horde_start_board();
    const auto startFeatures = extract_full_refresh_features(startBoard);
    assert(startFeatures.valid());
    assert(startFeatures.globalSize == MaxHordePieces);
    assert(startFeatures.royalSize == MaxRoyalInputPieces);
    assert(startFeatures.royalKey == royal_key(SQ_E8));
    assert(startFeatures.global.front()
           == fixed_role_piece_square_index(W_PAWN, SQ_A1));
    assert(startFeatures.global[startFeatures.globalSize - 1]
           == fixed_role_piece_square_index(B_ROOK, SQ_H8));

    // The Global domain remains absolute, while reflecting the complete board
    // must preserve the multiset of Royal rows.
    const auto reflectedFeatures =
      extract_full_refresh_features(horizontal_reflection(startBoard));
    assert(reflectedFeatures.valid());
    assert(reflectedFeatures.royalKey.bucket == startFeatures.royalKey.bucket);
    assert(reflectedFeatures.royalKey.mirror != startFeatures.royalKey.mirror);
    assert(sorted_prefix(reflectedFeatures.royal, reflectedFeatures.royalSize)
           == sorted_prefix(startFeatures.royal, startFeatures.royalSize));
    assert(sorted_prefix(reflectedFeatures.global, reflectedFeatures.globalSize)
           != sorted_prefix(startFeatures.global, startFeatures.globalSize));

    auto promotedBoard       = startBoard;
    promotedBoard[SQ_A1]     = W_QUEEN;
    const auto promotedFeatures = extract_full_refresh_features(promotedBoard);
    assert(promotedFeatures.valid());
    assert(promotedFeatures.globalSize == MaxHordePieces);
    assert(promotedFeatures.royalSize == MaxRoyalInputPieces);
    assert(promotedFeatures.global.front()
           == fixed_role_piece_square_index(W_QUEEN, SQ_A1));

    auto invalidBoard       = startBoard;
    invalidBoard[SQ_A1]     = W_KING;
    assert(extract_full_refresh_features(invalidBoard).error == FullRefreshError::WHITE_KING);

    invalidBoard            = startBoard;
    invalidBoard[SQ_E8]     = NO_PIECE;
    assert(extract_full_refresh_features(invalidBoard).error
           == FullRefreshError::BLACK_KING_COUNT);

    invalidBoard            = startBoard;
    invalidBoard[SQ_A6]     = B_KING;
    assert(extract_full_refresh_features(invalidBoard).error
           == FullRefreshError::BLACK_KING_COUNT);

    invalidBoard            = startBoard;
    invalidBoard[SQ_A6]     = W_PAWN;
    assert(extract_full_refresh_features(invalidBoard).error
           == FullRefreshError::TOO_MANY_WHITE_PIECES);

    invalidBoard            = startBoard;
    invalidBoard[SQ_A6]     = B_QUEEN;
    assert(extract_full_refresh_features(invalidBoard).error
           == FullRefreshError::TOO_MANY_BLACK_PIECES);

    invalidBoard            = startBoard;
    invalidBoard[SQ_A1]     = Piece(7);
    assert(extract_full_refresh_features(invalidBoard).error
           == FullRefreshError::INVALID_PIECE);

    constexpr std::array<Piece, 5> HordeRoles = {
      W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN};
    constexpr std::array<Piece, 5> RoyalRoles = {
      B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN};
    std::mt19937                     rng(0x48563230);
    std::uniform_int_distribution<> whiteCountDistribution(0, MaxHordeSidePieces);
    std::uniform_int_distribution<> blackCountDistribution(0, MaxRoyalSidePieces - 1);
    std::uniform_int_distribution<> roleDistribution(0, 4);

    for (int sample = 0; sample < 10000; ++sample)
    {
        std::array<Piece, SQUARE_NB> randomBoard{};
        std::array<int, SQUARE_NB>   squares{};
        std::iota(squares.begin(), squares.end(), 0);
        std::shuffle(squares.begin(), squares.end(), rng);

        const int whiteCount   = whiteCountDistribution(rng);
        const int blackNonKing = blackCountDistribution(rng);
        int       cursor       = 0;
        randomBoard[squares[cursor++]] = B_KING;
        for (int i = 0; i < whiteCount; ++i)
            randomBoard[squares[cursor++]] = HordeRoles[roleDistribution(rng)];
        for (int i = 0; i < blackNonKing; ++i)
            randomBoard[squares[cursor++]] = RoyalRoles[roleDistribution(rng)];

        const auto randomFeatures = extract_full_refresh_features(randomBoard);
        assert(randomFeatures.valid());
        assert(randomFeatures.globalSize == std::size_t(whiteCount + blackNonKing + 1));
        assert(randomFeatures.royalSize == std::size_t(whiteCount + blackNonKing));
        assert(std::all_of(randomFeatures.global.begin(),
                           randomFeatures.global.begin() + randomFeatures.globalSize,
                           [](const auto index) {
                               return index < FixedRolePieceSquareDimensions;
                           }));
        assert(std::all_of(randomFeatures.royal.begin(),
                           randomFeatures.royal.begin() + randomFeatures.royalSize,
                           [](const auto index) { return index < RoyalPieceSquareDimensions; }));

        const auto sortedGlobal =
          sorted_prefix(randomFeatures.global, randomFeatures.globalSize);
        const auto sortedRoyal = sorted_prefix(randomFeatures.royal, randomFeatures.royalSize);
        assert(std::adjacent_find(sortedGlobal.begin(),
                                  sortedGlobal.begin() + randomFeatures.globalSize)
               == sortedGlobal.begin() + randomFeatures.globalSize);
        assert(std::adjacent_find(sortedRoyal.begin(),
                                  sortedRoyal.begin() + randomFeatures.royalSize)
               == sortedRoyal.begin() + randomFeatures.royalSize);

        const auto randomReflected =
          extract_full_refresh_features(horizontal_reflection(randomBoard));
        assert(randomReflected.valid());
        assert(randomReflected.royalSize == randomFeatures.royalSize);
        assert(sorted_prefix(randomReflected.royal, randomReflected.royalSize) == sortedRoyal);
    }

    // Exercise the exact V2_BASE_P0 integer path with a non-zero deterministic
    // payload. Both STM heads share every preceding layer.
    ScalarNetwork deterministicNetwork(make_deterministic_parameters(0x4856325F42415345ULL));
    const auto    whiteTrace = deterministicNetwork.evaluate_full_refresh(startBoard, WHITE, 0);
    const auto    blackTrace = deterministicNetwork.evaluate_full_refresh(startBoard, BLACK, 0);
    assert(whiteTrace.valid());
    assert(blackTrace.valid());
    assert(whiteTrace.royalAccumulator == blackTrace.royalAccumulator);
    assert(whiteTrace.globalAccumulator == blackTrace.globalAccumulator);
    assert(whiteTrace.transformed == blackTrace.transformed);
    assert(whiteTrace.hidden0Affine == blackTrace.hidden0Affine);
    assert(whiteTrace.hidden0 == blackTrace.hidden0);
    assert(whiteTrace.hidden1Affine == blackTrace.hidden1Affine);
    assert(whiteTrace.hidden1 == blackTrace.hidden1);
    assert(whiteTrace.outputAffine != blackTrace.outputAffine);
    assert(whiteTrace.preRule50Value == 183);
    assert(blackTrace.preRule50Value == 130);

    // Recompute selected FT lanes independently from the exposed parameter
    // rows. This is the layer-by-layer trainer ABI receipt.
    const auto& deterministicParameters = deterministicNetwork.parameters();
    for (const Eval::NNUE::IndexType lane :
         {Eval::NNUE::IndexType(0), Eval::NNUE::IndexType(17), RoyalLanes - 1})
    {
        Accumulator expected = deterministicParameters.royalBias[lane];
        for (std::size_t active = 0; active < startFeatures.royalSize; ++active)
            expected += deterministicParameters.royalWeights[
              std::size_t(startFeatures.royal[active]) * RoyalLanes + lane];
        assert(whiteTrace.royalAccumulator[lane] == expected);
    }
    for (const Eval::NNUE::IndexType lane :
         {Eval::NNUE::IndexType(0), Eval::NNUE::IndexType(19), GlobalLanes - 1})
    {
        Accumulator expected = deterministicParameters.globalBias[lane];
        for (std::size_t active = 0; active < startFeatures.globalSize; ++active)
            expected += deterministicParameters.globalWeights[
              std::size_t(startFeatures.global[active]) * GlobalLanes + lane];
        assert(whiteTrace.globalAccumulator[lane] == expected);
    }

    // Royal canonicalization is invariant under complete horizontal
    // reflection. Global is deliberately absolute and therefore is not.
    const auto reflectedTrace =
      deterministicNetwork.evaluate_full_refresh(horizontal_reflection(startBoard), WHITE, 0);
    assert(reflectedTrace.valid());
    assert(reflectedTrace.royalAccumulator == whiteTrace.royalAccumulator);
    assert(reflectedTrace.globalAccumulator != whiteTrace.globalAccumulator);

    const auto rule50Half = deterministicNetwork.evaluate_full_refresh(startBoard, WHITE, 50);
    const auto rule50Full = deterministicNetwork.evaluate_full_refresh(startBoard, WHITE, 100);
    assert(rule50Half.valid());
    assert(rule50Full.valid());
    assert(rule50Half.value == apply_rule50_postprocessor(whiteTrace.preRule50Value, 50));
    assert(rule50Full.value == VALUE_ZERO);

    // Exact clipping and negative truncation checks use an otherwise zero
    // payload, keeping every expected intermediate value transparent.
    ScalarParameters clippingParameters;
    clippingParameters.royalBias[0]  = -1;
    clippingParameters.royalBias[1]  = 64;
    clippingParameters.royalBias[2]  = 64 * 127;
    clippingParameters.royalBias[3]  = 64 * 128;
    clippingParameters.outputBias[WHITE] = -511;
    clippingParameters.outputBias[BLACK] = 511;
    ScalarNetwork clippingNetwork(std::move(clippingParameters));
    const auto    clippingWhite = clippingNetwork.evaluate_full_refresh(startBoard, WHITE, 50);
    const auto    clippingBlack = clippingNetwork.evaluate_full_refresh(startBoard, BLACK, 50);
    assert(clippingWhite.valid());
    assert(clippingBlack.valid());
    assert(clippingWhite.transformed[0] == 0);
    assert(clippingWhite.transformed[1] == 1);
    assert(clippingWhite.transformed[2] == 127);
    assert(clippingWhite.transformed[3] == 127);
    assert(clippingWhite.preRule50Value == -31);
    assert(clippingWhite.value == Value(-15));
    assert(clippingBlack.preRule50Value == 31);
    assert(clippingBlack.value == Value(15));

    ScalarParameters invalidParameters;
    invalidParameters.royalWeights.pop_back();
    ScalarNetwork invalidNetwork(std::move(invalidParameters));
    assert(invalidNetwork.evaluate_full_refresh(startBoard, WHITE, 0).error
           == ScalarEvalError::INVALID_PARAMETERS);

    auto scalarInvalidBoard   = startBoard;
    scalarInvalidBoard[SQ_E8] = NO_PIECE;
    const auto invalidPositionTrace =
      deterministicNetwork.evaluate_full_refresh(scalarInvalidBoard, WHITE, 0);
    assert(invalidPositionTrace.error == ScalarEvalError::INVALID_POSITION);
    assert(invalidPositionTrace.featureError == FullRefreshError::BLACK_KING_COUNT);
    assert(deterministicNetwork.evaluate_full_refresh(startBoard, Color(COLOR_NB), 0).error
           == ScalarEvalError::INVALID_SIDE_TO_MOVE);

    std::cout << "Horde V2 feature contracts passed: "
              << FixedRolePieceSquareDimensions << " global and "
              << RoyalPieceSquareDimensions << " Royal indices; full-refresh "
              << startFeatures.globalSize << "+" << startFeatures.royalSize
              << " active rows; scalar P0=" << whiteTrace.preRule50Value << "/"
              << blackTrace.preRule50Value << "\n";
}
