/*
  Exhaustive contract checks for the experimental fixed-role Horde V2 index.
*/

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <numeric>
#include <random>
#include <string_view>

#include "../src/nnue/horde_legacy_features.h"
#include "../src/nnue/horde_v2_features.h"
#include "../src/nnue/horde_v2_full_refresh.h"
#include "../src/nnue/horde_v2_scalar.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE::HordeV2;

namespace {

constexpr u64 ScalarFixtureSeed = 0x4856325F42415345ULL;

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

std::array<Piece, SQUARE_NB> incremental_base_board() {
    std::array<Piece, SQUARE_NB> board{};
    board[SQ_E8] = B_KING;
    board[SQ_A8] = B_ROOK;
    board[SQ_H8] = B_ROOK;
    board[SQ_C4] = B_PAWN;
    board[SQ_F5] = B_PAWN;
    board[SQ_A2] = W_PAWN;
    board[SQ_B7] = W_PAWN;
    board[SQ_D4] = W_ROOK;
    board[SQ_E5] = W_PAWN;
    return board;
}

std::array<Piece, SQUARE_NB> horizontal_reflection(const std::array<Piece, SQUARE_NB>& board) {
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

template<typename T, std::size_t Size>
void emit_json_array(const char* name, const std::array<T, Size>& values, bool leadingComma) {
    std::cout << (leadingComma ? ",\"" : "\"") << name << "\":[";
    for (std::size_t index = 0; index < Size; ++index)
    {
        if (index != 0)
            std::cout << ',';
        std::cout << +values[index];
    }
    std::cout << ']';
}

template<typename T, std::size_t Size>
void emit_json_prefix(const char* name,
                      const std::array<T, Size>& values,
                      std::size_t                size,
                      bool                       leadingComma) {
    assert(size <= Size);
    std::cout << (leadingComma ? ",\"" : "\"") << name << "\":[";
    for (std::size_t index = 0; index < size; ++index)
    {
        if (index != 0)
            std::cout << ',';
        std::cout << +values[index];
    }
    std::cout << ']';
}

u8 physical_piece_code(Piece piece) {
    switch (piece)
    {
    case NO_PIECE :
        return 0;
    case W_PAWN :
        return 1;
    case W_KNIGHT :
        return 2;
    case W_BISHOP :
        return 3;
    case W_ROOK :
        return 4;
    case W_QUEEN :
        return 5;
    case B_PAWN :
        return 6;
    case B_KNIGHT :
        return 7;
    case B_BISHOP :
        return 8;
    case B_ROOK :
        return 9;
    case B_QUEEN :
        return 10;
    case B_KING :
        return 11;
    default :
        assert(false);
        return 0xFF;
    }
}

std::array<Piece, SQUARE_NB> promotion_fixture() {
    std::array<Piece, SQUARE_NB> board{};
    board[SQ_A8] = B_KING;
    board[SQ_D4] = B_QUEEN;
    board[SQ_G7] = B_PAWN;
    board[SQ_A2] = W_PAWN;
    board[SQ_B3] = W_KNIGHT;
    board[SQ_C4] = W_BISHOP;
    board[SQ_D5] = W_ROOK;
    board[SQ_E6] = W_QUEEN;
    return board;
}

std::array<Piece, SQUARE_NB> low_material_fixture() {
    std::array<Piece, SQUARE_NB> board{};
    board[SQ_H1] = B_KING;
    board[SQ_F2] = B_ROOK;
    board[SQ_A7] = W_PAWN;
    board[SQ_C6] = W_QUEEN;
    return board;
}

void emit_sparse_position(const std::array<Piece, SQUARE_NB>& board) {
    const auto features = extract_full_refresh_features(board);
    assert(features.valid());

    std::array<Eval::NNUE::IndexType, MaxHordePieces> legacyWhite{};
    std::array<Eval::NNUE::IndexType, MaxHordePieces> legacyBlack{};
    std::size_t                                        legacySize = 0;
    for (int rawSquare = 0; rawSquare < SQUARE_NB; ++rawSquare)
    {
        const Square square = Square(rawSquare);
        const Piece  piece  = board[square];
        if (piece == NO_PIECE)
            continue;
        legacyWhite[legacySize] = Eval::NNUE::HordeLegacy::feature_index(WHITE, square, piece);
        legacyBlack[legacySize] = Eval::NNUE::HordeLegacy::feature_index(BLACK, square, piece);
        assert(legacyWhite[legacySize] < Eval::NNUE::HordeLegacy::PieceSquareDimensions);
        assert(legacyBlack[legacySize] < Eval::NNUE::HordeLegacy::PieceSquareDimensions);
        ++legacySize;
    }

    std::cout << "{\"board\":[";
    for (int rawSquare = 0; rawSquare < SQUARE_NB; ++rawSquare)
    {
        if (rawSquare != 0)
            std::cout << ',';
        std::cout << +physical_piece_code(board[rawSquare]);
    }
    std::cout << ']';
    emit_json_prefix("legacy_white", legacyWhite, legacySize, true);
    emit_json_prefix("legacy_black", legacyBlack, legacySize, true);
    emit_json_prefix("v2_global", features.global, features.globalSize, true);
    emit_json_prefix("v2_royal", features.royal, features.royalSize, true);
    std::cout << ",\"royal_bucket\":" << features.royalKey.bucket;
    std::cout << ",\"royal_mirror\":" << (features.royalKey.mirror ? "true" : "false");
    std::cout << '}';
}

int emit_sparse_index_receipt() {
    const std::array<std::array<Piece, SQUARE_NB>, 5> boards = {
      horde_start_board(), horizontal_reflection(horde_start_board()), incremental_base_board(),
      promotion_fixture(), low_material_fixture()};

    std::cout << "{\"schema\":\"HORDE_SPARSE_INDEX_RECEIPT_V1\",\"positions\":[";
    for (std::size_t index = 0; index < boards.size(); ++index)
    {
        if (index != 0)
            std::cout << ',';
        emit_sparse_position(boards[index]);
    }
    std::cout << "]}\n";
    return 0;
}

int emit_scalar_receipt() {
    ScalarNetwork network(make_deterministic_parameters(ScalarFixtureSeed));
    const auto    board = horde_start_board();
    const auto    white = network.evaluate_full_refresh(board, WHITE, 0);
    const auto    black = network.evaluate_full_refresh(board, BLACK, 0);
    assert(white.valid());
    assert(black.valid());

    std::cout << '{';
    std::cout << "\"seed\":" << ScalarFixtureSeed;
    std::cout << ",\"parameter_bytes\":" << ScalarParameterBytes;
    emit_json_array("royal_accumulator", white.royalAccumulator, true);
    emit_json_array("global_accumulator", white.globalAccumulator, true);
    emit_json_array("transformed", white.transformed, true);
    emit_json_array("hidden0_affine", white.hidden0Affine, true);
    emit_json_array("hidden0", white.hidden0, true);
    emit_json_array("hidden1_affine", white.hidden1Affine, true);
    emit_json_array("hidden1", white.hidden1, true);
    std::cout << ",\"white_output_affine\":" << white.outputAffine;
    std::cout << ",\"black_output_affine\":" << black.outputAffine;
    std::cout << ",\"white_pre_rule50\":" << white.preRule50Value;
    std::cout << ",\"black_pre_rule50\":" << black.preRule50Value;
    std::cout << ",\"white_value\":" << int(white.value);
    std::cout << ",\"black_value\":" << int(black.value);
    std::cout << "}\n";
    return 0;
}

DirtyPiece make_dirty(Piece  piece,
                      Square from,
                      Square to,
                      Piece  removePiece  = NO_PIECE,
                      Square removeSquare = SQ_NONE,
                      Piece  addPiece     = NO_PIECE,
                      Square addSquare    = SQ_NONE) {
    DirtyPiece dirty{};
    dirty.pc        = piece;
    dirty.from      = from;
    dirty.to        = to;
    dirty.remove_sq = removeSquare;
    dirty.add_sq    = addSquare;
    dirty.remove_pc = removePiece;
    dirty.add_pc    = addPiece;
    return dirty;
}

void assert_same_evaluation(const ScalarTrace& actual, const ScalarTrace& expected) {
    assert(actual.valid());
    assert(expected.valid());
    assert(actual.royalAccumulator == expected.royalAccumulator);
    assert(actual.globalAccumulator == expected.globalAccumulator);
    assert(actual.transformed == expected.transformed);
    assert(actual.hidden0Affine == expected.hidden0Affine);
    assert(actual.hidden0 == expected.hidden0);
    assert(actual.hidden1Affine == expected.hidden1Affine);
    assert(actual.hidden1 == expected.hidden1);
    assert(actual.outputAffine == expected.outputAffine);
    assert(actual.preRule50Value == expected.preRule50Value);
    assert(actual.value == expected.value);
    assert(actual.royalKey == expected.royalKey);
}

void assert_incremental_transition(ScalarNetwork&                      network,
                                   const std::array<Piece, SQUARE_NB>& sourceBoard,
                                   const std::array<Piece, SQUARE_NB>& targetBoard,
                                   const DirtyPiece&                   dirty,
                                   Color                               targetSideToMove,
                                   int                                 targetRule50,
                                   bool                                expectRoyalRefresh) {
    const ScalarTrace source = network.evaluate_full_refresh(sourceBoard, ~targetSideToMove, 0);
    const ScalarTrace expected =
      network.evaluate_full_refresh(targetBoard, targetSideToMove, targetRule50);
    const ScalarTrace actual =
      network.evaluate_incremental(dirty, targetBoard, source, targetSideToMove, targetRule50);

    assert_same_evaluation(actual, expected);
    assert(actual.royalRefreshed == expectRoyalRefresh);

    // Undo restores the saved source frame; the transition must never mutate it.
    const ScalarTrace refreshedSource =
      network.evaluate_full_refresh(sourceBoard, ~targetSideToMove, 0);
    assert_same_evaluation(source, refreshedSource);
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc == 2 && std::string_view(argv[1]) == "--scalar-receipt")
        return emit_scalar_receipt();
    if (argc == 2 && std::string_view(argv[1]) == "--sparse-index-receipt")
        return emit_sparse_index_receipt();

    constexpr std::array<Piece, FIXED_ROLE_NB> FixedRolePieces = {
      W_PAWN,   W_KNIGHT, W_BISHOP, W_ROOK,  W_QUEEN, B_PAWN,
      B_KNIGHT, B_BISHOP, B_ROOK,   B_QUEEN, B_KING};
    constexpr std::array<Piece, RoyalNonKingRoleCount> RoyalPieces = {
      W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN, B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN};

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
                const auto index = royal_piece_square_index(piece, Square(rawSquare), kingSquare);
                assert(index < RoyalPieceSquareDimensions);
                assert(!royalSeen[index]);
                royalSeen[index] = true;
            }
    }

    for (const bool present : royalSeen)
        assert(present);

    // R8 keeps the same mirror orientation while sharing all four canonical
    // king files within one rank bucket.
    std::array<bool, RoyalRankPieceSquareDimensions> rank8Seen{};
    for (int rawRank = 0; rawRank < RANK_NB; ++rawRank)
    {
        const Square kingSquare = make_square(FILE_E, Rank(rawRank));
        const auto   key        = royal_rank_key(kingSquare);
        assert(is_valid_royal_rank_key(key));
        assert(key.bucket == u32(rawRank));
        assert(!key.mirror);

        for (const Piece piece : RoyalPieces)
            for (int rawSquare = 0; rawSquare < SQUARE_NB; ++rawSquare)
            {
                const auto index = royal_rank_piece_square_index(
                  piece, Square(rawSquare), key);
                assert(index < RoyalRankPieceSquareDimensions);
                assert(!rank8Seen[index]);
                rank8Seen[index] = true;
            }
    }
    for (const bool present : rank8Seen)
        assert(present);

    // Reflecting both the king and the piece preserves the Royal row.
    for (int rawKingSquare = 0; rawKingSquare < SQUARE_NB; ++rawKingSquare)
        for (const Piece piece : RoyalPieces)
            for (int rawSquare = 0; rawSquare < SQUARE_NB; ++rawSquare)
            {
                const Square kingSquare = Square(rawKingSquare);
                const Square square     = Square(rawSquare);
                assert(royal_piece_square_index(piece, square, kingSquare)
                       == royal_piece_square_index(piece, horizontal_flip(square),
                                                   horizontal_flip(kingSquare)));
            }

    const RoyalKey d4Key = royal_key(SQ_D4);
    const RoyalKey e4Key = royal_key(SQ_E4);
    assert(d4Key.bucket == e4Key.bucket);
    assert(d4Key != e4Key);
    assert(royal_piece_square_index(W_PAWN, SQ_A1, d4Key)
           == royal_piece_square_index(W_PAWN, SQ_H1, e4Key));

    const RoyalKey e4Rank8Key = royal_rank_key(SQ_E4);
    assert(e4Rank8Key == royal_rank_key(SQ_F4));
    assert(e4Rank8Key != royal_rank_key(SQ_E5));
    assert(royal_rank_key(SQ_D4).bucket == e4Rank8Key.bucket);
    assert(royal_rank_key(SQ_D4).mirror != e4Rank8Key.mirror);
    assert(royal_rank_piece_square_index(W_PAWN, SQ_A1, royal_rank_key(SQ_D4))
           == royal_rank_piece_square_index(W_PAWN, SQ_H1, e4Rank8Key));
    assert(royal_rank_index_from_royal(
             royal_piece_square_index(W_PAWN, SQ_A1, d4Key))
           == royal_rank_piece_square_index(W_PAWN, SQ_A1, royal_rank_key(SQ_D4)));
    assert(royal_rank_index_from_royal(InvalidRoyalFeatureIndex)
           == InvalidRoyalRankFeatureIndex);

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
    assert(startFeatures.global.front() == fixed_role_piece_square_index(W_PAWN, SQ_A1));
    assert(startFeatures.global[startFeatures.globalSize - 1]
           == fixed_role_piece_square_index(B_ROOK, SQ_H8));

    // The Global domain remains absolute, while reflecting the complete board
    // must preserve the multiset of Royal rows.
    const auto reflectedFeatures = extract_full_refresh_features(horizontal_reflection(startBoard));
    assert(reflectedFeatures.valid());
    assert(reflectedFeatures.royalKey.bucket == startFeatures.royalKey.bucket);
    assert(reflectedFeatures.royalKey.mirror != startFeatures.royalKey.mirror);
    assert(sorted_prefix(reflectedFeatures.royal, reflectedFeatures.royalSize)
           == sorted_prefix(startFeatures.royal, startFeatures.royalSize));
    assert(sorted_prefix(reflectedFeatures.global, reflectedFeatures.globalSize)
           != sorted_prefix(startFeatures.global, startFeatures.globalSize));

    auto promotedBoard          = startBoard;
    promotedBoard[SQ_A1]        = W_QUEEN;
    const auto promotedFeatures = extract_full_refresh_features(promotedBoard);
    assert(promotedFeatures.valid());
    assert(promotedFeatures.globalSize == MaxHordePieces);
    assert(promotedFeatures.royalSize == MaxRoyalInputPieces);
    assert(promotedFeatures.global.front() == fixed_role_piece_square_index(W_QUEEN, SQ_A1));

    auto invalidBoard   = startBoard;
    invalidBoard[SQ_A1] = W_KING;
    assert(extract_full_refresh_features(invalidBoard).error == FullRefreshError::WHITE_KING);

    invalidBoard        = startBoard;
    invalidBoard[SQ_E8] = NO_PIECE;
    assert(extract_full_refresh_features(invalidBoard).error == FullRefreshError::BLACK_KING_COUNT);

    invalidBoard        = startBoard;
    invalidBoard[SQ_A6] = B_KING;
    assert(extract_full_refresh_features(invalidBoard).error == FullRefreshError::BLACK_KING_COUNT);

    invalidBoard        = startBoard;
    invalidBoard[SQ_A6] = W_PAWN;
    assert(extract_full_refresh_features(invalidBoard).error
           == FullRefreshError::TOO_MANY_WHITE_PIECES);

    invalidBoard        = startBoard;
    invalidBoard[SQ_A6] = B_QUEEN;
    assert(extract_full_refresh_features(invalidBoard).error
           == FullRefreshError::TOO_MANY_BLACK_PIECES);

    invalidBoard        = startBoard;
    invalidBoard[SQ_A1] = Piece(7);
    assert(extract_full_refresh_features(invalidBoard).error == FullRefreshError::INVALID_PIECE);

    constexpr std::array<Piece, 5>  HordeRoles = {W_PAWN, W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN};
    constexpr std::array<Piece, 5>  RoyalRoles = {B_PAWN, B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN};
    std::mt19937                    rng(0x48563230);
    std::uniform_int_distribution<> whiteCountDistribution(0, MaxHordeSidePieces);
    std::uniform_int_distribution<> blackCountDistribution(0, MaxRoyalSidePieces - 1);
    std::uniform_int_distribution<> roleDistribution(0, 4);

    for (int sample = 0; sample < 10000; ++sample)
    {
        std::array<Piece, SQUARE_NB> randomBoard{};
        std::array<int, SQUARE_NB>   squares{};
        std::iota(squares.begin(), squares.end(), 0);
        std::shuffle(squares.begin(), squares.end(), rng);

        const int whiteCount           = whiteCountDistribution(rng);
        const int blackNonKing         = blackCountDistribution(rng);
        int       cursor               = 0;
        randomBoard[squares[cursor++]] = B_KING;
        for (int i = 0; i < whiteCount; ++i)
            randomBoard[squares[cursor++]] = HordeRoles[roleDistribution(rng)];
        for (int i = 0; i < blackNonKing; ++i)
            randomBoard[squares[cursor++]] = RoyalRoles[roleDistribution(rng)];

        const auto randomFeatures = extract_full_refresh_features(randomBoard);
        assert(randomFeatures.valid());
        assert(randomFeatures.globalSize == std::size_t(whiteCount + blackNonKing + 1));
        assert(randomFeatures.royalSize == std::size_t(whiteCount + blackNonKing));
        assert(std::all_of(
          randomFeatures.global.begin(), randomFeatures.global.begin() + randomFeatures.globalSize,
          [](const auto index) { return index < FixedRolePieceSquareDimensions; }));
        assert(std::all_of(randomFeatures.royal.begin(),
                           randomFeatures.royal.begin() + randomFeatures.royalSize,
                           [](const auto index) { return index < RoyalPieceSquareDimensions; }));

        const auto sortedGlobal = sorted_prefix(randomFeatures.global, randomFeatures.globalSize);
        const auto sortedRoyal  = sorted_prefix(randomFeatures.royal, randomFeatures.royalSize);
        assert(
          std::adjacent_find(sortedGlobal.begin(), sortedGlobal.begin() + randomFeatures.globalSize)
          == sortedGlobal.begin() + randomFeatures.globalSize);
        assert(
          std::adjacent_find(sortedRoyal.begin(), sortedRoyal.begin() + randomFeatures.royalSize)
          == sortedRoyal.begin() + randomFeatures.royalSize);

        const auto randomReflected =
          extract_full_refresh_features(horizontal_reflection(randomBoard));
        assert(randomReflected.valid());
        assert(randomReflected.royalSize == randomFeatures.royalSize);
        assert(sorted_prefix(randomReflected.royal, randomReflected.royalSize) == sortedRoyal);
    }

    // Exercise the exact V2_BASE_P0 integer path with a non-zero deterministic
    // payload. Both STM heads share every preceding layer.
    ScalarNetwork deterministicNetwork(make_deterministic_parameters(ScalarFixtureSeed));
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
            expected +=
              deterministicParameters
                .royalWeights[std::size_t(startFeatures.royal[active]) * RoyalLanes + lane];
        assert(whiteTrace.royalAccumulator[lane] == expected);
    }
    for (const Eval::NNUE::IndexType lane :
         {Eval::NNUE::IndexType(0), Eval::NNUE::IndexType(19), GlobalLanes - 1})
    {
        Accumulator expected = deterministicParameters.globalBias[lane];
        for (std::size_t active = 0; active < startFeatures.globalSize; ++active)
            expected +=
              deterministicParameters
                .globalWeights[std::size_t(startFeatures.global[active]) * GlobalLanes + lane];
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
    clippingParameters.royalBias[0]      = -1;
    clippingParameters.royalBias[1]      = 64;
    clippingParameters.royalBias[2]      = 64 * 127;
    clippingParameters.royalBias[3]      = 64 * 128;
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

    const auto incrementalBase = incremental_base_board();
    assert(extract_full_refresh_features(incrementalBase).valid());

    auto quietTarget      = incrementalBase;
    quietTarget[SQ_A2]    = NO_PIECE;
    quietTarget[SQ_A3]    = W_PAWN;
    const auto quietDirty = make_dirty(W_PAWN, SQ_A2, SQ_A3);
    assert_incremental_transition(deterministicNetwork, incrementalBase, quietTarget, quietDirty,
                                  BLACK, 0, false);

    auto captureTarget      = incrementalBase;
    captureTarget[SQ_D4]    = NO_PIECE;
    captureTarget[SQ_C4]    = W_ROOK;
    const auto captureDirty = make_dirty(W_ROOK, SQ_D4, SQ_C4, B_PAWN, SQ_C4);
    assert_incremental_transition(deterministicNetwork, incrementalBase, captureTarget,
                                  captureDirty, BLACK, 0, false);

    auto epTarget      = incrementalBase;
    epTarget[SQ_E5]    = NO_PIECE;
    epTarget[SQ_F5]    = NO_PIECE;
    epTarget[SQ_F6]    = W_PAWN;
    const auto epDirty = make_dirty(W_PAWN, SQ_E5, SQ_F6, B_PAWN, SQ_F5);
    assert_incremental_transition(deterministicNetwork, incrementalBase, epTarget, epDirty, BLACK,
                                  0, false);

    auto promotionTarget      = incrementalBase;
    promotionTarget[SQ_B7]    = NO_PIECE;
    promotionTarget[SQ_A8]    = W_QUEEN;
    const auto promotionDirty = make_dirty(W_PAWN, SQ_B7, SQ_NONE, B_ROOK, SQ_A8, W_QUEEN, SQ_A8);
    assert_incremental_transition(deterministicNetwork, incrementalBase, promotionTarget,
                                  promotionDirty, BLACK, 0, false);

    auto kingTarget      = incrementalBase;
    kingTarget[SQ_E8]    = NO_PIECE;
    kingTarget[SQ_E7]    = B_KING;
    const auto kingDirty = make_dirty(B_KING, SQ_E8, SQ_E7);
    assert_incremental_transition(deterministicNetwork, incrementalBase, kingTarget, kingDirty,
                                  WHITE, 1, true);

    // d8/e8 share a canonical Royal bucket, but the mirror bit is part of the
    // key and therefore still forces a refresh.
    auto mirrorCrossingTarget      = incrementalBase;
    mirrorCrossingTarget[SQ_E8]    = NO_PIECE;
    mirrorCrossingTarget[SQ_D8]    = B_KING;
    const auto mirrorCrossingDirty = make_dirty(B_KING, SQ_E8, SQ_D8);
    assert_incremental_transition(deterministicNetwork, incrementalBase, mirrorCrossingTarget,
                                  mirrorCrossingDirty, WHITE, 1, true);

    auto castlingTarget      = incrementalBase;
    castlingTarget[SQ_E8]    = NO_PIECE;
    castlingTarget[SQ_H8]    = NO_PIECE;
    castlingTarget[SQ_G8]    = B_KING;
    castlingTarget[SQ_F8]    = B_ROOK;
    const auto castlingDirty = make_dirty(B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_H8, B_ROOK, SQ_F8);
    assert_incremental_transition(deterministicNetwork, incrementalBase, castlingTarget,
                                  castlingDirty, WHITE, 1, true);

    // Randomized quiet transitions cover every fixed role and arbitrary
    // RoyalKey changes without relying on chess legality or move generation.
    std::mt19937 transitionRng(0x494E4352);
    for (int sample = 0; sample < 256; ++sample)
    {
        std::array<Piece, SQUARE_NB> sourceBoard{};
        std::array<int, SQUARE_NB>   transitionSquares{};
        std::iota(transitionSquares.begin(), transitionSquares.end(), 0);
        std::shuffle(transitionSquares.begin(), transitionSquares.end(), transitionRng);

        const int whiteCount                     = 1 + int(transitionRng() % MaxHordeSidePieces);
        const int blackNonKing                   = int(transitionRng() % MaxRoyalSidePieces);
        int       cursor                         = 0;
        sourceBoard[transitionSquares[cursor++]] = B_KING;
        for (int index = 0; index < whiteCount; ++index)
            sourceBoard[transitionSquares[cursor++]] = HordeRoles[roleDistribution(transitionRng)];
        for (int index = 0; index < blackNonKing; ++index)
            sourceBoard[transitionSquares[cursor++]] = RoyalRoles[roleDistribution(transitionRng)];

        const Square from        = Square(transitionSquares[transitionRng() % cursor]);
        const Square to          = Square(transitionSquares[cursor]);
        const Piece  piece       = sourceBoard[from];
        auto         targetBoard = sourceBoard;
        targetBoard[from]        = NO_PIECE;
        targetBoard[to]          = piece;

        const auto sourceFeatures = extract_full_refresh_features(sourceBoard);
        const auto targetFeatures = extract_full_refresh_features(targetBoard);
        assert(sourceFeatures.valid());
        assert(targetFeatures.valid());
        assert_incremental_transition(deterministicNetwork, sourceBoard, targetBoard,
                                      make_dirty(piece, from, to), ~color_of(piece), sample % 101,
                                      sourceFeatures.royalKey != targetFeatures.royalKey);
    }

    const ScalarTrace incrementalSource =
      deterministicNetwork.evaluate_full_refresh(incrementalBase, WHITE, 0);
    DirtyPiece invalidDirty = quietDirty;
    invalidDirty.pc         = W_KING;
    assert(deterministicNetwork
             .evaluate_incremental(invalidDirty, quietTarget, incrementalSource, BLACK, 0)
             .error
           == ScalarEvalError::INVALID_DIRTY_PIECE);
    assert(
      deterministicNetwork.evaluate_incremental(quietDirty, quietTarget, ScalarTrace{}, BLACK, 0)
        .error
      == ScalarEvalError::INVALID_SOURCE_TRACE);

    std::cout << "Horde V2 feature contracts passed: " << FixedRolePieceSquareDimensions
              << " global and " << RoyalPieceSquareDimensions << " Royal indices; full-refresh "
              << startFeatures.globalSize << "+" << startFeatures.royalSize
              << " active rows; scalar P0=" << whiteTrace.preRule50Value << "/"
              << blackTrace.preRule50Value << "; incremental=263/263\n";
}
