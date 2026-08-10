/*
  Real Position/StateInfo integration checks for the experimental Horde V2
  scalar reference and lean accumulator stacks.
*/

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../src/attacks.h"
#include "../src/movegen.h"
#include "../src/nnue/horde_v2_container.h"
#include "../src/position.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE::HordeV2;

namespace {

constexpr u64         ScalarFixtureSeed = 0x4856325F42415345ULL;
constexpr const char* HordeStartFen =
  "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1";

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

void set_position(Position&          pos,
                  StateInfo&         state,
                  const std::string& fen,
                  bool               chess960 = false) {
    const auto error = pos.set(fen, chess960, &state);
    assert(!error.has_value());
}

void assert_legal(const Position& pos, Move move) { assert(MoveList<LEGAL>(pos).contains(move)); }

void assert_dirty_equals(const DirtyPiece& actual, const DirtyPiece& expected) {
    assert(actual.pc == expected.pc);
    assert(actual.from == expected.from);
    assert(actual.to == expected.to);
    assert(actual.remove_sq == expected.remove_sq);
    assert(actual.add_sq == expected.add_sq);
    if (actual.remove_sq != SQ_NONE)
        assert(actual.remove_pc == expected.remove_pc);
    if (actual.add_sq != SQ_NONE)
        assert(actual.add_pc == expected.add_pc);
}

void assert_same_lean_result(const LeanEvalResult& actual, const LeanEvalResult& expected) {
    assert(actual.outputAffine == expected.outputAffine);
    assert(actual.preRule50Value == expected.preRule50Value);
    assert(actual.value == expected.value);
}

template<typename Width, bool ValidateExactBoard>
void assert_lean_current(
  const LeanNetwork<Width>& network,
  LazyAccumulatorStack<LeanNetwork<Width>, ValidateExactBoard>& stack,
  const Position& pos) {
    const FullRefreshFeatures positionFeatures = extract_full_refresh_features(pos);
    const FullRefreshFeatures boardFeatures    = extract_full_refresh_features(pos.piece_array());
    assert(positionFeatures.valid());
    assert(positionFeatures.global == boardFeatures.global);
    assert(positionFeatures.royal == boardFeatures.royal);
    assert(positionFeatures.globalSize == boardFeatures.globalSize);
    assert(positionFeatures.royalSize == boardFeatures.royalSize);
    assert(positionFeatures.royalKey == boardFeatures.royalKey);

    LeanAccumulatorFrame<Width> refreshed{};
    network.full_refresh(refreshed, positionFeatures);

    LeanDenseScratch<Width> scratch{};
    const LeanEvalResult    expected =
      network.propagate(refreshed, scratch, pos.side_to_move(), pos.rule50_count());
    const LeanStackEvaluation actual = stack.evaluate(pos);
    assert(actual.valid());

    const LeanAccumulatorFrame<Width>* current = stack.latest();
    assert(current != nullptr);
    assert(reinterpret_cast<std::uintptr_t>(current) % 64 == 0);
    assert(current->royal == refreshed.royal);
    assert(current->global == refreshed.global);
    assert(current->key == refreshed.key);
    assert_same_lean_result(actual.result, expected);
}

DirtyPiece expected_dirty(Piece  piece,
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

void exercise_move(const ScalarNetwork& network,
                   const std::string&   fen,
                   Move                 move,
                   const DirtyPiece&    expectedDirty,
                   bool                 expectRoyalRefresh,
                   bool                 chess960 = false) {
    Position  pos;
    StateInfo root{};
    StateInfo child{};
    set_position(pos, root, fen, chess960);

    ScalarAccumulatorStack stack(network);
    const ScalarTrace      rootTrace = stack.reset(pos);
    assert_same_evaluation(rootTrace, network.evaluate_full_refresh(pos));
    assert(stack.size() == 1);

    assert_legal(pos, move);
    const auto sourceBoard = pos.piece_array();
    Dirties    dirties{};
    pos.do_move(move, child, pos.gives_check(move), dirties, nullptr, nullptr);

    assert_dirty_equals(dirties.dirtyPiece, expectedDirty);
    assert(dirty_piece_matches_transition(sourceBoard, dirties.dirtyPiece, pos.piece_array()));

    const ScalarTrace incremental = stack.push(dirties, pos);
    const ScalarTrace refreshed   = network.evaluate_full_refresh(pos);
    assert_same_evaluation(incremental, refreshed);
    assert_same_evaluation(stack.evaluate(pos), refreshed);
    assert(incremental.royalRefreshed == expectRoyalRefresh);
    assert(stack.size() == 2);

    pos.undo_move(move);
    assert(stack.pop());
    assert(pos.piece_array() == sourceBoard);
    assert_same_evaluation(stack.evaluate(pos), network.evaluate_full_refresh(pos));
    assert(stack.size() == 1);
    assert(!stack.pop());
}

void exercise_special_moves(const ScalarNetwork& network) {
    exercise_move(network, "4k3/8/8/8/8/8/P7/8 w - - 0 1", Move(SQ_A2, SQ_A3),
                  expected_dirty(W_PAWN, SQ_A2, SQ_A3), false);

    exercise_move(network, "4k3/8/8/8/8/8/2pR4/P7 w - - 0 1", Move(SQ_D2, SQ_C2),
                  expected_dirty(W_ROOK, SQ_D2, SQ_C2, B_PAWN, SQ_C2), false);

    exercise_move(network, "4k3/8/8/4Pp2/8/8/P7/8 w - f6 0 2", Move::make<EN_PASSANT>(SQ_E5, SQ_F6),
                  expected_dirty(W_PAWN, SQ_E5, SQ_F6, B_PAWN, SQ_F5), false);

    exercise_move(network, "r3k3/1P6/8/8/8/8/P7/8 w - - 0 1",
                  Move::make<PROMOTION>(SQ_B7, SQ_A8, QUEEN),
                  expected_dirty(W_PAWN, SQ_B7, SQ_NONE, B_ROOK, SQ_A8, W_QUEEN, SQ_A8), false);

    exercise_move(network, "4k3/8/8/8/8/8/P7/8 b - - 0 1", Move(SQ_E8, SQ_D8),
                  expected_dirty(B_KING, SQ_E8, SQ_D8), true);

    exercise_move(network, "4k2r/8/8/8/8/8/P7/8 b k - 0 1", Move::make<CASTLING>(SQ_E8, SQ_H8),
                  expected_dirty(B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_H8, B_ROOK, SQ_F8), true);

    exercise_move(network, "4k1r1/8/8/8/8/8/P7/8 b g - 0 1",
                  Move::make<CASTLING>(SQ_E8, SQ_G8),
                  expected_dirty(B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_G8, B_ROOK, SQ_F8), true,
                  true);
    exercise_move(network, "5k1r/8/8/8/8/8/P7/8 b h - 0 1",
                  Move::make<CASTLING>(SQ_F8, SQ_H8),
                  expected_dirty(B_KING, SQ_F8, SQ_G8, B_ROOK, SQ_H8, B_ROOK, SQ_F8), true,
                  true);
    exercise_move(network, "6kr/8/8/8/8/8/P7/8 b h - 0 1",
                  Move::make<CASTLING>(SQ_G8, SQ_H8),
                  expected_dirty(B_KING, SQ_G8, SQ_G8, B_ROOK, SQ_H8, B_ROOK, SQ_F8), false,
                  true);
    exercise_move(network, "4kr2/8/8/8/8/8/P7/8 b f - 0 1",
                  Move::make<CASTLING>(SQ_E8, SQ_F8),
                  expected_dirty(B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_F8, B_ROOK, SQ_F8), true,
                  true);
}

template<typename Width>
void exercise_lean_move(const LeanNetwork<Width>& network,
                        const std::string&        fen,
                        Move                      move,
                        const DirtyPiece&         expectedDirty,
                        bool                      expectRoyalRefresh,
                        bool                      chess960 = false) {
    Position  pos;
    StateInfo root{};
    StateInfo child{};
    set_position(pos, root, fen, chess960);

    LeanAccumulatorStack<Width> stack(network);
    assert(stack.reset(pos) == LeanStackError::NONE);
    assert(stack.size() == 1);
    assert(stack.counters().fullRefreshes == 1);
    assert_lean_current(network, stack, pos);

    assert_legal(pos, move);
    Dirties dirties{};
    pos.do_move(move, child, pos.gives_check(move), dirties, nullptr, nullptr);
    assert_dirty_equals(dirties.dirtyPiece, expectedDirty);
    assert(stack.push(dirties, pos) == LeanStackError::NONE);
    assert(stack.size() == 2);
    assert(stack.counters().pushed == 1);
    assert(stack.counters().materialized == std::size_t(expectRoyalRefresh));
    assert(stack.counters().royalRefreshes == std::size_t(expectRoyalRefresh));
    assert_lean_current(network, stack, pos);
    assert(stack.counters().materialized == 1);

    pos.undo_move(move);
    assert(stack.pop());
    assert(stack.size() == 1);
    assert_lean_current(network, stack, pos);
    assert(!stack.pop());
}

template<typename Width>
void exercise_lean_lazy_batch(const LeanNetwork<Width>& network) {
    Position  pos;
    StateInfo root{};
    set_position(pos, root, "4k3/7p/8/8/8/8/P7/8 w - - 0 1");

    LeanAccumulatorStack<Width> stack(network);
    assert(stack.reset(pos) == LeanStackError::NONE);

    const std::array<Move, 6> moves = {Move(SQ_A2, SQ_A3), Move(SQ_H7, SQ_H6), Move(SQ_A3, SQ_A4),
                                       Move(SQ_H6, SQ_H5), Move(SQ_A4, SQ_A5), Move(SQ_H5, SQ_H4)};
    std::deque<StateInfo>     states;
    for (const Move move : moves)
    {
        assert_legal(pos, move);
        states.emplace_back();
        Dirties dirties{};
        pos.do_move(move, states.back(), pos.gives_check(move), dirties, nullptr, nullptr);
        assert(stack.push(dirties, pos) == LeanStackError::NONE);
        assert(stack.latest() == nullptr);
        assert(stack.counters().materialized == 0);
    }

    assert(stack.counters().pushed == moves.size());
    assert(stack.counters().royalRefreshes == 0);
    assert_lean_current(network, stack, pos);
    assert(stack.counters().materialized == moves.size());

    for (auto move = moves.rbegin(); move != moves.rend(); ++move)
    {
        pos.undo_move(*move);
        assert(stack.pop());
    }
    assert(stack.size() == 1);
    assert_lean_current(network, stack, pos);
}

void exercise_lean_special_moves() {
    LeanNetwork<Width64x192> network(
      make_lean_parameters<Width64x192>(ScalarFixtureSeed, FixturePayload::PERF_COMMON_V1));

    exercise_lean_move(network, "4k3/8/8/8/8/8/P7/8 w - - 0 1", Move(SQ_A2, SQ_A3),
                       expected_dirty(W_PAWN, SQ_A2, SQ_A3), false);
    exercise_lean_move(network, "4k3/8/8/8/8/8/2pR4/P7 w - - 0 1", Move(SQ_D2, SQ_C2),
                       expected_dirty(W_ROOK, SQ_D2, SQ_C2, B_PAWN, SQ_C2), false);
    exercise_lean_move(network, "4k3/8/8/4Pp2/8/8/P7/8 w - f6 0 2",
                       Move::make<EN_PASSANT>(SQ_E5, SQ_F6),
                       expected_dirty(W_PAWN, SQ_E5, SQ_F6, B_PAWN, SQ_F5), false);
    exercise_lean_move(
      network, "r3k3/1P6/8/8/8/8/P7/8 w - - 0 1", Move::make<PROMOTION>(SQ_B7, SQ_A8, QUEEN),
      expected_dirty(W_PAWN, SQ_B7, SQ_NONE, B_ROOK, SQ_A8, W_QUEEN, SQ_A8), false);
    exercise_lean_move(network, "4k3/8/8/8/8/8/P7/8 b - - 0 1", Move(SQ_E8, SQ_D8),
                       expected_dirty(B_KING, SQ_E8, SQ_D8), true);
    exercise_lean_move(network, "4k2r/8/8/8/8/8/P7/8 b k - 0 1", Move::make<CASTLING>(SQ_E8, SQ_H8),
                       expected_dirty(B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_H8, B_ROOK, SQ_F8), true);
    exercise_lean_move(network, "4k1r1/8/8/8/8/8/P7/8 b g - 0 1",
                       Move::make<CASTLING>(SQ_E8, SQ_G8),
                       expected_dirty(B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_G8, B_ROOK, SQ_F8), true,
                       true);
    exercise_lean_move(network, "5k1r/8/8/8/8/8/P7/8 b h - 0 1",
                       Move::make<CASTLING>(SQ_F8, SQ_H8),
                       expected_dirty(B_KING, SQ_F8, SQ_G8, B_ROOK, SQ_H8, B_ROOK, SQ_F8), true,
                       true);
    exercise_lean_move(network, "6kr/8/8/8/8/8/P7/8 b h - 0 1",
                       Move::make<CASTLING>(SQ_G8, SQ_H8),
                       expected_dirty(B_KING, SQ_G8, SQ_G8, B_ROOK, SQ_H8, B_ROOK, SQ_F8), false,
                       true);
    exercise_lean_move(network, "4kr2/8/8/8/8/8/P7/8 b f - 0 1",
                       Move::make<CASTLING>(SQ_E8, SQ_F8),
                       expected_dirty(B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_F8, B_ROOK, SQ_F8), true,
                       true);

    Position  pos;
    StateInfo root{};
    set_position(pos, root, "4k3/8/8/8/8/8/P7/8 w - - 0 1");
    LeanAccumulatorStack<Width64x192> stack(network);
    assert(stack.evaluate(pos).error == LeanStackError::STACK_UNINITIALIZED);
    assert(stack.push(expected_dirty(W_PAWN, SQ_A2, SQ_A3), pos)
           == LeanStackError::STACK_UNINITIALIZED);
    assert(stack.reset(pos) == LeanStackError::NONE);
    DirtyPiece invalid = expected_dirty(W_PAWN, SQ_A2, SQ_A3);
    invalid.pc         = W_KING;
    assert(stack.push(invalid, pos) == LeanStackError::INVALID_DIRTY_PIECE);
    assert(stack.size() == 1);

    exercise_lean_lazy_batch(network);
}

bool same_counters(const LeanStackCounters& left, const LeanStackCounters& right) {
    return left.fullRefreshes == right.fullRefreshes && left.pushed == right.pushed
        && left.materialized == right.materialized && left.royalRefreshes == right.royalRefreshes;
}

template<typename Width>
void exercise_lean_fail_closed(const LeanNetwork<Width>& network) {
    Position  pos;
    StateInfo root{};
    set_position(pos, root, "4k3/8/8/8/8/8/P7/8 w - - 0 1");

    ValidatingLeanAccumulatorStack<Width> stack(network);
    assert(stack.reset(pos) == LeanStackError::NONE);
    const LeanAccumulatorFrame<Width> rootFrame = *stack.latest();
    const LeanStackCounters            rootCounters = stack.counters();

    Position  unrelated;
    StateInfo unrelatedRoot{};
    set_position(unrelated, unrelatedRoot, "4k3/8/8/8/8/8/1P6/8 b - - 0 1");
    assert(stack.push(expected_dirty(W_PAWN, SQ_A2, SQ_A3), unrelated)
           == LeanStackError::DIRTY_TARGET_MISMATCH);
    assert(stack.size() == 1);
    assert(same_counters(stack.counters(), rootCounters));
    assert(stack.latest()->royal == rootFrame.royal);
    assert(stack.latest()->global == rootFrame.global);

    const Move move(SQ_A2, SQ_A3);
    StateInfo  child{};
    Dirties    dirties{};
    pos.do_move(move, child, pos.gives_check(move), dirties, nullptr, nullptr);
    assert(stack.evaluate(pos).error == LeanStackError::SOURCE_POSITION_MISMATCH);

    DirtyPiece wrongFrom = dirties.dirtyPiece;
    wrongFrom.from       = SQ_B2;
    assert(stack.push(wrongFrom, pos) == LeanStackError::DIRTY_TARGET_MISMATCH);
    assert(stack.size() == 1);
    assert(same_counters(stack.counters(), rootCounters));

    DirtyPiece invalidActive = dirties.dirtyPiece;
    invalidActive.remove_sq  = SQ_B2;
    invalidActive.remove_pc  = Piece(7);
    assert(stack.push(invalidActive, pos) == LeanStackError::INVALID_DIRTY_PIECE);
    assert(stack.size() == 1);
    assert(same_counters(stack.counters(), rootCounters));

    DirtyPiece malformedPromotion = dirties.dirtyPiece;
    malformedPromotion.add_sq      = SQ_A3;
    malformedPromotion.add_pc      = W_QUEEN;
    assert(stack.push(malformedPromotion, pos) == LeanStackError::INVALID_DIRTY_PIECE);
    assert(stack.size() == 1);
    assert(same_counters(stack.counters(), rootCounters));

    // Inactive piece fields are outside DirtyPiece's contract and may contain
    // stale values. Normalization must ignore them.
    DirtyPiece poisonedInactive = dirties.dirtyPiece;
    poisonedInactive.remove_pc  = Piece(7);
    poisonedInactive.add_pc     = Piece(7);
    assert(stack.push(poisonedInactive, pos) == LeanStackError::NONE);
    assert_lean_current(network, stack, pos);

    pos.undo_move(move);
    assert(stack.pop());
    assert_lean_current(network, stack, pos);

    Position  pendingPos;
    StateInfo pendingRoot{};
    StateInfo quietChild{};
    StateInfo kingChild{};
    set_position(pendingPos, pendingRoot, "4k3/7p/8/8/8/8/P7/8 w - - 0 1");
    ValidatingLeanAccumulatorStack<Width> pendingStack(network);
    assert(pendingStack.reset(pendingPos) == LeanStackError::NONE);

    const Move quiet(SQ_A2, SQ_A3);
    Dirties    quietDirties{};
    pendingPos.do_move(quiet, quietChild, pendingPos.gives_check(quiet), quietDirties, nullptr,
                       nullptr);
    assert(pendingStack.push(quietDirties, pendingPos) == LeanStackError::NONE);
    assert(pendingStack.latest() == nullptr);
    assert(pendingStack.counters().materialized == 0);

    const Move kingMove(SQ_E8, SQ_D8);
    Dirties    kingDirties{};
    pendingPos.do_move(kingMove, kingChild, pendingPos.gives_check(kingMove), kingDirties, nullptr,
                       nullptr);
    DirtyPiece wrongKing = kingDirties.dirtyPiece;
    wrongKing.from       = SQ_F8;
    assert(pendingStack.push(wrongKing, pendingPos) == LeanStackError::DIRTY_TARGET_MISMATCH);
    assert(pendingStack.size() == 2);
    assert(pendingStack.latest() == nullptr);
    assert(pendingStack.counters().materialized == 0);
    assert(pendingStack.counters().royalRefreshes == 0);

    pendingPos.undo_move(kingMove);
    assert_lean_current(network, pendingStack, pendingPos);
    pendingPos.undo_move(quiet);
    assert(pendingStack.pop());
    assert_lean_current(network, pendingStack, pendingPos);
}

template<typename Width>
void exercise_lean_null_child(const LeanNetwork<Width>& network) {
    Position  pos;
    StateInfo root{};
    StateInfo nullState{};
    StateInfo child{};
    set_position(pos, root, "4k3/8/8/8/8/8/P7/8 b - - 7 1");

    LeanAccumulatorStack<Width> stack(network);
    assert(stack.reset(pos) == LeanStackError::NONE);
    const std::size_t rootSize = stack.size();

    pos.do_null_move(nullState);
    assert(stack.size() == rootSize);
    assert_lean_current(network, stack, pos);

    const Move move(SQ_A2, SQ_A3);
    assert_legal(pos, move);
    Dirties dirties{};
    pos.do_move(move, child, pos.gives_check(move), dirties, nullptr, nullptr);
    assert(stack.push(dirties, pos) == LeanStackError::NONE);
    assert_lean_current(network, stack, pos);

    pos.undo_move(move);
    assert(stack.pop());
    assert_lean_current(network, stack, pos);
    pos.undo_null_move();
    assert_lean_current(network, stack, pos);
}

void exercise_null_move(const ScalarNetwork& network) {
    Position  pos;
    StateInfo root{};
    StateInfo nullState{};
    set_position(pos, root, "4k3/8/8/8/8/8/P7/8 b - - 7 1");

    ScalarAccumulatorStack stack(network);
    const ScalarTrace      before = stack.reset(pos);
    const auto             board  = pos.piece_array();
    assert(before.valid());

    pos.do_null_move(nullState);
    assert(pos.piece_array() == board);
    assert(stack.size() == 1);
    const ScalarTrace afterNull = stack.evaluate(pos);
    assert_same_evaluation(afterNull, network.evaluate_full_refresh(pos));

    pos.undo_null_move();
    assert(pos.piece_array() == board);
    assert(stack.size() == 1);
    assert_same_evaluation(stack.evaluate(pos), before);
}

void exercise_fail_closed(const ScalarNetwork& network) {
    ScalarAccumulatorStack emptyStack(network);
    Position               source;
    StateInfo              sourceRoot{};
    set_position(source, sourceRoot, "4k3/8/8/8/8/8/P7/8 w - - 0 1");
    assert(emptyStack.evaluate(source).error == ScalarEvalError::STACK_UNINITIALIZED);

    ScalarAccumulatorStack stack(network);
    assert(stack.reset(source).valid());

    Position  target;
    StateInfo targetRoot{};
    set_position(target, targetRoot, "4k3/8/8/8/8/P7/8/8 b - - 0 1");
    assert(stack.evaluate(target).error == ScalarEvalError::SOURCE_POSITION_MISMATCH);

    StateInfo  child{};
    Dirties    dirties{};
    const Move move(SQ_A2, SQ_A3);
    source.do_move(move, child, source.gives_check(move), dirties, nullptr, nullptr);

    DirtyPiece invalid = dirties.dirtyPiece;
    invalid.pc         = W_KING;
    assert(stack.push(invalid, source).error == ScalarEvalError::INVALID_DIRTY_PIECE);
    assert(stack.size() == 1);

    DirtyPiece mismatch = dirties.dirtyPiece;
    mismatch.from       = SQ_B2;
    assert(stack.push(mismatch, source).error == ScalarEvalError::DIRTY_BOARD_MISMATCH);
    assert(stack.size() == 1);

    assert(stack.push(dirties, source).valid());
    assert(stack.size() == 2);
    source.undo_move(move);
    assert(stack.pop());
    assert_same_evaluation(stack.evaluate(source), network.evaluate_full_refresh(source));
}

struct RandomReceipt {
    std::size_t moves          = 0;
    std::size_t nullMoves      = 0;
    std::size_t royalRefreshes = 0;
};

RandomReceipt exercise_legal_sequences(const ScalarNetwork& network) {
    Position  pos;
    StateInfo root{};
    set_position(pos, root, HordeStartFen);

    ScalarAccumulatorStack stack(network);
    assert(stack.reset(pos).valid());

    std::mt19937  rng(0x504F5354);
    RandomReceipt receipt{};

    for (int sequence = 0; sequence < 4; ++sequence)
    {
        std::deque<StateInfo> states;
        std::vector<Move>     moves;

        for (int ply = 0; ply < 48; ++ply)
        {
            const MoveList<LEGAL> legal(pos);
            if (legal.size() == 0)
                break;

            const Move move        = *(legal.begin() + (rng() % legal.size()));
            const auto sourceBoard = pos.piece_array();
            states.emplace_back();
            Dirties dirties{};
            pos.do_move(move, states.back(), pos.gives_check(move), dirties, nullptr, nullptr);

            assert(
              dirty_piece_matches_transition(sourceBoard, dirties.dirtyPiece, pos.piece_array()));
            const ScalarTrace incremental = stack.push(dirties, pos);
            assert_same_evaluation(incremental, network.evaluate_full_refresh(pos));
            assert_same_evaluation(stack.evaluate(pos), network.evaluate_full_refresh(pos));
            receipt.royalRefreshes += incremental.royalRefreshed;
            ++receipt.moves;
            moves.push_back(move);

            if ((ply % 11) == 5 && !pos.checkers())
            {
                StateInfo  nullState{};
                const auto board = pos.piece_array();
                const auto size  = stack.size();
                pos.do_null_move(nullState);
                assert(pos.piece_array() == board);
                assert(stack.size() == size);
                assert_same_evaluation(stack.evaluate(pos), network.evaluate_full_refresh(pos));
                pos.undo_null_move();
                assert(pos.piece_array() == board);
                assert(stack.size() == size);
                assert_same_evaluation(stack.evaluate(pos), network.evaluate_full_refresh(pos));
                ++receipt.nullMoves;
            }
        }

        while (!moves.empty())
        {
            pos.undo_move(moves.back());
            moves.pop_back();
            assert(stack.pop());
            assert_same_evaluation(stack.evaluate(pos), network.evaluate_full_refresh(pos));
        }

        assert(stack.size() == 1);
        assert(pos.fen() == HordeStartFen);
    }

    return receipt;
}

struct LeanPositionReceipt {
    u64         digest           = 0x4C45414E5F563200ULL;
    std::size_t moves            = 0;
    std::size_t nullMoves        = 0;
    std::size_t royalRefreshes   = 0;
    std::size_t materializations = 0;
};

void mix_lean_receipt(LeanPositionReceipt& receipt, const LeanEvalResult& result) {
    const auto mix = [&](u64 value) {
        receipt.digest ^=
          value + 0x9E3779B97F4A7C15ULL + (receipt.digest << 6) + (receipt.digest >> 2);
    };
    mix(u64(i64(result.outputAffine)));
    mix(u64(i64(result.preRule50Value)));
    mix(u64(i64(result.value)));
}

template<typename Width>
LeanPositionReceipt exercise_lean_legal_sequences() {
    LeanNetwork<Width> network(
      make_lean_parameters<Width>(ScalarFixtureSeed, FixturePayload::PERF_COMMON_V1));

    Position  pos;
    StateInfo root{};
    set_position(pos, root, HordeStartFen);

    LeanAccumulatorStack<Width> stack(network);
    assert(stack.reset(pos) == LeanStackError::NONE);
    assert_lean_current(network, stack, pos);

    std::mt19937        rng(0x504F5354);
    LeanPositionReceipt receipt{};

    for (int sequence = 0; sequence < 4; ++sequence)
    {
        std::deque<StateInfo> states;
        std::vector<Move>     moves;

        for (int ply = 0; ply < 48; ++ply)
        {
            const MoveList<LEGAL> legal(pos);
            if (legal.size() == 0)
                break;

            const Move move = *(legal.begin() + (rng() % legal.size()));
            states.emplace_back();
            Dirties dirties{};
            pos.do_move(move, states.back(), pos.gives_check(move), dirties, nullptr, nullptr);

            assert(stack.push(dirties, pos) == LeanStackError::NONE);
            assert_lean_current(network, stack, pos);
            const LeanStackEvaluation evaluation = stack.evaluate(pos);
            assert(evaluation.valid());
            mix_lean_receipt(receipt, evaluation.result);
            ++receipt.moves;
            moves.push_back(move);

            if ((ply % 11) == 5 && !pos.checkers())
            {
                StateInfo  nullState{};
                const auto size = stack.size();
                pos.do_null_move(nullState);
                assert(stack.size() == size);
                assert_lean_current(network, stack, pos);
                const LeanStackEvaluation nullEvaluation = stack.evaluate(pos);
                assert(nullEvaluation.valid());
                mix_lean_receipt(receipt, nullEvaluation.result);
                pos.undo_null_move();
                assert(stack.size() == size);
                assert_lean_current(network, stack, pos);
                ++receipt.nullMoves;
            }
        }

        while (!moves.empty())
        {
            pos.undo_move(moves.back());
            moves.pop_back();
            assert(stack.pop());
            assert_lean_current(network, stack, pos);
        }

        assert(stack.size() == 1);
        assert(pos.fen() == HordeStartFen);
    }

    receipt.royalRefreshes   = stack.counters().royalRefreshes;
    receipt.materializations = stack.counters().materialized;
    assert(stack.counters().pushed == receipt.moves);
    assert(receipt.materializations == receipt.moves);
    assert(receipt.royalRefreshes > 0);
    return receipt;
}

void assert_same_lean_receipt(const LeanPositionReceipt& actual,
                              const LeanPositionReceipt& expected) {
    assert(actual.digest == expected.digest);
    assert(actual.moves == expected.moves);
    assert(actual.nullMoves == expected.nullMoves);
    assert(actual.royalRefreshes == expected.royalRefreshes);
    assert(actual.materializations == expected.materializations);
}

template<typename Kernels, bool ValidateExactBoard>
void assert_container_current(
  const ContainerNetwork<Kernels>& network,
  LazyAccumulatorStack<ContainerNetwork<Kernels>, ValidateExactBoard>& stack,
  const Position& pos) {
    const FullRefreshFeatures features = extract_full_refresh_features(pos);
    assert(features.valid());

    ContainerAccumulatorFrame refreshed{};
    network.full_refresh(refreshed, features);
    ContainerDenseScratch scratch{};
    const LeanEvalResult expected =
      network.propagate(refreshed, scratch, pos.side_to_move(), pos.rule50_count());
    const LeanStackEvaluation actual = stack.evaluate(pos);
    assert(actual.valid());

    const ContainerAccumulatorFrame* current = stack.latest();
    assert(current != nullptr);
    assert(reinterpret_cast<std::uintptr_t>(current) % 64 == 0);
    assert(current->first == refreshed.first);
    assert(current->global == refreshed.global);
    assert(current->key == refreshed.key);
    assert_same_lean_result(actual.result, expected);
}

template<typename Kernels>
void exercise_container_move(const ContainerNetwork<Kernels>& network,
                             const std::string&                fen,
                             Move                              move,
                             const DirtyPiece&                 expectedDirty,
                             bool                              expectFirstRefresh,
                             bool                              chess960 = false) {
    Position  pos;
    StateInfo root{};
    StateInfo child{};
    set_position(pos, root, fen, chess960);

    ContainerAccumulatorStack<Kernels> stack(network);
    assert(stack.reset(pos) == LeanStackError::NONE);
    assert_container_current(network, stack, pos);

    assert_legal(pos, move);
    Dirties dirties{};
    pos.do_move(move, child, pos.gives_check(move), dirties, nullptr, nullptr);
    assert_dirty_equals(dirties.dirtyPiece, expectedDirty);
    assert(stack.push(dirties, pos) == LeanStackError::NONE);
    assert(stack.counters().materialized == std::size_t(expectFirstRefresh));
    assert(stack.counters().royalRefreshes == std::size_t(expectFirstRefresh));
    assert_container_current(network, stack, pos);
    assert(stack.counters().materialized == 1);

    pos.undo_move(move);
    assert(stack.pop());
    assert_container_current(network, stack, pos);
}

template<typename Kernels>
void exercise_container_special_moves(const ContainerNetwork<Kernels>& network) {
    const bool fullRoyal = network.first_domain() == FirstDomain::ROYAL;
    const bool keyedRoyal = network.first_domain() != FirstDomain::ABSOLUTE_NONKING;
    exercise_container_move(network, "4k3/8/8/8/8/8/P7/8 w - - 0 1", Move(SQ_A2, SQ_A3),
                            expected_dirty(W_PAWN, SQ_A2, SQ_A3), false);
    exercise_container_move(network, "4k3/8/8/8/8/8/2pR4/P7 w - - 0 1", Move(SQ_D2, SQ_C2),
                            expected_dirty(W_ROOK, SQ_D2, SQ_C2, B_PAWN, SQ_C2), false);
    exercise_container_move(network, "4k3/8/8/4Pp2/8/8/P7/8 w - f6 0 2",
                            Move::make<EN_PASSANT>(SQ_E5, SQ_F6),
                            expected_dirty(W_PAWN, SQ_E5, SQ_F6, B_PAWN, SQ_F5), false);
    exercise_container_move(
      network, "r3k3/1P6/8/8/8/8/P7/8 w - - 0 1",
      Move::make<PROMOTION>(SQ_B7, SQ_A8, QUEEN),
      expected_dirty(W_PAWN, SQ_B7, SQ_NONE, B_ROOK, SQ_A8, W_QUEEN, SQ_A8), false);
    exercise_container_move(network, "4k3/8/8/8/8/8/P7/8 b - - 0 1", Move(SQ_E8, SQ_D8),
                            expected_dirty(B_KING, SQ_E8, SQ_D8), keyedRoyal);
    exercise_container_move(network, "3k4/8/8/8/8/8/P7/8 b - - 0 1", Move(SQ_D8, SQ_D7),
                            expected_dirty(B_KING, SQ_D8, SQ_D7), keyedRoyal);
    exercise_container_move(network, "4k2r/8/8/8/8/8/P7/8 b k - 0 1",
                            Move::make<CASTLING>(SQ_E8, SQ_H8),
                            expected_dirty(B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_H8, B_ROOK, SQ_F8),
                            fullRoyal);
    exercise_container_move(network, "4k1r1/8/8/8/8/8/P7/8 b g - 0 1",
                            Move::make<CASTLING>(SQ_E8, SQ_G8),
                            expected_dirty(B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_G8, B_ROOK, SQ_F8),
                            fullRoyal, true);
    exercise_container_move(network, "5k1r/8/8/8/8/8/P7/8 b h - 0 1",
                            Move::make<CASTLING>(SQ_F8, SQ_H8),
                            expected_dirty(B_KING, SQ_F8, SQ_G8, B_ROOK, SQ_H8, B_ROOK, SQ_F8),
                            fullRoyal, true);
    exercise_container_move(network, "6kr/8/8/8/8/8/P7/8 b h - 0 1",
                            Move::make<CASTLING>(SQ_G8, SQ_H8),
                            expected_dirty(B_KING, SQ_G8, SQ_G8, B_ROOK, SQ_H8, B_ROOK, SQ_F8),
                            false, true);
    exercise_container_move(network, "4kr2/8/8/8/8/8/P7/8 b f - 0 1",
                            Move::make<CASTLING>(SQ_E8, SQ_F8),
                            expected_dirty(B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_F8, B_ROOK, SQ_F8),
                            fullRoyal, true);
}

template<typename Kernels>
void exercise_container_lazy_batch(const ContainerNetwork<Kernels>& network) {
    Position  pos;
    StateInfo root{};
    set_position(pos, root, "4k3/7p/8/8/8/8/P7/8 w - - 0 1");

    ContainerAccumulatorStack<Kernels> stack(network);
    assert(stack.reset(pos) == LeanStackError::NONE);
    const std::array<Move, 6> moves = {Move(SQ_A2, SQ_A3), Move(SQ_H7, SQ_H6),
                                       Move(SQ_A3, SQ_A4), Move(SQ_H6, SQ_H5),
                                       Move(SQ_A4, SQ_A5), Move(SQ_H5, SQ_H4)};
    std::deque<StateInfo> states;
    for (const Move move : moves)
    {
        states.emplace_back();
        Dirties dirties{};
        pos.do_move(move, states.back(), pos.gives_check(move), dirties, nullptr, nullptr);
        assert(stack.push(dirties, pos) == LeanStackError::NONE);
        assert(stack.latest() == nullptr);
    }
    assert(stack.counters().materialized == 0);
    assert_container_current(network, stack, pos);
    assert(stack.counters().materialized == moves.size());

    for (auto move = moves.rbegin(); move != moves.rend(); ++move)
    {
        pos.undo_move(*move);
        assert(stack.pop());
    }
    assert_container_current(network, stack, pos);
}

template<typename Kernels>
void exercise_container_fail_closed(const ContainerNetwork<Kernels>& network) {
    Position  pos;
    StateInfo root{};
    StateInfo child{};
    set_position(pos, root, "4k3/8/8/8/8/8/P7/8 w - - 0 1");

    ValidatingContainerAccumulatorStack<Kernels> stack(network);
    assert(stack.reset(pos) == LeanStackError::NONE);
    const LeanStackCounters rootCounters = stack.counters();

    const Move move(SQ_A2, SQ_A3);
    Dirties    dirties{};
    pos.do_move(move, child, pos.gives_check(move), dirties, nullptr, nullptr);
    assert(stack.evaluate(pos).error == LeanStackError::SOURCE_POSITION_MISMATCH);

    DirtyPiece wrong = dirties.dirtyPiece;
    wrong.from       = SQ_B2;
    assert(stack.push(wrong, pos) == LeanStackError::DIRTY_TARGET_MISMATCH);
    assert(stack.size() == 1);
    assert(same_counters(stack.counters(), rootCounters));

    DirtyPiece poisoned = dirties.dirtyPiece;
    poisoned.remove_pc  = Piece(7);
    poisoned.add_pc     = Piece(7);
    assert(stack.push(poisoned, pos) == LeanStackError::NONE);
    assert_container_current(network, stack, pos);

    pos.undo_move(move);
    assert(stack.pop());
    assert_container_current(network, stack, pos);
}

template<typename Kernels>
LeanPositionReceipt exercise_container_legal_sequences(const ContainerNetwork<Kernels>& network) {
    Position  pos;
    StateInfo root{};
    set_position(pos, root, HordeStartFen);

    ContainerAccumulatorStack<Kernels> stack(network);
    assert(stack.reset(pos) == LeanStackError::NONE);
    assert_container_current(network, stack, pos);

    std::mt19937        rng(0x504F5354);
    LeanPositionReceipt receipt{};
    for (int sequence = 0; sequence < 4; ++sequence)
    {
        std::deque<StateInfo> states;
        std::vector<Move>     moves;
        for (int ply = 0; ply < 48; ++ply)
        {
            const MoveList<LEGAL> legal(pos);
            if (legal.size() == 0)
                break;
            const Move move = *(legal.begin() + (rng() % legal.size()));
            states.emplace_back();
            Dirties dirties{};
            pos.do_move(move, states.back(), pos.gives_check(move), dirties, nullptr, nullptr);
            assert(stack.push(dirties, pos) == LeanStackError::NONE);
            assert_container_current(network, stack, pos);
            const LeanStackEvaluation evaluation = stack.evaluate(pos);
            assert(evaluation.valid());
            mix_lean_receipt(receipt, evaluation.result);
            ++receipt.moves;
            moves.push_back(move);

            if ((ply % 11) == 5 && !pos.checkers())
            {
                StateInfo nullState{};
                pos.do_null_move(nullState);
                assert_container_current(network, stack, pos);
                pos.undo_null_move();
                assert_container_current(network, stack, pos);
                ++receipt.nullMoves;
            }
        }

        while (!moves.empty())
        {
            pos.undo_move(moves.back());
            moves.pop_back();
            assert(stack.pop());
            assert_container_current(network, stack, pos);
        }
        assert(pos.fen() == HordeStartFen);
    }

    receipt.royalRefreshes   = stack.counters().royalRefreshes;
    receipt.materializations = stack.counters().materialized;
    assert(stack.counters().pushed == receipt.moves);
    assert(receipt.materializations == receipt.moves);
    if (network.first_domain() != FirstDomain::ABSOLUTE_NONKING)
        assert(receipt.royalRefreshes > 0);
    else
        assert(receipt.royalRefreshes == 0);
    return receipt;
}

LeanPositionReceipt exercise_container_file(const std::filesystem::path& path) {
    ContainerLoadResult loaded = load_integer_container(path);
    assert(loaded);
    ContainerNetwork<> network(loaded.parameters);
    exercise_container_special_moves(network);
    exercise_container_lazy_batch(network);
    exercise_container_fail_closed(network);
    return exercise_container_legal_sequences(network);
}

}  // namespace

int main(int argc, char** argv) {
    Attacks::init();
    Position::init();

    const ScalarNetwork network(make_deterministic_parameters(ScalarFixtureSeed));
    exercise_special_moves(network);
    exercise_null_move(network);
    exercise_fail_closed(network);
    const RandomReceipt random = exercise_legal_sequences(network);

    exercise_lean_special_moves();
    LeanNetwork<Width64x192> validationNetwork(
      make_lean_parameters<Width64x192>(ScalarFixtureSeed, FixturePayload::PERF_COMMON_V1));
    exercise_lean_fail_closed(validationNetwork);
    exercise_lean_null_child(validationNetwork);
    const LeanPositionReceipt lean256x256 = exercise_lean_legal_sequences<Width256x256>();
    assert_same_lean_receipt(exercise_lean_legal_sequences<Width128x256>(), lean256x256);
    assert_same_lean_receipt(exercise_lean_legal_sequences<Width128x128>(), lean256x256);
    assert_same_lean_receipt(exercise_lean_legal_sequences<Width64x192>(), lean256x256);

    std::size_t containerMoves = 0;
    for (int index = 1; index < argc; ++index)
        containerMoves += exercise_container_file(argv[index]).moves;

    std::cout << "Horde V2 real Position stack passed: special=10, legal=" << random.moves
              << ", null=" << random.nullMoves << ", royal-refresh=" << random.royalRefreshes
              << "; lean-special=10, lean-common=" << lean256x256.moves
              << ", lean-null=" << lean256x256.nullMoves
              << ", lean-royal-refresh=" << lean256x256.royalRefreshes
              << "; containers=" << (argc - 1) << ", container-legal=" << containerMoves << "\n";
}
