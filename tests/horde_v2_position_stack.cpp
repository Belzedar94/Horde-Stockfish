/*
  Real Position/StateInfo integration checks for the experimental Horde V2
  scalar accumulator stack.
*/

#include <array>
#include <cassert>
#include <cstddef>
#include <deque>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../src/attacks.h"
#include "../src/movegen.h"
#include "../src/nnue/horde_v2_stack.h"
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

void set_position(Position& pos, StateInfo& state, const std::string& fen) {
    const auto error = pos.set(fen, false, &state);
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
                   bool                 expectRoyalRefresh) {
    Position  pos;
    StateInfo root{};
    StateInfo child{};
    set_position(pos, root, fen);

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

}  // namespace

int main() {
    Attacks::init();
    Position::init();

    const ScalarNetwork network(make_deterministic_parameters(ScalarFixtureSeed));
    exercise_special_moves(network);
    exercise_null_move(network);
    exercise_fail_closed(network);
    const RandomReceipt random = exercise_legal_sequences(network);

    std::cout << "Horde V2 real Position stack passed: special=6, legal=" << random.moves
              << ", null=" << random.nullMoves << ", royal-refresh=" << random.royalRefreshes
              << "\n";
}
