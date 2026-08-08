/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_V2_STACK_H_INCLUDED
#define HORDE_V2_STACK_H_INCLUDED

#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include "horde_v2_scalar.h"

namespace Stockfish::Eval::NNUE::HordeV2 {

inline bool valid_scalar_dirty_piece(const DirtyPiece& dirty) noexcept {
    return is_registered_piece(dirty.pc) && is_ok(dirty.from)
        && (dirty.to == SQ_NONE || is_ok(dirty.to))
        && (dirty.remove_sq == SQ_NONE
            || (is_ok(dirty.remove_sq) && is_registered_piece(dirty.remove_pc)))
        && (dirty.add_sq == SQ_NONE || (is_ok(dirty.add_sq) && is_registered_piece(dirty.add_pc)));
}

// Validate the exact physical board transition represented by DirtyPiece.
// Removal is applied before addition, matching Position::do_move() for
// captures, promotions, en passant and (including overlapping Chess960
// squares) castling.
inline bool dirty_piece_matches_transition(const std::array<Piece, SQUARE_NB>& source,
                                           const DirtyPiece&                   dirty,
                                           const std::array<Piece, SQUARE_NB>& target) noexcept {
    if (!valid_scalar_dirty_piece(dirty))
        return false;

    std::array<Piece, SQUARE_NB> expected = source;
    if (expected[dirty.from] != dirty.pc)
        return false;
    expected[dirty.from] = NO_PIECE;

    if (dirty.remove_sq != SQ_NONE)
    {
        if (expected[dirty.remove_sq] != dirty.remove_pc)
            return false;
        expected[dirty.remove_sq] = NO_PIECE;
    }

    if (dirty.to != SQ_NONE)
    {
        if (expected[dirty.to] != NO_PIECE)
            return false;
        expected[dirty.to] = dirty.pc;
    }

    if (dirty.add_sq != SQ_NONE)
    {
        if (expected[dirty.add_sq] != NO_PIECE)
            return false;
        expected[dirty.add_sq] = dirty.add_pc;
    }

    return expected == target;
}

// Engineering-only reference for the real search stack contract. Position
// owns StateInfo, while Stockfish's AccumulatorStack owns Dirties; this class
// consumes the same Dirties after Position::do_move() and keeps a separate V2
// frame so the production Run 6B accumulator remains bit-identical.
class ScalarAccumulatorStack {
   public:
    static constexpr std::size_t MaxSize = MAX_PLY + 1;

    explicit ScalarAccumulatorStack(const ScalarNetwork& network) :
        network_(network) {
        frames_.reserve(MaxSize);
    }

    [[nodiscard]] ScalarTrace reset(const Position& pos) {
        frames_.clear();
        ScalarTrace trace = network_.evaluate_full_refresh(pos);
        if (trace.valid())
            frames_.push_back({pos.piece_array(), trace});
        return trace;
    }

    [[nodiscard]] ScalarTrace push(const Dirties& dirties, const Position& target) {
        return push(dirties.dirtyPiece, target);
    }

    [[nodiscard]] ScalarTrace push(const DirtyPiece& dirty, const Position& target) {
        if (frames_.empty())
            return error_trace(ScalarEvalError::STACK_UNINITIALIZED);
        if (frames_.size() >= MaxSize)
            return error_trace(ScalarEvalError::STACK_OVERFLOW);
        if (!valid_scalar_dirty_piece(dirty))
            return error_trace(ScalarEvalError::INVALID_DIRTY_PIECE);
        if (!dirty_piece_matches_transition(frames_.back().board, dirty, target.piece_array()))
            return error_trace(ScalarEvalError::DIRTY_BOARD_MISMATCH);

        ScalarTrace trace = network_.evaluate_incremental(dirty, target, frames_.back().trace);
        if (trace.valid())
            frames_.push_back({target.piece_array(), trace});
        return trace;
    }

    // This mirrors search: null moves do not push/pop the accumulator stack.
    // The board identity check prevents accidentally reusing an unrelated
    // frame while still allowing STM and rule50 to change.
    [[nodiscard]] ScalarTrace evaluate(const Position& pos) const noexcept {
        if (frames_.empty())
            return error_trace(ScalarEvalError::STACK_UNINITIALIZED);
        if (frames_.back().board != pos.piece_array())
            return error_trace(ScalarEvalError::SOURCE_POSITION_MISMATCH);
        return network_.evaluate_from_accumulators(frames_.back().trace, pos);
    }

    [[nodiscard]] bool pop() noexcept {
        if (frames_.size() <= 1)
            return false;
        frames_.pop_back();
        return true;
    }

    [[nodiscard]] std::size_t size() const noexcept { return frames_.size(); }

    [[nodiscard]] const ScalarTrace* latest() const noexcept {
        return frames_.empty() ? nullptr : &frames_.back().trace;
    }

   private:
    struct Frame {
        std::array<Piece, SQUARE_NB> board{};
        ScalarTrace                  trace{};
    };

    [[nodiscard]] static ScalarTrace error_trace(ScalarEvalError error) noexcept {
        ScalarTrace trace{};
        trace.error = error;
        return trace;
    }

    const ScalarNetwork& network_;
    std::vector<Frame>   frames_;
};

}  // namespace Stockfish::Eval::NNUE::HordeV2

#endif  // HORDE_V2_STACK_H_INCLUDED
