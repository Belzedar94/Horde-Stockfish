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
#include <cstdint>
#include <utility>
#include <vector>

#include "horde_v2_backend.h"

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

enum class LeanStackError : std::uint8_t {
    NONE,
    INVALID_POSITION,
    INVALID_DIRTY_PIECE,
    STACK_UNINITIALIZED,
    STACK_OVERFLOW,
};

struct LeanStackEvaluation {
    LeanEvalResult result{};
    LeanStackError error = LeanStackError::NONE;

    [[nodiscard]] constexpr bool valid() const noexcept { return error == LeanStackError::NONE; }
};

struct LeanStackCounters {
    u64 fullRefreshes  = 0;
    u64 pushed         = 0;
    u64 materialized   = 0;
    u64 royalRefreshes = 0;
};

// Production-layout stack for the width-templated backend. Frames are
// allocated and aligned once; push/pop never allocate, keep board copies or
// retain dense intermediates. Ordinary deltas are materialized lazily, just as
// in search. A Royal-key transition is rare and is materialized while its
// target Position is available.
template<typename Width, typename Kernels = DefaultLeanKernels>
class LeanAccumulatorStack {
   public:
    static constexpr std::size_t MaxSize = MAX_PLY + 1;

    using Network = LeanNetwork<Width, Kernels>;
    using Frame   = typename Network::Frame;
    using Scratch = typename Network::Scratch;

   private:
    struct Slot {
        Frame      frame{};
        DirtyPiece dirty{};
        bool       computed = false;
    };

   public:
    explicit LeanAccumulatorStack(const Network& network) :
        network_(network),
        slots_(MaxSize) {}

    [[nodiscard]] LeanStackError reset(const Position& pos) noexcept {
        size_     = 0;
        counters_ = {};

        const FullRefreshFeatures features = extract_full_refresh_features(pos);
        if (!features.valid())
            return LeanStackError::INVALID_POSITION;

        network_.full_refresh(slots_[0].frame, features);
        slots_[0].computed = true;
        size_              = 1;
        ++counters_.fullRefreshes;
        return LeanStackError::NONE;
    }

    [[nodiscard]] LeanStackError push(const Dirties& dirties, const Position& target) noexcept {
        return push(dirties.dirtyPiece, target);
    }

    [[nodiscard]] LeanStackError push(const DirtyPiece& dirty, const Position& target) noexcept {
        if (size_ == 0)
            return LeanStackError::STACK_UNINITIALIZED;
        if (size_ >= MaxSize)
            return LeanStackError::STACK_OVERFLOW;
        if (!valid_scalar_dirty_piece(dirty))
            return LeanStackError::INVALID_DIRTY_PIECE;
        if (target.count<KING>(WHITE) != 0 || target.count<KING>(BLACK) != 1)
            return LeanStackError::INVALID_POSITION;

        Slot&          child     = slots_[size_];
        const RoyalKey sourceKey = slots_[size_ - 1].frame.key;
        const RoyalKey targetKey = royal_key(target.square<KING>(BLACK));
        child.dirty              = dirty;
        child.frame.key          = targetKey;
        child.computed           = false;

        if (targetKey != sourceKey)
        {
            if (!materialize_through(size_ - 1)
                || !network_.materialize_child(child.frame, slots_[size_ - 1].frame, dirty, target))
                return LeanStackError::INVALID_POSITION;

            child.computed = true;
            ++counters_.materialized;
            ++counters_.royalRefreshes;
        }

        ++size_;
        ++counters_.pushed;
        return LeanStackError::NONE;
    }

    // Null moves reuse the latest frame and only change STM/rule50 inputs.
    [[nodiscard]] LeanStackEvaluation evaluate(const Position& pos) noexcept {
        if (size_ == 0)
            return {{}, LeanStackError::STACK_UNINITIALIZED};
        if (!materialize_through(size_ - 1))
            return {{}, LeanStackError::INVALID_POSITION};

        return {network_.propagate(slots_[size_ - 1].frame, scratch_, pos.side_to_move(),
                                   pos.rule50_count()),
                LeanStackError::NONE};
    }

    [[nodiscard]] bool pop() noexcept {
        if (size_ <= 1)
            return false;
        --size_;
        return true;
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    [[nodiscard]] const Frame* latest() const noexcept {
        return size_ == 0 || !slots_[size_ - 1].computed ? nullptr : &slots_[size_ - 1].frame;
    }

    [[nodiscard]] const LeanStackCounters& counters() const noexcept { return counters_; }

   private:
    [[nodiscard]] bool materialize_through(std::size_t target) noexcept {
        if (slots_[target].computed)
            return true;

        std::size_t source = target;
        while (source > 0 && !slots_[source].computed)
            --source;
        if (!slots_[source].computed)
            return false;

        for (std::size_t next = source + 1; next <= target; ++next)
        {
            Slot& child = slots_[next];
            if (!network_.materialize_child_same_key(child.frame, slots_[next - 1].frame,
                                                     child.dirty, child.frame.key))
                return false;
            child.computed = true;
            ++counters_.materialized;
        }
        return true;
    }

    const Network&    network_;
    std::vector<Slot> slots_;
    Scratch           scratch_{};
    std::size_t       size_ = 0;
    LeanStackCounters counters_{};
};

}  // namespace Stockfish::Eval::NNUE::HordeV2

#endif  // HORDE_V2_STACK_H_INCLUDED
