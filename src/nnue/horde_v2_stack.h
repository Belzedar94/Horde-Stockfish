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
    NormalizedDirty normalized{};
    return normalize_dirty_piece(dirty, normalized);
}

inline bool apply_normalized_dirty(std::array<Piece, SQUARE_NB>& board,
                                   const NormalizedDirty&        dirty) noexcept {
    if (board[dirty.from] != dirty.pc)
        return false;

    // Remove every source-state piece before adding any target-state piece.
    board[dirty.from] = NO_PIECE;
    if (dirty.remove_sq != SQ_NONE)
    {
        if (board[dirty.remove_sq] != dirty.remove_pc)
            return false;
        board[dirty.remove_sq] = NO_PIECE;
    }

    if (dirty.to != SQ_NONE)
    {
        if (board[dirty.to] != NO_PIECE)
            return false;
        board[dirty.to] = dirty.pc;
    }
    if (dirty.add_sq != SQ_NONE)
    {
        if (board[dirty.add_sq] != NO_PIECE)
            return false;
        board[dirty.add_sq] = dirty.add_pc;
    }
    return true;
}

inline bool undo_normalized_dirty(std::array<Piece, SQUARE_NB>& board,
                                  const NormalizedDirty&        dirty) noexcept {
    // Remove every target-state addition before restoring source-state pieces.
    if (dirty.add_sq != SQ_NONE)
    {
        if (board[dirty.add_sq] != dirty.add_pc)
            return false;
        board[dirty.add_sq] = NO_PIECE;
    }
    if (dirty.to != SQ_NONE)
    {
        if (board[dirty.to] != dirty.pc)
            return false;
        board[dirty.to] = NO_PIECE;
    }

    if (board[dirty.from] != NO_PIECE)
        return false;
    board[dirty.from] = dirty.pc;
    if (dirty.remove_sq != SQ_NONE)
    {
        if (board[dirty.remove_sq] != NO_PIECE)
            return false;
        board[dirty.remove_sq] = dirty.remove_pc;
    }
    return true;
}

// Validate the exact physical board transition represented by DirtyPiece.
// Removal is applied before addition, matching Position::do_move() for
// captures, promotions, en passant and (including overlapping Chess960
// squares) castling.
inline bool dirty_piece_matches_transition(const std::array<Piece, SQUARE_NB>& source,
                                           const DirtyPiece&                   dirty,
                                           const std::array<Piece, SQUARE_NB>& target) noexcept {
    NormalizedDirty normalized{};
    if (!normalize_dirty_piece(dirty, normalized))
        return false;
    std::array<Piece, SQUARE_NB> expected = source;
    return apply_normalized_dirty(expected, normalized) && expected == target;
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
    DIRTY_TARGET_MISMATCH,
    SOURCE_POSITION_MISMATCH,
    STACK_UNINITIALIZED,
    STACK_OVERFLOW,
    INTERNAL_ERROR,
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

// Shared production-layout orchestration for deterministic and trained V2
// networks. Frames are allocated once and dense intermediates are stack-owned.
// The validating specialization keeps one exact 64-byte board shadow for CI
// and assertions; the production specialization removes those board scans
// from push, pop and evaluate.
template<typename NetworkType, bool ValidateExactBoard = false>
class LazyAccumulatorStack {
   public:
    static constexpr std::size_t MaxSize = MAX_PLY + 1;

    using Network = NetworkType;
    using Frame   = typename Network::Frame;
    using Scratch = typename Network::Scratch;

   private:
    struct Slot {
        Frame           frame{};
        NormalizedDirty dirty{};
        bool            computed = false;
    };

   public:
    explicit LazyAccumulatorStack(const Network& network) :
        network_(network),
        slots_(MaxSize) {}

    [[nodiscard]] LeanStackError reset(const Position& pos) noexcept {
        size_     = 0;
        counters_ = {};

        if (!network_.valid())
            return LeanStackError::INVALID_POSITION;

        const FullRefreshFeatures features = extract_full_refresh_features(pos);
        if (!features.valid())
            return LeanStackError::INVALID_POSITION;

        network_.full_refresh(slots_[0].frame, features);
        slots_[0].computed = true;
        if constexpr (ValidateExactBoard)
            board_ = pos.piece_array();
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
        NormalizedDirty normalized{};
        if (!normalize_dirty_piece(dirty, normalized))
            return LeanStackError::INVALID_DIRTY_PIECE;

        if constexpr (ValidateExactBoard)
        {
            std::array<Piece, SQUARE_NB> candidate = board_;
            if (!apply_normalized_dirty(candidate, normalized)
                || candidate != target.piece_array())
                return LeanStackError::DIRTY_TARGET_MISMATCH;
        }
        if (target.count<KING>(WHITE) != 0 || target.count<KING>(BLACK) != 1)
            return LeanStackError::INVALID_POSITION;

        Slot&          child     = slots_[size_];
        const RoyalKey sourceKey = slots_[size_ - 1].frame.key;
        const RoyalKey targetKey = network_.first_domain_key(target);
        const bool     refresh   = network_.requires_refresh(sourceKey, targetKey);

        if (refresh)
        {
            const FullRefreshFeatures targetFeatures = extract_full_refresh_features(target);
            if (!targetFeatures.valid()
                || network_.first_domain_key(targetFeatures) != targetKey)
                return LeanStackError::INVALID_POSITION;

            child.dirty     = normalized;
            child.frame.key = targetKey;
            child.computed  = false;
            if (!materialize_through(size_ - 1))
                return LeanStackError::INTERNAL_ERROR;
            if (!network_.materialize_child(child.frame, slots_[size_ - 1].frame, normalized,
                                            targetFeatures))
                return LeanStackError::INTERNAL_ERROR;

            child.computed = true;
            ++counters_.materialized;
            ++counters_.royalRefreshes;
        }
        else
        {
            child.dirty     = normalized;
            child.frame.key = targetKey;
            child.computed  = false;
        }

        if constexpr (ValidateExactBoard)
        {
            const bool applied = apply_normalized_dirty(board_, normalized);
            assert(applied);
            (void) applied;
        }
        ++size_;
        ++counters_.pushed;
        return LeanStackError::NONE;
    }

    // Null moves reuse the latest frame and only change STM/rule50 inputs.
    [[nodiscard]] LeanStackEvaluation evaluate(const Position& pos) noexcept {
        if (size_ == 0)
            return {{}, LeanStackError::STACK_UNINITIALIZED};
        if constexpr (ValidateExactBoard)
            if (board_ != pos.piece_array())
                return {{}, LeanStackError::SOURCE_POSITION_MISMATCH};
        if (!materialize_through(size_ - 1))
            return {{}, LeanStackError::INTERNAL_ERROR};

        return {network_.propagate(slots_[size_ - 1].frame, scratch_, pos.side_to_move(),
                                   pos.rule50_count()),
                LeanStackError::NONE};
    }

    [[nodiscard]] bool pop() noexcept {
        if (size_ <= 1)
            return false;

        if constexpr (ValidateExactBoard)
        {
            std::array<Piece, SQUARE_NB> parent = board_;
            if (!undo_normalized_dirty(parent, slots_[size_ - 1].dirty))
                return false;
            board_ = parent;
        }
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
            if (!network_.materialize_child_delta(child.frame, slots_[next - 1].frame, child.dirty,
                                                  child.frame.key))
                return false;
            child.computed = true;
            ++counters_.materialized;
        }
        return true;
    }

    const Network&    network_;
    std::vector<Slot> slots_;
    Scratch           scratch_{};
    std::array<Piece, SQUARE_NB> board_{};
    std::size_t       size_ = 0;
    LeanStackCounters counters_{};
};

template<typename Width, typename Kernels = DefaultLeanKernels>
using LeanAccumulatorStack = LazyAccumulatorStack<LeanNetwork<Width, Kernels>>;

template<typename Width, typename Kernels = DefaultLeanKernels>
using ValidatingLeanAccumulatorStack = LazyAccumulatorStack<LeanNetwork<Width, Kernels>, true>;

}  // namespace Stockfish::Eval::NNUE::HordeV2

#endif  // HORDE_V2_STACK_H_INCLUDED
