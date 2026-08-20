/*
  Horde-Stockfish, a UCI chess engine derived from Stockfish
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#ifndef HORDE_V3_STACK_H_INCLUDED
#define HORDE_V3_STACK_H_INCLUDED

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "horde_v3_backend.h"

// The incremental V3 accumulator stack.
//
// G0 rows follow ordinary DirtyPiece deltas. The six contextual blocks do not:
// a move can change the contextual role of a pawn that never moved, so a
// DirtyPiece alone does not determine the contextual row set. Every frame
// therefore carries the physical state those blocks are keyed on -- the White
// pawn and full occupancy bitboards, the per-file frontier and rearmost ranks,
// the per-file White pawn count and the three per-file predicate bits.
//
// On a transition the candidate file set is derived from every physically
// changed square, which covers castling and en passant and not only the nominal
// from/to pair, and is widened by one file on each side because
// frontier_supported and frontier_phalanx read the neighbouring files. Only
// those files are recomputed, the source and target contextual row sets are
// diffed, and exactly the removed and added rows are applied.
//
// Undo restores the saved source frame. It never re-derives an old contextual
// role from the DirtyPiece list, because the source roles are not a function of
// that list.

namespace Stockfish::Eval::NNUE::HordeV3 {

enum class V3StackError : std::uint8_t {
    NONE,
    INVALID_NETWORK,
    INVALID_POSITION,
    INVALID_DIRTY_PIECE,
    SOURCE_NOT_COMPUTED,
    STACK_UNINITIALIZED,
    STACK_OVERFLOW,
    WHITE_PAWN_ON_RANK_8,
    TOO_MANY_WHITE_PIECES,
    TOO_MANY_BLACK_PIECES,
    TOO_MANY_PIECES,
    BLACK_KING_COUNT,
    CONTEXTUAL_OVERFLOW,
};

constexpr const char* v3_stack_error_name(V3StackError error) noexcept {
    switch (error)
    {
    case V3StackError::NONE :
        return "NONE";
    case V3StackError::INVALID_NETWORK :
        return "INVALID_NETWORK";
    case V3StackError::INVALID_POSITION :
        return "INVALID_POSITION";
    case V3StackError::INVALID_DIRTY_PIECE :
        return "INVALID_DIRTY_PIECE";
    case V3StackError::SOURCE_NOT_COMPUTED :
        return "SOURCE_NOT_COMPUTED";
    case V3StackError::STACK_UNINITIALIZED :
        return "STACK_UNINITIALIZED";
    case V3StackError::STACK_OVERFLOW :
        return "STACK_OVERFLOW";
    case V3StackError::WHITE_PAWN_ON_RANK_8 :
        return "WHITE_PAWN_ON_RANK_8";
    case V3StackError::TOO_MANY_WHITE_PIECES :
        return "TOO_MANY_WHITE_PIECES";
    case V3StackError::TOO_MANY_BLACK_PIECES :
        return "TOO_MANY_BLACK_PIECES";
    case V3StackError::TOO_MANY_PIECES :
        return "TOO_MANY_PIECES";
    case V3StackError::BLACK_KING_COUNT :
        return "BLACK_KING_COUNT";
    case V3StackError::CONTEXTUAL_OVERFLOW :
        return "CONTEXTUAL_OVERFLOW";
    }
    return "UNKNOWN";
}

// The exact delta a transition applied. Kept so a parity failure can name the
// removed and added indices instead of only the resulting accumulator.
struct V3Transition {
    NormalizedDirty                              dirty{};
    std::array<IndexType, 4>                     g0Removed{};
    std::array<IndexType, 4>                     g0Added{};
    std::array<IndexType, MaxContextualRows>     removed{};
    std::array<IndexType, MaxContextualRows>     added{};
    std::size_t                                  g0RemovedSize = 0;
    std::size_t                                  g0AddedSize   = 0;
    std::size_t                                  removedSize   = 0;
    std::size_t                                  addedSize     = 0;
    std::uint8_t                                 changedFiles   = 0;  // one bit per file
    std::uint8_t                                 candidateFiles = 0;  // widened by one file
};

// Every square the transition physically touches, in removal-then-addition
// order. Castling writes four squares, en passant three.
inline std::size_t v3_changed_squares(const NormalizedDirty& dirty,
                                      std::array<Square, 4>& out) noexcept {
    std::size_t written = 0;
    out[written++]      = dirty.from;
    if (dirty.remove_sq != SQ_NONE)
        out[written++] = dirty.remove_sq;
    if (dirty.to != SQ_NONE)
        out[written++] = dirty.to;
    if (dirty.add_sq != SQ_NONE)
        out[written++] = dirty.add_sq;
    return written;
}

template<typename Kernels>
class V3AccumulatorStack {
   public:
    using Network = V3Network<Kernels>;
    using Frame   = V3AccumulatorFrame;
    using Scratch = V3DenseScratch;

    static constexpr std::size_t MaxSize = MAX_PLY + 1;

    explicit V3AccumulatorStack(const Network& network) :
        network_(network),
        slots_(MaxSize) {}

    void clear() noexcept { size_ = 0; }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    [[nodiscard]] const Frame* latest() const noexcept {
        return size_ == 0 ? nullptr : &slots_[size_ - 1].frame;
    }

    [[nodiscard]] const V3Transition* last_transition() const noexcept {
        return size_ <= 1 ? nullptr : &slots_[size_ - 1].transition;
    }

    [[nodiscard]] V3StackError reset(const std::array<Piece, SQUARE_NB>& board) noexcept {
        clear();
        if (!network_.valid())
            return V3StackError::INVALID_NETWORK;

        const V3Features features = extract_v3_features(board);
        if (!features.valid())
            return feature_error(features.error);

        slots_[0].transition = V3Transition{};
        network_.full_refresh(slots_[0].frame, features);
        size_ = 1;
        return V3StackError::NONE;
    }

    [[nodiscard]] V3StackError push(const DirtyPiece& raw) noexcept {
        if (size_ == 0)
            return V3StackError::STACK_UNINITIALIZED;
        if (size_ >= MaxSize)
            return V3StackError::STACK_OVERFLOW;

        NormalizedDirty dirty{};
        if (!normalize_dirty_piece(raw, dirty))
            return V3StackError::INVALID_DIRTY_PIECE;

        const Frame& source = slots_[size_ - 1].frame;
        if (!source.computed)
            return V3StackError::SOURCE_NOT_COMPUTED;

        Slot&              child = slots_[size_];
        const V3StackError error = materialize_child(child.frame, child.transition, source, dirty);
        if (error != V3StackError::NONE)
        {
            child.frame.computed = false;
            return error;
        }
        ++size_;
        return V3StackError::NONE;
    }

    // Undo. The parent frame was never modified, so restoring it is a pure pop:
    // no contextual role is ever re-derived from the DirtyPiece list.
    [[nodiscard]] bool pop() noexcept {
        if (size_ <= 1)
            return false;
        --size_;
        return true;
    }

    // A null move leaves the physical board untouched, so it reuses the top
    // frame and changes only the selected output row, bucket * 2 + stm, which
    // selects both the output head and the PSQT column.
    [[nodiscard]] V3EvalResult evaluate(Scratch& scratch, Color sideToMove, int rule50Count) const
      noexcept {
        if (size_ == 0 || !slots_[size_ - 1].frame.computed)
            return V3EvalResult{};
        return network_.propagate(slots_[size_ - 1].frame, scratch, sideToMove, rule50Count);
    }

    // One transition, exposed for tests and for callers that own their frames.
    [[nodiscard]] V3StackError materialize_child(Frame&                 child,
                                                 V3Transition&          transition,
                                                 const Frame&           source,
                                                 const NormalizedDirty& dirty) const noexcept {
        assert(&child != &source);

        transition       = V3Transition{};
        transition.dirty = dirty;

        child.accumulator     = source.accumulator;
        child.psqt            = source.psqt;
        child.pawns           = source.pawns;
        child.whitePieceCount = source.whitePieceCount;
        child.blackPieceCount = source.blackPieceCount;
        child.blackKings      = source.blackKings;
        child.computed        = false;

        // 1. G0 rows: ordinary remove/add deltas, removals before additions.
        if (!remove_piece(child, transition, dirty.pc, dirty.from))
            return V3StackError::INVALID_DIRTY_PIECE;
        if (dirty.remove_sq != SQ_NONE
            && !remove_piece(child, transition, dirty.remove_pc, dirty.remove_sq))
            return V3StackError::INVALID_DIRTY_PIECE;
        if (dirty.to != SQ_NONE && !add_piece(child, transition, dirty.pc, dirty.to))
            return V3StackError::INVALID_DIRTY_PIECE;
        if (dirty.add_sq != SQ_NONE && !add_piece(child, transition, dirty.add_pc, dirty.add_sq))
            return V3StackError::INVALID_DIRTY_PIECE;

        if (child.blackKings != 1)
            return V3StackError::BLACK_KING_COUNT;
        if (child.whitePieceCount > MaxWhitePieces)
            return V3StackError::TOO_MANY_WHITE_PIECES;
        if (child.blackPieceCount > MaxBlackPieces)
            return V3StackError::TOO_MANY_BLACK_PIECES;
        if (std::size_t(child.whitePieceCount) + child.blackPieceCount > MaxHordePieces)
            return V3StackError::TOO_MANY_PIECES;

        // 2. The candidate file set. Every physically changed square counts, so
        //    castling and en passant are covered, and each is widened by one
        //    file because the supported and phalanx predicates read neighbours.
        std::array<Square, 4> changed{};
        const std::size_t     changedSize = v3_changed_squares(dirty, changed);
        for (std::size_t index = 0; index < changedSize; ++index)
        {
            const int file = int(changed[index]) & 7;
            transition.changedFiles = std::uint8_t(transition.changedFiles | (1u << file));
            for (int neighbour = file - 1; neighbour <= file + 1; ++neighbour)
                if (neighbour >= 0 && neighbour < FileNb)
                    transition.candidateFiles =
                      std::uint8_t(transition.candidateFiles | (1u << neighbour));
        }

        // 3. The source contextual rows of every candidate file, read from the
        //    source frame's retained per-file state.
        std::array<IndexType, MaxRowsPerFile> buffer{};
        for (int file = 0; file < FileNb; ++file)
        {
            if (!file_bit(transition.candidateFiles, file))
                continue;
            const std::size_t written = v3_file_contextual_rows(source.pawns, file, buffer.data());
            for (std::size_t index = 0; index < written; ++index)
            {
                if (transition.removedSize >= MaxContextualRows)
                    return V3StackError::CONTEXTUAL_OVERFLOW;
                transition.removed[transition.removedSize++] = buffer[index];
            }
        }

        // 4. Recompute only the candidate files, ranks first and predicates
        //    second, because a predicate reads its own file's new frontier.
        for (int file = 0; file < FileNb; ++file)
            if (file_bit(transition.candidateFiles, file) && !v3_recompute_file(child.pawns, file))
                return V3StackError::WHITE_PAWN_ON_RANK_8;
        for (int file = 0; file < FileNb; ++file)
            if (file_bit(transition.candidateFiles, file))
                v3_recompute_predicates(child.pawns, file);

        // 5. The target contextual rows of the same files.
        for (int file = 0; file < FileNb; ++file)
        {
            if (!file_bit(transition.candidateFiles, file))
                continue;
            const std::size_t written = v3_file_contextual_rows(child.pawns, file, buffer.data());
            for (std::size_t index = 0; index < written; ++index)
            {
                if (transition.addedSize >= MaxContextualRows)
                    return V3StackError::CONTEXTUAL_OVERFLOW;
                transition.added[transition.addedSize++] = buffer[index];
            }
        }

        // 6. Cancel the rows both sets share, so only the exact difference is
        //    applied and the reported delta is the real one.
        cancel_common(transition);

        for (std::size_t index = 0; index < transition.removedSize; ++index)
            network_.subtract_row(child, transition.removed[index]);
        for (std::size_t index = 0; index < transition.addedSize; ++index)
            network_.add_row(child, transition.added[index]);

        child.computed = true;
        return V3StackError::NONE;
    }

   private:
    struct Slot {
        Frame        frame{};
        V3Transition transition{};
    };

    [[nodiscard]] static constexpr V3StackError feature_error(V3FeatureError error) noexcept {
        switch (error)
        {
        case V3FeatureError::WHITE_PAWN_ON_RANK_8 :
            return V3StackError::WHITE_PAWN_ON_RANK_8;
        case V3FeatureError::TOO_MANY_WHITE_PIECES :
            return V3StackError::TOO_MANY_WHITE_PIECES;
        case V3FeatureError::TOO_MANY_BLACK_PIECES :
            return V3StackError::TOO_MANY_BLACK_PIECES;
        case V3FeatureError::TOO_MANY_PIECES :
            return V3StackError::TOO_MANY_PIECES;
        case V3FeatureError::BLACK_KING_COUNT :
            return V3StackError::BLACK_KING_COUNT;
        default :
            return V3StackError::INVALID_POSITION;
        }
    }

    [[nodiscard]] bool
    remove_piece(Frame& frame, V3Transition& transition, Piece piece, Square square) const noexcept {
        const IndexType row = fixed_role_piece_square_index(piece, square);
        if (row == InvalidFeatureIndex || transition.g0RemovedSize >= transition.g0Removed.size())
            return false;

        network_.subtract_row(frame, row);
        transition.g0Removed[transition.g0RemovedSize++] = row;

        frame.pawns.occupancy &= ~square_bit(int(square));
        if (piece == W_PAWN)
            frame.pawns.whitePawns &= ~square_bit(int(square));
        if (color_of(piece) == WHITE)
            --frame.whitePieceCount;
        else
        {
            --frame.blackPieceCount;
            if (piece == B_KING)
                --frame.blackKings;
        }
        return true;
    }

    [[nodiscard]] bool
    add_piece(Frame& frame, V3Transition& transition, Piece piece, Square square) const noexcept {
        const IndexType row = fixed_role_piece_square_index(piece, square);
        if (row == InvalidFeatureIndex || transition.g0AddedSize >= transition.g0Added.size())
            return false;

        network_.add_row(frame, row);
        transition.g0Added[transition.g0AddedSize++] = row;

        frame.pawns.occupancy |= square_bit(int(square));
        if (piece == W_PAWN)
            frame.pawns.whitePawns |= square_bit(int(square));
        if (color_of(piece) == WHITE)
            ++frame.whitePieceCount;
        else
        {
            ++frame.blackPieceCount;
            if (piece == B_KING)
                ++frame.blackKings;
        }
        return true;
    }

    static void cancel_common(V3Transition& transition) noexcept {
        for (std::size_t removed = 0; removed < transition.removedSize;)
        {
            bool cancelled = false;
            for (std::size_t added = 0; added < transition.addedSize; ++added)
                if (transition.removed[removed] == transition.added[added])
                {
                    transition.added[added] = transition.added[--transition.addedSize];
                    transition.removed[removed] =
                      transition.removed[--transition.removedSize];
                    cancelled = true;
                    break;
                }
            if (!cancelled)
                ++removed;
        }
    }

    const Network&    network_;
    std::vector<Slot> slots_;
    std::size_t       size_ = 0;
};

using V3ScalarStack  = V3AccumulatorStack<V3ScalarKernels>;
using V3DefaultStack = V3AccumulatorStack<V3DefaultKernels>;

}  // namespace Stockfish::Eval::NNUE::HordeV3

#endif  // HORDE_V3_STACK_H_INCLUDED
