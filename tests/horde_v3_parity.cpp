/*
  Horde NNUE V3 parity gates.

  Gate 1  trainer to C++, layer by layer. Driven by --positions: the caller
          feeds the FENs of the trainer's layer trace and compares the emitted
          integers against it.
  Gate 2  scalar equals AVX2, every intermediate layer, same positions and the
          same container.
  Gate 3  incremental equals full refresh over generated legal moves, the
          special move shapes, Black castling on overlapping Chess960 layouts,
          every Black king move, null moves and the matching undos.

  The binary loads one V3 container, runs the gates it can run and prints one
  JSON object. A mismatch prints the FEN, the move, the block, both contextual
  states, the removed and added indices and both accumulator values, then
  aborts.
*/

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "nnue/horde_v2_stack.h"  // apply_normalized_dirty, the board primitive V3 mirrors
#include "nnue/horde_v3_container.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;
using namespace Stockfish::Eval::NNUE::HordeV3;

namespace {

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "Horde V3 parity failure: " << message << '\n';
    std::exit(EXIT_FAILURE);
}

void require(bool condition, const std::string& message) {
    if (!condition)
        fail(message);
}

// ---------------------------------------------------------------------------
// Boards and FEN
// ---------------------------------------------------------------------------

struct PositionState {
    std::array<Piece, SQUARE_NB> board{};
    Color                        sideToMove = WHITE;
    Square                       epSquare   = SQ_NONE;
    int                          rule50     = 0;
};

Piece piece_from_char(char value) {
    switch (value)
    {
    case 'P' :
        return W_PAWN;
    case 'N' :
        return W_KNIGHT;
    case 'B' :
        return W_BISHOP;
    case 'R' :
        return W_ROOK;
    case 'Q' :
        return W_QUEEN;
    case 'K' :
        return W_KING;
    case 'p' :
        return B_PAWN;
    case 'n' :
        return B_KNIGHT;
    case 'b' :
        return B_BISHOP;
    case 'r' :
        return B_ROOK;
    case 'q' :
        return B_QUEEN;
    case 'k' :
        return B_KING;
    default :
        return PIECE_NB;
    }
}

char char_from_piece(Piece piece) {
    switch (piece)
    {
    case W_PAWN :
        return 'P';
    case W_KNIGHT :
        return 'N';
    case W_BISHOP :
        return 'B';
    case W_ROOK :
        return 'R';
    case W_QUEEN :
        return 'Q';
    case W_KING :
        return 'K';
    case B_PAWN :
        return 'p';
    case B_KNIGHT :
        return 'n';
    case B_BISHOP :
        return 'b';
    case B_ROOK :
        return 'r';
    case B_QUEEN :
        return 'q';
    case B_KING :
        return 'k';
    default :
        return '.';
    }
}

std::string square_name(Square square) {
    if (square == SQ_NONE)
        return "-";
    std::string name;
    name.push_back(char('a' + (int(square) & 7)));
    name.push_back(char('1' + (int(square) >> 3)));
    return name;
}

// Piece placement, side to move, castling, en passant, halfmove clock.
// Castling rights are not modelled: the castling gate uses explicit DirtyPiece
// fixtures, exactly like the V2 container test.
bool parse_fen(std::string_view fen, PositionState& out) {
    out = PositionState{};
    std::vector<std::string> fields;
    std::string              current;
    for (const char value : fen)
    {
        if (value == ' ')
        {
            if (!current.empty())
                fields.push_back(current);
            current.clear();
        }
        else
            current.push_back(value);
    }
    if (!current.empty())
        fields.push_back(current);
    if (fields.empty())
        return false;

    int rank = 7;
    int file = 0;
    for (const char value : fields[0])
    {
        if (value == '/')
        {
            if (file != 8 || rank == 0)
                return false;
            --rank;
            file = 0;
            continue;
        }
        if (value >= '1' && value <= '8')
        {
            file += value - '0';
            if (file > 8)
                return false;
            continue;
        }
        const Piece piece = piece_from_char(value);
        if (piece == PIECE_NB || file >= 8)
            return false;
        out.board[rank * 8 + file] = piece;
        ++file;
    }
    if (file != 8 || rank != 0)
        return false;

    if (fields.size() > 1)
    {
        if (fields[1] == "w")
            out.sideToMove = WHITE;
        else if (fields[1] == "b")
            out.sideToMove = BLACK;
        else
            return false;
    }
    if (fields.size() > 3 && fields[3] != "-")
    {
        if (fields[3].size() != 2 || fields[3][0] < 'a' || fields[3][0] > 'h'
            || fields[3][1] < '1' || fields[3][1] > '8')
            return false;
        out.epSquare = Square((fields[3][1] - '1') * 8 + (fields[3][0] - 'a'));
    }
    if (fields.size() > 4)
        out.rule50 = std::atoi(fields[4].c_str());
    return true;
}

std::string board_fen(const std::array<Piece, SQUARE_NB>& board) {
    std::string fen;
    for (int rank = 7; rank >= 0; --rank)
    {
        int empty = 0;
        for (int file = 0; file < 8; ++file)
        {
            const Piece piece = board[rank * 8 + file];
            if (piece == NO_PIECE)
            {
                ++empty;
                continue;
            }
            if (empty)
            {
                fen += std::to_string(empty);
                empty = 0;
            }
            fen.push_back(char_from_piece(piece));
        }
        if (empty)
            fen += std::to_string(empty);
        if (rank)
            fen.push_back('/');
    }
    return fen;
}

std::string position_fen(const PositionState& position) {
    return board_fen(position.board) + (position.sideToMove == WHITE ? " w " : " b ") + "- "
         + square_name(position.epSquare) + " " + std::to_string(position.rule50) + " 1";
}

// ---------------------------------------------------------------------------
// Self-contained Horde move generation
//
// The V3 headers deliberately do not depend on Position, so this test carries
// its own generator. White has no king, so every pseudo-legal White move is
// legal; Black must not leave its king attacked.
// ---------------------------------------------------------------------------

struct GeneratedMove {
    DirtyPiece                   dirty{};
    std::array<Piece, SQUARE_NB> target{};
    std::string                  label;
    std::string                  kind;
};

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
    dirty.remove_pc = removePiece;
    dirty.remove_sq = removeSquare;
    dirty.add_pc    = addPiece;
    dirty.add_sq    = addSquare;
    return dirty;
}

bool attacks(const std::array<Piece, SQUARE_NB>& board, int from, int to) {
    if (from == to)
        return false;
    const Piece piece = board[from];
    const int   dr    = (to >> 3) - (from >> 3);
    const int   df    = (to & 7) - (from & 7);
    const int   adr   = dr < 0 ? -dr : dr;
    const int   adf   = df < 0 ? -df : df;

    switch (type_of(piece))
    {
    case PAWN :
        return adf == 1 && dr == (color_of(piece) == WHITE ? 1 : -1);
    case KNIGHT :
        return (adf == 1 && adr == 2) || (adf == 2 && adr == 1);
    case KING :
        return adf <= 1 && adr <= 1;
    case BISHOP :
    case ROOK :
    case QUEEN : {
        const bool diagonal = adf == adr;
        const bool straight = adf == 0 || adr == 0;
        if (type_of(piece) == BISHOP && !diagonal)
            return false;
        if (type_of(piece) == ROOK && !straight)
            return false;
        if (type_of(piece) == QUEEN && !diagonal && !straight)
            return false;
        const int stepRank = dr > 0 ? 1 : (dr < 0 ? -1 : 0);
        const int stepFile = df > 0 ? 1 : (df < 0 ? -1 : 0);
        const int step     = stepRank * 8 + stepFile;
        for (int square = from + step; square != to; square += step)
            if (board[square] != NO_PIECE)
                return false;
        return true;
    }
    default :
        return false;
    }
}

bool square_attacked(const std::array<Piece, SQUARE_NB>& board, int square, Color by) {
    for (int from = 0; from < SQUARE_NB; ++from)
    {
        const Piece piece = board[from];
        if (piece == NO_PIECE || color_of(piece) != by)
            continue;
        if (attacks(board, from, square))
            return true;
    }
    return false;
}

bool black_king_safe(const std::array<Piece, SQUARE_NB>& board) {
    for (int square = 0; square < SQUARE_NB; ++square)
        if (board[square] == B_KING)
            return !square_attacked(board, square, WHITE);
    return false;
}

// Build the target board with the same primitive the accumulator stack models:
// every removal is applied before every addition.
bool try_move(const std::array<Piece, SQUARE_NB>& source,
              const DirtyPiece&                   dirty,
              std::array<Piece, SQUARE_NB>&       target) {
    HordeV2::NormalizedDirty normalized{};
    if (!HordeV2::normalize_dirty_piece(dirty, normalized))
        return false;
    target = source;
    return HordeV2::apply_normalized_dirty(target, normalized);
}

void push_move(std::vector<GeneratedMove>&         moves,
               const std::array<Piece, SQUARE_NB>& board,
               const DirtyPiece&                   dirty,
               const std::string&                  label,
               const std::string&                  kind,
               Color                               us) {
    GeneratedMove move{};
    move.dirty = dirty;
    if (!try_move(board, dirty, move.target))
        return;
    if (us == BLACK && !black_king_safe(move.target))
        return;
    move.label = label;
    move.kind  = kind;
    moves.push_back(move);
}

std::vector<GeneratedMove> generate_legal_moves(const PositionState& position) {
    std::vector<GeneratedMove> moves;
    const auto&                board = position.board;
    const Color                us    = position.sideToMove;
    const Color                them  = ~us;

    for (int from = 0; from < SQUARE_NB; ++from)
    {
        const Piece piece = board[from];
        if (piece == NO_PIECE || color_of(piece) != us)
            continue;
        const std::string origin = square_name(Square(from));

        if (type_of(piece) == PAWN)
        {
            const int forward     = us == WHITE ? 8 : -8;
            const int promoteRank = us == WHITE ? 7 : 0;
            const int startRankA  = us == WHITE ? 0 : 6;  // Horde: a White pawn on
            const int startRankB  = us == WHITE ? 1 : 6;  // rank 1 may double push
            const int single      = from + forward;

            if (single >= 0 && single < SQUARE_NB && board[single] == NO_PIECE)
            {
                if ((single >> 3) == promoteRank)
                {
                    for (const PieceType promoted : {QUEEN, ROOK, BISHOP, KNIGHT})
                        push_move(moves, board,
                                  make_dirty(piece, Square(from), SQ_NONE, NO_PIECE, SQ_NONE,
                                             make_piece(us, promoted), Square(single)),
                                  origin + "-" + square_name(Square(single)) + "="
                                    + char_from_piece(make_piece(WHITE, promoted)),
                                  "promotion", us);
                }
                else
                {
                    push_move(moves, board,
                              make_dirty(piece, Square(from), Square(single)),
                              origin + "-" + square_name(Square(single)), "quiet", us);
                    const int doubleStep = single + forward;
                    if (((from >> 3) == startRankA || (from >> 3) == startRankB)
                        && doubleStep >= 0 && doubleStep < SQUARE_NB
                        && board[doubleStep] == NO_PIECE)
                        push_move(moves, board,
                                  make_dirty(piece, Square(from), Square(doubleStep)),
                                  origin + "-" + square_name(Square(doubleStep)), "double-push",
                                  us);
                }
            }

            for (const int step : {forward - 1, forward + 1})
            {
                const int to = from + step;
                if (to < 0 || to >= SQUARE_NB)
                    continue;
                const int fileDelta = (to & 7) - (from & 7);
                if (fileDelta != 1 && fileDelta != -1)
                    continue;

                if (board[to] != NO_PIECE && color_of(board[to]) == them)
                {
                    if ((to >> 3) == promoteRank)
                    {
                        for (const PieceType promoted : {QUEEN, KNIGHT})
                            push_move(moves, board,
                                      make_dirty(piece, Square(from), SQ_NONE, board[to],
                                                 Square(to), make_piece(us, promoted), Square(to)),
                                      origin + "x" + square_name(Square(to)) + "="
                                        + char_from_piece(make_piece(WHITE, promoted)),
                                      "promotion-capture", us);
                    }
                    else
                        push_move(moves, board,
                                  make_dirty(piece, Square(from), Square(to), board[to],
                                             Square(to)),
                                  origin + "x" + square_name(Square(to)), "capture", us);
                }
                else if (position.epSquare != SQ_NONE && to == int(position.epSquare)
                         && board[to] == NO_PIECE)
                {
                    const int captured = to - forward;
                    if (captured >= 0 && captured < SQUARE_NB && board[captured] != NO_PIECE
                        && color_of(board[captured]) == them
                        && type_of(board[captured]) == PAWN)
                        push_move(moves, board,
                                  make_dirty(piece, Square(from), Square(to), board[captured],
                                             Square(captured)),
                                  origin + "x" + square_name(Square(to)) + "ep", "en-passant", us);
                }
            }
            continue;
        }

        static const std::array<std::pair<int, int>, 8> KnightSteps = {
          {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}}};
        static const std::array<std::pair<int, int>, 8> KingSteps = {
          {{0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}}};

        const bool slider = type_of(piece) == BISHOP || type_of(piece) == ROOK
                         || type_of(piece) == QUEEN;
        const auto& steps = type_of(piece) == KNIGHT ? KnightSteps : KingSteps;

        for (const auto& [stepFile, stepRank] : steps)
        {
            if (slider)
            {
                const bool diagonal = stepFile != 0 && stepRank != 0;
                if (type_of(piece) == BISHOP && !diagonal)
                    continue;
                if (type_of(piece) == ROOK && diagonal)
                    continue;
            }

            int file = (from & 7) + stepFile;
            int rank = (from >> 3) + stepRank;
            while (file >= 0 && file < 8 && rank >= 0 && rank < 8)
            {
                const int to = rank * 8 + file;
                if (board[to] != NO_PIECE && color_of(board[to]) == us)
                    break;
                const bool capture = board[to] != NO_PIECE;
                const std::string label =
                  std::string(1, char_from_piece(make_piece(WHITE, type_of(piece)))) + origin
                  + (capture ? "x" : "-") + square_name(Square(to));
                push_move(moves, board,
                          capture ? make_dirty(piece, Square(from), Square(to), board[to],
                                               Square(to))
                                  : make_dirty(piece, Square(from), Square(to)),
                          label,
                          piece == B_KING ? "black-king"
                                          : (capture ? "capture" : "quiet"),
                          us);
                if (capture || !slider)
                    break;
                file += stepFile;
                rank += stepRank;
            }
        }
    }
    return moves;
}

// ---------------------------------------------------------------------------
// Row bookkeeping, so a mismatch can be attributed to a block
// ---------------------------------------------------------------------------

using RowCounts = std::array<int, V3StreamRows>;

RowCounts row_counts(const V3Features& features) {
    RowCounts counts{};
    for (std::size_t index = 0; index < features.size; ++index)
        ++counts[features.rows[index]];
    return counts;
}

std::string describe_pawns(const V3PawnStructure& pawns) {
    std::ostringstream text;
    for (int file = 0; file < FileNb; ++file)
    {
        text << char('a' + file) << ":count=" << int(pawns.count[file])
             << ",frontier=" << int(pawns.frontier[file])
             << ",rearmost=" << int(pawns.rearmost[file])
             << ",blocked=" << int(file_bit(pawns.blocked, file))
             << ",supported=" << int(file_bit(pawns.supported, file))
             << ",phalanx=" << int(file_bit(pawns.phalanx, file)) << (file == 7 ? "" : " | ");
    }
    return text.str();
}

std::string describe_dirty(const V3Transition& transition) {
    const HordeV2::NormalizedDirty& dirty = transition.dirty;
    std::ostringstream              text;
    text << "pc=" << char_from_piece(dirty.pc) << " from=" << square_name(dirty.from)
         << " to=" << square_name(dirty.to) << " remove=" << char_from_piece(dirty.remove_pc)
         << "@" << square_name(dirty.remove_sq) << " add=" << char_from_piece(dirty.add_pc) << "@"
         << square_name(dirty.add_sq);
    return text.str();
}

std::string describe_rows(const IndexType* rows, std::size_t size) {
    std::ostringstream text;
    text << '[';
    for (std::size_t index = 0; index < size; ++index)
        text << (index ? ", " : "") << rows[index] << '(' << v3_row_block_name(rows[index]) << ')';
    text << ']';
    return text.str();
}

// ---------------------------------------------------------------------------
// Gate 3
// ---------------------------------------------------------------------------

struct Gate3Counters {
    std::size_t positions            = 0;
    std::size_t generatedMoves       = 0;
    std::size_t specialMoves         = 0;
    std::size_t undos                = 0;
    std::size_t nullMoves            = 0;
    std::size_t blackKingMoves       = 0;
    std::size_t castlingMoves        = 0;
    std::size_t enPassantMoves       = 0;
    std::size_t promotions           = 0;
    std::size_t promotionCaptures    = 0;
    std::size_t captures             = 0;
    std::size_t contextualDeltas     = 0;
    std::size_t neighbourOnlyFiles   = 0;
    std::size_t maxActiveRows        = 0;
    std::size_t maxContextualRows    = 0;
    std::size_t deepSequences        = 0;
};

template<typename Kernels>
class Gate3Runner {
   public:
    using Stack = V3AccumulatorStack<Kernels>;

    explicit Gate3Runner(const V3Network<Kernels>& network) :
        network_(network),
        stack_(network) {}

    Gate3Counters& counters() noexcept { return counters_; }

    void verify_position(const PositionState& position, const std::string& name) {
        const V3Features features = extract_v3_features(position.board);
        require(features.valid(), name + ": fixture board rejected by the enumerator ("
                                    + v3_feature_error_name(features.error) + ")");
        ++counters_.positions;

        require(stack_.reset(position.board) == V3StackError::NONE,
                name + ": stack reset rejected the fixture");

        const std::vector<GeneratedMove> moves = generate_legal_moves(position);
        for (const GeneratedMove& move : moves)
        {
            verify_transition(position.board, move.dirty, move.target,
                              name + " " + move.label, move.kind, position);
            ++counters_.generatedMoves;
            if (move.kind == "black-king")
                ++counters_.blackKingMoves;
            else if (move.kind == "en-passant")
                ++counters_.enPassantMoves;
            else if (move.kind == "promotion")
                ++counters_.promotions;
            else if (move.kind == "promotion-capture")
                ++counters_.promotionCaptures;
            else if (move.kind == "capture")
                ++counters_.captures;
        }

        verify_null_move(position, name);
        verify_sequence(position, name);
    }

    void verify_special(const std::array<Piece, SQUARE_NB>& board,
                        const DirtyPiece&                   dirty,
                        const std::string&                  name,
                        const std::string&                  kind) {
        std::array<Piece, SQUARE_NB> target{};
        require(try_move(board, dirty, target), name + ": fixture transition is not applicable");
        PositionState position{};
        position.board      = board;
        position.sideToMove = WHITE;
        verify_transition(board, dirty, target, name, kind, position);
        ++counters_.specialMoves;
        if (kind == "castling")
            ++counters_.castlingMoves;
    }

   private:
    // One transition. The child frame is compared against a fresh full refresh
    // of the target board, then popped and compared against the source frame.
    void verify_transition(const std::array<Piece, SQUARE_NB>& sourceBoard,
                           const DirtyPiece&                   dirty,
                           const std::array<Piece, SQUARE_NB>& targetBoard,
                           const std::string&                  name,
                           const std::string&                  kind,
                           const PositionState&                position) {
        require(stack_.reset(sourceBoard) == V3StackError::NONE, name + ": source reset failed");
        const V3AccumulatorFrame source = *stack_.latest();

        const V3StackError error = stack_.push(dirty);
        require(error == V3StackError::NONE,
                name + ": incremental push failed with " + v3_stack_error_name(error));

        const V3AccumulatorFrame& child       = *stack_.latest();
        const V3Transition&       transition  = *stack_.last_transition();
        const V3Features          targetRows  = extract_v3_features(targetBoard);
        require(targetRows.valid(), name + ": target board rejected by the enumerator");

        V3AccumulatorFrame refreshed{};
        network_.full_refresh(refreshed, targetRows);

        compare_frames(child, refreshed, sourceBoard, targetBoard, dirty, transition, source,
                       targetRows, name);

        // The dense trunk must agree too, on both sides to move.
        for (const Color side : {WHITE, BLACK})
        {
            V3DenseScratch incrementalScratch{};
            V3DenseScratch refreshScratch{};
            const V3EvalResult incremental =
              network_.propagate(child, incrementalScratch, side, position.rule50);
            const V3EvalResult full =
              network_.propagate(refreshed, refreshScratch, side, position.rule50);
            require(incrementalScratch.transformed == refreshScratch.transformed,
                    name + ": transformed differs");
            require(incrementalScratch.hidden0Affine == refreshScratch.hidden0Affine,
                    name + ": h0 affine differs");
            require(incrementalScratch.hidden0 == refreshScratch.hidden0,
                    name + ": h0 activation differs");
            require(incrementalScratch.hidden1Affine == refreshScratch.hidden1Affine,
                    name + ": h1 affine differs");
            require(incrementalScratch.hidden1 == refreshScratch.hidden1,
                    name + ": h1 activation differs");
            require(incremental.bucket == full.bucket, name + ": bucket differs");
            require(incremental.outputAffine == full.outputAffine, name + ": out_pre differs");
            require(incremental.psqtSum == full.psqtSum, name + ": psqt_sum differs");
            require(incremental.value == full.value, name + ": value differs");
        }

        counters_.maxActiveRows = std::max(counters_.maxActiveRows, targetRows.size);
        counters_.maxContextualRows =
          std::max(counters_.maxContextualRows, targetRows.size - targetRows.g0Size);
        if (transition.removedSize || transition.addedSize)
            ++counters_.contextualDeltas;
        count_neighbour_only(transition);
        (void) kind;

        // Undo. The parent frame must come back bit-identical without ever
        // re-deriving a contextual role from the DirtyPiece list.
        require(stack_.pop(), name + ": pop rejected");
        const V3AccumulatorFrame& restored = *stack_.latest();
        require(restored.accumulator == source.accumulator, name + ": undo accumulator differs");
        require(restored.psqt == source.psqt, name + ": undo psqt differs");
        require(restored.pawns == source.pawns, name + ": undo contextual state differs");
        require(restored.whitePieceCount == source.whitePieceCount,
                name + ": undo white_piece_count differs");
        ++counters_.undos;
    }

    // A contextual row that changed on a file no changed square sits on. That
    // is the case a from/to derived candidate set would miss.
    void count_neighbour_only(const V3Transition& transition) {
        std::array<bool, FileNb> touched{};
        for (std::size_t index = 0; index < transition.removedSize; ++index)
            touched[contextual_file(transition.removed[index])] = true;
        for (std::size_t index = 0; index < transition.addedSize; ++index)
            touched[contextual_file(transition.added[index])] = true;
        for (int file = 0; file < FileNb; ++file)
            if (touched[file] && !file_bit(transition.changedFiles, file))
                ++counters_.neighbourOnlyFiles;
    }

    static std::size_t contextual_file(IndexType row) {
        if (row >= ContextualBlockBase[FILE_COUNT]
            && row < ContextualBlockBase[FILE_COUNT] + ContextualBlockWidth[FILE_COUNT])
            return (row - ContextualBlockBase[FILE_COUNT]) / FileCountStates;
        if (row >= ContextualBlockBase[FRONTIER_BLOCKED])
            return (row - ContextualBlockBase[FRONTIER_BLOCKED]) % 8;
        if (row >= ContextualBlockBase[FRONTIER_SQUARE])
            return (row - ContextualBlockBase[FRONTIER_SQUARE]) % 8;
        return 0;
    }

    void compare_frames(const V3AccumulatorFrame&           child,
                        const V3AccumulatorFrame&           refreshed,
                        const std::array<Piece, SQUARE_NB>& sourceBoard,
                        const std::array<Piece, SQUARE_NB>& targetBoard,
                        const DirtyPiece&                   dirty,
                        const V3Transition&                 transition,
                        const V3AccumulatorFrame&           source,
                        const V3Features&                   targetRows,
                        const std::string&                  name) {
        if (child.accumulator == refreshed.accumulator && child.psqt == refreshed.psqt
            && child.pawns == refreshed.pawns && child.whitePieceCount == refreshed.whitePieceCount
            && child.blackPieceCount == refreshed.blackPieceCount)
            return;

        std::cerr << "\nHorde V3 incremental/full-refresh mismatch in " << name << "\n";
        std::cerr << "  source FEN : " << board_fen(sourceBoard) << "\n";
        std::cerr << "  target FEN : " << board_fen(targetBoard) << "\n";
        std::cerr << "  move       : " << describe_dirty(transition) << " (raw pc="
                  << char_from_piece(dirty.pc) << ")\n";

        const V3Features sourceRows = extract_v3_features(sourceBoard);
        const RowCounts  expected   = row_counts(targetRows);
        RowCounts        actual{};
        for (std::size_t index = 0; index < sourceRows.size; ++index)
            ++actual[sourceRows.rows[index]];
        for (std::size_t index = 0; index < transition.g0RemovedSize; ++index)
            --actual[transition.g0Removed[index]];
        for (std::size_t index = 0; index < transition.removedSize; ++index)
            --actual[transition.removed[index]];
        for (std::size_t index = 0; index < transition.g0AddedSize; ++index)
            ++actual[transition.g0Added[index]];
        for (std::size_t index = 0; index < transition.addedSize; ++index)
            ++actual[transition.added[index]];

        for (std::size_t row = 0; row < V3StreamRows; ++row)
            if (expected[row] != actual[row])
                std::cerr << "  row " << row << " (" << v3_row_block_name(IndexType(row))
                          << ") full=" << expected[row] << " incremental=" << actual[row] << "\n";

        std::cerr << "  source contextual state: " << describe_pawns(source.pawns) << "\n";
        std::cerr << "  target contextual state: " << describe_pawns(refreshed.pawns) << "\n";
        std::cerr << "  incremental contextual : " << describe_pawns(child.pawns) << "\n";
        std::cerr << "  removed rows: " << describe_rows(transition.removed.data(),
                                                          transition.removedSize)
                  << "\n";
        std::cerr << "  added rows  : "
                  << describe_rows(transition.added.data(), transition.addedSize) << "\n";
        std::cerr << "  g0 removed  : "
                  << describe_rows(transition.g0Removed.data(), transition.g0RemovedSize) << "\n";
        std::cerr << "  g0 added    : "
                  << describe_rows(transition.g0Added.data(), transition.g0AddedSize) << "\n";

        for (std::size_t lane = 0; lane < V3Lanes; ++lane)
            if (child.accumulator[lane] != refreshed.accumulator[lane])
            {
                std::cerr << "  first differing accumulator lane " << lane
                          << ": incremental=" << child.accumulator[lane]
                          << " full=" << refreshed.accumulator[lane] << "\n";
                break;
            }
        for (std::size_t column = 0; column < V3PsqtColumns; ++column)
            if (child.psqt[column] != refreshed.psqt[column])
            {
                std::cerr << "  first differing psqt column " << column
                          << ": incremental=" << child.psqt[column]
                          << " full=" << refreshed.psqt[column] << "\n";
                break;
            }
        fail(name + ": incremental frame differs from a full refresh");
    }

    // A null move leaves the board untouched, so nothing but the selected
    // output row bucket*2+stm may change.
    void verify_null_move(const PositionState& position, const std::string& name) {
        require(stack_.reset(position.board) == V3StackError::NONE, name + ": null reset failed");
        V3DenseScratch first{};
        V3DenseScratch second{};
        const V3EvalResult before =
          stack_.evaluate(first, position.sideToMove, position.rule50);
        const V3EvalResult after = stack_.evaluate(second, ~position.sideToMove, position.rule50);
        require(first.transformed == second.transformed, name + ": null move changed transformed");
        require(first.hidden0Affine == second.hidden0Affine, name + ": null move changed h0");
        require(first.hidden1Affine == second.hidden1Affine, name + ": null move changed h1");
        require(before.bucket == after.bucket, name + ": null move changed the bucket");
        require(before.outputRow != after.outputRow, name + ": null move kept the output row");
        ++counters_.nullMoves;
    }

    // A deterministic legal walk, so the stack is exercised past one ply and
    // every pop is checked against the frame it must restore.
    void verify_sequence(const PositionState& position, const std::string& name) {
        PositionState current = position;
        require(stack_.reset(current.board) == V3StackError::NONE, name + ": walk reset failed");

        std::vector<V3AccumulatorFrame> expected;
        expected.push_back(*stack_.latest());
        std::uint64_t state = 0x9E3779B97F4A7C15ULL ^ std::uint64_t(name.size());

        for (int ply = 0; ply < 8; ++ply)
        {
            const std::vector<GeneratedMove> moves = generate_legal_moves(current);
            if (moves.empty())
                break;
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            const GeneratedMove& move = moves[state % moves.size()];
            if (stack_.push(move.dirty) != V3StackError::NONE)
                break;

            V3AccumulatorFrame refreshed{};
            const V3Features   rows = extract_v3_features(move.target);
            require(rows.valid(), name + ": walk produced an invalid board");
            network_.full_refresh(refreshed, rows);
            const V3AccumulatorFrame& top = *stack_.latest();
            require(top.accumulator == refreshed.accumulator, name + ": walk accumulator differs");
            require(top.psqt == refreshed.psqt, name + ": walk psqt differs");
            require(top.pawns == refreshed.pawns, name + ": walk contextual state differs");
            expected.push_back(top);

            current.board      = move.target;
            current.sideToMove = ~current.sideToMove;
            current.epSquare   = SQ_NONE;
            ++counters_.deepSequences;
        }

        while (stack_.size() > 1)
        {
            require(stack_.pop(), name + ": walk pop rejected");
            expected.pop_back();
            const V3AccumulatorFrame& top = *stack_.latest();
            require(top.accumulator == expected.back().accumulator,
                    name + ": walk undo accumulator differs");
            require(top.pawns == expected.back().pawns, name + ": walk undo contextual differs");
            ++counters_.undos;
        }
    }

    const V3Network<Kernels>& network_;
    Stack                     stack_;
    Gate3Counters             counters_{};
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

struct Fixture {
    const char* name;
    const char* fen;
};

constexpr std::array<Fixture, 10> Fixtures = {{
  {"horde-start-white", "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w - - 0 1"},
  {"horde-start-black", "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP b - - 3 2"},
  {"midgame-white", "r2qkb1r/pp1n1ppp/4pn2/1PP5/P1PP1P2/2P1P3/PP1PP1PP/1P4P1 w - - 5 12"},
  {"midgame-black", "r2qkb1r/pp1n1ppp/4pn2/1PP5/P1PP1P2/2P1P3/PP1PP1PP/1P4P1 b - - 5 12"},
  {"stacked-files", "4k3/PP4P1/PP4P1/PP4P1/PP4P1/PP4P1/PP4P1/PP4P1 w - - 0 1"},
  {"en-passant-white", "4k3/8/8/3pP3/8/8/PPP5/8 w - d6 0 9"},
  {"en-passant-black", "4k3/8/8/8/2Pp4/8/PP6/8 b - c3 0 9"},
  {"promotion-ready", "r3k2r/PPP3P1/8/8/8/2n5/PP1PP3/8 w - - 0 1"},
  {"sparse-endgame", "8/5k2/8/2P5/8/1P6/P7/8 w - - 12 40"},
  {"king-centre", "8/8/3PPP2/3PkP2/3PPP2/8/8/8 b - - 0 1"},
}};

// Black castling, including the overlapping Chess960 layouts the V2 container
// test covers. White pawns stand on rank 7 so a rook landing on rank 8 changes
// frontier_blocked for a file the king never touched.
struct CastlingFixture {
    const char* name;
    const char* fen;
    Piece       pc;
    Square      from;
    Square      to;
    Piece       removePc;
    Square      removeSq;
    Piece       addPc;
    Square      addSq;
};

constexpr std::array<CastlingFixture, 5> CastlingFixtures = {{
  {"castle-orthodox", "r3k2r/PPPPPPPP/8/8/8/8/8/8 b - - 0 1", B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_H8,
   B_ROOK, SQ_F8},
  {"castle-king-to-rook", "r3k1r1/PPPPPPPP/8/8/8/8/8/8 b - - 0 1", B_KING, SQ_E8, SQ_G8, B_ROOK,
   SQ_G8, B_ROOK, SQ_F8},
  {"castle-rook-to-king", "r4k1r/PPPPPPPP/8/8/8/8/8/8 b - - 0 1", B_KING, SQ_F8, SQ_G8, B_ROOK,
   SQ_H8, B_ROOK, SQ_F8},
  {"castle-zero-king", "r5kr/PPPPPPPP/8/8/8/8/8/8 b - - 0 1", B_KING, SQ_G8, SQ_G8, B_ROOK, SQ_H8,
   B_ROOK, SQ_F8},
  {"castle-zero-rook", "r3kr2/PPPPPPPP/8/8/8/8/8/8 b - - 0 1", B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_F8,
   B_ROOK, SQ_F8},
}};

// Transitions whose contextual effect is not on the file that moved.
struct ContextualFixture {
    const char* name;
    const char* fen;
    Piece       pc;
    Square      from;
    Square      to;
    Piece       removePc;
    Square      removeSq;
    Piece       addPc;
    Square      addSq;
};

constexpr std::array<ContextualFixture, 4> ContextualFixtures = {{
  // A Black knight captures the frontier pawn of the d file: the file keeps
  // pawns, so frontier, blocked, supported and phalanx all move without any
  // pawn having moved.
  {"frontier-captured", "4k3/8/2n5/3P4/3P4/3P4/2PPP3/8 b - - 0 1", B_KNIGHT, SQ_C6, SQ_D4, W_PAWN,
   SQ_D4, NO_PIECE, SQ_NONE},
  // A White pawn push on the c file changes phalanx and supported for b and d.
  {"neighbour-phalanx", "4k3/8/8/8/1P1P4/2P5/8/8 w - - 0 1", W_PAWN, SQ_C3, SQ_C4, NO_PIECE,
   SQ_NONE, NO_PIECE, SQ_NONE},
  // A promotion empties the frontier of the b file entirely.
  {"promotion-empties-file", "4k3/1P6/8/8/8/8/P7/8 w - - 0 1", W_PAWN, SQ_B7, SQ_NONE, NO_PIECE,
   SQ_NONE, W_QUEEN, SQ_B8},
  // A Black rook lands on b8 and blocks the frontier pawn on b7.
  {"rank8-blocks-frontier", "r3k3/1P6/8/8/8/8/8/8 b - - 0 1", B_ROOK, SQ_A8, SQ_B8, NO_PIECE,
   SQ_NONE, NO_PIECE, SQ_NONE},
}};

// ---------------------------------------------------------------------------
// Gate 2 and the position trace
// ---------------------------------------------------------------------------

struct LayerTrace {
    std::string                                fen;
    Color                                      sideToMove = WHITE;
    int                                        rule50     = 0;
    std::size_t                                whitePieceCount = 0;
    std::size_t                                bucket          = 0;
    std::vector<IndexType>                     rows;
    std::array<Accumulator, 16>                accumulatorHead{};
    std::array<Accumulator, V3Hidden0Lanes>    hidden0Affine{};
    std::array<Activation, V3Hidden0Lanes>     hidden0{};
    std::array<Accumulator, V3Hidden1Lanes>    hidden1Affine{};
    std::array<Activation, V3Hidden1Lanes>     hidden1{};
    Accumulator                                outputAffine = 0;
    Accumulator                                psqtSum      = 0;
    i32                                        preRule50    = 0;
    int                                        value        = 0;
    std::array<Activation, 16>                 transformedHead{};
};

template<typename Kernels>
LayerTrace trace_position(const V3Network<Kernels>& network, const PositionState& position) {
    const V3Features features = extract_v3_features(position.board);
    require(features.valid(), std::string("trace position rejected: ")
                                + v3_feature_error_name(features.error));

    V3AccumulatorFrame frame{};
    V3DenseScratch     scratch{};
    network.full_refresh(frame, features);
    const V3EvalResult result = network.propagate(frame, scratch, position.sideToMove,
                                                  position.rule50);

    LayerTrace trace{};
    trace.fen             = position_fen(position);
    trace.sideToMove      = position.sideToMove;
    trace.rule50          = position.rule50;
    trace.whitePieceCount = features.whitePieceCount;
    trace.bucket          = result.bucket;
    trace.rows.assign(features.rows.begin(), features.rows.begin() + features.size);
    std::sort(trace.rows.begin(), trace.rows.end());
    for (std::size_t lane = 0; lane < 16; ++lane)
    {
        trace.accumulatorHead[lane] = frame.accumulator[lane];
        trace.transformedHead[lane] = scratch.transformed[lane];
    }
    trace.hidden0Affine = scratch.hidden0Affine;
    trace.hidden0       = scratch.hidden0;
    trace.hidden1Affine = scratch.hidden1Affine;
    trace.hidden1       = scratch.hidden1;
    trace.outputAffine  = result.outputAffine;
    trace.psqtSum       = result.psqtSum;
    trace.preRule50     = result.preRule50Value;
    trace.value         = int(result.value);
    return trace;
}

// Gate 2: the two kernels must agree on every intermediate value, not only on
// the final one.
std::size_t verify_scalar_equals_avx2(const V3Parameters&              parameters,
                                      const std::vector<PositionState>& positions) {
    const V3Network<V3ScalarKernels>  scalar(parameters);
    const V3Network<V3DefaultKernels> selected(parameters);
    std::size_t                       checked = 0;

    for (const PositionState& position : positions)
    {
        const V3Features features = extract_v3_features(position.board);
        require(features.valid(), "gate 2 fixture rejected by the enumerator");

        V3AccumulatorFrame scalarFrame{};
        V3AccumulatorFrame selectedFrame{};
        scalar.full_refresh(scalarFrame, features);
        selected.full_refresh(selectedFrame, features);
        require(scalarFrame.accumulator == selectedFrame.accumulator,
                "gate 2: accumulator differs after a full refresh");
        require(scalarFrame.psqt == selectedFrame.psqt, "gate 2: psqt differs after a full refresh");

        for (const Color side : {WHITE, BLACK})
        {
            V3DenseScratch scalarScratch{};
            V3DenseScratch selectedScratch{};
            const V3EvalResult scalarResult =
              scalar.propagate(scalarFrame, scalarScratch, side, position.rule50);
            const V3EvalResult selectedResult =
              selected.propagate(selectedFrame, selectedScratch, side, position.rule50);
            require(scalarScratch.transformed == selectedScratch.transformed,
                    "gate 2: transformed differs");
            require(scalarScratch.hidden0Affine == selectedScratch.hidden0Affine,
                    "gate 2: h0 affine differs");
            require(scalarScratch.hidden0 == selectedScratch.hidden0,
                    "gate 2: h0 activation differs");
            require(scalarScratch.hidden1Affine == selectedScratch.hidden1Affine,
                    "gate 2: h1 affine differs");
            require(scalarScratch.hidden1 == selectedScratch.hidden1,
                    "gate 2: h1 activation differs");
            require(scalarResult.bucket == selectedResult.bucket, "gate 2: bucket differs");
            require(scalarResult.outputAffine == selectedResult.outputAffine,
                    "gate 2: out_pre differs");
            require(scalarResult.psqtSum == selectedResult.psqtSum, "gate 2: psqt_sum differs");
            require(scalarResult.preRule50Value == selectedResult.preRule50Value,
                    "gate 2: pre-rule50 differs");
            require(scalarResult.value == selectedResult.value, "gate 2: value differs");
            ++checked;
        }

        // The incremental paths must agree kernel for kernel as well.
        V3AccumulatorStack<V3ScalarKernels>  scalarStack(scalar);
        V3AccumulatorStack<V3DefaultKernels> selectedStack(selected);
        require(scalarStack.reset(position.board) == V3StackError::NONE, "gate 2: scalar reset");
        require(selectedStack.reset(position.board) == V3StackError::NONE, "gate 2: selected reset");
        for (const GeneratedMove& move : generate_legal_moves(position))
        {
            require(scalarStack.reset(position.board) == V3StackError::NONE, "gate 2: reset");
            require(selectedStack.reset(position.board) == V3StackError::NONE, "gate 2: reset");
            require(scalarStack.push(move.dirty) == V3StackError::NONE, "gate 2: scalar push");
            require(selectedStack.push(move.dirty) == V3StackError::NONE, "gate 2: selected push");
            require(scalarStack.latest()->accumulator == selectedStack.latest()->accumulator,
                    "gate 2: incremental accumulator differs");
            require(scalarStack.latest()->psqt == selectedStack.latest()->psqt,
                    "gate 2: incremental psqt differs");
        }
    }
    return checked;
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

template<typename T, std::size_t Size>
void emit_array(std::ostream& out, const std::array<T, Size>& values) {
    out << '[';
    for (std::size_t index = 0; index < Size; ++index)
        out << (index ? "," : "") << i64(values[index]);
    out << ']';
}

void emit_rows(std::ostream& out, const std::vector<IndexType>& rows) {
    out << '[';
    for (std::size_t index = 0; index < rows.size(); ++index)
        out << (index ? "," : "") << rows[index];
    out << ']';
}

void emit_trace(std::ostream& out, const LayerTrace& trace) {
    out << "{\"fen\":\"" << trace.fen << "\",\"side_to_move\":" << int(trace.sideToMove)
        << ",\"rule50\":" << trace.rule50
        << ",\"white_piece_count\":" << trace.whitePieceCount << ",\"bucket\":" << trace.bucket
        << ",\"active_rows\":";
    emit_rows(out, trace.rows);
    out << ",\"accumulator_head\":";
    emit_array(out, trace.accumulatorHead);
    out << ",\"transformed_head\":";
    emit_array(out, trace.transformedHead);
    out << ",\"hidden0_affine\":";
    emit_array(out, trace.hidden0Affine);
    out << ",\"hidden0\":";
    emit_array(out, trace.hidden0);
    out << ",\"hidden1_affine\":";
    emit_array(out, trace.hidden1Affine);
    out << ",\"hidden1\":";
    emit_array(out, trace.hidden1);
    out << ",\"out_pre\":" << trace.outputAffine << ",\"psqt_sum\":" << trace.psqtSum
        << ",\"pre_rule50\":" << trace.preRule50 << ",\"value\":" << trace.value << '}';
}

std::vector<PositionState> fixture_positions() {
    std::vector<PositionState> positions;
    for (const Fixture& fixture : Fixtures)
    {
        PositionState position{};
        require(parse_fen(fixture.fen, position),
                std::string("fixture FEN did not parse: ") + fixture.name);
        positions.push_back(position);
    }
    return positions;
}

}  // namespace

int main(int argc, char** argv) {
    std::string networkPath;
    std::string positionsPath;
    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--network" && index + 1 < argc)
            networkPath = argv[++index];
        else if (argument == "--positions" && index + 1 < argc)
            positionsPath = argv[++index];
        else
        {
            std::cerr << "usage: horde-v3-parity --network FILE [--positions FILE]\n";
            return 2;
        }
    }
    if (networkPath.empty())
    {
        std::cerr << "usage: horde-v3-parity --network FILE [--positions FILE]\n";
        return 2;
    }

    V3ContainerLoadResult loaded = load_v3_container(networkPath);
    if (!loaded)
    {
        std::cerr << v3_container_load_error_name(loaded.error) << ": " << loaded.message << '\n';
        return 3;
    }
    const V3Parameters& parameters = loaded.parameters;

    const std::vector<PositionState> positions = fixture_positions();

    // Gate 2
    const std::size_t gate2Checks = verify_scalar_equals_avx2(parameters, positions);

    // Gate 3
    const V3Network<V3DefaultKernels> network(parameters);
    Gate3Runner<V3DefaultKernels>     runner(network);
    for (std::size_t index = 0; index < positions.size(); ++index)
        runner.verify_position(positions[index], Fixtures[index].name);

    for (const CastlingFixture& fixture : CastlingFixtures)
    {
        PositionState position{};
        require(parse_fen(fixture.fen, position),
                std::string("castling FEN did not parse: ") + fixture.name);
        runner.verify_special(position.board,
                              make_dirty(fixture.pc, fixture.from, fixture.to, fixture.removePc,
                                         fixture.removeSq, fixture.addPc, fixture.addSq),
                              fixture.name, "castling");
    }
    for (const ContextualFixture& fixture : ContextualFixtures)
    {
        PositionState position{};
        require(parse_fen(fixture.fen, position),
                std::string("contextual FEN did not parse: ") + fixture.name);
        runner.verify_special(position.board,
                              make_dirty(fixture.pc, fixture.from, fixture.to, fixture.removePc,
                                         fixture.removeSq, fixture.addPc, fixture.addSq),
                              fixture.name, "contextual");
    }

    // Gate 1 material: the requested positions, traced layer by layer.
    std::vector<LayerTrace> traces;
    if (!positionsPath.empty())
    {
        std::ifstream input(positionsPath);
        if (!input)
        {
            std::cerr << "cannot open position list " << positionsPath << '\n';
            return 4;
        }
        std::string line;
        while (std::getline(input, line))
        {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n'))
                line.pop_back();
            if (line.empty())
                continue;
            PositionState position{};
            require(parse_fen(line, position), "position list contains an unparsable FEN: " + line);
            traces.push_back(trace_position(network, position));
        }
    }

    const Gate3Counters& counters = runner.counters();
    require(counters.generatedMoves >= 192,
            "gate 3 generated only " + std::to_string(counters.generatedMoves)
              + " legal moves, the contract requires at least 192");

    std::ostream& out = std::cout;
    out << "{\"schema\":\"HORDE_V3_PARITY_ORACLE_V1\""
        << ",\"backend\":\"" << V3DefaultBackendName << "\""
        << ",\"avx2_available\":" << (V3Avx2Available ? "true" : "false")
        << ",\"network\":{\"schema\":\"" << parameters.schemaName << "\",\"file_sha256\":\""
        << parameters.fileSha256 << "\",\"parameter_sha256\":\"" << parameters.parameterSha256
        << "\",\"training_architecture_structural_sha256\":\"" << loaded.trainingStructuralSha256
        << "\"}"
        << ",\"gate2\":{\"status\":\"" << (V3Avx2Available ? "pass" : "skipped")
        << "\",\"reason\":\""
        << (V3Avx2Available ? "" : "binary compiled without USE_AVX2, both kernels are scalar")
        << "\",\"propagations\":" << gate2Checks << ",\"positions\":" << positions.size() << "}"
        << ",\"gate3\":{\"status\":\"pass\""
        << ",\"positions\":" << counters.positions
        << ",\"generated_legal_moves\":" << counters.generatedMoves
        << ",\"special_transitions\":" << counters.specialMoves
        << ",\"castling_transitions\":" << counters.castlingMoves
        << ",\"en_passant_moves\":" << counters.enPassantMoves
        << ",\"promotions\":" << counters.promotions
        << ",\"promotion_captures\":" << counters.promotionCaptures
        << ",\"captures\":" << counters.captures
        << ",\"black_king_moves\":" << counters.blackKingMoves
        << ",\"null_moves\":" << counters.nullMoves << ",\"undos\":" << counters.undos
        << ",\"walk_plies\":" << counters.deepSequences
        << ",\"transitions_with_contextual_delta\":" << counters.contextualDeltas
        << ",\"neighbour_only_file_updates\":" << counters.neighbourOnlyFiles
        << ",\"max_active_rows\":" << counters.maxActiveRows
        << ",\"max_contextual_rows\":" << counters.maxContextualRows << "}"
        << ",\"traces\":[";
    for (std::size_t index = 0; index < traces.size(); ++index)
    {
        if (index)
            out << ',';
        emit_trace(out, traces[index]);
    }
    out << "]}\n";
    return 0;
}
