/*
  Trained Horde V2 container and full-refresh parity oracle.
*/

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "nnue/horde_v2_container.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE::HordeV2;

namespace {

static_assert(std::is_constructible_v<ContainerNetwork<>, const ContainerParameters&>);
static_assert(!std::is_constructible_v<ContainerNetwork<>, ContainerParameters&&>);
static_assert(!std::is_constructible_v<ContainerNetwork<>, const ContainerParameters&&>);

struct PositionFixture {
    const char* name;
    const char* board;
    Color       sideToMove;
    int         rule50;
};

constexpr std::array<PositionFixture, 6> Fixtures = {{
  {"start-white",
   "PPPPPPPP"
   "PPPPPPPP"
   "PPPPPPPP"
   "PPPPPPPP"
   ".PP..PP."
   "........"
   "pppppppp"
   "rnbqkbnr",
   WHITE, 0},
  {"start-black-r37",
   "PPPPPPPP"
   "PPPPPPPP"
   "PPPPPPPP"
   "PPPPPPPP"
   ".PP..PP."
   "........"
   "pppppppp"
   "rnbqkbnr",
   BLACK, 37},
  {"minimal-r100",
   "Q......."
   "........"
   "........"
   "........"
   "........"
   "........"
   "........"
   "....k...",
   WHITE, 100},
  {"mirror-d4",
   "........"
   "P......."
   "........"
   "...k...."
   "........"
   "..N....."
   "........"
   ".......q",
   BLACK, 7},
  {"mirror-e4",
   "........"
   ".......P"
   "........"
   "....k..."
   "........"
   ".....N.."
   "........"
   "q.......",
   WHITE, 19},
  {"mixed-promotions",
   "R..Q...."
   "....p..."
   "B......n"
   "....P..."
   ".r...b.."
   "........"
   "...q...."
   "......k.",
   BLACK, 51},
}};

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "Horde V2 container test failure: " << message << '\n';
    std::exit(EXIT_FAILURE);
}

Piece piece_from_char(char value) {
    switch (value)
    {
    case '.' :
        return NO_PIECE;
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
        fail("fixture contains an unknown piece code");
    }
}

std::array<Piece, SQUARE_NB> decode_board(std::string_view encoded) {
    if (encoded.size() != SQUARE_NB)
        fail("fixture board length is not 64");
    std::array<Piece, SQUARE_NB> board{};
    for (std::size_t square = 0; square < board.size(); ++square)
        board[square] = piece_from_char(encoded[square]);
    return board;
}

template<typename T, std::size_t Size>
bool same_array(const std::array<T, Size>& left, const std::array<T, Size>& right) {
    return std::equal(left.begin(), left.end(), right.begin());
}

void require_same(const ContainerTrace& scalar,
                  const ContainerTrace& selected,
                  std::string_view      name) {
    if (scalar.error != selected.error || scalar.featureError != selected.featureError
        || !same_array(scalar.firstAccumulator, selected.firstAccumulator)
        || !same_array(scalar.globalAccumulator, selected.globalAccumulator)
        || !same_array(scalar.transformed, selected.transformed)
        || !same_array(scalar.hidden0Affine, selected.hidden0Affine)
        || !same_array(scalar.hidden0, selected.hidden0)
        || !same_array(scalar.hidden1Affine, selected.hidden1Affine)
        || !same_array(scalar.hidden1, selected.hidden1)
        || scalar.outputAffine != selected.outputAffine
        || scalar.preRule50Value != selected.preRule50Value || scalar.value != selected.value)
        fail(std::string("scalar/backend trace differs in ") + std::string(name));
}

std::array<Piece, SQUARE_NB>
make_board(std::initializer_list<std::pair<Square, Piece>> pieces) {
    std::array<Piece, SQUARE_NB> board{};
    for (const auto& [square, piece] : pieces)
        board[square] = piece;
    return board;
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
    dirty.remove_pc = removePiece;
    dirty.remove_sq = removeSquare;
    dirty.add_pc    = addPiece;
    dirty.add_sq    = addSquare;
    return dirty;
}

template<typename Kernels>
void verify_incremental_transition(const ContainerNetwork<Kernels>&       network,
                                   const std::array<Piece, SQUARE_NB>& sourceBoard,
                                   const DirtyPiece&                   rawDirty,
                                   std::string_view                    name) {
    NormalizedDirty dirty{};
    if (!normalize_dirty_piece(rawDirty, dirty))
        fail(std::string("invalid incremental fixture dirty: ") + std::string(name));

    std::array<Piece, SQUARE_NB> targetBoard = sourceBoard;
    if (!apply_normalized_dirty(targetBoard, dirty))
        fail(std::string("incremental fixture transition failed: ") + std::string(name));

    const FullRefreshFeatures sourceFeatures = extract_full_refresh_features(sourceBoard);
    const FullRefreshFeatures targetFeatures = extract_full_refresh_features(targetBoard);
    if (!sourceFeatures.valid() || !targetFeatures.valid())
        fail(std::string("incremental fixture position invalid: ") + std::string(name));

    typename ContainerNetwork<Kernels>::Frame source{};
    typename ContainerNetwork<Kernels>::Frame child{};
    typename ContainerNetwork<Kernels>::Frame refreshed{};
    network.full_refresh(source, sourceFeatures);

    const RoyalKey targetKey = network.first_domain_key(targetFeatures);
    const bool     refresh   = network.requires_refresh(source.key, targetKey);
    if (refresh)
    {
        if (!network.materialize_child(child, source, dirty, targetFeatures))
            fail(std::string("incremental refresh failed: ") + std::string(name));
    }
    else if (!network.materialize_child_delta(child, source, dirty, targetKey))
        fail(std::string("incremental delta was rejected: ") + std::string(name));

    network.full_refresh(refreshed, targetFeatures);
    if (child.first != refreshed.first || child.global != refreshed.global
        || child.key != refreshed.key)
        fail(std::string("incremental/full accumulator mismatch: ") + std::string(name));

    const bool expectedRefresh = network.first_domain() == FirstDomain::ROYAL
                              && sourceFeatures.royalKey != targetFeatures.royalKey;
    if (refresh != expectedRefresh)
        fail(std::string("wrong first-domain refresh decision: ") + std::string(name));

    if (network.first_domain() == FirstDomain::ABSOLUTE_NONKING && dirty.pc == B_KING
        && (dirty.add_sq == SQ_NONE
            || (dirty.remove_sq == dirty.add_sq && dirty.remove_pc == dirty.add_pc))
        && child.first != source.first)
        fail(std::string("Black king changed absolute first domain: ") + std::string(name));

    typename ContainerNetwork<Kernels>::Scratch incrementalScratch{};
    typename ContainerNetwork<Kernels>::Scratch refreshScratch{};
    const LeanEvalResult incremental = network.propagate(child, incrementalScratch, BLACK, 37);
    const LeanEvalResult full         = network.propagate(refreshed, refreshScratch, BLACK, 37);
    if (incrementalScratch.transformed != refreshScratch.transformed
        || incrementalScratch.hidden0Affine != refreshScratch.hidden0Affine
        || incrementalScratch.hidden0 != refreshScratch.hidden0
        || incrementalScratch.hidden1Affine != refreshScratch.hidden1Affine
        || incrementalScratch.hidden1 != refreshScratch.hidden1
        || incremental.outputAffine != full.outputAffine
        || incremental.preRule50Value != full.preRule50Value || incremental.value != full.value)
        fail(std::string("incremental/full dense mismatch: ") + std::string(name));
}

template<typename Kernels>
void verify_incremental_suite(const ContainerNetwork<Kernels>& network) {
    const auto run = [&](std::string_view name, std::array<Piece, SQUARE_NB> board,
                         const DirtyPiece& dirty) {
        verify_incremental_transition(network, board, dirty, name);
    };

    run("quiet", make_board({{SQ_A2, W_PAWN}, {SQ_E8, B_KING}}),
        make_dirty(W_PAWN, SQ_A2, SQ_A3));
    run("capture",
        make_board({{SQ_A2, W_PAWN}, {SQ_D2, W_ROOK}, {SQ_C2, B_PAWN}, {SQ_E8, B_KING}}),
        make_dirty(W_ROOK, SQ_D2, SQ_C2, B_PAWN, SQ_C2));
    run("en-passant",
        make_board({{SQ_A2, W_PAWN}, {SQ_E5, W_PAWN}, {SQ_F5, B_PAWN}, {SQ_E8, B_KING}}),
        make_dirty(W_PAWN, SQ_E5, SQ_F6, B_PAWN, SQ_F5));
    run("promotion", make_board({{SQ_A2, W_PAWN}, {SQ_B7, W_PAWN}, {SQ_E8, B_KING}}),
        make_dirty(W_PAWN, SQ_B7, SQ_NONE, NO_PIECE, SQ_NONE, W_QUEEN, SQ_B8));
    run("promotion-capture",
        make_board({{SQ_A2, W_PAWN}, {SQ_B7, W_PAWN}, {SQ_A8, B_ROOK}, {SQ_E8, B_KING}}),
        make_dirty(W_PAWN, SQ_B7, SQ_NONE, B_ROOK, SQ_A8, W_QUEEN, SQ_A8));
    run("king-same-mirror", make_board({{SQ_A2, W_PAWN}, {SQ_E8, B_KING}}),
        make_dirty(B_KING, SQ_E8, SQ_F8));
    run("king-cross-mirror", make_board({{SQ_A2, W_PAWN}, {SQ_D8, B_KING}}),
        make_dirty(B_KING, SQ_D8, SQ_E8));
    run("castle-orthodox",
        make_board({{SQ_A2, W_PAWN}, {SQ_E8, B_KING}, {SQ_H8, B_ROOK}}),
        make_dirty(B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_H8, B_ROOK, SQ_F8));
    run("castle-king-to-rook",
        make_board({{SQ_A2, W_PAWN}, {SQ_E8, B_KING}, {SQ_G8, B_ROOK}}),
        make_dirty(B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_G8, B_ROOK, SQ_F8));
    run("castle-rook-to-king",
        make_board({{SQ_A2, W_PAWN}, {SQ_F8, B_KING}, {SQ_H8, B_ROOK}}),
        make_dirty(B_KING, SQ_F8, SQ_G8, B_ROOK, SQ_H8, B_ROOK, SQ_F8));
    run("castle-zero-king",
        make_board({{SQ_A2, W_PAWN}, {SQ_G8, B_KING}, {SQ_H8, B_ROOK}}),
        make_dirty(B_KING, SQ_G8, SQ_G8, B_ROOK, SQ_H8, B_ROOK, SQ_F8));
    run("castle-zero-rook",
        make_board({{SQ_A2, W_PAWN}, {SQ_E8, B_KING}, {SQ_F8, B_ROOK}}),
        make_dirty(B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_F8, B_ROOK, SQ_F8));
}

template<typename T, std::size_t Size>
void emit_array(const std::array<T, Size>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index)
            std::cout << ',';
        std::cout << int(values[index]);
    }
    std::cout << ']';
}

void emit_trace(const PositionFixture& fixture, const ContainerTrace& trace) {
    std::cout << "{\"name\":\"" << fixture.name << "\",\"first_accumulator\":";
    emit_array(trace.firstAccumulator);
    std::cout << ",\"global_accumulator\":";
    emit_array(trace.globalAccumulator);
    std::cout << ",\"transformed\":";
    emit_array(trace.transformed);
    std::cout << ",\"hidden0_affine\":";
    emit_array(trace.hidden0Affine);
    std::cout << ",\"hidden0\":";
    emit_array(trace.hidden0);
    std::cout << ",\"hidden1_affine\":";
    emit_array(trace.hidden1Affine);
    std::cout << ",\"hidden1\":";
    emit_array(trace.hidden1);
    std::cout << ",\"output_affine\":" << trace.outputAffine
              << ",\"pre_rule50\":" << trace.preRule50Value << ",\"value\":" << int(trace.value)
              << '}';
}

int validate(const std::filesystem::path& path) {
    ContainerLoadResult loaded = load_integer_container(path);
    if (!loaded)
    {
        std::cerr << container_load_error_name(loaded.error) << ": " << loaded.message << '\n';
        return 2;
    }
    std::cout << "VALID " << loaded.parameters.schemaName << ' ' << loaded.parameters.fileSha256
              << ' ' << loaded.parameters.parameterSha256 << '\n';
    return 0;
}

int trace(const std::filesystem::path& path) {
    ContainerLoadResult loaded = load_integer_container(path);
    if (!loaded)
    {
        std::cerr << container_load_error_name(loaded.error) << ": " << loaded.message << '\n';
        return 2;
    }

    ContainerNetwork<ScalarLeanKernels>  scalar(loaded.parameters);
    ContainerNetwork<DefaultLeanKernels> selected(loaded.parameters);
    verify_incremental_suite(scalar);
    verify_incremental_suite(selected);
    std::cout << "{\"schema\":\"HORDE_V2_FULL_REFRESH_TRACE_V1\",\"network_schema\":\""
              << loaded.parameters.schemaName << "\",\"backend\":\"" << DefaultLeanBackendName
              << "\",\"file_sha256\":\"" << loaded.parameters.fileSha256
              << "\",\"parameter_sha256\":\"" << loaded.parameters.parameterSha256
              << "\",\"positions\":[";
    for (std::size_t index = 0; index < Fixtures.size(); ++index)
    {
        const auto& fixture = Fixtures[index];
        const auto  board   = decode_board(fixture.board);
        const auto  scalarTrace =
          scalar.evaluate_full_refresh(board, fixture.sideToMove, fixture.rule50);
        const auto selectedTrace =
          selected.evaluate_full_refresh(board, fixture.sideToMove, fixture.rule50);
        if (!scalarTrace.valid() || !selectedTrace.valid())
            fail(std::string("valid fixture was rejected: ") + fixture.name);
        require_same(scalarTrace, selectedTrace, fixture.name);
        if (index)
            std::cout << ',';
        emit_trace(fixture, scalarTrace);
    }
    std::cout << "]}\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3)
    {
        std::cerr << "usage: horde-v2-container-test --validate|--trace NETWORK\n";
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "--validate")
        return validate(argv[2]);
    if (mode == "--trace")
        return trace(argv[2]);
    std::cerr << "unknown mode: " << mode << '\n';
    return 2;
}
