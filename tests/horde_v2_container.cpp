/*
  Trained Horde V2 container and full-refresh parity oracle.
*/

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include "nnue/horde_v2_container.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE::HordeV2;

namespace {

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
    if (scalar.featureError != selected.featureError
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
