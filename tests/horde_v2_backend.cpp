/*
  Correctness receipts for the lean, width-templated Horde V2 backend.
*/

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

#include "../src/nnue/horde_v2_backend.h"

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE::HordeV2;

namespace {

constexpr u64 FixtureSeed = 0x4856325F50455246ULL;

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "Horde V2 backend failure: " << message << '\n';
    std::exit(EXIT_FAILURE);
}

void require(bool condition, const std::string& message) {
    if (!condition)
        fail(message);
}

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

ScalarParameters copy_scalar_parameters(const LeanParameters<Width256x256>& source) {
    ScalarParameters target;
    target.royalBias   = source.royalBias;
    target.globalBias  = source.globalBias;
    target.hidden0Bias = source.hidden0Bias;
    target.hidden1Bias = source.hidden1Bias;
    target.outputBias  = source.outputBias;
    std::copy(source.royalWeights.begin(), source.royalWeights.end(), target.royalWeights.begin());
    std::copy(source.globalWeights.begin(), source.globalWeights.end(),
              target.globalWeights.begin());
    std::copy(source.hidden0Weights.begin(), source.hidden0Weights.end(),
              target.hidden0Weights.begin());
    std::copy(source.hidden1Weights.begin(), source.hidden1Weights.end(),
              target.hidden1Weights.begin());
    std::copy(source.outputWeights.begin(), source.outputWeights.end(),
              target.outputWeights.begin());
    return target;
}

void require_frame_equals_trace(const LeanAccumulatorFrame<Width256x256>& frame,
                                const ScalarTrace&                        trace,
                                const std::string&                        label) {
    require(trace.valid(), label + ": scalar trace invalid");
    require(frame.royal == trace.royalAccumulator, label + ": Royal accumulator mismatch");
    require(frame.global == trace.globalAccumulator, label + ": Global accumulator mismatch");
    require(frame.key == trace.royalKey, label + ": Royal key mismatch");
}

void require_dense_equals_trace(const LeanDenseScratch<Width256x256>& scratch,
                                const LeanEvalResult&                 result,
                                const ScalarTrace&                    trace,
                                const std::string&                    label) {
    require(scratch.transformed == trace.transformed, label + ": transformed mismatch");
    require(scratch.hidden0Affine == trace.hidden0Affine, label + ": H0 affine mismatch");
    require(scratch.hidden0 == trace.hidden0, label + ": H0 activation mismatch");
    require(scratch.hidden1Affine == trace.hidden1Affine, label + ": H1 affine mismatch");
    require(scratch.hidden1 == trace.hidden1, label + ": H1 activation mismatch");
    require(result.outputAffine == trace.outputAffine, label + ": output affine mismatch");
    require(result.preRule50Value == trace.preRule50Value, label + ": pre-rule50 value mismatch");
    require(result.value == trace.value, label + ": final value mismatch");
}

void verify_transition(LeanNetwork<Width256x256>&          lean,
                       ScalarNetwork&                      scalar,
                       const std::array<Piece, SQUARE_NB>& sourceBoard,
                       const std::array<Piece, SQUARE_NB>& targetBoard,
                       const DirtyPiece&                   dirty,
                       Color                               targetSide,
                       int                                 targetRule50,
                       const std::string&                  label) {
    const FullRefreshFeatures sourceFeatures = extract_full_refresh_features(sourceBoard);
    const FullRefreshFeatures targetFeatures = extract_full_refresh_features(targetBoard);
    require(sourceFeatures.valid(), label + ": invalid source features");
    require(targetFeatures.valid(), label + ": invalid target features");

    LeanAccumulatorFrame<Width256x256> sourceFrame{};
    LeanAccumulatorFrame<Width256x256> childFrame{};
    lean.full_refresh(sourceFrame, sourceFeatures);
    lean.materialize_child(childFrame, sourceFrame, dirty, targetFeatures);

    const ScalarTrace scalarSource = scalar.evaluate_full_refresh(sourceBoard, ~targetSide, 0);
    const ScalarTrace scalarTarget =
      scalar.evaluate_incremental(dirty, targetBoard, scalarSource, targetSide, targetRule50);
    require_frame_equals_trace(sourceFrame, scalarSource, label + " source");
    require_frame_equals_trace(childFrame, scalarTarget, label + " child");

    LeanDenseScratch<Width256x256> scratch{};
    const LeanEvalResult result = lean.propagate(childFrame, scratch, targetSide, targetRule50);
    require_dense_equals_trace(scratch, result, scalarTarget, label + " dense");
}

void verify_parity_backend() {
    auto parameters =
      make_lean_parameters<Width256x256>(FixtureSeed, FixturePayload::PARITY_FULL_V1);
    require(reinterpret_cast<std::uintptr_t>(parameters.royalWeights.data()) % 64 == 0,
            "Royal table is not 64-byte aligned");
    require(reinterpret_cast<std::uintptr_t>(parameters.globalWeights.data()) % 64 == 0,
            "Global table is not 64-byte aligned");

    ScalarNetwork             scalar(copy_scalar_parameters(parameters));
    LeanNetwork<Width256x256> lean(std::move(parameters));

    const auto startBoard    = horde_start_board();
    const auto startFeatures = extract_full_refresh_features(startBoard);
    require(startFeatures.valid(), "invalid Horde start features");

    LeanAccumulatorFrame<Width256x256> startFrame{};
    require(reinterpret_cast<std::uintptr_t>(&startFrame) % 64 == 0,
            "lean frame is not 64-byte aligned");
    lean.full_refresh(startFrame, startFeatures);

    for (const Color side : {WHITE, BLACK})
    {
        const ScalarTrace scalarTrace = scalar.evaluate_full_refresh(startBoard, side, 37);
        require_frame_equals_trace(startFrame, scalarTrace, "start full refresh");
        LeanDenseScratch<Width256x256> scratch{};
        const LeanEvalResult           result = lean.propagate(startFrame, scratch, side, 37);
        require_dense_equals_trace(scratch, result, scalarTrace, "start propagation");
    }

    const auto base = incremental_base_board();

    auto quietTarget      = base;
    quietTarget[SQ_A2]    = NO_PIECE;
    quietTarget[SQ_A3]    = W_PAWN;
    const auto quietDirty = make_dirty(W_PAWN, SQ_A2, SQ_A3);
    verify_transition(lean, scalar, base, quietTarget, quietDirty, BLACK, 0, "quiet");

    auto captureTarget      = base;
    captureTarget[SQ_D4]    = NO_PIECE;
    captureTarget[SQ_C4]    = W_ROOK;
    const auto captureDirty = make_dirty(W_ROOK, SQ_D4, SQ_C4, B_PAWN, SQ_C4);
    verify_transition(lean, scalar, base, captureTarget, captureDirty, BLACK, 0, "capture");

    auto epTarget      = base;
    epTarget[SQ_E5]    = NO_PIECE;
    epTarget[SQ_F5]    = NO_PIECE;
    epTarget[SQ_F6]    = W_PAWN;
    const auto epDirty = make_dirty(W_PAWN, SQ_E5, SQ_F6, B_PAWN, SQ_F5);
    verify_transition(lean, scalar, base, epTarget, epDirty, BLACK, 0, "en passant");

    auto promotionTarget      = base;
    promotionTarget[SQ_B7]    = NO_PIECE;
    promotionTarget[SQ_A8]    = W_QUEEN;
    const auto promotionDirty = make_dirty(W_PAWN, SQ_B7, SQ_NONE, B_ROOK, SQ_A8, W_QUEEN, SQ_A8);
    verify_transition(lean, scalar, base, promotionTarget, promotionDirty, BLACK, 0,
                      "promotion capture");

    auto kingTarget      = base;
    kingTarget[SQ_E8]    = NO_PIECE;
    kingTarget[SQ_E7]    = B_KING;
    const auto kingDirty = make_dirty(B_KING, SQ_E8, SQ_E7);
    verify_transition(lean, scalar, base, kingTarget, kingDirty, WHITE, 1, "king move");

    auto castlingTarget      = base;
    castlingTarget[SQ_E8]    = NO_PIECE;
    castlingTarget[SQ_H8]    = NO_PIECE;
    castlingTarget[SQ_G8]    = B_KING;
    castlingTarget[SQ_F8]    = B_ROOK;
    const auto castlingDirty = make_dirty(B_KING, SQ_E8, SQ_G8, B_ROOK, SQ_H8, B_ROOK, SQ_F8);
    verify_transition(lean, scalar, base, castlingTarget, castlingDirty, WHITE, 1, "castling");
}

struct CommonReceipt {
    std::array<Accumulator, V2Hidden0Lanes> hidden0Affine{};
    std::array<Activation, V2Hidden0Lanes>  hidden0{};
    std::array<Accumulator, V2Hidden1Lanes> hidden1Affine{};
    std::array<Activation, V2Hidden1Lanes>  hidden1{};
    LeanEvalResult                          result{};
};

template<typename Width>
CommonReceipt evaluate_common(LeanNetwork<Width>&        network,
                              const FullRefreshFeatures& features,
                              Color                      side,
                              int                        rule50) {
    LeanAccumulatorFrame<Width> frame{};
    LeanDenseScratch<Width>     scratch{};
    network.full_refresh(frame, features);
    const LeanEvalResult result = network.propagate(frame, scratch, side, rule50);
    return {scratch.hidden0Affine, scratch.hidden0, scratch.hidden1Affine, scratch.hidden1, result};
}

void require_common_equal(const CommonReceipt& expected,
                          const CommonReceipt& actual,
                          const std::string&   label) {
    require(actual.hidden0Affine == expected.hidden0Affine, label + ": H0 affine differs");
    require(actual.hidden0 == expected.hidden0, label + ": H0 activation differs");
    require(actual.hidden1Affine == expected.hidden1Affine, label + ": H1 affine differs");
    require(actual.hidden1 == expected.hidden1, label + ": H1 activation differs");
    require(actual.result.outputAffine == expected.result.outputAffine,
            label + ": output affine differs");
    require(actual.result.preRule50Value == expected.result.preRule50Value,
            label + ": pre-rule50 differs");
    require(actual.result.value == expected.result.value, label + ": final value differs");
}

void verify_common_payload() {
    LeanNetwork<Width256x256> width0(
      make_lean_parameters<Width256x256>(FixtureSeed, FixturePayload::PERF_COMMON_V1));
    LeanNetwork<Width128x256> width1(
      make_lean_parameters<Width128x256>(FixtureSeed, FixturePayload::PERF_COMMON_V1));
    LeanNetwork<Width128x128> width2(
      make_lean_parameters<Width128x128>(FixtureSeed, FixturePayload::PERF_COMMON_V1));
    LeanNetwork<Width64x192> width3(
      make_lean_parameters<Width64x192>(FixtureSeed, FixturePayload::PERF_COMMON_V1));

    require(width0.parameters().hidden0Weights[64] == 0,
            "PERF_COMMON Royal extension is not zero-connected");
    require(width0.parameters().hidden0Weights[Width256x256::RoyalLanes + 128] == 0,
            "PERF_COMMON Global extension is not zero-connected");
    require(width3.parameters().royalWeights[Width64x192::RoyalLanes - 1] != 0,
            "PERF_COMMON Royal FT lane was zeroed");
    require(width3.parameters().globalWeights[Width64x192::GlobalLanes - 1] != 0,
            "PERF_COMMON Global FT lane was zeroed");

    auto start         = horde_start_board();
    auto sparse        = incremental_base_board();
    auto shiftedKing   = start;
    shiftedKing[SQ_E8] = NO_PIECE;
    shiftedKing[SQ_E6] = B_KING;

    for (const auto& board : {start, sparse, shiftedKing})
    {
        const FullRefreshFeatures features = extract_full_refresh_features(board);
        require(features.valid(), "invalid PERF_COMMON board");
        for (const Color side : {WHITE, BLACK})
        {
            const CommonReceipt expected = evaluate_common(width0, features, side, 43);
            require_common_equal(expected, evaluate_common(width1, features, side, 43),
                                 "128+256 PERF_COMMON");
            require_common_equal(expected, evaluate_common(width2, features, side, 43),
                                 "128+128 PERF_COMMON");
            require_common_equal(expected, evaluate_common(width3, features, side, 43),
                                 "64+192 PERF_COMMON");
        }
    }
}

}  // namespace

int main() {
    verify_parity_backend();
    verify_common_payload();
    std::cout << "Horde V2 lean backend passed: backend=" << DefaultLeanBackendName
              << ", parity=256+256, common=4 widths, transitions=6\n";
}
