/*
  Horde NNUE V3 kernel-level timing harness.

  It times the minimum matrix the V2 design specifies, adapted to the V3
  topology (one sparse transformer, 1024 lanes, 896 stream rows):

    1  frame copy                 V3AccumulatorFrame assignment
    2  full refresh               at 0, at 1 and at the maximum active row count
    3  quiet transition           G0 delta only, no contextual row moves
    4  contextual transition      at least one file's contextual block moves
    5  dense propagation          trunk + PSQT + rule50, accumulator already made
    6  composed evaluation        refresh + dense
    7  TWO-GATHER REFERENCE       two gathers of 52 rows over 512 lanes each,
                                  against the V3 refresh of the same row count
                                  over 1024 lanes in a single gather

  Case 7 is the point of the exercise. The 2x512 side is an EMULATION of the
  incumbent's arithmetic shape -- two perspectives, two independent row-index
  streams, two feature tables whose bytes sum to exactly the V3 table's bytes,
  the same total lane-adds -- written with the same kernel primitive as the V3
  side. It is not the incumbent's actual code and makes no claim to be.

  Measurement discipline:
    * std::chrono::steady_clock, one thread, no threading anywhere.
    * Every case runs under a hot schedule (one pinned pool slot, L1 resident)
      and a deterministic streaming schedule (a strided walk over distinct
      boards, frames and table regions, sized past L1 and L2).
    * The two schedules execute the same loop shape: the hot schedule is the
      streaming loop with a pool of one and a stride of zero, so no bookkeeping
      asymmetry can favour either.
    * Case order is reshuffled every round under the frozen seed and every case
      contributes exactly one sample per round, so the samples are paired and
      background drift cannot systematically favour one case.
    * Allocation, page faults, feature extraction, validation and logging stay
      outside the timed regions. Every page of every table and pool is touched
      once before timing.
    * Per case: sample count, median, median absolute deviation and a
      bootstrapped 95% confidence interval on the median. The bootstrap
      resamples ROUNDS, shared across cases, so the two-gather ratio interval
      is genuinely paired.

  --verify re-runs the timed kernels against the parity path -- fresh feature
  enumeration, fresh full refresh, incremental child against a full refresh of
  the post-move board, scalar against the selected backend, and the two-gather
  emulation against an independent scalar sum -- so a timing build that
  silently computes the wrong thing cannot pass.
*/

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "nnue/horde_v2_stack.h"      // apply_normalized_dirty, the board primitive V3 mirrors
#include "nnue/horde_v3_container.h"  // the optional real container, and the whole V3 stack

using namespace Stockfish;
using namespace Stockfish::Eval::NNUE;
using namespace Stockfish::Eval::NNUE::HordeV3;

namespace {

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

[[noreturn]] void fail(const std::string& message) {
    std::cerr << "Horde V3 kernel timing failure: " << message << '\n';
    std::exit(EXIT_FAILURE);
}

void require(bool condition, const std::string& message) {
    if (!condition)
        fail(message);
}

// ---------------------------------------------------------------------------
// Optimiser barriers. Every timed body writes through one of these, so no
// timed region can be folded away or hoisted out of its loop.
// ---------------------------------------------------------------------------

volatile i64 g_sink = 0;

#if defined(__GNUC__) || defined(__clang__)
inline void escape(const void* pointer) noexcept {
    __asm__ __volatile__("" : : "r"(pointer) : "memory");
}
inline void clobber() noexcept { __asm__ __volatile__("" : : : "memory"); }
#else
const void* volatile g_escapeSink = nullptr;
inline void escape(const void* pointer) noexcept { g_escapeSink = pointer; }
inline void clobber() noexcept { g_sink = g_sink; }
#endif

// ---------------------------------------------------------------------------
// The deterministic fixture network.
//
// A bit-for-bit mirror of tests/horde_v3_parity.py::deterministic_sections(),
// which is the fixture container the parity oracle is run against. The salt
// per section, the modulus per dtype and the phase lookup are the same, so the
// in-process fixture and the on-disk parity fixture hold the same parameters.
// No new fixture is invented here.
// ---------------------------------------------------------------------------

constexpr std::uint32_t NetworkSchemaId = 0x00020001u;

constexpr std::array<std::uint8_t, V3PhaseLookupSize> FixturePhaseLookup = {
  0, 0, 0, 0, 0, 0,          //
  1, 1, 1, 1,                //
  2, 2, 2, 2,                //
  3, 3, 3, 3, 3,             //
  4, 4, 4, 4,                //
  5, 5, 5, 5,                //
  6, 6, 6,                   //
  7, 7, 7, 7, 7, 7, 7};

constexpr i64 fixture_salt(i64 sectionId) { return i64(NetworkSchemaId) * 17 + sectionId * 101; }

void build_fixture_parameters(V3Parameters& parameters) {
    parameters.schemaName      = "HORDE_V3_INTEGER_NETWORK_V1";
    parameters.fileSha256      = "";
    parameters.parameterSha256 = "";

    const i64 ftWeightSalt      = fixture_salt(1);
    const i64 ftBiasSalt        = fixture_salt(2);
    const i64 psqtSalt          = fixture_salt(3);
    const i64 hidden0WeightSalt = fixture_salt(4);
    const i64 hidden0BiasSalt   = fixture_salt(5);
    const i64 hidden1WeightSalt = fixture_salt(6);
    const i64 hidden1BiasSalt   = fixture_salt(7);
    const i64 outputWeightSalt  = fixture_salt(8);
    const i64 outputBiasSalt    = fixture_salt(9);

    const auto i16Value = [](i64 index, i64 salt) {
        return FtWeight(((index * 97 + salt) % 63) - 31);
    };
    const auto i8Value = [](i64 index, i64 salt) {
        return DenseWeight(((index * 37 + salt) % 15) - 7);
    };
    const auto biasValue = [](i64 index, i64 salt) {
        return AffineBias(((index * 193 + salt) % 8193) - 4096);
    };

    for (std::size_t index = 0; index < FtWeightCount; ++index)
        parameters.ftWeights[index] = i16Value(i64(index), ftWeightSalt);
    for (std::size_t index = 0; index < V3Lanes; ++index)
        parameters.ftBias[index] =
          AffineBias(((i64(index) * 193 + ftBiasSalt) % 12289) - 6144 + 4096);
    for (std::size_t index = 0; index < PsqtWeightCount; ++index)
        parameters.psqtWeights[index] =
          PsqtWeight(((i64(index) * 8191 + psqtSalt) % 40001) - 20000);

    for (std::size_t index = 0; index < Hidden0WeightCount; ++index)
        parameters.hidden0Weights[index] = i8Value(i64(index), hidden0WeightSalt);
    for (std::size_t index = 0; index < parameters.hidden0Bias.size(); ++index)
        parameters.hidden0Bias[index] = biasValue(i64(index), hidden0BiasSalt);
    for (std::size_t index = 0; index < Hidden1WeightCount; ++index)
        parameters.hidden1Weights[index] = i8Value(i64(index), hidden1WeightSalt);
    for (std::size_t index = 0; index < parameters.hidden1Bias.size(); ++index)
        parameters.hidden1Bias[index] = biasValue(i64(index), hidden1BiasSalt);
    for (std::size_t index = 0; index < OutputWeightCount; ++index)
        parameters.outputWeights[index] = i8Value(i64(index), outputWeightSalt);
    for (std::size_t index = 0; index < parameters.outputBias.size(); ++index)
        parameters.outputBias[index] = biasValue(i64(index), outputBiasSalt);

    parameters.phaseLookup = FixturePhaseLookup;
}

// ---------------------------------------------------------------------------
// Boards and FEN. Same shape as tests/horde_v3_parity.cpp.
// ---------------------------------------------------------------------------

struct PositionState {
    std::array<Piece, SQUARE_NB> board{};
    Color                        sideToMove = WHITE;
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
            const Piece piece = board[std::size_t(rank * 8 + file)];
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

bool try_move(const std::array<Piece, SQUARE_NB>& source,
              const DirtyPiece&                   dirty,
              std::array<Piece, SQUARE_NB>&       target) {
    HordeV2::NormalizedDirty normalized{};
    if (!HordeV2::normalize_dirty_piece(dirty, normalized))
        return false;
    target = source;
    return HordeV2::apply_normalized_dirty(target, normalized);
}

// The seventeen parity fixtures, verbatim from tests/horde_v3_parity.cpp. They
// are reused for --verify and for the dense and evaluation pools.
constexpr std::array<const char*, 17> ParityFixtureFens = {
  "4k3/8/1n1b4/PPPPPPPP/PPPPPPPP/2r1q3/8/8 b - - 0 1",
  "4k3/8/8/2nbrq2/1PPPPP2/8/8/8 w - - 0 1",
  "4k3/8/8/8/8/8/8/QQRRBBNN w - - 0 1",
  "rnbq1bnr/ppp1kppp/3p4/8/8/2P5/PP1PPPPP/8 b - - 0 1",
  "4k3/8/2P2P2/1P2P3/P1P2P2/2P1P3/PP2PP2/8 w - - 0 1",
  "1r1qk2r/2p2ppp/p1np1n2/1p2p3/4P3/2PP1P2/PP2P1PP/8 b - - 0 1",
  "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w - - 0 1",
  "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP b - - 3 2",
  "r2qkb1r/pp1n1ppp/4pn2/1PP5/P1PP1P2/2P1P3/PP1PP1PP/1P4P1 w - - 5 12",
  "r2qkb1r/pp1n1ppp/4pn2/1PP5/P1PP1P2/2P1P3/PP1PP1PP/1P4P1 b - - 5 12",
  "4k3/PP4P1/PP4P1/PP4P1/PP4P1/PP4P1/PP4P1/PP4P1 w - - 0 1",
  "4k3/8/8/3pP3/8/8/PPP5/8 w - d6 0 9",
  "4k3/8/8/8/2Pp4/8/PP6/8 b - c3 0 9",
  "r3k2r/PPP3P1/8/8/8/2n5/PP1PP3/8 w - - 0 1",
  "8/5k2/8/2P5/8/1P6/P7/8 w - - 12 40",
  "8/8/3PPP2/3PkP2/3PPP2/8/8/8 b - - 0 1",
  "8/8/8/3k4/8/8/PPP5/8 b - - 0 1"};

// ---------------------------------------------------------------------------
// Board families
// ---------------------------------------------------------------------------

// The maximum-row family.
//
//   ranks 1..4          White pawns on every file                       32
//   rank 5   files 0..k-1 White pawns, files k..7 Black blockers      k + (8-k)
//   rank 6   files 0..k-1 Black blockers                                  k
//   rank 7   free, carries the 4-k White pieces that top White up to 36
//   rank 8   eight Black pieces, exactly one king
//
// Every file therefore has White pawns, every frontier is blocked from above,
// every frontier is supported from the rank below and every frontier has a
// phalanx neighbour, so all six contextual blocks fire on all eight files:
// 48 contextual rows. White is 36 and Black is 16, so 52 pieces give 52 G0
// rows. 52 + 48 = 100 = MaxActiveRows, the frozen ceiling.
std::array<Piece, SQUARE_NB> max_row_board(std::size_t variant) {
    constexpr std::array<Piece, 4> BlackFiller = {B_KNIGHT, B_BISHOP, B_ROOK, B_QUEEN};
    constexpr std::array<Piece, 4> WhiteFiller = {W_KNIGHT, W_BISHOP, W_ROOK, W_QUEEN};

    const std::size_t high      = 2 + variant % 3;             // k, in {2,3,4}
    const std::size_t extras    = 4 - high;                    // White pieces to top up to 36
    const std::size_t extraBase = (variant / 3) % 8;
    const std::size_t lowType   = (variant / 24) % 4;
    const std::size_t highType  = (variant / 96) % 4;
    const std::size_t kingFile  = (variant / 384) % 8;

    std::array<Piece, SQUARE_NB> board{};
    for (int rank = 0; rank < 4; ++rank)
        for (int file = 0; file < 8; ++file)
            board[std::size_t(rank * 8 + file)] = W_PAWN;

    for (std::size_t file = 0; file < 8; ++file)
        board[4 * 8 + file] = file < high ? W_PAWN : BlackFiller[lowType];
    for (std::size_t file = 0; file < high; ++file)
        board[5 * 8 + file] = BlackFiller[highType];

    for (std::size_t index = 0; index < extras; ++index)
    {
        const std::size_t file = (extraBase + index * 3) % 8;
        board[6 * 8 + file]    = WhiteFiller[(variant + index) % 4];
    }

    for (std::size_t file = 0; file < 8; ++file)
        board[7 * 8 + file] =
          file == kingFile ? B_KING : BlackFiller[(file + variant) % BlackFiller.size()];
    return board;
}

// One Black king and nothing else: exactly one active row, and the smallest
// row count any legal board can reach.
std::array<Piece, SQUARE_NB> single_row_board(std::size_t variant) {
    std::array<Piece, SQUARE_NB> board{};
    board[variant % SQUARE_NB] = B_KING;
    return board;
}

// A quiet Black king step on rank 8 over a static White pawn wall. The wall's
// frontier is never blocked, supported or moved, so the transition is a pure
// G0 delta: one row out, one row in, no contextual row moves at all.
struct QuietCase {
    std::array<Piece, SQUARE_NB> board{};
    DirtyPiece                   dirty{};
};

QuietCase quiet_case(std::size_t variant) {
    const int pawnRank = 1 + int(variant % 3);
    const int kingFile = int((variant / 3) % 8);
    const int gap      = int((variant / 24) % 9);  // 8 means no gap

    QuietCase result{};
    for (int file = 0; file < 8; ++file)
        if (file != gap)
            result.board[std::size_t(pawnRank * 8 + file)] = W_PAWN;

    const int from = 7 * 8 + kingFile;
    const int to   = 7 * 8 + (kingFile < 7 ? kingFile + 1 : kingFile - 1);
    result.board[std::size_t(from)] = B_KING;
    result.dirty                    = make_dirty(B_KING, Square(from), Square(to));
    return result;
}

// A White pawn push whose contextual effect reaches the neighbouring files:
// the pushed file's frontier and rearmost rows move, and both neighbours swap
// frontier_supported for frontier_phalanx without either pawn moving. Variant
// zero is the parity oracle's "neighbour-phalanx" contextual fixture.
struct ContextualCase {
    std::array<Piece, SQUARE_NB> board{};
    DirtyPiece                   dirty{};
};

ContextualCase contextual_case(std::size_t variant) {
    const int rank     = 1 + int(variant % 4);
    const int file     = 1 + int((variant / 4) % 6);
    const int kingFile = int((variant / 24) % 8);

    ContextualCase result{};
    result.board[std::size_t((rank + 1) * 8 + file - 1)] = W_PAWN;
    result.board[std::size_t((rank + 1) * 8 + file + 1)] = W_PAWN;
    result.board[std::size_t(rank * 8 + file)]           = W_PAWN;
    result.board[std::size_t(7 * 8 + kingFile)]          = B_KING;

    const int from = rank * 8 + file;
    const int to   = (rank + 1) * 8 + file;
    result.dirty   = make_dirty(W_PAWN, Square(from), Square(to));
    return result;
}

// ---------------------------------------------------------------------------
// Lane-parametric feature-transformer kernels.
//
// add_ft is copied verbatim from the selected V3 backend with the lane count
// lifted into a template parameter, so the 1x1024 and the 2x512 refreshes run
// the same primitive and only the gather structure differs. --verify checks
// that the 1024-lane instantiation is bit-identical to the backend's own
// add_ft on real rows.
// ---------------------------------------------------------------------------

template<std::size_t Lanes>
struct ScalarLaneKernels {
    static void add_ft(Accumulator* accumulator, const FtWeight* row) noexcept {
        for (std::size_t lane = 0; lane < Lanes; ++lane)
            accumulator[lane] += Accumulator(row[lane]);
    }
};

#if defined(USE_AVX2)
template<std::size_t Lanes>
struct Avx2LaneKernels {
    static void add_ft(Accumulator* accumulator, const FtWeight* row) noexcept {
        for (std::size_t lane = 0; lane < Lanes; lane += 8)
        {
            const __m256i current =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(accumulator + lane));
            const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + lane));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(accumulator + lane),
                                _mm256_add_epi32(current, _mm256_cvtepi16_epi32(packed)));
        }
    }
};
template<std::size_t Lanes>
using LaneKernels = Avx2LaneKernels<Lanes>;
#else
template<std::size_t Lanes>
using LaneKernels = ScalarLaneKernels<Lanes>;
#endif

template<std::size_t Lanes>
inline void ft_refresh(Accumulator*       accumulator,
                       const Accumulator* bias,
                       const FtWeight*    table,
                       const IndexType*   rows,
                       std::size_t        count) noexcept {
    std::memcpy(accumulator, bias, Lanes * sizeof(Accumulator));
    for (std::size_t index = 0; index < count; ++index)
        LaneKernels<Lanes>::add_ft(accumulator, table + std::size_t(rows[index]) * Lanes);
}

inline constexpr std::size_t IncumbentLanes    = 512;
inline constexpr std::size_t TwoGatherRowCount = 52;  // MaxG0Rows, the Horde piece ceiling

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

double median_of_sorted(const std::vector<double>& sorted) {
    const std::size_t count = sorted.size();
    if (count == 0)
        return 0.0;
    return count % 2 ? sorted[count / 2] : 0.5 * (sorted[count / 2 - 1] + sorted[count / 2]);
}

double median_of(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return median_of_sorted(values);
}

double median_absolute_deviation(const std::vector<double>& values, double centre) {
    std::vector<double> deviations(values.size());
    for (std::size_t index = 0; index < values.size(); ++index)
        deviations[index] = std::abs(values[index] - centre);
    return median_of(std::move(deviations));
}

// Bootstrap draws over ROUND indices, shared by every case in a schedule, so
// per-case intervals and the two-gather ratio interval come from one paired
// resampling of the same rounds.
using BootstrapDraws = std::vector<std::vector<std::uint32_t>>;

BootstrapDraws make_draws(std::size_t rounds, std::size_t resamples, std::uint64_t seed) {
    std::mt19937_64                              rng(seed);
    std::uniform_int_distribution<std::uint32_t> pick(0, std::uint32_t(rounds - 1));
    BootstrapDraws                               draws(resamples, std::vector<std::uint32_t>(rounds));
    for (std::size_t resample = 0; resample < resamples; ++resample)
        for (std::size_t round = 0; round < rounds; ++round)
            draws[resample][round] = pick(rng);
    return draws;
}

struct Interval {
    double low  = 0.0;
    double high = 0.0;
};

Interval percentile_interval(std::vector<double> values) {
    if (values.empty())
        return {};
    std::sort(values.begin(), values.end());
    const auto index = [&](double fraction) {
        const double position = fraction * double(values.size() - 1);
        const auto   floorPos = std::size_t(position);
        const auto   ceilPos  = std::min(floorPos + 1, values.size() - 1);
        const double weight   = position - double(floorPos);
        return values[floorPos] * (1.0 - weight) + values[ceilPos] * weight;
    };
    return {index(0.025), index(0.975)};
}

Interval median_interval(const std::vector<double>& samples, const BootstrapDraws& draws) {
    std::vector<double> medians;
    medians.reserve(draws.size());
    std::vector<double> buffer(samples.size());
    for (const std::vector<std::uint32_t>& draw : draws)
    {
        for (std::size_t index = 0; index < draw.size(); ++index)
            buffer[index] = samples[draw[index]];
        medians.push_back(median_of(buffer));
    }
    return percentile_interval(std::move(medians));
}

Interval ratio_interval(const std::vector<double>& numerator,
                        const std::vector<double>& denominator,
                        const BootstrapDraws&      draws) {
    std::vector<double> ratios;
    ratios.reserve(draws.size());
    std::vector<double> left(numerator.size());
    std::vector<double> right(denominator.size());
    for (const std::vector<std::uint32_t>& draw : draws)
    {
        for (std::size_t index = 0; index < draw.size(); ++index)
        {
            left[index]  = numerator[draw[index]];
            right[index] = denominator[draw[index]];
        }
        const double bottom = median_of(right);
        ratios.push_back(bottom > 0.0 ? median_of(left) / bottom : 0.0);
    }
    return percentile_interval(std::move(ratios));
}

// ---------------------------------------------------------------------------
// The timing engine
// ---------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;
static_assert(Clock::is_steady, "the harness requires a steady clock");

// One batch: reps operations, walking the pool by stride. The hot schedule is
// this same loop with poolCount == 1 and stride == 0, so both schedules pay
// exactly the same per-iteration bookkeeping.
template<typename Body>
double time_batch(Body&&       body,
                  std::size_t  reps,
                  std::size_t  poolCount,
                  std::size_t  stride,
                  std::size_t& cursor) {
    // The walk is one add and one conditional subtract, so the stride has to
    // be reduced into [0, poolCount) or a single subtract cannot bring the
    // cursor back into range.
    const std::size_t step = poolCount ? stride % poolCount : 0;
    std::size_t       slot = poolCount ? cursor % poolCount : 0;

    const Clock::time_point start = Clock::now();
    for (std::size_t index = 0; index < reps; ++index)
    {
        body(slot);
        slot += step;
        if (slot >= poolCount)
            slot -= poolCount;
    }
    const Clock::time_point stop = Clock::now();

    cursor = slot;
    clobber();
    return double(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
}

using BatchFn = std::function<double(std::size_t reps, std::size_t poolCount, std::size_t stride,
                                     std::size_t& cursor)>;

struct ScheduleResult {
    std::size_t         samples    = 0;
    std::size_t         innerReps  = 0;
    std::size_t         poolCount  = 0;
    double              median     = 0.0;
    double              mad        = 0.0;
    Interval            ci{};
    std::vector<double> perOperationNs;
};

struct TimingCase {
    std::string    name;
    std::string    description;
    BatchFn        batch;
    std::size_t    streamingPool = 1;
    std::size_t    streamingStride = 1;
    ScheduleResult hot{};
    ScheduleResult streaming{};
};

// Grow the inner repetition count until one batch clears the target duration,
// so a single steady_clock tick can never dominate a sample. Calibration also
// serves as the warm-up for the case.
//
// A hiccup during calibration can only inflate a batch, which would stop the
// doubling early and leave the case permanently under-batched, so the decision
// is taken on the MINIMUM of several measurements at each level.
std::size_t calibrate(const BatchFn& batch,
                      std::size_t    poolCount,
                      std::size_t    stride,
                      double         targetNs) {
    constexpr int Probes = 3;

    std::size_t cursor = 0;
    std::size_t reps   = 1;
    for (int attempt = 0; attempt < 30; ++attempt)
    {
        double shortest = 0.0;
        for (int probe = 0; probe < Probes; ++probe)
        {
            const double elapsed = batch(reps, poolCount, stride, cursor);
            if (probe == 0 || elapsed < shortest)
                shortest = elapsed;
        }
        if (shortest >= targetNs)
            break;
        const std::size_t next = reps * 2;
        if (next > (std::size_t(1) << 24))
            break;
        reps = next;
    }
    return reps;
}

double observed_clock_resolution_ns() {
    double best = 1e12;
    for (int attempt = 0; attempt < 64; ++attempt)
    {
        const Clock::time_point start = Clock::now();
        Clock::time_point       stop  = Clock::now();
        while (stop == start)
            stop = Clock::now();
        const double delta =
          double(std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
        if (delta > 0.0 && delta < best)
            best = delta;
    }
    return best;
}

void touch_pages(const void* data, std::size_t bytes) {
    if (bytes == 0)
        return;
    const volatile std::uint8_t* cursor = static_cast<const volatile std::uint8_t*>(data);
    std::uint64_t                sum    = 0;
    for (std::size_t offset = 0; offset < bytes; offset += 4096)
        sum += cursor[offset];
    sum += cursor[bytes - 1];
    g_sink += i64(sum);
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

std::string json_number(double value) {
    std::ostringstream out;
    out << std::setprecision(10) << value;
    return out.str();
}

void emit_schedule(std::ostream& out, const ScheduleResult& result) {
    out << "{\"samples\":" << result.samples << ",\"inner_repetitions\":" << result.innerReps
        << ",\"pool_slots\":" << result.poolCount
        << ",\"median_ns\":" << json_number(result.median)
        << ",\"mad_ns\":" << json_number(result.mad) << ",\"ci95_median_ns\":["
        << json_number(result.ci.low) << ',' << json_number(result.ci.high) << "]}";
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

struct Options {
    std::uint64_t seed        = 0x5EED1234u;
    std::size_t   repetitions = 201;
    std::string   outputPath;
    std::string   networkPath;
    bool          quick   = false;
    bool          verify  = false;
};

struct Workspace {
    // Pools. Every one is filled and page-touched before any timing starts.
    std::vector<V3AccumulatorFrame> copySource;
    std::vector<V3AccumulatorFrame> copyTarget;

    std::vector<V3Features>         zeroFeatures;
    std::vector<V3Features>         singleFeatures;
    std::vector<V3Features>         maxFeatures;
    std::vector<V3AccumulatorFrame> refreshFrames;

    std::vector<std::array<Piece, SQUARE_NB>> singleBoards;
    std::vector<std::array<Piece, SQUARE_NB>> maxBoards;

    std::vector<V3AccumulatorFrame> quietSource;
    std::vector<V3AccumulatorFrame> quietChild;
    std::vector<HordeV2::NormalizedDirty> quietDirty;
    std::vector<V3Transition>             quietTransition;
    std::vector<QuietCase>                quietCases;

    std::vector<V3AccumulatorFrame>       contextualSource;
    std::vector<V3AccumulatorFrame>       contextualChild;
    std::vector<HordeV2::NormalizedDirty> contextualDirty;
    std::vector<V3Transition>             contextualTransition;
    std::vector<ContextualCase>           contextualCases;

    std::vector<V3AccumulatorFrame> denseFrames;
    std::vector<V3DenseScratch>     denseScratch;
    std::vector<Color>              denseSideToMove;
    std::vector<int>                denseRule50;
    std::vector<V3Features>         denseFeatures;
    // The composed case refreshes from the maximum-row boards, so every pool
    // slot does identical work and the hot and streaming schedules stay
    // comparable, and so the composed median can be checked against
    // full_refresh_rows_max plus dense_propagation.
    std::vector<V3AccumulatorFrame> composedFrames;

    // The two-gather reference.
    std::vector<std::array<IndexType, TwoGatherRowCount>> v3Rows;
    std::vector<std::array<IndexType, TwoGatherRowCount>> perspectiveARows;
    std::vector<std::array<IndexType, TwoGatherRowCount>> perspectiveBRows;
};

std::size_t g_poolSlots = 64;

}  // namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Options options;

    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        if (argument == "--seed" && index + 1 < argc)
            options.seed = std::strtoull(argv[++index], nullptr, 0);
        else if (argument == "--repetitions" && index + 1 < argc)
            options.repetitions = std::size_t(std::strtoull(argv[++index], nullptr, 0));
        else if (argument == "--output" && index + 1 < argc)
            options.outputPath = argv[++index];
        else if (argument == "--network" && index + 1 < argc)
            options.networkPath = argv[++index];
        else if (argument == "--quick")
            options.quick = true;
        else if (argument == "--verify")
            options.verify = true;
        else
        {
            std::cerr << "usage: horde-v3-timing [--seed N] [--repetitions N] [--output FILE]"
                         " [--network FILE] [--quick] [--verify]\n";
            return 2;
        }
    }

    bool repetitionsGiven = false;
    for (int index = 1; index < argc; ++index)
        if (std::string(argv[index]) == "--repetitions")
            repetitionsGiven = true;

    if (options.quick)
    {
        if (!repetitionsGiven)
            options.repetitions = 15;
        g_poolSlots = 16;
    }
    require(options.repetitions >= 3, "at least three repetitions are needed for a median");

    const double      targetBatchNs = options.quick ? 20000.0 : 200000.0;
    const std::size_t resamples     = options.quick ? 400 : 2000;

    // -----------------------------------------------------------------------
    // Parameters
    // -----------------------------------------------------------------------

    V3Parameters fixtureParameters;
    std::string  networkSource;
    std::string  parameterDigest;

    V3ContainerLoadResult loaded;
    if (!options.networkPath.empty())
    {
        loaded = load_v3_container(options.networkPath);
        if (!loaded)
        {
            std::cerr << v3_container_load_error_name(loaded.error) << ": " << loaded.message
                      << '\n';
            return 3;
        }
        networkSource   = options.networkPath;
        parameterDigest = loaded.parameters.parameterSha256;
    }
    else
    {
        build_fixture_parameters(fixtureParameters);
        networkSource = "in-process deterministic fixture, a mirror of "
                        "tests/horde_v3_parity.py::deterministic_sections";
    }

    const V3Parameters& parameters =
      options.networkPath.empty() ? fixtureParameters : loaded.parameters;
    require(parameters.valid(), "the network parameters did not pass V3Parameters::valid()");

    const V3Network<V3DefaultKernels> network(parameters);
    const V3Network<V3ScalarKernels>  scalarNetwork(parameters);
    V3AccumulatorStack<V3DefaultKernels> stack(network);
    V3AccumulatorStack<V3ScalarKernels>  scalarStack(scalarNetwork);

    // -----------------------------------------------------------------------
    // Workspace construction. Everything here is outside every timed region.
    // -----------------------------------------------------------------------

    // The streaming stride must be inside the pool and coprime with it, so the
    // walk visits every slot before it repeats and never revisits the same
    // cache lines in a short cycle.
    const std::size_t pool = g_poolSlots;
    const auto        gcd  = [](std::size_t left, std::size_t right) {
        while (right)
        {
            const std::size_t rest = left % right;
            left                   = right;
            right                  = rest;
        }
        return left;
    };
    std::size_t stride = pool > 2 ? pool / 2 + 1 : 1;
    while (stride > 1 && gcd(stride, pool) != 1)
        --stride;
    require(pool >= 2 && stride >= 1 && stride < pool && gcd(stride, pool) == 1,
            "the streaming stride is not a coprime walk over the pool");
    Workspace work;

    // Case 1: frame copy.
    work.copySource.resize(pool);
    work.copyTarget.resize(pool);

    // Case 2/3/4: refresh at 0, at 1 and at the maximum row count.
    work.zeroFeatures.assign(pool, V3Features{});
    work.singleBoards.resize(pool);
    work.singleFeatures.resize(pool);
    work.maxBoards.resize(pool);
    work.maxFeatures.resize(pool);
    work.refreshFrames.resize(pool);

    std::size_t maxActiveRows     = 0;
    std::size_t maxG0Rows         = 0;
    std::size_t maxContextualRows = 0;

    for (std::size_t slot = 0; slot < pool; ++slot)
    {
        work.singleBoards[slot]   = single_row_board(slot);
        work.singleFeatures[slot] = extract_v3_features(work.singleBoards[slot]);
        require(work.singleFeatures[slot].valid(), "the one-row board was rejected");
        require(work.singleFeatures[slot].size == 1,
                "the one-row board produced " + std::to_string(work.singleFeatures[slot].size)
                  + " active rows");

        work.maxBoards[slot]   = max_row_board(slot);
        work.maxFeatures[slot] = extract_v3_features(work.maxBoards[slot]);
        require(work.maxFeatures[slot].valid(),
                std::string("the maximum-row board was rejected: ")
                  + v3_feature_error_name(work.maxFeatures[slot].error));
        maxActiveRows = std::max(maxActiveRows, work.maxFeatures[slot].size);
        maxG0Rows     = std::max(maxG0Rows, work.maxFeatures[slot].g0Size);
        maxContextualRows =
          std::max(maxContextualRows, work.maxFeatures[slot].size - work.maxFeatures[slot].g0Size);

        require(work.zeroFeatures[slot].valid() && work.zeroFeatures[slot].size == 0,
                "the zero-row feature set is not empty");
    }

    // Every board in the family must reach the frozen ceiling, otherwise the
    // "maximum" case is not the maximum and the spec has not closed.
    for (std::size_t slot = 0; slot < pool; ++slot)
        require(work.maxFeatures[slot].size == MaxActiveRows,
                "maximum-row board " + std::to_string(slot) + " produced "
                  + std::to_string(work.maxFeatures[slot].size) + " rows, not "
                  + std::to_string(MaxActiveRows));

    // Case 5: quiet transition, G0 delta only.
    work.quietCases.resize(pool);
    work.quietSource.resize(pool);
    work.quietChild.resize(pool);
    work.quietDirty.resize(pool);
    work.quietTransition.resize(pool);
    std::size_t quietG0Removed = 0;
    std::size_t quietG0Added   = 0;

    for (std::size_t slot = 0; slot < pool; ++slot)
    {
        work.quietCases[slot] = quiet_case(slot);
        require(network.full_refresh(work.quietSource[slot], work.quietCases[slot].board),
                "the quiet-transition board was rejected");
        require(HordeV2::normalize_dirty_piece(work.quietCases[slot].dirty, work.quietDirty[slot]),
                "the quiet DirtyPiece did not normalize");
        require(stack.materialize_child(work.quietChild[slot], work.quietTransition[slot],
                                        work.quietSource[slot], work.quietDirty[slot])
                  == V3StackError::NONE,
                "the quiet transition was rejected");
        require(work.quietTransition[slot].removedSize == 0
                  && work.quietTransition[slot].addedSize == 0,
                "the quiet transition moved contextual rows, so it is not a G0-only delta");
        quietG0Removed = work.quietTransition[slot].g0RemovedSize;
        quietG0Added   = work.quietTransition[slot].g0AddedSize;
        require(quietG0Removed == 1 && quietG0Added == 1,
                "the quiet transition is not a single-piece move");
    }

    // Case 6: contextual transition.
    work.contextualCases.resize(pool);
    work.contextualSource.resize(pool);
    work.contextualChild.resize(pool);
    work.contextualDirty.resize(pool);
    work.contextualTransition.resize(pool);
    std::size_t contextualRemovedMin = MaxContextualRows + 1;
    std::size_t contextualRemovedMax = 0;
    std::size_t contextualAddedMin   = MaxContextualRows + 1;
    std::size_t contextualAddedMax   = 0;

    for (std::size_t slot = 0; slot < pool; ++slot)
    {
        work.contextualCases[slot] = contextual_case(slot);
        require(network.full_refresh(work.contextualSource[slot], work.contextualCases[slot].board),
                "the contextual-transition board was rejected");
        require(HordeV2::normalize_dirty_piece(work.contextualCases[slot].dirty,
                                               work.contextualDirty[slot]),
                "the contextual DirtyPiece did not normalize");
        require(stack.materialize_child(work.contextualChild[slot], work.contextualTransition[slot],
                                        work.contextualSource[slot], work.contextualDirty[slot])
                  == V3StackError::NONE,
                "the contextual transition was rejected");
        require(work.contextualTransition[slot].removedSize
                    + work.contextualTransition[slot].addedSize
                  > 0,
                "the contextual transition moved no contextual row");
        contextualRemovedMin =
          std::min(contextualRemovedMin, work.contextualTransition[slot].removedSize);
        contextualRemovedMax =
          std::max(contextualRemovedMax, work.contextualTransition[slot].removedSize);
        contextualAddedMin = std::min(contextualAddedMin, work.contextualTransition[slot].addedSize);
        contextualAddedMax = std::max(contextualAddedMax, work.contextualTransition[slot].addedSize);
    }

    // Case 7/8: dense propagation and composed evaluation.
    work.denseFrames.resize(pool);
    work.denseScratch.resize(pool);
    work.denseSideToMove.resize(pool);
    work.denseRule50.resize(pool);
    work.denseFeatures.resize(pool);
    work.composedFrames.resize(pool);

    std::vector<PositionState> parityPositions;
    for (const char* fen : ParityFixtureFens)
    {
        PositionState position{};
        require(parse_fen(fen, position), std::string("parity fixture FEN did not parse: ") + fen);
        parityPositions.push_back(position);
    }

    for (std::size_t slot = 0; slot < pool; ++slot)
    {
        // Dense propagation walks the parity fixtures: the trunk's work is the
        // same for every frame, but the phase bucket differs, so the streaming
        // schedule walks distinct regions of the hidden0 table as well.
        const PositionState& position = parityPositions[slot % parityPositions.size()];
        work.denseFeatures[slot]      = extract_v3_features(position.board);
        require(work.denseFeatures[slot].valid(), "a parity fixture was rejected by the enumerator");
        network.full_refresh(work.denseFrames[slot], work.denseFeatures[slot]);
        network.full_refresh(work.composedFrames[slot], work.maxFeatures[slot]);
        work.denseSideToMove[slot] = (slot % 2) ? BLACK : WHITE;
        work.denseRule50[slot]     = int((slot * 7) % 101);
    }

    // Case 9: the two-gather reference. Each pool slot takes the 52 G0 rows of
    // a distinct maximum-row board, so the row lists are real piece rows, all
    // exactly 52 long, and spread over the G0 range.
    work.v3Rows.resize(pool);
    work.perspectiveARows.resize(pool);
    work.perspectiveBRows.resize(pool);
    for (std::size_t slot = 0; slot < pool; ++slot)
    {
        const V3Features& features = work.maxFeatures[slot];
        require(features.g0Size == TwoGatherRowCount,
                "a maximum-row board carries " + std::to_string(features.g0Size)
                  + " G0 rows, not " + std::to_string(TwoGatherRowCount));
        for (std::size_t index = 0; index < TwoGatherRowCount; ++index)
        {
            const IndexType row              = features.rows[index];
            work.v3Rows[slot][index]         = row;
            work.perspectiveARows[slot][index] = row;
            // The mirrored perspective, the way the incumbent's second view
            // indexes the same piece: same fixed role, vertically flipped
            // square. A bijection, so the list stays 52 distinct rows.
            work.perspectiveBRows[slot][index] =
              IndexType((row / 64) * 64 + ((row % 64) ^ 56));
        }
    }

    // The two 512-lane tables. Their bytes sum to exactly the V3 table's bytes
    // (896 * 1024 * 2 == 2 * 896 * 512 * 2), they use the same aligned
    // allocator, and they are filled deterministically.
    HordeV2::AlignedBuffer<FtWeight>   perspectiveATable(std::size_t(V3StreamRows) * IncumbentLanes);
    HordeV2::AlignedBuffer<FtWeight>   perspectiveBTable(std::size_t(V3StreamRows) * IncumbentLanes);
    HordeV2::AlignedBuffer<Accumulator> perspectiveABias(IncumbentLanes);
    HordeV2::AlignedBuffer<Accumulator> perspectiveBBias(IncumbentLanes);
    for (std::size_t index = 0; index < perspectiveATable.size(); ++index)
    {
        perspectiveATable[index] = FtWeight(((i64(index) * 97 + 11) % 63) - 31);
        perspectiveBTable[index] = FtWeight(((i64(index) * 89 + 29) % 63) - 31);
    }
    for (std::size_t lane = 0; lane < IncumbentLanes; ++lane)
    {
        perspectiveABias[lane] = Accumulator(parameters.ftBias[lane]);
        perspectiveBBias[lane] = Accumulator(parameters.ftBias[IncumbentLanes + lane]);
    }

    HordeV2::AlignedBuffer<Accumulator> v3Accumulators(pool * V3Lanes);
    HordeV2::AlignedBuffer<Accumulator> incumbentAccumulators(pool * V3Lanes);
    for (std::size_t index = 0; index < v3Accumulators.size(); ++index)
    {
        v3Accumulators[index]        = 0;
        incumbentAccumulators[index] = 0;
    }

    // -----------------------------------------------------------------------
    // --verify: the timed kernels against the parity path
    // -----------------------------------------------------------------------

    if (options.verify)
    {
        std::size_t checkedRefresh     = 0;
        std::size_t checkedTransitions = 0;
        std::size_t checkedPropagation = 0;
        std::size_t checkedGathers     = 0;

        const auto same_evaluation = [&](const V3AccumulatorFrame& left,
                                         const V3AccumulatorFrame& right, Color side, int rule50,
                                         const std::string& label) {
            V3DenseScratch     leftScratch{};
            V3DenseScratch     rightScratch{};
            const V3EvalResult leftResult  = network.propagate(left, leftScratch, side, rule50);
            const V3EvalResult rightResult = network.propagate(right, rightScratch, side, rule50);
            require(left.accumulator == right.accumulator, label + ": accumulator differs");
            require(left.psqt == right.psqt, label + ": psqt differs");
            require(leftResult.bucket == rightResult.bucket, label + ": bucket differs");
            require(leftResult.outputAffine == rightResult.outputAffine, label + ": out_pre differs");
            require(leftResult.psqtSum == rightResult.psqtSum, label + ": psqt_sum differs");
            require(leftResult.preRule50Value == rightResult.preRule50Value,
                    label + ": pre-rule50 differs");
            require(leftResult.value == rightResult.value, label + ": value differs");
        };

        // Every pooled feature set must still be what the enumerator produces
        // for its board, and every pooled frame must still be what a fresh
        // full refresh produces.
        const auto check_board = [&](const std::array<Piece, SQUARE_NB>& board,
                                     const V3Features&                   pooled,
                                     const V3AccumulatorFrame&           pooledFrame,
                                     const std::string&                  label) {
            const V3Features fresh = extract_v3_features(board);
            require(fresh.valid(), label + ": the enumerator rejected the board");
            require(fresh.size == pooled.size, label + ": active row count drifted");
            for (std::size_t index = 0; index < fresh.size; ++index)
                require(fresh.rows[index] == pooled.rows[index], label + ": active row drifted");
            V3AccumulatorFrame reference{};
            network.full_refresh(reference, fresh);
            same_evaluation(pooledFrame, reference, WHITE, 0, label);
            same_evaluation(pooledFrame, reference, BLACK, 37, label);
            ++checkedRefresh;
        };

        for (std::size_t slot = 0; slot < pool; ++slot)
        {
            V3AccumulatorFrame frame{};
            network.full_refresh(frame, work.singleFeatures[slot]);
            check_board(work.singleBoards[slot], work.singleFeatures[slot], frame,
                        "one-row board " + std::to_string(slot));

            check_board(work.maxBoards[slot], work.maxFeatures[slot], work.composedFrames[slot],
                        "maximum-row board " + std::to_string(slot));

            const PositionState& position = parityPositions[slot % parityPositions.size()];
            check_board(position.board, work.denseFeatures[slot], work.denseFrames[slot],
                        "parity fixture " + std::to_string(slot % parityPositions.size()));
            ++checkedPropagation;
        }

        // The zero-row refresh must be exactly the bias, with a zero PSQT.
        {
            V3AccumulatorFrame frame{};
            network.full_refresh(frame, work.zeroFeatures[0]);
            for (std::size_t lane = 0; lane < V3Lanes; ++lane)
                require(frame.accumulator[lane] == Accumulator(parameters.ftBias[lane]),
                        "the zero-row refresh is not the feature-transformer bias");
            for (std::size_t column = 0; column < V3PsqtColumns; ++column)
                require(frame.psqt[column] == 0, "the zero-row refresh left a non-zero PSQT");
            ++checkedRefresh;
        }

        // Every timed transition must equal a full refresh of the post-move
        // board. This is the gate-3 property, restated for the timed frames.
        const auto check_transition = [&](const std::array<Piece, SQUARE_NB>& board,
                                          const DirtyPiece&                   dirty,
                                          const V3AccumulatorFrame&           child,
                                          const std::string&                  label) {
            std::array<Piece, SQUARE_NB> target{};
            require(try_move(board, dirty, target), label + ": the transition is not applicable");
            V3AccumulatorFrame reference{};
            require(network.full_refresh(reference, target),
                    label + ": the post-move board was rejected");
            same_evaluation(child, reference, WHITE, 0, label);
            same_evaluation(child, reference, BLACK, 5, label);
            ++checkedTransitions;
        };

        for (std::size_t slot = 0; slot < pool; ++slot)
        {
            check_transition(work.quietCases[slot].board, work.quietCases[slot].dirty,
                             work.quietChild[slot], "quiet transition " + std::to_string(slot));
            check_transition(work.contextualCases[slot].board, work.contextualCases[slot].dirty,
                             work.contextualChild[slot],
                             "contextual transition " + std::to_string(slot));
        }

        // The scalar kernels and the selected kernels must agree, so an AVX2
        // timing build cannot be timing a different arithmetic.
        for (std::size_t slot = 0; slot < pool; ++slot)
        {
            V3AccumulatorFrame selected{};
            V3AccumulatorFrame plain{};
            network.full_refresh(selected, work.maxFeatures[slot]);
            scalarNetwork.full_refresh(plain, work.maxFeatures[slot]);
            require(selected.accumulator == plain.accumulator,
                    "scalar and the selected backend disagree on the accumulator");
            require(selected.psqt == plain.psqt,
                    "scalar and the selected backend disagree on the PSQT");

            V3DenseScratch     selectedScratch{};
            V3DenseScratch     plainScratch{};
            const V3EvalResult selectedResult =
              network.propagate(selected, selectedScratch, WHITE, 0);
            const V3EvalResult plainResult = scalarNetwork.propagate(plain, plainScratch, WHITE, 0);
            require(selectedScratch.transformed == plainScratch.transformed,
                    "scalar and the selected backend disagree on the transformed layer");
            require(selectedResult.value == plainResult.value,
                    "scalar and the selected backend disagree on the value");

            V3AccumulatorFrame selectedChild{};
            V3AccumulatorFrame plainChild{};
            V3Transition       selectedTransition{};
            V3Transition       plainTransition{};
            require(stack.materialize_child(selectedChild, selectedTransition,
                                            work.contextualSource[slot],
                                            work.contextualDirty[slot])
                      == V3StackError::NONE,
                    "the selected backend rejected the contextual transition");
            require(scalarStack.materialize_child(plainChild, plainTransition,
                                                  work.contextualSource[slot],
                                                  work.contextualDirty[slot])
                      == V3StackError::NONE,
                    "the scalar backend rejected the contextual transition");
            require(selectedChild.accumulator == plainChild.accumulator,
                    "scalar and the selected backend disagree on an incremental accumulator");
        }

        // The lane-parametric primitive at 1024 lanes must be bit-identical to
        // the backend's own add_ft, otherwise the two-gather comparison is not
        // running the V3 kernel at all.
        for (std::size_t slot = 0; slot < pool; ++slot)
        {
            alignas(64) std::array<Accumulator, V3Lanes> viaBackend{};
            alignas(64) std::array<Accumulator, V3Lanes> viaLaneKernel{};
            for (std::size_t lane = 0; lane < V3Lanes; ++lane)
            {
                viaBackend[lane]    = Accumulator(parameters.ftBias[lane]);
                viaLaneKernel[lane] = Accumulator(parameters.ftBias[lane]);
            }
            for (std::size_t index = 0; index < TwoGatherRowCount; ++index)
            {
                const FtWeight* row =
                  parameters.ftWeights.data() + std::size_t(work.v3Rows[slot][index]) * V3Lanes;
                V3DefaultKernels::add_ft(viaBackend.data(), row);
                LaneKernels<V3Lanes>::add_ft(viaLaneKernel.data(), row);
            }
            require(viaBackend == viaLaneKernel,
                    "the lane-parametric add_ft is not bit-identical to the backend's add_ft");

            // The 2x512 emulation must equal an independent scalar sum over
            // the same rows and the same tables.
            alignas(64) std::array<Accumulator, IncumbentLanes> emulatedA{};
            alignas(64) std::array<Accumulator, IncumbentLanes> emulatedB{};
            ft_refresh<IncumbentLanes>(emulatedA.data(), perspectiveABias.data(),
                                       perspectiveATable.data(), work.perspectiveARows[slot].data(),
                                       TwoGatherRowCount);
            ft_refresh<IncumbentLanes>(emulatedB.data(), perspectiveBBias.data(),
                                       perspectiveBTable.data(), work.perspectiveBRows[slot].data(),
                                       TwoGatherRowCount);
            for (std::size_t lane = 0; lane < IncumbentLanes; ++lane)
            {
                Accumulator referenceA = perspectiveABias[lane];
                Accumulator referenceB = perspectiveBBias[lane];
                for (std::size_t index = 0; index < TwoGatherRowCount; ++index)
                {
                    referenceA += Accumulator(
                      perspectiveATable[std::size_t(work.perspectiveARows[slot][index])
                                          * IncumbentLanes
                                        + lane]);
                    referenceB += Accumulator(
                      perspectiveBTable[std::size_t(work.perspectiveBRows[slot][index])
                                          * IncumbentLanes
                                        + lane]);
                }
                require(emulatedA[lane] == referenceA && emulatedB[lane] == referenceB,
                        "the 2x512 emulation disagrees with an independent scalar sum");
            }
            ++checkedGathers;
        }

        std::ostream& out = std::cout;
        out << "{\"schema\":\"HORDE_V3_KERNEL_TIMING_VERIFY_V1\""
            << ",\"status\":\"pass\""
            << ",\"backend\":\"" << V3DefaultBackendName << "\""
            << ",\"avx2_available\":" << (V3Avx2Available ? "true" : "false")
            << ",\"network_source\":\"" << networkSource << "\""
            << ",\"pool_slots\":" << pool << ",\"refresh_checks\":" << checkedRefresh
            << ",\"transition_checks\":" << checkedTransitions
            << ",\"propagation_checks\":" << checkedPropagation
            << ",\"gather_checks\":" << checkedGathers
            << ",\"max_active_rows\":" << maxActiveRows << "}\n";
        std::cerr << "verify: pass (" << checkedRefresh << " refresh, " << checkedTransitions
                  << " transition, " << checkedGathers << " gather checks)\n";
        return 0;
    }

    // -----------------------------------------------------------------------
    // The cases
    // -----------------------------------------------------------------------

    std::vector<TimingCase> cases;

    const auto add_case = [&](const std::string& name, const std::string& description,
                              BatchFn batch) {
        TimingCase entry;
        entry.name            = name;
        entry.description     = description;
        entry.batch           = std::move(batch);
        entry.streamingPool   = pool;
        entry.streamingStride = stride;
        cases.push_back(std::move(entry));
    };

    add_case("frame_copy", "V3AccumulatorFrame assignment, source and target both walked",
             [&](std::size_t reps, std::size_t poolCount, std::size_t walkStride,
                 std::size_t& cursor) {
                 return time_batch(
                   [&](std::size_t slot) {
                       work.copyTarget[slot] = work.copySource[slot];
                       escape(&work.copyTarget[slot]);
                   },
                   reps, poolCount, walkStride, cursor);
             });

    add_case("full_refresh_rows_0",
             "full refresh over an empty row set: the bias copy and the PSQT clear alone",
             [&](std::size_t reps, std::size_t poolCount, std::size_t walkStride,
                 std::size_t& cursor) {
                 return time_batch(
                   [&](std::size_t slot) {
                       network.full_refresh(work.refreshFrames[slot], work.zeroFeatures[slot]);
                       escape(&work.refreshFrames[slot]);
                   },
                   reps, poolCount, walkStride, cursor);
             });

    add_case("full_refresh_rows_1", "full refresh over one active row",
             [&](std::size_t reps, std::size_t poolCount, std::size_t walkStride,
                 std::size_t& cursor) {
                 return time_batch(
                   [&](std::size_t slot) {
                       network.full_refresh(work.refreshFrames[slot], work.singleFeatures[slot]);
                       escape(&work.refreshFrames[slot]);
                   },
                   reps, poolCount, walkStride, cursor);
             });

    add_case("full_refresh_rows_max", "full refresh over the maximum active row count",
             [&](std::size_t reps, std::size_t poolCount, std::size_t walkStride,
                 std::size_t& cursor) {
                 return time_batch(
                   [&](std::size_t slot) {
                       network.full_refresh(work.refreshFrames[slot], work.maxFeatures[slot]);
                       escape(&work.refreshFrames[slot]);
                   },
                   reps, poolCount, walkStride, cursor);
             });

    add_case("quiet_transition",
             "one G0-only transition: frame copy, one row subtracted, one row added, "
             "no contextual row moved",
             [&](std::size_t reps, std::size_t poolCount, std::size_t walkStride,
                 std::size_t& cursor) {
                 return time_batch(
                   [&](std::size_t slot) {
                       const V3StackError error = stack.materialize_child(
                         work.quietChild[slot], work.quietTransition[slot], work.quietSource[slot],
                         work.quietDirty[slot]);
                       g_sink += i64(error != V3StackError::NONE);
                       escape(&work.quietChild[slot]);
                   },
                   reps, poolCount, walkStride, cursor);
             });

    add_case("contextual_transition",
             "one transition that moves at least one file's contextual block",
             [&](std::size_t reps, std::size_t poolCount, std::size_t walkStride,
                 std::size_t& cursor) {
                 return time_batch(
                   [&](std::size_t slot) {
                       const V3StackError error = stack.materialize_child(
                         work.contextualChild[slot], work.contextualTransition[slot],
                         work.contextualSource[slot], work.contextualDirty[slot]);
                       g_sink += i64(error != V3StackError::NONE);
                       escape(&work.contextualChild[slot]);
                   },
                   reps, poolCount, walkStride, cursor);
             });

    add_case("dense_propagation",
             "trunk, PSQT skip and the rule50 postprocessor over a materialized accumulator",
             [&](std::size_t reps, std::size_t poolCount, std::size_t walkStride,
                 std::size_t& cursor) {
                 return time_batch(
                   [&](std::size_t slot) {
                       const V3EvalResult result =
                         network.propagate(work.denseFrames[slot], work.denseScratch[slot],
                                           work.denseSideToMove[slot], work.denseRule50[slot]);
                       g_sink += i64(result.value);
                       escape(&work.denseScratch[slot]);
                   },
                   reps, poolCount, walkStride, cursor);
             });

    add_case("full_evaluation",
             "composed: full refresh at the maximum row count followed by dense propagation",
             [&](std::size_t reps, std::size_t poolCount, std::size_t walkStride,
                 std::size_t& cursor) {
                 return time_batch(
                   [&](std::size_t slot) {
                       network.full_refresh(work.composedFrames[slot], work.maxFeatures[slot]);
                       const V3EvalResult result =
                         network.propagate(work.composedFrames[slot], work.denseScratch[slot],
                                           work.denseSideToMove[slot], work.denseRule50[slot]);
                       g_sink += i64(result.value);
                       escape(&work.composedFrames[slot]);
                   },
                   reps, poolCount, walkStride, cursor);
             });

    add_case("ft_refresh_v3_1x1024",
             "V3: one gather of 52 rows over 1024 lanes, feature transformer only",
             [&](std::size_t reps, std::size_t poolCount, std::size_t walkStride,
                 std::size_t& cursor) {
                 return time_batch(
                   [&](std::size_t slot) {
                       Accumulator* accumulator = v3Accumulators.data() + slot * V3Lanes;
                       ft_refresh<V3Lanes>(accumulator, parameters.ftBias.data(),
                                           parameters.ftWeights.data(), work.v3Rows[slot].data(),
                                           TwoGatherRowCount);
                       escape(accumulator);
                   },
                   reps, poolCount, walkStride, cursor);
             });

    add_case("ft_refresh_incumbent_2x512",
             "EMULATION of the incumbent's arithmetic shape: two gathers of 52 rows over "
             "512 lanes each, two tables whose bytes sum to the V3 table's bytes. Not the "
             "incumbent's code.",
             [&](std::size_t reps, std::size_t poolCount, std::size_t walkStride,
                 std::size_t& cursor) {
                 return time_batch(
                   [&](std::size_t slot) {
                       Accumulator* first  = incumbentAccumulators.data() + slot * V3Lanes;
                       Accumulator* second = first + IncumbentLanes;
                       ft_refresh<IncumbentLanes>(first, perspectiveABias.data(),
                                                  perspectiveATable.data(),
                                                  work.perspectiveARows[slot].data(),
                                                  TwoGatherRowCount);
                       ft_refresh<IncumbentLanes>(second, perspectiveBBias.data(),
                                                  perspectiveBTable.data(),
                                                  work.perspectiveBRows[slot].data(),
                                                  TwoGatherRowCount);
                       escape(first);
                   },
                   reps, poolCount, walkStride, cursor);
             });

    // -----------------------------------------------------------------------
    // Page touching. Every table and every pool, once, before any timing.
    // -----------------------------------------------------------------------

    touch_pages(parameters.ftWeights.data(), parameters.ftWeights.size() * sizeof(FtWeight));
    touch_pages(parameters.psqtWeights.data(), parameters.psqtWeights.size() * sizeof(PsqtWeight));
    touch_pages(parameters.hidden0Weights.data(),
                parameters.hidden0Weights.size() * sizeof(DenseWeight));
    touch_pages(parameters.hidden1Weights.data(),
                parameters.hidden1Weights.size() * sizeof(DenseWeight));
    touch_pages(parameters.outputWeights.data(),
                parameters.outputWeights.size() * sizeof(DenseWeight));
    touch_pages(parameters.ftBias.data(), parameters.ftBias.size() * sizeof(AffineBias));
    touch_pages(perspectiveATable.data(), perspectiveATable.size() * sizeof(FtWeight));
    touch_pages(perspectiveBTable.data(), perspectiveBTable.size() * sizeof(FtWeight));
    touch_pages(v3Accumulators.data(), v3Accumulators.size() * sizeof(Accumulator));
    touch_pages(incumbentAccumulators.data(), incumbentAccumulators.size() * sizeof(Accumulator));

    const auto touch_frames = [](const std::vector<V3AccumulatorFrame>& frames) {
        touch_pages(frames.data(), frames.size() * sizeof(V3AccumulatorFrame));
    };
    touch_frames(work.copySource);
    touch_frames(work.copyTarget);
    touch_frames(work.refreshFrames);
    touch_frames(work.quietSource);
    touch_frames(work.quietChild);
    touch_frames(work.contextualSource);
    touch_frames(work.contextualChild);
    touch_frames(work.denseFrames);
    touch_frames(work.composedFrames);
    touch_pages(work.denseScratch.data(), work.denseScratch.size() * sizeof(V3DenseScratch));
    touch_pages(work.maxFeatures.data(), work.maxFeatures.size() * sizeof(V3Features));
    touch_pages(work.singleFeatures.data(), work.singleFeatures.size() * sizeof(V3Features));
    touch_pages(work.denseFeatures.data(), work.denseFeatures.size() * sizeof(V3Features));

    const double clockResolutionNs = observed_clock_resolution_ns();

    // -----------------------------------------------------------------------
    // Calibration, then the paired rounds
    // -----------------------------------------------------------------------

    struct Schedule {
        const char* name;
        bool        streaming;
    };
    const std::array<Schedule, 2> schedules = {{{"hot", false}, {"streaming", true}}};

    for (const Schedule& schedule : schedules)
    {
        const std::size_t poolCount  = schedule.streaming ? pool : 1;
        const std::size_t walkStride = schedule.streaming ? stride : 0;

        for (TimingCase& entry : cases)
        {
            const std::size_t reps =
              calibrate(entry.batch, poolCount, walkStride, targetBatchNs);
            ScheduleResult& result = schedule.streaming ? entry.streaming : entry.hot;
            result.innerReps       = reps;
            result.poolCount       = poolCount;
            result.perOperationNs.reserve(options.repetitions);
        }

        std::vector<std::size_t> order(cases.size());
        std::iota(order.begin(), order.end(), std::size_t(0));
        std::mt19937_64 rng(options.seed ^ (schedule.streaming ? 0x9E3779B97F4A7C15ull : 0ull));
        std::vector<std::size_t> cursors(cases.size(), 0);

        for (std::size_t round = 0; round < options.repetitions; ++round)
        {
            std::shuffle(order.begin(), order.end(), rng);
            for (const std::size_t index : order)
            {
                TimingCase&     entry  = cases[index];
                ScheduleResult& result = schedule.streaming ? entry.streaming : entry.hot;
                const double    elapsed =
                  entry.batch(result.innerReps, poolCount, walkStride, cursors[index]);
                result.perOperationNs.push_back(elapsed / double(result.innerReps));
            }
        }

        const BootstrapDraws draws = make_draws(
          options.repetitions, resamples,
          options.seed ^ (schedule.streaming ? 0xC2B2AE3D27D4EB4Full : 0x165667B19E3779F9ull));

        for (TimingCase& entry : cases)
        {
            ScheduleResult& result = schedule.streaming ? entry.streaming : entry.hot;
            result.samples         = result.perOperationNs.size();
            result.median          = median_of(result.perOperationNs);
            result.mad = median_absolute_deviation(result.perOperationNs, result.median);
            result.ci  = median_interval(result.perOperationNs, draws);
        }
    }

    // The two-gather ratio, bootstrapped over the same rounds as the per-case
    // intervals, so the interval is genuinely paired.
    const auto find_case = [&](const std::string& name) -> const TimingCase& {
        for (const TimingCase& entry : cases)
            if (entry.name == name)
                return entry;
        fail("case " + name + " is missing");
    };
    const TimingCase& v3Case        = find_case("ft_refresh_v3_1x1024");
    const TimingCase& incumbentCase = find_case("ft_refresh_incumbent_2x512");

    struct RatioReport {
        double   ratio = 0.0;
        Interval ci{};
    };
    std::array<RatioReport, 2> ratios{};
    for (std::size_t index = 0; index < schedules.size(); ++index)
    {
        const bool            streaming = schedules[index].streaming;
        const ScheduleResult& v3Result  = streaming ? v3Case.streaming : v3Case.hot;
        const ScheduleResult& incumbentResult =
          streaming ? incumbentCase.streaming : incumbentCase.hot;
        const BootstrapDraws draws =
          make_draws(options.repetitions, resamples,
                     options.seed ^ (streaming ? 0xC2B2AE3D27D4EB4Full : 0x165667B19E3779F9ull));
        ratios[index].ratio =
          v3Result.median > 0.0 ? incumbentResult.median / v3Result.median : 0.0;
        ratios[index].ci =
          ratio_interval(incumbentResult.perOperationNs, v3Result.perOperationNs, draws);
    }

    // -----------------------------------------------------------------------
    // Report
    // -----------------------------------------------------------------------

    std::ostringstream json;
    json << "{\"schema\":\"HORDE_V3_KERNEL_TIMING_V1\""
         << ",\"backend\":\"" << V3DefaultBackendName << "\""
         << ",\"avx2_available\":" << (V3Avx2Available ? "true" : "false")
#if defined(NDEBUG)
         << ",\"asserts_enabled\":false"
#else
         << ",\"asserts_enabled\":true"
#endif
         << ",\"seed\":" << options.seed << ",\"repetitions\":" << options.repetitions
         << ",\"quick\":" << (options.quick ? "true" : "false")
         << ",\"bootstrap_resamples\":" << resamples
         << ",\"target_batch_ns\":" << json_number(targetBatchNs)
         << ",\"clock\":{\"source\":\"std::chrono::steady_clock\",\"is_steady\":true"
         << ",\"period_num\":" << Clock::period::num << ",\"period_den\":" << Clock::period::den
         << ",\"observed_resolution_ns\":" << json_number(clockResolutionNs) << "}"
         << ",\"threads\":1"
         << ",\"network\":{\"source\":\"" << networkSource << "\",\"parameter_sha256\":\""
         << parameterDigest << "\"}"
         << ",\"geometry\":{\"lanes\":" << V3Lanes << ",\"stream_rows\":" << V3StreamRows
         << ",\"psqt_columns\":" << V3PsqtColumns
         << ",\"ft_table_bytes\":" << (std::size_t(V3StreamRows) * V3Lanes * sizeof(FtWeight))
         << ",\"frame_bytes\":" << sizeof(V3AccumulatorFrame)
         << ",\"scratch_bytes\":" << sizeof(V3DenseScratch) << "}"
         << ",\"row_counts\":{\"refresh_zero\":0,\"refresh_one\":1"
         << ",\"refresh_max\":" << maxActiveRows << ",\"refresh_max_g0\":" << maxG0Rows
         << ",\"refresh_max_contextual\":" << maxContextualRows
         << ",\"frozen_ceiling\":" << MaxActiveRows << ",\"max_row_board_fen\":\""
         << board_fen(work.maxBoards[0]) << "\""
         << ",\"zero_row_note\":\"a legal board always carries the Black king, so one is the "
            "smallest board-reachable row count; the zero-row case is a V3Features with an "
            "empty row set and isolates the bias copy plus the PSQT clear\"}"
         << ",\"transitions\":{\"quiet\":{\"g0_removed\":" << quietG0Removed
         << ",\"g0_added\":" << quietG0Added
         << ",\"contextual_removed\":0,\"contextual_added\":0}"
         << ",\"contextual\":{\"g0_removed\":1,\"g0_added\":1"
         << ",\"contextual_removed_first\":" << work.contextualTransition[0].removedSize
         << ",\"contextual_added_first\":" << work.contextualTransition[0].addedSize
         << ",\"contextual_removed_min\":" << contextualRemovedMin
         << ",\"contextual_removed_max\":" << contextualRemovedMax
         << ",\"contextual_added_min\":" << contextualAddedMin
         << ",\"contextual_added_max\":" << contextualAddedMax << "}}"
         << ",\"schedules\":{\"hot\":{\"pool_slots\":1,\"stride\":0"
            ",\"note\":\"one pinned slot, L1 resident\"}"
            ",\"streaming\":{\"pool_slots\":"
         << pool << ",\"stride\":" << stride
         << ",\"note\":\"strided walk over distinct boards, frames and table regions; the same "
            "loop body as the hot schedule, so no bookkeeping asymmetry\"}}"
         << ",\"cases\":[";

    for (std::size_t index = 0; index < cases.size(); ++index)
    {
        const TimingCase& entry = cases[index];
        if (index)
            json << ',';
        json << "{\"name\":\"" << entry.name << "\",\"description\":\"" << entry.description
             << "\",\"unit\":\"ns_per_operation\",\"hot\":";
        emit_schedule(json, entry.hot);
        json << ",\"streaming\":";
        emit_schedule(json, entry.streaming);
        json << '}';
    }

    json << "],\"two_gather_reference\":{"
         << "\"claim\":\"V3 at 1024 lanes performs the same accumulator arithmetic as two "
            "512-lane perspectives while doing one gather instead of two\""
         << ",\"emulation_disclaimer\":\"ft_refresh_incumbent_2x512 is an EMULATION of the "
            "incumbent's arithmetic shape, not the incumbent's actual code: two perspectives of "
            "512 lanes, two independent row-index streams, two feature tables whose bytes sum "
            "exactly to the V3 table's bytes, driven by the same lane-parametric kernel "
            "primitive as the V3 side\""
         << ",\"rows_per_gather\":" << TwoGatherRowCount
         << ",\"v3_lanes\":" << V3Lanes << ",\"incumbent_lanes_per_perspective\":" << IncumbentLanes
         << ",\"lane_adds_v3\":" << (TwoGatherRowCount * V3Lanes)
         << ",\"lane_adds_incumbent\":" << (2 * TwoGatherRowCount * IncumbentLanes)
         << ",\"table_bytes_v3\":" << (std::size_t(V3StreamRows) * V3Lanes * sizeof(FtWeight))
         << ",\"table_bytes_incumbent\":"
         << (2 * std::size_t(V3StreamRows) * IncumbentLanes * sizeof(FtWeight))
         << ",\"psqt_included\":false"
         << ",\"ratio_definition\":\"incumbent_2x512 median / v3_1x1024 median; above one means "
            "the single gather is faster\""
         << ",\"hot\":{\"ratio\":" << json_number(ratios[0].ratio) << ",\"ci95\":["
         << json_number(ratios[0].ci.low) << ',' << json_number(ratios[0].ci.high) << "]}"
         << ",\"streaming\":{\"ratio\":" << json_number(ratios[1].ratio) << ",\"ci95\":["
         << json_number(ratios[1].ci.low) << ',' << json_number(ratios[1].ci.high) << "]}}"
         << ",\"notes\":["
         << "\"feature enumeration is outside every timed region; the refresh cases time the "
            "accumulation over a pre-enumerated row set\""
         << ",\"the transition cases include the frame copy the transition performs; subtract "
            "frame_copy to isolate the delta\""
         << ",\"page faults, allocation, validation and logging are outside every timed region; "
            "every table and pool page is touched once before timing\""
         << ",\"every pool slot of a given case does identical work, so the hot and streaming "
            "medians of a case are comparable; full_evaluation refreshes the maximum-row boards, "
            "so its median should track full_refresh_rows_max plus dense_propagation\""
         << ",\"dense_propagation walks the parity fixtures, whose phase buckets differ, so its "
            "streaming schedule also walks distinct regions of the hidden0 table; the trunk's "
            "instruction count does not depend on the frame\""
         << ",\"the two-gather accumulator pools are laid out at a 1024-accumulator stride from a "
            "64-byte aligned base, so both sides are page aligned; a production V3AccumulatorFrame "
            "is not, and 4K aliasing between the gathered rows and the accumulator can differ "
            "between the two shapes. Read the two-gather ratio as a statement about arithmetic "
            "and gather structure, not about the engine's frame placement\""
         << ",\"the scalar build's two-gather ratio also carries whatever the auto-vectorizer "
            "chose for the 1024-lane and the 512-lane instantiation of the same source loop; the "
            "AVX2 build's kernels are hand written and identical in shape, so the AVX2 ratio is "
            "the one that speaks to the design claim\""
         << "]}";

    std::cout << json.str() << '\n';

    if (!options.outputPath.empty())
    {
        std::ofstream output(options.outputPath);
        if (!output)
        {
            std::cerr << "cannot write " << options.outputPath << '\n';
            return 4;
        }
        output << json.str() << '\n';
    }

    // Human table on stderr.
    std::cerr << "\nHorde V3 kernel timing  backend=" << V3DefaultBackendName
              << "  seed=" << options.seed << "  rounds=" << options.repetitions
              << (options.quick ? "  (quick)" : "") << "\n"
              << "clock resolution " << clockResolutionNs << " ns, max active rows "
              << maxActiveRows << " (" << maxG0Rows << " G0 + " << maxContextualRows
              << " contextual)\n\n";
    std::cerr << std::left << std::setw(30) << "case" << std::setw(11) << "schedule"
              << std::right << std::setw(6) << "n" << std::setw(9) << "inner" << std::setw(14)
              << "median ns" << std::setw(12) << "MAD ns" << "   ci95\n";
    for (const TimingCase& entry : cases)
        for (int which = 0; which < 2; ++which)
        {
            const ScheduleResult& result = which ? entry.streaming : entry.hot;
            std::cerr << std::left << std::setw(30) << entry.name << std::setw(11)
                      << (which ? "streaming" : "hot") << std::right << std::setw(6)
                      << result.samples << std::setw(9) << result.innerReps << std::setw(14)
                      << std::fixed << std::setprecision(2) << result.median << std::setw(12)
                      << result.mad << "   [" << std::setprecision(2) << result.ci.low << ", "
                      << result.ci.high << "]\n";
        }
    std::cerr << "\ntwo-gather reference (2x512 emulation / V3 1x1024), above 1 favours V3:\n"
              << "  hot        ratio " << std::fixed << std::setprecision(4) << ratios[0].ratio
              << "  ci95 [" << ratios[0].ci.low << ", " << ratios[0].ci.high << "]\n"
              << "  streaming  ratio " << ratios[1].ratio << "  ci95 [" << ratios[1].ci.low << ", "
              << ratios[1].ci.high << "]\n"
              << "  the 2x512 side is an emulation of the incumbent's arithmetic shape, not the "
                 "incumbent's code.\n";

    return 0;
}
