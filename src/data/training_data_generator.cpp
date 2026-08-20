/*
  Horde-Stockfish training-data generator
  Copyright (C) 2026 The Horde-Stockfish developers

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "training_data_generator.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "engine.h"
#include "expansion_selection.h"
#include "horde_bin_v1.h"
#include "misc.h"
#include "movegen.h"
#include "position.h"
#include "random_seed.h"
#include "search.h"
#include "sha256.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"

#ifndef HORDE_SOURCE_COMMIT
    #define HORDE_SOURCE_COMMIT "0000000000000000000000000000000000000000"
#endif

#ifndef HORDE_SOURCE_DIRTY
    #define HORDE_SOURCE_DIRTY 1
#endif

namespace Stockfish::Data {
namespace {

constexpr u64              ReportEvery         = 5000;
constexpr int              MaximumGeneratedPly = 4096;
constexpr usize            DedupeTableSize     = usize(1) << 22;
constexpr std::string_view Run6BSha256 =
  "B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7";

struct GeneratorParams {
    int   threads = 1;
    usize hashMb  = 512;
    u64   count   = 0;
    int   depth   = 6;
    u64   nodes   = 0;

    int randomMoveMinPly  = 1;
    int randomMoveMaxPly  = 20;
    int randomMoveCount   = 8;
    int randomMultiPv     = 4;
    int randomMultiPvDiff = 200;

    int writeMinPly = 5;
    int writeMaxPly = 400;
    int maxGamePly  = 512;

    // HORDE_BIN_V1_R2 tactical expansion. All three default to zero, so a command
    // that does not ask for expansion produces exactly the bytes the pre-revision
    // generator produced, manifest included.
    int expandPromo       = 0;
    int expandCheck       = 0;
    int expandMaxChildren = 0;

    bool expansion_enabled() const { return expandPromo > 0 || expandCheck > 0; }
    int  expansion_ceiling() const {
        return std::min(expandPromo + expandCheck, expandMaxChildren);
    }

    std::string network;
    std::string networkSha256;
    std::string producerSha256;
    std::string book;
    std::string bookSha256;
    std::string outputFile;
    std::string seedText;

    bool setRecommendedUciOptions = false;
};

struct RequiredParams {
    bool threads        = false;
    bool hash           = false;
    bool count          = false;
    bool network        = false;
    bool networkSha256  = false;
    bool producerSha256 = false;
    bool book           = false;
    bool bookSha256     = false;
    bool output         = false;
    bool seed           = false;

    bool complete() const {
        return threads && hash && count && network && networkSha256 && producerSha256 && book
            && bookSha256 && output && seed;
    }
};

struct GameResolution {
    bool                 terminal = false;
    std::optional<Color> winner;
    HordeOutcomeReason   reason = HordeOutcomeReason::STALEMATE;
};

template<typename T>
bool read_value(std::istream& input, T& value, std::string_view name, std::string& error) {
    if (input >> value)
        return true;
    error = "Missing or invalid value for horde_generate_training_data option " + std::string(name);
    return false;
}

bool read_u64_value(std::istream& input, u64& value, std::string_view name, std::string& error) {
    std::string token;
    if (!(input >> token) || token.empty())
    {
        error =
          "Missing or invalid value for horde_generate_training_data option " + std::string(name);
        return false;
    }

    u64 parsed = 0;
    for (const unsigned char c : token)
    {
        if (c < '0' || c > '9')
        {
            error = "Unsigned option " + std::string(name) + " must contain decimal digits only";
            return false;
        }
        const u64 digit = c - '0';
        if (parsed > (std::numeric_limits<u64>::max() - digit) / 10)
        {
            error = "Unsigned option " + std::string(name) + " is out of range";
            return false;
        }
        parsed = parsed * 10 + digit;
    }
    value = parsed;
    return true;
}

std::string uppercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return char(std::toupper(c)); });
    return value;
}

bool parse_params(std::istream& input, GeneratorParams& params, std::string& error) {
    RequiredParams required;
    std::string    token;
    while (input >> token)
    {
        if (token == "threads")
        {
            required.threads = read_value(input, params.threads, token, error);
            if (!required.threads)
                return false;
        }
        else if (token == "hash")
        {
            required.hash = read_value(input, params.hashMb, token, error);
            if (!required.hash)
                return false;
        }
        else if (token == "count")
        {
            required.count = read_u64_value(input, params.count, token, error);
            if (!required.count)
                return false;
        }
        else if (token == "depth")
        {
            if (!read_value(input, params.depth, token, error))
                return false;
        }
        else if (token == "nodes")
        {
            if (!read_u64_value(input, params.nodes, token, error))
                return false;
        }
        else if (token == "network")
        {
            required.network = read_value(input, params.network, token, error);
            if (!required.network)
                return false;
        }
        else if (token == "network_sha256")
        {
            required.networkSha256 = read_value(input, params.networkSha256, token, error);
            if (!required.networkSha256)
                return false;
            params.networkSha256 = uppercase(params.networkSha256);
        }
        else if (token == "producer_sha256")
        {
            required.producerSha256 = read_value(input, params.producerSha256, token, error);
            if (!required.producerSha256)
                return false;
            params.producerSha256 = uppercase(params.producerSha256);
        }
        else if (token == "book")
        {
            required.book = read_value(input, params.book, token, error);
            if (!required.book)
                return false;
        }
        else if (token == "book_sha256")
        {
            required.bookSha256 = read_value(input, params.bookSha256, token, error);
            if (!required.bookSha256)
                return false;
            params.bookSha256 = uppercase(params.bookSha256);
        }
        else if (token == "out")
        {
            required.output = read_value(input, params.outputFile, token, error);
            if (!required.output)
                return false;
        }
        else if (token == "seed")
        {
            required.seed = read_value(input, params.seedText, token, error);
            if (!required.seed)
                return false;
        }
        else if (token == "random_move_min_ply")
        {
            if (!read_value(input, params.randomMoveMinPly, token, error))
                return false;
        }
        else if (token == "random_move_max_ply")
        {
            if (!read_value(input, params.randomMoveMaxPly, token, error))
                return false;
        }
        else if (token == "random_move_count")
        {
            if (!read_value(input, params.randomMoveCount, token, error))
                return false;
        }
        else if (token == "random_multi_pv")
        {
            if (!read_value(input, params.randomMultiPv, token, error))
                return false;
        }
        else if (token == "random_multi_pv_diff")
        {
            if (!read_value(input, params.randomMultiPvDiff, token, error))
                return false;
        }
        else if (token == "write_min_ply")
        {
            if (!read_value(input, params.writeMinPly, token, error))
                return false;
        }
        else if (token == "write_max_ply")
        {
            if (!read_value(input, params.writeMaxPly, token, error))
                return false;
        }
        else if (token == "max_game_ply")
        {
            if (!read_value(input, params.maxGamePly, token, error))
                return false;
        }
        else if (token == "expand_promo")
        {
            if (!read_value(input, params.expandPromo, token, error))
                return false;
        }
        else if (token == "expand_check")
        {
            if (!read_value(input, params.expandCheck, token, error))
                return false;
        }
        else if (token == "expand_max_children")
        {
            if (!read_value(input, params.expandMaxChildren, token, error))
                return false;
        }
        else if (token == "set_recommended_uci_options")
            params.setRecommendedUciOptions = true;
        else
        {
            error = "Unknown horde_generate_training_data option " + token;
            return false;
        }
    }

    if (!required.complete())
    {
        error = "The Horde DATAGEN identity contract requires threads, hash, count, network, "
                "network_sha256, producer_sha256, book, book_sha256, out, and seed";
        return false;
    }
    if (params.network == "NONE" || params.networkSha256 != Run6BSha256)
    {
        error = "Network identity does not match the registered Run 6B SHA-256";
        return false;
    }
    if (!is_upper_sha256(params.producerSha256) || params.outputFile.empty()
        || params.threads <= 0 || params.count == 0 || params.hashMb == 0 || params.depth <= 0
        || params.depth >= MAX_PLY || params.randomMoveMinPly < 0
        || params.randomMoveMaxPly < params.randomMoveMinPly
        || params.randomMoveMaxPly > MaximumGeneratedPly || params.randomMoveCount < 0
        || params.randomMoveCount > MaximumGeneratedPly || params.randomMultiPv < 0
        || params.randomMultiPv > int(MAX_MOVES) || params.randomMultiPvDiff < 0
        || params.writeMinPly < 0 || params.writeMaxPly <= params.writeMinPly
        || params.maxGamePly <= params.writeMaxPly || params.maxGamePly > MaximumGeneratedPly
        || !params.setRecommendedUciOptions)
    {
        error = "Invalid horde_generate_training_data parameter range";
        return false;
    }
    if ((params.book == "NONE") != (params.bookSha256 == "NONE")
        || (params.book != "NONE" && !is_upper_sha256(params.bookSha256)))
    {
        error = "book and book_sha256 must either both be NONE or identify one authenticated book";
        return false;
    }
    if (params.expandPromo < 0 || params.expandPromo > HordeBinV1MaxExpansionCap
        || params.expandCheck < 0 || params.expandCheck > HordeBinV1MaxExpansionCap
        || params.expandMaxChildren < 0 || params.expandMaxChildren > HordeBinV1MaxExpansionCap)
    {
        error = "expand_promo, expand_check and expand_max_children must be in [0, "
              + std::to_string(HordeBinV1MaxExpansionCap) + "]";
        return false;
    }
    if (params.expansion_enabled() != (params.expandMaxChildren > 0))
    {
        error = "expand_max_children must be positive exactly when a family cap is positive";
        return false;
    }
    return true;
}

std::string trim(std::string text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return {};
    const auto last = text.find_last_not_of(" \t\r\n");
    return text.substr(first, last - first + 1);
}

std::optional<std::string> normalize_book_line(std::string line) {
    line = trim(std::move(line));
    if (line.empty() || line[0] == '#')
        return std::nullopt;

    std::istringstream       stream(line);
    std::vector<std::string> fields;
    std::string              field;
    while (stream >> field)
        fields.push_back(field);
    if (fields.size() < 4)
        return std::nullopt;

    std::ostringstream fen;
    for (usize i = 0; i < 4; ++i)
        fen << (i ? " " : "") << fields[i];

    const auto decimal = [](const std::string& value) {
        return !value.empty() && std::all_of(value.begin(), value.end(), [](unsigned char c) {
            return std::isdigit(c) != 0;
        });
    };
    const bool hasClocks = fields.size() >= 6 && decimal(fields[4]) && decimal(fields[5]);
    fen << (hasClocks ? " " + fields[4] + " " + fields[5] : " 0 1");
    return fen.str();
}

bool load_book(const GeneratorParams&    params,
               std::vector<std::string>& positions,
               std::string&              error) {
    if (params.book == "NONE")
    {
        positions.emplace_back(StartFEN);
        return true;
    }

    std::string observedSha;
    if (!sha256_file(params.book, observedSha, error))
        return false;
    if (observedSha != params.bookSha256)
    {
        error =
          "Opening-book SHA-256 mismatch: expected " + params.bookSha256 + ", got " + observedSha;
        return false;
    }

    std::ifstream input(params.book);
    if (!input)
    {
        error = "Cannot open Horde training-data opening book";
        return false;
    }

    std::string line;
    while (std::getline(input, line))
    {
        auto candidate = normalize_book_line(std::move(line));
        if (!candidate)
            continue;

        Position  position;
        StateInfo state{};
        if (position.set(*candidate, false, &state) || position.is_chess960()
            || position.count<KING>(WHITE) != 0 || position.count<KING>(BLACK) != 1
            || position.outcome(0))
        {
            error = "Opening book contains an invalid, terminal, or unsupported Horde FEN";
            return false;
        }
        positions.push_back(position.fen());
    }

    if (positions.empty())
    {
        error = "Horde training-data opening book contains no usable positions";
        return false;
    }
    return true;
}

Color side_to_move_from_fen(const std::string& fen) {
    const auto separator = fen.find(' ');
    assert(separator != std::string::npos && separator + 1 < fen.size());
    return fen[separator + 1] == 'b' ? BLACK : WHITE;
}

HordeOutcomeReason convert_reason(OutcomeReason reason) {
    switch (reason)
    {
    case OutcomeReason::CHECKMATE :
        return HordeOutcomeReason::CHECKMATE;
    case OutcomeReason::EXTINCTION :
        return HordeOutcomeReason::EXTINCTION;
    case OutcomeReason::STALEMATE :
        return HordeOutcomeReason::STALEMATE;
    case OutcomeReason::HORDE_FORTRESS :
        return HordeOutcomeReason::HORDE_FORTRESS;
    case OutcomeReason::FIFTY_MOVE :
        return HordeOutcomeReason::FIFTY_MOVE;
    case OutcomeReason::FIVEFOLD_REPETITION :
        return HordeOutcomeReason::FIVEFOLD_REPETITION;
    }
    assert(false);
    return HordeOutcomeReason::STALEMATE;
}

GameResolution current_resolution(Position& position) {
    const auto outcome = position.outcome(0);
    if (!outcome)
        return {};

    std::optional<Color> winner;
    if (outcome->result > VALUE_DRAW)
        winner = position.side_to_move();
    else if (outcome->result < VALUE_DRAW)
        winner = ~position.side_to_move();
    return {true, winner, convert_reason(outcome->reason)};
}

class SeenPositions {
   public:
    SeenPositions() :
        slots(std::make_unique<std::atomic<Key>[]>(DedupeTableSize)) {
        for (usize i = 0; i < DedupeTableSize; ++i)
            slots[i].store(0, std::memory_order_relaxed);
    }

    bool already_seen(Key key) {
        const usize index = usize(key) & (DedupeTableSize - 1);
        return slots[index].exchange(key, std::memory_order_relaxed) == key;
    }

   private:
    std::unique_ptr<std::atomic<Key>[]> slots;
};

class Generator {
   public:
    Generator(const GeneratorParams&   params_,
              ThreadPool&              threads_,
              TranspositionTable&      tt_,
              std::filesystem::path    outputPath,
              HordeBinV1Manifest       manifest,
              std::vector<std::string> openingPositions_,
              u64                      resolvedSeed) :
        params(params_),
        threads(threads_),
        tt(tt_),
        output(std::move(outputPath), std::move(manifest)),
        openingPositions(std::move(openingPositions_)) {
        ReplayablePRNG source(resolvedSeed);
        workerSeeds.reserve(threads.size());
        for (usize i = 0; i < threads.size(); ++i)
            workerSeeds.push_back(i == 0 ? source.seed() : source.next_seed());

        if (openingPositions.size() > 1)
        {
            ReplayablePRNG bookRng(workerSeeds[0]);
            for (usize i = 0; i < openingPositions.size(); ++i)
                std::swap(openingPositions[i],
                          openingPositions[i + bookRng.rand(openingPositions.size() - i)]);
            workerSeeds[0] = bookRng.seed();
        }
    }

    bool run(std::string& error) {
        threads.wait_for_search_finished();
        threads.clear();
        tt.clear(threads);
        threads.stop = false;
        tt.new_search();

        std::vector<Thread*> workers;
        workers.reserve(threads.size());
        for (auto& thread : threads)
            workers.push_back(thread.get());

        for (usize i = 0; i < workers.size(); ++i)
        {
            Search::Worker* worker = workers[i]->worker.get();
            threads.run_on_thread(i, [this, worker, i]() { run_worker(*worker, i); });
        }
        for (usize i = 0; i < workers.size(); ++i)
            threads.wait_on_thread(i);
        threads.stop = false;

        std::lock_guard lock(outputMutex);
        if (!failure.empty())
        {
            const DataResult cleanup = output.abort();
            error                    = failure;
            if (!cleanup)
                error += "; cleanup failed: " + cleanup.message;
            return false;
        }
        if (DataResult result = output.finalize(); !result)
        {
            const DataResult cleanup = output.abort();
            error                    = result.message;
            if (!cleanup)
                error += "; cleanup failed: " + cleanup.message;
            return false;
        }
        return true;
    }

    u64 records_written() const { return written.load(std::memory_order_relaxed); }
    u64 draws_written() const { return draws.load(std::memory_order_relaxed); }
    u64 truncated_games() const { return truncated.load(std::memory_order_relaxed); }

    u64 promotion_children() const { return promotionChildren.load(std::memory_order_relaxed); }
    u64 check_children() const { return checkChildren.load(std::memory_order_relaxed); }
    u64 expanded_parents() const { return expandedParents.load(std::memory_order_relaxed); }
    u64 capped_parents() const { return cappedParents.load(std::memory_order_relaxed); }
    u64 terminal_children_skipped() const {
        return terminalChildren.load(std::memory_order_relaxed);
    }
    u64 duplicate_children_skipped() const {
        return duplicateChildren.load(std::memory_order_relaxed);
    }
    std::string children_histogram() const {
        std::ostringstream text;
        for (int i = 0; i <= HordeBinV1MaxExpansionCap; ++i)
            text << (i ? "," : "") << i << ":"
                 << childrenHistogram[usize(i)].load(std::memory_order_relaxed);
        return text.str();
    }

   private:
    const GeneratorParams&   params;
    ThreadPool&              threads;
    TranspositionTable&      tt;
    HordeBinV1Sink           output;
    std::vector<std::string> openingPositions;
    std::vector<u64>         workerSeeds;
    SeenPositions            seen;

    std::atomic<u64>                            written{0};
    std::atomic<u64>                            draws{0};
    std::atomic<u64>                            truncated{0};
    std::atomic<u64>                            promotionChildren{0};
    std::atomic<u64>                            checkChildren{0};
    std::atomic<u64>                            expandedParents{0};
    std::atomic<u64>                            cappedParents{0};
    std::atomic<u64>                            terminalChildren{0};
    std::atomic<u64>                            duplicateChildren{0};
    std::atomic<u64>                            childrenHistogram[HordeBinV1MaxExpansionCap + 1]{};
    std::atomic<bool>                           finished{false};
    std::mutex                                  outputMutex;
    std::mutex                                  openingMutex;
    usize                                       openingIndex = 0;
    std::string                                 failure;
    const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();

    void fail(std::string message) {
        {
            std::lock_guard lock(outputMutex);
            if (failure.empty())
                failure = std::move(message);
        }
        finished.store(true, std::memory_order_relaxed);
        threads.stop = true;
    }

    std::string next_opening() {
        std::lock_guard   lock(openingMutex);
        const std::string fen = openingPositions[openingIndex];
        if (++openingIndex == openingPositions.size())
            openingIndex = 0;
        return fen;
    }

    std::vector<u8> random_move_flags(ReplayablePRNG& rng) const {
        std::vector<u8>  flags(usize(params.maxGamePly), 0);
        std::vector<int> candidates;
        for (int ply = params.randomMoveMinPly; ply <= params.randomMoveMaxPly; ++ply)
            candidates.push_back(ply);

        const int count = std::min(params.randomMoveCount, int(candidates.size()));
        for (int i = 0; i < count; ++i)
        {
            const usize selected = usize(i) + usize(rng.rand(candidates.size() - usize(i)));
            std::swap(candidates[usize(i)], candidates[selected]);
            flags[usize(candidates[usize(i)])] = 1;
        }
        return flags;
    }

    Move choose_played_move(const Position&                     position,
                            const Search::TrainingSearchResult& search,
                            ReplayablePRNG&                     rng,
                            bool                                explore) const {
        assert(!search.pv.empty());
        const Move bestMove = search.pv[0];
        if (!explore)
            return bestMove;

        if (params.randomMultiPv == 0 || search.lines.empty())
        {
            const MoveList<LEGAL> moves(position);
            return *(moves.begin() + rng.rand(moves.size()));
        }

        usize candidates = search.lines.size();
        for (usize i = 1; i < candidates; ++i)
            if (std::int64_t(search.lines.front().value)
                > std::int64_t(search.lines[i].value) + params.randomMultiPvDiff)
            {
                candidates = i;
                break;
            }
        const auto& line = search.lines[rng.rand(candidates)];
        return line.pv.empty() ? bestMove : line.pv[0];
    }

    // Tactical expansion, HORDE_BIN_V1_R2.
    //
    // Two families are expanded, promotion and check, because those are the two
    // buckets where the teacher and every student we have measured are worst.
    // A promotion that also gives check belongs to PROMOTION and spends only the
    // promotion budget, so the check budget stays reserved for non-promoting
    // checks and no move is counted twice: at most expand_promo + expand_check
    // children per parent, before the hard ceiling.
    //
    // Selection is deterministic — legal-move generation order, one child per
    // distinct promotion push so a single pawn cannot spend the whole promotion
    // budget on its own underpromotions.
    //
    // The child is searched by the same worker, the same network and the same
    // budget as its parent: labelling across engine families is what lost all
    // three gates in the Spell round-2 experiment. Its result is not copied from
    // the parent — the child is appended to the same game's sample list and
    // commit_game derives every result from the record's own side to move, so the
    // sign flip falls out of the FEN and no result is fabricated.
    bool expand_parent(Search::Worker&                  worker,
                       Position&                        position,
                       std::vector<TrainingDataSample>& samples) {
        const int ceiling = params.expansion_ceiling();
        if (ceiling <= 0)
            return true;

        const ExpansionSelection selection = select_expansion_children(
          position, params.expandPromo, params.expandCheck, ceiling);
        if (selection.capped)
            cappedParents.fetch_add(1, std::memory_order_relaxed);

        u64 emitted = 0;
        for (const auto& [childMove, family] : selection.children)
        {
            if (finished.load(std::memory_order_relaxed))
                break;

            StateInfo childState{};
            position.do_move(childMove, childState, &tt);

            // A terminal child has no principal variation to label, and a child
            // that duplicates a position already written would be a correlated
            // duplicate rather than new signal. Both are counted, never silently
            // dropped: the July Spell lab swallowed child-generation failures and
            // left no trace of how many children it lost.
            if (current_resolution(position).terminal)
            {
                terminalChildren.fetch_add(1, std::memory_order_relaxed);
                position.undo_move(childMove);
                continue;
            }
            if (seen.already_seen(position.key()))
            {
                duplicateChildren.fetch_add(1, std::memory_order_relaxed);
                position.undo_move(childMove);
                continue;
            }

            Search::TrainingSearchRequest request;
            request.depth   = params.depth;
            request.nodes   = params.nodes;
            request.multiPV = 1;
            const auto search = worker.training_search(position, request);
            if (finished.load(std::memory_order_relaxed))
            {
                position.undo_move(childMove);
                break;
            }
            if (search.value == VALUE_NONE || search.pv.empty())
            {
                position.undo_move(childMove);
                fail("Synchronous Horde expansion search returned no principal variation");
                return false;
            }
            if (!search.exact)
            {
                position.undo_move(childMove);
                fail("Synchronous Horde expansion search returned a bound score");
                return false;
            }

            const Move childBest = search.pv[0];
            samples.push_back({position.fen(), int(search.value), childBest, childBest, 0,
                               HordeOutcomeReason::STALEMATE, family});
            if (family == HordeExpansionFamily::PROMOTION)
                promotionChildren.fetch_add(1, std::memory_order_relaxed);
            else
                checkChildren.fetch_add(1, std::memory_order_relaxed);
            ++emitted;
            position.undo_move(childMove);
        }

        childrenHistogram[usize(std::min(emitted, u64(HordeBinV1MaxExpansionCap)))].fetch_add(
          1, std::memory_order_relaxed);
        if (emitted)
            expandedParents.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    bool commit_game(std::vector<TrainingDataSample>& samples,
                     std::optional<Color>             winner,
                     HordeOutcomeReason               reason) {
        if (samples.empty())
            return false;

        std::lock_guard lock(outputMutex);
        if (!failure.empty() || finished.load(std::memory_order_relaxed))
            return true;

        const bool draw = !winner.has_value();
        for (auto& sample : samples)
        {
            if (written.load(std::memory_order_relaxed) >= params.count)
            {
                finished.store(true, std::memory_order_relaxed);
                threads.stop = true;
                return true;
            }

            sample.result = winner ? (side_to_move_from_fen(sample.fen) == *winner ? 1 : -1) : 0;
            sample.outcomeReason = reason;
            if (DataResult result = output.append(sample); !result)
            {
                failure = result.message;
                finished.store(true, std::memory_order_relaxed);
                threads.stop = true;
                return true;
            }

            const u64 done = written.fetch_add(1, std::memory_order_relaxed) + 1;
            if (draw)
                draws.fetch_add(1, std::memory_order_relaxed);
            if (done % ReportEvery == 0 || done == params.count)
            {
                const auto elapsed = std::max(
                  1.0, std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
                         .count());
                std::cout << "info string Horde training data " << done << "/" << params.count
                          << " records, " << u64(double(done) / elapsed) << " records/s"
                          << std::endl;
            }
            if (done >= params.count)
            {
                finished.store(true, std::memory_order_relaxed);
                threads.stop = true;
                return true;
            }
        }
        return false;
    }

    void run_worker(Search::Worker& worker, usize workerIndex) {
        ReplayablePRNG         rng(workerSeeds[workerIndex]);
        std::vector<StateInfo> states(usize(params.maxGamePly));

        while (!finished.load(std::memory_order_relaxed))
        {
            StateInfo         rootState{};
            Position          position;
            const std::string initialFen = next_opening();
            if (const auto setError = position.set(initialFen, false, &rootState))
            {
                fail(std::string("Cannot set Horde opening FEN: ") + setError->what());
                return;
            }

            std::vector<TrainingDataSample> samples;
            samples.reserve(usize(params.writeMaxPly - params.writeMinPly)
                            * usize(1 + std::max(0, params.expansion_ceiling())));
            const std::vector<u8> explore = random_move_flags(rng);

            for (int ply = 0; !finished.load(std::memory_order_relaxed); ++ply)
            {
                const GameResolution resolution = current_resolution(position);
                if (resolution.terminal)
                {
                    commit_game(samples, resolution.winner, resolution.reason);
                    break;
                }
                if (ply >= params.maxGamePly)
                {
                    truncated.fetch_add(1, std::memory_order_relaxed);
                    break;
                }

                Search::TrainingSearchRequest request;
                request.depth = params.depth;
                request.nodes = params.nodes;
                request.multiPV =
                  explore[usize(ply)] ? usize(std::max(params.randomMultiPv, 1)) : 1;
                const auto search = worker.training_search(position, request);
                if (finished.load(std::memory_order_relaxed))
                    break;
                if (search.value == VALUE_NONE || search.pv.empty())
                {
                    fail("Synchronous Horde training search returned no principal variation");
                    return;
                }
                if (!search.exact
                    || std::any_of(search.lines.begin(), search.lines.end(),
                                   [](const auto& line) { return !line.exact; }))
                {
                    fail("Synchronous Horde training search returned a bound score");
                    return;
                }

                const Move bestMove = search.pv[0];
                const Move playedMove =
                  choose_played_move(position, search, rng, explore[usize(ply)]);
                const MoveList<LEGAL> legalMoves(position);
                if (!legalMoves.contains(bestMove) || !legalMoves.contains(playedMove))
                {
                    fail("Horde training generator selected an illegal move");
                    return;
                }

                if (ply >= params.writeMinPly && ply < params.writeMaxPly
                    && !seen.already_seen(position.key()))
                {
                    samples.push_back({position.fen(), int(search.value), bestMove, playedMove, 0,
                                       HordeOutcomeReason::STALEMATE,
                                       HordeExpansionFamily::NONE});
                    // Children are appended immediately after their parent, so a
                    // reader recovers the (parent, children) unit by scanning
                    // forward from a record whose EXPANSION_CHILD bit is clear.
                    if (!expand_parent(worker, position, samples))
                        return;
                }

                position.do_move(playedMove, states[usize(ply)], &tt);
            }
        }
    }
};

void set_engine_option(Engine& engine, const std::string& name, const std::string& value) {
    std::istringstream option("name " + name + " value " + value);
    engine.get_options().setoption(option);
}

void print_error(std::string_view message) { sync_cout << "ERROR: " << message << sync_endl; }

}  // namespace

bool generate_training_data(Engine& engine, std::istream& input) {
    GeneratorParams params;
    std::string     error;
    if (!parse_params(input, params, error))
    {
        print_error(error);
        return false;
    }

    engine.wait_for_search_finished();

    std::string observedNetworkSha;
    if (!sha256_file(params.network, observedNetworkSha, error))
    {
        print_error(error);
        return false;
    }
    if (observedNetworkSha != params.networkSha256 || observedNetworkSha != Run6BSha256)
    {
        print_error("Network bytes do not match the registered Run 6B SHA-256");
        return false;
    }

    set_engine_option(engine, "Threads", std::to_string(params.threads));
    set_engine_option(engine, "Hash", std::to_string(params.hashMb));
    set_engine_option(engine, "UCI_Chess960", "false");
    set_engine_option(engine, "Skill Level", "20");
    set_engine_option(engine, "UCI_LimitStrength", "false");
    set_engine_option(engine, "EvalFile", params.network);
    if (int(engine.options["Threads"]) != params.threads
        || usize(engine.options["Hash"]) != params.hashMb
        || int(engine.options["UCI_Chess960"]) != 0
        || std::string(engine.options["EvalFile"]) != params.network)
    {
        print_error("Engine options did not accept the requested Horde DATAGEN configuration");
        return false;
    }
    engine.verify_network();

    std::vector<std::string> openings;
    if (!load_book(params, openings, error))
    {
        print_error(error);
        return false;
    }

    std::error_code ec;
    if (std::filesystem::exists(params.outputFile, ec) || ec)
    {
        print_error(ec ? "Cannot inspect Horde DATAGEN output path"
                       : "Horde DATAGEN output already exists");
        return false;
    }

    const u64          resolvedSeed = ReplayablePRNG::resolve(params.seedText);
    HordeBinV1Manifest manifest;
    manifest.sourceCommit      = HORDE_SOURCE_COMMIT;
    manifest.sourceDirty       = HORDE_SOURCE_DIRTY != 0;
    manifest.networkSha256     = params.networkSha256;
    manifest.bookSha256        = params.bookSha256;
    manifest.producerSha256    = params.producerSha256;
    manifest.requestedRecords  = params.count;
    manifest.seed              = resolvedSeed;
    manifest.threads           = usize(params.threads);
    manifest.hashMb            = params.hashMb;
    manifest.depth             = params.depth;
    manifest.nodes             = params.nodes;
    manifest.randomMoveMinPly  = params.randomMoveMinPly;
    manifest.randomMoveMaxPly  = params.randomMoveMaxPly;
    manifest.randomMoveCount   = params.randomMoveCount;
    manifest.randomMultiPv     = params.randomMultiPv;
    manifest.randomMultiPvDiff = params.randomMultiPvDiff;
    manifest.writeMinPly       = params.writeMinPly;
    manifest.writeMaxPly       = params.writeMaxPly;
    manifest.maxGamePly        = params.maxGamePly;
    manifest.openingCount      = openings.size();
    manifest.expandPromo       = params.expandPromo;
    manifest.expandCheck       = params.expandCheck;
    manifest.expandMaxChildren = params.expandMaxChildren;
    if (DataResult valid = validate_horde_bin_v1_manifest(manifest); !valid)
    {
        print_error(valid.message);
        return false;
    }

    std::cout << "INFO: Executing horde_generate_training_data command\n"
              << "INFO: schema = "
              << (manifest.expansion_enabled() ? HordeBinV1RevisionName : HordeBinV1SchemaName)
              << '\n'
              << "INFO: schema_sha256 = "
              << (manifest.expansion_enabled() ? HordeBinV1RevisionSha256 : HordeBinV1SchemaSha256)
              << '\n'
              << "INFO: expand_promo = " << params.expandPromo << '\n'
              << "INFO: expand_check = " << params.expandCheck << '\n'
              << "INFO: expand_max_children = " << params.expandMaxChildren << '\n'
              << "INFO: expansion_ceiling = " << std::max(0, params.expansion_ceiling()) << '\n'
              << "INFO: source_commit = " << manifest.sourceCommit << '\n'
              << "INFO: source_dirty = " << (manifest.sourceDirty ? "true" : "false") << '\n'
              << "PRNG::initial_seed = " << resolvedSeed << '\n'
              << "INFO: threads = " << params.threads << '\n'
              << "INFO: depth = " << params.depth << '\n'
              << "INFO: count = " << params.count << '\n'
              << "INFO: openings = " << openings.size() << std::endl;

    Generator generator(params, engine.threads, engine.tt, params.outputFile, std::move(manifest),
                        std::move(openings), resolvedSeed);
    if (!generator.run(error))
    {
        print_error(error);
        return false;
    }

    std::cout << "INFO: horde_generate_training_data finished.\n"
              << "INFO: records=" << generator.records_written()
              << " draws=" << generator.draws_written()
              << " truncated_games=" << generator.truncated_games() << std::endl;
    if (params.expansion_enabled())
        std::cout << "INFO: expansion promotion_children=" << generator.promotion_children()
                  << " check_children=" << generator.check_children()
                  << " expanded_parents=" << generator.expanded_parents()
                  << " capped_parents=" << generator.capped_parents()
                  << " terminal_children_skipped=" << generator.terminal_children_skipped()
                  << " duplicate_children_skipped=" << generator.duplicate_children_skipped()
                  << '\n'
                  << "INFO: expansion children_per_parent " << generator.children_histogram()
                  << std::endl;
    return true;
}

}  // namespace Stockfish::Data
