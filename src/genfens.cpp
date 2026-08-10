/*
  Horde-Stockfish, a UCI Horde chess engine derived from Stockfish
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Horde-Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "genfens.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fstream>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "misc.h"
#include "movegen.h"
#include "position.h"
#include "uci.h"

namespace Stockfish::GenFens {
namespace {

constexpr std::size_t MaxCount = 1'000'000;
constexpr std::size_t MaxPlies = 128;
constexpr u64         ZeroSeed = 0x9E3779B97F4A7C15ULL;

struct Settings {
    std::size_t count       = 0;
    u64         seed        = 0;
    std::string book        = "None";
    std::size_t minPlies    = 0;
    std::size_t maxPlies    = 0;
    bool        seedSeen    = false;
    bool        minPliesSet = false;
    bool        maxPliesSet = false;
};

bool parse_size(std::string_view text, std::size_t& value) {
    if (text.empty())
        return false;

    std::size_t parsed      = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), parsed);
    if (error != std::errc{} || end != text.data() + text.size())
        return false;

    value = parsed;
    return true;
}

std::optional<std::string> parse_settings(std::istream& args, Settings& settings) {
    if (!(args >> settings.count) || settings.count == 0 || settings.count > MaxCount)
        return "genfens count must be between 1 and 1000000";

    std::string token;
    while (args >> token)
    {
        if (token == "seed")
        {
            if (!(args >> settings.seed))
                return "genfens requires an unsigned integer after seed";
            settings.seedSeen = true;
        }
        else if (token == "book")
        {
            if (!(args >> settings.book) || settings.book.empty())
                return "genfens requires a path or None after book";
        }
        else if (token == "randmoves" || token == "plies")
        {
            std::size_t plies = 0;
            if (!(args >> plies))
                return "genfens requires an unsigned integer after " + token;
            settings.minPlies = settings.maxPlies = plies;
            settings.minPliesSet = settings.maxPliesSet = true;
        }
        else
        {
            const auto equal = token.find('=');
            if (equal == std::string::npos)
                return "unknown genfens argument: " + token;

            const std::string_view name(token.data(), equal);
            const std::string_view value(token.data() + equal + 1, token.size() - equal - 1);
            std::size_t            parsed = 0;
            if (!parse_size(value, parsed))
                return "invalid genfens integer: " + token;

            if (name == "plies")
            {
                settings.minPlies = settings.maxPlies = parsed;
                settings.minPliesSet = settings.maxPliesSet = true;
            }
            else if (name == "minplies")
            {
                settings.minPlies    = parsed;
                settings.minPliesSet = true;
            }
            else if (name == "maxplies")
            {
                settings.maxPlies    = parsed;
                settings.maxPliesSet = true;
            }
            else
                return "unknown genfens argument: " + token;
        }
    }

    if (!settings.seedSeen)
        return "genfens requires an explicit seed";

    const bool usesBook = settings.book != "None";
    if (!settings.minPliesSet && !settings.maxPliesSet)
    {
        settings.minPlies = usesBook ? 3 : 8;
        settings.maxPlies = usesBook ? 4 : 9;
    }
    else
    {
        // A single bound means a fixed count. This keeps each option useful on
        // its own while retaining fail-closed validation when both are given.
        if (!settings.minPliesSet)
            settings.minPlies = settings.maxPlies;
        if (!settings.maxPliesSet)
            settings.maxPlies = settings.minPlies;
    }

    if (settings.minPlies > settings.maxPlies || settings.maxPlies > MaxPlies)
        return "genfens plies must satisfy 0 <= minplies <= maxplies <= 128";

    return std::nullopt;
}

std::optional<std::string> normalize_epd(std::string_view line) {
    std::istringstream input{std::string(line)};
    std::string        board, side, castling, ep;
    if (!(input >> board >> side >> castling >> ep))
        return std::nullopt;
    return board + " " + side + " " + castling + " " + ep + " 0 1";
}

std::optional<std::string> load_openings(const Settings&           settings,
                                         std::vector<std::string>& openings) {
    if (settings.book == "None")
    {
        openings.emplace_back(StartFEN);
        return std::nullopt;
    }

    std::ifstream book(settings.book);
    if (!book)
        return "unable to open genfens book: " + settings.book;

    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(book, line))
    {
        ++lineNumber;
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        const auto first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#')
            continue;

        auto normalized = normalize_epd(std::string_view(line).substr(first));
        if (!normalized)
            return "invalid genfens book record at line " + std::to_string(lineNumber);

        StateInfo state;
        Position  position;
        if (auto error = position.set(*normalized, false, &state))
            return "invalid Horde FEN in genfens book at line " + std::to_string(lineNumber) + ": "
                 + error->what();

        openings.emplace_back(std::move(*normalized));
    }

    if (openings.empty())
        return "genfens book contains no Horde positions";

    return std::nullopt;
}

std::size_t choose(PRNG& rng, std::size_t count) {
    return count == 1 ? 0 : mul_hi64(rng.rand<u64>(), count);
}

}  // namespace

std::optional<std::string> run(std::istream& args, std::ostream& output) {
    Settings settings;
    if (auto error = parse_settings(args, settings))
        return error;

    std::vector<std::string> openings;
    if (auto error = load_openings(settings, openings))
        return error;

    PRNG              rng(settings.seed == 0 ? ZeroSeed : settings.seed);
    const std::size_t plySpan     = settings.maxPlies - settings.minPlies + 1;
    const std::size_t maxAttempts = settings.count * 4096 + 4096;
    std::size_t       generated   = 0;

    for (std::size_t attempt = 0; generated < settings.count && attempt < maxAttempts; ++attempt)
    {
        std::deque<StateInfo> states(1);
        Position              position;
        if (auto error =
              position.set(openings[choose(rng, openings.size())], false, &states.back()))
            return "validated genfens opening became invalid: " + std::string(error->what());

        const std::size_t plies  = settings.minPlies + choose(rng, plySpan);
        bool              usable = true;
        for (std::size_t ply = 0; ply < plies; ++ply)
        {
            if (position.outcome(0))
            {
                usable = false;
                break;
            }

            MoveList<LEGAL> moves(position);
            if (moves.size() == 0)
            {
                usable = false;
                break;
            }

            const Move move = *(moves.begin() + choose(rng, moves.size()));
            states.emplace_back();
            position.do_move(move, states.back());
        }

        if (!usable || position.outcome(0) || MoveList<LEGAL>(position).size() == 0)
            continue;

        output << "info string genfens " << position.fen() << std::endl;
        ++generated;
    }

    if (generated != settings.count)
        return "genfens could not produce the requested non-terminal Horde positions";

    return std::nullopt;
}

}  // namespace Stockfish::GenFens
