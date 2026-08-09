/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include "network.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>

#define INCBIN_SILENCE_BITCODE_WARNING
#include "../incbin/incbin.h"

#include "../misc.h"
#include "nnue_accumulator.h"

#if !defined(UNIVERSAL_BINARY) && !defined(_MSC_VER) && !defined(NNUE_EMBEDDING_OFF)
INCBIN(EmbeddedNNUE, EvalFileDefaultName);
#elif defined(UNIVERSAL_BINARY_MACOS_X86_SLICE)
extern const unsigned char* const gEmbeddedNNUEData;
extern const unsigned int         gEmbeddedNNUESize;
#elif defined(UNIVERSAL_BINARY)
extern const unsigned char gEmbeddedNNUEData[];
extern const unsigned int  gEmbeddedNNUESize;
#else
const unsigned char gEmbeddedNNUEData[1] = {0x0};
const unsigned int  gEmbeddedNNUESize    = 1;
#endif

namespace Stockfish::Eval::NNUE {

namespace fs = std::filesystem;

void Network::load(const fs::path& rootDirectory, fs::path evalfilePath, EvalFile& evalFile) {
#if defined(DEFAULT_NNUE_DIRECTORY)
    const std::vector<fs::path> dirs = {fs::path{}, rootDirectory,
                                        fs::path(stringify(DEFAULT_NNUE_DIRECTORY))};
#else
    const std::vector<fs::path> dirs = {fs::path{}, rootDirectory};
#endif

    if (evalfilePath.empty())
        evalfilePath = evalFile.defaultName;

    if (evalFile.current != evalfilePath && evalfilePath == evalFile.defaultName)
        load_internal(evalFile);

    for (const fs::path& directory : dirs)
        if (evalFile.current != evalfilePath)
            load_external(directory, evalfilePath, evalFile);
}

bool Network::save(const EvalFile&, const std::optional<fs::path>&) const {
    sync_cout << "Exporting the registered Run 6B network is disabled; use the canonical "
                 "manifested artifact instead."
              << sync_endl;
    return false;
}

NetworkOutput Network::evaluate(const Position&    pos,
                                AccumulatorStack&  accumulatorStack,
                                AccumulatorCaches& cache) const {
    const auto [psqt, positional] = evaluate_raw(pos, accumulatorStack, cache);
    return {static_cast<Value>(psqt / 16), static_cast<Value>(positional / 16)};
}

RawNetworkOutput Network::evaluate_raw(const Position& pos,
                                       AccumulatorStack& accumulatorStack,
                                       AccumulatorCaches&,
                                       int bucket) const {
    return hordeLegacyNetwork.evaluate_raw(pos, accumulatorStack, bucket);
}

void Network::verify(const std::function<void(std::string_view)>& f,
                     const EvalFile&                              evalFile,
                     fs::path                                     evalfilePath) const {
    if (evalfilePath.empty())
        evalfilePath = evalFile.defaultName;

    if (evalFile.current != evalfilePath || !hordeLegacyNetwork.loaded())
    {
        if (f)
        {
            const std::string msg1 =
              "The registered HORDETEST_HP_LEGACY_V1 Run 6B network must be available.";
            const std::string msg2 =
              "The network file " + evalfilePath.string() + " was not loaded successfully.";
            const std::string msg3 = "The UCI option EvalFile might need to specify the full path, "
                                     "including the directory name, to the network file.";
            const std::string msg4 = "Unregistered networks are rejected even when their NNUE "
                                     "header hashes match.";
            const std::string msg5 = "Search was not started.";

            f("ERROR: " + msg1 + '\n' + "ERROR: " + msg2 + '\n' + "ERROR: " + msg3 + '\n'
              + "ERROR: " + msg4 + '\n' + "ERROR: " + msg5 + '\n');
        }
        std::exit(EXIT_FAILURE);
    }

    if (f)
        f("NNUE evaluation using " + evalfilePath.string() + " ["
          + HordeLegacyNetwork::SchemaName + ", SHA-256 " + HordeLegacyNetwork::Sha256
          + ", (896, 1024, 16, 32, 1)]");
}

NnueEvalTrace Network::trace_evaluate(const Position&    pos,
                                      AccumulatorStack&  accumulatorStack,
                                      AccumulatorCaches& cache) const {
    NnueEvalTrace trace{};
    trace.correctBucket = hordeLegacyNetwork.bucket_for(pos);
    for (int bucket = 0; bucket < int(HordeLegacyNetwork::LayerStacks); ++bucket)
    {
        const auto [materialist, positional] =
          evaluate_raw(pos, accumulatorStack, cache, bucket);
        trace.psqt[bucket]       = static_cast<Value>(materialist / 16);
        trace.positional[bucket] = static_cast<Value>(positional / 16);
    }
    return trace;
}

void Network::load_external(const fs::path& dir,
                            const fs::path& evalfilePath,
                            EvalFile&       evalFile) {
    std::ifstream stream(dir / evalfilePath, std::ios::binary);
    if (!stream)
        return;

    const std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(stream), {}};
    auto                             candidate = std::make_unique<Network>();
    std::string                      description;
    if (candidate->hordeLegacyNetwork.load(bytes.data(), bytes.size(), description))
    {
        *this                   = std::move(*candidate);
        evalFile.current        = evalfilePath;
        evalFile.netDescription = std::move(description);
    }
    else
        sync_cout << "info string Rejected EvalFile: " << description << sync_endl;
}

void Network::load_internal(EvalFile& evalFile) {
#ifdef UNIVERSAL_BINARY_MACOS_X86_SLICE
    if (gEmbeddedNNUEData == nullptr)
        return;
#endif

    auto        candidate = std::make_unique<Network>();
    std::string description;
    if (candidate->hordeLegacyNetwork.load(gEmbeddedNNUEData, usize(gEmbeddedNNUESize),
                                           description))
    {
        *this                   = std::move(*candidate);
        evalFile.current        = evalFile.defaultName;
        evalFile.netDescription = std::move(description);
    }
}

usize Network::get_content_hash() const { return hordeLegacyNetwork.content_hash(); }

}  // namespace Stockfish::Eval::NNUE
