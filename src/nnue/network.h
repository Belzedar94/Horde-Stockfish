/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <filesystem>
#include <type_traits>

#include "../misc.h"
#include "../types.h"
#include "horde_legacy_network.h"
#include "nnue_misc.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE {

class AccumulatorStack;
struct AccumulatorCaches;

using NetworkOutput    = std::tuple<Value, Value>;
using RawNetworkOutput = HordeLegacyNetwork::RawOutput;

// Horde-Stockfish accepts exactly the registered Run 6B schema. The network
// remains an inline trivial type so NUMA replication and shared memory retain
// the Stockfish ownership contract.
class Network {
   public:
    Network() = default;

    Network(const Network& other) = default;
    Network(Network&& other)      = default;

    Network& operator=(const Network& other) = default;
    Network& operator=(Network&& other)      = default;

    void load(const std::filesystem::path& rootDirectory,
              std::filesystem::path        evalfilePath,
              EvalFile&                    evalFile);
    bool save(const EvalFile& evalFile, const std::optional<std::filesystem::path>& filename) const;

    usize get_content_hash() const;

    NetworkOutput evaluate(const Position&    pos,
                           AccumulatorStack&  accumulatorStack,
                           AccumulatorCaches& cache) const;

    RawNetworkOutput evaluate_raw(const Position&    pos,
                                  AccumulatorStack&  accumulatorStack,
                                  AccumulatorCaches& cache,
                                  int                bucket = -1) const;


    void verify(const std::function<void(std::string_view)>& f,
                const EvalFile&                              evalFile,
                std::filesystem::path                        evalfilePath) const;

    NnueEvalTrace trace_evaluate(const Position&    pos,
                                 AccumulatorStack&  accumulatorStack,
                                 AccumulatorCaches& cache) const;

    void load_external(const std::filesystem::path&, const std::filesystem::path&, EvalFile&);
    void load_internal(EvalFile&);

   private:
    HordeLegacyNetwork hordeLegacyNetwork;
};

static_assert(std::is_trivially_copyable_v<Network>);


}  // namespace Stockfish::Eval::NNUE

template<>
struct std::hash<Stockfish::Eval::NNUE::Network> {
    Stockfish::usize operator()(const Stockfish::Eval::NNUE::Network& network) const noexcept {
        return network.get_content_hash();
    }
};

#endif
