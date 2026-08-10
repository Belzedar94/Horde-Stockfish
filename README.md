# Horde-Stockfish

[![Horde correctness](https://github.com/Belzedar94/Horde-Stockfish/actions/workflows/horde-correctness.yml/badge.svg)](https://github.com/Belzedar94/Horde-Stockfish/actions/workflows/horde-correctness.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](Copying.txt)

Horde-Stockfish is a dedicated UCI engine for
[Lichess Horde chess](https://lichess.org/variant/horde), derived from modern
[Stockfish](https://github.com/official-stockfish/Stockfish). Its Horde rules
are checked against [scalachess](https://github.com/lichess-org/scalachess),
while [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish)
defines the frozen legacy NNUE format and formal development baseline.

> [!WARNING]
> This repository is under active development and does not yet have a stable,
> strength-qualified release. The production evaluator is the authenticated
> Run 6B network. NNUE V2 remains experimental until its full-data candidate
> passes parity, speed and equal-time strength gates.

## Scope

The project deliberately supports one variant and a narrow public contract:

- standard 8x8 Lichess Horde rules;
- UCI with `position startpos` loading the Horde starting position;
- `UCI_Variant` fixed to `horde`;
- deterministic Horde benchmark, multi-threading and configurable hash;
- the registered legacy HordeTest H/P NNUE schema; and
- Linux and Windows native release targets for AVX2 and BMI2 CPUs.

Chess960, orthodox chess, Syzygy tablebases, standard-chess NNUE networks and
silent evaluation fallbacks are outside the release contract.

## Neural network

The current default is
[`hordetest_run6b_e37_l06.nnue`](networks/hordetest_run6b_e37_l06.nnue),
SHA-256
`B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7`.
It was authored by Belzedar and is distributed under CC0 1.0; see the
[network notice](networks/CC0-1.0-NOTICE.md).

Public FEN and UCI positions continue to use `P` for every pawn. The historical
`H` identity exists only at the legacy NNUE feature boundary. The engine
authenticates compatible networks instead of inferring their schema from file
size or shared header hashes.

Native release packages call the same byte-identical network `Horde_v1.nnue`.
That name is a distribution alias; the tracked source filename and default
engine contract remain unchanged.

## Building

From `src`, inspect the supported targets with `make help`. A BMI2 build on a
compatible 64-bit Intel or AMD CPU can be produced with:

```sh
cd src
make -j2 ARCH=x86-64-bmi2 build
```

The direct Makefile output is `stockfish` (`stockfish.exe` on Windows), but its
UCI identity is `Horde-Stockfish`. Release packages rename the executable to
`horde-stockfish`.

## Usage

Horde-Stockfish does not include a graphical interface. Start the engine from
a UCI-compatible GUI with Horde support, or use it directly:

```text
uci
setoption name Threads value 1
setoption name Hash value 128
position startpos
go movetime 1000
```

The tracked default network is embedded by the native release build. Setting
`EvalFile` to an unregistered or incompatible network fails closed.

## Validation

The quickest local integrity and engine checks are:

```sh
python scripts/horde/verify_baseline.py
python tests/horde_rules.py src/stockfish
python tests/horde_network_contract.py src/stockfish
python tests/horde_bench.py src/stockfish
```

The complete CI matrix additionally covers the material corpus, canonical
perfts, 100,000-position legacy NNUE parity, incremental refresh, scalar/SIMD
and thread determinism, sanitizers, Windows/Linux builds, NNUE V2 experimental
contracts and reproducible release packaging.

See the [Horde documentation index](docs/horde/README.md), the
[testing and release contract](docs/horde/testing-and-release-contract.md) and
the [native release process](docs/horde/release-process.md) for the frozen
authorities, exact gates and publication procedure.

## Release policy

A public release requires an exact final `main` commit and network, green
correctness and packaging gates, and a paired 600/400/200-game strength panel
at the three frozen time controls. The release workflow builds authenticated
candidate artifacts only; it never creates or moves a tag and never publishes
a GitHub release automatically.

## Attribution and license

Horde-Stockfish retains Stockfish's copyright notices and is distributed under
the [GNU General Public License version 3](Copying.txt). See [AUTHORS](AUTHORS)
for project and upstream attribution. Fairy-Stockfish and scalachess remain
independently versioned references rather than vendored code.
