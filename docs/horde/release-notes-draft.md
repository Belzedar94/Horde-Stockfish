# Horde-Stockfish X.Y.Z release notes draft

> This is a publication template, not a release announcement. Replace every
> `TBD` value and remove this notice only after the exact candidate has passed
> the complete release contract.

Horde-Stockfish X.Y.Z is a UCI chess engine specialized in Horde chess,
derived from a current Stockfish framework with Horde-specific rules, search
and NNUE evaluation.

## Strength

The formal panel compares the latest reviewed Horde-Stockfish `main` candidate
(full commit `TBD`) with `Horde_v1.nnue` against Fairy-Stockfish dev (full
commit `TBD`) with `horde-28173ddccabe.nnue` (full SHA-256
`28173DDCCABE12306D02AFA1156DED2B6A69C6A8DB909895DB6E955F8B4AD6A6`).
The opening book SHA-256 and match-runner commit are also `TBD` until the panel
inputs are frozen.

Insert only results produced by that exact comparison. Include W/L/D, Elo with
confidence interval, LOS, crashes and time losses.

| Time control | Games | Score | Elo |
|---|---:|---|---:|
| 2s + 0.02s | 600 | TBD | TBD |
| 10s + 0.1s | 400 | TBD | TBD |
| 30s + 0.3s | 200 | TBD | TBD |

## Features

- Native Lichess Horde legality, including the kingless White side, Horde pawn
  double steps, Black-only castling, en passant and ordinary promotions.
- Horde terminal handling for checkmate, extinction, stalemate, closed
  fortresses and side-specific winning material.
- Authenticated legacy H/P NNUE evaluation with full-refresh/incremental and
  scalar/SIMD parity against the frozen Run 6B contract.
- A fail-closed `EvalFile` contract: unregistered networks are rejected instead
  of falling back to another evaluator.
- Horde-specific search behavior on a modern Stockfish chassis, with a
  deterministic Horde benchmark, multi-threading and configurable hash.
- Linux and Windows native packages for `x86-64-avx2` and `x86-64-bmi2`.

## Usage

Download the archive matching your operating system and CPU. Use the BMI2
package only when the CPU advertises BMI2 support; otherwise use AVX2.

Extract the complete archive without changing its directory layout, then start
the executable below `bin/` from a UCI-compatible graphical interface or the
command line. The authenticated Run 6B network is included as
`networks/Horde_v1.nnue` and is also embedded in the release binary.

Horde-Stockfish does not include a graphical interface. Chess960 and orthodox
Syzygy tablebases are not supported by this Horde engine.

## Network

The production asset name is `Horde_v1.nnue`. Its bytes are the frozen Run 6B
HordeTest network currently recorded in `BASELINE_MANIFEST.json`, authored by
Belzedar and made available under CC0-1.0. The source/default filename remains
unchanged; `Horde_v1.nnue` is the release-package alias.

The competing Fairy-Stockfish network is `horde-28173ddccabe.nnue`, SHA-256
`28173DDCCABE12306D02AFA1156DED2B6A69C6A8DB909895DB6E955F8B4AD6A6`.
Do not describe Run 6B as the official Fairy-Stockfish Horde network.

## Checksums (SHA-256)

Download `SHA256SUMS` and `horde-stockfish-release-manifest.json` with the
archives, then run:

```console
sha256sum --check SHA256SUMS
```

TBD: paste the four authenticated archive checksums and the manifest checksum
from the final candidate artifact.

## Known limitations

- The release does not include a GUI.
- Only standard 8x8 Lichess Horde is supported. Chess960 and orthodox chess are
  outside the public engine contract.
- Horde tablebase support is not included.
- The production evaluator is the registered Run 6B legacy network. NNUE V2 is
  experimental and is not part of this release.
- Initial native packages target x86-64 Linux and Windows with AVX2 or BMI2;
  macOS, ARM and portable baseline packages are not included.

## Acknowledgements

Run 6B was authored by Belzedar and is distributed under CC0 1.0. Horde-Stockfish
builds on the work of the Stockfish, Fairy-Stockfish, scalachess and Horde chess
communities. Distributed testing is provided by OpenBench.
