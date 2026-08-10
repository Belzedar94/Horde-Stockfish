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

- Horde-specific legality, terminal conditions and search behavior.
- NNUE evaluation authenticated against the network and baseline manifests.
- Standard UCI, deterministic Horde benchmark, multi-threading and large hash
  support.
- Linux and Windows packages for `x86-64-avx2` and `x86-64-bmi2`.

TBD: replace this list with the exact user-visible scope of the release. Do not
claim features or strength that are not covered by an exact-commit receipt.

## Usage

Download the archive matching your operating system and CPU. Use
`x86-64-bmi2` for compatible Haswell/Zen 3 or newer systems and
`x86-64-avx2` as the safer fallback.

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

- TBD: list any unresolved compatibility or performance limitations.
- The release does not include a GUI.
- Horde tablebase support is not claimed.

## Acknowledgements

Built on the work of the Stockfish, Fairy-Stockfish and Horde chess
communities. Testing infrastructure is based on OpenBench. Network authors and
data contributors must be credited in the final network section.
