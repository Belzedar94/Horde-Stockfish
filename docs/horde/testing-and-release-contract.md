# Testing and release contract

This contract defines the minimum evidence required to call a Horde-Stockfish
revision reproducible, rule-correct or releasable. Passing a strength match
does not waive a rule or provenance gate.

## 1. Roles and authority

The baseline deliberately separates four roles:

| Role | Frozen input | Authority |
|---|---|---|
| Formal engine source | `fairy-stockfish/Fairy-Stockfish@c19b5f6c66894fdb0e88d0dd100e3885f744760a` | Build and engine-behavior baseline |
| Executable rule reference | `lichess-org/scalachess@d5d47c16f65a005ca68e19bab702b02f66dd888c` | Lichess Horde legality, results and perft |
| Canonical evaluation network | `networks/hordetest_run6b_e37_l06.nnue` | Frozen NNUE baseline |
| Historical binary | SHA-256 `3501bd84...c3ad3` | Oracle-only diagnostic |

The historical executable is not formal source evidence. It must not be
packaged, used to establish build reproducibility, promoted to a regression
baseline or cited as proof that a new release implements the pinned source.

## 2. Gate A: artifact integrity

Run from the repository root:

```console
python scripts/horde/verify_baseline.py
```

The verifier must confirm the byte size and SHA-256 of:

- the canonical NNUE;
- the CC0 notice;
- the exact `variants.ini` fixture;
- the copied Lichess perft fixture; and
- the mechanically translated HordeTest perft fixture.

Any mismatch is a hard failure. Updating a hash in place to silence the failure
is prohibited. A changed input requires a new baseline identifier, rationale
and review.

Remote source checks must use the full 40-character commits and the pinned blob
hashes in `baseline-manifest.json`. A moving branch, release alias or web page
is supporting context only.

## 3. Gate B: formal source build

The formal comparison binary must be built from the exact Fairy-Stockfish
commit in the manifest. Record at least:

- repository URL and full commit;
- whether the worktree is clean;
- compiler name and version;
- target triple, operating system and CPU architecture;
- build command and architecture target;
- compile-time flags;
- resulting binary size and SHA-256; and
- whether NNUE is embedded or loaded at runtime.

Build reproducibility means that another operator can repeat the same source
and toolchain recipe. It does not mean that unrelated compilers must emit
identical binaries.

The source distribution must preserve Fairy-Stockfish's GPL-3.0 obligations.
The network's CC0 dedication does not relax those obligations.

## 4. Gate C: UCI and variant smoke test

With a formally built binary:

```console
python scripts/horde/verify_baseline.py --engine /absolute/path/to/fairy-stockfish
```

The script copies only `variants.ini` into a temporary runtime directory,
launches `fairy-stockfish load variants.ini`, then requires:

1. `uciok` after `uci`;
2. `readyok` after selecting `UCI_Variant=hordetest`, setting `EvalFile`,
   `Threads=1` and `Hash=64`;
3. every HordeTest perft count through depth four; and
4. a legal `bestmove` from a deterministic `go depth 8` search.

The verifier refuses the known historical oracle binary by hash. A successful
smoke test proves only that the runtime contract works; the recorded build
provenance is still required.

## 5. Gate D: rule equivalence

Run both encodings:

- built-in `horde` with `fixtures/lichess-horde.perft`; and
- custom `hordetest` with `fixtures/hordetest.perft`.

The expected corpus is:

| Position | d1 | d2 | d3 | d4 |
|---|---:|---:|---:|---:|
| Start | 8 | 128 | 1274 | 23310 |
| Open flank | 30 | 241 | 6633 | 56539 |
| En passant | 13 | 172 | 2205 | 33781 |

The two encodings must agree exactly at every depth. A mismatch is a rules bug,
not a tolerable evaluation difference.

Add focused move-list tests for:

- rank-one and rank-two white double steps;
- the forbidden en-passant response to a rank-one double step;
- ordinary legal en passant in both directions where applicable;
- queen, rook, bishop and knight promotions;
- capture of the last Horde piece;
- survival of a promoted Horde piece after all pawns are gone;
- black self-check and white kingless legality;
- checkmate of the black king;
- repetition and move-count draws;
- stalemate/closed-fortress handling; and
- insufficient-material cases from the pinned Lichess implementation.

For each test, retain the FEN, move sequence, expected legal moves and expected
result. Tests that depend on a referee must record its source commit.

## 6. Gate E: differential rule testing

Perft is necessary but not sufficient. Generate a deterministic corpus from
legal Horde positions and compare the engine against the pinned Lichess rule
reference for:

- legal move sets;
- game-over state;
- winner or draw;
- check/checkmate state for Black;
- FEN round trips; and
- insufficient-material decisions.

Record the seed, generator version, position count and any exclusions. Every
difference requires classification. Do not filter a disagreement merely
because it is rare or occurs late in an endgame.

## 7. Gate F: search and NNUE correctness

Before measuring Elo, demonstrate:

- the engine reports that the intended `hordetest` network is loaded;
- NNUE-enabled search completes without assertion, crash or silent fallback;
- `useNNUE=false` or the equivalent classical mode remains available as a
  diagnostic when the engine supports it;
- repeated single-thread searches at fixed depth return stable nodes and
  bestmove under a fixed build;
- Horde terminal scores use mate/extinction semantics consistently; and
- promotions, en passant and low-material fortress positions survive search at
  multiple depths.

An engine that produces moves with NPS zero, silently rejects the network or
uses a different variant has failed even if a GUI displays a plausible line.

## 8. Gate G: performance

Benchmark the candidate and formal baseline on the same machine and record:

- CPU model, frequency policy and available instruction set;
- threads, hash, NUMA policy and process affinity;
- compiler and architecture target;
- warm-up procedure;
- benchmark positions;
- nodes, elapsed time and NPS; and
- mean, dispersion and run count.

Separate speed from strength. A search change that gains Elo while causing an
unexplained speed regression is not ready. A network-only release should not
claim engine-speed changes.

## 9. Gate H: strength

Strength testing uses paired colors and a frozen opening set. The opening file
must have a SHA-256 in the test record. Do not use the historical `horde.epd`
implicitly; it is old, contains duplicate positions and may encode path
weights. If it is used, preserve duplicates and cite its frozen blob.

At minimum report:

- exact candidate and baseline source/network hashes;
- match runner and version;
- opening book hash and selection policy;
- adjudication and draw rules;
- time control, increment and concurrency;
- threads, hash and tablebases;
- games, wins, losses and draws;
- Elo estimate, confidence interval and LOS; and
- crashes, time losses and invalid games.

Use at least one fast and one slower time control. Historical cross-list Elo
claims are context, not acceptance evidence. A direct match cannot be compared
numerically with a rating-list delta without explaining the different baseline.

## 10. Gate I: packaging and release

A release candidate must include or link to:

- exact corresponding GPL source;
- build instructions;
- binary checksum and platform/architecture label;
- `baseline-manifest.json`;
- network checksum, Belzedar credit and CC0 notice;
- rule fixture provenance;
- test report satisfying the gates above; and
- known limitations and compatibility notes.

Never ship the historical oracle executable as the candidate. Never claim the
experimental Run 6B network is the current official Fairy-Stockfish Horde net;
the public Fairy-Stockfish list currently names `horde-28173ddccabe.nnue` for
the built-in `horde` representation. Run 6B belongs to the distinct
`hordetest` feature encoding.

## 11. Stop-ship conditions

Any of the following blocks release:

- artifact hash mismatch;
- unrecorded source commit or dirty formal source tree;
- legal-move or result disagreement with the pinned rule profile;
- failed perft at any required depth;
- inability to load the canonical network explicitly;
- crash, hang, protocol failure or silent NNUE fallback;
- use of the oracle-only binary as formal evidence;
- missing GPL source availability or CC0 notice;
- irreproducible opening/adjudication settings; or
- a result claim that cannot be traced to raw match totals.

Waivers must identify the failed gate, evidence, owner and expiry. A waiver may
permit further experimentation; it must not relabel a failed build as a
release.
