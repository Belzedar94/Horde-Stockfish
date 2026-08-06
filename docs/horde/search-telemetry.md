# Horde search telemetry

Search telemetry is available only in an explicitly instrumented build:

```text
make -C src ARCH=x86-64 EXTRACXXFLAGS=-DHORDE_SEARCH_TELEMETRY build
```

That build exposes `HordeSearchTelemetry`, defaulting to `false`. A normal
build contains neither the option nor the counters. With the runtime option
disabled, the deterministic Horde bench remains `440088` with best-move digest
`2783dc7f37887e5356802f77585b65d7e2776d65708187a4949cc89e2b810280`.

When enabled, the engine emits one summary followed by non-empty cells before
`bestmove`. Every cell is keyed by side to move, search-depth bucket, and White
piece-count bucket. Counters cover:

- legal and searched moves, fail-highs, final best-move rank, and branching;
- null-move, ProbCut, LMP, LMR, and PV re-search activity;
- capture and quiet futility, history, and SEE pruning;
- qsearch stand-pat, move-count, non-capture, futility, and SEE pruning;
- last-Horde-piece capture visibility and search;
- pruned White pawn pushes and White pawn candidates present when LMP fires;
- exact `horde_is_fortress()` sample count and elapsed nanoseconds.

The fortress predicate is sampled once per 1,024 visited nodes. Sampling runs
the const predicate without using its result, so it cannot alter the searched
value. It intentionally makes an enabled instrumented build slower.

These counters are observational. Counterfactual false-prune experiments must
use an isolated shadow search with separate transposition tables and histories;
they must never reuse the production search state.

## Isolated shadow search

Instrumented builds also expose `HordeSearchExperimentMask`, defaulting to
zero. Its additive bit IDs are fixed:

| Bit | Disabled search component |
| ---: | --- |
| 1 | Null-move pruning |
| 2 | ProbCut |
| 4 | Late-move pruning |
| 8 | Node futility |
| 16 | Capture futility |
| 32 | Capture SEE pruning |
| 64 | Quiet continuation-history pruning |
| 128 | Quiet futility |
| 256 | Quiet SEE pruning |
| 512 | Qsearch pruning |
| 1024 | Late-move reductions |
| 2048 | Razoring |

Mask zero preserves the accepted search. `tests/horde_shadow_search.py` starts
separate engine sessions, clears TT and histories for every position, compares
one disabled component at a time, and checks the result against a deeper
mask-zero reference. A changed shallow result that agrees with the deeper best
move is reported as a false-prune candidate, not as a confirmed regression.

Example:

```text
python tests/horde_shadow_search.py src/stockfish \
  --positions 1000 --depth 6 --reference-depth 8
```

The ordered positions come from the deterministic physical-`P` Horde generator.
The JSON receipt freezes the seed, depths, experiment IDs, changed results, and
candidate records.
