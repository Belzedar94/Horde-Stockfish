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
