# Horde-Stockfish Strength Testing Ledger

This ledger records reproducible strength experiments against the accepted
Horde-Stockfish baseline. Public OpenBench tests use Run 6B on both sides and
the `LICHESS_HORDE_V1` contract.

Current baseline: `cee98c4d2f41295378c9cc02a9fb5153ae956d73`.

## Experiment policy

- Test one orthogonal idea per branch and OpenBench workload.
- Reject local speed candidates when paired measurements are neutral or
  negative; do not spend distributed games on them.
- Stop statistically neutral OpenBench tests at approximately 10,000 games.
  Twenty thousand games is an absolute ceiling, not a target.
- Promote a clean STC pass to one LTC workload for the identical diff.
- Never promote a result with crashes, time losses, illegal moves, or an
  infrastructure defect.
- Local speed screening uses identical BMI2 builds and alternating paired
  benches. OpenBench remains authoritative for playing strength.

## Registered experiments

### 2026-08-07 — Legacy dirty-piece tracking only

- Branch: `test/legacy-dirty-piece-only`
- Commit: `69e52de5aebe2733862de5d75e17dd951e5495a2`
- OpenBench: [test 228](https://belzedar.duckdns.org/test/228/)
- Hypothesis: avoid building standard-NNUE threat and pawn-pair deltas in
  `Position::do_move()` because the fail-closed Run 6B legacy evaluator consumes
  only `DirtyPiece`.
- Scope: `src/position.cpp`, ten insertions and sixteen deletions. The dirty
  structures themselves remain unchanged so this workload measures only the
  per-move bookkeeping removal.
- Correctness: Horde rules, Run 6B network contract, three deterministic
  315,576-node benches, 10,000-position incremental/full-refresh determinism
  across one, two, and four threads, and 10,000-position differential parity
  against the pinned Fairy-Stockfish oracle all passed exactly. Coverage
  included 3,474 captures, 240 promotions, 33 en passant moves, 22 castlings,
  and every one-to-four-White-piece bucket.
- Shadow validation: an assertions build that compares every incremental
  evaluation with full refresh passed Horde rules and the three deterministic
  benches with digest
  `fe9a5001c1997125ce34bf0ef119eab44570f5f363227bd4bab8e0db1f4e8592`.
- Local speed screen: +8.7% combined geometric mean over two independent
  blocks of twelve alternating depth-16 pairs; every pair in the second block
  was positive.
- CI: all four OpenBench artifact jobs passed on Windows and Linux for AVX2
  pext and popcnt.
- Decision: registered for STC `[1.00, 6.00]` with a 10,000-game neutral
  cutoff and a 20,000-game absolute ceiling.

### 2026-08-07 — Fixed-role `do_move()` checkers update

- Branch: `test/fixed-role-do-move-checkers`
- Commit: `61a9a2d3c963975fce05288b7c6a698df35b319d`
- OpenBench: [test 227](https://belzedar.duckdns.org/test/227/)
- Hypothesis: avoid a dynamic king-presence lookup in the per-move checkers
  update by using Horde's fixed White-Horde and Black-Royal roles.
- Scope: `src/position.cpp`, two insertions and one deletion.
- Validation: Horde rules, Run 6B contract, and three deterministic benches
  passed at 315,576 nodes with best-move digest
  `fe9a5001c1997125ce34bf0ef119eab44570f5f363227bd4bab8e0db1f4e8592`.
- Local speed screen: approximately +1.1% geometric mean over 24 alternating
  depth-16 pairs, 1,710,990 nodes per run.
- Decision: registered for STC `[1.00, 6.00]`.

## Local rejects

### 2026-08-07 - Black node-futility margin scaling

- Hypothesis: preserve Black node futility while increasing its margin enough
  to recover the royal-side false-prune signal at much lower cost than a full
  disable.
- Scope: one Black-only integer percentage applied to the existing child-node
  futility margin; value 100 is bit-identical to the accepted baseline.
- Validation: value 100 reproduced the frozen 315,576-node bench and best-move
  digest exactly across three runs; the telemetry build also passed its
  switch-zero determinism gate.
- Local screen: values 101-103 produced no depth-confirmed corrections in two
  independent 128-position opening and midgame corpora. In a separate broad
  256-position corpus, values 105, 115, and 125 produced only 1, 1, and 2
  deeper-reference-aligned moves, respectively, with no stable stratum.
- Cost behavior: the frozen bench was discontinuous even for small changes
  (+22.9% nodes at 105), while corpus node deltas did not predict a robust
  correction rate.
- Decision: rejected locally; no commit, push, or OpenBench workload.
- Learning: margin scaling does not retain enough of the full-disable signal to
  justify remote testing, and bench sensitivity makes this a poor low-hanging
  parameter family.

### 2026-08-07 - Disable Black node futility

- Hypothesis: isolated shadow searches showed more depth-confirmed node-futility
  corrections on the royal side than on the Horde side, especially across
  positions generated between plies 20 and 80.
- Scope: disable child-node futility only when Black is to move; preserve every
  other pruning rule and all White-side node futility.
- Evidence: disabling node futility produced 35 deeper-reference-aligned move
  changes in 512 opening positions. In a separate 128-position midgame sample,
  13 changes aligned and 9 of them were Black-to-move positions.
- Validation: Horde rules and the fail-closed Run 6B contract passed. Three
  candidate benches were exact at 576,890 nodes with best-move digest
  `2bac2fb049b94fe02f1055efca8e6dfdeef8d01529e19db860233dfc61313f50`.
- Local cost screen: +82.8% nodes versus the frozen 315,576-node baseline bench.
- Decision: rejected locally as a disproportionate search expansion; no
  commit, push, or OpenBench workload.
- Learning: the asymmetric false-prune signal merits a Black-specific margin
  experiment, not wholesale removal of the pruning stage.

### 2026-08-07 - Skip terminal recomputation in singular verification

- Hypothesis: an excluded-move singular verification searches the unchanged
  position after its parent already established that the authoritative Horde
  state is non-terminal, so it can skip the repeated mobility and fortress
  computation.
- Scope: one guard in `search()`; no rule, score, move-ordering, pruning, or
  search-margin change.
- Validation: Horde rules, fail-closed Run 6B contract, and three deterministic
  315,576-node benches passed exactly.
- Local speed screen: -0.32% geometric mean, -0.42% median, and 4/12 positive
  alternating depth-16 pairs.
- Decision: rejected locally; no commit, push, or OpenBench workload.
- Learning: singular-verification terminal recomputation is redundant but not
  hot enough to overcome the resulting code-layout change.

### 2026-08-07 - Explicit legacy accumulator SIMD

- Branch: `test/legacy-accumulator-simd`.
- Hypothesis: replace scalar Run 6B accumulator and PSQT lane updates with the
  engine SIMD primitives.
- Scope: only the legacy accumulator update implementation; no search-policy
  or evaluation change.
- Validation: deterministic bench remained exact and the OpenBench artifact CI
  passed on commit `780ab46ca81dca1644366ecc7a3eae6ec10ccf14`.
- Local speed screen: -0.41% geometric mean, +0.18% median, and 4/8 positive
  alternating depth-16 pairs.
- Decision: rejected locally; no OpenBench workload.
- Learning: the optimized compiler already vectorizes these fixed-size loops
  effectively, while the explicit abstraction adds code-layout cost.

### 2026-08-07 - Fused legacy accumulator delta

- Branch: `test/legacy-accumulator-fused-delta`.
- Hypothesis: compute source-minus-from-plus-to/remove/add in one lane pass
  instead of copying the accumulator and applying each feature column in
  separate passes.
- Scope: only the legacy incremental accumulator delta, guarded by one UCI
  experiment switch during local validation.
- Validation: deterministic bench remained exact and the OpenBench artifact CI
  passed on commit `6acdd0f381099e1aa8ed55afb9822ed968a2ee1e`.
- Local speed screen: -1.97% geometric mean, -0.84% median, and 3/8 positive
  alternating depth-16 pairs.
- Decision: rejected locally; no OpenBench workload.
- Learning: the larger specialized dispatch and fused loop do not beat the
  compiler-optimized copy-plus-delta sequence on this workload.

### 2026-08-07 - Legacy-only NNUE chassis

- Hypothesis: remove the unreachable standard-NNUE transformer, network,
  Finny-table and multi-delta accumulator infrastructure so every search ply
  stores only the exact Run 6B 512-lane state and one `DirtyPiece`.
- Scope: one cohesive backend specialization across the NNUE network and
  accumulator adapters, `Position::do_move()`, and the build source list. No
  rule, evaluation, move-ordering, pruning, or search-policy change was mixed
  into the candidate.
- Validation: Horde rules, fail-closed Run 6B contract, three deterministic
  315,576-node benches, 10,000-position T1/T2/T4 determinism, and exact raw and
  final evaluation parity against the pinned Fairy-Stockfish oracle on 100,000
  reachable positions all passed. The differential corpus covered 37,263
  captures, 2,393 promotions, 459 en passant moves, 247 castlings, every
  one-to-four-White-piece bucket, and rule50 counts 0/50/90/99.
- Local speed screen: +2.75% geometric mean, -0.38% median, and only 5/12
  positive alternating depth-16 pairs. Per-pair noise ranged from -6.63% to
  +22.42% under the live worker load.
- Decision: rejected locally as insufficiently clear low-hanging fruit; no
  commit, push, or OpenBench workload.
- Learning: removing dead generic infrastructure substantially simplifies the
  binary and per-thread memory layout, but it does not remove enough measured
  hot-path work to justify distributed games on its own.

### 2026-08-07 — Match generic accumulator width to Run 6B

- Hypothesis: reduce the inherited generic NNUE accumulator from 1,024 to 512
  lanes, matching the only supported Run 6B evaluator and saving 2,048 bytes
  per ply (about 506 KiB per thread across the accumulator stack).
- Scope: one constant and its explanatory comment in
  `src/nnue/nnue_architecture.h`.
- Validation: BMI2 release build, Horde rules, Run 6B network contract, and
  three deterministic 315,576-node benches all passed exactly.
- Local speed screen: +0.81% geometric mean, +0.29% median, and 7/12 positive
  alternating depth-16 pairs. Per-pair noise ranged from -12.15% to +17.77%
  under the live worker load.
- Decision: rejected locally as insufficiently clear low-hanging fruit; no
  commit, push, or OpenBench workload. The source change was reverted.
- Learning: halving unused accumulator capacity improves memory footprint but
  does not remove enough hot-path work to justify STC by itself.

### 2026-08-07 — Fixed-role check-square setup

- Hypothesis: replace the generic opponent-king detection and orientation in
  `Position::set_check_info()` with fixed Horde roles.
- Scope: one isolated `src/position.cpp` change. The Black-side zeroing was
  deliberately retained because MovePicker reads `checkSquares` directly.
- Validation: Horde rules, Run 6B contract, and deterministic bench all passed.
- Local speed screen: -0.22% geometric mean and -0.67% median over 12
  alternating depth-16 pairs.
- Decision: rejected locally; no commit, push, or OpenBench workload.
- Learning: the generic role expressions are already optimized effectively;
  specializing them changed code layout without removing meaningful work.

### 2026-08-07 — Fixed-role SEE pin filtering

- Hypothesis: skip pin filtering for White attackers inside `Position::see_ge()`
  because the Horde side has no king and cannot have pinned attackers.
- Scope: one isolated `src/position.cpp` loop condition.
- Validation: Horde rules, Run 6B contract, and deterministic bench all passed.
- Local speed screen: -2.68% geometric mean and -2.15% median over 12
  alternating depth-16 pairs.
- Decision: rejected locally; no commit, push, or OpenBench workload.
- Learning: the additional color branch harmed the SEE loop more than the
  eliminated zero-valued pinner/blocker lookups saved.
