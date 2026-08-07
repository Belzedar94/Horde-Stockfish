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
