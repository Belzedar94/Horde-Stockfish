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

## Invalidated OpenBench samples

### 2026-08-08 - V1 opening book and clock-safety reset

- A complete database audit found 75 time losses and zero crashes in 56,632
  Horde games across tests 172-229. Every workload used
  `HORDE_openings.epd`, `8.0+0.08`, and the engine default
  `Move Overhead=10`.
- The affected tests were 172 (13 time losses), 173 (2), 176 (4), 177 (1),
  178 (1), 181 (1), 183 (1), 185 (1), 194 (5), 197 (2), 198 (3), 213 (1),
  224 (3), 228 (19), and 229 (18). Tests with no reported defect were also
  stopped before further sampling so results from different book and clock
  contracts cannot be combined.
- Seventeen raw PGN failures from tests 173, 176, 177, 183, 194, 213, and 224
  were reconstructed under Horde rules. All decisive failures ended in a
  legal, non-terminal position immediately before the losing engine should
  move. Sixteen affected the Horde side and one affected the royal side.
  Eleven were candidate losses and six were baseline losses, proving a common
  clock defect while also showing that some diffs amplified it.
- Summing the per-move UCI times left between -187 and 510 milliseconds before
  the missing move. Seven games had already exhausted their nominal budget;
  the other ten had only enough reserve to be consumed by accumulated process
  and controller latency over 55-67 moves. Direct searches from every critical
  FEN returned in 11-22 milliseconds with only 20 milliseconds on the clock,
  excluding a deterministic search hang.
- Independently, the matched V1 book control supplied only 11.5% assignment-
  decisive pairs, `U = 0.0925`, 88% Black-Black pairs, and 93.25% Black wins.
  The low information rate and repeated book wrap make the defect-free subset
  unsuitable for comparing small search changes.
- Decision: all V1 workloads are invalid and remain stopped. No LLR or Elo
  from tests 172-229 may promote a change.
- Restart contract: wait until OpenBench PR 39 is reviewed, merged, deployed,
  and `HORDE_openings_v2.epd` is visible in Supported. Recreate selected tests
  from game zero with Run 6B on both sides and exact symmetric options
  `Threads=1 Hash=32 "Move Overhead=50"`. Any time loss, crash, illegal move,
  or abort invalidates the new sample immediately.

## Opening-book sensitivity follow-up

### 2026-08-08 - Deep-evaluation V3 discovery gate

- Collaborator feedback correctly identified that V2 still contains many
  Black-Black pairs. The optimization target remains paired-test information,
  not an artificial 50/50 color split.
- The first V3 selector used the 1,000-node generation score in a fixed White-
  relative `+80..+200` centipawn band. Although this band looked favorable in
  the 200-pair discovery receipt, its first disjoint 40-pair sample produced
  only 15% assignment-decisive pairs and was rejected rather than retuned on
  the same outcomes.
- The orthogonal follow-up rescored all 5,608 V2 positions at 20,000 nodes with
  MultiPV 2, then applied the same predeclared White-relative `+80..+200` band.
  Every position received a finite cp score; the rescore manifest SHA-256 is
  `9e582051644036b3691c83777f818e6e5bc6706957f639f5fb868332fdfd9677`.
- The held-out pools conservatively excluded every opening whose outcome had
  already been observed, including complete pairs from host-truncated probes.
  Three valid foreground shards supplied 80 games, 40 complete pairs, 40
  unique openings, no incomplete games, no abnormal termination, and maximum
  opening reuse one.
- Combined result: pentanomial `[5, 0, 27, 3, 5]`, 32.5% assignment-decisive
  pairs, `U = 0.26875`, 67.5% Black-Black pairs, 82.5% Black wins, 13.75%
  White wins, and 3.75% draws. The analysis SHA-256 is
  `1a64754225088dca547958003c008534854141bbb9c23fc3e1b3c8c580307d47`.
- Decision: the deeper score is a promising V3 generation constraint, but it
  does not replace V2 yet. The latest held-out pool has only 1,522 positions,
  below the 5,000-pair no-wrap capacity gate, and the independent sample is
  still only 40 pairs.
- Next gate: scale fresh deterministic generation until at least 5,000
  balanced, prefix-capped positions survive the unchanged deep band, then run
  a fresh 200-pair audit. Local referee invocations must be at most ten pairs
  because longer balanced openings repeatedly hit the host runtime boundary.
- Deployment state at this receipt: OpenBench PR 39 remains open and clean at
  `fc9eb10e6983e2ecbe5dd7a216af522590c4ae22`; production still exposes only
  `HORDE_openings.epd`, so no V2 strength workload can be registered yet.

## Prepared V2 restart queue

The V1 results below are used only to prioritize fresh V2 workloads. They do
not count as strength evidence and will not be combined with the new samples.

First wave, all at priority 1000:

1. `test/black-outcome-fastpath` at
   `bca714f45b99524f92057f9bd7144b6397636856`. The candidate returned the
   exact same result and reason as the accepted baseline on 100,000
   deterministic reachable Black-to-move positions plus twelve edge fixtures;
   every call also preserved FEN, key, state pointer, ply, rule50, repetition,
   and `pos_is_ok()`. Forty-eight alternating start-position depth-18 pairs
   measured a `1.012807` speed ratio, while sixteen alternating depth-16
   ten-position bench pairs measured `1.038213`; fixed work and best moves were
   identical. All four artifact jobs pass.
2. `test/white-outcome-has-move-fastpath` at
   `4be2e1656d61a48273e360264057feaff800c41a`. It replaces the complete White
   legal-list construction used only to answer whether a terminal move exists
   with an exact fixed-role bitboard predicate. The predicate matched full
   legal generation on 100,000 deterministic reachable White-to-move
   positions plus ten edge fixtures covering stalemate, promotion, en passant,
   and every physical piece type. Horde rules, the Run 6B contract, and three
   deterministic 315,576-node benches passed with the accepted best-move
   digest. Forty-eight alternating start-position depth-18 pairs measured a
   `1.031855` speed ratio; thirty-two alternating depth-16 ten-position bench
   pairs measured `1.035770`, with 28/32 favorable and a `1.035564` ratio after
   trimming one extreme from each tail. Fixed work and best moves were
   identical. All four artifact jobs pass.
3. `test/fortress-white-mobility-fastpath` at
   `f25edacd54d897b97590c67edc09c5fcfc82b74b`. It applies the same proven
   White legal-move predicate only after each legal Black move in the fortress
   scan, leaving direct White terminal detection unchanged so the two hot
   paths remain independently measurable. Its movegen implementation is
   byte-identical to the predicate that matched 100,010 full legal-generation
   receipts. Horde rules, the Run 6B contract, and three deterministic
   315,576-node benches passed with the accepted best-move digest. A
   start-position sample was nearly neutral at `1.006114`, as expected for
   opening-heavy work. Two independent 32-pair depth-16 ten-position bench
   samples measured `1.084210` and `1.025097`; their combined geometric ratio
   is `1.054239`, with 55/64 favorable pairs and identical fixed work and best
   moves. All four artifact jobs pass.
4. `test/fixed-royal-slider-blockers` at
   `acbacc48fdaabe49512d2dff89880f8f9ead95c6`. It replaces only the generic
   White king lookup and early return in `set_check_info()` with the fixed
   zero blockers/pinners state required by Horde, while retaining the complete
   Black royal-side calculation. Horde rules, the Run 6B contract, and three
   deterministic 315,576-node benches passed with the accepted best-move
   digest. Forty-eight alternating start-position depth-18 pairs measured
   `1.037351` geometric, `1.039771` trimmed, and 39/48 favorable. Thirty-two
   alternating depth-16 ten-position bench pairs measured `1.019999`
   geometric, `1.018788` trimmed, and 25/32 favorable. Fixed work and best
   moves were identical, and all four artifact jobs pass.
5. `test/white-pawn-quiet-see` at `1f3eb1d5685241f76834ed9a8c03bbbdb3fef00a`.
   Its defect-free V1 sample had LLR `+0.97` after 4,096 games.
6. `test/movepicker-legacy-pawn` at
   `a7161dc472672ac09d9e9616766e9bb1b01c37a1`. Its early V1 signal was LLR
   `+0.51`, but one time loss invalidates the 1,536-game sample.

Do not register the older combined `test/outcome-fastpath` commit
`bffa9a10ff4c3ce7ea61361f98d2814ffa1c6256` in the first wave. Its early V1
signal was LLR `+0.73`, but four time losses invalidate the 1,600-game sample,
and it combines the two independently measurable White-mobility call sites
now isolated above. It may be rebuilt as a merge-confirmation test only after
both orthogonal components pass independently.

Second wave consists of zero-game structural candidates with green artifact
CI: `test/horde-material-correction` and `test/white-piece-count-see-guard`. The
last branch now points to
`dfa4746188baaaebad4fe675ed0f10f692293e74`; its previous CI failure was only a
stale deterministic-bench receipt, and all four artifact jobs pass on the new
head.

Third wave keeps four additional ideas orthogonal: exact promotion SEE at
`11e9cad9c2d7525383d6d1842b6fd410a16e6a8a`, exact en-passant SEE at
`8261fa2c44140b67f3c6736851dfc56f19eecea0`, legacy Horde-pawn capture stat
score at `5ffb8496c5465efc39ab8cf2eb24397ab647dfde`, and legacy Horde-pawn
capture-futility value at `805e5bc2e94a8b9419f4301903b8e44cc558d27f`. Each has
green artifact CI and zero previous games. Promotion ordering, qsearch
futility, SEE value, and main-search futility remain separate experiments.

Do not restart candidates whose V1 direction was already poor without new
evidence. This includes `test/qsearch-legacy-pawn-futility` (LLR `-1.62` after
2,048 games), `test/see-legacy-white-pawn-value` (LLR `-0.85` after 1,752),
`test/white-pawn-advance-ordering` (LLR `-1.08` after 2,048),
`test/white-pawn-support-ordering` (LLR `-0.49` after 640),
`test/promotion-capture-ordering` (LLR `-1.42` after 1,536), and
`test/white-pawn-lmr` (LLR `-0.72` after 2,048).

## Registered experiments

### 2026-08-07 - Treat White Horde pawns as null-move material

- Branch: `test/white-pawn-nmp`
- Commit: `3bd83ee9fd9336010a4fe3befd89335a4d95d47e`
- OpenBench: [test 172](https://belzedar.duckdns.org/test/172/)
- Hypothesis: White Horde pawn mass is genuine search material, so the
  orthodox non-pawn-material gate should not disable null-move pruning across
  the pawn-only Horde opening and middlegame.
- Scope: one default-off UCI switch extending only the existing NMP material
  predicate to physical White pawns. NMP margins, reductions, verification,
  every other pruning stage, and Black search remain unchanged.
- Telemetry: a depth-16 start-position search recorded thousands of White NMP
  opportunities blocked solely by the orthodox material gate. A separate
  128-position depth-8/depth-10 shadow corpus changed five search trees but no
  root best move.
- Local cost screen: enabled mode was deterministic at 329,596 nodes on the
  frozen bench versus 315,576 with the switch disabled (+4.4%); individual
  corpus positions also showed substantial node reductions, so distributed
  games remain necessary to measure the NMP tradeoff.
- CI: the OpenBench artifact workflow passed for the exact test commit.
- Decision: promoted from priority 300 to priority 1000 for STC `[1.00, 6.00]`,
  with a 10,000-game neutral cutoff and a 20,000-game absolute ceiling.

### 2026-08-07 - Disable main-search capture futility

- Branch: `test/disable-capture-futility`
- Commit: `a19765137296d319a1fa2bce193368ab8fdfc8d2`
- OpenBench: [test 229](https://belzedar.duckdns.org/test/229/)
- Hypothesis: classical captured-piece futility is too noisy for Horde's
  extinction objective and discards captures that a deeper search prefers.
- Scope: one default-off UCI switch around the existing main-search capture
  futility stage. SEE pruning, qsearch, move ordering, margins, and every other
  pruning rule remain unchanged.
- Shadow evidence: a broad 256-position corpus generated between plies 6 and
  80 produced eight shallow best-move changes that aligned with the deeper
  reference when capture futility was disabled.
- Validation: default-off rules, Run 6B contract, and three frozen benches
  passed exactly at 315,576 nodes. Enabled mode was deterministic across three
  321,458-node benches with best-move digest
  `0bdba1861b3be41d9c0638cb2b83b43c134b1d7a0120dbfb74883127128b53d9`,
  a 1.86% node increase.
- CI: all four OpenBench artifact jobs passed on Windows and Linux for AVX2
  pext and popcnt.
- Decision: registered for STC `[1.00, 6.00]` at priority 1000, with a
  10,000-game neutral cutoff and a 20,000-game absolute ceiling.

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

### 2026-08-09 - Scale White child-node futility margin to 90%

- Branch: `test/white-node-futility-margin-90`
- Commit: `00bee8e995aa104728552e7dd7d856d1ee053401`
- OpenBench: [test 332](https://belzedar.duckdns.org/test/332/)
- Hypothesis: the severe loss from disabling White child-node futility does
  not establish that the orthodox final margin is optimal for Horde. A small
  increase in White-side pruning may recover depth while preserving the
  structurally valuable pruning stage.
- Scope: one default-off UCI scale applied only to the fully assembled
  child-node futility margin when White is to move. Value 100 is bit-identical
  to the accepted baseline; Black, the margin terms themselves, and every
  other pruning stage remain unchanged. The experiment enables value 90.
- Validation: Horde rules, the Run 6B contract, and three deterministic
  default-mode benches passed exactly at 315,576 nodes with best-move digest
  `fe9a5001c1997125ce34bf0ef119eab44570f5f363227bd4bab8e0db1f4e8592`.
  Enabled mode was deterministic across three 324,893-node benches with
  best-move digest
  `89c8009d315c0872f7bd2216e2841be21788c650c7f3c21c0e3a0e8573570d70`.
- Local screen: two independent 256-position V3 depth-7 shards were nearly
  neutral in aggregate against deeper references, 48-46, while a separate
  128-position depth-9/depth-12 shard favored value 90 by 25-21 and used
  10.4% fewer nodes. These screens justify an exploratory STC only.
- CI: all four OpenBench artifact jobs passed on Windows and Linux for AVX2
  pext and popcnt.
- Decision: registered at priority 301 for STC `[1.00, 6.00]`, with a
  10,000-game neutral cutoff and a 20,000-game absolute ceiling.

## OpenBench outcomes

### 2026-08-09 - Legacy dirty-piece tracking only passes STC

- Branch: `test/legacy-dirty-piece-only`, commit
  `69e52de5aebe2733862de5d75e17dd951e5495a2`.
- OpenBench: [STC test 295](https://belzedar.duckdns.org/test/295/) and
  [LTC test 331](https://belzedar.duckdns.org/test/331/).
- STC result: green at 8,704 games, 4,352-4,181-171,
  pentanomial `[309, 80, 3488, 81, 394]`, raw Elo `+6.83 +/- 4.26`, and
  LLR `+2.95` for SPRT `[1.00, 6.00]`.
- Infrastructure audit: both sides used Run 6B and
  `HORDE_openings_v3.epd`; the test recorded no crash or Horde error event.
- Decision: advance the identical single change to LTC at `40.0+0.40`,
  Threads=1 and Hash=128. It is not accepted into development until LTC also
  passes and the exact build and correctness receipts are rechecked.

### 2026-08-09 - Reject White-wide child-node futility disable

- Branch: `test/white-side-node-futility`, commit
  `7d8ae885b89adb1d9d1305ddbfea19c2401c9cc1`.
- OpenBench: [test 305](https://belzedar.duckdns.org/test/305/).
- Scope: one default-off UCI switch that disabled only child-node futility
  when White was to move. All margins, Black-side futility, other pruning,
  rules, evaluation, network, and book remained unchanged.
- Result: red at 756 games, 304-440-12, pentanomial
  `[75, 9, 281, 3, 10]`, raw Elo `-63.19 +/- 16.35`, and LLR `-2.96` for
  SPRT `[1.00, 6.00]`.
- Infrastructure audit: both sides used Run 6B and
  `HORDE_openings_v3.epd`; the reporting machine showed matched candidate and
  baseline NPS, with zero crashes and no Horde error row.
- Decision: rejected. Blanket removal sacrifices far more effective depth
  than it recovers from false prunes; any future White futility experiment
  must target a measured tactical class or narrow margin instead of exempting
  the whole Horde side.

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

### 2026-08-07 - Disable qsearch move-count pruning by side

- Hypothesis: the fixed two-move qsearch quota may be unreliable for only one
  of Horde's asymmetric objectives, allowing a side-specific exception without
  removing qsearch futility or SEE pruning.
- Scope: local-only White and Black selectors around the qsearch move-count
  cutoff; futility, SEE, checks, promotions, and main search remained unchanged.
- Shadow evidence: on 128 positions generated between plies 20 and 80,
  White-only produced 2 deeper-reference-aligned changes out of 20 changed
  shallow searches; Black-only produced 3 out of 40.
- Local cost screen: the frozen bench expanded from 315,576 nodes to 408,903
  for White-only (+29.6%) and 358,657 for Black-only (+13.7%).
- Decision: rejected locally because neither side concentrated enough
  correction signal to justify the search expansion; no commit, push, or
  OpenBench workload.
- Learning: any qsearch move-count candidate must target a tactical move class
  or a measured position bucket rather than exempting an entire side.

### 2026-08-07 - Revert early one-king singular verification

- Hypothesis: the accepted two-ply Horde singular-verification bonus may start
  exclusion searches too early and amplify noisy TT bounds.
- Scope: revert only the one-king depth bonus; preserve singular margins,
  extensions, multi-cut, and all other search policy.
- Shadow evidence: only 2 of 59 changed shallow results aligned with the deeper
  reference on a 256-position broad corpus.
- Local cost screen: +2.71% nodes on the frozen bench.
- Decision: rejected locally for insufficient correction signal; no commit,
  push, or OpenBench workload.

### 2026-08-07 - Disable capture SEE pruning globally

- Hypothesis: orthodox exchange thresholds suppress strategically necessary
  Horde captures even when they are not immediate extinction captures.
- Scope: disable only main-search capture SEE pruning; preserve capture
  futility, qsearch SEE, MovePicker staging, and SEE itself.
- Shadow evidence: 8 of 108 changed shallow results aligned with the deeper
  reference on a 256-position broad corpus.
- Local cost screen: +14.12% nodes on the frozen bench.
- Decision: rejected locally as too broad and overlapping the already
  registered targeted last-piece SEE guard experiment; no new workload.
- Learning: any future SEE candidate must identify a narrower Horde tactical
  class rather than remove the complete pruning stage.

### 2026-08-07 - Disable ProbCut by Horde side

- Hypothesis: classical capture-based ProbCut is unreliable under Horde's
  asymmetric objectives, and one side may account for most of its false-prune
  signal.
- Scope: local-only selector for disabling ProbCut on White, Black, or both;
  no margin, MovePicker, SEE, or other pruning change.
- Shadow evidence: on the same 256-position broad corpus, White-only,
  Black-only, and global disable produced 5, 5, and 8 deeper-reference-aligned
  best moves, respectively.
- Local cost screen: the frozen bench expanded by 34.9%, 25.9%, and 24.0% for
  White-only, Black-only, and global disable.
- Decision: rejected locally as disproportionate search expansion; no commit,
  push, or OpenBench workload.
- Learning: the ProbCut signal is not concentrated enough by side to preserve
  it cheaply, so this is not low-hanging fruit without a more exact Horde
  capture model.

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
- Scope: one constant in `src/nnue/nnue_architecture.h`; no other NNUE,
  evaluation, or search change is included.
- Branch: `test/legacy-accumulator-width-512`, commit
  `5f81b966ad45db255e47c2897a8aa44f352f921b`.
- Validation: BMI2 release build, Horde rules, Run 6B network contract, and
  three deterministic 315,576-node benches all passed exactly. All four
  Linux/Windows PEXT/POPCNT artifact jobs passed in GitHub Actions run
  `31314383079`.
- Local speed screen: +0.81% geometric mean, +0.29% median, and 7/12 positive
  alternating depth-16 pairs. Per-pair noise ranged from -12.15% to +17.77%
  under the live worker load.
- Decision: the local screen was inconclusive, so the isolated candidate was
  promoted to distributed STC as
  [OpenBench #333](https://belzedar.duckdns.org/test/333/) at priority 301.
  The workload is pending; no strength result is claimed yet.
- Learning: halving unused accumulator capacity improves memory footprint but
  needs distributed testing because the expected speed signal is smaller than
  the noise of the local screen.

### 2026-08-08 - Fixed-role `gives_check()` specialization

- Branch: `test/fixed-role-gives-check`, commit
  `84a1d621e4af870154863cc47b5a2f3db7b11faf`.
- Hypothesis: exploit Horde's fixed roles inside `Position::gives_check()` by
  rejecting Black immediately and using the fixed Black king and White
  attacker sets for the remaining paths.
- Scope: only check detection for candidate moves; move generation, legality,
  rules, evaluation, ordering, and search policy remained unchanged.
- Validation: Horde rules, the Run 6B contract, and three deterministic
  315,576-node benches passed with the accepted best-move digest.
- Local speed screen: 48 start-position pairs measured `0.980558` geometric,
  `0.980038` trimmed geometric, and only 17/48 favorable. A separate 32-pair
  ten-position bench measured `1.011674` geometric, `1.014311` trimmed
  geometric, and 21/32 favorable.
- Decision: rejected locally because the two suites disagree and the main
  start-position screen regresses by about two percent; no OpenBench workload.
- Learning: the generic path is already cheap for fixed Horde roles, while the
  added early branch and specialized layout can cost more than they remove.

### 2026-08-08 - Fixed-role `do_move()` checkers update

- Branch: `test/fixed-role-do-move-checkers`, commit
  `61a9a2d3c963975fce05288b7c6a698df35b319d`.
- Hypothesis: replace generic king-presence and side expressions in the
  post-move checkers update with Horde's fixed White-attacker/Black-king roles.
- Scope: only the `st->checkersBB` assignment in `Position::do_move()`; no
  move generation, legality, rules, evaluation, ordering, or search-policy
  change.
- Validation: a clean GCC 16 AVX2 build passed Horde rules, the Run 6B
  contract, and three deterministic 315,576-node benches with the accepted
  best-move digest.
- Local speed screen: the raw geometric ratios were `0.993936` over 48
  start-position pairs and `1.017472` over a separate 32-pair ten-position
  bench, but each suite contained a large host pause. The robust trimmed ratios
  were `1.000365` and `0.989420`; bench favored the candidate only 10/32 times.
- Decision: rejected locally as neutral or negative after robust trimming; no
  OpenBench workload.
- Learning: specializing this single bookkeeping expression does not remove a
  measurable hot-path cost and its layout perturbation is at least as large as
  the saved generic tests.

### 2026-08-08 - White legal-generation early return

- Branch: `test/white-legal-generation-fastpath`, commit
  `916fca9952262d44f3a2fc9ff2ec3129e2c5300f`.
- Hypothesis: after pseudo-legal generation, return the White move list
  immediately because Horde's non-royal side cannot require king-safety
  filtering.
- Scope: only `generate<LEGAL>()`; Black pin, king-move, en-passant, and all
  rule/search behavior remained unchanged.
- Validation: a clean GCC 16 AVX2 build passed Horde rules, the Run 6B
  contract, and three deterministic 315,576-node benches with the accepted
  best-move digest.
- Local speed screen: the ten-position bench measured `1.035073` geometric,
  `1.033011` trimmed, and 28/32 favorable. Two independent start-position
  samples disagreed at `0.987434` over 48 pairs and `1.020050` over 32 pairs;
  the latter contained extreme host pauses. Their pooled raw geometric ratio
  is only `1.000353`, with 50/80 favorable.
- Decision: rejected as a standalone OpenBench workload because its principal
  suite is neutral and it overlaps the stronger direct White-terminal
  fastpath. Retain it only as an interaction candidate if that orthogonal
  terminal change first passes STC.
- Learning: removing the White legality scan helps some mixed positions but
  does not produce a stable whole-search gain from the standard Horde root.

### 2026-08-08 - Remove redundant qsearch terminal recheck

- Branch: `test/qsearch-terminal-recheck`, commit
  `0259173d71b07c49326554ab98dd1423340d0e15`.
- Hypothesis: remove the inherited mate/stalemate recheck at qsearch exit
  because authoritative Horde `Outcome` already resolves the unchanged root
  position before TT, stand-pat, or move search.
- Scope: delete only the final qsearch move-count terminal block; entry
  outcomes, move generation, pruning, evaluation, and score smoothing remained
  unchanged.
- Validation: a clean GCC 16 AVX2 build passed Horde rules, the Run 6B
  contract, and three deterministic 315,576-node benches with the accepted
  best-move digest.
- Local speed screen: 48 start-position pairs measured `0.976385` geometric,
  `0.975427` trimmed, and only 13/48 favorable. The ten-position bench measured
  `1.025345` raw because of one host pause, but only `1.004699` trimmed and
  8/32 favorable.
- Decision: rejected locally; no OpenBench workload.
- Learning: the block is semantically redundant under the Horde outcome
  contract, but it is cold and deleting it worsens the hot code layout.

### 2026-08-08 - Fixed-role White legality gate

- Hypothesis: replace the generic `has_king(WHITE)` piece-count lookup in
  `Position::legal()` with the invariant fixed Horde role `WHITE`.
- Scope: one condition in `src/position.cpp`; castling rejection and every
  Black legality path remained unchanged.
- Validation: Horde rules, the Run 6B contract, and three deterministic
  315,576-node benches passed with the accepted best-move digest.
- Local speed screen: 48 start-position pairs measured `0.999089` geometric
  and only 19/48 favorable. Two independent 32-pair ten-position bench samples
  measured `1.020430` and `0.989705`; after removing one external pause from
  each tail, the second sample was only `1.011058`. Across the complete local
  screen the robust gain remained below one percent and was not stable between
  suites.
- Decision: rejected locally as insufficiently clear low-hanging fruit; no
  commit, push, or OpenBench workload. The source change was reverted.

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

### 2026-08-08 - Key White correction history by Horde material

- Branch: `test/horde-material-correction`, commit
  `10409b93f430c183e1729afbebcac43f315204b1`.
- Hypothesis: key the White non-pawn correction field by complete piece-count
  material, so changes in the Horde pawn mass can distinguish states that the
  orthodox White non-pawn key collapses.
- Scope: one default-off UCI switch and only the lookup/update key for the
  existing `nonPawnWhite` correction field; no rules, NNUE, pruning threshold,
  move ordering, or other correction weight changed.
- Validation: all four artifact CI jobs pass. A clean GCC 16 AVX2 build passed
  Horde rules and the Run 6B contract. With the switch disabled, three benches
  remained exactly 315,576 nodes with the accepted digest. With it enabled,
  three benches were deterministically 398,170 nodes with digest
  `8c236f6f19cc248bbb40c584ddb1bc97958b9e8d27faac5b5fcd2334f7206082`.
- Local search proxy: across two disjoint midgame corpora totaling 640
  positions at depth 7 against a depth-10 baseline reference, the changed
  moves matched the reference 29 times and the baseline moves 34 times. Node
  ratios were 0.974 and 1.074 on the two shards.
- Decision: retain only in the second OpenBench wave. The local proxy is
  neutral to slightly negative, so it does not displace candidates with a more
  concentrated signal and should be stopped promptly if STC stays neutral.
- Learning: the material key is technically sound and isolated in its own
  correction bundle field, but a count-only key is coarse and its search-tree
  effect is much larger than its shallow-reference signal.

### 2026-08-08 - Exact White piece-count SEE guard

- Branch: `test/white-piece-count-see-guard`, commit
  `dfa4746188baaaebad4fe675ed0f10f692293e74`.
- Hypothesis: replace the inherited White non-pawn-material approximation with
  an exact Horde unit count in the shallow capture-SEE stalemate guard.
- Scope: only that White guard; the Black condition and all SEE calculations,
  margins, ordering, and other pruning remained unchanged.
- Validation: all four artifact CI jobs pass, and the committed deterministic
  receipt is 372,356 nodes with digest
  `50e1f561153b9d016a65c4ccbf6c9a4049e7f2454a1fca96a9012f01b422b129`.
- Decision: rejected before OpenBench. The change increases the frozen bench
  tree by 18.0% and preserves an orthodox premise that does not transfer:
  sacrificing White's final Horde unit loses by extinction rather than
  creating a stalemate resource.
- Learning: exact Horde counting does not rescue a heuristic whose underlying
  terminal objective belongs to orthodox chess.

### 2026-08-08 - Remove White stalemate exception from capture SEE

- Hypothesis: keep the inherited last-piece stalemate exception for Black but
  always permit ordinary capture-SEE pruning for White, whose final-unit
  sacrifice is an extinction loss.
- Scope: one side condition in the shallow capture/check SEE guard; the
  extinction-capture override remained untouched.
- Validation: a clean GCC 16 AVX2 build passed Horde rules, the Run 6B
  contract, and three deterministic candidate benches at 335,277 nodes with
  digest
  `39dbe20b08451ec781b8ec4288487f8e380c4fb031e52a2132588fec6a71cd00`.
  In a 512-position depth-7 screen it changed only one best move; that move
  agreed less well with the depth-10 baseline reference. Total nodes were
  1.0047 times baseline.
- Decision: rejected locally as too rare and without a positive quality or
  speed signal; no commit, push, or OpenBench workload. The source and bench
  receipt were reverted to the accepted baseline.
- Learning: although the rule argument is cleaner, this inherited exception
  fires too rarely in representative Horde searches to qualify as low-hanging
  fruit on its own.
