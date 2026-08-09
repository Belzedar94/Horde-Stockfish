# Horde-Stockfish baseline documentation

This directory freezes the evidence and contracts required to develop
Horde-Stockfish without losing rule, source or network provenance.

## Start here

- [Intelligence dossier](intelligence-dossier.md) — historical research,
  current upstream state, NNUE chronology, books, defects and open questions.
- [Horde from first principles](first-principles-horde.md) — game semantics and
  the distinction between Horde rules and the custom `H` feature encoding.
- [Baseline manifest](baseline-manifest.json) — machine-readable immutable
  source, rule, fixture, book, binary-oracle and network identities.
- [Reproducing the HordeTest baseline](hordetest-baseline.md) — the minimal
  formal-source build and UCI/perft probe.
- [Testing and release contract](testing-and-release-contract.md) — mandatory
  integrity, rules, search, performance, strength and packaging gates.
- [V1 compatibility receipt](v1-compatibility-receipt.md) — completed rules,
  Run 6B parity, incremental, determinism and benchmark evidence.
- [Search telemetry](search-telemetry.md) — opt-in Horde pruning and branching
  counters for instrumented builds.
- [NNUE V2 design](nnue-v2-design.md) — fixed-role, dual-domain architecture
  and the orthogonal feature-ablation ladder.
- [NNUE V2 width receipt](nnue-v2-width-receipt.json) - value-identical AVX2
  search NPS, confidence intervals and the first training-width gate.
- [NNUE V2 integer-container receipt](nnue-v2-integer-container-receipt.json) -
  authenticated checkpoint export, C++ full refresh and scalar/AVX2/Python
  parity for the first two registered experimental schemas.
- [NNUE V2 incremental-container receipt](nnue-v2-incremental-container-receipt.json) -
  lazy real-`Position` make/undo/null integration, domain-specific delta and
  refresh parity, sanitizer coverage and exact-source width evidence.
- [NNUE V2 Rank-8 control receipt](nnue-v2-rank8-control-receipt.json) -
  compact Royal topology, three-schema container parity and the frozen
  ABS/Rank-8/Royal-32 comparison gate.
- [NNUE V2 C1 campaign contract](../../schemas/horde-v2-c1-campaign-v1.json) -
  authenticated 250k/250k split, coverage audit, paired-seed training matrix
  and fail-closed pre-selection evidence for ABS, Rank-8 and Royal-32.
- [NNUE V2 C1 data-repair addendum](../../schemas/horde-v2-c1-data-repair-v1.json) -
  frozen overproduction and label-blind first-eligible selection of the exact
  250,000-record validation role after the direct split exposed transpositions.
- [NNUE V2 C1 coverage addendum](../../schemas/horde-v2-c1-coverage-addendum-v1.json) -
  pre-training, exact-data amendment that preserves the failed V1 preflight and
  gates ABS, Rank-8 and Royal-32 on structural keys, roles and seen row mass.
- [NNUE V2 C1 quantized screen](../../schemas/horde-v2-c1-quantized-screen-v1.json) -
  exact integer validation, paired three-seed stability gates and deterministic
  nomination of at most one subsequent fixed-node comparison.
- [Training-data contract V1](datagen-v1.md) — the isolated generator,
  physical-position wire format and G0 audit boundary.
- [Horde WDL calibration V1](wdl-calibration-v1.md) defines the frozen
  side-specific Davidson link and half-Brier training objective.
- [Fresh legacy-control V3 canary receipt](fresh-legacy-control-v3-canary-receipt.json)
  freezes authenticated labels, WDL calibration and byte-identical training.
- [Fresh legacy-control canary receipt](fresh-legacy-control-canary-receipt.json)
  preserves the original scalar-loss plumbing as historical evidence.
- [Fixtures](fixtures/README.md) — pinned Lichess perft and its exact HordeTest
  `P`-to-`H` translation.

The canonical NNUE is stored at
[`networks/hordetest_run6b_e37_l06.nnue`](../../networks/hordetest_run6b_e37_l06.nnue)
with a [CC0 notice](../../networks/CC0-1.0-NOTICE.md).

## Trust boundaries

- Fairy-Stockfish commit
  `c19b5f6c66894fdb0e88d0dd100e3885f744760a` is the formal engine-source
  baseline.
- Lichess `scalachess` commit
  `d5d47c16f65a005ca68e19bab702b02f66dd888c` is the executable Horde-rule
  reference.
- Run 6B is the canonical `hordetest` evaluation network, credited to Belzedar
  and frozen by SHA-256.
- The 2025 BMI2 executable recorded in the manifest is oracle-only. It is not a
  formal baseline, release input or distributable artifact under this contract.
- The old `hordetest` upstream branch is research evidence. It is not the source
  base for current implementation.

Run the offline integrity check from the repository root:

```console
python scripts/horde/verify_baseline.py
```

No strength or release claim follows from artifact integrity alone. Apply the
full testing and release contract.
