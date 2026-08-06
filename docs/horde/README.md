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
