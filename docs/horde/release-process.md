# Horde-Stockfish native release process

This document defines the native release-candidate path for Horde-Stockfish.
It follows the public Atomic-Stockfish and Spell-Stockfish convention of Linux
and Windows builds for AVX2 and BMI2 CPUs, while adopting Atomic-Stockfish's
authenticated manifest and reproducible-archive contract.

The current automation does not create tags or GitHub releases. It produces an
internal GitHub Actions artifact for inspection. Publication remains blocked
until every applicable gate in
[`testing-and-release-contract.md`](testing-and-release-contract.md) has a
receipt for the exact candidate commit.

## Candidate inventory

For version `X.Y.Z`, the workflow must produce exactly these archives:

- `Horde-Stockfish-X.Y.Z-linux-x86-64-avx2.tar.xz`
- `Horde-Stockfish-X.Y.Z-linux-x86-64-bmi2.tar.xz`
- `Horde-Stockfish-X.Y.Z-windows-x86-64-avx2.zip`
- `Horde-Stockfish-X.Y.Z-windows-x86-64-bmi2.zip`

The assembled artifact also contains:

- `horde-stockfish-release-manifest.json`
- `SHA256SUMS`

Each native archive is accompanied during assembly by a
`.provenance.json` descriptor. Those descriptors are authenticated inputs to
the manifest and are not separate public downloads.

Every archive has the same top-level layout:

```text
Horde-Stockfish-X.Y.Z/
  AUTHORS
  CITATION.cff
  Copying.txt
  README.md
  SOURCE.md
  bin/
    horde-stockfish[.exe]
  docs/
    BASELINE_MANIFEST.json
    RELEASE_NOTES_DRAFT.md
    RELEASE_PROCESS.md
    TESTING_AND_RELEASE_CONTRACT.md
  networks/
    CC0-1.0-NOTICE.md
    README.md
    Horde_v1.nnue
```

`Horde_v1.nnue` is the production package name for the byte-identical Run 6B
network. It is a packaging alias only: the source filename and
`EvalFileDefaultName` remain unchanged. Release smoke tests copy the executable
to a directory containing no `.nnue` file, so a successful benchmark also
proves that the candidate contains its embedded default network.

## Reproducibility boundary

`.github/workflows/horde-release.yml` is manual (`workflow_dispatch`) and has
read-only repository permissions. It resolves the requested source ref once,
then uses the resulting full commit for every job.

Linux and Windows archives are built in the same digest-pinned GCC and static
MinGW images used by the Atomic-Stockfish native release path. For each
platform and architecture, the workflow:

1. expands two independent `git archive` copies of the exact commit;
2. sets `SOURCE_DATE_EPOCH` to that commit's timestamp;
3. runs `scripts/horde/build_native_release.sh` in each clean copy;
4. requires both executables and both archives to be byte-identical;
5. runs the frozen Horde benchmark from an isolated runtime directory; and
6. writes a provenance schema-v2 descriptor with the source commit, toolchain,
   build command and producer-side SHA-256.

The assembler accepts exactly the four archives above. It rejects missing,
extra, duplicated, symlinked or tampered inputs, copies and re-hashes each
asset, then writes the Atomic-compatible schema-v1 manifest and `SHA256SUMS`.
The checksum file covers all four public archives and the manifest.

Reproducibility here means two isolated builds under the same immutable
toolchain recipe emit identical bytes. It does not claim that unrelated
compilers or operating systems produce identical binaries.

## Network boundary

The release workflow does not download, select, rename or rewrite an NNUE
network. The exact source commit supplies the default path in `src/evaluate.h`
and the tracked bytes below `networks/`. `scripts/horde/verify_baseline.py`
authenticates the current baseline before any candidate build.

A future champion-network change must be a separate reviewed source change. It
must update the network bytes, default path, attribution, license notice and
baseline manifest together, then repeat all affected correctness and strength
gates. Running the release workflow cannot make that decision.

## Formal strength panel

The public strength panel is frozen to this comparison contract:

- candidate: the latest reviewed Horde-Stockfish `main` commit, full commit
  `TBD`, with the Run 6B bytes distributed as `Horde_v1.nnue`;
- baseline: the Fairy-Stockfish development revision frozen at test start,
  full commit `TBD`, with `horde-28173ddccabe.nnue` (full SHA-256 `TBD`);
- paired game counts: 600 at 2s + 0.02s, 400 at 10s + 0.1s, and 200 at
  30s + 0.3s.

The opening-book SHA-256, runner commit, candidate commit, Fairy-Stockfish
commit and any other moving input remain `TBD` until the final panel is frozen.
Release notes must record those full identifiers before presenting scores.

[`scripts/horde/run_release_panel.py`](../../scripts/horde/run_release_panel.py)
implements this exact panel. It authenticates the two executables, two
networks, Horde referee and opening book before starting; rejects a book with
fewer than 300 positions; runs the three paired time controls concurrently;
and prints the complete WDL, pentanomial, Elo, confidence interval and LOS
table every five minutes. Any missing game, incomplete pair, crash, illegal
move, disconnect, stall or time loss writes `INVALID.json` and invalidates the
entire panel. It does not enable draw or resignation adjudication.

Before consuming machine time, invoke the runner with `--dry-run` and all
complete 40-character commits and 64-character SHA-256 values. Review the
three emitted cutechess commands, then repeat the same invocation without
`--dry-run` in an empty output directory. A valid completion produces the
original PGNs and logs plus `manifest.json`, one `result.json` per time
control, and `panel-receipt.json` with authenticated output hashes. The
candidate network SHA is fixed in the runner to the frozen Run 6B hash; every
moving asset hash must be supplied explicitly.

## Creating a candidate artifact

From GitHub Actions, run **Horde release candidate** with:

- `version`: an `X.Y.Z` candidate label;
- `source_ref`: a branch, tag or full commit in this repository (default:
  `main`).

The result is the single Actions artifact
`Horde-Stockfish-X.Y.Z-release`. No tag, draft or public release is created.

After downloading it, verify from inside the artifact directory:

```console
sha256sum --check SHA256SUMS
```

Also confirm that `commit` in `horde-stockfish-release-manifest.json` is the
intended full source commit and that its four artifact names exactly match the
inventory above.

## Publication checklist

Publication is a later, manual operation and requires all of the following:

1. The candidate commit is the reviewed commit intended for the release.
2. Required correctness, sanitizer, cross-platform, rule, NNUE, benchmark and
   strength gates have passed for that exact commit and network.
3. The final notes replace every placeholder in
   [`release-notes-draft.md`](release-notes-draft.md) with traceable evidence.
4. The downloaded candidate passes `SHA256SUMS`, manifest inventory and
   isolated Linux/Windows runtime verification.
5. The tag resolves to the manifest commit and the exact corresponding GPL
   source remains available.
6. A GitHub release is created as a draft, all advertised assets are attached,
   and the draft's downloaded bytes are re-authenticated before it becomes
   visible.

Never publish release notes before their complete asset set is attached. Never
move a stable release tag. A rolling prerelease, if added in a separate change,
must use a dedicated movable tag and must not be treated as a stable release.
