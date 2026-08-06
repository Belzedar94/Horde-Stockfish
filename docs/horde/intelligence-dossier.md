# Horde-Stockfish intelligence dossier

Status date: 2026-08-06

This dossier consolidates the available technical record for Horde rules,
Fairy-Stockfish support, neural networks, books, tests, known defects and the
HordeTest line of work. It distinguishes primary-source facts, archived
discussion evidence and interpretation. Historical Elo statements are retained
with their original baselines so they are not mistaken for current ratings.

## 1. Executive findings

1. Horde is a genuinely asymmetric game, not standard chess with extra white
   pawns. White has no king, Black has a royal king, Black wins by extinction
   and White wins by checkmate.
2. Correct move generation requires first-rank white double steps, complete
   promotion generation and Horde-specific en-passant treatment. Each has
   produced historical bugs.
3. The main NNUE limitation was color-shared piece identity. A black pawn and a
   white Horde pawn received the same piece-type features even though their
   functions differ sharply.
4. HordeTest separated white pawns into a custom `H` type. The 2023 experiments
   produced direct gains of roughly +149 to +258 Elo over the then-strongest
   Horde network across three time controls.
5. The general solution requested in
   [Fairy-Stockfish #723](https://github.com/fairy-stockfish/Fairy-Stockfish/issues/723)
   is still open. The experimental `hordetest` branch still exists but is two
   commits ahead and 140 commits behind current `master`; it is a historical
   prototype, not a merge base.
6. The public Fairy-Stockfish network list currently names
   `horde-28173ddccabe.nnue`, credited to Belzedar, at +490 Elo versus classical
   evaluation for the built-in `horde` representation. The project's Run 6B
   network is a different, explicitly frozen `hordetest` artifact.
7. The historical `horde.epd` book is reproducible but unchanged since 2016. It
   contains 2,486 lines but only 1,413 unique positions, so deduplication would
   change its implicit weighting.
8. Rule correctness, source provenance and strength must remain separate
   release gates. An old binary that behaves correctly is not a substitute for
   reproducible source.

## 2. Research coverage

The archived Discord corpus contained 49,996 messages spanning 2020-11-20 to
2026-07-13. The full-text index contained exactly the same 49,996 message IDs.
Searching by a case-insensitive `horde` substring produced 247 messages from 34
authors across 12 channels; the exact FTS token produced 230 and the stemmed
query `hord*` produced 247.

The largest Horde discussion groups were:

| Archived channel | Matching messages |
|---|---:|
| `nnue-training` | 80 |
| `nnue-general` | 39 |
| `test-results` | 39 |
| `general` | 35 |
| `help` | 22 |
| `analysis` | 15 |
| `development` | 6 |
| `variant-configuration` | 6 |

The search vocabulary included `horde nnue`, `horde net`, `horde perft`,
`horde book`, `horde opening`, `horde elo`, `horde rating`, `horde bug`,
`pawn structure`, `en passant`, `promotion`, `underpromotion`, `double step`,
`extinction`, `king`, `lichess`, `hordetest`, known filenames and network
hashes. GitHub records were deduplicated by issue/comment identity because the
archive retained multiple indexed snapshots.

The Discord archive ends 24 days before this dossier's status date. Current
GitHub state, current Lichess prose and the Fairy-Stockfish NNUE page were
checked directly. Statements about unarchived Discord activity after
2026-07-13 remain outside this record.

### Evidence classes

- **Primary:** pinned source, issue, commit, file, public rule page or immutable
  artifact checksum.
- **Direct archived evidence:** a dated message containing a result, filename,
  configuration or first-person report.
- **Interpretation:** a conclusion supported by several facts but not stated as
  a formal specification.
- **Unresolved:** a question that needs a new experiment or a newer source.

## 3. Rules and executable references

### 3.1 Initial state

The canonical start FEN is:

```text
rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1
```

It contains 36 white pawns, 8 black pawns and Black's normal back rank. White
has no king. The [Lichess Horde page](https://lichess.org/variant/horde) states
the terminal rules directly: Black wins by capturing every Horde piece,
including promoted pieces, and White wins by checkmating the black king.

The pinned executable rule reference is
[`lichess-org/scalachess@d5d47c16`](https://github.com/lichess-org/scalachess/blob/d5d47c16f65a005ca68e19bab702b02f66dd888c/core/src/main/scala/variant/Horde.scala).
It additionally documents and implements low-material and closed-fortress draw
logic. This matters because Horde insufficient material cannot be represented
by a standard-chess material table.

### 3.2 First-rank double step

[Fairy-Stockfish issue #1](https://github.com/fairy-stockfish/Fairy-Stockfish/issues/1)
identified the core requirements early: piece-count termination, no required
white king and pawns on the first rank. In
[issue #209](https://github.com/fairy-stockfish/Fairy-Stockfish/issues/209#issuecomment-731647109),
Fabian Fichter described first-rank double steps as a natural Horde rule.

The pinned Fairy-Stockfish implementation adds rank one to White's double-step
region. The HordeTest configuration expresses the same behavior as
`doubleStepRegionWhite = *1 *2`.

The special double step must not create a black en-passant capture against a
pawn that started on rank one. This exception is stated on the Lichess rule
page. It must not disable ordinary valid en-passant cases.

### 3.3 Promotions

The legal promotion set is queen, rook, bishop and knight. In July 2021 an
archived differential perft test showed that Multi-Variant Stockfish generated
30,273 nodes instead of 33,781 because knight underpromotions were missing.
The omitted leaf moves included moves such as `a2a1n`, `d2e1n` and `d2d1n`.

That defect is significant beyond perft: promotion is strategically dominant
in Horde, and an omitted underpromotion can alter both legality and search
outcomes.

### 3.4 Perft record

The current pinned Lichess corpus supplies three positions:

| Position | d1 | d2 | d3 | d4 |
|---|---:|---:|---:|---:|
| Start | 8 | 128 | 1274 | 23310 |
| Open flank | 30 | 241 | 6633 | 56539 |
| En passant | 13 | 172 | 2205 | 33781 |

The original and `H`-encoded copies are tracked under `docs/horde/fixtures`.

Another archived reference position is:

```text
4k3/7r/8/P7/2p1n2P/3p2P1/1P3P2/PPP1PPP1 w - - 0 1
```

Its archived depth-four expectation is `128809`. It is useful additional
evidence but is not part of the pinned current Lichess perft file.

## 4. NNUE chronology

### 4.1 First Horde network

On 2021-10-08, archived message `895947248578494525` announced the first Horde
NNUE. Adjacent messages thanked Fabian and described a roughly one-megabyte
network with an estimated gain near 200 Elo. Fabian explained that it was
trained from classical evaluation rather than reinforcement learning.

The same discussion identified a structural surprise: the architecture did not
explicitly encode Horde's color asymmetry. It apparently inferred the side from
piece distribution and the absent white king well enough to improve, but this
was not a robust representation.

### 4.2 v2.5 and v2.6

`5dfe2ffb19ac.nnue` v2.5 was tested on 2021-10-09:

- 840 games against MV-SF 120919;
- MV-SF scored 448-387-5;
- direct estimate +25.3 ±23.5 Elo for MV-SF; and
- archived narrative placed the net roughly +100 over classical, around 60
  below the best MV build and much further behind MultiAra.

The direct result and rating-list narrative use different baselines and must
not be collapsed into one number.

`44e236c323d7.nnue` v2.6 followed on 2021-10-17:

- Fairy-Stockfish scored 354-241-5 against MV-SF in 600 games;
- direct estimate approximately +66.2 ±28.2 Elo for Fairy-Stockfish; and
- the archived narrative described approximately +210 over v2.5 and about 200
  behind MultiAra.

The archived attachment was named `Horde_Rating7.pgn` in message
`899252871470649394`. Its old Discord CDN URL no longer resolves as of the
status date. The later reported MD5 for the 931 KB network file was
`b08c6f48255242e4448aca3fa5f21630`.

### 4.3 Compatibility incidents

Fairy-Stockfish 14 failed to load the v2.6-era Horde network for at least one
user. The same file worked with then-current development code; archived advice
later identified 14.0.1 or a recent development build as the supported floor.

The pattern repeated with newer reports: a network may appear corrupt or cause
the engine to stop moving when the actual problem is an old binary or
architecture mismatch. A direct `uci`, `isready`, network-load and shallow
search probe is required before classifying such a report as a damaged net.

### 4.4 Current public built-in Horde network

The current [Fairy-Stockfish NNUE list](https://fairy-stockfish.github.io/nnue/)
names:

- variant: `horde`;
- file: `horde-28173ddccabe.nnue`;
- author: `belzedar_ in discord`; and
- listed gain: +490 Elo over classical evaluation.

The vault records a 2024 Discord copy of 953,243 bytes in message
`1243085701205987400`; its old CDN URL no longer resolves. The maintained
public reference is the Fairy-Stockfish NNUE list above.

This is the public network for the built-in `horde` piece representation. It is
not interchangeable with the Run 6B `hordetest` network merely because both
play the same rules.

## 5. The HordeTest program

### 5.1 Diagnosis

The August 2023 training discussion isolated the core representation problem:
the same NNUE pawn feature served a black pawn on its home ranks and a white
Horde pawn embedded in a 36-pawn mass. The strategic roles and spatial priors
are not color-symmetric.

The experiment introduced a custom white pawn `H`. Initial configuration work
had to recover three pieces of Horde behavior:

- first- and second-rank double steps;
- promotion/pawn classification; and
- en passant between the custom `H` type and the ordinary `P` type.

The custom-pawn en-passant failure was isolated in
[Fairy-Stockfish #680](https://github.com/fairy-stockfish/Fairy-Stockfish/issues/680).
After the fix, archived discussion reported parity through perft depth six.

### 5.2 Direct 2023 results

The first reinforcement-learning cycle was reported near +300 Elo. A second
cycle scored 145-37-1 in 183 games against the then-strongest network, an
estimate of +235.53 ±63.4 Elo.

The clearest three-time-control HordeTest v5 comparison was:

| Time control | W-L-D | Games | Elo estimate |
|---|---:|---:|---:|
| 2s + 0.02s | 1046-220-42 | 1308 | +258.46 ±23.7 |
| 10s + 0.1s | 186-77-7 | 270 | +148.73 ±44.9 |
| 30s + 0.3s | 65-27-0 | 92 | +152.62 ±79.9 |

These are direct archived match results against the named contemporary
baseline. They are not a claim that the same delta holds against today's
public network or on a different source revision.

### 5.3 General support remains open

Belzedar opened
[#723, “Support for color specific letters”](https://github.com/fairy-stockfish/Fairy-Stockfish/issues/723)
on 2023-09-19. The issue body records a roughly +200 Elo improvement from the
custom-white-pawn workaround.

As of 2026-08-06:

- the issue is open, labeled `enhancement` and assigned to milestone `next`;
- it has no comments or assignee;
- its only 2025 update was the milestone assignment; and
- no linked implementation PR or commit is present.

### 5.4 Historical branch state

The [`hordetest` branch](https://github.com/fairy-stockfish/Fairy-Stockfish/tree/hordetest)
still exists at
[`5e19625cb332022f31d357d3da6d3149c3e92e3a`](https://github.com/fairy-stockfish/Fairy-Stockfish/commit/5e19625cb332022f31d357d3da6d3149c3e92e3a),
dated 2023-08-21. Its two commits are:

1. [`e77520bd206e9c76c3e7a411771d2239a343c3d0`](https://github.com/fairy-stockfish/Fairy-Stockfish/commit/e77520bd206e9c76c3e7a411771d2239a343c3d0),
   which adds `hordetest` with a pawn-letter synonym.
2. `5e19625c`, which adds a custom `H` pawn, changes the public `horde` mapping
   to the new representation and retains the old behavior as `hordeold`.

The [current comparison](https://github.com/fairy-stockfish/Fairy-Stockfish/compare/master...hordetest)
is two commits ahead and 140 behind `master`, with a cumulative one-file diff
of +18/-1 in `src/variant.cpp`. Current `master` contains none of
`hordetest`, `hordenew_variant` or `hordeold`.

This confirms the archived description of the branch as an “ugly hack”: it
preserved standard-looking FEN input by manipulating piece letters and replaced
the public variant mapping. It was useful for the experiment but creates GUI,
WinBoard, serialization and maintenance risk. New work should reproduce the
behavior on pinned current source, not branch from it.

### 5.5 Run 6B baseline

The project freezes the following experimental artifact:

- `networks/hordetest_run6b_e37_l06.nnue`;
- 1,088,416 bytes;
- SHA-256
  `b71108587968ac544eb2e62c2333feca880da5aca52866787f1402163444adf7`;
- credited to Belzedar; and
- dedicated under CC0 1.0 Universal.

Run 6B is the canonical project baseline because its bytes and representation
are frozen, not because it has replaced the public Fairy-Stockfish Horde net.
Any strength claim for it must name the exact opponent, source, book, time
control and totals.

Archived 2025 training notes reported that the rerun reached the previous best
strength after six runs and roughly 1.35 billion positions from depth five to
depth seven. The earlier process used eight runs and roughly 1.45 billion
positions from depth five to depth eight. The discussion also reported large
early gains followed by diminishing returns. These are training-process
observations, not release acceptance measurements.

## 6. Books and opening control

The historical book is
[`fairy-stockfish/books/horde.epd`](https://github.com/fairy-stockfish/books/blob/master/horde.epd).
The old `ianfab/books` repository URL now redirects to the organization-owned
repository.

Frozen metadata:

- Git blob: `bd0e510e745f5ade30aaf8a9d8b7b6376db129ef`;
- size: 196,008 bytes;
- SHA-256:
  `93e97b27d5df054b8a649b8be92a0a8b058384dae35bad142f9a610896eb6958`;
- last content commit:
  [`1b0cf1f9473b5412e1631a9327098ac1b38b096b`](https://github.com/fairy-stockfish/books/commit/1b0cf1f9473b5412e1631a9327098ac1b38b096b),
  2016-12-03;
- 2,486 lines and 1,413 unique lines; and
- every position has Black to move at full move three.

The line structure supports the archived statement that it was generated to
depth five. Exact duplicates occur up to six times. They likely preserve path
multiplicity through transpositions; deduplicating them would change selection
weights.

Archived advice recommends MultiPV book generation for asymmetric variants.
Filtering a Horde book around evaluation zero is unsafe because the starting
position can have a large, arbitrary evaluation offset. The relevant public
reference is the
[MultiPV book-generation example](https://github.com/ianfab/bookgen/wiki/Examples#multipv-based-generation).

No book is silently canonical for a new match. Every experiment must record the
book bytes, selection policy and whether duplicates are preserved.

## 7. Known defect and risk register

| Area | Observed failure | Current interpretation |
|---|---|---|
| Promotions | MV-SF omitted knight underpromotions | Confirmed move-generation bug; include full promotion tests |
| En passant | Custom `H` could not capture ordinary `P` en passant | Confirmed in #680 and fixed experimentally; retain cross-type regression tests |
| Network loading | Old FSF builds rejected or stopped with newer Horde nets | Classify binary/network architecture before blaming platform or file corruption |
| Color symmetry | RL plateaued or regressed with shared pawn identity | Architectural limitation motivating HordeTest |
| Classical evaluation | Custom pieces lost Horde-specific HCE behavior and produced extreme scores | Do not use HCE score scale or ordinary adjudication thresholds blindly |
| Protocols | Historical branch was expected to risk WinBoard failure | Test UCI, CECP/WinBoard, GUI FEN and variant discovery separately |
| Branch age | `hordetest` is 140 commits behind | Port the concept; do not merge or release the branch directly |
| Draw logic | Horde low-material and fortress cases are complex | Differential-test against pinned Lichess source |
| Evidence | Cross-list and head-to-head Elo were mixed in discussion | Preserve opponent and method with every numeric claim |

## 8. Historical strength record

The following figures are useful context, not a common rating scale:

- December 2020 round robin, 438 games per engine: MV310720 314.4, FSF
  development classical 154.6, FSF 11.2 15.3 and MV development 0 in the
  reported Ordo output.
- September 2021 MultiAra 0.9.5 versus MV120919: 705-255-6 in 966 games,
  +175.4 ±24.7 direct. Contemporary list narratives described larger gaps
  against other anchors.
- Horde NNUE v2.5 and v2.6 results are given in section 4.
- HordeTest v5's three-control direct comparison is given in section 5.

Archived analysis increasingly favored Black and reported a stronger Black
trend at longer time controls. Some participants described the game as nearly
solved for Black. This is experimental opinion, not a proof of the game-theory
value. A release document must not call Horde a proven Black win without a
formal solution artifact.

## 9. Contributors and ownership map

| Contributor/handle | Evidence-backed role |
|---|---|
| Belzedar / `belzedar_` | Rating lists, tournaments, trainer experiments, HordeTest, #723, 2025 rerun and Run 6B network |
| Fabian Fichter / `ianfab` / archived `ubdip` | Fairy-Stockfish rules, perft analysis, books, NNUE architecture, early networks and prototype branch; the handle-to-name link is strongly supported but partly inferential |
| `iq_qi94` | MultiAra-related testing; ported perft checks and identified MV-SF failures |
| `mtaktikos` | Custom-piece and training collaboration; isolated the `H`/`P` en-passant defect |
| `autocorr` | Trainer and representation review, especially separate pawn identity |
| Bianca / `bianca5` | Strong historical MV-SF baseline and Horde analysis |
| `gemoflife` | Practical online/bot testing through UltimateVariants |
| `lessnessrandomness` | Opening and refutation analysis |
| `dpldgr`, `louisxxiv` | 2024 binary/network compatibility triage |

A different variant also called “Horde Chess” was discussed by `couchtomato`.
Search results from that variant must not be mixed into Lichess Horde evidence
without explicit confirmation.

## 10. Formal project baseline

This repository's documentation baseline is:

- formal engine source:
  [`fairy-stockfish/Fairy-Stockfish@c19b5f6c`](https://github.com/fairy-stockfish/Fairy-Stockfish/commit/c19b5f6c66894fdb0e88d0dd100e3885f744760a);
- rule reference:
  [`lichess-org/scalachess@d5d47c16`](https://github.com/lichess-org/scalachess/commit/d5d47c16f65a005ca68e19bab702b02f66dd888c);
- HordeTest fixture: custom `H` pawn with Betza `fmWfceFifmnD`;
- canonical evaluation: Run 6B by Belzedar, frozen by SHA-256; and
- historical executable `stockfish_x86-64-bmi2_all_26082025.exe`, SHA-256
  `3501bd84eea8a08938df8d1998976aeb496673f9fdcb8bfcf1e9b06e894c3ad3`,
  designated oracle-only.

The historical executable can help classify behavioral regressions. It cannot
establish formal source identity, satisfy release provenance or be repackaged
as the baseline.

## 11. Open questions

1. What general piece-letter-by-color design, if any, will close #723 without a
   FEN/protocol compatibility hack?
2. Does a current formal-source build load Run 6B and reproduce all HordeTest
   perft values on every supported platform and architecture?
3. What current opening set and adjudication profile should be frozen for
   strength testing?
4. How does Run 6B compare directly against the current public built-in Horde
   network after controlling for the incompatible feature encodings?
5. Which exact low-material cases should be mirrored from the pinned Lichess
   suite into this repository's differential tests?
6. Do CECP/WinBoard and the target GUIs round-trip the custom `H` representation
   correctly?
7. Is there newer Discord evidence after 2026-07-13 about #723, a published
   HordeTest network or the unresolved 2026 network-loading report?

Until these questions are answered, claims should stay narrow: the artifacts
and historical results are real and reproducible; a current engine release and
current comparative strength are not established by this dossier alone.

## 12. Selected archived evidence ledger

| Message ID | Date | Channel | Author | Evidence |
|---|---|---|---|---|
| `862754266786955294` | 2021-07-08 | `general` | `iq_qi94` | Two Horde perft failures in MV-SF |
| `862762854808551454` | 2021-07-08 | `general` | `ubdip` | Missing knight underpromotions identified |
| `895947248578494525` | 2021-10-08 | `nnue-general` | Belzedar | First Horde NNUE announcement |
| `895948476393873428` | 2021-10-08 | `nnue-general` | `ubdip` | Classical-eval training, no RL yet |
| `895953873049620520` | 2021-10-08 | `nnue-general` | `ubdip` | Color asymmetry not explicit in architecture |
| `896418558425301053` | 2021-10-09 | `test-results` | Belzedar | v2.5 direct match |
| `899252647364812810` | 2021-10-17 | `test-results` | Belzedar | v2.6 direct match |
| `928640079939895306` | 2022-01-06 | `help` | `mateon1` | FSF 14 network-loading failure |
| `1142377701852848159` | 2023-08-19 | `nnue-training` | `ubdip` | Shared pawn identity diagnosed |
| `1142707611158978653` | 2023-08-20 | `nnue-training` | `mtaktikos` | Custom `H`/`P` en-passant defect |
| `1142752020093214762` | 2023-08-20 | `nnue-training` | `ubdip` | Perft parity reported through depth six |
| `1143099936406241310` | 2023-08-21 | `nnue-training` | Belzedar | +235.53 ±63.4 second-cycle result |
| `1143493537875382413` | 2023-08-22 | `nnue-training` | Belzedar | HordeTest v5 three-TC results |
| `1153674244945874975` | 2023-09-19 | `nnue-training` | Belzedar | Unreleased network about +198 over baseline |
| `1243085701205987400` | 2024-05-23 | `nnue-general` | `louisxxiv` | Public `horde-28173ddccabe.nnue` copy |
| `1410236528277590066` | 2025-08-27 | `nnue-training` | Belzedar | 2025 filter experiment summary |
| `1412086592679444610` | 2025-09-01 | `nnue-training` | Belzedar | Six-run rerun efficiency report |
| `1466372286243147887` | 2026-01 | `nnue-training` | `ubdip` | Built-in color symmetry remains a problem |
| `1479700369574465587` | 2026-03 | archived Discord | reporter | New 931 KB network crash report, unresolved in snapshot |

The message ledger supports provenance and chronology. Normative game behavior
comes from the pinned primary sources and tracked fixtures, not from chat text.
