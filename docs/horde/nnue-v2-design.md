# Horde NNUE V2 design

Status: engineering draft. This document defines an experimental path; it does
not change the production evaluator. `HORDETEST_HP_LEGACY_V1` and the registered
Run 6B network remain the default until a V2 network passes every technical and
strength gate in the testing contract.

## Goals

V2 should model Horde's asymmetric objectives directly while remaining cheap
enough for search. In particular, it should:

- encode fixed Horde and Royal roles instead of pretending that the position
  has two interchangeable kings;
- model the unique Black king and the White pawn mass in different refresh
  domains;
- distinguish useful White-pawn roles without duplicating information already
  present in piece-square features;
- preserve exact full-refresh, incremental, scalar, SIMD, and trainer parity;
- use a unique, self-describing network schema and reject any dimension,
  quantization, or payload mismatch;
- support controlled ablations in which every trained candidate changes one
  architectural idea at a time.

## Non-goals

- V2 does not reinterpret White pawns as a new physical piece. Positions, move
  generation, FEN, Zobrist keys, SEE, and search continue to use `PAWN`.
- V2 does not load Run 6B through a heuristic header or size match. The legacy
  network remains identified by its complete SHA-256 and manifest.
- V2 does not combine threat, pawn-structure, phase, output, and width changes
  in one strength test.
- Training loss alone does not select a production network.

## Why rank-specific White-pawn planes are insufficient

The legacy H/P encoder already uses piece-square inputs. A White pawn on `d3`
and one on `d6` therefore activate different rows. A second feature whose only
predicate is the absolute rank repeats information the model already has.

The useful distinction is contextual. Two White pawns on the same square in
different positions can have different jobs: one can be the exposed frontier,
one the rear reserve, one can be supported by a phalanx, one can be blocked
behind another Horde pawn, and one can be a viable promotion runner. V2 adds
such predicates only when their invalidation set is explicit and bounded.

## Fixed evaluation frame

The network uses one fixed White/Horde frame:

- White always advances toward rank 8.
- Piece families are fixed roles, not side-relative families:
  `HP, HN, HB, HR, HQ, RP, RN, RB, RR, RQ, RK`.
- There is no White-king feature and exactly one `RK` is required.
- Vertical color flipping is forbidden.
- Horizontal reflection is allowed where specified and as data augmentation.
- Positive output always means good for the side to move.

The first prototype transforms the position once, runs one shared dense trunk,
and selects one of two final output rows by side to move. It does not construct
two complete perspectives or two complete dense heads. No sign flip is applied
after selecting the row.

An explicit side-to-move scalar or embedding is a later comparator. Toggling a
256-lane sparse side-to-move row on every ply is not part of the base design.

## Engineering prototype: `V2_BASE_P0`

The first executable V2 network is deliberately small in scope:

- Royal transformer: 256 lanes, R0 only;
- Global transformer: 256 lanes, G0 only;
- no contextual features and no PSQT outputs;
- concatenated 512-lane activation;
- shared dense trunk `512 -> 32 -> 32`;
- two final one-unit rows selected by side to move;
- one phase bucket;
- shared Royal transformer bias across all king buckets;
- deterministic bounded quantized weights or a deterministic micro-fit, never
  an all-zero network.

The 32 Royal buckets are an engineering stress prototype, not a frozen
production choice. They make refresh cost measurable before expensive training
decides whether the king map has enough value.

The current implementation checkpoint provides the G0/R0 index contract, a
fail-closed full-refresh enumerator, and an engineering-only scalar P0 forward.
The enumerator walks physical squares from A1 to H8, emits at most 52 Global
and 51 Royal rows, rejects a White king, requires exactly one Black king, and
enforces the 36/16 side capacities. The scalar path exercises non-zero bounded
deterministic weights, both sparse transforms, the shared dense trunk, the two
STM output rows, and the external rule-50 postprocessor. It has no production
dispatch, file parser, production-layout accumulator, SIMD backend, or trained
weights and therefore cannot replace the production evaluator. A separate
engineering scalar incremental oracle exists for make/undo parity.

## Dual refresh domains

V2 has two independent sparse affine transformers. Their activated outputs are
concatenated before the shared dense trunk.

### R0: Royal-context piece-square

Purpose: model the spatial relation between every non-king piece and the unique
Black king.

R0 has 32 horizontally canonical Black-king buckets and ten fixed non-king
roles. For Black king square `k` and a non-king piece on `s`:

```text
mirror          = file(k) <= FILE_D
orient(x)       = mirror ? horizontal_flip(x) : x
canonical_king  = orient(k)
bucket          = rank(canonical_king) * 4
                  + (file(canonical_king) - FILE_E)
index           = ((bucket * 10 + non_king_role) * 64) + orient(s)
RoyalKey        = (bucket, mirror)
```

The mirror bit is part of the refresh key. For example, kings on `d4` and `e4`
share a canonical square but use opposite board orientations. Every legal Black
king move therefore changes `RoyalKey` under the 32-bucket map.

R0 has `32 * 10 * 64 = 20,480` input rows and at most 51 active rows. With 256
signed 16-bit weights per row, the table alone is exactly 10 MiB. That cost and
the refresh rate must be measured rather than hidden in a combined strength
test.

Refresh policy:

- unchanged `RoyalKey`: apply ordinary remove/add deltas to non-king pieces;
- changed `RoyalKey`: rebuild the Royal accumulator once from the final board;
- castling: refresh Royal once from the final board; Global receives the king
  and rook deltas;
- promotion and promotion capture: change the non-king role in both domains;
- en passant: remove the pawn from its physical captured square;
- null move: change only the selected output row; neither transformer changes.

The engineering reference now implements this policy directly from the
engine's `DirtyPiece` contract. Global always receives remove/add row deltas.
Royal receives the same non-king deltas while `RoyalKey` is unchanged and is
rebuilt exactly once from the target board when the bucket or mirror bit
changes. The source trace is immutable, so undo is a stack restore rather than
an inferred inverse update. Focused synthetic `DirtyPiece` parity covers quiet
moves, ordinary captures, en passant, promotion captures, Black-king moves,
the d/e mirror boundary, castling, null-head selection, 256 randomized
fixed-role transitions, and source restoration.

The separate scalar reference stack now consumes the exact `Dirties` object
filled by real `Position::do_move()`. It validates that each `DirtyPiece`
reconstructs the complete target board before accepting a frame, keeps Run 6B's
production `AccumulatorStack` untouched, restores undo by popping the saved
frame, and mirrors search by not pushing for null moves. The deterministic
integration receipt covers six focused special moves, 192 generated legal
moves, 15 null transitions, every corresponding undo, and 17 Royal refreshes.
Full refresh is compared after every transition.

That trace-heavy scalar stack remains a correctness oracle. It validates the
parameter object, scans the target board, stores a complete board identity and
keeps every dense intermediate. None of those costs represents the intended
search layout. The performance path therefore uses a separate width-templated
frame containing only aligned Royal and Global accumulators plus `RoyalKey`,
and reusable dense scratch. A Royal-key change copies only Global state before
rebuilding Royal; it never copies a Royal accumulator that will immediately be
discarded.

Two deterministic payload classes prevent width timing from changing the
workload:

- `PARITY_FULL_V1` derives every non-zero weight from its semantic block, row
  and lane. Shared coordinates are identical across widths and exercise
  scalar/full, SIMD/scalar and incremental/full parity.
- `PERF_COMMON_V1` exposes a common R64/G128 output subspace. Transformer
  extension lanes remain non-zero and are still updated and activated, but
  their runtime-loaded H0 weights are zero. All four widths must consequently
  return identical dense intermediates and evaluations before their elapsed
  time or engine NPS can be compared.

The lean scalar checkpoint matches the trace oracle layer by layer at
`256+256`, covers six dirty-piece transition forms, and proves identical
`PERF_COMMON_V1` dense results for all four widths on both sides to move. It is
paired with AVX2 row-update and dense kernels using the same frame and payload;
the AVX2 path passes the same layer and transition receipts.

The lean backend also has a production-layout `Position` stack. It allocates
aligned frames once, stores no board copies or dense traces, and reuses the top
frame across null moves. Ordinary children derive `RoyalKey` directly from the
Black king and queue their sparse delta without enumerating the board. The
pending same-key chain is materialized only when evaluation needs it. A
Black-king key transition refreshes Royal while its target position is still
available; Global remains incremental. The direct `Position` extractor
preserves the A1-to-H8 trainer order without first copying or scanning a
64-square array. Make/undo/null receipts compare every materialized frame with
full refresh, include an unevaluated six-ply lazy batch, and require the same
evaluation digest for all four `PERF_COMMON_V1` widths. The stack is not yet
owned by `Thread` or selected by production evaluation dispatch, so standalone
timings alone are not width-selection evidence.

In an 80-game V3 opening-book probe, 876 of 5,303 Black mainline moves were king
moves (16.5%, including 10 castlings). Search-node rates can differ materially,
so instrumented engine measurements are mandatory before freezing the bucket
map.

### G0: Global fixed-role piece-square

Purpose: preserve the complete absolute board independently of the exact Black
king square.

One row is active per physical piece:

```text
index = fixed_role * 64 + square
```

G0 has eleven roles, 704 input rows, and at most 52 active rows. Unlike R0, it
includes the Black king. All physical moves use ordinary remove/add deltas. The
engine index is absolute; horizontal reflection is only a trainer augmentation.

The 256+256 split preserves 512 activated lanes while avoiding the legacy
evaluator's two 512-lane perspectives. Alternative allocations such as 384+128
or 320+192 are later single-variable experiments.

## Contextual feature state and invalidation

`DirtyPiece` is sufficient for G0 and for R0 while `RoyalKey` is unchanged. It
is not sufficient for contextual pawn features: a move can change the role of
a pawn that did not move.

Every accumulator frame that enables contextual blocks must therefore retain:

- `RoyalKey`;
- the categorical code for every enabled per-file summary;
- the frontier/rearmost bitboard for every enabled boundary feature;
- one predicate bitboard per enabled local P2 feature.

For an incremental update, the target position recomputes only the candidate
files or local neighborhoods, diffs the source and target feature sets, and
applies the exact removed and added rows. Undo restores the saved source frame;
it must not infer old contextual roles from the restored `DirtyPiece` list.

A king move or castling can affect blocked/frontier state on several files. The
candidate set is derived from every physically changed square, not merely from
the nominal move source and destination.

## Candidate feature blocks

Each block has its own index range and structural hash. Every bullet below is a
separate experiment unless explicitly stated otherwise.

Two measured constraints shape the first pawn experiments:

- A literal orthodox `PP_3Wide` adaptation activates 317 pawn pairs in Horde
  startpos: 210 Horde-Horde, 100 mixed, and 7 Royal-Royal. The orthodox maximum
  of 128 is invalid and the block is too dense for the first prototype.
- In the 1,500-position V3 book, positions average 33.1 White pawns; 29.9 have
  a same-rank neighbour, 24.8 have diagonal support, and 24.2 are immediately
  blocked. Common per-pawn predicates are therefore dense, not sparse.

The cheap initial path is boundary-oriented: at most one front and one rear
pawn per file, followed by compact per-file summaries.

### S1: objective-state factorizations

Candidate one-hot counts are:

- White total piece count;
- White pawn count;
- Black non-king material count, one role at a time.

White promoted-piece count is exactly `White total - White pawns`; all three
must not be added together. Counts are deterministically recoverable from G0,
so each must independently earn its playing-speed cost.

### S2: per-file Horde shape

The following are separate alternatives or increments, not one initial bundle:

- White pawn count on each file;
- frontmost White-pawn rank, or empty;
- rearmost White-pawn rank, or empty;
- whether the frontmost White pawn is immediately blocked.

Frontmost-rank summaries and a frontier piece-square plane are alternative
parameterizations first. They are combined only after both individual receipts
exist. Blocked-front state is high-risk because any piece move, including
castling, can change it.

A worst-case bundled update across two changed files, with remove and add rows,
already costs up to 16 transformer row operations for four fields. This is why
the fields are introduced independently.

### P1: boundary pawn identities

- Frontier: the most advanced White pawn on each non-empty file.
- Rear guard: the least advanced White pawn on each non-empty file.

Each identity activates a piece-square row, so either plane has at most eight
active rows. Moving or removing a boundary pawn can expose one replacement on
the affected file.

### P2: local White-pawn roles

Separate predicates may cover:

- same-rank phalanx support on an adjacent file;
- diagonal support from a White pawn one rank behind;
- an immediately blocked White pawn with another White pawn directly behind.

Each predicate uses a fixed local neighborhood and is introduced alone. The
blocked-plus-behind predicate has the highest invalidation risk. A narrower
relational pawn block may reuse the existing pawn-pair delta mechanism only
after publishing its active-row distribution and maximum update count.

### P3: promotion runners

Distance to promotion alone is redundant with G0. A useful runner predicate
must combine blockers and enemy pawn control while retaining bounded updates.
A broad passed-pawn predicate can change many White pawns after one Black move,
so it is deferred until cheaper blocks have been exhausted.

### R1 and R2: Royal ring and relations

King-ring occupancy, supported contact, escape-square state, attacker-target
relations, and support graphs belong to the Royal domain. They can be larger
and dirtier than piece-square inputs, so they come after the base topology and
cheap pawn blocks have passed both Elo and NPS gates.

## Side to move and phase

`V2_BASE_P0` uses one phase bucket and two final scalar rows selected by side to
move. The two rows share both transformers and the complete dense trunk.

Later, side-to-move alternatives are isolated comparisons:

- the two final rows;
- one appended scalar or tiny embedding after transformer activation.

Phase is also introduced separately. The first bucketed implementation uses an
exact serialized lookup:

```text
phase = phase_for_white_piece_count[0..36]
```

Coverage is reported per bucket, side to move, result class, and train/holdout
split. A White-piece count feature and White-piece phase heads are alternatives
first; they are not added simultaneously.

## Dense inference and integer contract

The base topology is:

```text
Royal FT 256 --+
               +-- concat 512 -- clipped activation -- 32 -- 32 -- STM row
Global FT 256 -+
```

The engineering scalar reference currently fixes the following P0 receipt:

- Royal and Global FT weights are signed 16-bit values;
- FT biases and accumulators are signed 32-bit values;
- both FT activations compute `clip(value >> 6, 0, 127)` into unsigned bytes;
- dense weights are signed 8-bit values and dense biases/sums are signed
  32-bit values;
- both hidden layers use the same `clip(value >> 6, 0, 127)` activation;
- the selected STM affine output is divided by 16 using truncation toward zero;
- the legacy external rule-50 damping is then applied exactly once, followed
  by the tablebase-safe clamp.

The P0 payload contains 10,865,992 parameter bytes before container metadata.
Of those, 10,485,760 bytes belong to the 32-bucket Royal transformer. This is
an intentionally expensive stress point whose NPS cost must be measured before
the bucket map is retained for training.

An independent Python trainer-side reference regenerates the deterministic P0
payload without loading C++ parameters and compares every emitted accumulator,
activation, affine layer, STM output, and final value against the C++ scalar
receipt. This closes the initial layer-by-layer integer parity gate; it is not
a substitute for parity on a trained, serialized network.

The reference admits only biases whose magnitude is at most `2^30`. Combined
with the fixed active-row capacities and weight types, this analytically keeps
every signed 32-bit affine sum in range. It does not use saturation or depend
on feature update order.

Before a serialized schema is frozen, it must additionally specify every
remaining discrete inference detail:

- signed integer type for every weight, bias, accumulator, product, and sum;
- clip bounds, activation formula, shifts, rounding, and clamp order;
- feature-row and lane order;
- output scale and conversion to engine `Value`;
- Royal bucket/orientation map and shared-bias policy;
- side-to-move row and phase lookup;
- rule-50 postprocessor version.

Saturating accumulation is forbidden. The trainer must execute the exact
integer forward path, ideally through the same C++ reference; fake
quantization alone is not a parity proof. A later optimized accumulator type
must independently prove its bounds and remain bit-identical to the signed
32-bit scalar reference.

Changing width, activation, output scale, accumulator type, or split is a
separate experiment.

## Training data and label contract

`HORDE_BIN_V1` contains the complete physical board, side to move, clocks,
castling and en-passant state, best and played moves, raw search score, result,
and terminal reason. It is sufficient to derive every proposed feature.

The existing orthodox loader cannot be reused unchanged: it assumes at most 32
pieces, requires one king per color, emits two king-relative perspectives, and
its material bucket expression can exceed the orthodox range in 33-36-pawn
Horde positions.

The Horde-native sparse batch ABI is:

```text
royal_offsets, royal_indices,
global_offsets, global_indices,
side_to_move, white_piece_count, rule50_count,
score_stm, result_stm
```

Its decoder has invariant and horizontal-reflection tests before training.

The dataset manifest states:

- whether `score_stm` is static, qsearch, or root-search output;
- teacher source, settings, network, binary, and complete hashes;
- engine `Value` scale and mate-value handling;
- whether rule-50 scaling is already present in the label;
- position, terminal, mate, and score filters;
- result provenance and every sampling/oversampling rule.

Mate-distance values are never regressed as ordinary centipawn targets. They
are either clipped under a documented policy or trained only through the result
term. Fivefold repetition cannot be recovered from a board record alone and is
measured separately.

Horde's asymmetric WDL calibration is measured independently for White-to-move
and Black-to-move samples. The initial result model is a three-class monotone
link whose fitted parameters are frozen across architecture candidates.

Every strength-comparable rung uses the same dataset split, labels, optimizer,
schedule, loss, lambda policy, filters, and at least three seeds. The manifest
records trainer commit, dataset hashes, structural schema hash, seed,
validation metrics, engine NPS, and refresh rates.

## Rule-50 contract

The first V2 evaluator preserves the current external rule-50 postprocessor
exactly once:

```text
r  = min(rule50_count, 100)
v1 = trunc_toward_zero(v0 * (100 - r) / 100)
v  = clamp(v1, tablebase_value_bounds)
```

Python must emulate truncation toward zero; integer `//` is wrong for negative
scores. The trainer objective must state whether it predicts `v0` or the
postprocessed `v`, so damping is never learned and applied twice. A learned
rule-50 input, changed postprocessor, or no damping is a later isolated test.

## Network container and dispatch

V2 uses a new little-endian container and feature-transform identity. It must
not resemble Run 6B. It contains explicit, length-delimited sections rather
than serialized compiler structs.

The container records at least:

- schema name and version;
- authoritative structural schema SHA-256;
- ordered feature blocks, ranges, roles, lane order, and hashes;
- Royal bucket map, mirror semantics, and bias policy;
- transformer, dense-layer, phase, and side-to-move dimensions;
- integer types, quantization, clipping, shifts, rounding, and output scale;
- phase lookup and rule-50 postprocessor version;
- section offsets, section lengths, and payload SHA-256;
- whole-file SHA-256 registered by the engine;
- training and data manifest identities.

The engine dispatch order is explicit:

1. a complete SHA-256 match selects registered Run 6B and
   `HORDETEST_HP_LEGACY_V1`;
2. a V2 signature selects the V2 parser, which validates every structural and
   integrity field;
3. every other file is rejected.

Network replacement clears the transposition table, accumulator stacks,
contextual feature frames, and evaluation caches. The fresh H/P control uses a
separate experimental identity and cannot be mistaken for production Run 6B.

## Correctness and performance gates

The first implementation is a scalar full-refresh reference. Exactly three
parity gates are required before strength testing:

1. trainer integer forward equals C++ full refresh layer by layer;
2. scalar C++ equals every supported SIMD backend;
3. incremental equals full refresh after make/undo and search transitions.

Coverage includes ordinary moves, captures, promotion, promotion capture, en
passant, Black castling, every Black-king move, null moves, network replacement,
and every contextual block. A mismatch reports the FEN, move, domain, source
and target contextual state, removed and added indices, and both accumulator
values before aborting.

Instrumented builds sample shadow full refreshes and record:

- full-refresh evaluations per second;
- scalar, SIMD, and incremental engine NPS;
- Royal refreshes per materialized accumulator and per evaluation, separated
  by Black-king move, castling, and other causes;
- average and maximum removed/added rows per block;
- Royal-key reuse distance, unique rows/cache lines touched and memory
  footprint; a cache hit rate is reported only after an actual cache exists;
- benchmark and search NPS against the production evaluator.

`V2_BASE_P0` is an expressivity ceiling, not a presumed production size. Its
10,865,992-byte payload must not silently become the budget for later feature
blocks. Before adding contextual pawn or relational features, the optimized
backend benchmarks these isolated width points with the same integer fixture:

| Royal + Global lanes | Parameter bytes | Accumulator bytes | H0 MACs | Start refresh lane ops | King-move lane ops |
| --- | ---: | ---: | ---: | ---: | ---: |
| `256+256` | 10,865,992 | 2,048 | 16,384 | 26,368 | 13,568 |
| `128+256` | 5,618,504 | 1,536 | 12,288 | 19,840 | 7,040 |
| `128+128` | 5,433,672 | 1,024 | 8,192 | 13,184 | 6,784 |
| `64+192` | 2,902,344 | 1,024 | 8,192 | 13,248 | 3,648 |

The exact serialized payloads are 10.363, 5.358, 5.182 and 2.768 MiB. The
`128+128` versus `64+192` comparison is the clean allocation control: dense
work, accumulator bytes and quiet-move lane operations are identical; maximum
full-refresh work is nearly identical; only the Royal/Global allocation and
king-refresh cost differ materially.

The topology control keeps the same split `256+256` tables, payload, frame and
propagation. On a Royal-key change it compares the intended split policy
(Royal refresh plus Global delta) with a forced-full policy (refresh both
domains). A literal combined 512-lane table is not used because it would also
change row layout, zero storage and cache footprint.

The minimum timed matrix isolates frame copy; Royal refresh at 0, 1 and 51
rows; Global refresh at 0, 1 and 52 rows; one quiet piece transition; one Black
king transition; dense propagation; and composed full evaluation. Maximum-row,
quiet and king cases run with both a repeated hot schedule and a deterministic
streaming schedule covering all 32 Royal keys and every table region. Page
faults, validation, allocation, logging, move generation and explicit cache
flushes stay outside timed regions. Timings use the exact production kernels,
not a benchmark-only rewrite.

Engine NPS uses `PERF_COMMON_V1` and is accepted only after all four widths
produce identical per-position values, best moves, node counts and trace
hashes. Sanitizer and telemetry builds are never timed. Paired run order is
randomized with a frozen seed; raw samples, median, MAD and paired 95%
confidence intervals are retained. The headline NPS-ratio interval must have a
half-width no larger than 0.5%, and a width advances to training only if its
NPS lower bound is at least 95% of the fastest surviving width.

After training, fixed-node Elo and uninstrumented NPS remain separate axes.
Practical equivalence margins are 2 Elo and 1% NPS. A larger/slower point must
show a positive 95% lower confidence bound in fixed-node Elo against the
nearest faster survivor before equal-time testing. No width or feature block
advances because of validation loss alone.

## Orthogonal experiment ladder

### Engineering gates

1. Horde decoder invariants and horizontal-reflection round trips.
2. Pinned Run 6B replay through the new data path, without changing production
   dispatch.
3. `V2_BASE_P0`: R0 256 + G0 256, one bucket, shared trunk, two final STM rows,
   deterministic micro-fit.
4. Integer/full-refresh, scalar/SIMD, and incremental/full-refresh parity;
   real `Position` make/undo/null parity; split versus forced-full refresh
   performance on identical `256+256` tables; Royal-key refresh telemetry.

### Training control

5. Fresh legacy H/P, exact legacy architecture, three seeds, separate
   experimental schema identity.

### Architecture ablations

6. A0: G0-only 512, one bucket, same dense trunk.
7. A1: G0 256 + G0 256 implementation control, identical feature content and
   total parameter budget.
8. A2: replace the first G0 half with R0 256.
9. If R0 is promising, test one Royal bucket map at a time, then the fixed
   `256+256`, `128+256`, `128+128`, and `64+192` width points one at a time.
10. Compare two final STM rows with one dense STM scalar or tiny embedding.
11. Compare no count, one count feature, and White-count phase buckets as
    alternatives.
12. Test each remaining scalar count independently.
13. Test each per-file shape representation independently; frontier plane and
    front-rank summary are alternatives first.
14. Test each P2 predicate independently.
15. Only then test promotion-runner, king-ring, and relational threat blocks.

A network that differs in more than its named rung is not a valid ablation. If
two individually losing blocks are believed to interact, their combination is
tested only after both individual receipts exist.

## Open questions before a frozen V2 schema

- Does R0 beat the equal-parameter A1 control after its refresh cost?
- Is the 32-bucket Royal map worth 10 MiB, or should it be coarser?
- Can a Royal refresh cache amortize the measured search-node king-move rate?
- Do two final STM rows beat a post-transform STM scalar at equal NPS?
- Which count or phase representation has adequate late-extinction coverage?
- Which boundary pawn representation adds information beyond G0?
- Which exact integer scales and bounds give safe, efficient inference?
- Which score/result calibration best fits each side to move?
- How much near-extinction and near-fortress oversampling helps without
  distorting ordinary positions?

These questions are resolved through isolated technical and strength receipts,
not by changing the production Run 6B path.
