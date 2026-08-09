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
fail-closed full-refresh enumerator, scalar and AVX2 integer forwards, an
authenticated network container, and a production-layout incremental stack.
The enumerator walks physical squares from A1 to H8, emits at most 52 Global
and 51 Royal rows, rejects a White king, requires exactly one Black king, and
enforces the 36/16 side capacities. The integer path exercises non-zero bounded
weights, both sparse transforms, the shared dense trunk, the two STM output
rows, and the external rule-50 postprocessor. It still has no production UCI
dispatch or strength-qualified trained weights and therefore cannot replace
the production evaluator.

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

The separate scalar reference stack consumes the exact `Dirties` object filled
by real `Position::do_move()`. It validates that each `DirtyPiece` reconstructs
the complete target board before accepting a frame, keeps Run 6B's production
`AccumulatorStack` untouched, restores undo by popping the saved frame, and
mirrors search by not pushing for null moves. The deterministic integration
receipt covers ten focused special moves, including four overlapping Chess960
castling layouts, 192 generated legal moves, 15 null transitions, every
corresponding undo, and 17 Royal refreshes. Full refresh is compared after
every transition.

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

The lean backend also has a generic production-layout `Position` stack shared
with authenticated containers. It allocates aligned frames once, stores no
dense traces, and reuses the top frame across null moves. `DirtyPiece` is first
normalized so inactive, potentially indeterminate piece fields are never read;
all removals precede additions, including promotion, en passant and overlapping
Chess960 castling squares. Ordinary children derive the first-domain key in
O(1) and queue their sparse delta without enumerating the board. The pending
same-key chain is materialized only when evaluation needs it. A Black-king key
transition extracts the final sparse rows once and refreshes Royal while Global
remains incremental.

The stack has two compile-time policies over the same transition code. The
production policy performs no 64-square scan on push, pop or evaluate. The
validating policy retains one exact 64-byte board shadow and rejects any source,
dirty-list or target mismatch transactionally. Correctness tests exercise both
policies: the production policy covers special moves, lazy batches, null moves
and generated legal sequences; the validating policy covers poisoned inactive
fields and malformed or contradictory transitions. Make/undo/null receipts
compare every materialized frame with full refresh and require the same integer
layers under scalar and AVX2. The stack is not yet selected by production
evaluation dispatch, so its engineering timings make no playing-strength claim.

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

The first five contextual candidates are ordered by information added per
sparse row. Every candidate is additive alongside the immutable G0 stream and
is trained and timed alone before any combination:

| Order | Candidate | Rows | Maximum active rows | FT payload at G192 | FT payload at G128 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | frontier White-pawn square | 56 | 8 | 21,504 bytes | 14,336 bytes |
| 2 | rearmost White-pawn square | 56 | 8 | 21,504 bytes | 14,336 bytes |
| 3 | frontmost pawn blocked by any occupancy, per file | 8 | 8 | 3,072 bytes | 2,048 bytes |
| 4 | frontier pawn diagonally supported, per file | 8 | 8 | 3,072 bytes | 2,048 bytes |
| 5 | White-pawn count by file | 56 | 8 | 21,504 bytes | 14,336 bytes |

The 56-row square planes exclude rank 8, where a White pawn must promote, and
retain the exact physical square rather than adding a
redundant rank-only identity. The count block uses seven non-empty count states
per file (`1..7`); an empty file activates no row. Payload figures count signed
16-bit transformer weights only and are therefore direct additions to the
chosen Global width.

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
legacy_piece_offsets, legacy_white_indices, legacy_black_indices,
royal_offsets, royal_indices,
global_offsets, global_indices,
physical_piece_count, white_piece_count, side_to_move, rule50_count,
score_stm, result_stm
```

Its decoder has invariant and horizontal-reflection tests before training.
It retains the physical board, castling rights, en-passant square, and source
payload identity instead of discarding fields that are invisible to the first
network. `white_piece_count` is an explicit decoded value. Receipts bind the
whole input file, fixed header, manifest, record payload, book, producer,
teacher network, sparse rows, physical states, evaluator inputs, and mate-mask
eligibility. A sample identity is `(payload_sha256, local_record_index)`.

Legacy, Global and Royal offsets are independent even while base G0 emits one
Global row per physical piece. `physical_piece_count` and
`white_piece_count` are derived from the retained board, never from the
complete Global stream. Decoder receipts separately count and hash physical
G0, every contextual block, complete Global, Royal and legacy streams for both
the source position and its horizontal reflection.

Opening roots are assigned by a horizontal-reflection canonical key before
generation. After generation, both physical-state keys and complete
legacy-evaluator-input keys must have zero train/validation overlap. The split
and record-level audit are independent gates; neither substitutes for the
other. Until the wire format gains a game identifier, weighting and validation
metrics are per record and make no game-clustered statistical claim.

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
link whose fitted parameters are frozen across architecture candidates. The
reference trainer maps both network and non-mate teacher scores through that
same frozen link, then combines their half-Brier distance with the half-Brier
distance to the one-hot game result. Mate scores contribute only through the
result term. The calibration artifact must identify the exact training split;
its SHA-256 is part of every comparable training recipe and checkpoint.

Every strength-comparable rung uses the same dataset split, labels, optimizer,
schedule, loss, lambda policy, filters, and at least three seeds. The manifest
records trainer commit, dataset hashes, structural schema hash, seed,
validation metrics, engine NPS, and refresh rates.

Initialization uses independent SHA-256-derived streams keyed by semantic
parameter name. Identically shaped shared trunks and output heads therefore
start byte-identically across width candidates even when their sparse
transformers consume different parameter counts.

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

The control trainer predicts `v0`, applies this integer forward exactly once
inside the loss, and uses a straight-through estimator only for the gradient
through truncation. Its checkpoint freezes model, optimizer, scheduler,
device-specific RNG, deterministic sample-order hash chain, partial-epoch
cursor and metrics, examples consumed, source/data/recipe identities, and the
exact runtime. Uninterrupted and mid-epoch-resumed runs must be semantically
identical across every field and tensor; container-level `torch.save` bytes are
not accepted as a substitute for this comparison.

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

The first executable contract is `HORDE_V2_INTEGER_NETWORK_V1`. It uses the
distinct eight-byte magic `HSV2INT\0`, a fixed 2,048-byte little-endian header,
and ten authenticated parameter sections. Schema `0x00010001` registers
`V2_BASE_P0_64X192`; schema `0x00010002` registers
`V2_C1_ABS_NONKING_64X192`; and schema `0x00010003` registers
`V2_C1_ROYAL_RANK8_64X192`. The header carries both the container structural
hash and the training architecture structural hash, plus the exact checkpoint,
training receipt, train split, validation split, WDL calibration, and clean
source-commit identities. A schema is never inferred from the file size or a
shared hash.

The frozen integer conversion uses round-to-nearest with ties to even,
feature-transform scale 8,128, dense-weight scale 64, signed `int16` feature
weights, signed `int8` dense weights, and signed `int32` biases and
accumulators. Both activation stages compute
`clip(max(affine, 0) >> 6, 0, 127)`. The selected side-to-move output is divided
by 16 with truncation toward zero before the versioned rule-50 postprocessor.
Every bias is bounded to magnitude `2^30`; the registered dimensions keep all
legal full-refresh and dense sums inside signed 32-bit range.

`tools/horde_v2_export.py` accepts only a clean, receipt-matched training
checkpoint and writes the container exclusively. `tools/horde_v2_container.py`
owns the canonical descriptor and an independent fail-closed reader.
`src/nnue/horde_v2_container.cpp` owns the C++ reader and the container-specific
full-refresh and incremental adapters. All three registered schemas use the
shared lazy stack and dense propagation path. `tests/horde_v2_container_parity.py`
independently reconstructs sparse rows and every integer layer, compares Python
with scalar and AVX2 C++, invokes scalar and AVX2 real-`Position` stack oracles,
and verifies adversarial header, provenance, directory, payload, truncation,
and parameter-range failures on Linux and Windows. This path has no UCI
dispatch and cannot replace Run 6B; it is an engineering gate for trained V2
checkpoints only.

The accepted implementation and the two exported canary checkpoints are frozen
in `docs/horde/nnue-v2-integer-container-receipt.json` at source commit
`f38a1a7c`. The receipt records byte identities, exact training provenance,
layer traces, scalar/AVX2/Python parity, malformed-container cases, and the green
Linux/Windows CI runs. It carries no playing-strength or production-dispatch
claim.

The subsequent lazy incremental stack is frozen separately in
`docs/horde/nnue-v2-incremental-container-receipt.json` at source commit
`a1b318ae`. It verifies real-`Position` make/undo/null transitions, ordinary
delta materialization, Royal-only refreshes after king-bucket changes, both
schemas registered at that source commit, scalar/AVX2 parity, and ASan/UBSan.
The older full-refresh receipt remains immutable and therefore retains its
historical `incremental_eligible: false` field. Neither receipt enables
production UCI dispatch or makes a playing-strength claim.

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

The compile-time-only `HORDE_V2_PERF` path connects one registered width and
its lazy stack to the real search worker. It is absent from normal builds and
uses no UCI switch that could leak into a release. The dispatch-only V2 width
option in the `Horde correctness` workflow builds all four AVX2 binaries
sequentially on one runner. `tests/horde_v2_engine_widths.py` randomizes their
paired order, rejects any root-evaluation, per-position root score, node-count
or best-move mismatch, retains every raw NPS sample and bootstraps paired ratio
intervals. The root-evaluation pass remains outside the timed search. These
synthetic results select which widths are cheap enough to train; they are not
playing-strength evidence.

The accepted AVX2 search receipt is frozen in
`docs/horde/nnue-v2-width-receipt.json` at source commit `2468720a`. All four
builds searched exactly 2,190,067 nodes and produced the same ten root
evaluations, per-position root scores and bounds, node counts, and best moves.

| Width | Median NPS | Ratio to fastest, paired 95% CI | Precision | Training-speed gate |
| --- | ---: | ---: | --- | --- |
| `64+192` | 1,180,947 | 1.0000 | pass | pass |
| `128+128` | 1,168,659 | 0.9876--0.9975 | pass | pass |
| `128+256` | 1,051,148 | 0.8856--0.8961 | fail | fail |
| `256+256` | 939,137 | 0.7934--0.8007 | pass | fail |

Only `64+192` and `128+128` advance to the first training comparison. This
receipt does not choose a production width: the two survivors still require
controlled training and fixed-node strength evidence.

The deterministic reference trainer accepts both survivors directly from
`HORDE_BIN_V1`. It uses the same authenticated split, side-specific WDL link,
half-Brier objective, optimizer, schedule, rule-50 graph and semantic
initialization as the fresh legacy control. Width-specific checkpoints carry a
canonical structural hash and reject cross-width resume. This closes the
real-data gradient and restart path; it does not authorize a width strength
test before the C0 split-equivalence and C1 absolute-content controls establish
that the Royal domain itself is useful.

The first gradient-plumbing gate is frozen in
`docs/horde/nnue-v2-microfit-receipt.json`. A 32-position engineering fixture
covers both sides to move and all eight legacy material buckets. On one CPU
thread, two independent runs of each model produced identical loss and final
state hashes. The fresh legacy H/P topology exercised its transformer, PSQT,
all layer stacks, and dense path; both surviving V2 widths exercised the Royal
transformer, Global transformer, shared trunk, and both side-to-move heads.
All three reduced the frozen lambda-0.6 objective in 24 steps. The receipt uses
synthetic labels and therefore proves only deterministic data and gradient
plumbing; its losses do not rank architectures and make no strength claim.

The C0 split-equivalence gate is frozen in
`docs/horde/nnue-v2-c0-receipt.json`. It initializes `G0_SINGLE_256`, clones
exact row and lane slices into `G0_SPLIT_64_192` and
`G0_SPLIT_128_128`, and exercises the same 32-position fixture through four
RAdam steps. Both splits reproduce the single model's floating-point forward
values, reassembled gradients, parameters, and optimizer state exactly. A
canonical engineering quantizer also produces identical integer accumulator,
hidden-layer, and output traces after reassembly. PyTorch floating-point hashes
are explicitly runtime-scoped because named initialization can differ by an
ULP across operating systems; CI instead requires the quantized payload and
complete integer trace to match on Windows and Linux. The framed payload is
marked `production_schema: false`: it is an equality oracle, not a V2 network
container, trained candidate, or strength result. C0 therefore closes only the
mechanical split control; C1 remains the first test of whether Royal-relative
content is useful.

The C1 trainer control is `v2-c1-abs64x192`, schema
`V2_C1_ABS_NONKING_64X192`. Against `v2-64x192`, it preserves G0 192, the
256-lane dense input, trunk, two STM heads, initialization policy, labels,
optimizer, and schedule. Its first 64 lanes instead consume the ten absolute
non-king G0 role planes: 640 rows, no Black-king bucket, and no reflection.
This makes the first-domain representation the only semantic variable. It is
not parameter matched: the absolute control serializes 362,824 parameter bytes
and the Royal candidate 2,902,344. C1 must therefore judge any fixed-node gain
against the Royal table's measured refresh, cache, and equal-time cost.

The compact Royal control is `v2-c1-rank8-64x192`, schema
`V2_C1_ROYAL_RANK8_64X192`. It keeps the same ten non-king roles, 64 first-domain
lanes, G0 192, trunk, heads, initialization, labels, optimizer, and schedule.
Its Royal key contains only the Black king rank and horizontal reflection: eight
buckets, 5,120 rows, and 936,264 serialized parameter bytes. A king move within
the same rank and mirrored half keeps the key and uses ordinary deltas; a rank
or mirror-key transition refreshes only the first domain. This is a topology
control, not a contextual pawn feature.

The decisive C1 comparison is three-way: absolute, Rank-8 Royal, and the full
32-bucket Royal map. It uses disjoint 250,000-position training and
250,000-position validation sets, three paired seeds, and exactly 2,000,000
training-example exposures per model. Dataset, labels, calibration, optimizer,
widths, initialization policy, and export quantization remain identical. A tie
selects the cheaper model. The 32-bucket map is retained only if it beats Rank-8
after quantized training, establishes a fixed-node 95% lower bound above
`+2 Elo`, and then establishes a positive equal-time 95% lower bound. No larger
map advances on floating-point validation loss alone.

The exact Rank-8 implementation and its cross-platform engineering evidence
are frozen in `docs/horde/nnue-v2-rank8-control-receipt.json`. That receipt does
not claim a speed result, completed training gate, playing strength, or
production dispatch.

The production comparison is registered in
`schemas/horde-v2-c1-campaign-v1.json`. The planner
`tools/horde_v2_c1_campaign.py` accepts no CLI override for record counts,
architectures, seeds, epochs, optimizer, device, or selection margins. Its
`plan` command authenticates both `HORDE_BIN_V1` roles, the reflection-safe V2
book split, zero physical and legacy-input cross-role overlap, the exact WDL
calibration, Run 6B, the Rank-8 receipt, a clean trainer commit, all 32 Royal
buckets, STM-by-Horde-material slices, side-specific WDL support, and a below
0.1% unseen Royal-row activation rate before writing nine explicit
train/export commands. Seed one is designated for any later playing gate
before training metrics exist. The plan remains a preflight artifact and makes
no training or strength claim.

After all nine runs and integer exports exist, the `verify` command checks the
complete training receipts, checkpoint and metrics hashes, quantized container
provenance, equal environments, and sample-order chains independently rebuilt
from the frozen dataset, seed and shuffle schedule. Successful verification
authenticates the evidence but deliberately leaves architecture selection and
playing-gate eligibility false. A later training-screen artifact plus
fixed-node and equal-time evidence remain mandatory.

That next artifact is registered as
`schemas/horde-v2-c1-quantized-screen-v1.json`. It evaluates every authenticated
container over the complete 250,000-position validation role using the exact
integer forward path. The receipt reports the frozen half-Brier objective
overall, by side to move and by all six White-piece-count bins. It also rejects
weight sections with less than 1% non-zero parameters or more than 5% values at
their storage-type boundaries.

Each larger topology is compared with its nearest cheaper control and, for
Royal-32, also with the absolute control. Advancement requires all three paired
seeds to improve after quantization, a paired 95% lower bound above zero, an
improvement for both sides to move in every seed, the same ordering in each of
the last two floating-point epochs, no float-to-integer ranking reversal, and
parameter health on both models. The paired interval uses the pre-registered
Student critical value `t(0.975, 2) = 4.3026527297`. A tie therefore cannot
promote the larger model.

The screen may nominate at most one predesignated-seed fixed-node pairing. It
does not select an architecture or provide strength evidence. Royal-32 must
clear both Rank-8 and absolute before it can be nominated against Rank-8;
otherwise a passing Rank-8 is compared with absolute. If Rank-8 fails absolute,
Royal-32 can proceed only after clearing both controls and is then compared
directly with absolute.

```console
python tools/horde_v2_c1_campaign.py plan TRAIN.bin VALIDATION.bin \
  --book-split-receipt BOOK-SPLIT.json \
  --wdl-calibration WDL-CALIBRATION.json --output C1-PLAN.json

python tools/horde_v2_c1_campaign.py verify C1-PLAN.json RUNS-DIRECTORY \
  --output C1-VERIFICATION.json

python tools/horde_v2_c1_screen.py C1-PLAN.json RUNS-DIRECTORY \
  --validation VALIDATION.bin --wdl-calibration WDL-CALIBRATION.json \
  --output C1-QUANTIZED-SCREEN.json
```

The existing real-data C1 plumbing canary is frozen in
`docs/horde/nnue-v2-c1-real-canary-receipt.json`. Both architectures completed
two byte-identical three-epoch CPU runs on the same authenticated 4,096/1,024
split, and every first-domain, Global, dense, and output gradient group was
non-zero. Their final validation losses differ by only `5.10e-6` in favour of
the absolute control. One seed on this integration sample cannot rank either
architecture, so the receipt explicitly forbids architecture selection and
makes no strength claim.

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

5. Fresh legacy H/P, exact serialized legacy architecture, three seeds,
   separate experimental schema identity. The common reference recipe excludes
   the historical training-only first-layer factorizer; adding that factorizer
   is a separate training-method experiment, not part of an architecture rung.

### Architecture ablations

6. C0 is an engineering-equality receipt, not an Elo test: compare one
   `G0_SINGLE_256` table with `G0_SPLIT_64_192` and optionally
   `G0_SPLIT_128_128`. Initialize split tensors from exact row/lane slices of
   the single table and require identical forward values, gradients, optimizer
   state and exported integer evaluations after reassembly.
7. C1 isolates Royal content at the fastest surviving split: compare
   `ABS_NONKING_64 + G0_192`, `ROYAL_RANK8_64 + G0_192`, and
   `ROYAL_32_64 + G0_192`. All three use the same ten non-king roles and at most
   51 active rows. This is a content/topology control, not a parameter-matched
   claim; the three first domains intentionally own different row counts.
8. C2 runs only if a Royal map passes C1: compare the accepted map at
   `Royal_128 + G0_128` with `Royal_64 + G0_192`. These are the two width points
   that survived the real AVX2 gate and have equal dense work, accumulator bytes
   and quiet-move lane work.
9. Freeze one no-context map and width before testing side-to-move, phase,
   count, pawn-boundary, or relational features. Prefer the cheaper topology
   unless a larger point establishes the required fixed-node lower bound and
   then wins equal-time.
10. Compare two final STM rows with one dense STM scalar or tiny embedding.
11. Compare no count, one count feature, and White-count phase buckets as
    alternatives.
12. Test each remaining scalar count independently.
13. Test one frontier-square pawn representation by itself. A front-rank
    summary is a separate alternative and is not combined without individual
    receipts.
14. Test each P2 predicate independently.
15. Only then test promotion-runner, king-ring, and relational threat blocks.

A network that differs in more than its named rung is not a valid ablation. If
two individually losing blocks are believed to interact, their combination is
tested only after both individual receipts exist.

## Open questions before a frozen V2 schema

- Does either Royal map beat the absolute non-king content control after its
  refresh cost?
- Is rank-only Royal context sufficient, or does the 32-bucket map justify its
  larger table and higher refresh/cache cost?
- Can a Royal refresh cache amortize the measured search-node king-move rate?
- Do two final STM rows beat a post-transform STM scalar at equal NPS?
- Which count or phase representation has adequate late-extinction coverage?
- Does a frontier-square pawn representation add information beyond G0 before
  any richer rank, file, or support encoding is introduced?
- Which exact integer scales and bounds give safe, efficient inference?
- Which score/result calibration best fits each side to move?
- How much near-extinction and near-fortress oversampling helps without
  distorting ordinary positions?

These questions are resolved through isolated technical and strength receipts,
not by changing the production Run 6B path.
