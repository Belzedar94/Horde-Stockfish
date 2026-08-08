# Horde training-data contract V1

`HORDE_BIN_V1` is the canonical Horde-Stockfish self-play format. It stores the physical game position, not an evaluator-specific feature vector. White pawns therefore remain ordinary `P` pieces in the dataset. A trainer targeting `HORDETEST_HP_LEGACY_V1` maps those white pawns to the legacy `H` feature plane when it loads a record; future role-aware architectures may consume the same record without changing the game representation.

## File structure

Each file starts with a 2,048-byte header and continues with fixed 48-byte records. The first eight bytes are ASCII `HORDEBIN`. Little-endian `uint16` values at offsets 8 and 10 contain format version `1` and header size `2048`. A little-endian `uint32` at offset 12 gives the length of the compact UTF-8 JSON manifest beginning at offset 16. Every unused header byte is zero.

The manifest binds the file to the full source commit, dirty-state marker, Run 6B network SHA-256, opening-book SHA-256, producer executable SHA-256, exact generation settings, record count, and SHA-256 of the record payload. It also binds the external [`HORDE_LABEL_CONTRACT_V1`](../../schemas/horde-label-contract-v1.json) by schema name and complete SHA-256. Seeds are decimal strings so consumers do not lose 64-bit precision.

The complete byte layout, piece codes, flags, result perspective, and terminal-reason values are frozen in [`schemas/horde-bin-v1.schema.json`](../../schemas/horde-bin-v1.schema.json). The SHA-256 of that schema is part of the generator capability handshake and every file manifest.

## Position and label semantics

The board is encoded as 64 four-bit physical piece codes. It supports the kingless White Horde, the single Black king, all promoted White pieces, up to 36 White pieces, and up to 52 pieces in total. Only Black castling rights are representable. The en-passant square, rule-50 clock, game ply, side to move, best move, and played move are sufficient for exact physical FEN reconstruction and audit; repetition history remains a game-level property.

Scores and results are relative to the side to move in the stored position. `score` is the raw internal `Value` produced by a completed exact Horde-Stockfish root search, before UCI centipawn conversion. The producer rejects bound scores. Search evaluation and terminal logic have already incorporated rule 50 into this teacher value, so a trainer must never apply the prediction postprocessor to the stored score. `best_move` is the label selected by the principal search. `played_move` records the move used to advance self-play and may differ when deterministic exploration is scheduled.

Games are labeled only after `Position::outcome()` returns an exact terminal result. The side-to-move result is derived by comparing each stored position with the terminal winner. Checkmate and Horde extinction require a decisive result; stalemate, Horde fortress, the automatic fifty-move rule, and fivefold repetition require a draw. The encoder rejects every contradictory result/reason pair. The separate per-color insufficient-winning-material predicate is never treated as an automatic draw. A game that reaches the generator safety ply limit without a terminal result is discarded, not mislabeled. No class weighting or result resampling occurs at generation time.

## Generation boundary

The normal `horde-stockfish` executable does not expose data-generation commands. OpenBench uses the separately built `horde-stockfish-data-generator` artifact. The generator rejects an unregistered network, a mismatched network or book hash, a malformed producer hash, an existing output path, and unsupported or contradictory Horde openings. A failed or interrupted run removes its partial output.

OpenBench publication protocol 41 remains the outer transport and authentication envelope. Its archive receipt binds the compressed file to the workload, producer artifact, network, book, worker, and upload. The embedded `HORDE_BIN_V1` manifest independently binds the uncompressed payload and generation parameters.

## G0 audit

Before a canary may be expanded, the decoder must validate the header, schema hash, payload hash, record framing, physical piece constraints, move encodings and origins, and terminal reason range. Exact move legality is enforced before encoding by the producer. The coverage report must include side-to-move balance, White piece-count buckets, promoted-piece presence, en-passant states and moves, Black castling rights and moves, best-versus-played divergence, score distribution, game results, and every terminal reason observed. Capture, check, and promotion samples are measured rather than filtered out.

## Trainer reference decoder

[`tools/horde_training_decoder.py`](../../tools/horde_training_decoder.py) is
the fail-closed reference boundary between physical `HORDE_BIN_V1` records and
evaluator-specific sparse rows. It verifies the manifest, exact file framing
and payload SHA-256 before exposing a read-only memory map. Variable-length
batches use CSR-style offsets and retain the search labels without copying the
whole dataset into memory.

Every decoded record exposes four independent sparse views:

- legacy White-perspective and Black-perspective rows in the 896-dimensional
  `HORDETEST_HP_LEGACY_V1` table;
- absolute fixed-role rows in the 704-dimensional V2 Global table;
- Black-king-relative fixed-role rows in the 20,480-dimensional V2 Royal
  table, excluding the Black king itself.

The legacy implementation and the C++ conformance oracle share
[`src/nnue/horde_legacy_features.h`](../../src/nnue/horde_legacy_features.h),
so the trainer cannot silently collapse the legacy White `H` plane into the
Black `P` plane. The V2 oracle covers canonical Horde start, horizontal
reflection, every promoted White role, both king-mirror halves and low
material. The normal generator integration also decodes a deterministic real
file into uneven batches and checks every sparse table bound.

Generate a deterministic decoder receipt with:

```console
python tools/horde_training_decoder.py chunk.bin --batch-size 4096
```

This pure-Python implementation is the conformance reference and deterministic
micro-fit input path, not a throughput claim for a full 50-million-position
training run. Any future compiled loader must reproduce its sparse receipt
exactly before replacing it in large-scale training.

[`tools/horde_run6b.py`](../../tools/horde_run6b.py) then reads only the
registered Run 6B artifact and replays its integer network directly from those
legacy sparse rows. Its independent implementation covers feature-transformer
wrapping, both perspectives, piece-count buckets, PSQT, all eight layer stacks,
activation shifts and the serialized dense-weight order. CI compares its raw
PSQT, positional and total outputs against the production engine on 512
reachable positions containing captures, promotion, en passant and castling.
This pins the complete legacy trainer input path before any fresh network is
optimized or exported.

## Fresh legacy control

[`tools/horde_split_training_book.py`](../../tools/horde_split_training_book.py)
partitions one source EPD by the SHA-256 of a horizontal-reflection canonical
key made from its first four normalized FEN fields. A position and its file
mirror therefore cannot cross training and validation roles. It rejects
duplicate physical states and writes exclusive train, validation, and receipt
artifacts. The old exact-key V1 assignment remains available only through
`--legacy-exact-key-v1` for receipt replay; new generation uses
`HORDE_TRAINING_BOOK_SPLIT_V2`.

The fresh-control trainer requires the split receipt and independently checks
that the two `HORDE_BIN_V1` files identify different book hashes, identical
teacher settings, and the same producer and Run 6B identities. It then runs
[`tools/horde_training_split_audit.py`](../../tools/horde_training_split_audit.py)
over the generated records. The audit requires zero cross-role overlap for
both the complete physical state and the complete legacy evaluator input.
Every sample is globally identified by `(payload_sha256, local_record_index)`;
the full file, header, manifest, payload, producer, book, and network identities
are retained in decoder and trainer receipts.

[`tools/horde_training_control.py`](../../tools/horde_training_control.py)
trains the exact serialized legacy topology: one shared 896-by-512 H/P feature
transformer with PSQT outputs and eight `1024 -> 16 -> 32 -> 1` layer stacks.
It intentionally omits the historical training-only first-layer factorizer.
That hidden parameterization is not part of the serialized network and has no
matching meaning in the V2 candidates; it may be tested later as a separate
training-method ablation, but it is not mixed into the architecture control.

The reference recipe uses lambda 0.6, RAdam, a lower output-head learning rate,
the legacy dense quantization bounds, and a deterministic bounded-memory
SplitMix64 block shuffle. Parameter initialization derives an independent
SHA-256 seed from each semantic parameter name, so changing one transformer's
shape cannot perturb identically shaped trunks or heads in another candidate.
Scores with absolute value at least 31,507 are excluded from the score-derived
WDL term and retained in the game-result term, so mate-distance values are
never regressed as ordinary centipawns. The network predicts the
pre-postprocessor value. The loss graph applies the engine's integer rule-50
damping exactly once, including truncation toward zero and the tablebase-safe
clamp, while using a straight-through gradient for truncation.

Every run requires a validated `HORDE_WDL_CALIBRATION_V1` artifact fitted from
the exact authenticated training file. Its side-specific Davidson parameters
are frozen across architecture candidates. Both the postprocessed network
prediction and stored non-mate teacher score are mapped to loss/draw/win
probabilities. The score and one-hot result terms are half-Brier losses combined
per record with lambda 0.6 and then averaged without class weighting,
resampling, or side pooling. The calibration SHA-256 and source identities are
bound into settings, checkpoints, metrics, and the final receipt. See
[`wdl-calibration-v1.md`](wdl-calibration-v1.md) for the complete contract.

CPU and CUDA runs record complete environment, data, schedule, sample-order
chain, state, checkpoint, and metric hashes. A checkpoint contains the model,
optimizer, scheduler, device-specific RNG, sampler cursor, partial-epoch
metrics, examples consumed, and every identity needed to reject an incompatible
resume. [`tests/horde_training_resume.py`](../../tests/horde_training_resume.py)
compares an uninterrupted run with a mid-epoch stop/resume. It requires exact
semantic equality of every checkpoint field and byte-identical metrics; the
PyTorch archive SHA itself is not treated as semantic identity. CPU is the
exact cross-run receipt path; CUDA determinism is verified independently on
the target trainer host.

This Python path is a correctness and convergence reference. A compiled loader
must reproduce its batches, masks, loss, and state transitions before the
50-million-position ladder; the canary throughput is not a production training
claim.

The clean CPU canary at source commit `ff30b366` is frozen in
[`fresh-legacy-control-canary-receipt.json`](fresh-legacy-control-canary-receipt.json).
Two independent three-epoch runs produced byte-identical checkpoints, metrics,
and receipts. Their checkpoint SHA-256 is
`D60C946E3943681EE7B0ADA6FE496E324D6F84C67AC0AF3D4D9626701D12A495`;
the validation loss moved from `0.1912389414` to `0.1884044660`. This is an
integration and determinism receipt only, not a comparison against Run 6B. It
predates split V2, cross-role generated-record auditing, exact rule-50 loss,
and resumable checkpoints, so it remains historical evidence rather than the
authorization for a 50-million-position campaign.

The current authenticated integration canary at source commit `491e5227` is
frozen in
[`fresh-legacy-control-v3-canary-receipt.json`](fresh-legacy-control-v3-canary-receipt.json).
Its regenerated files bind `HORDE_LABEL_CONTRACT_V1`, the clean CI producer,
Run 6B, and the V2 book split. A training-only side-specific Davidson artifact
binds the exact 4,096-record training file. Two three-epoch CPU runs produced
byte-identical checkpoints, metrics, and receipts under
`HORDE_WDL_HALF_BRIER_V1`. This closes the label, loss, calibration, resume,
and repeatability plumbing gates. The sample remains an integration canary and
is too small for architecture selection or strength claims.
