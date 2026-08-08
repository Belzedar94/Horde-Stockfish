# Horde training-data contract V1

`HORDE_BIN_V1` is the canonical Horde-Stockfish self-play format. It stores the physical game position, not an evaluator-specific feature vector. White pawns therefore remain ordinary `P` pieces in the dataset. A trainer targeting `HORDETEST_HP_LEGACY_V1` maps those white pawns to the legacy `H` feature plane when it loads a record; future role-aware architectures may consume the same record without changing the game representation.

## File structure

Each file starts with a 2,048-byte header and continues with fixed 48-byte records. The first eight bytes are ASCII `HORDEBIN`. Little-endian `uint16` values at offsets 8 and 10 contain format version `1` and header size `2048`. A little-endian `uint32` at offset 12 gives the length of the compact UTF-8 JSON manifest beginning at offset 16. Every unused header byte is zero.

The manifest binds the file to the full source commit, dirty-state marker, Run 6B network SHA-256, opening-book SHA-256, producer executable SHA-256, exact generation settings, record count, and SHA-256 of the record payload. Seeds are decimal strings so consumers do not lose 64-bit precision.

The complete byte layout, piece codes, flags, result perspective, and terminal-reason values are frozen in [`schemas/horde-bin-v1.schema.json`](../../schemas/horde-bin-v1.schema.json). The SHA-256 of that schema is part of the generator capability handshake and every file manifest.

## Position and label semantics

The board is encoded as 64 four-bit physical piece codes. It supports the kingless White Horde, the single Black king, all promoted White pieces, up to 36 White pieces, and up to 52 pieces in total. Only Black castling rights are representable. The en-passant square, rule-50 clock, game ply, side to move, best move, and played move are sufficient for exact physical FEN reconstruction and audit; repetition history remains a game-level property.

Scores and results are relative to the side to move in the stored position. `score` is the raw internal `Value` produced by the exact Horde-Stockfish search, before UCI centipawn conversion. `best_move` is the label selected by the principal search. `played_move` records the move used to advance self-play and may differ when deterministic exploration is scheduled.

Games are labeled only after `Position::outcome()` returns an exact terminal result. Checkmate, Horde extinction, stalemate, Horde fortress, the automatic fifty-move rule, and fivefold repetition have distinct reason codes. The separate per-color insufficient-winning-material predicate is never treated as an automatic draw. A game that reaches the generator safety ply limit without a terminal result is discarded, not mislabeled.

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
