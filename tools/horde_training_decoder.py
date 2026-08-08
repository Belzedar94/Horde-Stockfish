#!/usr/bin/env python3
"""Decode HORDE_BIN_V1 into legacy H/P and Horde V2 sparse features."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import sys
from typing import Any, Iterator, Sequence

try:
    from . import horde_bin_v1 as wire
except ImportError:
    import horde_bin_v1 as wire


WHITE = 0
BLACK = 1
BLACK_KING = 11

LEGACY_SCHEMA = "HORDETEST_HP_LEGACY_V1"
LEGACY_DIMENSIONS = 896
V2_SCHEMA = "V2_BASE_P0"
V2_GLOBAL_DIMENSIONS = 704
V2_ROYAL_DIMENSIONS = 20_480

# HORDE_BIN_V1 physical piece codes are deliberately aligned with V2 fixed
# roles after subtracting one. Legacy families preserve Run 6B's H/P split.
LEGACY_FAMILY_BY_CODE = (-1, 5, 1, 2, 3, 4, 0, 1, 2, 3, 4, 6)


@dataclass(frozen=True, slots=True)
class SparseFeatures:
    legacy_white: tuple[int, ...]
    legacy_black: tuple[int, ...]
    v2_global: tuple[int, ...]
    v2_royal: tuple[int, ...]
    royal_bucket: int
    royal_mirror: bool


@dataclass(frozen=True, slots=True)
class TrainingRecord:
    index: int
    features: SparseFeatures
    side_to_move: int
    rule50_count: int
    game_ply: int
    score: int
    best_move: int
    played_move: int
    result: int
    outcome_reason: int


@dataclass(frozen=True, slots=True)
class SparseBatch:
    record_indices: tuple[int, ...]
    piece_offsets: tuple[int, ...]
    royal_offsets: tuple[int, ...]
    legacy_white: tuple[int, ...]
    legacy_black: tuple[int, ...]
    v2_global: tuple[int, ...]
    v2_royal: tuple[int, ...]
    royal_buckets: tuple[int, ...]
    royal_mirrors: tuple[bool, ...]
    side_to_move: tuple[int, ...]
    rule50_count: tuple[int, ...]
    game_ply: tuple[int, ...]
    scores: tuple[int, ...]
    best_moves: tuple[int, ...]
    played_moves: tuple[int, ...]
    results: tuple[int, ...]
    outcome_reasons: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.record_indices)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise wire.FormatError(message)


def legacy_feature_index(perspective: int, square: int, piece_code: int) -> int:
    _require(perspective in (WHITE, BLACK), f"invalid legacy perspective {perspective}")
    _require(0 <= square < 64, f"invalid feature square {square}")
    _require(1 <= piece_code < len(LEGACY_FAMILY_BY_CODE), f"invalid piece code {piece_code}")

    family = LEGACY_FAMILY_BY_CODE[piece_code]
    color = WHITE if piece_code <= 5 else BLACK
    plane = 2 * family + int(color != perspective)
    oriented_square = square if perspective == WHITE else square ^ 56
    index = plane * 64 + oriented_square
    _require(index < LEGACY_DIMENSIONS, f"legacy feature index overflow {index}")
    return index


def extract_sparse_features(board: Sequence[int]) -> SparseFeatures:
    _require(len(board) == 64, f"feature board has {len(board)} squares instead of 64")
    _require(all(type(code) is int and 0 <= code <= BLACK_KING for code in board),
             "feature board contains an invalid physical piece code")

    occupied = [(square, code) for square, code in enumerate(board) if code]
    white_count = sum(code <= 5 for _, code in occupied)
    black_count = len(occupied) - white_count
    _require(white_count <= 36, "feature board exceeds 36 White pieces")
    _require(black_count <= 16, "feature board exceeds 16 Black pieces")
    _require(len(occupied) <= 52, "feature board exceeds 52 total pieces")

    king_squares = [square for square, code in occupied if code == BLACK_KING]
    _require(len(king_squares) == 1, "feature board does not contain exactly one Black king")
    king_square = king_squares[0]
    king_file = king_square & 7
    royal_mirror = king_file <= 3
    canonical_king_file = king_file ^ 7 if royal_mirror else king_file
    royal_bucket = (king_square >> 3) * 4 + canonical_king_file - 4
    _require(0 <= royal_bucket < 32, f"invalid Royal bucket {royal_bucket}")

    legacy_white: list[int] = []
    legacy_black: list[int] = []
    v2_global: list[int] = []
    v2_royal: list[int] = []

    for square, code in occupied:
        legacy_white.append(legacy_feature_index(WHITE, square, code))
        legacy_black.append(legacy_feature_index(BLACK, square, code))

        role = code - 1
        global_index = role * 64 + square
        _require(global_index < V2_GLOBAL_DIMENSIONS, f"V2 Global index overflow {global_index}")
        v2_global.append(global_index)

        if code == BLACK_KING:
            continue
        oriented_square = square ^ 7 if royal_mirror else square
        royal_index = (royal_bucket * 10 + role) * 64 + oriented_square
        _require(royal_index < V2_ROYAL_DIMENSIONS, f"V2 Royal index overflow {royal_index}")
        v2_royal.append(royal_index)

    _require(len(set(legacy_white)) == len(legacy_white), "duplicate legacy White feature")
    _require(len(set(legacy_black)) == len(legacy_black), "duplicate legacy Black feature")
    _require(len(set(v2_global)) == len(v2_global), "duplicate V2 Global feature")
    _require(len(set(v2_royal)) == len(v2_royal), "duplicate V2 Royal feature")
    return SparseFeatures(
        tuple(legacy_white),
        tuple(legacy_black),
        tuple(v2_global),
        tuple(v2_royal),
        royal_bucket,
        royal_mirror,
    )


def decode_training_record(raw: bytes, index: int) -> TrainingRecord:
    decoded = wire.validate_record(raw, index)
    return TrainingRecord(
        index=index,
        features=extract_sparse_features(decoded["board"]),
        side_to_move=decoded["side"],
        rule50_count=decoded["rule50"],
        game_ply=decoded["game_ply"],
        score=decoded["score"],
        best_move=decoded["best_move"],
        played_move=decoded["played_move"],
        result=decoded["result"],
        outcome_reason=decoded["reason"],
    )


def make_sparse_batch(records: Sequence[TrainingRecord]) -> SparseBatch:
    record_indices: list[int] = []
    piece_offsets = [0]
    royal_offsets = [0]
    legacy_white: list[int] = []
    legacy_black: list[int] = []
    v2_global: list[int] = []
    v2_royal: list[int] = []
    royal_buckets: list[int] = []
    royal_mirrors: list[bool] = []
    side_to_move: list[int] = []
    rule50_count: list[int] = []
    game_ply: list[int] = []
    scores: list[int] = []
    best_moves: list[int] = []
    played_moves: list[int] = []
    results: list[int] = []
    outcome_reasons: list[int] = []

    for record in records:
        features = record.features
        _require(
            len(features.legacy_white) == len(features.legacy_black) == len(features.v2_global),
            f"record {record.index} piece-domain lengths differ",
        )
        _require(
            len(features.v2_royal) + 1 == len(features.v2_global),
            f"record {record.index} Royal domain did not exclude exactly the Black king",
        )
        record_indices.append(record.index)
        legacy_white.extend(features.legacy_white)
        legacy_black.extend(features.legacy_black)
        v2_global.extend(features.v2_global)
        v2_royal.extend(features.v2_royal)
        piece_offsets.append(len(v2_global))
        royal_offsets.append(len(v2_royal))
        royal_buckets.append(features.royal_bucket)
        royal_mirrors.append(features.royal_mirror)
        side_to_move.append(record.side_to_move)
        rule50_count.append(record.rule50_count)
        game_ply.append(record.game_ply)
        scores.append(record.score)
        best_moves.append(record.best_move)
        played_moves.append(record.played_move)
        results.append(record.result)
        outcome_reasons.append(record.outcome_reason)

    return SparseBatch(
        tuple(record_indices),
        tuple(piece_offsets),
        tuple(royal_offsets),
        tuple(legacy_white),
        tuple(legacy_black),
        tuple(v2_global),
        tuple(v2_royal),
        tuple(royal_buckets),
        tuple(royal_mirrors),
        tuple(side_to_move),
        tuple(rule50_count),
        tuple(game_ply),
        tuple(scores),
        tuple(best_moves),
        tuple(played_moves),
        tuple(results),
        tuple(outcome_reasons),
    )


class HordeBinV1Dataset:
    """Read-only, payload-verified mmap view of one HORDE_BIN_V1 file."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self._file = self.path.open("rb")
        self._mapping: mmap.mmap | None = None
        try:
            header = self._file.read(wire.HEADER_SIZE)
            self.manifest = wire.parse_header(header)
            expected_size = wire.HEADER_SIZE + self.manifest["record_count"] * wire.RECORD_SIZE
            actual_size = os.fstat(self._file.fileno()).st_size
            _require(actual_size == expected_size,
                     f"file size {actual_size} does not match manifest framing {expected_size}")

            payload_sha256 = hashlib.sha256()
            while chunk := self._file.read(8 * 1024 * 1024):
                payload_sha256.update(chunk)
            observed = payload_sha256.hexdigest().upper()
            _require(observed == self.manifest["payload_sha256"], "payload SHA-256 mismatch")
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> HordeBinV1Dataset:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __len__(self) -> int:
        return int(self.manifest["record_count"])

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if not self._file.closed:
            self._file.close()

    def record(self, index: int) -> TrainingRecord:
        _require(0 <= index < len(self), f"record index {index} is out of range")
        _require(self._mapping is not None, "dataset is closed")
        offset = wire.HEADER_SIZE + index * wire.RECORD_SIZE
        raw = self._mapping[offset : offset + wire.RECORD_SIZE]
        return decode_training_record(raw, index)

    def batches(self, batch_size: int) -> Iterator[SparseBatch]:
        _require(batch_size > 0, f"invalid batch size {batch_size}")
        for begin in range(0, len(self), batch_size):
            end = min(begin + batch_size, len(self))
            yield make_sparse_batch(tuple(self.record(index) for index in range(begin, end)))


def _update_index_list(digest: Any, indices: Sequence[int]) -> None:
    digest.update(struct.pack("<I", len(indices)))
    for index in indices:
        digest.update(struct.pack("<I", index))


def dataset_receipt(path: Path, batch_size: int) -> dict[str, object]:
    digest = hashlib.sha256()
    batches = 0
    piece_rows = 0
    royal_rows = 0
    with HordeBinV1Dataset(path) as dataset:
        for batch in dataset.batches(batch_size):
            batches += 1
            for row in range(len(batch)):
                piece_begin, piece_end = batch.piece_offsets[row : row + 2]
                royal_begin, royal_end = batch.royal_offsets[row : row + 2]
                digest.update(
                    struct.pack(
                        "<QBHHhHHbBBBB",
                        batch.record_indices[row],
                        batch.side_to_move[row],
                        batch.rule50_count[row],
                        batch.game_ply[row],
                        batch.scores[row],
                        batch.best_moves[row],
                        batch.played_moves[row],
                        batch.results[row],
                        batch.outcome_reasons[row],
                        batch.royal_buckets[row],
                        int(batch.royal_mirrors[row]),
                        0,
                    )
                )
                _update_index_list(digest, batch.legacy_white[piece_begin:piece_end])
                _update_index_list(digest, batch.legacy_black[piece_begin:piece_end])
                _update_index_list(digest, batch.v2_global[piece_begin:piece_end])
                _update_index_list(digest, batch.v2_royal[royal_begin:royal_end])
            piece_rows += len(batch.v2_global)
            royal_rows += len(batch.v2_royal)

        return {
            "schema": "HORDE_TRAINING_DECODER_V1",
            "source_schema": dataset.manifest["schema"],
            "record_count": len(dataset),
            "batch_size": batch_size,
            "batch_count": batches,
            "legacy": {"schema": LEGACY_SCHEMA, "dimensions": LEGACY_DIMENSIONS},
            "v2": {
                "schema": V2_SCHEMA,
                "global_dimensions": V2_GLOBAL_DIMENSIONS,
                "royal_dimensions": V2_ROYAL_DIMENSIONS,
            },
            "piece_rows": piece_rows,
            "royal_rows": royal_rows,
            "sparse_sha256": digest.hexdigest().upper(),
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(json.dumps(dataset_receipt(args.file, args.batch_size), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (wire.FormatError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
