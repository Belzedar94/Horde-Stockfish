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
from typing import Any, Callable, Iterator, Sequence

try:
    from . import horde_bin_v1 as wire
except ImportError:
    import horde_bin_v1 as wire


WHITE = 0
BLACK = 1
WHITE_PAWN = 1
BLACK_ROOK = 9
BLACK_KING = 11

LEGACY_SCHEMA = "HORDETEST_HP_LEGACY_V1"
LEGACY_DIMENSIONS = 896
V2_SCHEMA = "V2_BASE_P0"
V2_GLOBAL_DIMENSIONS = 704
V2_ROYAL_DIMENSIONS = 20_480
V2_ROYAL_RANK8_DIMENSIONS = 5_120

# V3 appends six contextual White-pawn blocks after the immutable G0 rows of the
# same stream. Every block is keyed on the frontier pawn of a file, so each has
# at most eight active rows and the whole contextual tail has at most 48.
V3_SCHEMA = "V3_G1024_PAWN_WPC8"
V3_PAWN_SQUARES = 56  # ranks 1 through 7; a White pawn on rank 8 is illegal
V3_FILE_COUNT_STATES = 7  # a file holds 1 to 7 White pawns, never 8
V3_BLOCK_BASES = {
    "frontier_square": V2_GLOBAL_DIMENSIONS,
    "rearmost_square": V2_GLOBAL_DIMENSIONS + 56,
    "file_count": V2_GLOBAL_DIMENSIONS + 112,
    "frontier_blocked": V2_GLOBAL_DIMENSIONS + 168,
    "frontier_supported": V2_GLOBAL_DIMENSIONS + 176,
    "frontier_phalanx": V2_GLOBAL_DIMENSIONS + 184,
}
V3_BLOCK_WIDTHS = {
    "frontier_square": 56,
    "rearmost_square": 56,
    "file_count": 56,
    "frontier_blocked": 8,
    "frontier_supported": 8,
    "frontier_phalanx": 8,
}
V3_BLOCK_ORDER = (
    "frontier_square",
    "rearmost_square",
    "file_count",
    "frontier_blocked",
    "frontier_supported",
    "frontier_phalanx",
)
V3_CONTEXTUAL_ROWS = 192
V3_GLOBAL_DIMENSIONS = V2_GLOBAL_DIMENSIONS + V3_CONTEXTUAL_ROWS  # 896
V3_MAX_ACTIVE_CONTEXTUAL_ROWS = 48
V3_MAX_ACTIVE_ROWS = 52 + V3_MAX_ACTIVE_CONTEXTUAL_ROWS  # 100

# HORDE_BIN_V1 physical piece codes are deliberately aligned with V2 fixed
# roles after subtracting one. Legacy families preserve Run 6B's H/P split.
LEGACY_FAMILY_BY_CODE = (-1, 5, 1, 2, 3, 4, 0, 1, 2, 3, 4, 6)
FEN_PIECE_BY_CODE = ".PNBRQpnbrqk"


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
    # HORDE_BIN_V1_R2: 0 for a self-play line sample, 1-3 for an expansion child.
    # Children are not independent samples; they share their parent's game and
    # inherit its result, so any dedup or shuffle must treat the (parent,
    # children) unit as one.
    expansion_family: int = 0
    board: tuple[int, ...] = ()
    castling_rights: int = 0
    ep_square: int = 64
    source_payload_sha256: str = ""


@dataclass(frozen=True, slots=True)
class SparseBatch:
    source_payload_sha256: str
    record_indices: tuple[int, ...]
    physical_boards: tuple[tuple[int, ...], ...]
    physical_position_keys: tuple[bytes, ...]
    legacy_model_input_keys: tuple[bytes, ...]
    legacy_piece_offsets: tuple[int, ...]
    global_offsets: tuple[int, ...]
    royal_offsets: tuple[int, ...]
    legacy_white: tuple[int, ...]
    legacy_black: tuple[int, ...]
    v2_global: tuple[int, ...]
    v2_royal: tuple[int, ...]
    royal_buckets: tuple[int, ...]
    royal_mirrors: tuple[bool, ...]
    side_to_move: tuple[int, ...]
    physical_piece_count: tuple[int, ...]
    white_piece_count: tuple[int, ...]
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


def horizontal_reflect_board(board: Sequence[int]) -> tuple[int, ...]:
    _require(len(board) == 64, f"reflection board has {len(board)} squares instead of 64")
    return tuple(board[square ^ 7] for square in range(64))


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


def v3_contextual_rows(board: Sequence[int]) -> tuple[int, ...]:
    """Enumerate the six V3 contextual White-pawn blocks from a physical board.

    Every predicate is keyed on the frontier pawn of a non-empty file, so each
    block contributes at most one row per file. Rows are emitted block by block
    in ``V3_BLOCK_ORDER`` and file by file in ascending order, so the stream is
    deterministic and independent of board scan order.
    """

    _require(len(board) == 64, f"contextual board has {len(board)} squares instead of 64")
    _require(
        all(type(code) is int and 0 <= code <= BLACK_KING for code in board),
        "contextual board contains an invalid physical piece code",
    )

    frontier: list[int] = [-1] * 8
    rearmost: list[int] = [-1] * 8
    counts: list[int] = [0] * 8
    for square in range(64):
        if board[square] != WHITE_PAWN:
            continue
        rank, file_index = divmod(square, 8)
        _require(rank != 7, "a White pawn occupies rank 8")
        counts[file_index] += 1
        if rank > frontier[file_index]:
            frontier[file_index] = rank
        if rearmost[file_index] < 0 or rank < rearmost[file_index]:
            rearmost[file_index] = rank

    blocks: dict[str, list[int]] = {name: [] for name in V3_BLOCK_ORDER}
    for file_index in range(8):
        rank = frontier[file_index]
        if rank < 0:
            _require(counts[file_index] == 0, "empty file reported a White pawn count")
            continue
        _require(
            1 <= counts[file_index] <= V3_FILE_COUNT_STATES,
            f"file {file_index} holds {counts[file_index]} White pawns",
        )
        blocks["frontier_square"].append(rank * 8 + file_index)
        blocks["rearmost_square"].append(rearmost[file_index] * 8 + file_index)
        blocks["file_count"].append(file_index * V3_FILE_COUNT_STATES + counts[file_index] - 1)
        if board[(rank + 1) * 8 + file_index] != 0:
            blocks["frontier_blocked"].append(file_index)
        neighbours = [file_index - 1, file_index + 1]
        if rank >= 1 and any(
            0 <= neighbour < 8 and board[(rank - 1) * 8 + neighbour] == WHITE_PAWN
            for neighbour in neighbours
        ):
            blocks["frontier_supported"].append(file_index)
        if any(
            0 <= neighbour < 8 and board[rank * 8 + neighbour] == WHITE_PAWN
            for neighbour in neighbours
        ):
            blocks["frontier_phalanx"].append(file_index)

    rows: list[int] = []
    for name in V3_BLOCK_ORDER:
        base = V3_BLOCK_BASES[name]
        width = V3_BLOCK_WIDTHS[name]
        offsets = blocks[name]
        _require(len(offsets) <= 8, f"contextual block {name} activated more than eight rows")
        for offset in offsets:
            _require(0 <= offset < width, f"contextual block {name} offset {offset} is out of range")
            rows.append(base + offset)
    _require(
        len(rows) <= V3_MAX_ACTIVE_CONTEXTUAL_ROWS,
        f"contextual tail activated {len(rows)} rows",
    )
    _require(len(set(rows)) == len(rows), "duplicate V3 contextual feature")
    _require(
        all(V2_GLOBAL_DIMENSIONS <= row < V3_GLOBAL_DIMENSIONS for row in rows),
        "V3 contextual row escaped its reserved range",
    )
    return tuple(rows)


def v3_global_rows(board: Sequence[int]) -> tuple[int, ...]:
    """Return the complete V3 stream: immutable G0 rows then the contextual tail."""

    rows = extract_sparse_features(board).v2_global + v3_contextual_rows(board)
    _require(len(rows) <= V3_MAX_ACTIVE_ROWS, f"V3 stream activated {len(rows)} rows")
    _require(len(set(rows)) == len(rows), "duplicate V3 stream feature")
    return rows


def reflect_v3_row(row: int) -> int:
    """Map one V3 stream row to its horizontally reflected counterpart."""

    if row < V2_GLOBAL_DIMENSIONS:
        role, square = divmod(row, 64)
        return role * 64 + (square ^ 7)
    for name in V3_BLOCK_ORDER:
        base = V3_BLOCK_BASES[name]
        width = V3_BLOCK_WIDTHS[name]
        if base <= row < base + width:
            offset = row - base
            if name in ("frontier_square", "rearmost_square"):
                return base + (offset ^ 7)
            if name == "file_count":
                file_index, state = divmod(offset, V3_FILE_COUNT_STATES)
                return base + (7 - file_index) * V3_FILE_COUNT_STATES + state
            return base + (7 - offset)
    raise wire.FormatError(f"row {row} is outside the V3 stream")


def synthetic_boards(count: int, seed: int = 20260820) -> list[tuple[int, ...]]:
    """Deterministic legal-shaped boards for environments without a corpus.

    Continuous integration has no access to the authenticated training volume,
    so every V3 suite must be able to run without it. These boards satisfy the
    physical contract the decoder enforces: one Black king, no White pawn on
    rank 8, no Black pawn on rank 1 or 8, at most 36 White and 16 Black pieces.
    """

    import random

    generator = random.Random(seed)
    boards: list[tuple[int, ...]] = []
    while len(boards) < count:
        board = [0] * 64
        board[generator.randrange(56, 64)] = BLACK_KING
        for _ in range(generator.randrange(1, 34)):
            square = generator.randrange(0, 56)
            if board[square] == 0:
                board[square] = WHITE_PAWN
        for _ in range(generator.randrange(0, 6)):
            square = generator.randrange(0, 64)
            if board[square] == 0:
                board[square] = generator.choice((2, 3, 4, 5))
        for _ in range(generator.randrange(0, 10)):
            square = generator.randrange(8, 56)
            if board[square] == 0:
                board[square] = generator.choice((6, 7, 8, 9, 10))
        white = sum(1 for code in board if 1 <= code <= 5)
        black = sum(1 for code in board if 6 <= code <= 11)
        if 1 <= white <= 36 and 1 <= black <= 16 and white + black <= 52:
            boards.append(tuple(board))
    return boards


def synthetic_record_payload(count: int, seed: int = 20260820) -> bytes:
    """Deterministic HORDE_BIN_V1 record payload for corpus-free environments.

    Each record carries a synthetic board plus a quiet White pawn push as both
    best and played move, so it satisfies every check in
    ``horde_bin_v1.validate_record`` without needing the authenticated volume.
    Labels are synthetic and carry no meaning; only the physical board and the
    clock fields are consumed by the V3 suites.
    """

    import random
    import struct

    generator = random.Random(seed ^ 0x5EED)
    out = bytearray()
    produced = 0
    attempt = 0
    while produced < count:
        attempt += 1
        board = list(synthetic_boards(1, seed=seed + attempt)[0])
        pushes = [
            square
            for square in range(48)
            if board[square] == WHITE_PAWN and board[square + 8] == 0
        ]
        if not pushes:
            continue
        origin = generator.choice(pushes)
        move = ((origin & 63) << 6) | ((origin + 8) & 63)
        packed = bytearray(48)
        for square in range(0, 64, 2):
            packed[square >> 1] = (board[square] & 0x0F) | ((board[square + 1] & 0x0F) << 4)
        rule50 = generator.randrange(0, 120)
        game_ply = generator.randrange(0, 150) * 2
        score = generator.randrange(-2500, 2500)
        result = generator.choice((-1, 0, 1))
        reason = 2 if result else 3
        packed[32] = WHITE
        packed[33] = 0
        packed[34] = 64
        packed[35] = 0
        struct.pack_into("<HHhHHbB", packed, 36, rule50, game_ply, score, move, move, result, reason)
        wire.validate_record(bytes(packed), produced)
        out.extend(packed)
        produced += 1
    return bytes(out)


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
        expansion_family=decoded["family"],
        board=tuple(decoded["board"]),
        castling_rights=decoded["castling"],
        ep_square=decoded["ep_square"],
    )


def physical_position_key(record: TrainingRecord) -> bytes:
    """Encode the first four FEN fields, independently of clock labels."""

    _require(len(record.board) == 64, f"record {record.index} has no physical board")
    _require(all(0 <= code <= BLACK_KING for code in record.board),
             f"record {record.index} has an invalid physical board")
    return bytes(record.board) + struct.pack(
        "<BBB",
        record.side_to_move,
        record.castling_rights,
        record.ep_square,
    )


def legacy_model_input_key(record: TrainingRecord) -> bytes:
    """Encode the complete legacy evaluator input before learned parameters."""

    features = record.features
    piece_count = len(features.legacy_white)
    _require(1 <= piece_count <= 52, f"record {record.index} has invalid legacy row count")
    bucket = min((piece_count - 1) * 8 // 52, 7)
    payload = bytearray(struct.pack("<BBH", record.side_to_move, bucket, record.rule50_count))
    for indices in (features.legacy_white, features.legacy_black):
        payload.extend(struct.pack("<H", len(indices)))
        for feature in indices:
            payload.extend(struct.pack("<H", feature))
    return bytes(payload)


def training_record_fen(record: TrainingRecord) -> str:
    """Reconstruct the exact board-local Horde FEN stored by one record."""

    _require(len(record.board) == 64, f"record {record.index} has no physical board")
    ranks: list[str] = []
    for rank in range(7, -1, -1):
        encoded: list[str] = []
        empty = 0
        for file_index in range(8):
            code = record.board[rank * 8 + file_index]
            _require(0 <= code < len(FEN_PIECE_BY_CODE),
                     f"record {record.index} has an invalid FEN piece code")
            if code == 0:
                empty += 1
            else:
                if empty:
                    encoded.append(str(empty))
                    empty = 0
                encoded.append(FEN_PIECE_BY_CODE[code])
        if empty:
            encoded.append(str(empty))
        ranks.append("".join(encoded))

    side = "w" if record.side_to_move == WHITE else "b"
    castling = ("k" if record.castling_rights & 1 else "") + (
        "q" if record.castling_rights & 2 else ""
    )
    castling = castling or "-"
    if record.ep_square == 64:
        ep_square = "-"
    else:
        _require(0 <= record.ep_square < 64, f"record {record.index} has an invalid EP square")
        ep_square = chr(ord("a") + (record.ep_square & 7)) + str((record.ep_square >> 3) + 1)
    _require(
        (record.game_ply & 1) == record.side_to_move,
        f"record {record.index} game ply contradicts side to move",
    )
    fullmove = record.game_ply // 2 + 1
    return " ".join(
        (
            "/".join(ranks),
            side,
            castling,
            ep_square,
            str(record.rule50_count),
            str(fullmove),
        )
    )


def make_sparse_batch(
    records: Sequence[TrainingRecord],
    contextual: Callable[[Sequence[int]], tuple[int, ...]] | None = None,
) -> SparseBatch:
    """Build one CSR sparse batch.

    ``contextual`` optionally appends a named contextual tail after the
    immutable G0 rows of the Global stream, which is the extension point the
    offsets were separated for. ``physical_piece_count`` still counts only the
    physical G0 rows, so the split stays recoverable downstream.
    """

    payload_identities = {record.source_payload_sha256 for record in records if record.source_payload_sha256}
    _require(len(payload_identities) <= 1, "batch mixes multiple source payload identities")
    source_payload_sha256 = next(iter(payload_identities), "")
    record_indices: list[int] = []
    physical_boards: list[tuple[int, ...]] = []
    physical_position_keys: list[bytes] = []
    legacy_model_input_keys: list[bytes] = []
    legacy_piece_offsets = [0]
    global_offsets = [0]
    royal_offsets = [0]
    legacy_white: list[int] = []
    legacy_black: list[int] = []
    v2_global: list[int] = []
    v2_royal: list[int] = []
    royal_buckets: list[int] = []
    royal_mirrors: list[bool] = []
    side_to_move: list[int] = []
    physical_piece_count: list[int] = []
    white_piece_count: list[int] = []
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
            len(record.board) == 64,
            f"record {record.index} does not retain its complete physical board",
        )
        physical_features = extract_sparse_features(record.board)
        physical_rows = len(physical_features.v2_global)
        _require(
            len(features.legacy_white) == len(features.legacy_black) == physical_rows,
            f"record {record.index} physical piece-domain lengths differ",
        )
        _require(
            len(features.v2_royal) + 1 == physical_rows,
            f"record {record.index} Royal domain did not exclude exactly the Black king",
        )
        _require(
            features.legacy_white == physical_features.legacy_white
            and features.legacy_black == physical_features.legacy_black
            and features.v2_royal == physical_features.v2_royal,
            f"record {record.index} changed a physical or Royal base stream",
        )
        _require(
            features.v2_global[:physical_rows] == physical_features.v2_global,
            f"record {record.index} Global stream does not begin with complete physical G0 rows",
        )
        record_indices.append(record.index)
        physical_boards.append(record.board)
        physical_position_keys.append(physical_position_key(record))
        legacy_model_input_keys.append(legacy_model_input_key(record))
        legacy_white.extend(features.legacy_white)
        legacy_black.extend(features.legacy_black)
        v2_global.extend(features.v2_global)
        if contextual is not None:
            extra = contextual(record.board)
            _require(
                all(
                    V2_GLOBAL_DIMENSIONS <= row < V3_GLOBAL_DIMENSIONS for row in extra
                ),
                f"record {record.index} contextual row escaped its reserved range",
            )
            _require(
                len(set(extra)) == len(extra),
                f"record {record.index} repeats a contextual row",
            )
            v2_global.extend(extra)
        v2_royal.extend(features.v2_royal)
        legacy_piece_offsets.append(len(legacy_white))
        global_offsets.append(len(v2_global))
        royal_offsets.append(len(v2_royal))
        royal_buckets.append(features.royal_bucket)
        royal_mirrors.append(features.royal_mirror)
        side_to_move.append(record.side_to_move)
        physical_piece_count.append(sum(code != 0 for code in record.board))
        white_piece_count.append(sum(1 <= code <= 5 for code in record.board))
        rule50_count.append(record.rule50_count)
        game_ply.append(record.game_ply)
        scores.append(record.score)
        best_moves.append(record.best_move)
        played_moves.append(record.played_move)
        results.append(record.result)
        outcome_reasons.append(record.outcome_reason)

    return SparseBatch(
        source_payload_sha256=source_payload_sha256,
        record_indices=tuple(record_indices),
        physical_boards=tuple(physical_boards),
        physical_position_keys=tuple(physical_position_keys),
        legacy_model_input_keys=tuple(legacy_model_input_keys),
        legacy_piece_offsets=tuple(legacy_piece_offsets),
        global_offsets=tuple(global_offsets),
        royal_offsets=tuple(royal_offsets),
        legacy_white=tuple(legacy_white),
        legacy_black=tuple(legacy_black),
        v2_global=tuple(v2_global),
        v2_royal=tuple(v2_royal),
        royal_buckets=tuple(royal_buckets),
        royal_mirrors=tuple(royal_mirrors),
        side_to_move=tuple(side_to_move),
        physical_piece_count=tuple(physical_piece_count),
        white_piece_count=tuple(white_piece_count),
        rule50_count=tuple(rule50_count),
        game_ply=tuple(game_ply),
        scores=tuple(scores),
        best_moves=tuple(best_moves),
        played_moves=tuple(played_moves),
        results=tuple(results),
        outcome_reasons=tuple(outcome_reasons),
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
            manifest_length = struct.unpack_from("<I", header, 12)[0]
            self.header_sha256 = hashlib.sha256(header).hexdigest().upper()
            self.manifest_sha256 = hashlib.sha256(
                header[16 : 16 + manifest_length]
            ).hexdigest().upper()
            expected_size = wire.HEADER_SIZE + self.manifest["record_count"] * wire.RECORD_SIZE
            actual_size = os.fstat(self._file.fileno()).st_size
            _require(actual_size == expected_size,
                     f"file size {actual_size} does not match manifest framing {expected_size}")

            payload_sha256 = hashlib.sha256()
            file_sha256 = hashlib.sha256(header)
            while chunk := self._file.read(8 * 1024 * 1024):
                payload_sha256.update(chunk)
                file_sha256.update(chunk)
            observed = payload_sha256.hexdigest().upper()
            _require(observed == self.manifest["payload_sha256"], "payload SHA-256 mismatch")
            self.file_sha256 = file_sha256.hexdigest().upper()
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

    def raw_record(self, index: int) -> bytes:
        """Return the exact authenticated record bytes at ``index``."""

        _require(0 <= index < len(self), f"record index {index} is out of range")
        _require(self._mapping is not None, "dataset is closed")
        offset = wire.HEADER_SIZE + index * wire.RECORD_SIZE
        return bytes(self._mapping[offset : offset + wire.RECORD_SIZE])

    def record(self, index: int) -> TrainingRecord:
        raw = self.raw_record(index)
        record = decode_training_record(raw, index)
        return TrainingRecord(
            index=record.index,
            features=record.features,
            side_to_move=record.side_to_move,
            rule50_count=record.rule50_count,
            game_ply=record.game_ply,
            score=record.score,
            best_move=record.best_move,
            played_move=record.played_move,
            result=record.result,
            outcome_reason=record.outcome_reason,
            expansion_family=record.expansion_family,
            board=record.board,
            castling_rights=record.castling_rights,
            ep_square=record.ep_square,
            source_payload_sha256=self.manifest["payload_sha256"],
        )

    def label(self, index: int) -> tuple[int, int, int, int]:
        """Fully validate one record and return its calibration label fields."""

        raw = self.raw_record(index)
        decoded = wire.validate_record(raw, index)
        return decoded["side"], decoded["score"], decoded["result"], decoded["reason"]

    def expansion_family(self, index: int) -> int:
        """0 for a self-play line sample, 1-3 for an expansion child."""

        return int(wire.validate_record(self.raw_record(index), index)["family"])

    def batches(self, batch_size: int) -> Iterator[SparseBatch]:
        _require(batch_size > 0, f"invalid batch size {batch_size}")
        for begin in range(0, len(self), batch_size):
            end = min(begin + batch_size, len(self))
            yield make_sparse_batch(tuple(self.record(index) for index in range(begin, end)))


def _update_index_list(digest: Any, indices: Sequence[int]) -> None:
    digest.update(struct.pack("<I", len(indices)))
    for index in indices:
        digest.update(struct.pack("<I", index))


def dataset_receipt(
    path: Path,
    batch_size: int,
    *,
    dataset_factory: Callable[[Path], Any] = HordeBinV1Dataset,
) -> dict[str, object]:
    digest = hashlib.sha256()
    legacy_stream_digest = hashlib.sha256()
    global_g0_stream_digest = hashlib.sha256()
    global_contextual_stream_digest = hashlib.sha256()
    global_complete_stream_digest = hashlib.sha256()
    royal_stream_digest = hashlib.sha256()
    reflected_legacy_stream_digest = hashlib.sha256()
    reflected_global_g0_stream_digest = hashlib.sha256()
    reflected_royal_stream_digest = hashlib.sha256()
    physical_digest = hashlib.sha256()
    model_input_digest = hashlib.sha256()
    eligibility_digest = hashlib.sha256()
    batches = 0
    legacy_rows = 0
    global_g0_rows = 0
    global_contextual_rows = 0
    global_complete_rows = 0
    royal_rows = 0
    with dataset_factory(path) as dataset:
        for batch in dataset.batches(batch_size):
            batches += 1
            for row in range(len(batch)):
                legacy_begin, legacy_end = batch.legacy_piece_offsets[row : row + 2]
                global_begin, global_end = batch.global_offsets[row : row + 2]
                royal_begin, royal_end = batch.royal_offsets[row : row + 2]
                digest.update(
                    struct.pack(
                        "<QBHHhHHbBBBBB",
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
                        batch.physical_piece_count[row],
                        batch.white_piece_count[row],
                    )
                )
                physical_digest.update(batch.physical_position_keys[row])
                model_input_digest.update(batch.legacy_model_input_keys[row])
                eligibility_digest.update(bytes((abs(batch.scores[row]) < 31_507,)))
                legacy_white = batch.legacy_white[legacy_begin:legacy_end]
                legacy_black = batch.legacy_black[legacy_begin:legacy_end]
                physical_count = batch.physical_piece_count[row]
                global_g0_end = global_begin + physical_count
                _require(
                    global_g0_end <= global_end,
                    f"record {batch.record_indices[row]} Global offsets omit physical G0 rows",
                )
                global_g0 = batch.v2_global[global_begin:global_g0_end]
                global_contextual = batch.v2_global[global_g0_end:global_end]
                global_complete = batch.v2_global[global_begin:global_end]
                royal = batch.v2_royal[royal_begin:royal_end]
                for stream in (legacy_white, legacy_black, global_complete, royal):
                    _update_index_list(digest, stream)
                _update_index_list(legacy_stream_digest, legacy_white)
                _update_index_list(legacy_stream_digest, legacy_black)
                _update_index_list(global_g0_stream_digest, global_g0)
                _update_index_list(global_contextual_stream_digest, global_contextual)
                _update_index_list(global_complete_stream_digest, global_complete)
                _update_index_list(royal_stream_digest, royal)

                board = batch.physical_boards[row]
                reflected_board = horizontal_reflect_board(board)
                _require(
                    horizontal_reflect_board(reflected_board) == board,
                    f"record {batch.record_indices[row]} failed horizontal reflection round trip",
                )
                reflected = extract_sparse_features(reflected_board)
                _require(
                    not global_contextual,
                    "horizontal-reflection receipt requires a named contextual extractor",
                )
                _require(
                    tuple(sorted(reflected.v2_royal)) == tuple(sorted(royal)),
                    f"record {batch.record_indices[row]} changed canonical Royal rows "
                    "under reflection",
                )
                _update_index_list(reflected_legacy_stream_digest, reflected.legacy_white)
                _update_index_list(reflected_legacy_stream_digest, reflected.legacy_black)
                _update_index_list(reflected_global_g0_stream_digest, reflected.v2_global)
                _update_index_list(reflected_royal_stream_digest, reflected.v2_royal)
            legacy_rows += len(batch.legacy_white)
            global_g0_rows += sum(batch.physical_piece_count)
            global_complete_rows += len(batch.v2_global)
            global_contextual_rows += len(batch.v2_global) - sum(batch.physical_piece_count)
            royal_rows += len(batch.v2_royal)

        return {
            "schema": "HORDE_TRAINING_DECODER_V2",
            "source_schema": dataset.manifest["schema"],
            "source": {
                "file_sha256": dataset.file_sha256,
                "header_sha256": dataset.header_sha256,
                "manifest_sha256": dataset.manifest_sha256,
                "payload_sha256": dataset.manifest["payload_sha256"],
                "source_commit": dataset.manifest["source_commit"],
                "producer_sha256": dataset.manifest["producer_sha256"],
                "book_sha256": dataset.manifest["book_sha256"],
                "network": dataset.manifest["network"],
                "label_contract": dataset.manifest["label_contract"],
            },
            "sample_identity": "(payload_sha256, local_record_index)",
            "record_count": len(dataset),
            "batch_size": batch_size,
            "batch_count": batches,
            "legacy": {"schema": LEGACY_SCHEMA, "dimensions": LEGACY_DIMENSIONS},
            "v2": {
                "schema": V2_SCHEMA,
                "global_dimensions": V2_GLOBAL_DIMENSIONS,
                "royal_dimensions": V2_ROYAL_DIMENSIONS,
            },
            "piece_rows": global_g0_rows,
            "royal_rows": royal_rows,
            "row_counts": {
                "legacy_physical": legacy_rows,
                "global_g0": global_g0_rows,
                "global_contextual": (
                    {"unnamed": global_contextual_rows} if global_contextual_rows else {}
                ),
                "global_contextual_total": global_contextual_rows,
                "global_complete": global_complete_rows,
                "royal": royal_rows,
            },
            "feature_stream_sha256": {
                "unreflected": {
                    "legacy": legacy_stream_digest.hexdigest().upper(),
                    "global_g0": global_g0_stream_digest.hexdigest().upper(),
                    "global_contextual": global_contextual_stream_digest.hexdigest().upper(),
                    "global_complete": global_complete_stream_digest.hexdigest().upper(),
                    "royal": royal_stream_digest.hexdigest().upper(),
                },
                "horizontal_reflection": {
                    "legacy": reflected_legacy_stream_digest.hexdigest().upper(),
                    "global_g0": reflected_global_g0_stream_digest.hexdigest().upper(),
                    "global_contextual": global_contextual_stream_digest.hexdigest().upper(),
                    "global_complete": reflected_global_g0_stream_digest.hexdigest().upper(),
                    "royal": reflected_royal_stream_digest.hexdigest().upper(),
                },
            },
            "sparse_sha256": digest.hexdigest().upper(),
            "physical_position_sha256": physical_digest.hexdigest().upper(),
            "legacy_model_input_sha256": model_input_digest.hexdigest().upper(),
            "eval_eligibility_sha256": eligibility_digest.hexdigest().upper(),
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
