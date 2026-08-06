#!/usr/bin/env python3
"""Validate and summarize HORDE_BIN_V1 training-data files."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
from typing import Any, Iterator


MAGIC = b"HORDEBIN"
FORMAT_VERSION = 1
HEADER_SIZE = 2048
RECORD_SIZE = 48
SCHEMA_NAME = "HORDE_BIN_V1"
SCHEMA_SHA256 = "B46ADE18AB8954A6AB232593484273E50C12B51550A938763A7A7D94DCCB63E4"
RUN6B_SHA256 = "B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7"

PIECE_NAMES = {
    0: "empty",
    1: "white_pawn",
    2: "white_knight",
    3: "white_bishop",
    4: "white_rook",
    5: "white_queen",
    6: "black_pawn",
    7: "black_knight",
    8: "black_bishop",
    9: "black_rook",
    10: "black_queen",
    11: "black_king",
}
OUTCOME_NAMES = {
    1: "checkmate",
    2: "extinction",
    3: "stalemate",
    4: "horde_fortress",
    5: "fifty_move",
    6: "fivefold_repetition",
}
FLAG_NAMES = {
    0: "best_capture",
    1: "best_check",
    2: "best_promotion",
    3: "best_en_passant",
    4: "best_castling",
    5: "played_differs",
    6: "royal_in_check",
}


class FormatError(ValueError):
    """Raised when a file violates the frozen HORDE_BIN_V1 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormatError(message)


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in document, f"duplicate manifest key {key}")
        document[key] = value
    return document


def parse_header(payload: bytes) -> dict[str, Any]:
    _require(len(payload) >= HEADER_SIZE, "file is shorter than the fixed header")
    _require(payload[:8] == MAGIC, "file magic is not HORDEBIN")
    version, header_size, manifest_length = struct.unpack_from("<HHI", payload, 8)
    _require(version == FORMAT_VERSION, f"unsupported format version {version}")
    _require(header_size == HEADER_SIZE, f"unexpected header size {header_size}")
    _require(
        0 < manifest_length <= HEADER_SIZE - 16,
        f"invalid manifest length {manifest_length}",
    )
    manifest_end = 16 + manifest_length
    raw_manifest = payload[16:manifest_end]
    try:
        manifest = json.loads(
            raw_manifest.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
        canonical = json.dumps(
            manifest,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as error:
        raise FormatError(f"manifest is not valid UTF-8 JSON: {error}") from error
    _require(isinstance(manifest, dict), "manifest root is not an object")
    _require(raw_manifest == canonical, "manifest is not canonical compact JSON")
    _require(
        all(byte == 0 for byte in payload[manifest_end:HEADER_SIZE]),
        "reserved header bytes are not zero",
    )
    _require(
        list(manifest) == [
            "schema",
            "schema_sha256",
            "format_version",
            "header_bytes",
            "record_bytes",
            "record_count",
            "byte_order",
            "source_commit",
            "source_dirty",
            "network",
            "book_sha256",
            "producer_sha256",
            "payload_sha256",
            "generation",
        ],
        "manifest fields are incomplete or out of order",
    )
    _require(manifest.get("schema") == SCHEMA_NAME, "manifest schema name mismatch")
    _require(
        manifest.get("schema_sha256") == SCHEMA_SHA256,
        "manifest schema SHA-256 mismatch",
    )
    _require(manifest.get("format_version") == FORMAT_VERSION, "manifest version mismatch")
    _require(manifest.get("header_bytes") == HEADER_SIZE, "manifest header size mismatch")
    _require(manifest.get("record_bytes") == RECORD_SIZE, "manifest record size mismatch")
    _require(manifest.get("byte_order") == "little", "manifest byte order mismatch")
    _require(
        type(manifest.get("record_count")) is int and manifest["record_count"] > 0,
        "manifest record count is invalid",
    )
    _require(
        isinstance(manifest.get("source_commit"), str)
        and re.fullmatch(r"[0-9a-fA-F]{40}", manifest["source_commit"])
        and manifest["source_commit"] != "0" * 40,
        "manifest source commit is invalid",
    )
    _require(manifest.get("source_dirty") is False, "manifest source is dirty")
    _require(
        isinstance(manifest.get("network"), dict)
        and list(manifest["network"]) == ["schema", "sha256"],
        "manifest network object mismatch",
    )
    _require(
        manifest.get("network", {}).get("schema") == "HORDETEST_HP_LEGACY_V1",
        "manifest network schema mismatch",
    )
    _require(
        manifest.get("network", {}).get("sha256") == RUN6B_SHA256,
        "manifest network SHA-256 mismatch",
    )
    _require(
        manifest.get("book_sha256") == "NONE"
        or bool(re.fullmatch(r"[0-9A-F]{64}", str(manifest.get("book_sha256")))),
        "manifest book SHA-256 is invalid",
    )
    _require(
        bool(re.fullmatch(r"[0-9A-F]{64}", str(manifest.get("producer_sha256")))),
        "manifest producer SHA-256 is invalid",
    )
    _require(
        bool(re.fullmatch(r"[0-9A-F]{64}", str(manifest.get("payload_sha256")))),
        "manifest payload SHA-256 is invalid",
    )
    generation = manifest.get("generation")
    generation_keys = [
        "requested_records",
        "seed",
        "threads",
        "hash_mb",
        "depth",
        "nodes",
        "random_move_min_ply",
        "random_move_max_ply",
        "random_move_count",
        "random_multi_pv",
        "random_multi_pv_diff",
        "write_min_ply",
        "write_max_ply",
        "max_game_ply",
        "opening_count",
    ]
    _require(
        isinstance(generation, dict) and list(generation) == generation_keys,
        "manifest generation object is incomplete or out of order",
    )
    _require(
        type(generation["requested_records"]) is int
        and generation["requested_records"] == manifest["record_count"],
        "manifest requested record count mismatch",
    )
    _require(
        isinstance(generation["seed"], str)
        and bool(re.fullmatch(r"[0-9]+", generation["seed"]))
        and int(generation["seed"]) > 0,
        "manifest seed is invalid",
    )
    numeric_generation = [
        generation[key] for key in generation_keys if key not in ("seed",)
    ]
    _require(
        all(type(value) is int for value in numeric_generation),
        "manifest generation values are not integers",
    )
    _require(
        generation["threads"] > 0
        and generation["hash_mb"] > 0
        and 0 < generation["depth"] < 246
        and generation["nodes"] >= 0
        and 0 <= generation["random_move_min_ply"]
        <= generation["random_move_max_ply"]
        and generation["random_move_count"] >= 0
        and generation["random_multi_pv"] >= 0
        and generation["random_multi_pv_diff"] >= 0
        and generation["write_min_ply"] >= 0
        and generation["write_min_ply"] < generation["write_max_ply"]
        < generation["max_game_ply"]
        and generation["opening_count"] > 0,
        "manifest generation values are outside the format domain",
    )
    return manifest


def decode_board(record: bytes) -> list[int]:
    board: list[int] = []
    for byte in record[:32]:
        board.extend((byte & 0x0F, byte >> 4))
    return board


def _piece_color(code: int) -> int | None:
    if 1 <= code <= 5:
        return 0
    if 6 <= code <= 11:
        return 1
    return None


def _validate_move(
    raw: int,
    board: list[int],
    side: int,
    ep_square: int,
    label: str,
) -> tuple[int, int]:
    _require(raw not in (0, 65), f"{label} is none or null")
    from_square = (raw >> 6) & 63
    to_square = raw & 63
    move_type = (raw >> 14) & 3
    promotion_bits = (raw >> 12) & 3
    piece = board[from_square]
    destination = board[to_square]
    _require(from_square != to_square, f"{label} has identical origin and destination")
    _require(_piece_color(piece) == side, f"{label} origin has the wrong color")
    if move_type != 3:  # Internal castling targets the friendly rook square.
        _require(
            _piece_color(destination) != side,
            f"{label} destination contains a friendly piece",
        )
    if move_type != 1:
        _require(promotion_bits == 0, f"{label} has stray promotion bits")
    from_rank = from_square // 8
    to_rank = to_square // 8
    from_file = from_square % 8
    to_file = to_square % 8
    if move_type == 0:
        _require(
            not (piece == 1 and to_rank == 7)
            and not (piece == 6 and to_rank == 0),
            f"{label} reaches the last rank without promotion",
        )
    if move_type == 1:
        _require(piece in (1, 6), f"{label} promotes a non-pawn")
        _require(
            (piece == 1 and to_rank == 7) or (piece == 6 and to_rank == 0),
            f"{label} does not reach the promotion rank",
        )
    elif move_type == 2:
        _require(piece in (1, 6) and destination == 0, f"{label} is not en passant")
        _require(
            to_square == ep_square
            and abs(to_file - from_file) == 1
            and (
                (piece == 1 and to_rank == from_rank + 1)
                or (piece == 6 and to_rank == from_rank - 1)
            ),
            f"{label} contradicts the en-passant state",
        )
    elif move_type == 3:
        _require(
            side == 1
            and piece == 11
            and destination == 9
            and from_square == 60
            and to_square in (56, 63),
            f"{label} is not registered Black castling",
        )
    return move_type, destination


def validate_record(record: bytes, index: int) -> dict[str, Any]:
    _require(len(record) == RECORD_SIZE, f"record {index} is truncated")
    board = decode_board(record)
    _require(all(code in PIECE_NAMES for code in board), f"record {index} has a reserved piece code")

    white_count = sum(1 for code in board if 1 <= code <= 5)
    black_count = sum(1 for code in board if 6 <= code <= 11)
    _require(1 <= white_count <= 36, f"record {index} has invalid White piece count")
    _require(1 <= black_count <= 16, f"record {index} has invalid Black piece count")
    _require(white_count + black_count <= 52, f"record {index} exceeds 52 pieces")
    _require(board.count(11) == 1, f"record {index} does not have exactly one Black king")
    for square, code in enumerate(board):
        rank = square // 8
        _require(not (code == 1 and rank == 7), f"record {index} has a White pawn on rank 8")
        _require(
            not (code == 6 and rank in (0, 7)),
            f"record {index} has a Black pawn on rank 1 or 8",
        )

    side, castling, ep_square, flags = record[32:36]
    _require(side in (0, 1), f"record {index} has invalid side to move")
    _require(castling & ~0x03 == 0, f"record {index} has reserved castling bits")
    _require(ep_square <= 64, f"record {index} has invalid en-passant square")
    _require(flags & 0x80 == 0, f"record {index} has reserved sample flags")
    if castling:
        _require(board[60] == 11, f"record {index} castling king is not on e8")
        _require(
            not (castling & 1) or board[63] == 9,
            f"record {index} kingside castling rook is missing",
        )
        _require(
            not (castling & 2) or board[56] == 9,
            f"record {index} queenside castling rook is missing",
        )
    if ep_square != 64:
        ep_rank = ep_square // 8
        ep_file = ep_square % 8
        if side == 0:
            expected_rank = 5
            captured_square = ep_square - 8
            attacker_offsets = (-9, -7)
            attacker, captured = 1, 6
        else:
            expected_rank = 2
            captured_square = ep_square + 8
            attacker_offsets = (7, 9)
            attacker, captured = 6, 1
        _require(
            ep_rank == expected_rank and board[captured_square] == captured,
            f"record {index} has an impossible en-passant target",
        )
        attackers = [
            ep_square + offset
            for offset in attacker_offsets
            if 0 <= ep_square + offset < 64
            and abs((ep_square + offset) % 8 - ep_file) == 1
        ]
        _require(
            any(board[square] == attacker for square in attackers),
            f"record {index} has no en-passant attacker",
        )

    rule50, game_ply, score, best_move, played_move, result, reason = struct.unpack_from(
        "<HHhHHbB", record, 36
    )
    _require(result in (-1, 0, 1), f"record {index} has invalid result")
    _require(reason in OUTCOME_NAMES, f"record {index} has invalid terminal reason")
    _require(
        (reason in (1, 2) and result in (-1, 1))
        or (reason in (3, 4, 5, 6) and result == 0),
        f"record {index} result contradicts its terminal reason",
    )
    best_type, best_destination = _validate_move(
        best_move, board, side, ep_square, f"record {index} best move"
    )
    _validate_move(
        played_move, board, side, ep_square, f"record {index} played move"
    )
    expected_capture = best_type == 2 or (
        best_type != 3 and _piece_color(best_destination) == 1 - side
    )
    _require(
        bool(flags & 1) == expected_capture,
        f"record {index} capture flag mismatch",
    )
    _require(
        bool(flags & (1 << 2)) == (best_type == 1)
        and bool(flags & (1 << 3)) == (best_type == 2)
        and bool(flags & (1 << 4)) == (best_type == 3),
        f"record {index} special-move flag mismatch",
    )
    _require(
        bool(flags & (1 << 5)) == (played_move != best_move),
        f"record {index} best-versus-played flag mismatch",
    )
    return {
        "board": board,
        "side": side,
        "castling": castling,
        "ep_square": ep_square,
        "flags": flags,
        "rule50": rule50,
        "game_ply": game_ply,
        "score": score,
        "best_move": best_move,
        "played_move": played_move,
        "result": result,
        "reason": reason,
        "white_count": white_count,
    }


def iter_records(payload: bytes) -> Iterator[bytes]:
    records = payload[HEADER_SIZE:]
    _require(len(records) % RECORD_SIZE == 0, "payload has a partial record")
    for offset in range(0, len(records), RECORD_SIZE):
        yield records[offset : offset + RECORD_SIZE]


def validate_file(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = path.read_bytes()
    manifest = parse_header(payload)
    record_count = (len(payload) - HEADER_SIZE) // RECORD_SIZE
    _require(
        len(payload) == HEADER_SIZE + record_count * RECORD_SIZE,
        "file length does not match fixed framing",
    )
    _require(manifest.get("record_count") == record_count, "manifest record count mismatch")
    actual_payload_sha = hashlib.sha256(payload[HEADER_SIZE:]).hexdigest().upper()
    _require(
        manifest.get("payload_sha256") == actual_payload_sha,
        "payload SHA-256 mismatch",
    )

    sides: Counter[str] = Counter()
    results: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    white_buckets: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    promoted_white = 0
    ep_states = 0
    black_castling_states = 0
    scores: list[int] = []

    for index, raw in enumerate(iter_records(payload)):
        record = validate_record(raw, index)
        sides["white" if record["side"] == 0 else "black"] += 1
        results[str(record["result"])] += 1
        reasons[OUTCOME_NAMES[record["reason"]]] += 1
        count = record["white_count"]
        bucket = "1-4" if count <= 4 else "5-12" if count <= 12 else "13-24" if count <= 24 else "25-36"
        white_buckets[bucket] += 1
        promoted_white += int(any(code in (2, 3, 4, 5) for code in record["board"]))
        ep_states += int(record["ep_square"] != 64)
        black_castling_states += int(record["castling"] != 0)
        scores.append(record["score"])
        for bit, name in FLAG_NAMES.items():
            flags[name] += int(bool(record["flags"] & (1 << bit)))

    summary = {
        "file": str(path.resolve()),
        "file_sha256": hashlib.sha256(payload).hexdigest().upper(),
        "payload_sha256": actual_payload_sha,
        "record_count": record_count,
        "sides": dict(sides),
        "results": dict(results),
        "outcome_reasons": dict(reasons),
        "white_piece_buckets": dict(white_buckets),
        "promoted_white_positions": promoted_white,
        "en_passant_positions": ep_states,
        "black_castling_positions": black_castling_states,
        "sample_flags": dict(flags),
        "score": {
            "minimum": min(scores) if scores else None,
            "maximum": max(scores) if scores else None,
            "mean": (sum(scores) / len(scores)) if scores else None,
        },
    }
    return manifest, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--expect-records", type=int)
    parser.add_argument("--expect-network-sha256")
    parser.add_argument("--expect-producer-sha256")
    parser.add_argument("--expect-book-sha256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.schema:
        observed = hashlib.sha256(args.schema.read_bytes()).hexdigest().upper()
        _require(observed == SCHEMA_SHA256, f"schema file SHA-256 mismatch: {observed}")

    manifest, summary = validate_file(args.file)
    if args.expect_records is not None:
        _require(summary["record_count"] == args.expect_records, "unexpected record count")
    if args.expect_network_sha256:
        _require(
            manifest["network"]["sha256"] == args.expect_network_sha256.upper(),
            "unexpected network SHA-256",
        )
    if args.expect_producer_sha256:
        _require(
            manifest["producer_sha256"] == args.expect_producer_sha256.upper(),
            "unexpected producer SHA-256",
        )
    if args.expect_book_sha256:
        _require(
            manifest["book_sha256"] == args.expect_book_sha256.upper(),
            "unexpected book SHA-256",
        )
    print(json.dumps({"manifest": manifest, "coverage": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FormatError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
