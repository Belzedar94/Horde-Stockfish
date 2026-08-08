#!/usr/bin/env python3
"""Split a Horde EPD into deterministic, position-disjoint training books."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence


SCHEMA_V1 = "HORDE_TRAINING_BOOK_SPLIT_V1"
SCHEMA_V2 = "HORDE_TRAINING_BOOK_SPLIT_V2"


class SplitError(ValueError):
    """Raised when a source book or requested split violates the contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitError(message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _mirror_board(board: str) -> str:
    mirrored: list[str] = []
    ranks = board.split("/")
    _require(len(ranks) == 8, "FEN board does not contain eight ranks")
    for rank in ranks:
        expanded: list[str] = []
        for symbol in rank:
            if symbol.isdigit():
                count = int(symbol)
                _require(1 <= count <= 8, f"invalid empty-square run {symbol}")
                expanded.extend("." for _ in range(count))
            else:
                _require(symbol in "PNBRQKpnbrqk", f"invalid Horde FEN piece {symbol}")
                expanded.append(symbol)
        _require(len(expanded) == 8, f"FEN rank {rank} does not contain eight files")

        compressed: list[str] = []
        empty = 0
        for symbol in reversed(expanded):
            if symbol == ".":
                empty += 1
            else:
                if empty:
                    compressed.append(str(empty))
                    empty = 0
                compressed.append(symbol)
        if empty:
            compressed.append(str(empty))
        mirrored.append("".join(compressed))
    return "/".join(mirrored)


def horizontal_mirror_position_key(position_key: str) -> str:
    fields = position_key.split(" ")
    _require(len(fields) == 4, "position key does not contain four FEN fields")
    board, side, castling, ep_square = fields
    _require(side in ("w", "b"), f"invalid FEN side to move {side}")

    if castling == "-":
        mirrored_castling = "-"
    else:
        _require(
            len(set(castling)) == len(castling)
            and all(symbol in "KQkq" for symbol in castling),
            f"invalid FEN castling field {castling}",
        )
        mapping = {"K": "Q", "Q": "K", "k": "q", "q": "k"}
        reflected = {mapping[symbol] for symbol in castling}
        mirrored_castling = "".join(
            symbol for symbol in "KQkq" if symbol in reflected
        )

    if ep_square == "-":
        mirrored_ep = "-"
    else:
        _require(
            len(ep_square) == 2
            and "a" <= ep_square[0] <= "h"
            and "1" <= ep_square[1] <= "8",
            f"invalid FEN en-passant field {ep_square}",
        )
        mirrored_ep = chr(ord("h") - (ord(ep_square[0]) - ord("a"))) + ep_square[1]
    return " ".join((_mirror_board(board), side, mirrored_castling, mirrored_ep))


def canonical_horizontal_position_key(position_key: str) -> str:
    mirrored = horizontal_mirror_position_key(position_key)
    return min(position_key, mirrored)


def _canonical_lines(
    payload: bytes,
    canonical_horizontal: bool,
) -> list[tuple[bytes, str, str, str]]:
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise SplitError(f"source book is not ASCII: {error}") from error

    entries: list[tuple[bytes, str, str, str]] = []
    seen_keys: dict[str, int] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = " ".join(raw_line.split())
        if not line:
            continue
        fields = line.split(" ")
        _require(len(fields) >= 4, f"line {line_number} does not contain four FEN fields")
        position_key = " ".join(fields[:4])
        previous = seen_keys.get(position_key)
        _require(
            previous is None,
            f"line {line_number} duplicates the physical position from line {previous}",
        )
        seen_keys[position_key] = line_number
        assignment_key = (
            canonical_horizontal_position_key(position_key)
            if canonical_horizontal
            else position_key
        )
        digest = hashlib.sha256(assignment_key.encode("ascii")).digest()
        entries.append((digest, assignment_key, position_key, line))

    _require(entries, "source book is empty")
    return entries


def _payload(entries: list[tuple[bytes, str, str, str]]) -> bytes:
    ordered = sorted(entries, key=lambda entry: (entry[0], entry[1], entry[2], entry[3]))
    return ("\n".join(entry[3] for entry in ordered) + "\n").encode("ascii")


def split_book(
    source: Path,
    modulus: int,
    validation_residue: int,
    canonical_horizontal: bool = True,
) -> tuple[bytes, bytes, dict[str, object]]:
    _require(modulus >= 2, "split modulus must be at least two")
    _require(0 <= validation_residue < modulus, "validation residue is outside the modulus")
    source_payload = source.expanduser().resolve().read_bytes()
    entries = _canonical_lines(source_payload, canonical_horizontal)

    train: list[tuple[bytes, str, str, str]] = []
    validation: list[tuple[bytes, str, str, str]] = []
    for entry in entries:
        residue = int.from_bytes(entry[0][:8], "big") % modulus
        (validation if residue == validation_residue else train).append(entry)

    _require(train, "training split is empty")
    _require(validation, "validation split is empty")
    train_keys = {entry[2] for entry in train}
    validation_keys = {entry[2] for entry in validation}
    _require(train_keys.isdisjoint(validation_keys), "training and validation positions overlap")
    _require(len(train_keys | validation_keys) == len(entries), "split lost a source position")
    train_groups = {entry[1] for entry in train}
    validation_groups = {entry[1] for entry in validation}
    _require(
        train_groups.isdisjoint(validation_groups),
        "a canonical opening group crosses training and validation",
    )
    canonical_groups = {entry[1] for entry in entries}
    grouped_records: dict[str, int] = {}
    for entry in entries:
        grouped_records[entry[1]] = grouped_records.get(entry[1], 0) + 1

    train_payload = _payload(train)
    validation_payload = _payload(validation)
    receipt = {
        "schema": SCHEMA_V2 if canonical_horizontal else SCHEMA_V1,
        "assignment": {
            "key": (
                "lexicographically smaller of the first four normalized FEN fields "
                "and their horizontal reflection"
                if canonical_horizontal
                else "first four normalized FEN fields"
            ),
            "hash": "SHA-256",
            "integer": "first eight digest bytes, unsigned big-endian",
            "modulus": modulus,
            "validation_residue": validation_residue,
            "horizontal_reflection_canonicalization": canonical_horizontal,
        },
        "source": {
            "name": source.name,
            "records": len(entries),
            "canonical_groups": len(canonical_groups),
            "multi_record_groups": sum(count > 1 for count in grouped_records.values()),
            "bytes": len(source_payload),
            "sha256": _sha256(source_payload),
        },
        "train": {
            "records": len(train),
            "bytes": len(train_payload),
            "sha256": _sha256(train_payload),
        },
        "validation": {
            "records": len(validation),
            "bytes": len(validation_payload),
            "sha256": _sha256(validation_payload),
        },
        "disjoint_position_keys": True,
        "disjoint_canonical_groups": True,
        "complete_partition": True,
    }
    return train_payload, validation_payload, receipt


def _write_exclusive(path: Path, payload: bytes) -> None:
    resolved = path.expanduser().resolve()
    _require(resolved.parent.is_dir(), f"output directory does not exist: {resolved.parent}")
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            resolved.unlink(missing_ok=True)
        finally:
            raise


def write_split(
    source: Path,
    train_output: Path,
    validation_output: Path,
    receipt_output: Path,
    modulus: int,
    validation_residue: int,
    canonical_horizontal: bool = True,
) -> dict[str, object]:
    destinations = [path.expanduser().resolve() for path in (train_output, validation_output, receipt_output)]
    _require(len(set(destinations)) == len(destinations), "split outputs must be distinct paths")
    _require(all(not path.exists() for path in destinations), "a split output already exists")
    train, validation, receipt = split_book(
        source,
        modulus,
        validation_residue,
        canonical_horizontal,
    )
    receipt_payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")

    created: list[Path] = []
    try:
        for path, payload in zip(destinations, (train, validation, receipt_payload), strict=True):
            _write_exclusive(path, payload)
            created.append(path)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--modulus", type=int, default=5)
    parser.add_argument("--validation-residue", type=int, default=0)
    parser.add_argument(
        "--legacy-exact-key-v1",
        action="store_true",
        help="reproduce V1 without horizontal-reflection grouping",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = write_split(
        args.source,
        args.train_output,
        args.validation_output,
        args.receipt_output,
        args.modulus,
        args.validation_residue,
        not args.legacy_exact_key_v1,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SplitError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
