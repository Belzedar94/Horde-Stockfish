#!/usr/bin/env python3
"""Focused fail-closed checks for HORDE_BIN_V1 chunk-set assembly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import horde_bin_v1 as wire  # noqa: E402
import horde_training_chunk_set as chunk_set  # noqa: E402


TRAIN_BOOK = "A" * 64
VALIDATION_BOOK = "B" * 64
SOURCE_COMMIT = "1" * 40
PRODUCER_SHA256 = "2" * 64


def _record(index: int) -> bytes:
    board = [0] * 64
    side = index & 1
    result = (-1, 0, 1)[(index // 2) % 3]
    board[0] = 2
    board[8 + index % 40] = 4
    board[48 + (index // 40) % 8] = 9
    board[57] = 7
    board[60] = 11
    packed_board = bytes(
        board[square] | (board[square + 1] << 4) for square in range(0, 64, 2)
    )
    move = (0 << 6) | 7 if side == 0 else (57 << 6) | 56
    state = bytes((side, 0, 64, 0))
    score = result * 200 + (index * 137) % 1201 - 600
    reason = 3 if result == 0 else 1
    labels = struct.pack("<HHhHHbB", index % 100, side, score, move, move, result, reason)
    record = packed_board + state + labels
    if len(record) != wire.RECORD_SIZE:
        raise AssertionError("synthetic record has the wrong size")
    wire.validate_record(record, index)
    return record


def _write_contract(path: Path) -> None:
    contract = {
        "schema_name": chunk_set.SCALE_SCHEMA,
        "dependencies": {
            "teacher": {
                "source_commit": SOURCE_COMMIT,
                "producer_sha256": PRODUCER_SHA256,
                "network_schema": "HORDETEST_HP_LEGACY_V1",
                "network_sha256": wire.RUN6B_SHA256,
            },
            "labels": {
                "schema": wire.LABEL_CONTRACT_NAME,
                "schema_sha256": wire.LABEL_CONTRACT_SHA256,
            },
        },
        "openbench": {
            "campaign_id": "fixture-rank8-scale",
            "cohort": "fixture-v3-split",
        },
        "books": {
            "training": {"records": 3, "raw_sha256": TRAIN_BOOK},
            "validation": {"records": 2, "raw_sha256": VALIDATION_BOOK},
        },
        "generation": {
            "common": {
                "hash_mb": 16,
                "depth": 1,
                "nodes": 0,
                "random_move_min_ply": 1,
                "random_move_max_ply": 1,
                "random_move_count": 0,
                "random_multi_pv": 0,
                "random_multi_pv_diff": 0,
                "write_min_ply": 0,
                "write_max_ply": 1,
                "max_game_ply": 2,
            },
            "training": {
                "records": 6,
                "positions_per_chunk": 2,
                "chunk_count": 3,
                "base_seed": 1000,
            },
            "validation_candidate": {
                "records": 4,
                "positions_per_chunk": 2,
                "chunk_count": 2,
                "base_seed": 2000,
            },
        },
    }
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_chunk(
    path: Path,
    *,
    first: int,
    seed: int,
    threads: int,
    book_sha256: str = TRAIN_BOOK,
) -> list[bytes]:
    records = [_record(first), _record(first + 1)]
    payload = b"".join(records)
    manifest = {
        "schema": wire.SCHEMA_NAME,
        "schema_sha256": wire.SCHEMA_SHA256,
        "format_version": wire.FORMAT_VERSION,
        "header_bytes": wire.HEADER_SIZE,
        "record_bytes": wire.RECORD_SIZE,
        "record_count": len(records),
        "byte_order": "little",
        "source_commit": SOURCE_COMMIT,
        "source_dirty": False,
        "network": {
            "schema": "HORDETEST_HP_LEGACY_V1",
            "sha256": wire.RUN6B_SHA256,
        },
        "book_sha256": book_sha256,
        "producer_sha256": PRODUCER_SHA256,
        "payload_sha256": hashlib.sha256(payload).hexdigest().upper(),
        "label_contract": {
            "schema": wire.LABEL_CONTRACT_NAME,
            "schema_sha256": wire.LABEL_CONTRACT_SHA256,
        },
        "generation": {
            "requested_records": len(records),
            "seed": str(seed),
            "threads": threads,
            "hash_mb": 16,
            "depth": 1,
            "nodes": 0,
            "random_move_min_ply": 1,
            "random_move_max_ply": 1,
            "random_move_count": 0,
            "random_multi_pv": 0,
            "random_multi_pv_diff": 0,
            "write_min_ply": 0,
            "write_max_ply": 1,
            "max_game_ply": 2,
            "opening_count": 3,
        },
    }
    encoded = json.dumps(
        manifest,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    header = wire.MAGIC + struct.pack(
        "<HHI", wire.FORMAT_VERSION, wire.HEADER_SIZE, len(encoded)
    ) + encoded
    header += bytes(wire.HEADER_SIZE - len(header))
    path.write_bytes(header + payload)
    return records


def _expect_failure(callable_object: object, needle: str) -> None:
    try:
        callable_object()  # type: ignore[operator]
    except (chunk_set.ChunkSetError, wire.FormatError) as error:
        if needle not in str(error):
            raise AssertionError(f"expected {needle!r}, got {error!r}") from error
    else:
        raise AssertionError(f"expected failure containing {needle!r}")


def _rewrite_receipt(source: Path, destination: Path, mutate: object) -> None:
    document = json.loads(source.read_text(encoding="utf-8"))
    mutate(document)  # type: ignore[operator]
    destination.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    schema_sha256 = hashlib.sha256(
        (ROOT / "schemas" / "horde-training-chunk-set-v1.json").read_bytes()
    ).hexdigest().upper()
    if schema_sha256 != chunk_set.SCHEMA_SHA256:
        raise AssertionError("chunk-set schema identity drifted")

    with tempfile.TemporaryDirectory(prefix="horde-chunk-set-") as directory:
        root = Path(directory)
        contract = root / "campaign.json"
        _write_contract(contract)
        chunks = [root / f"chunk-{index}.bin" for index in range(3)]
        expected_records: list[bytes] = []
        expected_records.extend(_write_chunk(chunks[0], first=0, seed=1000, threads=1))
        expected_records.extend(_write_chunk(chunks[1], first=2, seed=1001, threads=4))
        expected_records.extend(_write_chunk(chunks[2], first=4, seed=1002, threads=1))

        receipt_path = root / "train-chunk-set.json"
        receipt = chunk_set.assemble_chunk_set(
            contract,
            "training",
            receipt_path,
            [chunks[2], chunks[0], chunks[1]],
        )
        if [entry["index"] for entry in receipt["chunks"]] != [0, 1, 2]:
            raise AssertionError("assembler did not canonicalize chunk order")
        if receipt["totals"]["threads_seen"] != [1, 4]:
            raise AssertionError("per-chunk thread inventory was collapsed or lost")
        logical = hashlib.sha256(b"".join(expected_records)).hexdigest().upper()
        if receipt["identity"]["logical_payload_sha256"] != logical:
            raise AssertionError("logical payload identity is not the concatenated record stream")

        verified = chunk_set.verify_chunk_set(receipt_path, contract)
        if verified["records"] != 6 or verified["chunks"] != 3:
            raise AssertionError("verified chunk-set dimensions drifted")
        with chunk_set.HordeChunkSetDataset(receipt_path, contract) as dataset:
            if len(dataset) != 6:
                raise AssertionError("chunk-set dataset length drifted")
            if [dataset.raw_record(index) for index in range(6)] != expected_records:
                raise AssertionError("global random access changed record order")
            first_payload = receipt["chunks"][0]["payload_sha256"]
            second_payload = receipt["chunks"][1]["payload_sha256"]
            if dataset.sample_identity(1) != (first_payload, 1):
                raise AssertionError("last record before a chunk boundary maps incorrectly")
            if dataset.sample_identity(2) != (second_payload, 0):
                raise AssertionError("first record after a chunk boundary maps incorrectly")
            batches = list(dataset.batches(4))
            if [len(batch) for batch in batches] != [4, 2]:
                raise AssertionError("logical batches were split at physical chunk boundaries")
            if any(batch.source_payload_sha256 != logical for batch in batches):
                raise AssertionError("batch identity is not the exact logical payload identity")

        duplicate_receipt = root / "duplicate.json"
        _expect_failure(
            lambda: chunk_set.assemble_chunk_set(
                contract,
                "training",
                duplicate_receipt,
                [chunks[0], chunks[0], chunks[2]],
            ),
            "path is duplicated",
        )

        gap_chunk = root / "gap.bin"
        _write_chunk(gap_chunk, first=6, seed=1004, threads=2)
        _expect_failure(
            lambda: chunk_set.assemble_chunk_set(
                contract,
                "training",
                root / "gap.json",
                [chunks[0], chunks[1], gap_chunk],
            ),
            "outside campaign range",
        )

        wrong_book = root / "wrong-book.bin"
        _write_chunk(
            wrong_book,
            first=6,
            seed=1002,
            threads=2,
            book_sha256=VALIDATION_BOOK,
        )
        _expect_failure(
            lambda: chunk_set.assemble_chunk_set(
                contract,
                "training",
                root / "wrong-book.json",
                [chunks[0], chunks[1], wrong_book],
            ),
            "common manifest identity drifted",
        )

        reordered = root / "reordered.json"
        _rewrite_receipt(
            receipt_path,
            reordered,
            lambda document: document["chunks"].reverse(),
        )
        _expect_failure(
            lambda: chunk_set.verify_chunk_set(reordered, contract),
            "not index ordered",
        )

        escaping = root / "escaping.json"
        _rewrite_receipt(
            receipt_path,
            escaping,
            lambda document: document["chunks"][0].__setitem__("path", "../outside.bin"),
        )
        _expect_failure(
            lambda: chunk_set.verify_chunk_set(escaping, contract),
            "escapes root",
        )

        original = chunks[2].read_bytes()
        corrupted = bytearray(original)
        corrupted[-1] ^= 0x01
        chunks[2].write_bytes(corrupted)
        _expect_failure(
            lambda: chunk_set.verify_chunk_set(receipt_path, contract),
            "payload SHA-256 mismatch",
        )
        chunks[2].write_bytes(original[:-1])
        _expect_failure(
            lambda: chunk_set.verify_chunk_set(receipt_path, contract),
            "file size",
        )
        chunks[2].write_bytes(original)
        chunk_set.verify_chunk_set(receipt_path, contract)

    print("Horde training chunk set: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
