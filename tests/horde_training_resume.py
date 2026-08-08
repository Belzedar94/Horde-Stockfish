#!/usr/bin/env python3
"""Prove exact Horde trainer state across an interrupted/resumed run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import horde_compare_training_checkpoints as checkpoint_compare  # noqa: E402
import horde_training_control as control  # noqa: E402
from horde_bin_v1 import (  # noqa: E402
    HEADER_SIZE,
    MAGIC,
    RECORD_SIZE,
    RUN6B_SHA256,
    SCHEMA_SHA256,
)


def _record(index: int) -> bytes:
    board = [0] * 64
    source = index
    target = 40 + index
    board[source] = 2  # White knight; geometry is immaterial to the wire contract.
    board[60] = 11  # The unique Black king.
    packed_board = bytes(
        board[square] | (board[square + 1] << 4) for square in range(0, 64, 2)
    )
    move = (source << 6) | target
    state = bytes((0, 0, 64, 0))
    labels = struct.pack("<HHhHHbB", index, 0, index * 7 - 30, move, move, 0, 3)
    record = packed_board + state + labels
    if len(record) != RECORD_SIZE:
        raise AssertionError("synthetic HORDE_BIN_V1 record has the wrong size")
    return record


def _write_dataset(path: Path, *, first: int, count: int, book_sha256: str, seed: int) -> None:
    payload = b"".join(_record(index) for index in range(first, first + count))
    manifest = {
        "schema": "HORDE_BIN_V1",
        "schema_sha256": SCHEMA_SHA256,
        "format_version": 1,
        "header_bytes": HEADER_SIZE,
        "record_bytes": RECORD_SIZE,
        "record_count": count,
        "byte_order": "little",
        "source_commit": "1" * 40,
        "source_dirty": False,
        "network": {
            "schema": "HORDETEST_HP_LEGACY_V1",
            "sha256": RUN6B_SHA256,
        },
        "book_sha256": book_sha256,
        "producer_sha256": "2" * 64,
        "payload_sha256": hashlib.sha256(payload).hexdigest().upper(),
        "generation": {
            "requested_records": count,
            "seed": str(seed),
            "threads": 1,
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
            "opening_count": count,
        },
    }
    encoded = json.dumps(
        manifest,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    header = MAGIC + struct.pack("<HHI", 1, HEADER_SIZE, len(encoded)) + encoded
    header += bytes(HEADER_SIZE - len(header))
    path.write_bytes(header + payload)


def _write_split_receipt(
    path: Path,
    *,
    train_count: int,
    validation_count: int,
    train_book_sha256: str,
    validation_book_sha256: str,
) -> None:
    receipt = {
        "assignment": {
            "hash": "SHA-256",
            "horizontal_reflection_canonicalization": True,
            "integer": "first eight digest bytes, unsigned big-endian",
            "key": "synthetic horizontal-reflection canonical key",
            "modulus": 5,
            "validation_residue": 0,
        },
        "complete_partition": True,
        "disjoint_canonical_groups": True,
        "disjoint_position_keys": True,
        "schema": "HORDE_TRAINING_BOOK_SPLIT_V2",
        "source": {
            "bytes": 0,
            "canonical_groups": train_count + validation_count,
            "multi_record_groups": 0,
            "name": "synthetic.epd",
            "records": train_count + validation_count,
            "sha256": "C" * 64,
        },
        "train": {
            "bytes": 0,
            "records": train_count,
            "sha256": train_book_sha256,
        },
        "validation": {
            "bytes": 0,
            "records": validation_count,
            "sha256": validation_book_sha256,
        },
    }
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _arguments(
    train: Path,
    validation: Path,
    split_receipt: Path,
    output: Path,
    *,
    resume: Path | None = None,
    stop_after_steps: int | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        train=train,
        validation=validation,
        book_split_receipt=split_receipt,
        output=output,
        seed=2026080811,
        epochs=2,
        lambda_value=0.6,
        learning_rate=control.DEFAULT_LEARNING_RATE,
        scheduler_gamma=control.DEFAULT_SCHEDULER_GAMMA,
        batch_size=2,
        block_size=4,
        device="cpu",
        cpu_threads=1,
        resume=resume,
        stop_after_steps=stop_after_steps,
        allow_legacy_book_split_v1=False,
        allow_dirty=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="horde-training-resume-") as directory:
        root = Path(directory)
        train = root / "train.bin"
        validation = root / "validation.bin"
        split_receipt = root / "split-receipt.json"
        train_book_sha256 = "A" * 64
        validation_book_sha256 = "B" * 64
        _write_dataset(
            train,
            first=0,
            count=6,
            book_sha256=train_book_sha256,
            seed=101,
        )
        _write_dataset(
            validation,
            first=8,
            count=4,
            book_sha256=validation_book_sha256,
            seed=202,
        )
        _write_split_receipt(
            split_receipt,
            train_count=6,
            validation_count=4,
            train_book_sha256=train_book_sha256,
            validation_book_sha256=validation_book_sha256,
        )

        full = root / "full"
        partial = root / "partial"
        resumed = root / "resumed"
        full_receipt = control.train(_arguments(train, validation, split_receipt, full))
        partial_receipt = control.train(
            _arguments(
                train,
                validation,
                split_receipt,
                partial,
                stop_after_steps=2,
            )
        )
        if partial_receipt["run"]["complete"] is not False:
            raise AssertionError("partial trainer run was incorrectly marked complete")
        resumed_receipt = control.train(
            _arguments(
                train,
                validation,
                split_receipt,
                resumed,
                resume=partial / "checkpoint.pt",
            )
        )
        if resumed_receipt["run"]["complete"] is not True:
            raise AssertionError("resumed trainer run did not reach the target")

        full_checkpoint = checkpoint_compare.load(full / "checkpoint.pt")
        resumed_checkpoint = checkpoint_compare.load(resumed / "checkpoint.pt")
        full_semantic_sha = checkpoint_compare.compare_checkpoints(
            full_checkpoint,
            resumed_checkpoint,
        )
        if (full / "metrics.jsonl").read_bytes() != (resumed / "metrics.jsonl").read_bytes():
            raise AssertionError("resumed metrics are not byte-identical")

        for field in (
            "optimizer_steps",
            "samples_consumed",
            "sample_order_chain_sha256",
            "final_state_sha256",
            "stop_validation",
            "epochs_receipt",
        ):
            if full_receipt["run"][field] != resumed_receipt["run"][field]:
                raise AssertionError(f"resumed run field changed: {field}")

        print(
            "Horde trainer resume parity passed: "
            f"semantic_sha256={full_semantic_sha}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
