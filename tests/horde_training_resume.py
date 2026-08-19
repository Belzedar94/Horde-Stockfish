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
import horde_training_decoder as decoder  # noqa: E402
import horde_wdl as wdl  # noqa: E402
from horde_bin_v1 import (  # noqa: E402
    HEADER_SIZE,
    LABEL_CONTRACT_NAME,
    LABEL_CONTRACT_SHA256,
    MAGIC,
    RECORD_SIZE,
    RUN6B_SHA256,
    SCHEMA_SHA256,
)


def _record(index: int) -> bytes:
    board = [0] * 64
    side = index & 1
    result = (-1, 0, 1)[(index // 2) % 3]
    board[0] = 2  # White knight.
    board[8 + index % 40] = 4  # Unique White-rook component of the physical key.
    board[48 + (index // 40) % 8] = 9  # Unique Black-rook component.
    board[57] = 7  # Black knight.
    board[60] = 11  # The unique Black king.
    packed_board = bytes(
        board[square] | (board[square + 1] << 4) for square in range(0, 64, 2)
    )
    move = (0 << 6) | 7 if side == 0 else (57 << 6) | 56
    state = bytes((side, 0, 64, 0))
    # Keep the three outcome distributions overlapping while giving both
    # side-to-move fits the positive score/result relation required by the
    # frozen Davidson contract.
    score = result * 200 + (index * 137) % 1201 - 600
    reason = 3 if result == 0 else 1
    labels = struct.pack(
        "<HHhHHbB", index % 100, side, score, move, move, result, reason
    )
    record = packed_board + state + labels
    if len(record) != RECORD_SIZE:
        raise AssertionError("synthetic HORDE_BIN_V1 record has the wrong size")
    return record


def _write_dataset(
    path: Path,
    *,
    first: int,
    count: int,
    book_sha256: str,
    seed: int,
    opening_count: int | None = None,
) -> None:
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
        "label_contract": {
            "schema": LABEL_CONTRACT_NAME,
            "schema_sha256": LABEL_CONTRACT_SHA256,
        },
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
            "opening_count": count if opening_count is None else opening_count,
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


def _write_wdl_calibration(
    path: Path,
    train: Path,
    *,
    software_commit: str = "3" * 40,
) -> None:
    with decoder.HordeBinV1Dataset(train) as dataset:
        aggregated = wdl.aggregate_labels(dataset)
        manifest = dataset.manifest
        source = {
            "training_file": {
                "name": train.name,
                "sha256": dataset.file_sha256,
                "payload_sha256": manifest["payload_sha256"],
                "manifest_sha256": dataset.manifest_sha256,
                "records": len(dataset),
            },
            "teacher": {
                "source_commit": manifest["source_commit"],
                "producer_sha256": manifest["producer_sha256"],
                "network": manifest["network"],
                "label_contract": manifest["label_contract"],
            },
            "software": {
                "commit": software_commit,
                "dirty": False,
                "python": "3.12.0",
                "implementation": "CPython",
            },
        }
    path.write_bytes(wdl.canonical_json(wdl.build_artifact(aggregated, source)))


def _arguments(
    train: Path,
    validation: Path,
    split_receipt: Path,
    wdl_calibration: Path,
    output: Path,
    *,
    architecture: str = control.LEGACY_ARCHITECTURE,
    resume: Path | None = None,
    stop_after_steps: int | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        train=train,
        validation=validation,
        book_split_receipt=split_receipt,
        wdl_calibration=wdl_calibration,
        architecture=architecture,
        output=output,
        seed=2026080811,
        epochs=2,
        lambda_value=0.6,
        learning_rate=control.DEFAULT_LEARNING_RATE,
        scheduler_gamma=control.DEFAULT_SCHEDULER_GAMMA,
        batch_size=64,
        block_size=128,
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
        wdl_calibration = root / "wdl-calibration.json"
        train_book_sha256 = "A" * 64
        validation_book_sha256 = "B" * 64
        _write_dataset(
            train,
            first=0,
            count=192,
            book_sha256=train_book_sha256,
            seed=101,
        )
        _write_dataset(
            validation,
            first=192,
            count=96,
            book_sha256=validation_book_sha256,
            seed=202,
        )
        _write_split_receipt(
            split_receipt,
            train_count=192,
            validation_count=96,
            train_book_sha256=train_book_sha256,
            validation_book_sha256=validation_book_sha256,
        )
        _write_wdl_calibration(wdl_calibration, train)

        other_train = root / "other-train.bin"
        other_calibration = root / "other-wdl-calibration.json"
        _write_dataset(
            other_train,
            first=384,
            count=192,
            book_sha256=train_book_sha256,
            seed=303,
        )
        _write_wdl_calibration(other_calibration, other_train)
        mismatched_dataset_output = root / "mismatched-dataset"
        try:
            control.train(
                _arguments(
                    train,
                    validation,
                    split_receipt,
                    other_calibration,
                    mismatched_dataset_output,
                )
            )
        except control.TrainingError as error:
            if "exact training dataset" not in str(error):
                raise
        else:
            raise AssertionError("trainer accepted calibration from another dataset")
        if mismatched_dataset_output.exists():
            raise AssertionError("calibration mismatch created a partial output directory")

        full = root / "full"
        partial = root / "partial"
        resumed = root / "resumed"
        full_receipt = control.train(
            _arguments(train, validation, split_receipt, wdl_calibration, full)
        )
        for role in ("train", "validation"):
            decoder_receipt = full_receipt["data"]["decoder"][role]
            if (
                decoder_receipt["schema"] != "HORDE_TRAINING_DECODER_V2"
                or decoder_receipt["row_counts"]["global_contextual_total"] != 0
                or not decoder_receipt["feature_stream_sha256"]["unreflected"]
                or not decoder_receipt["feature_stream_sha256"]["horizontal_reflection"]
            ):
                raise AssertionError(f"{role} decoder receipt is incomplete")
        partial_receipt = control.train(
            _arguments(
                train,
                validation,
                split_receipt,
                wdl_calibration,
                partial,
                stop_after_steps=2,
            )
        )
        if partial_receipt["run"]["complete"] is not False:
            raise AssertionError("partial trainer run was incorrectly marked complete")

        alternate_calibration = root / "alternate-wdl-calibration.json"
        _write_wdl_calibration(
            alternate_calibration,
            train,
            software_commit="4" * 40,
        )
        mismatched_resume_output = root / "mismatched-resume"
        try:
            control.train(
                _arguments(
                    train,
                    validation,
                    split_receipt,
                    alternate_calibration,
                    mismatched_resume_output,
                    resume=partial / "checkpoint.pt",
                )
            )
        except control.TrainingError as error:
            if "resume settings mismatch" not in str(error):
                raise
        else:
            raise AssertionError("resume accepted a different calibration artifact")
        if mismatched_resume_output.exists():
            raise AssertionError("resume calibration mismatch created a partial output directory")

        resumed_receipt = control.train(
            _arguments(
                train,
                validation,
                split_receipt,
                wdl_calibration,
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

        v2_full = root / "v2-64x192-full"
        v2_partial = root / "v2-64x192-partial"
        v2_resumed = root / "v2-64x192-resumed"
        v2_full_receipt = control.train(
            _arguments(
                train,
                validation,
                split_receipt,
                wdl_calibration,
                v2_full,
                architecture="v2-64x192",
            )
        )
        v2_partial_receipt = control.train(
            _arguments(
                train,
                validation,
                split_receipt,
                wdl_calibration,
                v2_partial,
                architecture="v2-64x192",
                stop_after_steps=2,
            )
        )
        if v2_partial_receipt["run"]["complete"] is not False:
            raise AssertionError("partial V2 trainer run was incorrectly marked complete")

        wrong_architecture_output = root / "wrong-v2-architecture-resume"
        try:
            control.train(
                _arguments(
                    train,
                    validation,
                    split_receipt,
                    wdl_calibration,
                    wrong_architecture_output,
                    architecture="v2-c1-abs64x192",
                    resume=v2_partial / "checkpoint.pt",
                )
            )
        except control.TrainingError as error:
            if "architecture mismatch" not in str(error):
                raise
        else:
            raise AssertionError("V2 trainer resumed a checkpoint from another architecture")
        if wrong_architecture_output.exists():
            raise AssertionError(
                "wrong-architecture V2 resume created a partial output directory"
            )

        v2_resumed_receipt = control.train(
            _arguments(
                train,
                validation,
                split_receipt,
                wdl_calibration,
                v2_resumed,
                architecture="v2-64x192",
                resume=v2_partial / "checkpoint.pt",
            )
        )
        v2_full_checkpoint = checkpoint_compare.load(v2_full / "checkpoint.pt")
        v2_resumed_checkpoint = checkpoint_compare.load(v2_resumed / "checkpoint.pt")
        v2_semantic_sha = checkpoint_compare.compare_checkpoints(
            v2_full_checkpoint,
            v2_resumed_checkpoint,
        )
        if (v2_full / "metrics.jsonl").read_bytes() != (
            v2_resumed / "metrics.jsonl"
        ).read_bytes():
            raise AssertionError("resumed V2 metrics are not byte-identical")
        if v2_full_receipt["schema"] != control.V2_TRAINING_SCHEMA:
            raise AssertionError("V2 trainer emitted the legacy receipt schema")
        if v2_full_receipt["architecture"]["schema"] != "V2_BASE_P0_64X192":
            raise AssertionError("V2 trainer emitted the wrong architecture schema")
        for field in (
            "optimizer_steps",
            "samples_consumed",
            "sample_order_chain_sha256",
            "final_state_sha256",
            "stop_validation",
            "epochs_receipt",
        ):
            if v2_full_receipt["run"][field] != v2_resumed_receipt["run"][field]:
                raise AssertionError(f"resumed V2 run field changed: {field}")

        v2_wide = root / "v2-128x128-full"
        v2_wide_receipt = control.train(
            _arguments(
                train,
                validation,
                split_receipt,
                wdl_calibration,
                v2_wide,
                architecture="v2-128x128",
            )
        )
        if (
            v2_wide_receipt["architecture"]["schema"] != "V2_BASE_P0_128X128"
            or not v2_wide_receipt["run"]["complete"]
        ):
            raise AssertionError("wide V2 training control did not complete correctly")

        c1_absolute = root / "v2-c1-abs64x192-full"
        c1_receipt = control.train(
            _arguments(
                train,
                validation,
                split_receipt,
                wdl_calibration,
                c1_absolute,
                architecture="v2-c1-abs64x192",
            )
        )
        if (
            c1_receipt["architecture"]["schema"]
            != "V2_C1_ABS_NONKING_64X192"
            or c1_receipt["architecture"]["domains"][0]["name"]
            != "absolute_nonking"
            or not c1_receipt["run"]["complete"]
        ):
            raise AssertionError("C1 absolute-content training control did not complete")

        c1_rank8 = root / "v2-c1-rank8-64x192-full"
        c1_rank8_receipt = control.train(
            _arguments(
                train,
                validation,
                split_receipt,
                wdl_calibration,
                c1_rank8,
                architecture="v2-c1-rank8-64x192",
            )
        )
        if (
            c1_rank8_receipt["architecture"]["schema"]
            != "V2_C1_ROYAL_RANK8_64X192"
            or c1_rank8_receipt["architecture"]["domains"][0]["name"]
            != "royal_rank8"
            or not c1_rank8_receipt["run"]["complete"]
        ):
            raise AssertionError("C1 R8 training control did not complete")

        print(
            "Horde trainer resume parity passed: "
            f"legacy_sha256={full_semantic_sha} v2_sha256={v2_semantic_sha}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
