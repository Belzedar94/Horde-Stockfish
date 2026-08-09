#!/usr/bin/env python3
"""Fail-closed tests for the C1 selected validation-role contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import horde_training_resume as fixtures  # noqa: E402
import horde_training_control as training_control  # noqa: E402
import horde_training_selected_role as selected_role  # noqa: E402
from horde_bin_v1 import (  # noqa: E402
    HEADER_SIZE,
    LABEL_CONTRACT_NAME,
    LABEL_CONTRACT_SHA256,
    MAGIC,
    RECORD_SIZE,
    RUN6B_SHA256,
    SCHEMA_SHA256,
)


SOURCE = {"commit": "a" * 40, "dirty": False}
TRAIN_BOOK = "A" * 64
VALIDATION_BOOK = "B" * 64


def _write_records(path: Path, records: list[bytes], *, seed: int) -> None:
    payload = b"".join(records)
    manifest = {
        "schema": "HORDE_BIN_V1",
        "schema_sha256": SCHEMA_SHA256,
        "format_version": 1,
        "header_bytes": HEADER_SIZE,
        "record_bytes": RECORD_SIZE,
        "record_count": len(records),
        "byte_order": "little",
        "source_commit": "1" * 40,
        "source_dirty": False,
        "network": {
            "schema": "HORDETEST_HP_LEGACY_V1",
            "sha256": RUN6B_SHA256,
        },
        "book_sha256": VALIDATION_BOOK,
        "producer_sha256": "2" * 64,
        "payload_sha256": hashlib.sha256(payload).hexdigest().upper(),
        "label_contract": {
            "schema": LABEL_CONTRACT_NAME,
            "schema_sha256": LABEL_CONTRACT_SHA256,
        },
        "generation": {
            "requested_records": len(records),
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
            "opening_count": len(records),
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


def _change_rule50(record: bytes, value: int) -> bytes:
    changed = bytearray(record)
    struct.pack_into("<H", changed, 36, value)
    return bytes(changed)


def _change_score(record: bytes, value: int) -> bytes:
    changed = bytearray(record)
    struct.pack_into("<h", changed, 40, value)
    return bytes(changed)


def _black_h8_record(record: bytes, castling: int) -> bytes:
    board: list[int] = []
    for byte in record[:32]:
        board.extend((byte & 0x0F, byte >> 4))
    for square, piece in enumerate(board):
        if piece == 9:
            board[square] = 0
            break
    board[63] = 9
    changed = bytearray(record)
    changed[:32] = bytes(
        board[square] | (board[square + 1] << 4)
        for square in range(0, 64, 2)
    )
    changed[33] = castling
    return bytes(changed)


def _expect_failure(callable_object: object, needle: str) -> None:
    try:
        callable_object()
    except selected_role.SelectedRoleError as error:
        if needle not in str(error):
            raise AssertionError(f"unexpected selected-role error: {error}") from error
    else:
        raise AssertionError(f"selected role unexpectedly accepted invalid input: {needle}")


def main() -> int:
    contract, contract_sha = selected_role.load_contract()
    if contract["schema_name"] != selected_role.CONTRACT_SCHEMA:
        raise AssertionError("selected-role contract schema drifted")
    if contract_sha != selected_role.CONTRACT_SHA256:
        raise AssertionError("selected-role contract hash drifted")

    with tempfile.TemporaryDirectory(prefix="horde-selected-role-") as directory:
        root = Path(directory)
        train = root / "train.bin"
        candidate = root / "candidate.bin"
        candidate_labels = root / "candidate-labels.bin"
        fixtures._write_dataset(
            train,
            first=0,
            count=192,
            book_sha256=TRAIN_BOOK,
            seed=101,
        )

        royal_base = _black_h8_record(fixtures._record(300), 0)
        records = [
            fixtures._record(0),
            _change_rule50(fixtures._record(0), 999),
            royal_base,
            _black_h8_record(fixtures._record(300), 1),
            royal_base,
            fixtures._record(301),
            fixtures._record(302),
            fixtures._record(303),
            fixtures._record(304),
        ]
        _write_records(candidate, records, seed=202)
        changed_labels = list(records)
        changed_labels[5] = _change_score(changed_labels[5], 1234)
        _write_records(candidate_labels, changed_labels, seed=203)

        output = root / "selected"
        receipt = selected_role.create_selected_role(
            train,
            candidate,
            output,
            _allow_fixture=True,
            _target_records=4,
            _source_override=SOURCE,
        )
        if receipt["selection"]["rejection_reason_masks"] != {
            "1": 1,
            "3": 1,
            "8": 1,
            "12": 1,
        }:
            raise AssertionError("selected-role rejection masks drifted")
        if receipt["selection"]["cutoff_candidate_index"] != 7:
            raise AssertionError("selected-role cutoff drifted")
        with selected_role.SelectedRoleDataset(output / "receipt.json") as dataset:
            if dataset.source_indices != (2, 5, 6, 7):
                raise AssertionError("selected-role first-eligible order drifted")
            if len(dataset) != 4 or len(tuple(dataset.batches(3))) != 2:
                raise AssertionError("selected-role adapter framing drifted")

        verification = selected_role.verify_selected_role(
            train,
            candidate,
            output / "receipt.json",
            _allow_fixture=True,
            _source_override=SOURCE,
        )
        if verification["claims"]["canonical_selection_recomputed"] is not True:
            raise AssertionError("selected role was not independently recomputed")

        split = root / "book-split.json"
        calibration = root / "wdl-calibration.json"
        fixtures._write_split_receipt(
            split,
            train_count=192,
            validation_count=len(records),
            train_book_sha256=TRAIN_BOOK,
            validation_book_sha256=VALIDATION_BOOK,
        )
        fixtures._write_wdl_calibration(calibration, train)
        training_output = root / "training"
        training_receipt = training_control.train(
            argparse.Namespace(
                train=train,
                validation=output / "receipt.json",
                validation_selected_role=True,
                validation_candidate=candidate,
                selected_role_fixture=True,
                selected_role_source_override=SOURCE,
                book_split_receipt=split,
                wdl_calibration=calibration,
                architecture="v2-c1-abs64x192",
                output=training_output,
                seed=2026080811,
                epochs=1,
                lambda_value=0.6,
                learning_rate=training_control.DEFAULT_LEARNING_RATE,
                scheduler_gamma=training_control.DEFAULT_SCHEDULER_GAMMA,
                batch_size=64,
                block_size=128,
                device="cpu",
                cpu_threads=1,
                resume=None,
                stop_after_steps=None,
                allow_legacy_book_split_v1=False,
                allow_dirty=True,
            )
        )
        if training_receipt["data"]["validation_file"] != verification["selected_role"]:
            raise AssertionError("trainer did not bind the selected-role identity")
        if training_receipt["run"]["initial_validation"]["samples"] != 4:
            raise AssertionError("trainer did not evaluate the complete selected role")

        label_output = root / "selected-labels"
        label_receipt = selected_role.create_selected_role(
            train,
            candidate_labels,
            label_output,
            _allow_fixture=True,
            _target_records=4,
            _source_override=SOURCE,
        )
        if (
            label_receipt["selection"]["selected_indices"]["sha256"]
            != receipt["selection"]["selected_indices"]["sha256"]
        ):
            raise AssertionError("score-only mutation changed the label-blind selected indices")
        if (
            label_receipt["materialized_output"]["payload_sha256"]
            == receipt["materialized_output"]["payload_sha256"]
        ):
            raise AssertionError("score-only mutation did not change materialized provenance")

        original_indices = (output / selected_role.INDEX_FILENAME).read_bytes()
        tampered = bytearray(original_indices)
        tampered[-1] ^= 1
        (output / selected_role.INDEX_FILENAME).write_bytes(tampered)
        _expect_failure(
            lambda: selected_role.SelectedRoleDataset(output / "receipt.json"),
            "selected index hash",
        )
        (output / selected_role.INDEX_FILENAME).write_bytes(original_indices)

        cherry_picked = root / "cherry-picked"
        shutil.copytree(output, cherry_picked)
        cherry_indices = (2, 5, 6, 8)
        cherry_index_payload = b"".join(
            struct.pack("<Q", index) for index in cherry_indices
        )
        cherry_records = b"".join(records[index] for index in cherry_indices)
        (cherry_picked / selected_role.INDEX_FILENAME).write_bytes(cherry_index_payload)
        (cherry_picked / selected_role.RECORDS_FILENAME).write_bytes(cherry_records)
        cherry_receipt_path = cherry_picked / selected_role.RECEIPT_FILENAME
        cherry_receipt = json.loads(cherry_receipt_path.read_text(encoding="ascii"))
        cherry_receipt["selection"]["records_examined"] = 9
        cherry_receipt["selection"]["cutoff_candidate_index"] = 8
        cherry_receipt["selection"]["rejected_records"] = 5
        cherry_receipt["selection"]["selected_indices"]["sha256"] = hashlib.sha256(
            cherry_index_payload
        ).hexdigest().upper()
        materialized_sha = hashlib.sha256(cherry_records).hexdigest().upper()
        cherry_receipt["materialized_output"]["sha256"] = materialized_sha
        cherry_receipt["materialized_output"]["payload_sha256"] = materialized_sha
        order_digest = hashlib.sha256()
        for index in cherry_indices:
            order_digest.update(struct.pack("<Q", index))
            order_digest.update(records[index])
        cherry_receipt["materialized_output"][
            "record_order_sha256"
        ] = order_digest.hexdigest().upper()
        cherry_receipt_path.write_bytes(selected_role._canonical_json(cherry_receipt))
        _expect_failure(
            lambda: selected_role.verify_selected_role(
                train,
                candidate,
                cherry_receipt_path,
                _allow_fixture=True,
                _source_override=SOURCE,
            ),
            "canonical first-eligible result",
        )

        insufficient = root / "insufficient"
        _expect_failure(
            lambda: selected_role.create_selected_role(
                train,
                candidate,
                insufficient,
                _allow_fixture=True,
                _target_records=6,
                _source_override=SOURCE,
            ),
            "candidate exhausted",
        )
        if (insufficient / selected_role.RECEIPT_FILENAME).exists():
            raise AssertionError("failed selection left an eligible receipt")

    print(
        "Horde selected-role contract passed: dual-key train exclusion, "
        "validation deduplication, label blindness and fail-closed artifacts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
