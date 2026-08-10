#!/usr/bin/env python3
"""Fail-closed tests for the fresh C3 confirmation-role selector."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import horde_training_resume as fixtures  # noqa: E402
import horde_v2_c3_confirmation_role as confirmation  # noqa: E402
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
EXCLUSION_ONE_BOOK = "A" * 64
EXCLUSION_TWO_BOOK = "B" * 64
CANDIDATE_BOOK = "C" * 64


def _write_records(path: Path, records: list[bytes], *, seed: int, book: str) -> None:
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
        "network": {"schema": "HORDETEST_HP_LEGACY_V1", "sha256": RUN6B_SHA256},
        "book_sha256": book,
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


def _change_score(record: bytes, value: int) -> bytes:
    changed = bytearray(record)
    struct.pack_into("<h", changed, 40, value)
    return bytes(changed)


def _expect_failure(callable_object: object, needle: str) -> None:
    try:
        callable_object()
    except confirmation.ConfirmationRoleError as error:
        if needle not in str(error):
            raise AssertionError(f"unexpected confirmation-role error: {error}") from error
    else:
        raise AssertionError(f"confirmation role accepted invalid input: {needle}")


def main() -> int:
    contract, digest = confirmation.load_contract()
    if contract["schema_name"] != confirmation.CONTRACT_SCHEMA:
        raise AssertionError("C3 confirmation contract schema drifted")
    if digest != confirmation.CONTRACT_SHA256:
        raise AssertionError("C3 confirmation contract hash drifted")

    with tempfile.TemporaryDirectory(prefix="horde-c3-confirmation-") as directory:
        root = Path(directory)
        exclusion_one = root / "train.bin"
        exclusion_two = root / "tuning-validation.bin"
        candidate = root / "candidate.bin"
        candidate_labels = root / "candidate-labels.bin"
        _write_records(
            exclusion_one,
            [fixtures._record(index) for index in range(6)],
            seed=101,
            book=EXCLUSION_ONE_BOOK,
        )
        _write_records(
            exclusion_two,
            [fixtures._record(index) for index in range(10, 16)],
            seed=202,
            book=EXCLUSION_TWO_BOOK,
        )
        records = [
            fixtures._record(0),
            fixtures._record(10),
            fixtures._record(100),
            fixtures._record(100),
            fixtures._record(101),
            fixtures._record(102),
            fixtures._record(103),
            fixtures._record(104),
            fixtures._record(105),
            fixtures._record(106),
            fixtures._record(107),
        ]
        _write_records(candidate, records, seed=303, book=CANDIDATE_BOOK)
        mutated = [_change_score(record, 1000 + index) for index, record in enumerate(records)]
        _write_records(candidate_labels, mutated, seed=303, book=CANDIDATE_BOOK)

        output = root / "selected"
        receipt = confirmation.create_confirmation_role(
            candidate,
            (exclusion_one, exclusion_two),
            output,
            _allow_fixture=True,
            _target_records=4,
            _source_override=SOURCE,
        )
        eligible = [2, 4, 5, 6, 7, 8, 9, 10]
        expected = tuple(
            index
            for _, index in sorted(
                (confirmation._order_key(index), index) for index in eligible
            )[:4]
        )
        with confirmation.ConfirmationRoleDataset(output / "receipt.json") as dataset:
            if dataset.source_indices != expected:
                raise AssertionError("C3 fixed hash order drifted")
            if len(dataset) != 4 or len(tuple(dataset.batches(3))) != 2:
                raise AssertionError("C3 materialized adapter framing drifted")
        if receipt["postconditions"] != {
            "physical_overlap_with_each_excluded_role": [0, 0],
            "legacy_overlap_with_each_excluded_role": [0, 0],
            "physical_confirmation_duplicates": 0,
            "legacy_confirmation_duplicates": 0,
        }:
            raise AssertionError("C3 postconditions drifted")

        verification = confirmation.verify_confirmation_role(
            candidate,
            (exclusion_one, exclusion_two),
            output / "receipt.json",
            _allow_fixture=True,
            _source_override=SOURCE,
        )
        if verification["claims"]["canonical_selection_recomputed"] is not True:
            raise AssertionError("C3 confirmation role was not independently recomputed")

        label_output = root / "selected-labels"
        label_receipt = confirmation.create_confirmation_role(
            candidate_labels,
            (exclusion_one, exclusion_two),
            label_output,
            _allow_fixture=True,
            _target_records=4,
            _source_override=SOURCE,
        )
        if (
            label_receipt["selection"]["selected_indices"]["sha256"]
            != receipt["selection"]["selected_indices"]["sha256"]
        ):
            raise AssertionError("label-only mutations changed the C3 selected indices")
        if (
            label_receipt["materialized_output"]["payload_sha256"]
            == receipt["materialized_output"]["payload_sha256"]
        ):
            raise AssertionError("label-only mutations did not change materialized provenance")

        original_indices = (output / confirmation.INDEX_FILENAME).read_bytes()
        tampered_indices = bytearray(original_indices)
        tampered_indices[0] ^= 1
        (output / confirmation.INDEX_FILENAME).write_bytes(tampered_indices)
        _expect_failure(
            lambda: confirmation.ConfirmationRoleDataset(output / "receipt.json"),
            "index artifact drifted",
        )
        (output / confirmation.INDEX_FILENAME).write_bytes(original_indices)

        insufficient = root / "insufficient"
        _expect_failure(
            lambda: confirmation.create_confirmation_role(
                candidate,
                (exclusion_one, exclusion_two),
                insufficient,
                _allow_fixture=True,
                _target_records=9,
                _source_override=SOURCE,
            ),
            "eligible records",
        )
        if insufficient.exists():
            raise AssertionError("failed C3 selection left a materialized directory")

    print(
        "Horde V2 C3 confirmation-role tests passed: two-role dual-key exclusion, "
        "label-blind hash ordering, canonical verification and fail-closed artifacts"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
