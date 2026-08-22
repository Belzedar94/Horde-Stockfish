#!/usr/bin/env python3
"""Fail-closed checks for HORDE_BIN_V1_R2 chunk sets.

The record reader learned the revision on the datagen branch. The chunk-set
layer did not, and it is the layer that decides whether a corpus may be trained
on: it hardcoded the plain identity, so no corpus A chunk could assemble, and it
left the expansion caps out of the compared common manifest, so a chunk
generated at other caps would have assembled into a campaign declaring 2/2/5
without anything objecting.

These tests pin both directions, and pin that a plain-identity contract keeps
producing byte-identical receipts.
"""

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

BASE_COMMON = {
    "hash_mb": 16,
    "depth": 64,
    "nodes": 10000,
    "random_move_min_ply": 1,
    "random_move_max_ply": 1,
    "random_move_count": 0,
    "random_multi_pv": 0,
    "random_multi_pv_diff": 0,
    "write_min_ply": 0,
    "write_max_ply": 1,
    "max_game_ply": 2,
}
EXPANSION = {"expand_promo": 2, "expand_check": 2, "expand_max_children": 5}


def _record(index: int, *, family: int = 0) -> bytes:
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
    # The EXPANSION_CHILD flag and the family must agree; the writer derives one
    # from the other, and the reader cross-checks them.
    flags = 0x80 if family else 0
    state = bytes((side, 0, 64, flags))
    score = result * 200 + (index * 137) % 1201 - 600
    reason = 3 if result == 0 else 1
    outcome = reason | (family << 3)
    labels = struct.pack(
        "<HHhHHbB", index % 100, side, score, move, move, result, outcome
    )
    record = packed_board + state + labels
    if len(record) != wire.RECORD_SIZE:
        raise AssertionError("synthetic record has the wrong size")
    wire.validate_record(record, index)
    return record


def _write_contract(
    path: Path,
    *,
    expanded: bool = True,
    common_overrides: dict | None = None,
    dataset_schema: str | None = None,
) -> None:
    schema = dataset_schema or (
        wire.SCHEMA_R2_NAME if expanded else wire.SCHEMA_NAME
    )
    common = dict(BASE_COMMON)
    if expanded:
        common.update(EXPANSION)
    common.update(common_overrides or {})
    contract = {
        "schema_name": "HORDE_CORPUS_A_LEGACY_SCALE_V1",
        "dependencies": {
            "dataset": {
                "schema": schema,
                "schema_sha256": wire.SCHEMA_IDENTITIES[schema],
            },
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
            "campaign_id": "fixture-corpus-a",
            "cohort": "fixture-n10k-expand2x2",
        },
        "books": {
            "training": {"records": 3, "raw_sha256": TRAIN_BOOK},
            "validation": {"records": 2, "raw_sha256": VALIDATION_BOOK},
        },
        "generation": {
            "common": common,
            "training": {
                "records": 4,
                "positions_per_chunk": 2,
                "chunk_count": 2,
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
    threads: int = 1,
    expanded: bool = True,
    generation_overrides: dict | None = None,
    child: bool = True,
) -> None:
    # A parent followed by one of its children, which is the layout the reader
    # requires: every child immediately follows the parent it came from.
    records = [_record(first)]
    records.append(_record(first + 1, family=3 if (expanded and child) else 0))
    payload = b"".join(records)
    generation = {
        "requested_records": len(records),
        "seed": str(seed),
        "threads": threads,
        **BASE_COMMON,
        "opening_count": 3,
    }
    if expanded:
        generation.update(EXPANSION)
    generation.update(generation_overrides or {})
    schema = wire.SCHEMA_R2_NAME if expanded else wire.SCHEMA_NAME
    manifest = {
        "schema": schema,
        "schema_sha256": wire.SCHEMA_IDENTITIES[schema],
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
        "book_sha256": TRAIN_BOOK,
        "producer_sha256": PRODUCER_SHA256,
        "payload_sha256": hashlib.sha256(payload).hexdigest().upper(),
        "label_contract": {
            "schema": wire.LABEL_CONTRACT_NAME,
            "schema_sha256": wire.LABEL_CONTRACT_SHA256,
        },
        "generation": generation,
    }
    encoded = json.dumps(
        manifest, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    header = wire.MAGIC + struct.pack(
        "<HHI", wire.FORMAT_VERSION, wire.HEADER_SIZE, len(encoded)
    ) + encoded
    header += bytes(wire.HEADER_SIZE - len(header))
    path.write_bytes(header + payload)


def _expect_failure(callable_object, needle: str) -> None:
    try:
        callable_object()
    except (chunk_set.ChunkSetError, wire.FormatError) as error:
        if needle not in str(error):
            raise AssertionError(f"expected {needle!r}, got {error!r}") from error
    else:
        raise AssertionError(f"expected a failure containing {needle!r}")


def _assemble(directory: Path, contract: Path, name: str = "chunk-set.json"):
    output = directory / name
    return chunk_set.assemble_chunk_set(
        contract,
        "training",
        output,
        sorted(directory.glob("chunk_*.bin")),
    ), output


def test_r2_receipt_states_the_declared_identity() -> None:
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        contract = directory / "contract.json"
        _write_contract(contract)
        _write_chunk(directory / "chunk_0.bin", first=0, seed=1000)
        _write_chunk(directory / "chunk_1.bin", first=2, seed=1001)
        receipt, path = _assemble(directory, contract)

        assert receipt["format"]["schema"] == wire.SCHEMA_R2_NAME
        dataset = receipt["dataset"]
        assert dataset["schema"] == wire.SCHEMA_R2_NAME
        assert dataset["schema_sha256"] == wire.SCHEMA_R2_SHA256
        assert dataset["source"] == chunk_set.DATASET_SOURCE
        assert dataset["expansion_common_fields"] == list(
            chunk_set.EXPANSION_COMMON_FIELDS
        )
        # The caps are authenticated, not merely present in the chunk.
        generation = receipt["common_manifest"]["generation"]
        for key, value in EXPANSION.items():
            assert generation[key] == value, key
        chunk_set.verify_chunk_set(path, contract)
        chunk_set.verify_chunk_set(path)


def test_expansion_cap_drift_is_rejected() -> None:
    # The hole this closes: before the caps entered the common manifest, a chunk
    # generated at other caps assembled into this campaign without complaint.
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        contract = directory / "contract.json"
        _write_contract(contract)
        _write_chunk(directory / "chunk_0.bin", first=0, seed=1000)
        _write_chunk(
            directory / "chunk_1.bin",
            first=2,
            seed=1001,
            generation_overrides={"expand_promo": 3},
        )
        _expect_failure(
            lambda: _assemble(directory, contract),
            "chunk common manifest identity drifted",
        )


def test_contract_and_chunk_identities_must_agree() -> None:
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        contract = directory / "contract.json"
        _write_contract(contract, expanded=True)
        _write_chunk(directory / "chunk_0.bin", first=0, seed=1000, expanded=False)
        _write_chunk(directory / "chunk_1.bin", first=2, seed=1001, expanded=False)
        _expect_failure(
            lambda: _assemble(directory, contract),
            "chunk format identity drifted",
        )


def test_expansion_keys_only_belong_to_the_revision_contract() -> None:
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        # A plain-identity contract may not declare the expansion caps.
        contract = directory / "plain-with-caps.json"
        _write_contract(contract, expanded=False, common_overrides=dict(EXPANSION))
        _expect_failure(
            lambda: chunk_set.load_campaign_expectation(contract, "training"),
            "campaign common generation fields are incomplete or unexpected",
        )
        # A revision contract must declare all three.
        stripped = dict(BASE_COMMON)
        partial = directory / "revision-without-caps.json"
        _write_contract(partial, expanded=True)
        document = json.loads(partial.read_text())
        document["generation"]["common"] = stripped
        partial.write_text(json.dumps(document, indent=2) + "\n", newline="\n")
        _expect_failure(
            lambda: chunk_set.load_campaign_expectation(partial, "training"),
            "campaign common generation fields are incomplete or unexpected",
        )


def test_unregistered_dataset_identity_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        contract = directory / "contract.json"
        _write_contract(contract)
        document = json.loads(contract.read_text())
        document["dependencies"]["dataset"]["schema"] = "HORDE_BIN_V9"
        contract.write_text(json.dumps(document, indent=2) + "\n", newline="\n")
        _expect_failure(
            lambda: chunk_set.load_campaign_expectation(contract, "training"),
            "campaign dataset identity is not a registered",
        )


def test_receipt_dataset_block_is_required_exactly_under_the_revision() -> None:
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        contract = directory / "contract.json"
        _write_contract(contract)
        _write_chunk(directory / "chunk_0.bin", first=0, seed=1000)
        _write_chunk(directory / "chunk_1.bin", first=2, seed=1001)
        _, path = _assemble(directory, contract)

        document = json.loads(path.read_text())
        document.pop("dataset")
        stripped = directory / "stripped.json"
        stripped.write_text(json.dumps(document, indent=2) + "\n", newline="\n")
        _expect_failure(
            lambda: chunk_set.verify_chunk_set(stripped),
            "chunk-set receipt fields are incomplete",
        )

        document = json.loads(path.read_text())
        document["dataset"]["expansion_common_fields"] = ["expand_promo"]
        weakened = directory / "weakened.json"
        weakened.write_text(json.dumps(document, indent=2) + "\n", newline="\n")
        _expect_failure(
            lambda: chunk_set.verify_chunk_set(weakened),
            "does not authenticate the expansion settings",
        )


def test_plain_identity_receipt_carries_no_dataset_block() -> None:
    # Backwards compatibility, stated as a test rather than as a claim: a
    # plain-identity campaign produces the receipt it always produced, so every
    # receipt already written re-authenticates unchanged.
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        contract = directory / "contract.json"
        _write_contract(contract, expanded=False)
        _write_chunk(directory / "chunk_0.bin", first=0, seed=1000, expanded=False)
        _write_chunk(directory / "chunk_1.bin", first=2, seed=1001, expanded=False)
        receipt, path = _assemble(directory, contract)

        assert "dataset" not in receipt
        assert receipt["format"]["schema"] == wire.SCHEMA_NAME
        assert set(receipt["common_manifest"]["generation"]) == set(
            chunk_set.GENERATION_COMMON_FIELDS
        )
        chunk_set.verify_chunk_set(path, contract)

        document = json.loads(path.read_text())
        document["dataset"] = {
            "schema": wire.SCHEMA_R2_NAME,
            "schema_sha256": wire.SCHEMA_R2_SHA256,
            "source": chunk_set.DATASET_SOURCE,
            "expansion_common_fields": list(chunk_set.EXPANSION_COMMON_FIELDS),
        }
        smuggled = directory / "smuggled.json"
        smuggled.write_text(json.dumps(document, indent=2) + "\n", newline="\n")
        _expect_failure(
            lambda: chunk_set.verify_chunk_set(smuggled),
            "chunk-set receipt fields are incomplete",
        )


def main() -> int:
    test_r2_receipt_states_the_declared_identity()
    test_expansion_cap_drift_is_rejected()
    test_contract_and_chunk_identities_must_agree()
    test_expansion_keys_only_belong_to_the_revision_contract()
    test_unregistered_dataset_identity_is_rejected()
    test_receipt_dataset_block_is_required_exactly_under_the_revision()
    test_plain_identity_receipt_carries_no_dataset_block()
    print("HORDE_BIN_V1_R2 chunk set: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
