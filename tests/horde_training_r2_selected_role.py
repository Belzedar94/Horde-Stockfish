#!/usr/bin/env python3
"""The validation role of an expanded corpus is carved from parents only.

An expansion child shares its parent's game and sits one move from it. The
dual-key index cannot exclude it, because a child is a different position from
its parent, so without an explicit filter a child lands in the validation role
as a near duplicate of a training record. The contract declares the filter and
the receipt records that it ran.
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
import horde_training_scale_selected_role as selector  # noqa: E402


TRAIN_BOOK = "A" * 64
VALIDATION_BOOK = "B" * 64
SOURCE_COMMIT = "1" * 40
PRODUCER_SHA256 = "2" * 64

COMMON = {
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
    "expand_promo": 2,
    "expand_check": 2,
    "expand_max_children": 5,
}


def _record(identity: int, *, family: int = 0) -> bytes:
    board = [0] * 64
    side = identity & 1
    result = (-1, 0, 1)[(identity // 2) % 3]
    board[0] = 2
    board[8 + identity % 32] = 4
    board[48 + (identity // 32) % 8] = 9
    board[57] = 7
    board[60] = 11
    packed_board = bytes(
        board[square] | (board[square + 1] << 4) for square in range(0, 64, 2)
    )
    move = (0 << 6) | 7 if side == 0 else (57 << 6) | 56
    state = bytes((side, 0, 64, 0x80 if family else 0))
    score = result * 200 + (identity * 137) % 1201 - 600
    reason = 3 if result == 0 else 1
    labels = struct.pack(
        "<HHhHHbB",
        identity % 100,
        side,
        score,
        move,
        move,
        result,
        reason | (family << 3),
    )
    record = packed_board + state + labels
    wire.validate_record(record, identity)
    return record


def _write_contract(
    path: Path,
    *,
    dataset_schema: str = wire.SCHEMA_R2_NAME,
    parents_only: bool | None = True,
) -> None:
    common = dict(COMMON)
    if dataset_schema == wire.SCHEMA_NAME:
        for key in wire.EXPANSION_GENERATION_KEYS:
            common.pop(key)
    selection = {
        "target_records": 3,
        "algorithm": selector.ALGORITHM,
        "candidate_order": "chunk index ascending, then local record index ascending",
        "reject_training_physical_key": True,
        "reject_training_legacy_model_input_key": True,
        "reject_selected_physical_duplicate": True,
        "reject_selected_legacy_model_input_duplicate": True,
        "label_blind": True,
        "insufficient_candidate_records_fail_closed": True,
    }
    if parents_only is not None:
        selection["parents_only"] = parents_only
    contract = {
        "schema_name": selector.CORPUS_A_LEGACY_CONTRACT_SCHEMA,
        "dependencies": {
            "dataset": {
                "schema": dataset_schema,
                "schema_sha256": wire.SCHEMA_IDENTITIES[dataset_schema],
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
            "selected_validation_schema": selector.SCHEMA,
        },
        "openbench": {
            "campaign_id": "fixture-corpus-a",
            "cohort": "fixture-n10k-expand2x2",
        },
        "books": {
            "training": {"records": 3, "raw_sha256": TRAIN_BOOK},
            "validation": {"records": 3, "raw_sha256": VALIDATION_BOOK},
        },
        "generation": {
            "common": common,
            "training": {
                "records": 6,
                "positions_per_chunk": 2,
                "chunk_count": 3,
                "base_seed": 1000,
            },
            "validation_candidate": {
                "records": 6,
                "positions_per_chunk": 2,
                "chunk_count": 3,
                "base_seed": 2000,
            },
        },
        "validation_selection": selection,
    }
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_chunk(
    path: Path,
    records: list[bytes],
    *,
    seed: int,
    threads: int,
    book_sha256: str,
    dataset_schema: str = wire.SCHEMA_R2_NAME,
) -> None:
    payload = b"".join(records)
    generation = {
        "requested_records": len(records),
        "seed": str(seed),
        "threads": threads,
        **{key: value for key, value in COMMON.items()
           if key not in wire.EXPANSION_GENERATION_KEYS},
        "opening_count": 3,
    }
    if dataset_schema == wire.SCHEMA_R2_NAME:
        generation.update(
            {key: COMMON[key] for key in wire.EXPANSION_GENERATION_KEYS}
        )
    manifest = {
        "schema": dataset_schema,
        "schema_sha256": wire.SCHEMA_IDENTITIES[dataset_schema],
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


def _assemble(root, contract, role, records, base_seed, book) -> Path:
    chunks = []
    for index in range(3):
        chunk = root / f"{role}-{index}.bin"
        _write_chunk(
            chunk,
            records[2 * index : 2 * index + 2],
            seed=base_seed + index,
            threads=(1, 4, 2)[index],
            book_sha256=book,
        )
        chunks.append(chunk)
    receipt = root / f"{role}-chunk-set.json"
    chunk_set.assemble_chunk_set(contract, role, receipt, chunks)
    return receipt


def _expect_failure(callable_object, needle: str) -> None:
    try:
        callable_object()
    except (selector.ScaleSelectedRoleError, chunk_set.ChunkSetError,
            wire.FormatError) as error:
        if needle not in str(error):
            raise AssertionError(f"expected {needle!r}, got {error!r}") from error
    else:
        raise AssertionError(f"expected a failure containing {needle!r}")


def test_children_are_never_selected() -> None:
    with tempfile.TemporaryDirectory(prefix="horde-r2-selected-") as directory:
        root = Path(directory)
        contract = root / "campaign.json"
        _write_contract(contract)

        training_records = [_record(identity) for identity in range(6)]
        # Three eligible parents, interleaved with children that must never be
        # selected however eligible their keys look.
        candidate_records = [
            _record(20), _record(21, family=2),
            _record(22, family=3), _record(23),
            _record(24, family=2), _record(25),
        ]
        train = _assemble(root, contract, "training", training_records, 1000, TRAIN_BOOK)
        candidate = _assemble(
            root, contract, "validation_candidate", candidate_records, 2000,
            VALIDATION_BOOK,
        )

        output = root / "selected"
        receipt = selector.create_scale_selected_role(
            train,
            candidate,
            output,
            root / "scratch",
            contract_path=contract,
            _allow_fixture=True,
            _source_override={"commit": SOURCE_COMMIT, "dirty": False},
        )

        assert receipt["selection"]["parents_only"] is True
        assert receipt["record_schema"]["schema"] == wire.SCHEMA_R2_NAME
        assert receipt["contract"]["schema"] == (
            selector.CORPUS_A_LEGACY_CONTRACT_SCHEMA
        )

        payload = (output / selector.RECORDS_FILENAME).read_bytes()
        assert len(payload) == 3 * wire.RECORD_SIZE
        for index in range(3):
            record = payload[index * wire.RECORD_SIZE:(index + 1) * wire.RECORD_SIZE]
            family = (record[47] >> 3) & 0x07
            assert family == 0, f"selected record {index} is an expansion child"
            assert record[35] & 0x80 == 0, f"selected record {index} carries the child flag"

        # Every child that was walked past is accounted for in the histogram.
        masks = receipt["selection"]["rejection_reason_masks"]
        children_rejected = sum(
            count for mask, count in masks.items()
            if int(mask) & selector.REJECT_CHILD
        )
        assert children_rejected == 3, masks


def test_the_filter_is_declared_not_inferred() -> None:
    with tempfile.TemporaryDirectory(prefix="horde-r2-declare-") as directory:
        root = Path(directory)
        # A revision corpus that does not declare the filter must not load.
        silent = root / "silent.json"
        _write_contract(silent, parents_only=None)
        _expect_failure(
            lambda: selector.load_contract(silent, allow_fixture=True),
            "parents only under the revision identity",
        )
        off = root / "off.json"
        _write_contract(off, parents_only=False)
        _expect_failure(
            lambda: selector.load_contract(off, allow_fixture=True),
            "parents only under the revision identity",
        )
        # A plain corpus has no children, so claiming the filter is a statement
        # about data that cannot exist.
        plain = root / "plain.json"
        _write_contract(plain, dataset_schema=wire.SCHEMA_NAME, parents_only=True)
        _expect_failure(
            lambda: selector.load_contract(plain, allow_fixture=True),
            "parents only under the revision identity",
        )
        # A plain corpus that stays silent is the pre-existing behaviour.
        quiet = root / "quiet.json"
        _write_contract(quiet, dataset_schema=wire.SCHEMA_NAME, parents_only=None)
        selector.load_contract(quiet, allow_fixture=True)


def main() -> int:
    test_children_are_never_selected()
    test_the_filter_is_declared_not_inferred()
    print("HORDE_BIN_V1_R2 selected role: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
