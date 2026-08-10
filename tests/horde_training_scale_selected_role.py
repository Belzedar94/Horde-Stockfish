#!/usr/bin/env python3
"""Focused checks for exact bounded-memory scale validation selection."""

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
import horde_training_decoder as decoder  # noqa: E402
import horde_training_scale_selected_role as selector  # noqa: E402


TRAIN_BOOK = "A" * 64
VALIDATION_BOOK = "B" * 64
SOURCE_COMMIT = "1" * 40
PRODUCER_SHA256 = "2" * 64


def _record(identity: int, *, rule50: int | None = None) -> bytes:
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
    state = bytes((side, 0, 64, 0))
    score = result * 200 + (identity * 137) % 1201 - 600
    reason = 3 if result == 0 else 1
    labels = struct.pack(
        "<HHhHHbB",
        identity % 100 if rule50 is None else rule50,
        side,
        score,
        move,
        move,
        result,
        reason,
    )
    record = packed_board + state + labels
    wire.validate_record(record, identity)
    return record


def _write_contract(path: Path) -> None:
    contract = {
        "schema_name": selector.CONTRACT_SCHEMA,
        "dependencies": {
            "dataset": {
                "schema": wire.SCHEMA_NAME,
                "schema_sha256": wire.SCHEMA_SHA256,
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
            "campaign_id": "fixture-rank8-scale",
            "cohort": "fixture-v3-split",
        },
        "books": {
            "training": {"records": 3, "raw_sha256": TRAIN_BOOK},
            "validation": {"records": 3, "raw_sha256": VALIDATION_BOOK},
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
                "records": 6,
                "positions_per_chunk": 2,
                "chunk_count": 3,
                "base_seed": 2000,
            },
        },
        "validation_selection": {
            "target_records": 3,
            "algorithm": selector.ALGORITHM,
            "candidate_order": "chunk index ascending, then local record index ascending",
            "reject_training_physical_key": True,
            "reject_training_legacy_model_input_key": True,
            "reject_selected_physical_duplicate": True,
            "reject_selected_legacy_model_input_duplicate": True,
            "label_blind": True,
            "insufficient_candidate_records_fail_closed": True,
        },
    }
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_chunk(
    path: Path,
    records: list[bytes],
    *,
    seed: int,
    threads: int,
    book_sha256: str,
) -> None:
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
        manifest, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    header = wire.MAGIC + struct.pack(
        "<HHI", wire.FORMAT_VERSION, wire.HEADER_SIZE, len(encoded)
    ) + encoded
    header += bytes(wire.HEADER_SIZE - len(header))
    path.write_bytes(header + payload)


def _assemble(
    root: Path,
    contract: Path,
    role: str,
    records: list[bytes],
    base_seed: int,
    book: str,
) -> Path:
    chunks: list[Path] = []
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
    chunk_set.assemble_chunk_set(contract, role, receipt, list(reversed(chunks)))
    return receipt


def _expect_failure(callable_object: object, needle: str) -> None:
    try:
        callable_object()  # type: ignore[operator]
    except (selector.ScaleSelectedRoleError, wire.FormatError) as error:
        if needle not in str(error):
            raise AssertionError(f"expected {needle!r}, got {error!r}") from error
    else:
        raise AssertionError(f"expected failure containing {needle!r}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="horde-scale-selected-") as directory:
        root = Path(directory)
        contract = root / "campaign.json"
        _write_contract(contract)
        training_records = [_record(identity) for identity in range(6)]
        candidate_records = [
            training_records[0],
            _record(10),
            _record(10),
            _record(11),
            _record(11, rule50=99),
            _record(12),
        ]
        train = _assemble(root, contract, "training", training_records, 1000, TRAIN_BOOK)
        candidate = _assemble(
            root,
            contract,
            "validation_candidate",
            candidate_records,
            2000,
            VALIDATION_BOOK,
        )

        for index, raw in enumerate(training_records + candidate_records):
            fast = selector._selection_key_digests(raw, index)
            decoded = decoder.decode_training_record(raw, index)
            reference = (
                hashlib.sha256(decoder.physical_position_key(decoded)).digest(),
                hashlib.sha256(decoder.legacy_model_input_key(decoded)).digest(),
            )
            if fast != reference:
                raise AssertionError("optimized key extraction differs from the frozen keys")

        output = root / "selected"
        scratch = root / "scratch"
        receipt = selector.create_scale_selected_role(
            train,
            candidate,
            output,
            scratch,
            contract_path=contract,
            _allow_fixture=True,
            _source_override={"commit": SOURCE_COMMIT, "dirty": False},
        )
        indices = [
            struct.unpack_from("<Q", (output / selector.INDEX_FILENAME).read_bytes(), offset)[0]
            for offset in range(0, 24, 8)
        ]
        if indices != [1, 3, 5]:
            raise AssertionError(f"first-eligible global ordering drifted: {indices}")
        if receipt["selection"]["records_examined"] != 6:
            raise AssertionError("selector stopped at the wrong candidate boundary")
        masks = receipt["selection"]["rejection_reason_masks"]
        if masks != {"3": 1, "4": 1, "12": 1}:
            raise AssertionError(f"rejection accounting drifted: {masks}")
        exact = receipt["selection"]["exact_membership_index"]
        if exact["bucket_count"] != 256:
            raise AssertionError("bounded-memory bucket count drifted")
        if exact["rejection_flags"]["training_physical_matches"] != 1:
            raise AssertionError("training physical overlap was not indexed")

        with selector.ScaleSelectedRoleDataset(
            output / selector.RECEIPT_FILENAME,
            contract,
            _allow_fixture=True,
        ) as selected:
            if len(selected) != 3:
                raise AssertionError("selected-role length drifted")
            if [selected.raw_record(index) for index in range(3)] != [
                candidate_records[1],
                candidate_records[3],
                candidate_records[5],
            ]:
                raise AssertionError("materialized selected-role order drifted")
            if selected.identity()["selected_role"]["candidate_chunk_set_sha256"] != receipt[
                "candidate_source"
            ]["chunk_set_sha256"]:
                raise AssertionError("selected role lost its candidate chunk-set identity")

        corrupted = bytearray((output / selector.RECORDS_FILENAME).read_bytes())
        corrupted[-1] ^= 1
        (output / selector.RECORDS_FILENAME).write_bytes(corrupted)
        _expect_failure(
            lambda: selector.ScaleSelectedRoleDataset(
                output / selector.RECEIPT_FILENAME,
                contract,
                _allow_fixture=True,
            ),
            "hash drifted",
        )

        insufficient_contract = root / "insufficient.json"
        changed = json.loads(contract.read_text(encoding="utf-8"))
        changed["validation_selection"]["target_records"] = 6
        insufficient_contract.write_text(
            json.dumps(changed, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        insufficient_root = root / "insufficient-fixture"
        insufficient_root.mkdir()
        insufficient_train = _assemble(
            insufficient_root,
            insufficient_contract,
            "training",
            training_records,
            1000,
            TRAIN_BOOK,
        )
        insufficient_candidate = _assemble(
            insufficient_root,
            insufficient_contract,
            "validation_candidate",
            candidate_records,
            2000,
            VALIDATION_BOOK,
        )
        _expect_failure(
            lambda: selector.create_scale_selected_role(
                insufficient_train,
                insufficient_candidate,
                root / "insufficient-selected",
                root / "insufficient-scratch",
                contract_path=insufficient_contract,
                _allow_fixture=True,
                _source_override={"commit": SOURCE_COMMIT, "dirty": False},
            ),
            "candidate exhausted",
        )

    print("scale selected-role tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
