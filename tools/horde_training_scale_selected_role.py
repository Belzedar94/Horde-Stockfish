#!/usr/bin/env python3
"""Build an exact, bounded-memory selected role from Horde chunk sets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence

try:
    from . import horde_bin_v1 as wire
    from .horde_training_chunk_set import HordeChunkSetDataset
    from .horde_training_decoder import (
        BLACK,
        WHITE,
        SparseBatch,
        TrainingRecord,
        decode_training_record,
        legacy_feature_index,
        legacy_model_input_key,
        make_sparse_batch,
        physical_position_key,
    )
except ImportError:
    import horde_bin_v1 as wire
    from horde_training_chunk_set import HordeChunkSetDataset
    from horde_training_decoder import (
        BLACK,
        WHITE,
        SparseBatch,
        TrainingRecord,
        decode_training_record,
        legacy_feature_index,
        legacy_model_input_key,
        make_sparse_batch,
        physical_position_key,
    )


SCHEMA = "HORDE_TRAINING_SELECTED_ROLE_V1"
VERIFICATION_SCHEMA = "HORDE_TRAINING_SCALE_SELECTED_ROLE_VERIFICATION_V1"
ALGORITHM = "FIRST_ELIGIBLE_DUAL_KEY_V1"
INDEX_ALGORITHM = "PARTITIONED_SHA256_BUCKETS_V1"
CONTRACT_SCHEMA = "HORDE_V2_RANK8_SCALE_V1"
CONTRACT_RELATIVE_PATH = Path("schemas/horde-v2-rank8-scale-v1.json")
CONTRACT_SHA256 = "B8A8512D32930C88CAD0248C05A2AAD3B2CE8E2096A46DCAB84D87F1C532E1D1"
V3_CONTRACT_SCHEMA = "HORDE_V3_SCALE_V1"
V3_CONTRACT_RELATIVE_PATH = Path("schemas/horde-v3-scale-v1.json")
V3_CONTRACT_SHA256 = "405A7FD075A050FA5090F5567FB1BE36547151B85EA9C571CA31A37F0BA925C1"

# Every registered scale contract, keyed by its schema name. A contract is
# accepted only when its own SHA-256 matches the entry pinned here, and each
# entry names the one architecture that contract may bind. A contract can never
# be loaded under another contract's identity.
SCALE_CONTRACTS: dict[str, dict[str, object]] = {
    CONTRACT_SCHEMA: {
        "relative_path": CONTRACT_RELATIVE_PATH,
        "sha256": CONTRACT_SHA256,
        "architecture": "v2-c1-rank8-64x192",
    },
    V3_CONTRACT_SCHEMA: {
        "relative_path": V3_CONTRACT_RELATIVE_PATH,
        "sha256": V3_CONTRACT_SHA256,
        "architecture": "v3-g1024-pawn-wpc8",
    },
}
SELECTOR_RELATIVE_PATH = Path("tools/horde_training_scale_selected_role.py")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

INDEX_FILENAME = "selected-indices.bin"
RECORDS_FILENAME = "selected-records.bin"
RECEIPT_FILENAME = "receipt.json"
FLAGS_FILENAME = "candidate-rejection-flags.bin"
KEYS_FILENAME = "candidate-keys.bin"
BUCKET_COUNT = 256
TRAIN_ENTRY_BYTES = 32
QUERY_ENTRY_BYTES = 40
KEY_PAIR_BYTES = 64
BUFFER_BYTES = 64 * 1024

PHYSICAL_TAG = 0
LEGACY_TAG = 1
REJECT_TRAIN_PHYSICAL = 1
REJECT_TRAIN_LEGACY = 2
REJECT_SELECTED_PHYSICAL = 4
REJECT_SELECTED_LEGACY = 8


class ScaleSelectedRoleError(ValueError):
    """Raised when the scale selected-role contract is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScaleSelectedRoleError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"{label} does not exist: {resolved}")
    payload = resolved.read_bytes()
    try:
        root = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScaleSelectedRoleError(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(root, dict), f"{label} root is not an object")
    return root, payload


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789ABCDEF" for character in value)
    )


def _valid_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value != "0" * 40
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _repository_identity(
    source_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if source_override is not None:
        source = dict(source_override)
    else:
        def git(*arguments: str) -> str:
            result = subprocess.run(
                ["git", *arguments],
                cwd=REPOSITORY_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            return result.stdout.strip()

        source = {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
        }
    _require(
        _valid_commit(source.get("commit")) and type(source.get("dirty")) is bool,
        "selector source identity is invalid",
    )
    selector = REPOSITORY_ROOT / SELECTOR_RELATIVE_PATH
    return {
        **source,
        "path": SELECTOR_RELATIVE_PATH.as_posix(),
        "file_sha256": _sha256_file(selector),
    }


def load_contract(
    path: Path | None = None,
    *,
    allow_fixture: bool = False,
) -> tuple[dict[str, Any], str]:
    resolved = (path or REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH).expanduser().resolve()
    contract, payload = _read_json(resolved, "scale contract")
    digest = _sha256_bytes(payload)
    schema_name = contract.get("schema_name")
    _require(schema_name in SCALE_CONTRACTS, f"unknown scale contract schema: {schema_name}")
    registered = SCALE_CONTRACTS[str(schema_name)]
    _require(
        allow_fixture or digest == registered["sha256"],
        f"scale contract SHA-256 mismatch: {digest}",
    )

    dependencies = _mapping(contract.get("dependencies"), "scale dependencies")
    generation = _mapping(contract.get("generation"), "scale generation")
    training = _mapping(generation.get("training"), "training generation")
    candidate = _mapping(generation.get("validation_candidate"), "candidate generation")
    selection = _mapping(contract.get("validation_selection"), "validation selection")
    _require(
        dependencies.get("selected_validation_schema") == SCHEMA
        and dependencies.get("dataset", {}).get("schema") == wire.SCHEMA_NAME
        and dependencies.get("dataset", {}).get("schema_sha256") == wire.SCHEMA_SHA256,
        "scale dataset dependencies drifted",
    )
    _require(
        selection.get("target_records") > 0
        and selection.get("algorithm") == ALGORITHM
        and selection.get("candidate_order")
        == "chunk index ascending, then local record index ascending"
        and selection.get("reject_training_physical_key") is True
        and selection.get("reject_training_legacy_model_input_key") is True
        and selection.get("reject_selected_physical_duplicate") is True
        and selection.get("reject_selected_legacy_model_input_duplicate") is True
        and selection.get("label_blind") is True
        and selection.get("insufficient_candidate_records_fail_closed") is True,
        "scale selection contract drifted",
    )
    for role, section in (("training", training), ("validation candidate", candidate)):
        _require(
            type(section.get("records")) is int
            and type(section.get("positions_per_chunk")) is int
            and type(section.get("chunk_count")) is int
            and section["records"] == section["positions_per_chunk"] * section["chunk_count"],
            f"{role} chunk accounting drifted",
        )
    _require(
        selection["target_records"] <= candidate["records"],
        "selected target exceeds candidate role",
    )
    return contract, digest


def _dataset_identity(dataset: HordeChunkSetDataset) -> dict[str, object]:
    identity = dataset.identity()
    return {
        "receipt_name": dataset.path.name,
        "receipt_sha256": dataset.receipt_sha256,
        "chunk_set_sha256": dataset.chunk_set_sha256,
        "logical_payload_sha256": dataset.logical_payload_sha256,
        "record_count": len(dataset),
        "book_sha256": dataset.manifest["book_sha256"],
        "role": dataset.receipt["role"],
        "sample_identity": dataset.receipt["identity"]["sample_identity"],
        "chunks": identity["chunks"],
        "manifest": dataset.manifest,
    }


def _validate_sources(
    train: HordeChunkSetDataset,
    candidate: HordeChunkSetDataset,
    contract: Mapping[str, Any],
) -> tuple[dict[str, object], dict[str, object]]:
    _require(train.path != candidate.path, "training and candidate receipts are identical")
    _require(train.receipt["role"] == "training", "training chunk-set role drifted")
    _require(
        candidate.receipt["role"] == "validation_candidate",
        "candidate chunk-set role drifted",
    )
    _require(
        train.chunk_set_sha256 != candidate.chunk_set_sha256,
        "training and candidate chunk-set identities match",
    )
    _require(
        train.manifest["book_sha256"] != candidate.manifest["book_sha256"],
        "training and candidate use the same opening book",
    )
    for field in (
        "source_commit",
        "source_dirty",
        "producer_sha256",
        "network",
        "label_contract",
    ):
        _require(
            train.manifest[field] == candidate.manifest[field],
            f"training and candidate field {field} differs",
        )
    books = _mapping(contract.get("books"), "scale books")
    _require(
        train.manifest["book_sha256"] == books["training"]["raw_sha256"]
        and candidate.manifest["book_sha256"] == books["validation"]["raw_sha256"],
        "chunk-set book identities differ from the scale contract",
    )
    return _dataset_identity(train), _dataset_identity(candidate)


def _selection_key_digests(raw: bytes, index: int) -> tuple[bytes, bytes]:
    """Return exact physical and legacy-input digests without building V2 rows."""

    decoded = wire.validate_record(raw, index)
    board = decoded["board"]
    physical = bytes(board) + struct.pack(
        "<BBB", decoded["side"], decoded["castling"], decoded["ep_square"]
    )
    occupied = [(square, code) for square, code in enumerate(board) if code]
    piece_count = len(occupied)
    bucket = min((piece_count - 1) * 8 // 52, 7)
    legacy = bytearray(struct.pack("<BBH", decoded["side"], bucket, decoded["rule50"]))
    for perspective in (WHITE, BLACK):
        legacy.extend(struct.pack("<H", piece_count))
        for square, code in occupied:
            legacy.extend(struct.pack("<H", legacy_feature_index(perspective, square, code)))
    return hashlib.sha256(physical).digest(), hashlib.sha256(legacy).digest()


class _BucketWriter:
    def __init__(self, root: Path, prefix: str, entry_bytes: int) -> None:
        self.paths = tuple(root / f"{prefix}-{index:02X}.bin" for index in range(BUCKET_COUNT))
        self.entry_bytes = entry_bytes
        self._files = [path.open("xb", buffering=BUFFER_BYTES) for path in self.paths]
        self._buffers = [bytearray() for _ in range(BUCKET_COUNT)]
        self._digests = [hashlib.sha256() for _ in range(BUCKET_COUNT)]
        self.counts = [0] * BUCKET_COUNT

    def write(self, bucket: int, entry: bytes) -> None:
        _require(len(entry) == self.entry_bytes, "bucket entry framing drifted")
        self._buffers[bucket].extend(entry)
        self._digests[bucket].update(entry)
        self.counts[bucket] += 1
        if len(self._buffers[bucket]) >= BUFFER_BYTES:
            self._files[bucket].write(self._buffers[bucket])
            self._buffers[bucket].clear()

    def close(self) -> None:
        for index, output in enumerate(self._files):
            if output.closed:
                continue
            if self._buffers[index]:
                output.write(self._buffers[index])
                self._buffers[index].clear()
            output.flush()
            os.fsync(output.fileno())
            output.close()

    def inventory(self) -> list[dict[str, object]]:
        _require(all(output.closed for output in self._files), "bucket writers remain open")
        return [
            {
                "bucket": index,
                "records": self.counts[index],
                "bytes": self.paths[index].stat().st_size,
                "sha256": self._digests[index].hexdigest().upper(),
            }
            for index in range(BUCKET_COUNT)
        ]

    def __enter__(self) -> "_BucketWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _partition_training(
    dataset: HordeChunkSetDataset,
    scratch: Path,
) -> list[dict[str, object]]:
    with _BucketWriter(scratch, "training", TRAIN_ENTRY_BYTES) as buckets:
        for index in range(len(dataset)):
            physical, legacy = _selection_key_digests(dataset.raw_record(index), index)
            buckets.write(physical[0], bytes((PHYSICAL_TAG,)) + physical[1:])
            buckets.write(legacy[0], bytes((LEGACY_TAG,)) + legacy[1:])
    inventory = buckets.inventory()
    _require(
        sum(entry["records"] for entry in inventory) == 2 * len(dataset),
        "training bucket accounting drifted",
    )
    return inventory


def _partition_candidate(
    dataset: HordeChunkSetDataset,
    scratch: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    keys_path = scratch / KEYS_FILENAME
    key_digest = hashlib.sha256()
    with _BucketWriter(scratch, "candidate", QUERY_ENTRY_BYTES) as buckets:
        with keys_path.open("xb", buffering=BUFFER_BYTES) as keys:
            for index in range(len(dataset)):
                physical, legacy = _selection_key_digests(dataset.raw_record(index), index)
                pair = physical + legacy
                keys.write(pair)
                key_digest.update(pair)
                encoded_index = struct.pack("<Q", index)
                buckets.write(
                    physical[0], bytes((PHYSICAL_TAG,)) + encoded_index + physical[1:]
                )
                buckets.write(legacy[0], bytes((LEGACY_TAG,)) + encoded_index + legacy[1:])
            keys.flush()
            os.fsync(keys.fileno())
    inventory = buckets.inventory()
    expected_bytes = len(dataset) * KEY_PAIR_BYTES
    _require(keys_path.stat().st_size == expected_bytes, "candidate key spool framing drifted")
    _require(
        sum(entry["records"] for entry in inventory) == 2 * len(dataset),
        "candidate bucket accounting drifted",
    )
    return inventory, {
        "name": KEYS_FILENAME,
        "bytes": expected_bytes,
        "sha256": key_digest.hexdigest().upper(),
    }


def _verified_bucket_payload(
    path: Path,
    expected: Mapping[str, object],
    entry_bytes: int,
) -> bytes:
    payload = path.read_bytes()
    _require(expected.get("bytes") == len(payload), f"bucket {path.name} size drifted")
    _require(_sha256_bytes(payload) == expected.get("sha256"), f"bucket {path.name} hash drifted")
    _require(len(payload) % entry_bytes == 0, f"bucket {path.name} is truncated")
    _require(
        expected.get("records") == len(payload) // entry_bytes,
        f"bucket {path.name} record count drifted",
    )
    return payload


def _load_training_bucket(
    path: Path,
    expected: Mapping[str, object],
) -> tuple[set[bytes], set[bytes], tuple[int, int]]:
    payload = _verified_bucket_payload(path, expected, TRAIN_ENTRY_BYTES)
    physical: set[bytes] = set()
    legacy: set[bytes] = set()
    physical_duplicates = 0
    legacy_duplicates = 0
    for offset in range(0, len(payload), TRAIN_ENTRY_BYTES):
        tag = payload[offset]
        suffix = payload[offset + 1 : offset + TRAIN_ENTRY_BYTES]
        _require(tag in (PHYSICAL_TAG, LEGACY_TAG), "training bucket tag is invalid")
        target = physical if tag == PHYSICAL_TAG else legacy
        duplicate = suffix in target
        target.add(suffix)
        if tag == PHYSICAL_TAG:
            physical_duplicates += duplicate
        else:
            legacy_duplicates += duplicate
    return physical, legacy, (physical_duplicates, legacy_duplicates)


def _mark_training_overlaps(
    scratch: Path,
    candidate_records: int,
    training_inventory: Sequence[Mapping[str, object]],
    candidate_inventory: Sequence[Mapping[str, object]],
) -> tuple[dict[str, int], dict[str, object]]:
    _require(
        len(training_inventory) == len(candidate_inventory) == BUCKET_COUNT,
        "bucket inventory count drifted",
    )
    flags_path = scratch / FLAGS_FILENAME
    with flags_path.open("xb") as output:
        output.truncate(candidate_records)
    duplicate_counts = [0, 0]
    matched_counts = [0, 0]
    with flags_path.open("r+b") as flags_file:
        flags = mmap.mmap(flags_file.fileno(), 0)
        try:
            for bucket in range(BUCKET_COUNT):
                train_path = scratch / f"training-{bucket:02X}.bin"
                query_path = scratch / f"candidate-{bucket:02X}.bin"
                train_expected = training_inventory[bucket]
                candidate_expected = candidate_inventory[bucket]
                _require(
                    train_expected.get("bucket") == candidate_expected.get("bucket") == bucket,
                    "bucket inventory ordering drifted",
                )
                physical, legacy, duplicates = _load_training_bucket(
                    train_path, train_expected
                )
                duplicate_counts[0] += duplicates[0]
                duplicate_counts[1] += duplicates[1]
                payload = _verified_bucket_payload(
                    query_path,
                    candidate_expected,
                    QUERY_ENTRY_BYTES,
                )
                for offset in range(0, len(payload), QUERY_ENTRY_BYTES):
                    tag = payload[offset]
                    index = struct.unpack_from("<Q", payload, offset + 1)[0]
                    suffix = payload[offset + 9 : offset + QUERY_ENTRY_BYTES]
                    _require(index < candidate_records, "candidate query index is out of range")
                    _require(tag in (PHYSICAL_TAG, LEGACY_TAG), "candidate bucket tag is invalid")
                    matched = suffix in (physical if tag == PHYSICAL_TAG else legacy)
                    if matched:
                        bit = REJECT_TRAIN_PHYSICAL if tag == PHYSICAL_TAG else REJECT_TRAIN_LEGACY
                        flags[index] = flags[index] | bit
                        matched_counts[tag] += 1
            flags.flush()
        finally:
            flags.close()
    return {
        "physical": duplicate_counts[0],
        "legacy_model_input": duplicate_counts[1],
    }, {
        "name": FLAGS_FILENAME,
        "bytes": flags_path.stat().st_size,
        "sha256": _sha256_file(flags_path),
        "training_physical_matches": matched_counts[0],
        "training_legacy_model_input_matches": matched_counts[1],
    }


def _inventory_sha256(inventory: Sequence[Mapping[str, object]]) -> str:
    return _sha256_bytes(_canonical_json(list(inventory)))


def _select_records(
    candidate: HordeChunkSetDataset,
    scratch: Path,
    target_records: int,
    candidate_keys: Mapping[str, object],
    rejection_flags: Mapping[str, object],
) -> dict[str, object]:
    flags_path = scratch / FLAGS_FILENAME
    keys_path = scratch / KEYS_FILENAME
    selected_physical: set[bytes] = set()
    selected_legacy: set[bytes] = set()
    selected_indices: list[int] = []
    selected_records = bytearray()
    rejection_masks: Counter[int] = Counter()
    decision_chain = hashlib.sha256()
    record_order = hashlib.sha256()
    examined = 0

    for path, expected, expected_bytes, label in (
        (keys_path, candidate_keys, len(candidate) * KEY_PAIR_BYTES, "candidate keys"),
        (flags_path, rejection_flags, len(candidate), "candidate rejection flags"),
    ):
        _require(path.stat().st_size == expected.get("bytes") == expected_bytes, f"{label} size drifted")
        _require(_sha256_file(path) == expected.get("sha256"), f"{label} hash drifted")

    with flags_path.open("rb") as flags_file, keys_path.open("rb") as keys_file:
        flags = mmap.mmap(flags_file.fileno(), 0, access=mmap.ACCESS_READ)
        keys = mmap.mmap(keys_file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for index in range(len(candidate)):
                offset = index * KEY_PAIR_BYTES
                physical = bytes(keys[offset : offset + 32])
                legacy = bytes(keys[offset + 32 : offset + KEY_PAIR_BYTES])
                mask = int(flags[index])
                if physical in selected_physical:
                    mask |= REJECT_SELECTED_PHYSICAL
                if legacy in selected_legacy:
                    mask |= REJECT_SELECTED_LEGACY
                decision_chain.update(struct.pack("<QB", index, mask))
                examined = index + 1
                if mask:
                    rejection_masks[mask] += 1
                    continue
                raw = candidate.raw_record(index)
                selected_indices.append(index)
                selected_records.extend(raw)
                selected_physical.add(physical)
                selected_legacy.add(legacy)
                record_order.update(struct.pack("<Q", index))
                record_order.update(raw)
                if len(selected_indices) == target_records:
                    break
        finally:
            keys.close()
            flags.close()

    _require(
        len(selected_indices) == target_records,
        f"candidate exhausted after {len(selected_indices)} eligible records; target is {target_records}",
    )
    index_payload = b"".join(struct.pack("<Q", index) for index in selected_indices)
    records_payload = bytes(selected_records)
    _require(
        len(records_payload) == target_records * wire.RECORD_SIZE,
        "selected record payload framing drifted",
    )
    return {
        "index_payload": index_payload,
        "records_payload": records_payload,
        "records_examined": examined,
        "cutoff_candidate_index": selected_indices[-1],
        "rejection_reason_masks": {
            str(mask): rejection_masks[mask] for mask in sorted(rejection_masks)
        },
        "decision_chain_sha256": decision_chain.hexdigest().upper(),
        "selected_index_sha256": _sha256_bytes(index_payload),
        "materialized_payload_sha256": _sha256_bytes(records_payload),
        "record_order_sha256": record_order.hexdigest().upper(),
    }


def _build_receipt(
    *,
    contract_sha256: str,
    source: Mapping[str, object],
    train_identity: Mapping[str, object],
    candidate_identity: Mapping[str, object],
    target_records: int,
    selected: Mapping[str, object],
    training_inventory: Sequence[Mapping[str, object]],
    candidate_inventory: Sequence[Mapping[str, object]],
    candidate_keys: Mapping[str, object],
    rejection_flags: Mapping[str, object],
    training_duplicates: Mapping[str, int],
    fixture_mode: bool,
) -> dict[str, object]:
    index_payload = selected["index_payload"]
    records_payload = selected["records_payload"]
    _require(isinstance(index_payload, bytes), "selected index payload is invalid")
    _require(isinstance(records_payload, bytes), "selected record payload is invalid")
    return {
        "schema": SCHEMA,
        "contract": {"schema": CONTRACT_SCHEMA, "sha256": contract_sha256},
        "role": "validation",
        "record_schema": {
            "schema": wire.SCHEMA_NAME,
            "schema_sha256": wire.SCHEMA_SHA256,
            "record_bytes": wire.RECORD_SIZE,
        },
        "selector_source": dict(source),
        "training_reference": dict(train_identity),
        "candidate_source": dict(candidate_identity),
        "selection": {
            "algorithm": ALGORITHM,
            "candidate_order": "chunk index ascending, then local record index ascending",
            "selected_index_encoding": "uint64_little_endian_global_logical_index",
            "target_records": target_records,
            "records_examined": selected["records_examined"],
            "cutoff_candidate_index": selected["cutoff_candidate_index"],
            "accepted_records": target_records,
            "rejected_records": selected["records_examined"] - target_records,
            "rejection_reason_masks": selected["rejection_reason_masks"],
            "decision_chain_sha256": selected["decision_chain_sha256"],
            "selected_indices": {
                "name": INDEX_FILENAME,
                "bytes": len(index_payload),
                "sha256": selected["selected_index_sha256"],
            },
            "exact_membership_index": {
                "schema": INDEX_ALGORITHM,
                "hash": "SHA-256",
                "bucket": "digest_byte_0",
                "bucket_count": BUCKET_COUNT,
                "training_entry": "key_family_u8 || digest_bytes_1_31",
                "candidate_entry": "key_family_u8 || global_index_u64le || digest_bytes_1_31",
                "training_inventory_sha256": _inventory_sha256(training_inventory),
                "candidate_inventory_sha256": _inventory_sha256(candidate_inventory),
                "candidate_keys": dict(candidate_keys),
                "rejection_flags": dict(rejection_flags),
            },
        },
        "materialized_output": {
            "name": RECORDS_FILENAME,
            "bytes": len(records_payload),
            "sha256": selected["materialized_payload_sha256"],
            "payload_sha256": selected["materialized_payload_sha256"],
            "record_count": target_records,
            "record_order_sha256": selected["record_order_sha256"],
        },
        "training_internal_duplicates": dict(training_duplicates),
        "postconditions": {
            "physical_cross_role_overlap_samples": 0,
            "legacy_cross_role_overlap_samples": 0,
            "physical_validation_duplicate_samples": 0,
            "legacy_validation_duplicate_samples": 0,
        },
        "sample_identity": {
            "effective": "(selected_role_receipt_sha256, effective_index)",
            "source": "(candidate_chunk_set_sha256, global_logical_index)",
            "physical_source": "(chunk_payload_sha256, chunk_local_record_index)",
        },
        "claims": {
            "fixture_mode": fixture_mode,
            "label_blind_selection": True,
            "bounded_memory_exact_membership": True,
            "eligible_validation_role": not fixture_mode and source.get("dirty") is False,
            "training_started": False,
            "architecture_selected": True,
            "strength_evidence": False,
            "production_network": False,
        },
    }


def create_scale_selected_role(
    train_receipt: Path,
    candidate_receipt: Path,
    output_directory: Path,
    scratch_directory: Path,
    *,
    contract_path: Path | None = None,
    _allow_fixture: bool = False,
    _target_records: int | None = None,
    _source_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    contract, contract_sha256 = load_contract(contract_path, allow_fixture=_allow_fixture)
    selection_contract = _mapping(contract.get("validation_selection"), "validation selection")
    frozen_target = int(selection_contract["target_records"])
    target_records = _target_records if _target_records is not None else frozen_target
    _require(target_records > 0, "selected target must be positive")
    _require(_allow_fixture or target_records == frozen_target, "production target drifted")

    source = _repository_identity(_source_override)
    _require(_allow_fixture or source["dirty"] is False, "selector source tree is dirty")
    output = output_directory.expanduser().resolve()
    scratch = scratch_directory.expanduser().resolve()
    _require(output.parent.is_dir(), f"output parent does not exist: {output.parent}")
    _require(scratch.parent.is_dir(), f"scratch parent does not exist: {scratch.parent}")
    _require(not output.exists(), f"output already exists: {output}")
    _require(not scratch.exists(), f"scratch already exists: {scratch}")
    _require(output != scratch, "output and scratch directories are identical")
    scratch.mkdir()

    contract_resolved = (
        contract_path or REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH
    ).expanduser().resolve()
    with HordeChunkSetDataset(train_receipt, contract_resolved) as train, HordeChunkSetDataset(
        candidate_receipt, contract_resolved
    ) as candidate:
        train_identity, candidate_identity = _validate_sources(train, candidate, contract)
        training_inventory = _partition_training(train, scratch)
        candidate_inventory, candidate_keys = _partition_candidate(candidate, scratch)
        training_duplicates, rejection_flags = _mark_training_overlaps(
            scratch,
            len(candidate),
            training_inventory,
            candidate_inventory,
        )
        selected = _select_records(
            candidate,
            scratch,
            target_records,
            candidate_keys,
            rejection_flags,
        )

    receipt = _build_receipt(
        contract_sha256=contract_sha256,
        source=source,
        train_identity=train_identity,
        candidate_identity=candidate_identity,
        target_records=target_records,
        selected=selected,
        training_inventory=training_inventory,
        candidate_inventory=candidate_inventory,
        candidate_keys=candidate_keys,
        rejection_flags=rejection_flags,
        training_duplicates=training_duplicates,
        fixture_mode=_allow_fixture,
    )
    output.mkdir()
    _write_exclusive(output / INDEX_FILENAME, selected["index_payload"])
    _write_exclusive(output / RECORDS_FILENAME, selected["records_payload"])
    _write_exclusive(output / RECEIPT_FILENAME, _canonical_json(receipt))
    return receipt


class ScaleSelectedRoleDataset:
    """Read-only view of a scale selected-role materialization."""

    def __init__(
        self,
        receipt_path: Path,
        contract_path: Path | None = None,
        *,
        _allow_fixture: bool = False,
    ) -> None:
        self.receipt_path = receipt_path.expanduser().resolve()
        self.receipt, payload = _read_json(self.receipt_path, "scale selected-role receipt")
        _require(payload == _canonical_json(self.receipt), "selected-role receipt is not canonical")
        _require(self.receipt.get("schema") == SCHEMA, "selected-role schema mismatch")
        self.receipt_sha256 = _sha256_bytes(payload)
        _, contract_sha256 = load_contract(contract_path, allow_fixture=_allow_fixture)
        _require(
            self.receipt.get("contract")
            == {"schema": CONTRACT_SCHEMA, "sha256": contract_sha256},
            "selected-role contract identity drifted",
        )
        selection = _mapping(self.receipt.get("selection"), "selected-role selection")
        output = _mapping(self.receipt.get("materialized_output"), "materialized output")
        index_artifact = _mapping(selection.get("selected_indices"), "selected indices")
        _require(selection.get("algorithm") == ALGORITHM, "selected-role algorithm drifted")
        _require(
            selection.get("candidate_order")
            == "chunk index ascending, then local record index ascending"
            and selection.get("selected_index_encoding")
            == "uint64_little_endian_global_logical_index",
            "selected-role ordering drifted",
        )
        exact_index = _mapping(selection.get("exact_membership_index"), "exact membership index")
        _require(
            exact_index.get("schema") == INDEX_ALGORITHM
            and exact_index.get("bucket_count") == BUCKET_COUNT
            and _valid_sha256(exact_index.get("training_inventory_sha256"))
            and _valid_sha256(exact_index.get("candidate_inventory_sha256")),
            "exact membership index receipt drifted",
        )

        self.index_path = self.receipt_path.parent / INDEX_FILENAME
        self.path = self.receipt_path.parent / RECORDS_FILENAME
        _require(self.index_path.is_file(), "selected index artifact is missing")
        _require(self.path.is_file(), "selected record artifact is missing")
        index_payload = self.index_path.read_bytes()
        _require(index_artifact.get("name") == INDEX_FILENAME, "selected index filename drifted")
        _require(len(index_payload) == index_artifact.get("bytes"), "selected index size drifted")
        _require(_sha256_bytes(index_payload) == index_artifact.get("sha256"), "selected index hash drifted")
        target = selection.get("target_records")
        _require(type(target) is int and target > 0, "selected target is invalid")
        _require(len(index_payload) == target * 8, "selected index framing drifted")
        self.source_indices = tuple(
            struct.unpack_from("<Q", index_payload, offset)[0]
            for offset in range(0, len(index_payload), 8)
        )
        _require(
            all(left < right for left, right in zip(self.source_indices, self.source_indices[1:])),
            "selected indices are not strictly increasing",
        )
        _require(selection.get("accepted_records") == target, "accepted record count drifted")
        _require(
            selection.get("cutoff_candidate_index") == self.source_indices[-1],
            "selected cutoff index drifted",
        )

        self._file = self.path.open("rb")
        self._mapping: mmap.mmap | None = None
        try:
            size = os.fstat(self._file.fileno()).st_size
            _require(output.get("name") == RECORDS_FILENAME, "selected record filename drifted")
            _require(size == output.get("bytes") == target * wire.RECORD_SIZE, "selected record framing drifted")
            observed = _sha256_file(self.path)
            _require(observed == output.get("sha256") == output.get("payload_sha256"), "selected record hash drifted")
            _require(output.get("record_count") == target, "selected record count drifted")
            self.file_sha256 = observed
            self.payload_sha256 = observed
            self.header_sha256 = self.receipt_sha256
            self.manifest_sha256 = self.receipt_sha256
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            self.close()
            raise

        candidate = _mapping(self.receipt.get("candidate_source"), "candidate source")
        candidate_manifest = _mapping(candidate.get("manifest"), "candidate manifest")
        self.manifest = {
            "schema": SCHEMA,
            "schema_sha256": contract_sha256,
            "record_count": target,
            "payload_sha256": self.payload_sha256,
            "source_commit": candidate_manifest["source_commit"],
            "source_dirty": candidate_manifest["source_dirty"],
            "producer_sha256": candidate_manifest["producer_sha256"],
            "book_sha256": candidate_manifest["book_sha256"],
            "network": candidate_manifest["network"],
            "label_contract": candidate_manifest["label_contract"],
            "selection_source": {
                "candidate_chunk_set_sha256": candidate["chunk_set_sha256"],
                "candidate_logical_payload_sha256": candidate["logical_payload_sha256"],
            },
        }

    def __enter__(self) -> "ScaleSelectedRoleDataset":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __len__(self) -> int:
        return int(self.manifest["record_count"])

    def close(self) -> None:
        if getattr(self, "_mapping", None) is not None:
            self._mapping.close()
            self._mapping = None
        if hasattr(self, "_file") and not self._file.closed:
            self._file.close()

    def raw_record(self, index: int) -> bytes:
        _require(0 <= index < len(self), f"record index {index} is out of range")
        _require(self._mapping is not None, "selected-role dataset is closed")
        offset = index * wire.RECORD_SIZE
        return bytes(self._mapping[offset : offset + wire.RECORD_SIZE])

    def record(self, index: int) -> TrainingRecord:
        decoded = decode_training_record(self.raw_record(index), index)
        return TrainingRecord(
            index=decoded.index,
            features=decoded.features,
            side_to_move=decoded.side_to_move,
            rule50_count=decoded.rule50_count,
            game_ply=decoded.game_ply,
            score=decoded.score,
            best_move=decoded.best_move,
            played_move=decoded.played_move,
            result=decoded.result,
            outcome_reason=decoded.outcome_reason,
            board=decoded.board,
            castling_rights=decoded.castling_rights,
            ep_square=decoded.ep_square,
            source_payload_sha256=self.payload_sha256,
        )

    def label(self, index: int) -> tuple[int, int, int, int]:
        decoded = wire.validate_record(self.raw_record(index), index)
        return decoded["side"], decoded["score"], decoded["result"], decoded["reason"]

    def batches(self, batch_size: int) -> Iterator[SparseBatch]:
        _require(batch_size > 0, "batch size must be positive")
        for begin in range(0, len(self), batch_size):
            end = min(begin + batch_size, len(self))
            yield make_sparse_batch(tuple(self.record(index) for index in range(begin, end)))

    def identity(self) -> dict[str, object]:
        candidate = _mapping(self.receipt.get("candidate_source"), "candidate source")
        selection = _mapping(self.receipt.get("selection"), "selection")
        return {
            "name": RECORDS_FILENAME,
            "sha256": self.file_sha256,
            "payload_sha256": self.payload_sha256,
            "manifest_sha256": self.manifest_sha256,
            "records": len(self),
            "book_sha256": self.manifest["book_sha256"],
            "selected_role": {
                "schema": SCHEMA,
                "receipt_name": self.receipt_path.name,
                "receipt_sha256": self.receipt_sha256,
                "contract_sha256": self.receipt["contract"]["sha256"],
                "candidate_chunk_set_sha256": candidate["chunk_set_sha256"],
                "candidate_logical_payload_sha256": candidate["logical_payload_sha256"],
                "selected_index_sha256": selection["selected_indices"]["sha256"],
                "decision_chain_sha256": selection["decision_chain_sha256"],
                "record_order_sha256": self.receipt["materialized_output"]["record_order_sha256"],
            },
        }


def verify_scale_selected_role(
    train_receipt: Path,
    candidate_receipt: Path,
    selected_receipt: Path,
    *,
    contract_path: Path | None = None,
) -> dict[str, object]:
    contract, contract_sha256 = load_contract(contract_path)
    source = _repository_identity()
    _require(source["dirty"] is False, "selector source tree is dirty")
    contract_resolved = (contract_path or REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH).resolve()
    with HordeChunkSetDataset(train_receipt, contract_resolved) as train, HordeChunkSetDataset(
        candidate_receipt, contract_resolved
    ) as candidate, ScaleSelectedRoleDataset(selected_receipt, contract_resolved) as selected:
        train_identity, candidate_identity = _validate_sources(train, candidate, contract)
        receipt = selected.receipt
        _require(receipt.get("contract") == {"schema": CONTRACT_SCHEMA, "sha256": contract_sha256}, "contract receipt drifted")
        _require(receipt.get("selector_source") == source, "selector source identity drifted")
        _require(receipt.get("training_reference") == train_identity, "training reference drifted")
        _require(receipt.get("candidate_source") == candidate_identity, "candidate source drifted")
        _require(
            receipt.get("postconditions")
            == {
                "physical_cross_role_overlap_samples": 0,
                "legacy_cross_role_overlap_samples": 0,
                "physical_validation_duplicate_samples": 0,
                "legacy_validation_duplicate_samples": 0,
            },
            "selected-role postconditions drifted",
        )
        identity = selected.identity()
    return {
        "schema": VERIFICATION_SCHEMA,
        "contract_sha256": contract_sha256,
        "selected_role": identity,
        "claims": {
            "chunk_sets_reauthenticated": True,
            "selected_artifacts_reauthenticated": True,
            "source_binding_verified": True,
            "deep_selection_replay": False,
            "strength_evidence": False,
            "production_network": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="materialize the scale selected role")
    create.add_argument("train", type=Path, help="training chunk-set receipt")
    create.add_argument("candidate", type=Path, help="candidate chunk-set receipt")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--scratch", type=Path, required=True)
    create.add_argument("--contract", type=Path)
    verify = subparsers.add_parser("verify", help="reauthenticate a scale selected role")
    verify.add_argument("train", type=Path)
    verify.add_argument("candidate", type=Path)
    verify.add_argument("receipt", type=Path)
    verify.add_argument("--contract", type=Path)
    verify.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "create":
        result = create_scale_selected_role(
            args.train,
            args.candidate,
            args.output,
            args.scratch,
            contract_path=args.contract,
        )
    else:
        result = verify_scale_selected_role(
            args.train,
            args.candidate,
            args.receipt,
            contract_path=args.contract,
        )
    payload = _canonical_json(result)
    if args.command == "verify" and args.output is not None:
        _write_exclusive(args.output.expanduser().resolve(), payload)
    print(payload.decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        ScaleSelectedRoleError,
        subprocess.SubprocessError,
        wire.FormatError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
