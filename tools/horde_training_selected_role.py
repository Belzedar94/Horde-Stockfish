#!/usr/bin/env python3
"""Build and verify a label-blind selected Horde training-data role."""

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
    from .horde_training_decoder import (
        HordeBinV1Dataset,
        SparseBatch,
        TrainingRecord,
        decode_training_record,
        legacy_model_input_key,
        make_sparse_batch,
        physical_position_key,
    )
except ImportError:
    import horde_bin_v1 as wire
    from horde_training_decoder import (
        HordeBinV1Dataset,
        SparseBatch,
        TrainingRecord,
        decode_training_record,
        legacy_model_input_key,
        make_sparse_batch,
        physical_position_key,
    )


SCHEMA = "HORDE_TRAINING_SELECTED_ROLE_V1"
ALGORITHM = "FIRST_ELIGIBLE_DUAL_KEY_V1"
CONTRACT_SCHEMA = "HORDE_V2_C1_DATA_REPAIR_V1"
CONTRACT_RELATIVE_PATH = Path("schemas/horde-v2-c1-data-repair-v1.json")
CONTRACT_SHA256 = "307B7CA068A025B74443D1A657789D6AB0FDFA34E8EEB6A573E025ACA218535D"
INDEX_FILENAME = "selected-indices.bin"
RECORDS_FILENAME = "selected-records.bin"
RECEIPT_FILENAME = "receipt.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SELECTOR_RELATIVE_PATH = Path("tools/horde_training_selected_role.py")

REJECT_TRAIN_PHYSICAL = 1
REJECT_TRAIN_LEGACY = 2
REJECT_SELECTED_PHYSICAL = 4
REJECT_SELECTED_LEGACY = 8


class SelectedRoleError(ValueError):
    """Raised when a selected role violates its frozen data contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectedRoleError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"{label} does not exist: {resolved}")
    payload = resolved.read_bytes()
    try:
        root = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelectedRoleError(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(root, dict), f"{label} root is not an object")
    return root, payload


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


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


def _repository_identity(root: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    _require(_valid_commit(commit), "selector source is not a full Git identity")
    return {"commit": commit.lower(), "dirty": dirty}


def _normalized_generation(manifest: Mapping[str, Any]) -> dict[str, Any]:
    generation = _mapping(manifest.get("generation"), "generation manifest")
    ignored = {"requested_records", "seed", "opening_count"}
    return {key: value for key, value in generation.items() if key not in ignored}


def direct_dataset_identity(dataset: HordeBinV1Dataset) -> dict[str, object]:
    return {
        "file_sha256": dataset.file_sha256,
        "header_sha256": dataset.header_sha256,
        "manifest_sha256": dataset.manifest_sha256,
        "payload_sha256": dataset.manifest["payload_sha256"],
        "record_count": len(dataset),
        "book_sha256": dataset.manifest["book_sha256"],
        "seed": dataset.manifest["generation"]["seed"],
        "manifest": dataset.manifest,
    }


def load_contract(path: Path | None = None) -> tuple[dict[str, Any], str]:
    resolved = (path or REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH).expanduser().resolve()
    contract, payload = _read_json(resolved, "C1 data-repair contract")
    digest = _sha256_bytes(payload)
    _require(digest == CONTRACT_SHA256, f"C1 data-repair contract SHA-256 mismatch: {digest}")
    _require(contract.get("schema_name") == CONTRACT_SCHEMA, "data-repair schema drifted")
    candidate = _mapping(contract.get("candidate"), "data-repair candidate")
    selection = _mapping(contract.get("selection"), "data-repair selection")
    postconditions = _mapping(contract.get("postconditions"), "data-repair postconditions")
    _require(
        candidate.get("schema") == wire.SCHEMA_NAME
        and candidate.get("schema_sha256") == wire.SCHEMA_SHA256,
        "candidate record schema drifted",
    )
    _require(
        selection.get("schema") == SCHEMA
        and selection.get("algorithm") == ALGORITHM
        and selection.get("role") == "validation"
        and selection.get("target_records") == 250_000
        and selection.get("candidate_order") == "local_record_index_ascending"
        and selection.get("selected_index_encoding") == "uint64_little_endian"
        and selection.get("reject_training_physical_key") is True
        and selection.get("reject_training_legacy_model_input_key") is True
        and selection.get("reject_selected_physical_duplicate") is True
        and selection.get("reject_selected_legacy_model_input_duplicate") is True
        and selection.get("label_blind") is True,
        "selected-role algorithm drifted",
    )
    _require(
        postconditions.get("effective_validation_records") == 250_000
        and postconditions.get("physical_cross_role_overlap_samples") == 0
        and postconditions.get("legacy_cross_role_overlap_samples") == 0
        and postconditions.get("physical_validation_duplicate_samples") == 0
        and postconditions.get("legacy_validation_duplicate_samples") == 0
        and postconditions.get("insufficient_candidate_records_fail_closed") is True,
        "selected-role postconditions drifted",
    )
    return contract, digest


def _identity_matches_contract(
    identity: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    for key in (
        "file_sha256",
        "header_sha256",
        "manifest_sha256",
        "payload_sha256",
        "record_count",
        "book_sha256",
    ):
        _require(identity.get(key) == expected.get(key), f"{label} field {key} drifted")


def _validate_direct_sources(
    train: HordeBinV1Dataset,
    candidate: HordeBinV1Dataset,
    contract: Mapping[str, Any],
    *,
    fixture_mode: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    train_identity = direct_dataset_identity(train)
    candidate_identity = direct_dataset_identity(candidate)
    if not fixture_mode:
        _identity_matches_contract(
            train_identity,
            _mapping(contract.get("training_reference"), "training reference"),
            "training reference",
        )
        frozen_candidate = _mapping(contract.get("candidate"), "candidate contract")
        _require(
            candidate_identity["record_count"] == frozen_candidate.get("requested_records"),
            "candidate record count drifted",
        )
        _require(
            candidate_identity["seed"] == frozen_candidate.get("seed"),
            "candidate seed drifted",
        )
        _require(
            candidate_identity["book_sha256"] == frozen_candidate.get("book_sha256"),
            "candidate book hash drifted",
        )

    train_manifest = train.manifest
    candidate_manifest = candidate.manifest
    _require(train.path != candidate.path, "training and candidate paths are identical")
    _require(train.file_sha256 != candidate.file_sha256, "training and candidate files match")
    _require(
        train_manifest["book_sha256"] != candidate_manifest["book_sha256"],
        "training and candidate use the same opening book",
    )
    for field in ("schema", "schema_sha256", "source_commit", "source_dirty", "producer_sha256"):
        _require(
            train_manifest[field] == candidate_manifest[field],
            f"training and candidate manifest field {field} differs",
        )
    _require(
        train_manifest["network"] == candidate_manifest["network"],
        "training and candidate teacher networks differ",
    )
    _require(
        train_manifest["label_contract"] == candidate_manifest["label_contract"],
        "training and candidate label contracts differ",
    )
    _require(
        _normalized_generation(train_manifest) == _normalized_generation(candidate_manifest),
        "training and candidate generation settings differ",
    )
    return train_identity, candidate_identity


def _key_digest(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def _compute_selection(
    train: HordeBinV1Dataset,
    candidate: HordeBinV1Dataset,
    target_records: int,
) -> dict[str, object]:
    _require(target_records > 0, "selected-role target must be positive")
    train_physical: set[bytes] = set()
    train_legacy: set[bytes] = set()
    train_physical_duplicates = 0
    train_legacy_duplicates = 0
    for index in range(len(train)):
        record = train.record(index)
        physical = _key_digest(physical_position_key(record))
        legacy = _key_digest(legacy_model_input_key(record))
        train_physical_duplicates += physical in train_physical
        train_legacy_duplicates += legacy in train_legacy
        train_physical.add(physical)
        train_legacy.add(legacy)

    selected_physical: set[bytes] = set()
    selected_legacy: set[bytes] = set()
    selected_indices: list[int] = []
    selected_records = bytearray()
    rejection_masks: Counter[int] = Counter()
    decision_chain = hashlib.sha256()
    record_order = hashlib.sha256()
    examined = 0

    for index in range(len(candidate)):
        record = candidate.record(index)
        physical = _key_digest(physical_position_key(record))
        legacy = _key_digest(legacy_model_input_key(record))
        mask = 0
        if physical in train_physical:
            mask |= REJECT_TRAIN_PHYSICAL
        if legacy in train_legacy:
            mask |= REJECT_TRAIN_LEGACY
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

    _require(
        len(selected_indices) == target_records,
        f"candidate exhausted after {len(selected_indices)} eligible records; "
        f"target is {target_records}",
    )
    index_payload = b"".join(struct.pack("<Q", index) for index in selected_indices)
    records_payload = bytes(selected_records)
    _require(
        len(records_payload) == target_records * wire.RECORD_SIZE,
        "materialized record payload has invalid framing",
    )
    return {
        "index_payload": index_payload,
        "records_payload": records_payload,
        "selection": {
            "algorithm": ALGORITHM,
            "candidate_order": "local_record_index_ascending",
            "selected_index_encoding": "uint64_little_endian",
            "records_examined": examined,
            "cutoff_candidate_index": selected_indices[-1],
            "accepted_records": target_records,
            "rejected_records": examined - target_records,
            "rejection_reason_masks": {
                str(mask): rejection_masks[mask] for mask in sorted(rejection_masks)
            },
            "decision_chain_sha256": decision_chain.hexdigest().upper(),
            "selected_index_sha256": _sha256_bytes(index_payload),
            "materialized_payload_sha256": _sha256_bytes(records_payload),
            "record_order_sha256": record_order.hexdigest().upper(),
        },
        "training_duplicates": {
            "physical": train_physical_duplicates,
            "legacy_model_input": train_legacy_duplicates,
        },
    }


def _selector_source(
    source_override: Mapping[str, object] | None,
) -> dict[str, object]:
    source = (
        dict(source_override)
        if source_override is not None
        else _repository_identity(REPOSITORY_ROOT)
    )
    _require(
        _valid_commit(source.get("commit")) and type(source.get("dirty")) is bool,
        "selector source override is invalid",
    )
    selector_path = REPOSITORY_ROOT / SELECTOR_RELATIVE_PATH
    return {
        **source,
        "path": SELECTOR_RELATIVE_PATH.as_posix(),
        "file_sha256": sha256_file(selector_path),
    }


def _build_receipt(
    contract_sha256: str,
    source: Mapping[str, object],
    train_identity: Mapping[str, object],
    candidate_identity: Mapping[str, object],
    computed: Mapping[str, object],
    *,
    target_records: int,
    fixture_mode: bool,
) -> dict[str, object]:
    selection = _mapping(computed.get("selection"), "computed selection")
    index_payload = computed["index_payload"]
    records_payload = computed["records_payload"]
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
            "algorithm": selection["algorithm"],
            "candidate_order": selection["candidate_order"],
            "selected_index_encoding": selection["selected_index_encoding"],
            "target_records": target_records,
            "records_examined": selection["records_examined"],
            "cutoff_candidate_index": selection["cutoff_candidate_index"],
            "accepted_records": selection["accepted_records"],
            "rejected_records": selection["rejected_records"],
            "rejection_reason_masks": selection["rejection_reason_masks"],
            "decision_chain_sha256": selection["decision_chain_sha256"],
            "selected_indices": {
                "name": INDEX_FILENAME,
                "bytes": len(index_payload),
                "sha256": selection["selected_index_sha256"],
            },
        },
        "materialized_output": {
            "name": RECORDS_FILENAME,
            "bytes": len(records_payload),
            "sha256": selection["materialized_payload_sha256"],
            "payload_sha256": selection["materialized_payload_sha256"],
            "record_count": target_records,
            "record_order_sha256": selection["record_order_sha256"],
        },
        "training_internal_duplicates": computed["training_duplicates"],
        "postconditions": {
            "physical_cross_role_overlap_samples": 0,
            "legacy_cross_role_overlap_samples": 0,
            "physical_validation_duplicate_samples": 0,
            "legacy_validation_duplicate_samples": 0,
        },
        "sample_identity": {
            "effective": "(selected_role_receipt_sha256, effective_index)",
            "source": "(candidate_payload_sha256, candidate_local_index)",
        },
        "claims": {
            "fixture_mode": fixture_mode,
            "label_blind_selection": True,
            "eligible_validation_role": not fixture_mode and source.get("dirty") is False,
            "training_started": False,
            "architecture_selected": False,
            "strength_evidence": False,
            "production_network": False,
        },
    }


def create_selected_role(
    train_path: Path,
    candidate_path: Path,
    output_directory: Path,
    *,
    contract_path: Path | None = None,
    _allow_fixture: bool = False,
    _target_records: int | None = None,
    _source_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    contract, contract_sha256 = load_contract(contract_path)
    frozen_target = int(_mapping(contract.get("selection"), "selection contract")["target_records"])
    target_records = _target_records if _target_records is not None else frozen_target
    _require(
        not _allow_fixture or target_records > 0,
        "fixture selected-role target must be positive",
    )
    _require(
        _allow_fixture or target_records == frozen_target,
        "production selected-role target drifted",
    )
    source = _selector_source(_source_override)
    _require(_allow_fixture or source["dirty"] is False, "selector source tree is dirty")
    output = output_directory.expanduser().resolve()
    _require(output.parent.is_dir(), f"output parent does not exist: {output.parent}")
    _require(not output.exists(), f"output already exists: {output}")

    with HordeBinV1Dataset(train_path) as train, HordeBinV1Dataset(candidate_path) as candidate:
        train_identity, candidate_identity = _validate_direct_sources(
            train,
            candidate,
            contract,
            fixture_mode=_allow_fixture,
        )
        computed = _compute_selection(train, candidate, target_records)
    receipt = _build_receipt(
        contract_sha256,
        source,
        train_identity,
        candidate_identity,
        computed,
        target_records=target_records,
        fixture_mode=_allow_fixture,
    )

    output.mkdir()
    _write_exclusive(output / INDEX_FILENAME, computed["index_payload"])
    _write_exclusive(output / RECORDS_FILENAME, computed["records_payload"])
    _write_exclusive(output / RECEIPT_FILENAME, _canonical_json(receipt))
    return receipt


class SelectedRoleDataset:
    """Read-only view of one authenticated materialized selected role."""

    def __init__(self, receipt_path: Path) -> None:
        self.receipt_path = receipt_path.expanduser().resolve()
        self.receipt, payload = _read_json(self.receipt_path, "selected-role receipt")
        _require(payload == _canonical_json(self.receipt), "selected-role receipt is not canonical")
        _require(self.receipt.get("schema") == SCHEMA, "selected-role schema mismatch")
        self.receipt_sha256 = _sha256_bytes(payload)
        contract = _mapping(self.receipt.get("contract"), "selected-role contract")
        _require(
            contract.get("schema") == CONTRACT_SCHEMA
            and contract.get("sha256") == CONTRACT_SHA256,
            "selected-role contract identity drifted",
        )
        selection = _mapping(self.receipt.get("selection"), "selected-role selection")
        output = _mapping(self.receipt.get("materialized_output"), "materialized output")
        index_artifact = _mapping(selection.get("selected_indices"), "selected index artifact")
        _require(selection.get("algorithm") == ALGORITHM, "selected-role algorithm mismatch")
        _require(
            selection.get("candidate_order") == "local_record_index_ascending"
            and selection.get("selected_index_encoding") == "uint64_little_endian",
            "selected-role ordering contract drifted",
        )
        for artifact, expected_name, label in (
            (index_artifact, INDEX_FILENAME, "selected indices"),
            (output, RECORDS_FILENAME, "selected records"),
        ):
            name = artifact.get("name")
            _require(name == expected_name, f"{label} filename drifted")
            _require(Path(str(name)).name == name, f"{label} path is not a basename")
            _require(
                type(artifact.get("bytes")) is int and artifact["bytes"] >= 0,
                f"{label} size is invalid",
            )
            _require(_valid_sha256(artifact.get("sha256")), f"{label} hash is invalid")

        self.index_path = self.receipt_path.parent / INDEX_FILENAME
        self.path = self.receipt_path.parent / RECORDS_FILENAME
        _require(self.index_path.is_file(), "selected index artifact is missing")
        _require(self.path.is_file(), "selected record artifact is missing")
        index_payload = self.index_path.read_bytes()
        _require(len(index_payload) == index_artifact["bytes"], "selected index size drifted")
        _require(
            _sha256_bytes(index_payload) == index_artifact["sha256"],
            "selected index hash drifted",
        )
        target = selection.get("target_records")
        _require(type(target) is int and target > 0, "selected-role target is invalid")
        _require(len(index_payload) == target * 8, "selected index framing drifted")
        self.source_indices = tuple(
            struct.unpack_from("<Q", index_payload, offset)[0]
            for offset in range(0, len(index_payload), 8)
        )
        _require(
            all(
                left < right
                for left, right in zip(self.source_indices, self.source_indices[1:])
            ),
            "selected source indices are not strictly increasing",
        )
        _require(selection.get("accepted_records") == target, "selected accepted count drifted")
        _require(
            selection.get("cutoff_candidate_index") == self.source_indices[-1],
            "selected cutoff drifted",
        )

        self._file = self.path.open("rb")
        self._mapping: mmap.mmap | None = None
        try:
            actual_size = os.fstat(self._file.fileno()).st_size
            _require(actual_size == output["bytes"], "selected record size drifted")
            _require(actual_size == target * wire.RECORD_SIZE, "selected record framing drifted")
            observed_sha = sha256_file(self.path)
            _require(observed_sha == output["sha256"], "selected record hash drifted")
            _require(output.get("payload_sha256") == observed_sha, "selected payload hash drifted")
            _require(output.get("record_count") == target, "selected record count drifted")
            self.file_sha256 = observed_sha
            self.payload_sha256 = observed_sha
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
            "schema_sha256": CONTRACT_SHA256,
            "record_count": target,
            "payload_sha256": self.payload_sha256,
            "source_commit": candidate_manifest["source_commit"],
            "source_dirty": candidate_manifest["source_dirty"],
            "producer_sha256": candidate_manifest["producer_sha256"],
            "book_sha256": candidate_manifest["book_sha256"],
            "network": candidate_manifest["network"],
            "label_contract": candidate_manifest["label_contract"],
            "generation": candidate_manifest["generation"],
        }

    def __enter__(self) -> "SelectedRoleDataset":
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
        _require(batch_size > 0, f"invalid batch size {batch_size}")
        for begin in range(0, len(self), batch_size):
            end = min(begin + batch_size, len(self))
            yield make_sparse_batch(tuple(self.record(index) for index in range(begin, end)))

    def identity(self) -> dict[str, object]:
        candidate = _mapping(self.receipt.get("candidate_source"), "candidate source")
        selection = _mapping(self.receipt.get("selection"), "selected-role selection")
        return {
            "name": RECORDS_FILENAME,
            "sha256": self.file_sha256,
            "payload_sha256": self.payload_sha256,
            "records": len(self),
            "book_sha256": self.manifest["book_sha256"],
            "seed": self.manifest["generation"]["seed"],
            "selected_role": {
                "schema": SCHEMA,
                "receipt_name": self.receipt_path.name,
                "receipt_sha256": self.receipt_sha256,
                "contract_sha256": CONTRACT_SHA256,
                "candidate_file_sha256": candidate["file_sha256"],
                "candidate_payload_sha256": candidate["payload_sha256"],
                "selected_index_sha256": selection["selected_indices"]["sha256"],
                "decision_chain_sha256": selection["decision_chain_sha256"],
                "record_order_sha256": self.receipt["materialized_output"]["record_order_sha256"],
            },
        }


def validate_selected_role_binding(
    train: HordeBinV1Dataset,
    candidate: HordeBinV1Dataset,
    selected: SelectedRoleDataset,
    *,
    _allow_fixture: bool = False,
    _source_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Authenticate a selected role and its direct parents without rescanning keys."""

    contract, contract_sha256 = load_contract()
    train_identity, candidate_identity = _validate_direct_sources(
        train,
        candidate,
        contract,
        fixture_mode=_allow_fixture,
    )
    receipt = selected.receipt
    source = _selector_source(_source_override)
    claims = _mapping(receipt.get("claims"), "selected-role claims")
    _require(claims.get("fixture_mode") is _allow_fixture, "selected-role fixture claim drifted")
    _require(
        _allow_fixture or claims.get("eligible_validation_role") is True,
        "selected role is not production-eligible",
    )
    _require(
        receipt.get("contract")
        == {"schema": CONTRACT_SCHEMA, "sha256": contract_sha256},
        "selected-role contract receipt drifted",
    )
    _require(receipt.get("selector_source") == source, "selector source identity drifted")
    _require(receipt.get("training_reference") == train_identity, "training reference drifted")
    _require(receipt.get("candidate_source") == candidate_identity, "candidate source drifted")
    postconditions = _mapping(receipt.get("postconditions"), "selected-role postconditions")
    _require(
        postconditions
        == {
            "physical_cross_role_overlap_samples": 0,
            "legacy_cross_role_overlap_samples": 0,
            "physical_validation_duplicate_samples": 0,
            "legacy_validation_duplicate_samples": 0,
        },
        "selected-role postconditions drifted",
    )
    return selected.identity()


def verify_selected_role(
    train_path: Path,
    candidate_path: Path,
    receipt_path: Path,
    *,
    contract_path: Path | None = None,
    _allow_fixture: bool = False,
    _source_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    contract, contract_sha256 = load_contract(contract_path)
    source = _selector_source(_source_override)
    _require(_allow_fixture or source["dirty"] is False, "selector source tree is dirty")
    with HordeBinV1Dataset(train_path) as train, HordeBinV1Dataset(candidate_path) as candidate:
        train_identity, candidate_identity = _validate_direct_sources(
            train,
            candidate,
            contract,
            fixture_mode=_allow_fixture,
        )
        with SelectedRoleDataset(receipt_path) as selected:
            receipt = selected.receipt
            validate_selected_role_binding(
                train,
                candidate,
                selected,
                _allow_fixture=_allow_fixture,
                _source_override=source,
            )
            target = int(_mapping(receipt.get("selection"), "selection")["target_records"])
            if not _allow_fixture:
                _require(target == 250_000, "production selected-role target drifted")
            computed = _compute_selection(train, candidate, target)
            expected = _build_receipt(
                contract_sha256,
                source,
                train_identity,
                candidate_identity,
                computed,
                target_records=target,
                fixture_mode=_allow_fixture,
            )
            _require(
                receipt == expected,
                "selected-role receipt is not the canonical first-eligible result",
            )
            _require(
                selected.index_path.read_bytes() == computed["index_payload"],
                "selected index sequence differs from canonical selection",
            )
            _require(
                selected.path.read_bytes() == computed["records_payload"],
                "materialized records differ from canonical selection",
            )
            identity = selected.identity()
    return {
        "schema": "HORDE_TRAINING_SELECTED_ROLE_VERIFICATION_V1",
        "contract_sha256": contract_sha256,
        "selected_role": identity,
        "claims": {
            "fixture_mode": _allow_fixture,
            "canonical_selection_recomputed": True,
            "materialized_records_reconstructed": True,
            "zero_cross_role_overlap": True,
            "zero_validation_duplicates": True,
            "training_eligible": not _allow_fixture,
            "strength_evidence": False,
            "production_network": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create", help="materialize the frozen selected role")
    create.add_argument("train", type=Path)
    create.add_argument("candidate", type=Path)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--contract", type=Path)
    verify = subparsers.add_parser("verify", help="recompute and authenticate a selected role")
    verify.add_argument("train", type=Path)
    verify.add_argument("candidate", type=Path)
    verify.add_argument("receipt", type=Path)
    verify.add_argument("--contract", type=Path)
    verify.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "create":
        result = create_selected_role(
            args.train,
            args.candidate,
            args.output,
            contract_path=args.contract,
        )
    else:
        result = verify_selected_role(
            args.train,
            args.candidate,
            args.receipt,
            contract_path=args.contract,
        )
    payload = _canonical_json(result)
    if getattr(args, "output", None) is not None and args.command == "verify":
        _write_exclusive(args.output.expanduser().resolve(), payload)
    print(payload.decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, SelectedRoleError, subprocess.SubprocessError, wire.FormatError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
