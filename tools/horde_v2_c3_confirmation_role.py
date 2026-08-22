#!/usr/bin/env python3
"""Build and verify the fresh label-blind Horde V2 C3 confirmation role."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any, Iterator, Mapping, Protocol, Sequence

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
    from .horde_training_selected_role import SelectedRoleDataset
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
    from horde_training_selected_role import SelectedRoleDataset


SCHEMA = "HORDE_V2_C3_CONFIRMATION_ROLE_RECEIPT_V1"
CONTRACT_SCHEMA = "HORDE_V2_C3_REPRESENTATION_QUALIFICATION_V1"
CONTRACT_RELATIVE_PATH = Path("schemas/horde-v2-c3-representation-qualification-v1.json")
CONTRACT_SHA256 = "33F48B363AB6B4B20303E586E484DB7F45BF6BDBBFB5DF82C9FB034542A7B7DA"
ALGORITHM = "HASH_ORDERED_MULTI_ROLE_DUAL_KEY_V1"
ORDER_LABEL = b"HORDE_V2_C3_CONFIRMATION_ROLE_V1:select:"
SEED_LABEL = b"HORDE_V2_C3_CONFIRMATION_ROLE_V1:generator-seed"
INDEX_FILENAME = "selected-indices.bin"
RECORDS_FILENAME = "selected-records.bin"
RECEIPT_FILENAME = "receipt.json"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL_RELATIVE_PATH = Path("tools/horde_v2_c3_confirmation_role.py")


class ConfirmationRoleError(ValueError):
    """Raised when C3 confirmation-role evidence violates the frozen contract."""


class RecordDataset(Protocol):
    manifest: Mapping[str, Any]
    file_sha256: str
    payload_sha256: str
    header_sha256: str
    manifest_sha256: str

    def __len__(self) -> int: ...

    def record(self, index: int) -> TrainingRecord: ...

    def raw_record(self, index: int) -> bytes: ...


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfirmationRoleError(message)


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"{label} does not exist: {resolved}")
    payload = resolved.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConfirmationRoleError(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(value, dict), f"{label} root is not an object")
    return value, payload


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
        and all(character in "0123456789abcdef" for character in value)
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
    _require(_valid_commit(commit), "confirmation selector source is not a full Git identity")
    tool = root / TOOL_RELATIVE_PATH
    _require(tool.is_file(), "confirmation selector source file is missing")
    return {
        "commit": commit,
        "dirty": dirty,
        "path": TOOL_RELATIVE_PATH.as_posix(),
        "file_sha256": _sha256_file(tool),
    }


def load_contract(path: Path | None = None) -> tuple[dict[str, Any], str]:
    resolved = (path or REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH).expanduser().resolve()
    contract, payload = _read_json(resolved, "C3 representation contract")
    _require(payload == _canonical_json(contract), "C3 representation contract is not canonical JSON")
    digest = _sha256_bytes(payload)
    _require(digest == CONTRACT_SHA256, f"C3 representation contract SHA-256 mismatch: {digest}")
    _require(contract.get("schema_name") == CONTRACT_SCHEMA, "C3 contract schema drifted")
    role = _mapping(contract.get("confirmation_role"), "confirmation-role contract")
    generation = _mapping(role.get("generation"), "confirmation generation contract")
    selection = _mapping(role.get("selection"), "confirmation selection contract")
    derived = int.from_bytes(hashlib.sha256(SEED_LABEL).digest()[:8], "big") & ((1 << 63) - 1)
    _require(
        role.get("status") == "preregistered_unmaterialized"
        and generation.get("candidate_records") == 262_144
        and generation.get("seed") == derived
        and generation.get("seed_derivation_sha256") == _sha256_bytes(SEED_LABEL)
        and selection.get("selected_records") == 250_000
        and selection.get("fail_closed_if_eligible_records_below") == 250_000
        and selection.get("label_blind") is True
        and selection.get("candidate_order_key")
        == "SHA-256('HORDE_V2_C3_CONFIRMATION_ROLE_V1:select:' || uint64be(local_record_index)), ascending digest then ascending local_record_index",
        "C3 confirmation-role preregistration drifted",
    )
    return contract, digest


def _direct_identity(dataset: HordeBinV1Dataset) -> dict[str, object]:
    return {
        "name": dataset.path.name,
        "file_sha256": dataset.file_sha256,
        "header_sha256": dataset.header_sha256,
        "manifest_sha256": dataset.manifest_sha256,
        "payload_sha256": dataset.manifest["payload_sha256"],
        "record_count": len(dataset),
        "book_sha256": dataset.manifest["book_sha256"],
        "seed": dataset.manifest["generation"]["seed"],
        "manifest": dataset.manifest,
    }


def _selected_identity(dataset: SelectedRoleDataset) -> dict[str, object]:
    identity = dataset.identity()
    return {
        "name": identity["name"],
        "file_sha256": dataset.file_sha256,
        "header_sha256": dataset.header_sha256,
        "manifest_sha256": dataset.manifest_sha256,
        "payload_sha256": dataset.payload_sha256,
        "record_count": len(dataset),
        "book_sha256": dataset.manifest["book_sha256"],
        "seed": dataset.manifest["generation"]["seed"],
        "manifest": dataset.manifest,
        "selected_role": identity["selected_role"],
    }


def _open_exclusion(stack: ExitStack, path: Path) -> tuple[RecordDataset, dict[str, object]]:
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() == ".json":
        selected = stack.enter_context(SelectedRoleDataset(resolved))
        return selected, _selected_identity(selected)
    direct = stack.enter_context(HordeBinV1Dataset(resolved))
    return direct, _direct_identity(direct)


def _validate_sources(
    candidate: HordeBinV1Dataset,
    candidate_identity: Mapping[str, object],
    exclusion_identities: Sequence[Mapping[str, object]],
    contract: Mapping[str, Any],
    *,
    fixture_mode: bool,
) -> None:
    _require(candidate.manifest.get("schema") == wire.SCHEMA_NAME, "candidate schema drifted")
    _require(
        candidate.manifest.get("schema_sha256") == wire.SCHEMA_SHA256,
        "candidate schema hash drifted",
    )
    _require(len(exclusion_identities) == 2, "C3 requires exactly two exclusion roles")
    _require(
        all(candidate.file_sha256 != identity.get("file_sha256") for identity in exclusion_identities),
        "candidate duplicates an exclusion role",
    )
    if fixture_mode:
        return

    role = _mapping(contract.get("confirmation_role"), "confirmation-role contract")
    generation = _mapping(role.get("generation"), "confirmation generation contract")
    expected_book = _mapping(generation.get("book"), "confirmation book contract")
    expected_producer = _mapping(generation.get("producer"), "confirmation producer contract")
    expected_network = _mapping(generation.get("network"), "confirmation network contract")
    expected_labels = _mapping(generation.get("label_contract"), "confirmation label contract")
    manifest_generation = _mapping(candidate.manifest.get("generation"), "candidate generation manifest")
    _require(
        candidate_identity.get("name") == generation.get("candidate_output_name")
        and candidate_identity.get("record_count") == generation.get("candidate_records")
        and int(str(candidate_identity.get("seed"))) == generation.get("seed")
        and candidate_identity.get("book_sha256") == expected_book.get("sha256")
        and candidate.manifest.get("source_commit") == expected_producer.get("source_commit")
        and candidate.manifest.get("source_dirty") is False
        and candidate.manifest.get("producer_sha256") == expected_producer.get("sha256")
        and candidate.manifest.get("network") == expected_network
        and candidate.manifest.get("label_contract") == expected_labels,
        "candidate generation identity drifted",
    )
    configuration = _mapping(generation.get("configuration"), "generation configuration contract")
    manifest_names = {
        "threads": "threads",
        "hash_mb": "hash_mb",
        "depth": "depth",
        "nodes": "nodes",
        "random_move_min_ply": "random_move_min_ply",
        "random_move_max_ply": "random_move_max_ply",
        "random_move_count": "random_move_count",
        "random_multi_pv": "random_multi_pv",
        "random_multi_pv_diff": "random_multi_pv_diff",
        "write_min_ply": "write_min_ply",
        "write_max_ply": "write_max_ply",
        "max_game_ply": "max_game_ply",
    }
    for contract_name, manifest_name in manifest_names.items():
        _require(
            manifest_generation.get(manifest_name) == configuration.get(contract_name),
            f"candidate generation field {contract_name} drifted",
        )

    expected_exclusions = _mapping(role.get("selection"), "confirmation selection contract").get(
        "excluded_roles"
    )
    _require(isinstance(expected_exclusions, list), "excluded-role contract is invalid")
    _require(len(expected_exclusions) == len(exclusion_identities), "excluded-role count drifted")
    for index, (observed, expected_value) in enumerate(zip(exclusion_identities, expected_exclusions)):
        expected = _mapping(expected_value, f"excluded-role contract {index}")
        _require(
            observed.get("name") == expected.get("name")
            and observed.get("file_sha256") == expected.get("file_sha256")
            and observed.get("payload_sha256") == expected.get("payload_sha256")
            and observed.get("record_count") == expected.get("records"),
            f"excluded role {index} identity drifted",
        )


def _key_digest(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def _order_key(index: int) -> bytes:
    return hashlib.sha256(ORDER_LABEL + struct.pack(">Q", index)).digest()


def _compute_selection(
    candidate: HordeBinV1Dataset,
    exclusions: Sequence[RecordDataset],
    target_records: int,
) -> dict[str, object]:
    _require(target_records > 0, "confirmation target must be positive")
    exclusion_keys: list[tuple[set[bytes], set[bytes]]] = []
    exclusion_duplicates: list[dict[str, int]] = []
    for dataset in exclusions:
        physical_keys: set[bytes] = set()
        legacy_keys: set[bytes] = set()
        physical_duplicates = 0
        legacy_duplicates = 0
        for index in range(len(dataset)):
            record = dataset.record(index)
            physical = _key_digest(physical_position_key(record))
            legacy = _key_digest(legacy_model_input_key(record))
            physical_duplicates += physical in physical_keys
            legacy_duplicates += legacy in legacy_keys
            physical_keys.add(physical)
            legacy_keys.add(legacy)
        exclusion_keys.append((physical_keys, legacy_keys))
        exclusion_duplicates.append(
            {"physical": physical_duplicates, "legacy_model_input": legacy_duplicates}
        )

    seen_physical: set[bytes] = set()
    seen_legacy: set[bytes] = set()
    eligible: list[tuple[bytes, int]] = []
    rejection_masks: Counter[int] = Counter()
    decision_chain = hashlib.sha256()
    duplicate_physical_bit = 1 << (2 * len(exclusions))
    duplicate_legacy_bit = 1 << (2 * len(exclusions) + 1)

    for index in range(len(candidate)):
        record = candidate.record(index)
        physical = _key_digest(physical_position_key(record))
        legacy = _key_digest(legacy_model_input_key(record))
        mask = 0
        for exclusion_index, (physical_keys, legacy_keys) in enumerate(exclusion_keys):
            if physical in physical_keys:
                mask |= 1 << (2 * exclusion_index)
            if legacy in legacy_keys:
                mask |= 1 << (2 * exclusion_index + 1)
        if physical in seen_physical:
            mask |= duplicate_physical_bit
        if legacy in seen_legacy:
            mask |= duplicate_legacy_bit
        seen_physical.add(physical)
        seen_legacy.add(legacy)
        order = _order_key(index)
        decision_chain.update(struct.pack(">QI", index, mask))
        decision_chain.update(physical)
        decision_chain.update(legacy)
        decision_chain.update(order)
        if mask:
            rejection_masks[mask] += 1
        else:
            eligible.append((order, index))

    _require(
        len(eligible) >= target_records,
        f"candidate has {len(eligible)} eligible records after frozen exclusions; target is {target_records}",
    )
    eligible.sort(key=lambda item: (item[0], item[1]))
    selected = eligible[:target_records]
    selected_indices = [index for _, index in selected]
    index_payload = b"".join(struct.pack("<Q", index) for index in selected_indices)
    records_payload = b"".join(candidate.raw_record(index) for index in selected_indices)
    _require(
        len(records_payload) == target_records * wire.RECORD_SIZE,
        "confirmation record payload framing drifted",
    )
    record_order = hashlib.sha256()
    for index in selected_indices:
        record_order.update(struct.pack("<Q", index))
        record_order.update(candidate.raw_record(index))
    return {
        "index_payload": index_payload,
        "records_payload": records_payload,
        "exclusion_duplicates": exclusion_duplicates,
        "selection": {
            "algorithm": ALGORITHM,
            "candidate_order": "sha256_fixed_label_and_uint64be_local_index",
            "selected_index_encoding": "uint64_little_endian_in_selected_order",
            "candidate_records": len(candidate),
            "eligible_records": len(eligible),
            "accepted_records": target_records,
            "rejected_records": len(candidate) - len(eligible),
            "rejection_reason_masks": {
                str(mask): rejection_masks[mask] for mask in sorted(rejection_masks)
            },
            "decision_chain_sha256": decision_chain.hexdigest().upper(),
            "selected_indices_sha256": _sha256_bytes(index_payload),
            "selected_records_sha256": _sha256_bytes(records_payload),
            "record_order_sha256": record_order.hexdigest().upper(),
            "selected_local_index_minimum": min(selected_indices),
            "selected_local_index_maximum": max(selected_indices),
        },
    }


def _build_receipt(
    contract_sha256: str,
    source: Mapping[str, object],
    candidate_identity: Mapping[str, object],
    exclusion_identities: Sequence[Mapping[str, object]],
    computed: Mapping[str, object],
    *,
    target_records: int,
    fixture_mode: bool,
) -> dict[str, object]:
    selection = _mapping(computed.get("selection"), "computed confirmation selection")
    index_payload = computed.get("index_payload")
    records_payload = computed.get("records_payload")
    _require(isinstance(index_payload, bytes), "confirmation index payload is invalid")
    _require(isinstance(records_payload, bytes), "confirmation record payload is invalid")
    return {
        "schema": SCHEMA,
        "contract": {"schema": CONTRACT_SCHEMA, "sha256": contract_sha256},
        "role": "fresh_label_blind_confirmation",
        "record_schema": {
            "schema": wire.SCHEMA_NAME,
            "schema_sha256": wire.SCHEMA_SHA256,
            "record_bytes": wire.RECORD_SIZE,
        },
        "selector_source": dict(source),
        "candidate_source": dict(candidate_identity),
        "excluded_roles": [dict(identity) for identity in exclusion_identities],
        "selection": {
            "algorithm": selection["algorithm"],
            "candidate_order": selection["candidate_order"],
            "selected_index_encoding": selection["selected_index_encoding"],
            "target_records": target_records,
            "candidate_records": selection["candidate_records"],
            "eligible_records": selection["eligible_records"],
            "accepted_records": selection["accepted_records"],
            "rejected_records": selection["rejected_records"],
            "rejection_reason_masks": selection["rejection_reason_masks"],
            "decision_chain_sha256": selection["decision_chain_sha256"],
            "selected_local_index_minimum": selection["selected_local_index_minimum"],
            "selected_local_index_maximum": selection["selected_local_index_maximum"],
            "selected_indices": {
                "name": INDEX_FILENAME,
                "bytes": len(index_payload),
                "sha256": selection["selected_indices_sha256"],
            },
        },
        "materialized_output": {
            "name": RECORDS_FILENAME,
            "bytes": len(records_payload),
            "sha256": selection["selected_records_sha256"],
            "payload_sha256": selection["selected_records_sha256"],
            "record_count": target_records,
            "record_order_sha256": selection["record_order_sha256"],
        },
        "excluded_role_internal_duplicates": computed["exclusion_duplicates"],
        "postconditions": {
            "physical_overlap_with_each_excluded_role": [0] * len(exclusion_identities),
            "legacy_overlap_with_each_excluded_role": [0] * len(exclusion_identities),
            "physical_confirmation_duplicates": 0,
            "legacy_confirmation_duplicates": 0,
        },
        "sample_identity": {
            "effective": "(confirmation_receipt_sha256, effective_index)",
            "source": "(candidate_payload_sha256, candidate_local_index)",
        },
        "claims": {
            "fixture_mode": fixture_mode,
            "label_blind_selection": True,
            "confirmation_eligible": not fixture_mode and source.get("dirty") is False,
            "network_inference_performed": False,
            "architecture_selected": False,
            "strength_evidence": False,
            "production_network": False,
        },
    }


def create_confirmation_role(
    candidate_path: Path,
    exclusion_paths: Sequence[Path],
    output_directory: Path,
    *,
    contract_path: Path | None = None,
    _allow_fixture: bool = False,
    _target_records: int | None = None,
    _source_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    contract, contract_sha256 = load_contract(contract_path)
    selection_contract = _mapping(
        _mapping(contract.get("confirmation_role"), "confirmation-role contract").get("selection"),
        "confirmation selection contract",
    )
    frozen_target = int(selection_contract["selected_records"])
    target_records = _target_records if _target_records is not None else frozen_target
    _require(_allow_fixture or target_records == frozen_target, "production target drifted")
    source = dict(_source_override) if _source_override is not None else _repository_identity(REPOSITORY_ROOT)
    _require(
        _valid_commit(source.get("commit")) and type(source.get("dirty")) is bool,
        "confirmation selector source override is invalid",
    )
    if "path" not in source:
        source["path"] = TOOL_RELATIVE_PATH.as_posix()
        source["file_sha256"] = _sha256_file(REPOSITORY_ROOT / TOOL_RELATIVE_PATH)
    _require(_allow_fixture or source.get("dirty") is False, "confirmation selector source is dirty")
    output = output_directory.expanduser().resolve()
    _require(output.parent.is_dir(), f"output parent does not exist: {output.parent}")
    _require(not output.exists(), f"output already exists: {output}")

    with ExitStack() as stack:
        candidate = stack.enter_context(HordeBinV1Dataset(candidate_path))
        candidate_identity = _direct_identity(candidate)
        opened = [_open_exclusion(stack, path) for path in exclusion_paths]
        exclusions = [item[0] for item in opened]
        exclusion_identities = [item[1] for item in opened]
        _validate_sources(
            candidate,
            candidate_identity,
            exclusion_identities,
            contract,
            fixture_mode=_allow_fixture,
        )
        computed = _compute_selection(candidate, exclusions, target_records)
    receipt = _build_receipt(
        contract_sha256,
        source,
        candidate_identity,
        exclusion_identities,
        computed,
        target_records=target_records,
        fixture_mode=_allow_fixture,
    )
    output.mkdir()
    _write_exclusive(output / INDEX_FILENAME, computed["index_payload"])
    _write_exclusive(output / RECORDS_FILENAME, computed["records_payload"])
    _write_exclusive(output / RECEIPT_FILENAME, _canonical_json(receipt))
    return receipt


class ConfirmationRoleDataset:
    """Read-only view of an authenticated materialized C3 confirmation role."""

    def __init__(self, receipt_path: Path) -> None:
        self.receipt_path = receipt_path.expanduser().resolve()
        self.receipt, payload = _read_json(self.receipt_path, "confirmation-role receipt")
        _require(payload == _canonical_json(self.receipt), "confirmation receipt is not canonical")
        _require(self.receipt.get("schema") == SCHEMA, "confirmation receipt schema mismatch")
        self.receipt_sha256 = _sha256_bytes(payload)
        _require(
            self.receipt.get("contract")
            == {"schema": CONTRACT_SCHEMA, "sha256": CONTRACT_SHA256},
            "confirmation contract identity drifted",
        )
        selection = _mapping(self.receipt.get("selection"), "confirmation selection")
        output = _mapping(self.receipt.get("materialized_output"), "confirmation output")
        index_artifact = _mapping(selection.get("selected_indices"), "confirmation indices")
        _require(selection.get("algorithm") == ALGORITHM, "confirmation algorithm drifted")
        _require(
            selection.get("candidate_order") == "sha256_fixed_label_and_uint64be_local_index"
            and selection.get("selected_index_encoding") == "uint64_little_endian_in_selected_order",
            "confirmation ordering drifted",
        )
        target = selection.get("target_records")
        _require(type(target) is int and target > 0, "confirmation target is invalid")
        self.index_path = self.receipt_path.parent / INDEX_FILENAME
        self.path = self.receipt_path.parent / RECORDS_FILENAME
        _require(self.index_path.is_file(), "confirmation index artifact is missing")
        _require(self.path.is_file(), "confirmation record artifact is missing")
        index_payload = self.index_path.read_bytes()
        _require(
            index_artifact.get("name") == INDEX_FILENAME
            and index_artifact.get("bytes") == len(index_payload)
            and index_artifact.get("sha256") == _sha256_bytes(index_payload)
            and len(index_payload) == target * 8,
            "confirmation index artifact drifted",
        )
        self.source_indices = tuple(
            struct.unpack_from("<Q", index_payload, offset)[0]
            for offset in range(0, len(index_payload), 8)
        )
        _require(len(set(self.source_indices)) == target, "confirmation indices are not unique")
        _require(
            selection.get("accepted_records") == target
            and selection.get("selected_local_index_minimum") == min(self.source_indices)
            and selection.get("selected_local_index_maximum") == max(self.source_indices),
            "confirmation selected-index accounting drifted",
        )
        self._file = self.path.open("rb")
        self._mapping: mmap.mmap | None = None
        try:
            size = os.fstat(self._file.fileno()).st_size
            observed_sha = _sha256_file(self.path)
            _require(
                output.get("name") == RECORDS_FILENAME
                and output.get("bytes") == size
                and size == target * wire.RECORD_SIZE
                and output.get("sha256") == observed_sha
                and output.get("payload_sha256") == observed_sha
                and output.get("record_count") == target,
                "confirmation record artifact drifted",
            )
            self.file_sha256 = observed_sha
            self.payload_sha256 = observed_sha
            self.header_sha256 = self.receipt_sha256
            self.manifest_sha256 = self.receipt_sha256
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except BaseException:
            self.close()
            raise
        candidate = _mapping(self.receipt.get("candidate_source"), "confirmation candidate")
        manifest = _mapping(candidate.get("manifest"), "confirmation candidate manifest")
        self.manifest = {
            "schema": SCHEMA,
            "schema_sha256": CONTRACT_SHA256,
            "record_count": target,
            "payload_sha256": self.payload_sha256,
            "source_commit": manifest["source_commit"],
            "source_dirty": manifest["source_dirty"],
            "producer_sha256": manifest["producer_sha256"],
            "book_sha256": manifest["book_sha256"],
            "network": manifest["network"],
            "label_contract": manifest["label_contract"],
            "generation": manifest["generation"],
        }

    def __enter__(self) -> "ConfirmationRoleDataset":
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
        _require(0 <= index < len(self), f"confirmation record {index} is out of range")
        _require(self._mapping is not None, "confirmation dataset is closed")
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
            expansion_family=decoded.expansion_family,
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
        candidate = _mapping(self.receipt.get("candidate_source"), "confirmation candidate")
        selection = _mapping(self.receipt.get("selection"), "confirmation selection")
        return {
            "name": RECORDS_FILENAME,
            "sha256": self.file_sha256,
            "payload_sha256": self.payload_sha256,
            "records": len(self),
            "book_sha256": self.manifest["book_sha256"],
            "seed": self.manifest["generation"]["seed"],
            "confirmation_role": {
                "schema": SCHEMA,
                "receipt_name": self.receipt_path.name,
                "receipt_sha256": self.receipt_sha256,
                "contract_sha256": CONTRACT_SHA256,
                "candidate_file_sha256": candidate["file_sha256"],
                "candidate_payload_sha256": candidate["payload_sha256"],
                "selected_index_sha256": selection["selected_indices"]["sha256"],
                "decision_chain_sha256": selection["decision_chain_sha256"],
                "record_order_sha256": self.receipt["materialized_output"][
                    "record_order_sha256"
                ],
            },
        }


def verify_confirmation_role(
    candidate_path: Path,
    exclusion_paths: Sequence[Path],
    receipt_path: Path,
    *,
    contract_path: Path | None = None,
    _allow_fixture: bool = False,
    _source_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    contract, contract_sha256 = load_contract(contract_path)
    with ConfirmationRoleDataset(receipt_path) as selected:
        receipt = selected.receipt
        source = dict(_source_override) if _source_override is not None else _repository_identity(REPOSITORY_ROOT)
        if "path" not in source:
            source["path"] = TOOL_RELATIVE_PATH.as_posix()
            source["file_sha256"] = _sha256_file(REPOSITORY_ROOT / TOOL_RELATIVE_PATH)
        _require(_allow_fixture or source.get("dirty") is False, "confirmation verifier source is dirty")
        with ExitStack() as stack:
            candidate = stack.enter_context(HordeBinV1Dataset(candidate_path))
            candidate_identity = _direct_identity(candidate)
            opened = [_open_exclusion(stack, path) for path in exclusion_paths]
            exclusions = [item[0] for item in opened]
            exclusion_identities = [item[1] for item in opened]
            _validate_sources(
                candidate,
                candidate_identity,
                exclusion_identities,
                contract,
                fixture_mode=_allow_fixture,
            )
            target = int(_mapping(receipt.get("selection"), "confirmation selection")["target_records"])
            computed = _compute_selection(candidate, exclusions, target)
        expected = _build_receipt(
            contract_sha256,
            source,
            candidate_identity,
            exclusion_identities,
            computed,
            target_records=target,
            fixture_mode=_allow_fixture,
        )
        _require(receipt == expected, "confirmation receipt is not the canonical frozen selection")
        _require(
            selected.index_path.read_bytes() == computed["index_payload"],
            "confirmation index sequence differs from canonical selection",
        )
        _require(
            selected.path.read_bytes() == computed["records_payload"],
            "confirmation records differ from canonical selection",
        )
        identity = selected.identity()
    return {
        "schema": "HORDE_V2_C3_CONFIRMATION_ROLE_VERIFICATION_V1",
        "contract_sha256": contract_sha256,
        "confirmation_role": identity,
        "claims": {
            "fixture_mode": _allow_fixture,
            "canonical_selection_recomputed": True,
            "materialized_records_reconstructed": True,
            "zero_overlap_with_both_excluded_roles": True,
            "zero_confirmation_duplicates": True,
            "label_blind_selection": True,
            "network_inference_performed": False,
            "architecture_selected": False,
            "strength_evidence": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("create", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("candidate", type=Path)
        command.add_argument("training", type=Path)
        command.add_argument("tuning_validation_receipt", type=Path)
        command.add_argument("--contract", type=Path)
        if name == "create":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("receipt", type=Path)
            command.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    exclusions = (args.training, args.tuning_validation_receipt)
    if args.command == "create":
        result = create_confirmation_role(
            args.candidate,
            exclusions,
            args.output,
            contract_path=args.contract,
        )
    else:
        result = verify_confirmation_role(
            args.candidate,
            exclusions,
            args.receipt,
            contract_path=args.contract,
        )
        if args.output is not None:
            _write_exclusive(args.output.expanduser().resolve(), _canonical_json(result))
    print(_canonical_json(result).decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConfirmationRoleError, OSError, subprocess.SubprocessError, wire.FormatError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
