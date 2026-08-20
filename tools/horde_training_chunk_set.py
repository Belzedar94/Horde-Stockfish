#!/usr/bin/env python3
"""Assemble and read authenticated HORDE_BIN_V1 chunk sets."""

from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Iterator, Mapping, Sequence

try:
    from . import horde_bin_v1 as wire
    from .horde_training_decoder import (
        HordeBinV1Dataset,
        SparseBatch,
        TrainingRecord,
        decode_training_record,
        make_sparse_batch,
    )
except ImportError:
    import horde_bin_v1 as wire
    from horde_training_decoder import (
        HordeBinV1Dataset,
        SparseBatch,
        TrainingRecord,
        decode_training_record,
        make_sparse_batch,
    )


SCHEMA = "HORDE_TRAINING_CHUNK_SET_V1"
SCHEMA_SHA256 = "CAAF9D19B4A04BA8854FDBC24B4A7D1948577B17FB652E39EF4DB4287BB0DD4E"
SCALE_SCHEMA = "HORDE_V2_RANK8_SCALE_V1"

ROLE_BOOK = {
    "training": "training",
    "validation_candidate": "validation",
}
GENERATION_COMMON_FIELDS = (
    "hash_mb",
    "depth",
    "nodes",
    "random_move_min_ply",
    "random_move_max_ply",
    "random_move_count",
    "random_multi_pv",
    "random_multi_pv_diff",
    "write_min_ply",
    "write_max_ply",
    "max_game_ply",
    "opening_count",
)
FORMAT_FIELDS = (
    "schema",
    "schema_sha256",
    "format_version",
    "header_bytes",
    "record_bytes",
    "byte_order",
)
CHUNK_IDENTITY_FIELDS = (
    "index",
    "file_bytes",
    "file_sha256",
    "header_sha256",
    "manifest_sha256",
    "payload_sha256",
    "records",
    "seed",
    "threads",
    "global_begin",
    "global_end",
)
CAMPAIGN_FIELDS = {
    "contract_name",
    "contract_sha256",
    "contract_schema",
    "campaign_id",
    "cohort",
}
ORDERING_FIELDS = {
    "key",
    "base_seed",
    "chunk_count",
    "expected_indices",
    "expected_seed_range",
}
TOTAL_FIELDS = {"chunks", "records", "payload_bytes", "file_bytes", "threads_seen"}
IDENTITY_FIELDS = {
    "logical_payload_sha256",
    "chunk_set_sha256",
    "sample_identity",
}
GATE_FIELDS = {
    "all_chunks_authenticated",
    "paths_confined_to_receipt_directory",
    "indices_complete_and_unique",
    "seeds_match_base_plus_index",
    "record_counts_match_campaign",
    "common_manifest_identity_matches",
    "file_identities_unique",
    "payload_identities_unique",
    "logical_payload_identity_computed",
    "passed",
}


class ChunkSetError(ValueError):
    """Raised when a chunk set violates its fail-closed contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ChunkSetError(message)


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in document, f"duplicate JSON key {key}")
        document[key] = value
    return document


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = path.expanduser().resolve()
    payload = resolved.read_bytes()
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ChunkSetError(f"{label} is not strict UTF-8 JSON: {error}") from error
    _require(isinstance(document, dict), f"{label} root is not an object")
    return document, payload


def _mapping(value: object, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> None:
    resolved = path.expanduser().resolve()
    _require(resolved.parent.is_dir(), f"output parent does not exist: {resolved.parent}")
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        resolved.unlink(missing_ok=True)
        raise


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789ABCDEF" for character in value)
    )


def _relative_chunk_path(root: Path, path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ChunkSetError(f"chunk path escapes the receipt directory: {resolved}") from error
    encoded = relative.as_posix()
    _require(encoded not in ("", "."), "chunk path does not name a file")
    return encoded


def _resolve_chunk_path(root: Path, encoded: object) -> Path:
    _require(isinstance(encoded, str) and encoded, "chunk path is invalid")
    _require("\\" not in encoded, "chunk path is not canonical POSIX form")
    pure = PurePosixPath(encoded)
    _require(not pure.is_absolute(), "chunk path is absolute")
    _require(all(part not in ("", ".", "..") for part in pure.parts), "chunk path escapes root")
    resolved = (root / Path(*pure.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ChunkSetError(f"chunk path escapes the receipt directory: {encoded}") from error
    return resolved


@dataclass(frozen=True, slots=True)
class CampaignExpectation:
    contract_name: str
    contract_sha256: str
    campaign_id: str
    cohort: str
    role: str
    records: int
    positions_per_chunk: int
    chunk_count: int
    base_seed: int
    source_commit: str
    producer_sha256: str
    network: dict[str, str]
    label_contract: dict[str, str]
    book_sha256: str
    generation_common: dict[str, int]


def load_campaign_expectation(contract_path: Path, role: str) -> CampaignExpectation:
    _require(role in ROLE_BOOK, f"unsupported chunk-set role {role}")
    contract_resolved = contract_path.expanduser().resolve()
    contract, payload = _load_json(contract_resolved, "campaign contract")
    _require(contract.get("schema_name") == SCALE_SCHEMA, "campaign contract schema mismatch")

    openbench = _mapping(contract.get("openbench"), "campaign OpenBench section")
    generation = _mapping(contract.get("generation"), "campaign generation section")
    role_generation = _mapping(generation.get(role), f"campaign {role} generation")
    common_generation = _mapping(generation.get("common"), "campaign common generation")
    books = _mapping(contract.get("books"), "campaign books section")
    book = _mapping(books.get(ROLE_BOOK[role]), f"campaign {role} book")
    dependencies = _mapping(contract.get("dependencies"), "campaign dependencies")
    teacher = _mapping(dependencies.get("teacher"), "campaign teacher")
    labels = _mapping(dependencies.get("labels"), "campaign labels")

    records = role_generation.get("records")
    positions_per_chunk = role_generation.get("positions_per_chunk")
    chunk_count = role_generation.get("chunk_count")
    base_seed = role_generation.get("base_seed")
    _require(
        all(type(value) is int and value > 0 for value in (
            records,
            positions_per_chunk,
            chunk_count,
            base_seed,
        )),
        f"campaign {role} dimensions are invalid",
    )
    assert isinstance(records, int)
    assert isinstance(positions_per_chunk, int)
    assert isinstance(chunk_count, int)
    assert isinstance(base_seed, int)
    _require(records == positions_per_chunk * chunk_count, f"campaign {role} totals drifted")

    expected_common_keys = set(GENERATION_COMMON_FIELDS) - {"opening_count"}
    _require(
        set(common_generation) == expected_common_keys,
        "campaign common generation fields are incomplete or unexpected",
    )
    _require(
        all(type(common_generation[key]) is int for key in expected_common_keys),
        "campaign common generation values are not integers",
    )
    opening_count = book.get("records")
    _require(type(opening_count) is int and opening_count > 0, "campaign book count is invalid")
    generation_common = {key: int(common_generation[key]) for key in common_generation}
    generation_common["opening_count"] = opening_count

    source_commit = teacher.get("source_commit")
    producer_sha256 = teacher.get("producer_sha256")
    network_schema = teacher.get("network_schema")
    network_sha256 = teacher.get("network_sha256")
    book_sha256 = book.get("raw_sha256")
    label_schema = labels.get("schema")
    label_sha256 = labels.get("schema_sha256")
    _require(
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(character in "0123456789abcdefABCDEF" for character in source_commit),
        "campaign teacher source commit is invalid",
    )
    for value, label in (
        (producer_sha256, "producer"),
        (network_sha256, "network"),
        (book_sha256, "book"),
        (label_sha256, "label contract"),
    ):
        _require(_valid_sha256(value), f"campaign {label} SHA-256 is invalid")
    _require(isinstance(network_schema, str) and network_schema, "network schema is invalid")
    _require(isinstance(label_schema, str) and label_schema, "label schema is invalid")

    campaign_id = openbench.get("campaign_id")
    cohort = openbench.get("cohort")
    _require(isinstance(campaign_id, str) and campaign_id, "campaign id is invalid")
    _require(isinstance(cohort, str) and cohort, "campaign cohort is invalid")
    return CampaignExpectation(
        contract_name=contract_resolved.name,
        contract_sha256=_sha256_bytes(payload),
        campaign_id=campaign_id,
        cohort=cohort,
        role=role,
        records=records,
        positions_per_chunk=positions_per_chunk,
        chunk_count=chunk_count,
        base_seed=base_seed,
        source_commit=source_commit.lower(),
        producer_sha256=str(producer_sha256),
        network={"schema": str(network_schema), "sha256": str(network_sha256)},
        label_contract={"schema": str(label_schema), "schema_sha256": str(label_sha256)},
        book_sha256=str(book_sha256),
        generation_common=generation_common,
    )


def _format_identity(manifest: Mapping[str, Any]) -> dict[str, object]:
    return {field: manifest[field] for field in FORMAT_FIELDS}


def _common_manifest_identity(manifest: Mapping[str, Any]) -> dict[str, object]:
    generation = _mapping(manifest.get("generation"), "chunk generation")
    return {
        "source_commit": str(manifest["source_commit"]).lower(),
        "source_dirty": manifest["source_dirty"],
        "network": manifest["network"],
        "book_sha256": manifest["book_sha256"],
        "producer_sha256": manifest["producer_sha256"],
        "label_contract": manifest["label_contract"],
        "generation": {field: generation[field] for field in GENERATION_COMMON_FIELDS},
    }


def _expected_format() -> dict[str, object]:
    return {
        "schema": wire.SCHEMA_NAME,
        "schema_sha256": wire.SCHEMA_SHA256,
        "format_version": wire.FORMAT_VERSION,
        "header_bytes": wire.HEADER_SIZE,
        "record_bytes": wire.RECORD_SIZE,
        "byte_order": "little",
    }


def _expected_common(expectation: CampaignExpectation) -> dict[str, object]:
    return {
        "source_commit": expectation.source_commit,
        "source_dirty": False,
        "network": expectation.network,
        "book_sha256": expectation.book_sha256,
        "producer_sha256": expectation.producer_sha256,
        "label_contract": expectation.label_contract,
        "generation": expectation.generation_common,
    }


def _validate_chunk_manifest(
    manifest: Mapping[str, Any],
    expectation: CampaignExpectation,
) -> int:
    _require(_format_identity(manifest) == _expected_format(), "chunk format identity drifted")
    _require(
        _common_manifest_identity(manifest) == _expected_common(expectation),
        "chunk common manifest identity drifted",
    )
    _require(
        manifest.get("record_count") == expectation.positions_per_chunk,
        "chunk record count differs from campaign",
    )
    generation = _mapping(manifest.get("generation"), "chunk generation")
    _require(
        generation.get("requested_records") == expectation.positions_per_chunk,
        "chunk requested record count differs from campaign",
    )
    seed_text = generation.get("seed")
    _require(isinstance(seed_text, str) and seed_text.isdecimal(), "chunk seed is invalid")
    seed = int(seed_text)
    index = seed - expectation.base_seed
    _require(0 <= index < expectation.chunk_count, f"chunk seed {seed} is outside campaign range")
    return index


def _campaign_section(expectation: CampaignExpectation) -> dict[str, object]:
    return {
        "contract_name": expectation.contract_name,
        "contract_sha256": expectation.contract_sha256,
        "contract_schema": SCALE_SCHEMA,
        "campaign_id": expectation.campaign_id,
        "cohort": expectation.cohort,
    }


def _ordering_section(expectation: CampaignExpectation) -> dict[str, object]:
    return {
        "key": "generation.seed",
        "base_seed": expectation.base_seed,
        "chunk_count": expectation.chunk_count,
        "expected_indices": [0, expectation.chunk_count - 1],
        "expected_seed_range": [
            str(expectation.base_seed),
            str(expectation.base_seed + expectation.chunk_count - 1),
        ],
    }


def _identity_chunk(entry: Mapping[str, Any]) -> dict[str, object]:
    return {field: entry[field] for field in CHUNK_IDENTITY_FIELDS}


def _chunk_set_identity_document(receipt: Mapping[str, Any]) -> dict[str, object]:
    campaign = _mapping(receipt.get("campaign"), "receipt campaign")
    campaign_identity = {
        key: campaign[key]
        for key in ("contract_sha256", "contract_schema", "campaign_id", "cohort")
    }
    chunks = receipt.get("chunks")
    _require(isinstance(chunks, list), "receipt chunks are not a list")
    return {
        "schema": receipt["schema"],
        "schema_sha256": receipt["schema_sha256"],
        "campaign": campaign_identity,
        "role": receipt["role"],
        "format": receipt["format"],
        "common_manifest": receipt["common_manifest"],
        "ordering": receipt["ordering"],
        "chunks": [_identity_chunk(_mapping(chunk, "receipt chunk")) for chunk in chunks],
    }


def _logical_payload_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as source:
            source.seek(wire.HEADER_SIZE)
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest().upper()


def assemble_chunk_set(
    contract_path: Path,
    role: str,
    output_path: Path,
    chunk_paths: Sequence[Path],
) -> dict[str, object]:
    expectation = load_campaign_expectation(contract_path, role)
    output = output_path.expanduser().resolve()
    _require(output.parent.is_dir(), f"output parent does not exist: {output.parent}")
    _require(not output.exists(), f"output already exists: {output}")
    _require(len(chunk_paths) == expectation.chunk_count, "input chunk count differs from campaign")

    resolved_paths = [path.expanduser().resolve() for path in chunk_paths]
    _require(len(set(resolved_paths)) == len(resolved_paths), "input chunk path is duplicated")
    entries_by_index: dict[int, dict[str, object]] = {}
    paths_by_index: dict[int, Path] = {}
    file_hashes: set[str] = set()
    payload_hashes: set[str] = set()

    for path in resolved_paths:
        _require(path.is_file(), f"chunk does not exist: {path}")
        relative_path = _relative_chunk_path(output.parent, path)
        with HordeBinV1Dataset(path) as dataset:
            index = _validate_chunk_manifest(dataset.manifest, expectation)
            _require(index not in entries_by_index, f"duplicate chunk index {index}")
            _require(dataset.file_sha256 not in file_hashes, "duplicate chunk file identity")
            payload_sha256 = dataset.manifest["payload_sha256"]
            _require(payload_sha256 not in payload_hashes, "duplicate chunk payload identity")
            generation = _mapping(dataset.manifest["generation"], "chunk generation")
            entry = {
                "index": index,
                "path": relative_path,
                "file_bytes": path.stat().st_size,
                "file_sha256": dataset.file_sha256,
                "header_sha256": dataset.header_sha256,
                "manifest_sha256": dataset.manifest_sha256,
                "payload_sha256": payload_sha256,
                "records": len(dataset),
                "seed": generation["seed"],
                "threads": generation["threads"],
                "global_begin": 0,
                "global_end": 0,
            }
        entries_by_index[index] = entry
        paths_by_index[index] = path
        file_hashes.add(str(entry["file_sha256"]))
        payload_hashes.add(str(entry["payload_sha256"]))

    expected_indices = list(range(expectation.chunk_count))
    _require(sorted(entries_by_index) == expected_indices, "chunk indices contain a gap")
    entries: list[dict[str, object]] = []
    ordered_paths: list[Path] = []
    cursor = 0
    for index in expected_indices:
        entry = entries_by_index[index]
        entry["global_begin"] = cursor
        cursor += int(entry["records"])
        entry["global_end"] = cursor
        entries.append(entry)
        ordered_paths.append(paths_by_index[index])
    _require(cursor == expectation.records, "assembled record total differs from campaign")

    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "schema_sha256": SCHEMA_SHA256,
        "campaign": _campaign_section(expectation),
        "role": role,
        "format": _expected_format(),
        "common_manifest": _expected_common(expectation),
        "ordering": _ordering_section(expectation),
        "chunks": entries,
        "totals": {
            "chunks": len(entries),
            "records": cursor,
            "payload_bytes": cursor * wire.RECORD_SIZE,
            "file_bytes": sum(int(entry["file_bytes"]) for entry in entries),
            "threads_seen": sorted({int(entry["threads"]) for entry in entries}),
        },
        "identity": {
            "logical_payload_sha256": _logical_payload_sha256(ordered_paths),
            "chunk_set_sha256": "",
            "sample_identity": "(chunk_payload_sha256, chunk_local_record_index)",
        },
        "gates": {
            "all_chunks_authenticated": True,
            "paths_confined_to_receipt_directory": True,
            "indices_complete_and_unique": True,
            "seeds_match_base_plus_index": True,
            "record_counts_match_campaign": True,
            "common_manifest_identity_matches": True,
            "file_identities_unique": True,
            "payload_identities_unique": True,
            "logical_payload_identity_computed": True,
            "passed": True,
        },
    }
    identity = _sha256_bytes(_canonical_bytes(_chunk_set_identity_document(receipt)))
    _mapping(receipt["identity"], "receipt identity")["chunk_set_sha256"] = identity
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_exclusive(output, payload)
    return receipt


class HordeChunkSetDataset:
    """Read-only random-access view across authenticated HORDE_BIN_V1 chunks."""

    def __init__(self, receipt_path: Path, contract_path: Path | None = None) -> None:
        self.path = receipt_path.expanduser().resolve()
        self.receipt, payload = _load_json(self.path, "chunk-set receipt")
        self.receipt_sha256 = _sha256_bytes(payload)
        self.file_sha256 = self.receipt_sha256
        self.manifest_sha256 = self.receipt_sha256
        self.header_sha256 = self.receipt_sha256
        self._datasets: list[HordeBinV1Dataset] = []
        self._ends: list[int] = []
        self._begins: list[int] = []
        try:
            self._open_and_verify(contract_path)
        except BaseException:
            self.close()
            raise

    def _open_and_verify(self, contract_path: Path | None) -> None:
        expected_top = {
            "schema",
            "schema_sha256",
            "campaign",
            "role",
            "format",
            "common_manifest",
            "ordering",
            "chunks",
            "totals",
            "identity",
            "gates",
        }
        _require(set(self.receipt) == expected_top, "chunk-set receipt fields are incomplete")
        _require(self.receipt["schema"] == SCHEMA, "chunk-set receipt schema mismatch")
        _require(self.receipt["schema_sha256"] == SCHEMA_SHA256, "chunk-set schema SHA-256 mismatch")
        role = self.receipt["role"]
        _require(role in ROLE_BOOK, "chunk-set role is invalid")
        campaign = _mapping(self.receipt["campaign"], "receipt campaign")
        ordering = _mapping(self.receipt["ordering"], "receipt ordering")
        totals = _mapping(self.receipt["totals"], "receipt totals")
        identity = _mapping(self.receipt["identity"], "receipt identity")
        gates = _mapping(self.receipt["gates"], "receipt gates")
        common_manifest = _mapping(self.receipt["common_manifest"], "receipt common manifest")
        chunks = self.receipt["chunks"]
        _require(isinstance(chunks, list) and chunks, "chunk-set receipt has no chunks")
        _require(set(campaign) == CAMPAIGN_FIELDS, "receipt campaign fields are incomplete")
        _require(set(ordering) == ORDERING_FIELDS, "receipt ordering fields are incomplete")
        _require(set(totals) == TOTAL_FIELDS, "receipt total fields are incomplete")
        _require(set(identity) == IDENTITY_FIELDS, "receipt identity fields are incomplete")
        _require(set(gates) == GATE_FIELDS, "receipt gate fields are incomplete")
        _require(all(value is True for value in gates.values()), "chunk-set receipt contains a failed gate")
        _require(_valid_sha256(identity.get("logical_payload_sha256")), "logical payload SHA-256 is invalid")
        _require(_valid_sha256(identity.get("chunk_set_sha256")), "chunk-set SHA-256 is invalid")
        _require(
            identity.get("sample_identity")
            == "(chunk_payload_sha256, chunk_local_record_index)",
            "receipt sample identity drifted",
        )
        _require(isinstance(campaign.get("contract_name"), str) and campaign["contract_name"],
                 "receipt contract name is invalid")
        _require(isinstance(campaign.get("campaign_id"), str) and campaign["campaign_id"],
                 "receipt campaign id is invalid")
        _require(isinstance(campaign.get("cohort"), str) and campaign["cohort"],
                 "receipt cohort is invalid")
        _require(ordering.get("key") == "generation.seed", "receipt ordering key drifted")
        _require(type(ordering.get("base_seed")) is int and ordering["base_seed"] > 0,
                 "receipt base seed is invalid")
        _require(type(ordering.get("chunk_count")) is int and ordering["chunk_count"] > 0,
                 "receipt chunk count is invalid")
        _require(
            set(common_manifest)
            == {
                "source_commit",
                "source_dirty",
                "network",
                "book_sha256",
                "producer_sha256",
                "label_contract",
                "generation",
            },
            "receipt common manifest fields are incomplete",
        )
        common_generation = _mapping(common_manifest.get("generation"), "common generation")
        _require(
            set(common_generation) == set(GENERATION_COMMON_FIELDS),
            "receipt common generation fields are incomplete",
        )
        _require(common_manifest.get("source_dirty") is False, "receipt source is dirty")
        _require(
            isinstance(common_manifest.get("source_commit"), str)
            and len(common_manifest["source_commit"]) == 40
            and all(
                character in "0123456789abcdef"
                for character in common_manifest["source_commit"]
            ),
            "receipt source commit is invalid",
        )
        for value, label in (
            (common_manifest.get("book_sha256"), "book"),
            (common_manifest.get("producer_sha256"), "producer"),
        ):
            _require(_valid_sha256(value), f"receipt {label} SHA-256 is invalid")
        _require(
            common_manifest.get("network")
            == {"schema": "HORDETEST_HP_LEGACY_V1", "sha256": wire.RUN6B_SHA256},
            "receipt network identity drifted",
        )
        _require(
            common_manifest.get("label_contract")
            == {
                "schema": wire.LABEL_CONTRACT_NAME,
                "schema_sha256": wire.LABEL_CONTRACT_SHA256,
            },
            "receipt label contract drifted",
        )

        expectation: CampaignExpectation | None = None
        if contract_path is not None:
            expectation = load_campaign_expectation(contract_path, str(role))
            _require(campaign == _campaign_section(expectation), "receipt campaign identity drifted")
            _require(ordering == _ordering_section(expectation), "receipt ordering contract drifted")
            _require(self.receipt["format"] == _expected_format(), "receipt format drifted")
            _require(
                self.receipt["common_manifest"] == _expected_common(expectation),
                "receipt common manifest drifted",
            )
        else:
            _require(campaign.get("contract_schema") == SCALE_SCHEMA, "receipt campaign schema drifted")
            _require(_valid_sha256(campaign.get("contract_sha256")), "receipt contract SHA-256 is invalid")
            _require(self.receipt["format"] == _expected_format(), "receipt format drifted")

        root = self.path.parent.resolve()
        expected_index = 0
        cursor = 0
        paths: list[Path] = []
        observed_file_hashes: set[str] = set()
        observed_payload_hashes: set[str] = set()
        observed_threads: set[int] = set()
        for raw_entry in chunks:
            entry = _mapping(raw_entry, "receipt chunk")
            _require(
                set(entry) == set(CHUNK_IDENTITY_FIELDS) | {"path"},
                "receipt chunk fields are incomplete",
            )
            for field in ("index", "file_bytes", "records", "threads", "global_begin", "global_end"):
                _require(type(entry[field]) is int, f"receipt chunk field {field} is not an integer")
            for field in ("file_sha256", "header_sha256", "manifest_sha256", "payload_sha256"):
                _require(_valid_sha256(entry[field]), f"receipt chunk field {field} is invalid")
            _require(
                isinstance(entry["seed"], str) and entry["seed"].isdecimal(),
                "receipt chunk seed is invalid",
            )
            _require(entry["index"] == expected_index, "receipt chunks are not index ordered")
            _require(entry["global_begin"] == cursor, "receipt chunk global range has a gap")
            path = _resolve_chunk_path(root, entry["path"])
            _require(path.is_file(), f"receipt chunk is missing: {path}")
            dataset = HordeBinV1Dataset(path)
            self._datasets.append(dataset)
            manifest = dataset.manifest
            if expectation is not None:
                _require(
                    _validate_chunk_manifest(manifest, expectation) == expected_index,
                    "receipt chunk seed maps to another index",
                )
            else:
                _require(
                    _format_identity(manifest) == self.receipt["format"],
                    "receipt chunk format identity drifted",
                )
                _require(
                    _common_manifest_identity(manifest) == self.receipt["common_manifest"],
                    "receipt chunk common manifest identity drifted",
                )
                base_seed = ordering.get("base_seed")
                _require(type(base_seed) is int and base_seed > 0, "receipt base seed is invalid")
                _require(
                    int(manifest["generation"]["seed"]) - base_seed == expected_index,
                    "receipt chunk seed maps to another index",
                )

            expected_entry = {
                "index": expected_index,
                "path": entry["path"],
                "file_bytes": path.stat().st_size,
                "file_sha256": dataset.file_sha256,
                "header_sha256": dataset.header_sha256,
                "manifest_sha256": dataset.manifest_sha256,
                "payload_sha256": manifest["payload_sha256"],
                "records": len(dataset),
                "seed": manifest["generation"]["seed"],
                "threads": manifest["generation"]["threads"],
                "global_begin": cursor,
                "global_end": cursor + len(dataset),
            }
            _require(entry == expected_entry, f"receipt chunk {expected_index} identity drifted")
            _require(dataset.file_sha256 not in observed_file_hashes, "duplicate chunk file identity")
            _require(
                manifest["payload_sha256"] not in observed_payload_hashes,
                "duplicate chunk payload identity",
            )
            observed_file_hashes.add(dataset.file_sha256)
            observed_payload_hashes.add(manifest["payload_sha256"])
            observed_threads.add(int(manifest["generation"]["threads"]))
            self._begins.append(cursor)
            cursor += len(dataset)
            self._ends.append(cursor)
            paths.append(path)
            expected_index += 1

        _require(ordering.get("chunk_count") == len(chunks), "receipt chunk count drifted")
        _require(ordering.get("expected_indices") == [0, len(chunks) - 1], "receipt index range drifted")
        expected_seed_range = [
            str(ordering["base_seed"]),
            str(ordering["base_seed"] + len(chunks) - 1),
        ]
        _require(ordering.get("expected_seed_range") == expected_seed_range, "receipt seed range drifted")
        _require(totals.get("chunks") == len(chunks), "receipt total chunk count drifted")
        _require(totals.get("records") == cursor, "receipt total record count drifted")
        _require(totals.get("payload_bytes") == cursor * wire.RECORD_SIZE, "payload byte total drifted")
        _require(
            totals.get("file_bytes") == sum(path.stat().st_size for path in paths),
            "file byte total drifted",
        )
        _require(totals.get("threads_seen") == sorted(observed_threads), "thread inventory drifted")
        if expectation is not None:
            _require(cursor == expectation.records, "receipt record total differs from campaign")

        observed_logical = _logical_payload_sha256(paths)
        _require(
            observed_logical == identity["logical_payload_sha256"],
            "logical payload SHA-256 drifted",
        )
        observed_chunk_set = _sha256_bytes(_canonical_bytes(_chunk_set_identity_document(self.receipt)))
        _require(observed_chunk_set == identity["chunk_set_sha256"], "chunk-set SHA-256 drifted")
        self.logical_payload_sha256 = observed_logical
        self.chunk_set_sha256 = observed_chunk_set
        self.manifest = {
            "schema": SCHEMA,
            "schema_sha256": SCHEMA_SHA256,
            "record_count": cursor,
            "logical_payload_sha256": observed_logical,
            "chunk_set_sha256": observed_chunk_set,
            **_mapping(self.receipt["common_manifest"], "receipt common manifest"),
            "chunk_generation": {
                "base_seed": ordering["base_seed"],
                "chunk_count": len(chunks),
                "threads_seen": sorted(observed_threads),
            },
        }

    def __enter__(self) -> "HordeChunkSetDataset":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __len__(self) -> int:
        return int(self.manifest["record_count"])

    def close(self) -> None:
        for dataset in reversed(self._datasets):
            dataset.close()
        self._datasets.clear()

    def _location(self, index: int) -> tuple[int, int]:
        _require(0 <= index < len(self), f"record index {index} is out of range")
        chunk_index = bisect_right(self._ends, index)
        return chunk_index, index - self._begins[chunk_index]

    def sample_identity(self, index: int) -> tuple[str, int]:
        chunk_index, local_index = self._location(index)
        return self._datasets[chunk_index].manifest["payload_sha256"], local_index

    def raw_record(self, index: int) -> bytes:
        chunk_index, local_index = self._location(index)
        return self._datasets[chunk_index].raw_record(local_index)

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
            source_payload_sha256=self.logical_payload_sha256,
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
        return {
            "name": self.path.name,
            "receipt_sha256": self.receipt_sha256,
            "chunk_set_sha256": self.chunk_set_sha256,
            "logical_payload_sha256": self.logical_payload_sha256,
            "records": len(self),
            "chunks": len(self._datasets),
            "role": self.receipt["role"],
        }


def verify_chunk_set(
    receipt_path: Path,
    contract_path: Path | None = None,
) -> dict[str, object]:
    with HordeChunkSetDataset(receipt_path, contract_path) as dataset:
        return dataset.identity()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    assemble = subparsers.add_parser("assemble", help="authenticate chunks and write a receipt")
    assemble.add_argument("--contract", type=Path, required=True)
    assemble.add_argument("--role", choices=tuple(ROLE_BOOK), required=True)
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--chunks-dir", type=Path)
    assemble.add_argument("chunks", type=Path, nargs="*")

    verify = subparsers.add_parser("verify", help="re-authenticate a chunk-set receipt")
    verify.add_argument("receipt", type=Path)
    verify.add_argument("--contract", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "assemble":
        _require(
            bool(args.chunks) != bool(args.chunks_dir),
            "provide either positional chunks or --chunks-dir",
        )
        chunks = args.chunks
        if args.chunks_dir:
            chunks_dir = args.chunks_dir.expanduser().resolve()
            _require(chunks_dir.is_dir(), f"chunk directory does not exist: {chunks_dir}")
            chunks = sorted(chunks_dir.glob("*.bin"))
            _require(chunks, f"chunk directory contains no .bin files: {chunks_dir}")
        receipt = assemble_chunk_set(args.contract, args.role, args.output, chunks)
        summary = {
            "receipt": str(args.output.expanduser().resolve()),
            "chunk_set_sha256": receipt["identity"]["chunk_set_sha256"],
            "logical_payload_sha256": receipt["identity"]["logical_payload_sha256"],
            "chunks": receipt["totals"]["chunks"],
            "records": receipt["totals"]["records"],
        }
    else:
        summary = verify_chunk_set(args.receipt, args.contract)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ChunkSetError, wire.FormatError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
