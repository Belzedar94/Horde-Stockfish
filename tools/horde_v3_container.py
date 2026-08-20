#!/usr/bin/env python3
"""Canonical codec for trained Horde NNUE V3 integer networks.

V3 keeps the HORDE_V2_INTEGER_NETWORK_V1 envelope unchanged: the same
``HSV2INT\\0`` magic, the same 2,048 byte little-endian header, the same
authenticated section directory, and the same fail-closed reader discipline.
It registers one new network schema, ``0x00020001``.

Two contract changes are deliberate and are recorded in
``docs/horde/nnue-v3-integer-bounds.md``.

1. The output layer is quantized at the scale the trainer actually optimizes.
   ``NNUE_TO_SCORE`` is 600 and the legacy exporter reproduces that exactly, so
   V3 uses ``OUTPUT_WEIGHT_SCALE = 9600 / 127`` and ``OUTPUT_BIAS_SCALE = 9600``
   and evaluates at ``600 * v``. The V2 container exporter quantized the output
   layer at the dense scale instead and evaluates at ``508 * v``; that defect is
   left untouched in the V2 path and is corrected separately.
2. V3 restores the PSQT skip, which V2 does not have, so the contract gains one
   new bound: ``MAX_PSQT_MAGNITUDE = 2**20``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Mapping

try:
    from .horde_v2_container import (
        DIRECTORY_ENTRY_BYTES,
        DIRECTORY_OFFSET,
        DTYPE_BYTES,
        DTYPE_CODES,
        ENDIAN_TAG,
        FORMAT_VERSION,
        HEADER_BYTES,
        MAGIC,
        MAXIMUM_CONTAINER_BYTES,
        canonical_json,
        sha256_bytes,
    )
    from .horde_training_decoder import (
        V3_BLOCK_BASES,
        V3_BLOCK_ORDER,
        V3_BLOCK_WIDTHS,
        V3_GLOBAL_DIMENSIONS,
        V3_MAX_ACTIVE_ROWS,
    )
except ImportError:
    from horde_v2_container import (
        DIRECTORY_ENTRY_BYTES,
        DIRECTORY_OFFSET,
        DTYPE_BYTES,
        DTYPE_CODES,
        ENDIAN_TAG,
        FORMAT_VERSION,
        HEADER_BYTES,
        MAGIC,
        MAXIMUM_CONTAINER_BYTES,
        canonical_json,
        sha256_bytes,
    )
    from horde_training_decoder import (
        V3_BLOCK_BASES,
        V3_BLOCK_ORDER,
        V3_BLOCK_WIDTHS,
        V3_GLOBAL_DIMENSIONS,
        V3_MAX_ACTIVE_ROWS,
    )


CONTAINER_SCHEMA = "HORDE_V2_INTEGER_NETWORK_V1"
NETWORK_SCHEMA_NAME = "V3_G1024_PAWN_WPC8"
NETWORK_SCHEMA_ID = 0x00020001
SECTION_COUNT = 10

FT_ROWS = V3_GLOBAL_DIMENSIONS
FT_LANES = 1024
PHASE_BUCKETS = 8
HIDDEN0_LANES = 16
HIDDEN1_LANES = 32
OUTPUT_HEADS = 2
PSQT_COLUMNS = PHASE_BUCKETS * OUTPUT_HEADS
PHASE_LOOKUP_ENTRIES = 37

FT_SCALE = 127 * 64
DENSE_SCALE = 64
HIDDEN_BIAS_SCALE = 127 * 64
OUTPUT_BIAS_SCALE = 600 * 16
OUTPUT_WEIGHT_SCALE_NUMERATOR = 600 * 16
OUTPUT_WEIGHT_SCALE_DENOMINATOR = 127
PSQT_SCALE = 600 * 16
OUTPUT_DIVISOR = 16
NETWORK_TO_SCORE = 600

FT_ACTIVATION_SHIFT = 6
DENSE_ACTIVATION_SHIFT = 6
ACTIVATION_MIN = 0
ACTIVATION_MAX = 127
MAX_SAFE_BIAS_MAGNITUDE = 1 << 30
MAX_PSQT_MAGNITUDE = 1 << 20
ROUND_TO_NEAREST_TIES_EVEN = 1
RULE50_POSTPROCESSOR_V1 = 1
FEATURE_ORDER_A1_H8_V1 = 1
FIRST_DOMAIN_V3_STREAM = 4

SHA256_RE = re.compile(r"^[0-9A-F]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")

PROVENANCE_KEYS = {
    "checkpoint_sha256",
    "container_schema",
    "source_commit",
    "source_dirty",
    "train_file_sha256",
    "training_architecture_structural_sha256",
    "training_receipt_sha256",
    "validation_file_sha256",
    "wdl_calibration_sha256",
}


class ContainerError(ValueError):
    """Raised when a V3 network violates the frozen container contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContainerError(message)


@dataclass(frozen=True, slots=True)
class SectionSpec:
    section_id: int
    name: str
    dtype: str
    shape: tuple[int, ...]

    @property
    def elements(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    @property
    def byte_length(self) -> int:
        return self.elements * DTYPE_BYTES[self.dtype]

    def descriptor(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "id": self.section_id,
            "name": self.name,
            "order": "row-major" if len(self.shape) == 2 else "lane-major",
            "shape": list(self.shape),
        }


# Every section is at most two dimensional so the frozen 64 byte directory entry
# is reused verbatim. Bucketed tensors are flattened bucket-major.
SECTIONS: tuple[SectionSpec, ...] = (
    SectionSpec(1, "ft_weights", "i16", (FT_ROWS, FT_LANES)),
    SectionSpec(2, "ft_bias", "i32", (FT_LANES,)),
    SectionSpec(3, "psqt_weights", "i32", (FT_ROWS, PSQT_COLUMNS)),
    SectionSpec(4, "hidden0_weights", "i8", (PHASE_BUCKETS * HIDDEN0_LANES, FT_LANES)),
    SectionSpec(5, "hidden0_bias", "i32", (PHASE_BUCKETS * HIDDEN0_LANES,)),
    SectionSpec(6, "hidden1_weights", "i8", (PHASE_BUCKETS * HIDDEN1_LANES, HIDDEN0_LANES)),
    SectionSpec(7, "hidden1_bias", "i32", (PHASE_BUCKETS * HIDDEN1_LANES,)),
    SectionSpec(8, "output_weights", "i8", (PHASE_BUCKETS * OUTPUT_HEADS, HIDDEN1_LANES)),
    SectionSpec(9, "output_bias", "i32", (PHASE_BUCKETS * OUTPUT_HEADS,)),
    SectionSpec(10, "phase_lookup", "i8", (PHASE_LOOKUP_ENTRIES,)),
)
SECTIONS_BY_NAME = {section.name: section for section in SECTIONS}
PARAMETER_BYTES = sum(section.byte_length for section in SECTIONS)

ROLES = ["HP", "HN", "HB", "HR", "HQ", "RP", "RN", "RB", "RR", "RQ", "RK"]


def contextual_descriptor() -> list[dict[str, object]]:
    predicate = {
        "frontier_square": "square of the most advanced White pawn on the file, ranks 1-7",
        "rearmost_square": "square of the least advanced White pawn on the file, ranks 1-7",
        "file_count": "number of White pawns on the file, states 1-7, file-major",
        "frontier_blocked": "any occupancy on the square directly ahead of the frontier pawn",
        "frontier_supported": "a White pawn one rank behind on an adjacent file",
        "frontier_phalanx": "a White pawn on the same rank on an adjacent file",
    }
    return [
        {
            "base": V3_BLOCK_BASES[name],
            "max_active_rows": 8,
            "name": name,
            "predicate": predicate[name],
            "width": V3_BLOCK_WIDTHS[name],
        }
        for name in V3_BLOCK_ORDER
    ]


def descriptor() -> dict[str, object]:
    return {
        "container_schema": CONTAINER_SCHEMA,
        "feature_order": "physical squares A1 through H8",
        "feature_order_version": FEATURE_ORDER_A1_H8_V1,
        "feature_schema": NETWORK_SCHEMA_NAME,
        "format_version": FORMAT_VERSION,
        "integer": {
            "accumulation": "signed int32, non-saturating",
            "activation": "clip(max(affine, 0) >> 6, 0, 127)",
            "activation_type": "uint8",
            "bias_type": "int32",
            "byte_order": "little-endian",
            "dense_scale": DENSE_SCALE,
            "dense_weight_type": "int8",
            "feature_scale": FT_SCALE,
            "feature_weight_type": "int16",
            "hidden_bias_scale": HIDDEN_BIAS_SCALE,
            "maximum_bias_magnitude": MAX_SAFE_BIAS_MAGNITUDE,
            "maximum_psqt_magnitude": MAX_PSQT_MAGNITUDE,
            "output_bias_scale": OUTPUT_BIAS_SCALE,
            "output_conversion": "signed int32 / 16, truncation toward zero",
            "output_weight_scale": f"{OUTPUT_WEIGHT_SCALE_NUMERATOR}/{OUTPUT_WEIGHT_SCALE_DENOMINATOR}",
            "quantization_rounding": "nearest, ties to even",
            "value_semantics": "container value equals 600 times the trained model output",
        },
        "network_schema": NETWORK_SCHEMA_NAME,
        "network_schema_id": NETWORK_SCHEMA_ID,
        "phase": {
            "buckets": PHASE_BUCKETS,
            "lookup_section": "phase_lookup",
            "source": "white_piece_count",
            "domain": [0, 36],
            "monotone_nondecreasing": True,
        },
        "physical_piece_contract": "all pawns remain PAWN; White has no king; Black has one king",
        "psqt": {
            "columns": PSQT_COLUMNS,
            "index": "bucket * 2 + side_to_move",
            "note": "one perspective only; the legacy signed two-perspective fold does not apply",
            "scale": PSQT_SCALE,
        },
        "rule50": {
            "clamp": "VALUE_TB_LOSS_IN_MAX_PLY+1 through VALUE_TB_WIN_IN_MAX_PLY-1",
            "formula": "trunc_toward_zero(value * (100 - min(rule50, 100)) / 100)",
            "version": RULE50_POSTPROCESSOR_V1,
        },
        "sections": [section.descriptor() for section in SECTIONS],
        "side_to_move_heads": ["WHITE", "BLACK"],
        "stream": {
            "active_rows_max": V3_MAX_ACTIVE_ROWS,
            "contextual_blocks": contextual_descriptor(),
            "contextual_rows": V3_GLOBAL_DIMENSIONS - 704,
            "g0_index": "fixed_role * 64 + absolute_square",
            "g0_rows": 704,
            "lanes": FT_LANES,
            "name": "v3_stream",
            "roles": ROLES,
            "rows": FT_ROWS,
        },
        "topology": [FT_LANES, HIDDEN0_LANES, HIDDEN1_LANES, OUTPUT_HEADS],
    }


def structural_bytes() -> bytes:
    return canonical_json(descriptor())


def structural_sha256() -> str:
    return sha256_bytes(structural_bytes())


def _digest_bytes(value: object, label: str) -> bytes:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} is not uppercase SHA-256",
    )
    _require(value != "0" * 64, f"{label} must be nonzero")
    return bytes.fromhex(str(value))


def _validate_provenance(
    provenance: Mapping[str, object], training_structural_sha256: str
) -> dict[str, object]:
    _require(set(provenance) == PROVENANCE_KEYS, "provenance fields do not match the contract")
    result = dict(provenance)
    _require(result["container_schema"] == CONTAINER_SCHEMA, "provenance container schema mismatch")
    _require(result["source_dirty"] is False, "dirty training source is forbidden")
    commit = result["source_commit"]
    _require(
        isinstance(commit, str) and COMMIT_RE.fullmatch(commit) is not None and commit != "0" * 40,
        "source commit is not a nonzero full Git object ID",
    )
    result["source_commit"] = str(commit).lower()
    _require(
        result["training_architecture_structural_sha256"] == training_structural_sha256,
        "training architecture structural hash mismatch",
    )
    for key in (
        "checkpoint_sha256",
        "training_receipt_sha256",
        "train_file_sha256",
        "validation_file_sha256",
        "wdl_calibration_sha256",
    ):
        _digest_bytes(result[key], key)
    return result


def _unpack(payload: bytes, dtype: str) -> list[int]:
    code = {"i8": "b", "i16": "<h", "i32": "<i"}[dtype]
    width = DTYPE_BYTES[dtype]
    return [struct.unpack_from(code, payload, offset)[0] for offset in range(0, len(payload), width)]


def validate_parameter_values(sections: Mapping[str, bytes]) -> None:
    """Enforce every registered magnitude bound before a container is sealed."""

    for section in SECTIONS:
        values = _unpack(sections[section.name], section.dtype)
        if section.name == "psqt_weights":
            worst = max((abs(value) for value in values), default=0)
            _require(
                worst <= MAX_PSQT_MAGNITUDE,
                f"psqt_weights magnitude {worst} exceeds the registered bound",
            )
        elif section.name.endswith("_bias"):
            worst = max((abs(value) for value in values), default=0)
            _require(
                worst <= MAX_SAFE_BIAS_MAGNITUDE,
                f"section {section.name} exceeds the registered safe bias magnitude",
            )
        elif section.name == "phase_lookup":
            _require(len(values) == PHASE_LOOKUP_ENTRIES, "phase lookup length drifted")
            _require(
                all(0 <= value < PHASE_BUCKETS for value in values),
                "phase lookup contains an out-of-range bucket",
            )
            _require(
                all(values[index] <= values[index + 1] for index in range(len(values) - 1)),
                "phase lookup is not monotone nondecreasing in white_piece_count",
            )
            _require(
                sorted(set(values)) == list(range(PHASE_BUCKETS)),
                "phase lookup does not reach every bucket",
            )


def _put_u16(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<H", buffer, offset, value)


def _put_u32(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<I", buffer, offset, value)


def _put_i32(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<i", buffer, offset, value)


def _put_u64(buffer: bytearray, offset: int, value: int) -> None:
    struct.pack_into("<Q", buffer, offset, value)


def build_container(
    sections: Mapping[str, bytes],
    provenance: Mapping[str, object],
    training_structural_sha256: str,
) -> tuple[bytes, dict[str, object]]:
    """Build one deterministic, fully authenticated V3 network container."""

    provenance_value = _validate_provenance(provenance, training_structural_sha256)
    _require(
        set(sections) == set(SECTIONS_BY_NAME), "parameter section names do not match the schema"
    )
    validate_parameter_values(sections)

    payloads: list[tuple[SectionSpec, bytes, int, str]] = []
    parameter_payload = bytearray()
    for section in SECTIONS:
        payload = bytes(sections[section.name])
        _require(
            len(payload) == section.byte_length,
            f"section {section.name} is {len(payload)} bytes instead of {section.byte_length}",
        )
        offset = len(parameter_payload)
        payloads.append((section, payload, offset, sha256_bytes(payload)))
        parameter_payload.extend(payload)
    _require(len(parameter_payload) == PARAMETER_BYTES, "parameter byte count drifted")

    structure = structural_bytes()
    provenance_payload = canonical_json(provenance_value)
    parameter_offset = HEADER_BYTES + len(structure) + len(provenance_payload)
    file_bytes = parameter_offset + len(parameter_payload)
    _require(file_bytes <= MAXIMUM_CONTAINER_BYTES, "container exceeds the registered size limit")
    parameter_sha = sha256_bytes(bytes(parameter_payload))
    provenance_sha = sha256_bytes(provenance_payload)
    structure_sha = structural_sha256()

    header = bytearray(HEADER_BYTES)
    header[0:8] = MAGIC
    _put_u16(header, 8, FORMAT_VERSION)
    _put_u16(header, 10, HEADER_BYTES)
    _put_u32(header, 12, ENDIAN_TAG)
    _put_u32(header, 16, NETWORK_SCHEMA_ID)
    _put_u16(header, 20, SECTION_COUNT)
    _put_u16(header, 22, 0)
    _put_u64(header, 24, file_bytes)
    _put_u64(header, 32, HEADER_BYTES)
    _put_u64(header, 40, len(structure))
    header[48:80] = bytes.fromhex(structure_sha)
    _put_u64(header, 80, HEADER_BYTES + len(structure))
    _put_u64(header, 88, len(provenance_payload))
    header[96:128] = bytes.fromhex(provenance_sha)
    _put_u64(header, 128, parameter_offset)
    _put_u64(header, 136, len(parameter_payload))
    header[144:176] = bytes.fromhex(parameter_sha)

    schema_name = NETWORK_SCHEMA_NAME.encode("ascii")
    _require(len(schema_name) < 64, "network schema name does not fit the header")
    header[176 : 176 + len(schema_name)] = schema_name
    header[240:272] = bytes.fromhex(training_structural_sha256)
    header[272:304] = _digest_bytes(provenance_value["checkpoint_sha256"], "checkpoint_sha256")
    header[304:336] = _digest_bytes(
        provenance_value["training_receipt_sha256"], "training_receipt_sha256"
    )
    header[336:368] = _digest_bytes(provenance_value["train_file_sha256"], "train_file_sha256")
    header[368:400] = _digest_bytes(
        provenance_value["validation_file_sha256"], "validation_file_sha256"
    )
    header[400:432] = _digest_bytes(
        provenance_value["wdl_calibration_sha256"], "wdl_calibration_sha256"
    )
    header[432:452] = bytes.fromhex(str(provenance_value["source_commit"]))
    header[452] = 0
    header[453] = ROUND_TO_NEAREST_TIES_EVEN
    header[454] = FT_ACTIVATION_SHIFT
    header[455] = DENSE_ACTIVATION_SHIFT
    _put_u32(header, 456, FIRST_DOMAIN_V3_STREAM)
    _put_u32(header, 460, FT_ROWS)
    _put_u32(header, 464, FT_LANES)
    _put_u32(header, 468, 704)
    _put_u32(header, 472, V3_GLOBAL_DIMENSIONS - 704)
    _put_u32(header, 476, HIDDEN0_LANES)
    _put_u32(header, 480, HIDDEN1_LANES)
    _put_u32(header, 484, OUTPUT_HEADS)
    _put_u32(header, 488, PHASE_BUCKETS)
    _put_u32(header, 492, FT_SCALE)
    _put_u32(header, 496, DENSE_SCALE)
    _put_i32(header, 500, ACTIVATION_MIN)
    _put_i32(header, 504, ACTIVATION_MAX)
    _put_i32(header, 508, OUTPUT_DIVISOR)
    _put_i32(header, 512, MAX_SAFE_BIAS_MAGNITUDE)
    _put_u32(header, 516, RULE50_POSTPROCESSOR_V1)
    _put_u32(header, 520, FEATURE_ORDER_A1_H8_V1)
    _put_u32(header, 524, DIRECTORY_OFFSET)
    _put_u32(header, 528, DIRECTORY_ENTRY_BYTES)
    # V3 extension block. Offsets 532 through 576 are zero in every V2 container.
    _put_u32(header, 532, OUTPUT_WEIGHT_SCALE_NUMERATOR)
    _put_u32(header, 536, OUTPUT_WEIGHT_SCALE_DENOMINATOR)
    _put_u32(header, 540, OUTPUT_BIAS_SCALE)
    _put_u32(header, 544, PSQT_SCALE)
    _put_u32(header, 548, PSQT_COLUMNS)
    _put_i32(header, 552, MAX_PSQT_MAGNITUDE)
    _put_u32(header, 556, V3_MAX_ACTIVE_ROWS)
    _put_u32(header, 560, PHASE_LOOKUP_ENTRIES)
    _put_u32(header, 564, NETWORK_TO_SCORE)
    _put_u32(header, 568, HIDDEN_BIAS_SCALE)
    _put_u32(header, 572, len(V3_BLOCK_ORDER))

    for index, (section, payload, offset, digest) in enumerate(payloads):
        entry = DIRECTORY_OFFSET + index * DIRECTORY_ENTRY_BYTES
        _put_u16(header, entry, section.section_id)
        header[entry + 2] = DTYPE_CODES[section.dtype]
        header[entry + 3] = len(section.shape)
        _put_u32(header, entry + 4, section.shape[0])
        _put_u32(header, entry + 8, section.shape[1] if len(section.shape) == 2 else 0)
        _put_u64(header, entry + 12, offset)
        _put_u64(header, entry + 20, len(payload))
        header[entry + 28 : entry + 60] = bytes.fromhex(digest)
        _put_u32(header, entry + 60, 0)

    container = bytes(header) + structure + provenance_payload + bytes(parameter_payload)
    _require(len(container) == file_bytes, "container length does not match its header")
    receipt = {
        "schema": "HORDE_V3_INTEGER_EXPORT_RECEIPT_V1",
        "container_schema": CONTAINER_SCHEMA,
        "network_schema": NETWORK_SCHEMA_NAME,
        "network_schema_id": NETWORK_SCHEMA_ID,
        "format_version": FORMAT_VERSION,
        "file_bytes": file_bytes,
        "file_sha256": sha256_bytes(container),
        "parameter_bytes": len(parameter_payload),
        "parameter_sha256": parameter_sha,
        "provenance": provenance_value,
        "provenance_bytes": len(provenance_payload),
        "provenance_sha256": provenance_sha,
        "structure_bytes": len(structure),
        "container_structural_sha256": structure_sha,
        "training_architecture_structural_sha256": training_structural_sha256,
        "sections": [
            {**section.descriptor(), "bytes": len(payload), "offset": offset, "sha256": digest}
            for section, payload, offset, digest in payloads
        ],
        "claims": {
            "production_dispatch": False,
            "strength_evidence": False,
        },
    }
    return container, receipt


@dataclass(frozen=True, slots=True)
class ParsedContainer:
    provenance: dict[str, object]
    sections: dict[str, bytes]
    parameter_sha256: str
    file_sha256: str
    training_structural_sha256: str


def read_container(path: Path) -> ParsedContainer:
    """Fail-closed reader. Every framing, identity and bound is re-checked."""

    payload = Path(path).read_bytes()
    _require(len(payload) > HEADER_BYTES, "container is shorter than its fixed header")
    _require(len(payload) <= MAXIMUM_CONTAINER_BYTES, "container exceeds the registered size limit")
    header = payload[:HEADER_BYTES]
    _require(header[:8] == MAGIC, "container magic is not HSV2INT")
    version, header_bytes = struct.unpack_from("<HH", header, 8)
    _require(version == FORMAT_VERSION, f"unsupported container format version {version}")
    _require(header_bytes == HEADER_BYTES, "container header size mismatch")
    _require(struct.unpack_from("<I", header, 12)[0] == ENDIAN_TAG, "container byte order mismatch")
    schema_id = struct.unpack_from("<I", header, 16)[0]
    _require(schema_id == NETWORK_SCHEMA_ID, f"unknown V3 network schema id {schema_id:#010x}")
    section_count, reserved = struct.unpack_from("<HH", header, 20)
    _require(section_count == SECTION_COUNT, "container section count mismatch")
    _require(reserved == 0, "reserved header field is not zero")

    file_bytes = struct.unpack_from("<Q", header, 24)[0]
    _require(file_bytes == len(payload), "container length disagrees with its header")
    structure_offset, structure_length = struct.unpack_from("<QQ", header, 32)
    _require(structure_offset == HEADER_BYTES, "structure does not follow the header")
    structure = payload[structure_offset : structure_offset + structure_length]
    _require(len(structure) == structure_length, "structure section is truncated")
    _require(
        header[48:80] == bytes.fromhex(sha256_bytes(structure)),
        "structure SHA-256 mismatch",
    )
    _require(structure == structural_bytes(), "structure does not match the registered V3 schema")

    provenance_offset, provenance_length = struct.unpack_from("<QQ", header, 80)
    _require(
        provenance_offset == structure_offset + structure_length,
        "provenance does not follow the structure",
    )
    provenance_payload = payload[provenance_offset : provenance_offset + provenance_length]
    _require(len(provenance_payload) == provenance_length, "provenance section is truncated")
    _require(
        header[96:128] == bytes.fromhex(sha256_bytes(provenance_payload)),
        "provenance SHA-256 mismatch",
    )
    try:
        provenance = json.loads(provenance_payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContainerError(f"provenance is not canonical ASCII JSON: {error}") from error
    _require(
        canonical_json(provenance) == provenance_payload, "provenance is not canonical JSON"
    )
    training_structural = str(provenance.get("training_architecture_structural_sha256", ""))
    _require(
        header[240:272] == bytes.fromhex(training_structural),
        "training structural hash disagrees with the header",
    )
    provenance = _validate_provenance(provenance, training_structural)

    parameter_offset, parameter_length = struct.unpack_from("<QQ", header, 128)
    _require(
        parameter_offset == provenance_offset + provenance_length,
        "parameters do not follow the provenance",
    )
    _require(parameter_length == PARAMETER_BYTES, "parameter byte count mismatch")
    parameters = payload[parameter_offset : parameter_offset + parameter_length]
    _require(len(parameters) == parameter_length, "parameter payload is truncated")
    parameter_sha = sha256_bytes(parameters)
    _require(header[144:176] == bytes.fromhex(parameter_sha), "parameter SHA-256 mismatch")

    for index, expected in enumerate(SECTIONS):
        entry = DIRECTORY_OFFSET + index * DIRECTORY_ENTRY_BYTES
        section_id = struct.unpack_from("<H", header, entry)[0]
        _require(section_id == expected.section_id, f"section {index} id mismatch")
        _require(header[entry + 2] == DTYPE_CODES[expected.dtype], f"section {index} dtype mismatch")
        _require(header[entry + 3] == len(expected.shape), f"section {index} rank mismatch")
        rows, columns = struct.unpack_from("<II", header, entry + 4)
        _require(rows == expected.shape[0], f"section {index} row count mismatch")
        _require(
            columns == (expected.shape[1] if len(expected.shape) == 2 else 0),
            f"section {index} column count mismatch",
        )
        offset, length = struct.unpack_from("<QQ", header, entry + 12)
        _require(length == expected.byte_length, f"section {index} byte length mismatch")
        _require(offset + length <= parameter_length, f"section {index} escapes the payload")
        blob = parameters[offset : offset + length]
        _require(
            header[entry + 28 : entry + 60] == bytes.fromhex(sha256_bytes(blob)),
            f"section {expected.name} SHA-256 mismatch",
        )
        _require(struct.unpack_from("<I", header, entry + 60)[0] == 0, "section padding is not zero")

    sections = {}
    cursor = 0
    for expected in SECTIONS:
        sections[expected.name] = parameters[cursor : cursor + expected.byte_length]
        cursor += expected.byte_length
    _require(cursor == parameter_length, "section directory does not tile the payload")
    validate_parameter_values(sections)

    for offset, value, label in (
        (532, OUTPUT_WEIGHT_SCALE_NUMERATOR, "output weight scale numerator"),
        (536, OUTPUT_WEIGHT_SCALE_DENOMINATOR, "output weight scale denominator"),
        (540, OUTPUT_BIAS_SCALE, "output bias scale"),
        (544, PSQT_SCALE, "psqt scale"),
        (556, V3_MAX_ACTIVE_ROWS, "active row bound"),
        (564, NETWORK_TO_SCORE, "network to score"),
    ):
        _require(
            struct.unpack_from("<I", header, offset)[0] == value,
            f"header {label} disagrees with the registered contract",
        )

    return ParsedContainer(
        provenance=provenance,
        sections=sections,
        parameter_sha256=parameter_sha,
        file_sha256=sha256_bytes(payload),
        training_structural_sha256=training_structural,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, nargs="?")
    parser.add_argument("--descriptor", action="store_true")
    args = parser.parse_args()
    if args.descriptor or args.file is None:
        print(json.dumps(descriptor(), indent=2, sort_keys=True))
        print(f"structural_sha256 = {structural_sha256()}", flush=True)
        print(f"parameter_bytes   = {PARAMETER_BYTES}")
        return 0
    container = read_container(args.file)
    print(json.dumps({
        "file_sha256": container.file_sha256,
        "parameter_sha256": container.parameter_sha256,
        "provenance": container.provenance,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
