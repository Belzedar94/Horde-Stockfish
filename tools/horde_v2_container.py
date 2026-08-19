#!/usr/bin/env python3
"""Canonical codec for trained Horde NNUE V2 integer networks.

The codec is deliberately independent of PyTorch.  Training checkpoints are
converted by ``horde_v2_export.py``; this module owns only the immutable binary
contract, integrity checks, and deterministic container construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Mapping, Sequence


CONTAINER_SCHEMA = "HORDE_V2_INTEGER_NETWORK_V1"
FORMAT_VERSION = 1
MAGIC = b"HSV2INT\0"
HEADER_BYTES = 2048
ENDIAN_TAG = 0x01020304
DIRECTORY_OFFSET = 640
DIRECTORY_ENTRY_BYTES = 64
SECTION_COUNT = 10
MAXIMUM_CONTAINER_BYTES = 16 * 1024 * 1024

ROUND_TO_NEAREST_TIES_EVEN = 1
FIRST_DOMAIN_ROYAL = 1
FIRST_DOMAIN_ABSOLUTE_NONKING = 2
FIRST_DOMAIN_ROYAL_RANK8 = 3
RULE50_POSTPROCESSOR_V1 = 1
FEATURE_ORDER_A1_H8_V1 = 1

FT_SCALE = 127 * 64
DENSE_SCALE = 64
FT_ACTIVATION_SHIFT = 6
DENSE_ACTIVATION_SHIFT = 6
ACTIVATION_MIN = 0
ACTIVATION_MAX = 127
OUTPUT_DIVISOR = 16
MAX_SAFE_BIAS_MAGNITUDE = 1 << 30

GLOBAL_ROWS = 11 * 64
GLOBAL_LANES = 192
FIRST_LANES = 64
HIDDEN0_LANES = 32
HIDDEN1_LANES = 32
OUTPUT_HEADS = 2
PHASE_BUCKETS = 1

DTYPE_CODES = {"i8": 1, "i16": 2, "i32": 3}
DTYPE_BYTES = {"i8": 1, "i16": 2, "i32": 4}
SHA256_RE = re.compile(r"^[0-9A-F]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


class ContainerError(ValueError):
    """Raised when a V2 network violates the frozen container contract."""


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


@dataclass(frozen=True, slots=True)
class NetworkSpec:
    architecture: str
    schema_id: int
    schema_name: str
    training_structural_sha256: str
    first_domain_code: int
    first_domain_name: str
    first_rows: int

    @property
    def first_weight_name(self) -> str:
        return f"{self.first_domain_name}_weights"

    @property
    def first_bias_name(self) -> str:
        return f"{self.first_domain_name}_bias"

    @property
    def sections(self) -> tuple[SectionSpec, ...]:
        return (
            SectionSpec(1, self.first_weight_name, "i16", (self.first_rows, FIRST_LANES)),
            SectionSpec(2, self.first_bias_name, "i32", (FIRST_LANES,)),
            SectionSpec(3, "global_weights", "i16", (GLOBAL_ROWS, GLOBAL_LANES)),
            SectionSpec(4, "global_bias", "i32", (GLOBAL_LANES,)),
            SectionSpec(5, "hidden0_weights", "i8", (HIDDEN0_LANES, FIRST_LANES + GLOBAL_LANES)),
            SectionSpec(6, "hidden0_bias", "i32", (HIDDEN0_LANES,)),
            SectionSpec(7, "hidden1_weights", "i8", (HIDDEN1_LANES, HIDDEN0_LANES)),
            SectionSpec(8, "hidden1_bias", "i32", (HIDDEN1_LANES,)),
            SectionSpec(9, "output_weights", "i8", (OUTPUT_HEADS, HIDDEN1_LANES)),
            SectionSpec(10, "output_bias", "i32", (OUTPUT_HEADS,)),
        )

    @property
    def parameter_bytes(self) -> int:
        return sum(section.byte_length for section in self.sections)

    def descriptor(self) -> dict[str, object]:
        roles = ["HP", "HN", "HB", "HR", "HQ", "RP", "RN", "RB", "RR", "RQ"]
        if self.first_domain_code == FIRST_DOMAIN_ROYAL:
            first: dict[str, object] = {
                "active_rows_max": 51,
                "bias_policy": "one shared bias across all king buckets",
                "black_king_buckets": 32,
                "dimensions": self.first_rows,
                "index": "((bucket * 10 + role) * 64) + oriented_square",
                "lanes": FIRST_LANES,
                "mirror": "files A-D horizontally reflected; king canonicalized to E-H",
                "name": "royal",
                "roles": roles,
            }
        elif self.first_domain_code == FIRST_DOMAIN_ROYAL_RANK8:
            first = {
                "active_rows_max": 51,
                "bias_policy": "one shared bias across all king-rank buckets",
                "black_king_buckets": 8,
                "bucket_map": "black king rank",
                "dimensions": self.first_rows,
                "index": "((rank_bucket * 10 + role) * 64) + oriented_square",
                "lanes": FIRST_LANES,
                "mirror": "files A-D horizontally reflected; king canonicalized to E-H",
                "name": "royal_rank8",
                "refresh_key": "black king rank plus horizontal mirror bit",
                "roles": roles,
            }
        else:
            first = {
                "active_rows_max": 51,
                "dimensions": self.first_rows,
                "index": "role * 64 + absolute_square",
                "lanes": FIRST_LANES,
                "name": "absolute_nonking",
                "orientation": "absolute A1-H8",
                "roles": roles,
                "source_projection": "G0 rows 0-639; Black king omitted",
            }
        return {
            "container_schema": CONTAINER_SCHEMA,
            "feature_order": "physical squares A1 through H8",
            "feature_order_version": FEATURE_ORDER_A1_H8_V1,
            "feature_schema": "V2_BASE_P0",
            "first_domain": first,
            "format_version": FORMAT_VERSION,
            "global_domain": {
                "active_rows_max": 52,
                "dimensions": GLOBAL_ROWS,
                "index": "fixed_role * 64 + absolute_square",
                "lanes": GLOBAL_LANES,
                "name": "global",
                "roles": roles + ["RK"],
            },
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
                "maximum_bias_magnitude": MAX_SAFE_BIAS_MAGNITUDE,
                "output_conversion": "signed int32 / 16, truncation toward zero",
                "quantization_rounding": "nearest, ties to even",
            },
            "network_schema": self.schema_name,
            "network_schema_id": self.schema_id,
            "phase_buckets": PHASE_BUCKETS,
            "physical_piece_contract": "all pawns remain PAWN; White has no king; Black has one king",
            "rule50": {
                "clamp": "VALUE_TB_LOSS_IN_MAX_PLY+1 through VALUE_TB_WIN_IN_MAX_PLY-1",
                "formula": "trunc_toward_zero(value * (100 - min(rule50, 100)) / 100)",
                "version": RULE50_POSTPROCESSOR_V1,
            },
            "sections": [section.descriptor() for section in self.sections],
            "side_to_move_heads": ["WHITE", "BLACK"],
            "topology": [FIRST_LANES + GLOBAL_LANES, HIDDEN0_LANES, HIDDEN1_LANES, OUTPUT_HEADS],
            "training_architecture_structural_sha256": self.training_structural_sha256,
        }

    @property
    def structural_bytes(self) -> bytes:
        return canonical_json(self.descriptor())

    @property
    def structural_sha256(self) -> str:
        return sha256_bytes(self.structural_bytes)


SPECS: tuple[NetworkSpec, ...] = (
    NetworkSpec(
        architecture="v2-64x192",
        schema_id=0x00010001,
        schema_name="V2_BASE_P0_64X192",
        training_structural_sha256="5360985A5B08E43A6F7C23E8601DF159BAEE38E9306C3652867AA93EEAD39862",
        first_domain_code=FIRST_DOMAIN_ROYAL,
        first_domain_name="royal",
        first_rows=32 * 10 * 64,
    ),
    NetworkSpec(
        architecture="v2-c1-abs64x192",
        schema_id=0x00010002,
        schema_name="V2_C1_ABS_NONKING_64X192",
        training_structural_sha256="66538DB76A0248662A824545A2ECCD2AE84CD4805DDFCA206B5239FA3BDE45B6",
        first_domain_code=FIRST_DOMAIN_ABSOLUTE_NONKING,
        first_domain_name="absolute_nonking",
        first_rows=10 * 64,
    ),
    NetworkSpec(
        architecture="v2-c1-rank8-64x192",
        schema_id=0x00010003,
        schema_name="V2_C1_ROYAL_RANK8_64X192",
        training_structural_sha256="53A82F734EBCAD97508AD91D54027ADD10772A1DC85A612F5D395AEE08567083",
        first_domain_code=FIRST_DOMAIN_ROYAL_RANK8,
        first_domain_name="royal_rank8",
        first_rows=8 * 10 * 64,
    ),
)
SPECS_BY_ARCHITECTURE = {spec.architecture: spec for spec in SPECS}
SPECS_BY_SCHEMA = {spec.schema_name: spec for spec in SPECS}
SPECS_BY_ID = {spec.schema_id: spec for spec in SPECS}


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


@dataclass(frozen=True, slots=True)
class ParsedContainer:
    spec: NetworkSpec
    provenance: dict[str, object]
    sections: dict[str, bytes]
    parameter_sha256: str
    file_sha256: str


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContainerError(message)


def _digest_bytes(value: object, label: str) -> bytes:
    _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"{label} is not uppercase SHA-256")
    _require(value != "0" * 64, f"{label} must be nonzero")
    return bytes.fromhex(value)


def _validate_provenance(spec: NetworkSpec, provenance: Mapping[str, object]) -> dict[str, object]:
    _require(set(provenance) == PROVENANCE_KEYS, "provenance fields do not match the V1 contract")
    result = dict(provenance)
    _require(result["container_schema"] == CONTAINER_SCHEMA, "provenance container schema mismatch")
    _require(result["source_dirty"] is False, "dirty training source is forbidden")
    _require(
        isinstance(result["source_commit"], str) and COMMIT_RE.fullmatch(result["source_commit"]) is not None,
        "source commit is not a full Git object ID",
    )
    _require(result["source_commit"] != "0" * 40, "source commit must be nonzero")
    result["source_commit"] = str(result["source_commit"]).lower()
    _require(
        result["training_architecture_structural_sha256"] == spec.training_structural_sha256,
        "training architecture structural hash mismatch",
    )
    for key in (
        "checkpoint_sha256",
        "training_receipt_sha256",
        "train_file_sha256",
        "validation_file_sha256",
        "wdl_calibration_sha256",
        "training_architecture_structural_sha256",
    ):
        _digest_bytes(result[key], key)
    return result


def _validate_parameter_values(spec: NetworkSpec, sections: Mapping[str, bytes]) -> None:
    for section in spec.sections:
        if section.dtype != "i32" or not section.name.endswith("_bias"):
            continue
        payload = sections[section.name]
        _require(len(payload) % 4 == 0, f"section {section.name} is not aligned to int32")
        for (value,) in struct.iter_unpack("<i", payload):
            _require(
                abs(value) <= MAX_SAFE_BIAS_MAGNITUDE,
                f"section {section.name} exceeds the registered safe bias magnitude",
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
    spec: NetworkSpec,
    sections: Mapping[str, bytes],
    provenance: Mapping[str, object],
) -> tuple[bytes, dict[str, object]]:
    """Build one deterministic, fully authenticated V2 network container."""

    provenance_value = _validate_provenance(spec, provenance)
    expected_names = {section.name for section in spec.sections}
    _require(set(sections) == expected_names, "parameter section names do not match the schema")
    _validate_parameter_values(spec, sections)

    section_payloads: list[tuple[SectionSpec, bytes, int, str]] = []
    parameter_payload = bytearray()
    for section in spec.sections:
        payload = bytes(sections[section.name])
        _require(
            len(payload) == section.byte_length,
            f"section {section.name} is {len(payload)} bytes instead of {section.byte_length}",
        )
        offset = len(parameter_payload)
        digest = sha256_bytes(payload)
        section_payloads.append((section, payload, offset, digest))
        parameter_payload.extend(payload)
    _require(len(parameter_payload) == spec.parameter_bytes, "parameter byte count drifted")

    structure = spec.structural_bytes
    provenance_payload = canonical_json(provenance_value)
    parameter_offset = HEADER_BYTES + len(structure) + len(provenance_payload)
    file_bytes = parameter_offset + len(parameter_payload)
    parameter_sha = sha256_bytes(bytes(parameter_payload))
    provenance_sha = sha256_bytes(provenance_payload)

    header = bytearray(HEADER_BYTES)
    header[0:8] = MAGIC
    _put_u16(header, 8, FORMAT_VERSION)
    _put_u16(header, 10, HEADER_BYTES)
    _put_u32(header, 12, ENDIAN_TAG)
    _put_u32(header, 16, spec.schema_id)
    _put_u16(header, 20, SECTION_COUNT)
    _put_u16(header, 22, 0)
    _put_u64(header, 24, file_bytes)
    _put_u64(header, 32, HEADER_BYTES)
    _put_u64(header, 40, len(structure))
    header[48:80] = bytes.fromhex(spec.structural_sha256)
    _put_u64(header, 80, HEADER_BYTES + len(structure))
    _put_u64(header, 88, len(provenance_payload))
    header[96:128] = bytes.fromhex(provenance_sha)
    _put_u64(header, 128, parameter_offset)
    _put_u64(header, 136, len(parameter_payload))
    header[144:176] = bytes.fromhex(parameter_sha)

    schema_name = spec.schema_name.encode("ascii")
    _require(len(schema_name) < 64, "network schema name does not fit the header")
    header[176 : 176 + len(schema_name)] = schema_name
    header[240:272] = bytes.fromhex(spec.training_structural_sha256)
    header[272:304] = _digest_bytes(provenance_value["checkpoint_sha256"], "checkpoint_sha256")
    header[304:336] = _digest_bytes(provenance_value["training_receipt_sha256"], "training_receipt_sha256")
    header[336:368] = _digest_bytes(provenance_value["train_file_sha256"], "train_file_sha256")
    header[368:400] = _digest_bytes(provenance_value["validation_file_sha256"], "validation_file_sha256")
    header[400:432] = _digest_bytes(provenance_value["wdl_calibration_sha256"], "wdl_calibration_sha256")
    header[432:452] = bytes.fromhex(str(provenance_value["source_commit"]))
    header[452] = 0
    header[453] = ROUND_TO_NEAREST_TIES_EVEN
    header[454] = FT_ACTIVATION_SHIFT
    header[455] = DENSE_ACTIVATION_SHIFT
    _put_u32(header, 456, spec.first_domain_code)
    _put_u32(header, 460, spec.first_rows)
    _put_u32(header, 464, FIRST_LANES)
    _put_u32(header, 468, GLOBAL_ROWS)
    _put_u32(header, 472, GLOBAL_LANES)
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

    for index, (section, payload, offset, digest) in enumerate(section_payloads):
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
        "schema": "HORDE_V2_INTEGER_EXPORT_RECEIPT_V1",
        "container_schema": CONTAINER_SCHEMA,
        "format_version": FORMAT_VERSION,
        "network_schema": spec.schema_name,
        "network_schema_id": spec.schema_id,
        "container_structural_sha256": spec.structural_sha256,
        "training_architecture_structural_sha256": spec.training_structural_sha256,
        "file_bytes": len(container),
        "file_sha256": sha256_bytes(container),
        "parameter_bytes": len(parameter_payload),
        "parameter_sha256": parameter_sha,
        "structure_bytes": len(structure),
        "provenance_bytes": len(provenance_payload),
        "provenance_sha256": provenance_sha,
        "sections": [
            {
                **section.descriptor(),
                "bytes": len(payload),
                "offset": offset,
                "sha256": digest,
            }
            for section, payload, offset, digest in section_payloads
        ],
        "provenance": provenance_value,
    }
    return container, receipt


def _u16(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<H", payload, offset)[0]


def _u32(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<I", payload, offset)[0]


def _i32(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<i", payload, offset)[0]


def _u64(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", payload, offset)[0]


def parse_container(payload: bytes) -> ParsedContainer:
    """Read and authenticate a complete V2 container, failing closed."""

    _require(len(payload) >= HEADER_BYTES, "container is shorter than its fixed header")
    _require(len(payload) <= MAXIMUM_CONTAINER_BYTES, "container exceeds the V2 size ceiling")
    header = payload[:HEADER_BYTES]
    _require(header[:8] == MAGIC, "container magic mismatch")
    _require(_u16(header, 8) == FORMAT_VERSION, "container version mismatch")
    _require(_u16(header, 10) == HEADER_BYTES, "container header size mismatch")
    _require(_u32(header, 12) == ENDIAN_TAG, "container byte-order tag mismatch")
    spec = SPECS_BY_ID.get(_u32(header, 16))
    _require(spec is not None, "unregistered V2 network schema id")
    assert spec is not None
    _require(_u16(header, 20) == SECTION_COUNT and _u16(header, 22) == 0, "section directory contract mismatch")
    _require(_u64(header, 24) == len(payload), "declared file length mismatch")

    nul = header[176:240].find(b"\0")
    _require(nul >= 0, "schema name is not NUL terminated")
    _require(header[176 + nul : 240] == b"\0" * (64 - nul), "schema name padding is nonzero")
    _require(header[176 : 176 + nul].decode("ascii") == spec.schema_name, "schema name mismatch")

    structure_offset = _u64(header, 32)
    structure_bytes = _u64(header, 40)
    provenance_offset = _u64(header, 80)
    provenance_bytes = _u64(header, 88)
    parameter_offset = _u64(header, 128)
    parameter_bytes = _u64(header, 136)
    _require(structure_offset == HEADER_BYTES, "structural descriptor offset mismatch")
    _require(provenance_offset == structure_offset + structure_bytes, "provenance is not contiguous")
    _require(parameter_offset == provenance_offset + provenance_bytes, "parameter payload is not contiguous")
    _require(parameter_offset + parameter_bytes == len(payload), "parameter payload length mismatch")
    _require(parameter_bytes == spec.parameter_bytes, "parameter byte count does not match schema")

    structure = payload[structure_offset:provenance_offset]
    _require(structure == spec.structural_bytes, "structural descriptor bytes mismatch")
    _require(header[48:80] == bytes.fromhex(spec.structural_sha256), "structural SHA-256 mismatch")
    provenance_payload = payload[provenance_offset:parameter_offset]
    _require(hashlib.sha256(provenance_payload).digest() == header[96:128], "provenance SHA-256 mismatch")
    try:
        provenance_root = json.loads(provenance_payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContainerError(f"invalid provenance JSON: {error}") from error
    _require(isinstance(provenance_root, dict), "provenance root is not an object")
    provenance = _validate_provenance(spec, provenance_root)
    _require(canonical_json(provenance) == provenance_payload, "provenance JSON is not canonical")

    fixed_hashes = (
        (240, spec.training_structural_sha256, "training structural"),
        (272, provenance["checkpoint_sha256"], "checkpoint"),
        (304, provenance["training_receipt_sha256"], "training receipt"),
        (336, provenance["train_file_sha256"], "training file"),
        (368, provenance["validation_file_sha256"], "validation file"),
        (400, provenance["wdl_calibration_sha256"], "WDL calibration"),
    )
    for offset, value, label in fixed_hashes:
        _require(header[offset : offset + 32] == bytes.fromhex(str(value)), f"{label} header identity mismatch")
    _require(header[432:452] == bytes.fromhex(str(provenance["source_commit"])), "source commit header mismatch")
    _require(header[452] == 0, "dirty source flag is forbidden")
    _require(header[453] == ROUND_TO_NEAREST_TIES_EVEN, "quantization rounding mismatch")
    _require(header[454] == FT_ACTIVATION_SHIFT and header[455] == DENSE_ACTIVATION_SHIFT, "activation shift mismatch")

    expected_u32 = {
        456: spec.first_domain_code,
        460: spec.first_rows,
        464: FIRST_LANES,
        468: GLOBAL_ROWS,
        472: GLOBAL_LANES,
        476: HIDDEN0_LANES,
        480: HIDDEN1_LANES,
        484: OUTPUT_HEADS,
        488: PHASE_BUCKETS,
        492: FT_SCALE,
        496: DENSE_SCALE,
        516: RULE50_POSTPROCESSOR_V1,
        520: FEATURE_ORDER_A1_H8_V1,
        524: DIRECTORY_OFFSET,
        528: DIRECTORY_ENTRY_BYTES,
    }
    for offset, expected in expected_u32.items():
        _require(_u32(header, offset) == expected, f"header field at {offset} mismatches the schema")
    expected_i32 = {500: ACTIVATION_MIN, 504: ACTIVATION_MAX, 508: OUTPUT_DIVISOR, 512: MAX_SAFE_BIAS_MAGNITUDE}
    for offset, expected in expected_i32.items():
        _require(_i32(header, offset) == expected, f"integer field at {offset} mismatches the schema")
    _require(header[532:DIRECTORY_OFFSET] == b"\0" * (DIRECTORY_OFFSET - 532), "reserved fixed-header bytes are nonzero")

    parameter_payload = payload[parameter_offset:]
    _require(hashlib.sha256(parameter_payload).digest() == header[144:176], "parameter payload SHA-256 mismatch")
    sections: dict[str, bytes] = {}
    expected_offset = 0
    for index, section in enumerate(spec.sections):
        entry = DIRECTORY_OFFSET + index * DIRECTORY_ENTRY_BYTES
        _require(_u16(header, entry) == section.section_id, f"section {section.name} id mismatch")
        _require(header[entry + 2] == DTYPE_CODES[section.dtype], f"section {section.name} dtype mismatch")
        _require(header[entry + 3] == len(section.shape), f"section {section.name} rank mismatch")
        _require(_u32(header, entry + 4) == section.shape[0], f"section {section.name} first dimension mismatch")
        expected_second = section.shape[1] if len(section.shape) == 2 else 0
        _require(_u32(header, entry + 8) == expected_second, f"section {section.name} second dimension mismatch")
        offset = _u64(header, entry + 12)
        length = _u64(header, entry + 20)
        _require(offset == expected_offset, f"section {section.name} is not contiguous")
        _require(length == section.byte_length, f"section {section.name} byte length mismatch")
        _require(_u32(header, entry + 60) == 0, f"section {section.name} reserved field is nonzero")
        section_payload = parameter_payload[offset : offset + length]
        _require(len(section_payload) == length, f"section {section.name} is truncated")
        _require(hashlib.sha256(section_payload).digest() == header[entry + 28 : entry + 60], f"section {section.name} SHA-256 mismatch")
        sections[section.name] = section_payload
        expected_offset += length
    directory_end = DIRECTORY_OFFSET + SECTION_COUNT * DIRECTORY_ENTRY_BYTES
    _require(header[directory_end:] == b"\0" * (HEADER_BYTES - directory_end), "reserved header tail is nonzero")
    _require(expected_offset == len(parameter_payload), "parameter directory does not consume the payload")
    _validate_parameter_values(spec, sections)

    return ParsedContainer(
        spec=spec,
        provenance=provenance,
        sections=sections,
        parameter_sha256=sha256_bytes(parameter_payload),
        file_sha256=sha256_bytes(payload),
    )


def read_container(path: Path) -> ParsedContainer:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"network container does not exist: {resolved}")
    return parse_container(resolved.read_bytes())


def write_container_exclusive(path: Path, payload: bytes) -> None:
    resolved = path.expanduser().resolve()
    _require(resolved.parent.is_dir(), f"output parent does not exist: {resolved.parent}")
    with resolved.open("xb") as output:
        output.write(payload)


if __name__ == "__main__":
    for network_spec in SPECS:
        print(
            json.dumps(
                {
                    "architecture": network_spec.architecture,
                    "network_schema": network_spec.schema_name,
                    "network_schema_id": network_spec.schema_id,
                    "parameter_bytes": network_spec.parameter_bytes,
                    "container_structural_sha256": network_spec.structural_sha256,
                    "training_architecture_structural_sha256": network_spec.training_structural_sha256,
                },
                sort_keys=True,
            )
        )
