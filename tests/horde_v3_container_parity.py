#!/usr/bin/env python3
"""Fail-closed gates for the Horde V3 integer container, exporter and evaluator.

Four groups of checks, all of which must pass:

a) Build then read round trip. Sections come back byte identical, every digest
   agrees, and the registered structural hash and parameter byte count hold.
b) Adversarial rejections. Framing, identity, authentication and every
   registered magnitude bound must raise ``ContainerError``.
c) Quantization parity against the float model on authenticated records. The
   integer container must agree with ``trunc(v * 600)`` up to quantization, and
   critically must carry no systematic scale factor. That last assertion is the
   regression test for the V2 defect, where the same ratio was 0.849.
d) A layer trace fixture written to ``tests/data/horde_v3_layer_trace.json`` so
   the C++ evaluator can be compared lane by lane against this implementation.

The float reference in (c) and the truncation helpers are deliberately
reimplemented here rather than imported from the evaluator: a parity test that
shares code with the thing it tests proves nothing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPO_ROOT / "tools"
sys.path.insert(0, str(TOOLS))

# This file shares its module name with the codec it tests. If ``tests`` ever
# precedes ``tools`` on sys.path the import below silently resolves to this
# file, so bind the codec explicitly and fail loudly instead.
if "horde_v3_container" in sys.modules:
    _resolved = Path(getattr(sys.modules["horde_v3_container"], "__file__", "")).resolve()
    if _resolved != (TOOLS / "horde_v3_container.py").resolve():
        raise SystemExit(
            f"horde_v3_container resolved to {_resolved}; the tests/ copy is shadowing "
            f"the codec in {TOOLS}. Run this file as a script, not as a package module."
        )

import numpy as np  # noqa: E402
import torch  # noqa: E402

import horde_bin_v1 as wire  # noqa: E402
import horde_training_decoder as dec  # noqa: E402
import horde_training_models as models  # noqa: E402
import horde_v3_export as exporter  # noqa: E402
import horde_v3_integer_eval as integer_eval  # noqa: E402
from horde_v2_container import (  # noqa: E402
    DIRECTORY_ENTRY_BYTES,
    DIRECTORY_OFFSET,
    HEADER_BYTES,
    sha256_bytes,
)
from horde_v3_container import (  # noqa: E402
    MAX_PSQT_MAGNITUDE,
    MAX_SAFE_BIAS_MAGNITUDE,
    NETWORK_SCHEMA_ID,
    NETWORK_SCHEMA_NAME,
    PARAMETER_BYTES,
    PHASE_BUCKETS,
    PHASE_LOOKUP_ENTRIES,
    SECTIONS,
    SECTIONS_BY_NAME,
    ContainerError,
    build_container,
    read_container,
    structural_sha256,
)
import horde_v3_container as _codec  # noqa: E402

if Path(_codec.__file__).resolve() != (TOOLS / "horde_v3_container.py").resolve():
    raise SystemExit(
        f"horde_v3_container resolved to {_codec.__file__}; expected the codec in {TOOLS}"
    )


REGISTERED_STRUCTURAL_SHA256 = (
    "5FD8D6DE6B375312C1610597260528150A1D49A6C54C4A8FB1B1E8A6A78A854C"
)
REGISTERED_PARAMETER_BYTES = 2_033_765
TRACE_SCHEMA = "HORDE_V3_INTEGER_LAYER_TRACE_V1"
TRACE_PATH = REPO_ROOT / "tests" / "data" / "horde_v3_layer_trace.json"
DEFAULT_RECORDS = Path(r"D:/horde-train/validation-selected/selected-records.bin")

VALUE_TB_WIN_IN_MAX_PLY = 31_507
NETWORK_TO_SCORE = 600
OUTPUT_DIVISOR = 16

FIXTURE_SEED = 987654321
# HordeV3Model initializes for training, not for quantization: its dense
# weights are around 0.012, which at the frozen dense scale of 64 collapses the
# whole trunk onto {-1, 0, 1}. Parity is only meaningful in the regime the
# quantization was designed for, so the fixture is deterministically rescaled
# to the magnitudes a trained network occupies. Nothing else is touched.
FIXTURE_GAINS = {
    "ft_weights": 6.0,
    "psqt_weights": 25.0,
    "hidden0_weights": 20.0,
    "hidden1_weights": 12.0,
    "output_weights": 10.0,
}

# Thresholds. The median absolute difference is expressed against the score
# scale itself, so it stays meaningful if NETWORK_TO_SCORE ever moves.
MAX_MEDIAN_ABSOLUTE_DIFFERENCE = 0.02 * NETWORK_TO_SCORE
MINIMUM_CORRELATION = 0.999
RATIO_TOLERANCE = 0.02
RATIO_FLOOR = 100
MINIMUM_RATIO_SAMPLES = 256

# Eight records chosen once from the default corpus so that all eight phase
# buckets, both sides to move and a damping rule-50 count are all exercised.
TRACE_RECORD_INDICES = (5_819, 36_320, 62_518, 192_065, 125_081, 156_260, 187_501, 218_797)

FAILURES: list[str] = []
REJECTIONS: list[tuple[str, str]] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def expect_error(
    label: str, expected: type[Exception], action: Callable[[], object]
) -> None:
    """``action`` must fail with exactly ``expected``, not succeed and not fail otherwise."""

    try:
        action()
    except expected as error:
        REJECTIONS.append((label, str(error)))
        return
    except Exception as error:  # noqa: BLE001 - the wrong failure mode is a failure
        FAILURES.append(
            f"{label}: raised {type(error).__name__} instead of {expected.__name__}: {error}"
        )
        return
    FAILURES.append(f"{label}: was accepted; it must raise {expected.__name__}")


def expect_container_error(label: str, action: Callable[[], object]) -> None:
    expect_error(label, ContainerError, action)


# --------------------------------------------------------------------------
# Independent reference arithmetic
# --------------------------------------------------------------------------


def trunc_div(values: np.ndarray, divisor: int) -> np.ndarray:
    magnitude = np.abs(values) // divisor
    return np.where(values < 0, -magnitude, magnitude)


def reference_score(raw_value: np.ndarray, rule50: np.ndarray) -> np.ndarray:
    """trunc(v * 600), then the registered rule-50 damping and clamp."""

    value = np.trunc(raw_value * float(NETWORK_TO_SCORE)).astype(np.int64)
    damped = trunc_div(value * (100 - np.clip(rule50, 0, 100)), 100)
    return np.clip(damped, -VALUE_TB_WIN_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1)


def scalar_trunc(value: int, divisor: int) -> int:
    magnitude = abs(value) // divisor
    return -magnitude if value < 0 else magnitude


class ScalarOracle:
    """A pure-Python transcription of the V3 contract.

    No NumPy, no GEMM, no fancy indexing: the point is to re-derive the bucket
    -major addressing and the shift/clip arithmetic by hand so a mistake in the
    vectorized evaluator cannot hide behind the same mistake in its oracle.
    """

    def __init__(self, sections: dict[str, bytes]) -> None:
        def unpack(name: str, code: str) -> tuple[int, ...]:
            spec = SECTIONS_BY_NAME[name]
            return struct.unpack(f"<{spec.elements}{code}", sections[name])

        self.ft_weights = unpack("ft_weights", "h")
        self.ft_bias = unpack("ft_bias", "i")
        self.psqt_weights = unpack("psqt_weights", "i")
        self.hidden0_weights = unpack("hidden0_weights", "b")
        self.hidden0_bias = unpack("hidden0_bias", "i")
        self.hidden1_weights = unpack("hidden1_weights", "b")
        self.hidden1_bias = unpack("hidden1_bias", "i")
        self.output_weights = unpack("output_weights", "b")
        self.output_bias = unpack("output_bias", "i")
        self.phase_lookup = unpack("phase_lookup", "b")

    def evaluate(
        self, rows: Sequence[int], side_to_move: int, white_piece_count: int, rule50: int
    ) -> dict[str, object]:
        lanes = 1024
        accumulator = list(self.ft_bias)
        for row in rows:
            base = row * lanes
            for lane in range(lanes):
                accumulator[lane] += self.ft_weights[base + lane]
        activation = [min(max(value, 0) >> 6, 127) for value in accumulator]

        bucket = self.phase_lookup[white_piece_count]

        hidden0 = []
        for lane in range(16):
            row = bucket * 16 + lane
            base = row * lanes
            total = self.hidden0_bias[row]
            for index in range(lanes):
                total += self.hidden0_weights[base + index] * activation[index]
            hidden0.append(total)
        hidden0_activated = [min(max(value, 0) >> 6, 127) for value in hidden0]

        hidden1 = []
        for lane in range(32):
            row = bucket * 32 + lane
            base = row * 16
            total = self.hidden1_bias[row]
            for index in range(16):
                total += self.hidden1_weights[base + index] * hidden0_activated[index]
            hidden1.append(total)
        hidden1_activated = [min(max(value, 0) >> 6, 127) for value in hidden1]

        head = bucket * 2 + side_to_move
        output_pre = self.output_bias[head]
        for index in range(32):
            output_pre += self.output_weights[head * 32 + index] * hidden1_activated[index]

        psqt_sum = 0
        for row in rows:
            psqt_sum += self.psqt_weights[row * 16 + head]

        value = scalar_trunc(output_pre + psqt_sum, OUTPUT_DIVISOR)
        damped = scalar_trunc(value * (100 - min(rule50, 100)), 100)
        score = max(
            -VALUE_TB_WIN_IN_MAX_PLY + 1, min(VALUE_TB_WIN_IN_MAX_PLY - 1, damped)
        )
        return {
            "bucket": bucket,
            "accumulator": accumulator,
            "activation": activation,
            "hidden0_pre": hidden0,
            "hidden0": hidden0_activated,
            "hidden1_pre": hidden1,
            "hidden1": hidden1_activated,
            "output_pre": output_pre,
            "psqt_sum": psqt_sum,
            "value": value,
            "score": score,
        }


# --------------------------------------------------------------------------
# Fixture construction
# --------------------------------------------------------------------------


def digest_of(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest().upper()


def commit_of(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()[:40]


def training_structural_sha256() -> str:
    """A deterministic stand-in for the hash the V3 trainer will register."""

    structure = {
        "schema": exporter.CHECKPOINT_SCHEMA,
        "feature_schema": NETWORK_SCHEMA_NAME,
        "dense_lanes": [1024, 16, 32, 2],
        "phase_buckets": PHASE_BUCKETS,
        "psqt_columns": 16,
        "rows": 896,
        "serialized_parameter_bytes": exporter.TRAINED_PARAMETER_BYTES,
        "training_activation": "clamp(x, 0, 1)",
        "network_to_score": NETWORK_TO_SCORE,
    }
    payload = json.dumps(structure, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def fixture_model(seed: int = FIXTURE_SEED) -> models.HordeV3Model:
    model = models.HordeV3Model(seed=seed)
    with torch.no_grad():
        for name, gain in FIXTURE_GAINS.items():
            getattr(model, name).mul_(float(gain))
    return model


def fixture_provenance() -> dict[str, object]:
    return {
        "checkpoint_sha256": digest_of("v3-fixture-checkpoint"),
        "container_schema": "HORDE_V2_INTEGER_NETWORK_V1",
        "source_commit": commit_of("v3-fixture-commit"),
        "source_dirty": False,
        "train_file_sha256": digest_of("v3-fixture-train"),
        "training_architecture_structural_sha256": training_structural_sha256(),
        "training_receipt_sha256": digest_of("v3-fixture-receipt"),
        "validation_file_sha256": digest_of("v3-fixture-validation"),
        "wdl_calibration_sha256": digest_of("v3-fixture-wdl"),
    }


@dataclass
class Fixture:
    sections: dict[str, bytes]
    provenance: dict[str, object]
    training_sha256: str
    container: bytes
    receipt: dict[str, object]
    offsets: dict[str, int]


def build_fixture(model: models.HordeV3Model) -> Fixture:
    state = {name: value.detach() for name, value in model.state_dict().items()}
    sections, _ = exporter.quantized_sections(state)
    provenance = fixture_provenance()
    training = str(provenance["training_architecture_structural_sha256"])
    container, receipt = build_container(sections, provenance, training)
    offsets: dict[str, int] = {}
    cursor = 0
    for section in SECTIONS:
        offsets[section.name] = cursor
        cursor += section.byte_length
    return Fixture(sections, provenance, training, container, receipt, offsets)


def synthesize_training_artifacts(
    model: models.HordeV3Model, directory: Path
) -> tuple[Path, Path]:
    """Write a checkpoint and training receipt the V3 exporter will accept."""

    training = training_structural_sha256()
    source = {"commit": commit_of("v3-export-commit"), "dirty": False}
    data = {
        "train_file": {"sha256": digest_of("v3-export-train")},
        "validation_file": {"sha256": digest_of("v3-export-validation")},
    }
    checkpoint = {
        "schema": exporter.CHECKPOINT_SCHEMA,
        "architecture": NETWORK_SCHEMA_NAME,
        "source": source,
        "settings": {
            "architecture": {
                "name": "v3_g1024_pawn_wpc8",
                "schema": NETWORK_SCHEMA_NAME,
                "structural_sha256": training,
            },
            "wdl_calibration_sha256": digest_of("v3-export-wdl"),
        },
        "data": data,
        "model_state": {name: value.detach() for name, value in model.state_dict().items()},
    }
    checkpoint_path = directory / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest().upper()

    receipt = {
        "schema": exporter.TRAINING_RECEIPT_SCHEMA,
        "architecture": {
            "name": "v3_g1024_pawn_wpc8",
            "schema": NETWORK_SCHEMA_NAME,
            "structural_sha256": training,
            "serialized_parameter_bytes": exporter.TRAINED_PARAMETER_BYTES,
        },
        "source": source,
        "data": data,
        "artifacts": {"checkpoint": {"sha256": checkpoint_sha}},
        "claims": {"strength_evidence": False},
    }
    receipt_path = directory / "training-receipt.json"
    receipt_path.write_bytes(
        json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    return checkpoint_path, receipt_path


class FloatView:
    """The batch view HordeV3Model.forward expects."""

    def __init__(self, batch: integer_eval.V3EvaluationBatch) -> None:
        self.v3_global = torch.tensor(np.asarray(batch.v3_rows), dtype=torch.long)
        self.global_offsets = torch.tensor(np.asarray(batch.row_offsets), dtype=torch.long)
        self.side_to_move = torch.tensor(np.asarray(batch.side_to_move), dtype=torch.long)
        self.phase_buckets = models.v3_phase_bucket(
            torch.tensor(np.asarray(batch.white_piece_count), dtype=torch.long)
        )


def batch_for_indices(
    payload: bytes, indices: Sequence[int]
) -> tuple[integer_eval.V3EvaluationBatch, list[dec.TrainingRecord]]:
    stride = wire.RECORD_SIZE
    streams: list[tuple[int, ...]] = []
    sides: list[int] = []
    white: list[int] = []
    rule50: list[int] = []
    records: list[dec.TrainingRecord] = []
    for index in indices:
        raw = payload[index * stride : (index + 1) * stride]
        decoded = wire.validate_record(raw, index)
        streams.append(dec.v3_global_rows(decoded["board"]))
        sides.append(int(decoded["side"]))
        white.append(int(decoded["white_count"]))
        rule50.append(int(decoded["rule50"]))
        records.append(dec.decode_training_record(raw, index))
    batch = integer_eval.make_batch(streams, sides, white, rule50, indices)
    return batch, records


# --------------------------------------------------------------------------
# Byte level mutation helpers
# --------------------------------------------------------------------------


def u64_at(payload: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", payload, offset)[0]


def repair_parameter_digest(payload: bytearray) -> None:
    offset = u64_at(payload, 128)
    length = u64_at(payload, 136)
    digest = sha256_bytes(bytes(payload[offset : offset + length]))
    payload[144:176] = bytes.fromhex(digest)


def read_bytes_as_container(payload: bytes, scratch: Path, name: str) -> object:
    path = scratch / f"{name}.nnue"
    path.write_bytes(payload)
    try:
        return read_container(path)
    finally:
        path.unlink(missing_ok=True)


def non_canonical_provenance_container(container: bytes) -> bytes:
    """Rebuild the container with a spaced provenance payload, digests repaired."""

    header = bytearray(container[:HEADER_BYTES])
    structure_offset = u64_at(header, 32)
    structure_length = u64_at(header, 40)
    provenance_offset = u64_at(header, 80)
    provenance_length = u64_at(header, 88)
    parameter_offset = u64_at(header, 128)
    parameter_length = u64_at(header, 136)

    structure = container[structure_offset : structure_offset + structure_length]
    provenance = json.loads(
        container[provenance_offset : provenance_offset + provenance_length].decode("ascii")
    )
    parameters = container[parameter_offset : parameter_offset + parameter_length]
    # Valid ASCII JSON with the same content but spaced separators, so only the
    # canonical-form check can reject it.
    payload = json.dumps(provenance, sort_keys=True, separators=(", ", ": ")).encode("ascii")

    struct.pack_into("<Q", header, 88, len(payload))
    header[96:128] = bytes.fromhex(sha256_bytes(payload))
    struct.pack_into("<Q", header, 128, provenance_offset + len(payload))
    total = provenance_offset + len(payload) + parameter_length
    struct.pack_into("<Q", header, 24, total)
    return bytes(header) + structure + payload + parameters


def patched_sections(fixture: Fixture, name: str, mutate) -> dict[str, bytes]:
    sections = dict(fixture.sections)
    buffer = bytearray(sections[name])
    mutate(buffer)
    sections[name] = bytes(buffer)
    return sections


# --------------------------------------------------------------------------
# (a) Structural contract and round trip
# --------------------------------------------------------------------------


def test_structural_contract() -> None:
    print("a) structural contract")
    actual = structural_sha256()
    check(
        actual == REGISTERED_STRUCTURAL_SHA256,
        f"descriptor structural hash is {actual}, not the registered constant",
    )
    check(
        PARAMETER_BYTES == REGISTERED_PARAMETER_BYTES,
        f"parameter_bytes is {PARAMETER_BYTES}, not {REGISTERED_PARAMETER_BYTES}",
    )
    check(len(SECTIONS) == 10, f"the V3 schema declares {len(SECTIONS)} sections, not 10")
    check(
        exporter.TRAINED_PARAMETER_BYTES + PHASE_LOOKUP_ENTRIES == PARAMETER_BYTES,
        "trained parameter bytes plus the phase lookup do not tile the payload",
    )
    expected_shapes = {
        "ft_weights": ((896, 1024), "i16"),
        "ft_bias": ((1024,), "i32"),
        "psqt_weights": ((896, 16), "i32"),
        "hidden0_weights": ((128, 1024), "i8"),
        "hidden0_bias": ((128,), "i32"),
        "hidden1_weights": ((256, 16), "i8"),
        "hidden1_bias": ((256,), "i32"),
        "output_weights": ((16, 32), "i8"),
        "output_bias": ((16,), "i32"),
        "phase_lookup": ((37,), "i8"),
    }
    actual_shapes = {s.name: (s.shape, s.dtype) for s in SECTIONS}
    check(actual_shapes == expected_shapes, f"V3 section layout drifted: {actual_shapes}")
    print(f"  structural_sha256      : {actual}")
    print(f"  parameter_bytes        : {PARAMETER_BYTES}")


def test_round_trip(fixture: Fixture, scratch: Path) -> None:
    print("a) build then read round trip")
    path = scratch / "round-trip.nnue"
    path.write_bytes(fixture.container)
    parsed = read_container(path)

    identical = all(
        parsed.sections[section.name] == fixture.sections[section.name] for section in SECTIONS
    )
    check(identical, "a section did not survive the round trip byte identical")
    check(
        sum(len(blob) for blob in parsed.sections.values()) == REGISTERED_PARAMETER_BYTES,
        "round-tripped parameter byte count drifted",
    )
    check(
        parsed.parameter_sha256 == fixture.receipt["parameter_sha256"],
        "parameter SHA-256 disagrees with the build receipt",
    )
    check(
        parsed.file_sha256 == fixture.receipt["file_sha256"],
        "file SHA-256 disagrees with the build receipt",
    )
    check(
        parsed.file_sha256 == sha256_bytes(path.read_bytes()),
        "file SHA-256 does not describe the bytes on disk",
    )
    check(
        parsed.training_structural_sha256 == fixture.training_sha256,
        "training structural hash did not survive the round trip",
    )
    check(parsed.provenance == fixture.provenance, "provenance did not survive the round trip")
    check(
        fixture.receipt["container_structural_sha256"] == REGISTERED_STRUCTURAL_SHA256,
        "build receipt structural hash is not the registered constant",
    )
    check(
        fixture.receipt["network_schema_id"] == NETWORK_SCHEMA_ID,
        "build receipt network schema id drifted",
    )
    print(f"  file_bytes             : {len(fixture.container)}")
    print(f"  parameter_sha256       : {parsed.parameter_sha256}")
    path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# (b) Adversarial rejections
# --------------------------------------------------------------------------


def test_framing_rejections(fixture: Fixture, scratch: Path) -> None:
    print("b) adversarial rejections: framing and authentication")
    good = fixture.container

    def mutated(name: str, mutate) -> Callable[[], object]:
        def action() -> object:
            payload = bytearray(good)
            mutate(payload)
            return read_bytes_as_container(bytes(payload), scratch, name)

        return action

    expect_container_error(
        "wrong magic", mutated("magic", lambda p: p.__setitem__(slice(0, 8), b"HSV3INT\0"))
    )
    expect_container_error(
        "wrong format version",
        mutated("version", lambda p: struct.pack_into("<H", p, 8, 2)),
    )
    expect_container_error(
        "wrong endian tag",
        mutated("endian", lambda p: struct.pack_into("<I", p, 12, 0x04030201)),
    )
    expect_container_error(
        "unknown network schema id",
        mutated("schema", lambda p: struct.pack_into("<I", p, 16, 0x00020002)),
    )
    expect_container_error(
        "truncated file",
        lambda: read_bytes_as_container(good[:-1], scratch, "truncated"),
    )
    expect_container_error(
        "truncated below the header",
        lambda: read_bytes_as_container(good[:100], scratch, "stub"),
    )
    expect_container_error(
        "file longer than the header claims",
        lambda: read_bytes_as_container(good + b"\x00", scratch, "long"),
    )
    expect_container_error(
        "corrupted structure bytes",
        mutated("structure", lambda p: p.__setitem__(HEADER_BYTES + 16, p[HEADER_BYTES + 16] ^ 0x01)),
    )
    provenance_offset = u64_at(good, 80)
    expect_container_error(
        "corrupted provenance bytes",
        mutated(
            "provenance",
            lambda p: p.__setitem__(provenance_offset + 8, p[provenance_offset + 8] ^ 0x01),
        ),
    )
    expect_container_error(
        "non-canonical provenance JSON",
        lambda: read_bytes_as_container(
            non_canonical_provenance_container(good), scratch, "noncanonical"
        ),
    )


def test_section_payload_rejections(fixture: Fixture, scratch: Path) -> None:
    print("b) adversarial rejections: per-section authentication")
    good = fixture.container
    parameter_offset = u64_at(good, 128)

    for index, section in enumerate(SECTIONS):
        start = parameter_offset + fixture.offsets[section.name]

        def flip(payload: bytearray, position: int = start) -> None:
            payload[position] ^= 0x01
            # Repair the aggregate digest so the per-section directory digest,
            # not the payload-wide one, is the gate under test.
            repair_parameter_digest(payload)

        def action(mutate=flip, name: str = section.name) -> object:
            payload = bytearray(good)
            mutate(payload)
            return read_bytes_as_container(bytes(payload), scratch, f"flip-{name}")

        expect_container_error(f"flipped bit in section {section.name}", action)

        entry = DIRECTORY_OFFSET + index * DIRECTORY_ENTRY_BYTES + 28

        def digest_action(position: int = entry, name: str = section.name) -> object:
            payload = bytearray(good)
            payload[position] ^= 0x01
            return read_bytes_as_container(bytes(payload), scratch, f"digest-{name}")

        expect_container_error(
            f"corrupted directory digest for section {section.name}", digest_action
        )


def test_provenance_rejections(fixture: Fixture) -> None:
    print("b) adversarial rejections: provenance and identity")

    def build_with(provenance: dict[str, object], training: str | None = None):
        return lambda: build_container(
            fixture.sections, provenance, training or fixture.training_sha256
        )

    dirty = dict(fixture.provenance)
    dirty["source_dirty"] = True
    expect_container_error("dirty source flag", build_with(dirty))

    for label, commit in (
        ("zero source commit", "0" * 40),
        ("short source commit", "abc123"),
        ("non-hex source commit", "z" * 40),
    ):
        bad = dict(fixture.provenance)
        bad["source_commit"] = commit
        expect_container_error(label, build_with(bad))

    expect_container_error(
        "mismatched training structural hash",
        build_with(dict(fixture.provenance), digest_of("some-other-architecture")),
    )

    missing = dict(fixture.provenance)
    missing.pop("wdl_calibration_sha256")
    expect_container_error("provenance field removed", build_with(missing))

    extra = dict(fixture.provenance)
    extra["unexpected"] = "value"
    expect_container_error("provenance field added", build_with(extra))

    lowercase = dict(fixture.provenance)
    lowercase["train_file_sha256"] = str(lowercase["train_file_sha256"]).lower()
    expect_container_error("lowercase provenance digest", build_with(lowercase))

    zeroed = dict(fixture.provenance)
    zeroed["checkpoint_sha256"] = "0" * 64
    expect_container_error("zero provenance digest", build_with(zeroed))


def test_bound_rejections(fixture: Fixture) -> None:
    print("b) adversarial rejections: registered magnitude bounds")

    def build_with(sections: dict[str, bytes]):
        return lambda: build_container(
            sections, fixture.provenance, fixture.training_sha256
        )

    expect_container_error(
        "psqt weight above 2**20",
        build_with(
            patched_sections(
                fixture,
                "psqt_weights",
                lambda buffer: struct.pack_into("<i", buffer, 0, MAX_PSQT_MAGNITUDE + 1),
            )
        ),
    )
    expect_container_error(
        "psqt weight below -2**20",
        build_with(
            patched_sections(
                fixture,
                "psqt_weights",
                lambda buffer: struct.pack_into("<i", buffer, 4, -MAX_PSQT_MAGNITUDE - 1),
            )
        ),
    )
    for name in ("ft_bias", "hidden0_bias", "hidden1_bias", "output_bias"):
        expect_container_error(
            f"{name} above 2**30",
            build_with(
                patched_sections(
                    fixture,
                    name,
                    lambda buffer: struct.pack_into(
                        "<i", buffer, 0, MAX_SAFE_BIAS_MAGNITUDE + 1
                    ),
                )
            ),
        )

    lookup = list(struct.unpack(f"<{PHASE_LOOKUP_ENTRIES}b", fixture.sections["phase_lookup"]))

    def lookup_section(values: list[int]) -> dict[str, bytes]:
        sections = dict(fixture.sections)
        sections["phase_lookup"] = struct.pack(f"<{len(values)}b", *values)
        return sections

    non_monotone = list(lookup)
    non_monotone[0] = 1
    check(
        non_monotone[0] > non_monotone[1] and sorted(set(non_monotone)) == list(range(8)),
        "the non-monotone phase lookup fixture is not actually non-monotone",
    )
    expect_container_error("non-monotone phase lookup", build_with(lookup_section(non_monotone)))

    out_of_range = list(lookup)
    out_of_range[-1] = PHASE_BUCKETS
    expect_container_error("out-of-range phase lookup", build_with(lookup_section(out_of_range)))

    negative = list(lookup)
    negative[0] = -1
    expect_container_error("negative phase lookup", build_with(lookup_section(negative)))

    incomplete = [min(value, PHASE_BUCKETS - 2) for value in lookup]
    check(
        sorted(set(incomplete)) == list(range(PHASE_BUCKETS - 1)),
        "the incomplete phase lookup fixture still reaches every bucket",
    )
    expect_container_error(
        "phase lookup does not reach all 8 buckets", build_with(lookup_section(incomplete))
    )

    short = lookup[:-1]
    expect_container_error("short phase lookup", build_with(lookup_section(short)))

    truncated = dict(fixture.sections)
    truncated["ft_bias"] = truncated["ft_bias"][:-4]
    expect_container_error("short section payload", build_with(truncated))

    renamed = dict(fixture.sections)
    renamed["unexpected_section"] = renamed.pop("output_bias")
    expect_container_error("section name set mismatch", build_with(renamed))


# --------------------------------------------------------------------------
# (c) Quantization parity
# --------------------------------------------------------------------------


def test_quantization_parity(
    model: models.HordeV3Model, records: Path, count: int, scratch: Path
) -> tuple[integer_eval.IntegerNetworkV3, Path]:
    print("c) quantization parity against the float model")
    work = scratch / "export"
    work.mkdir(parents=True, exist_ok=True)
    checkpoint_path, receipt_path = synthesize_training_artifacts(model, work)
    network_path = work / "horde-v3-fixture.nnue"
    export_receipt_path = work / "horde-v3-fixture-export.json"
    export_receipt = exporter.export_checkpoint(
        checkpoint_path, receipt_path, network_path, export_receipt_path
    )
    check(
        export_receipt["container"]["parameter_bytes"] == REGISTERED_PARAMETER_BYTES,
        "export receipt parameter byte count drifted",
    )
    check(
        export_receipt["container_structural_sha256"] == REGISTERED_STRUCTURAL_SHA256,
        "export receipt structural hash is not the registered constant",
    )
    expect_error(
        "exporter refuses to overwrite an existing network",
        exporter.ExportError,
        lambda: exporter.export_checkpoint(
            checkpoint_path, receipt_path, network_path, work / "second-receipt.json"
        ),
    )
    expect_error(
        "exporter refuses a checkpoint whose parameter names drifted",
        exporter.ExportError,
        lambda: exporter.quantized_sections({"ft_weights": torch.zeros(896, 1024)}),
    )

    network = integer_eval.IntegerNetworkV3.load(network_path)
    health = network.parameter_health()
    check(bool(health["passed"]), f"parameter health failed: {health}")

    batch = integer_eval.batch_from_file(records, count)
    layers = network.evaluate_layers(batch)
    integer_scores = layers["score"].astype(np.int64)

    with torch.no_grad():
        raw = model(FloatView(batch)).numpy().astype(np.float64)
    float_scores = reference_score(raw, np.asarray(batch.rule50_count, dtype=np.int64))

    check(
        int(np.unique(layers["bucket"]).size) == PHASE_BUCKETS,
        f"the parity batch only reached buckets {sorted(set(layers['bucket'].tolist()))}",
    )
    check(
        len(batch) >= 512, f"the parity batch has {len(batch)} records; at least 512 are required"
    )

    difference = integer_scores - float_scores
    median_difference = float(np.median(difference))
    median_absolute = float(np.median(np.abs(difference)))
    p95_absolute = float(np.percentile(np.abs(difference), 95))
    maximum_absolute = int(np.max(np.abs(difference)))
    correlation = float(np.corrcoef(integer_scores.astype(float), float_scores.astype(float))[0, 1])

    eligible = np.abs(float_scores) > RATIO_FLOOR
    sample_count = int(np.count_nonzero(eligible))
    ratio = integer_scores[eligible] / float_scores[eligible]
    median_ratio = float(np.median(ratio)) if sample_count else float("nan")

    print(f"  records evaluated      : {len(batch)}")
    print(
        f"  float value range      : [{int(float_scores.min())}, {int(float_scores.max())}]"
    )
    print(
        f"  integer value range    : [{int(integer_scores.min())}, {int(integer_scores.max())}]"
    )
    print(f"  difference median      : {median_difference:+.1f}")
    print(f"  difference median |.|  : {median_absolute:.1f}")
    print(f"  difference p95 |.|     : {p95_absolute:.1f}")
    print(f"  difference max |.|     : {maximum_absolute}")
    print(f"  exact agreement        : {float(np.mean(difference == 0)) * 100:.1f}%")
    print(f"  correlation            : {correlation:.6f}")
    print(f"  ratio samples (|f|>{RATIO_FLOOR}) : {sample_count}")
    print(f"  median integer/float   : {median_ratio:.4f}")

    check(
        median_absolute <= MAX_MEDIAN_ABSOLUTE_DIFFERENCE,
        f"median absolute difference {median_absolute:.1f} exceeds "
        f"{MAX_MEDIAN_ABSOLUTE_DIFFERENCE:.1f}",
    )
    check(
        correlation >= MINIMUM_CORRELATION,
        f"correlation {correlation:.6f} is below {MINIMUM_CORRELATION}",
    )
    check(
        sample_count >= MINIMUM_RATIO_SAMPLES,
        f"only {sample_count} records exceed |{RATIO_FLOOR}|; the scale check is not meaningful",
    )
    check(
        abs(median_ratio - 1.0) <= RATIO_TOLERANCE,
        f"median integer/float ratio {median_ratio:.4f} is not within "
        f"{RATIO_TOLERANCE} of 1.000; the output layer carries a systematic scale factor",
    )
    return network, network_path


# --------------------------------------------------------------------------
# (d) Layer trace fixture
# --------------------------------------------------------------------------


def test_layer_trace(
    network: integer_eval.IntegerNetworkV3, records: Path, payload: bytes
) -> None:
    print("d) layer trace fixture")
    available = len(payload) // wire.RECORD_SIZE
    indices = [index for index in TRACE_RECORD_INDICES if index < available]
    check(
        len(indices) == len(TRACE_RECORD_INDICES),
        f"the record corpus holds {available} records; the trace needs {TRACE_RECORD_INDICES}",
    )
    check(len(indices) == 8, f"the trace fixture covers {len(indices)} records, not 8")

    batch, decoded = batch_for_indices(payload, indices)
    layers = network.evaluate_layers(batch)

    # Re-derive every traced position by hand before it is written down.
    oracle = ScalarOracle(dict(network.container.sections))
    for position, index in enumerate(indices):
        expected = oracle.evaluate(
            batch.active_rows(position),
            int(batch.side_to_move[position]),
            int(batch.white_piece_count[position]),
            int(batch.rule50_count[position]),
        )
        for key in (
            "bucket",
            "output_pre",
            "psqt_sum",
            "value",
            "score",
        ):
            actual = int(layers[key][position])
            check(
                actual == int(expected[key]),
                f"record {index}: vectorized {key} is {actual}, scalar oracle says "
                f"{int(expected[key])}",
            )
        for key in ("accumulator", "activation", "hidden0_pre", "hidden0", "hidden1_pre", "hidden1"):
            actual_lanes = [int(value) for value in layers[key][position]]
            check(
                actual_lanes == list(expected[key]),
                f"record {index}: vectorized {key} disagrees with the scalar oracle",
            )

    positions = []
    for position, index in enumerate(indices):
        raw = payload[index * wire.RECORD_SIZE : (index + 1) * wire.RECORD_SIZE]
        rows = sorted(batch.active_rows(position))
        positions.append(
            {
                "record_index": index,
                "record_bytes_hex": raw.hex().upper(),
                "fen": dec.training_record_fen(decoded[position]),
                "side_to_move": int(batch.side_to_move[position]),
                "white_piece_count": int(batch.white_piece_count[position]),
                "phase_bucket": int(layers["bucket"][position]),
                "rule50": int(batch.rule50_count[position]),
                "active_row_count": len(rows),
                "active_rows": rows,
                "accumulator_first16": [
                    int(value) for value in layers["accumulator"][position][:16]
                ],
                "activation_first16": [
                    int(value) for value in layers["activation"][position][:16]
                ],
                "hidden0_pre": [int(value) for value in layers["hidden0_pre"][position]],
                "hidden0": [int(value) for value in layers["hidden0"][position]],
                "hidden1_pre": [int(value) for value in layers["hidden1_pre"][position]],
                "hidden1": [int(value) for value in layers["hidden1"][position]],
                "output_pre": int(layers["output_pre"][position]),
                "psqt_sum": int(layers["psqt_sum"][position]),
                "value": int(layers["value"][position]),
                "score": int(layers["score"][position]),
            }
        )

    for entry in positions:
        expected = trunc_div(
            np.asarray([entry["output_pre"] + entry["psqt_sum"]], dtype=np.int64),
            OUTPUT_DIVISOR,
        )[0]
        check(
            int(expected) == entry["value"],
            f"record {entry['record_index']}: value is not trunc((out_pre + psqt_sum) / 16)",
        )
        check(
            all(0 <= lane <= 127 for lane in entry["hidden0"] + entry["hidden1"]),
            f"record {entry['record_index']}: an activation escaped [0, 127]",
        )
        check(
            len(entry["active_rows"]) == len(set(entry["active_rows"])),
            f"record {entry['record_index']}: duplicate active row",
        )

    trace = {
        "schema": TRACE_SCHEMA,
        "network_schema": NETWORK_SCHEMA_NAME,
        "network_schema_id": NETWORK_SCHEMA_ID,
        "container_structural_sha256": structural_sha256(),
        "parameter_sha256": network.container.parameter_sha256,
        "file_sha256": network.container.file_sha256,
        "binding": (
            "parameter_sha256 is reproducible from fixture.model_seed and fixture.gains; "
            "file_sha256 additionally covers the synthetic provenance of this fixture"
        ),
        "fixture": {
            "model_seed": FIXTURE_SEED,
            "gains": FIXTURE_GAINS,
            "note": "HordeV3Model(seed) with each named parameter multiplied by its gain",
        },
        "record_source": {
            "path": str(records),
            "record_bytes": wire.RECORD_SIZE,
            "record_indices": list(indices),
        },
        "arithmetic": {
            "ft_activation": "clip(max(acc, 0) >> 6, 0, 127)",
            "dense_activation": "clip(max(pre, 0) >> 6, 0, 127)",
            "value": "trunc_toward_zero(out_pre + psqt_sum, 16)",
            "rule50": "trunc_toward_zero(value * (100 - min(rule50, 100)), 100)",
            "clamp": [-VALUE_TB_WIN_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1],
        },
        "positions": positions,
    }
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_bytes(
        json.dumps(trace, indent=2, sort_keys=True, allow_nan=False).encode("ascii") + b"\n"
    )
    buckets = sorted({entry["phase_bucket"] for entry in positions})
    sides = sorted({entry["side_to_move"] for entry in positions})
    damped = sum(1 for entry in positions if entry["rule50"] > 0)
    check(
        buckets == list(range(PHASE_BUCKETS)),
        f"the trace fixture only reaches buckets {buckets}; it must cover all "
        f"{PHASE_BUCKETS}. Re-select TRACE_RECORD_INDICES for this corpus.",
    )
    check(sides == [0, 1], f"the trace fixture only reaches sides {sides}")
    check(damped > 0, "no traced record exercises the rule-50 damping")
    print(f"  trace written          : {TRACE_PATH}")
    print(f"  trace bytes            : {TRACE_PATH.stat().st_size}")
    print(f"  scalar oracle          : agrees on all layers for all 8 records")
    print(f"  buckets covered        : {buckets}")
    print(f"  sides covered          : {sides}")
    print(f"  rule50-damped records  : {damped}")
    print(f"  values                 : {[entry['value'] for entry in positions]}")


# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--count", type=int, default=4096)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if not args.records.is_file():
        print(f"ERROR: record corpus does not exist: {args.records}", file=sys.stderr)
        return 1

    print("V3 integer container checks\n")
    model = fixture_model()
    fixture = build_fixture(model)
    with tempfile.TemporaryDirectory(prefix="horde-v3-container-") as directory:
        scratch = Path(directory)
        test_structural_contract()
        print()
        test_round_trip(fixture, scratch)
        print()
        test_framing_rejections(fixture, scratch)
        test_section_payload_rejections(fixture, scratch)
        test_provenance_rejections(fixture)
        test_bound_rejections(fixture)
        print(f"  attacks rejected       : {len(REJECTIONS)}")
        for label, reason in REJECTIONS:
            print(f"    {label:<52s} -> {reason}")
        print()
        network, _ = test_quantization_parity(model, args.records, args.count, scratch)
        print()
        test_layer_trace(network, args.records, args.records.read_bytes())

    if FAILURES:
        print(f"\nFAILED with {len(FAILURES)} problems:")
        for failure in FAILURES[:40]:
            print(f"  {failure}")
        return 1
    print("\nall V3 integer container checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
