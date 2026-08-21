#!/usr/bin/env python3
"""Export an authenticated Horde V3 training checkpoint to integer format.

The V3 container contract is frozen in ``horde_v3_container``; this exporter
never restates a bound or a scale, it imports every one of them.  Three things
differ from the V2 exporter and each is deliberate.

1. The bucketed trunk tensors are stored flattened *bucket-major*.  The trainer
   holds ``(8, 16, 1024)``; the container holds ``(128, 1024)`` with row
   ``bucket * 16 + lane``.  A contiguous ``reshape`` is exactly that mapping.
2. The output layer is quantized at the scale the trainer optimizes, so
   ``output_weights`` uses the rational scale ``9600/127`` and ``output_bias``
   uses ``9600``.  The container value is then exactly ``600 * v``.  The V2
   exporter used the dense scale here and evaluates at ``508 * v``; that defect
   is not reproduced.
3. ``phase_lookup`` is not a trained parameter.  It is the frozen
   ``V3_PHASE_LOOKUP`` tuple serialized as int8 and re-validated on the way out.

Quantization rounds to nearest with ties to even.  The product is taken in
float64 rather than the float32 the V2 exporter used, because ``9600/127`` is
not representable and a float32 product would decide ties on rounding noise.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import pickle
import struct
import sys
from typing import Mapping, Sequence

try:
    import torch
    from torch import Tensor
except ImportError as error:  # pragma: no cover - CLI dependency failure
    raise SystemExit("PyTorch is required for Horde V3 checkpoint export") from error

try:
    from .horde_training_models import V3_PHASE_LOOKUP
    from .horde_v2_container import sha256_bytes, sha256_file, write_container_exclusive
    from .horde_v3_container import (
        CONTAINER_SCHEMA,
        DENSE_SCALE,
        FT_LANES,
        FT_ROWS,
        FT_SCALE,
        HIDDEN0_LANES,
        HIDDEN1_LANES,
        HIDDEN_BIAS_SCALE,
        MAX_PSQT_MAGNITUDE,
        MAX_SAFE_BIAS_MAGNITUDE,
        NETWORK_SCHEMA_ID,
        NETWORK_SCHEMA_NAME,
        OUTPUT_BIAS_SCALE,
        OUTPUT_HEADS,
        OUTPUT_WEIGHT_SCALE_DENOMINATOR,
        OUTPUT_WEIGHT_SCALE_NUMERATOR,
        PARAMETER_BYTES,
        PHASE_BUCKETS,
        PHASE_LOOKUP_ENTRIES,
        PSQT_COLUMNS,
        PSQT_SCALE,
        SECTIONS_BY_NAME,
        SHA256_RE,
        ContainerError,
        build_container,
        structural_sha256,
    )
except ImportError:
    from horde_training_models import V3_PHASE_LOOKUP
    from horde_v2_container import sha256_bytes, sha256_file, write_container_exclusive
    from horde_v3_container import (
        CONTAINER_SCHEMA,
        DENSE_SCALE,
        FT_LANES,
        FT_ROWS,
        FT_SCALE,
        HIDDEN0_LANES,
        HIDDEN1_LANES,
        HIDDEN_BIAS_SCALE,
        MAX_PSQT_MAGNITUDE,
        MAX_SAFE_BIAS_MAGNITUDE,
        NETWORK_SCHEMA_ID,
        NETWORK_SCHEMA_NAME,
        OUTPUT_BIAS_SCALE,
        OUTPUT_HEADS,
        OUTPUT_WEIGHT_SCALE_DENOMINATOR,
        OUTPUT_WEIGHT_SCALE_NUMERATOR,
        PARAMETER_BYTES,
        PHASE_BUCKETS,
        PHASE_LOOKUP_ENTRIES,
        PSQT_COLUMNS,
        PSQT_SCALE,
        SECTIONS_BY_NAME,
        SHA256_RE,
        ContainerError,
        build_container,
        structural_sha256,
    )


# The trainer writes HORDE_V3_CHECKPOINT_V1. V2 carried "BASE" because it had a
# V2_BASE_P0 topology to distinguish; V3 has no base variant, so the trainer's
# name is the correct one and the exporter follows the artifact, never the
# reverse: a produced checkpoint is evidence and is not renamed to suit a reader.
CHECKPOINT_SCHEMA = "HORDE_V3_CHECKPOINT_V1"
TRAINING_RECEIPT_SCHEMA = "HORDE_V3_TRAINING_V1"  # matches the trainer, see above
EXPORT_RECEIPT_SCHEMA = "HORDE_V3_INTEGER_CHECKPOINT_EXPORT_V1"

OUTPUT_WEIGHT_SCALE = Fraction(
    OUTPUT_WEIGHT_SCALE_NUMERATOR, OUTPUT_WEIGHT_SCALE_DENOMINATOR
)

# ``phase_lookup`` is a frozen constant rather than a trained tensor, so the
# training side declares only the nine learned sections.
# The training architecture receipt declares the FULL serialized payload,
# 2,033,765 bytes, phase lookup included. The lookup is not trained, but it is
# serialized, and "serialized_parameter_bytes" is a statement about the
# serialized form. This value is baked into the pinned structural hash
# 5B49DC20 that the contract, the trainer and every completed run already
# carry, so the exporter follows it rather than subtracting the lookup and
# forcing a hash change that would invalidate finished work.
TRAINED_PARAMETER_BYTES = PARAMETER_BYTES

DTYPE_LIMITS = {
    "i8": (-(1 << 7), (1 << 7) - 1),
    "i16": (-(1 << 15), (1 << 15) - 1),
    "i32": (-(1 << 31), (1 << 31) - 1),
}
NUMPY_DTYPES = {"i8": "<i1", "i16": "<i2", "i32": "<i4"}


class ExportError(ValueError):
    """Raised when a checkpoint cannot be exported without ambiguity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


@dataclass(frozen=True, slots=True)
class SectionSource:
    """One container section and the trainer tensor it is derived from."""

    name: str
    state_key: str
    state_shape: tuple[int, ...]
    scale: int | Fraction

    @property
    def container_shape(self) -> tuple[int, ...]:
        return SECTIONS_BY_NAME[self.name].shape

    @property
    def dtype(self) -> str:
        return SECTIONS_BY_NAME[self.name].dtype


SECTION_SOURCES: tuple[SectionSource, ...] = (
    SectionSource("ft_weights", "ft_weights", (FT_ROWS, FT_LANES), FT_SCALE),
    SectionSource("ft_bias", "ft_bias", (FT_LANES,), FT_SCALE),
    SectionSource("psqt_weights", "psqt_weights", (FT_ROWS, PSQT_COLUMNS), PSQT_SCALE),
    SectionSource(
        "hidden0_weights",
        "hidden0_weights",
        (PHASE_BUCKETS, HIDDEN0_LANES, FT_LANES),
        DENSE_SCALE,
    ),
    SectionSource(
        "hidden0_bias", "hidden0_bias", (PHASE_BUCKETS, HIDDEN0_LANES), HIDDEN_BIAS_SCALE
    ),
    SectionSource(
        "hidden1_weights",
        "hidden1_weights",
        (PHASE_BUCKETS, HIDDEN1_LANES, HIDDEN0_LANES),
        DENSE_SCALE,
    ),
    SectionSource(
        "hidden1_bias", "hidden1_bias", (PHASE_BUCKETS, HIDDEN1_LANES), HIDDEN_BIAS_SCALE
    ),
    SectionSource(
        "output_weights",
        "output_weights",
        (PHASE_BUCKETS, OUTPUT_HEADS, HIDDEN1_LANES),
        OUTPUT_WEIGHT_SCALE,
    ),
    SectionSource(
        "output_bias", "output_bias", (PHASE_BUCKETS, OUTPUT_HEADS), OUTPUT_BIAS_SCALE
    ),
)
TRAINED_STATE_KEYS = frozenset(source.state_key for source in SECTION_SOURCES)


def _scale_terms(scale: int | Fraction) -> tuple[float, str, int, int]:
    if isinstance(scale, Fraction):
        numerator, denominator = scale.numerator, scale.denominator
    else:
        numerator, denominator = int(scale), 1
    _require(numerator > 0 and denominator > 0, "quantization scale must be positive")
    return numerator / denominator, f"{numerator}/{denominator}", numerator, denominator


def _quantize(
    tensor: Tensor, scale: int | Fraction, dtype: str, name: str
) -> tuple[bytes, dict[str, object]]:
    """Round to nearest with ties to even and enforce every registered bound."""

    _require(tensor.device.type == "cpu", f"{name} is not on CPU")
    _require(tensor.dtype == torch.float32, f"{name} is not float32")
    _require(bool(torch.isfinite(tensor).all()), f"{name} contains a non-finite value")
    factor, scale_text, numerator, denominator = _scale_terms(scale)
    exact = tensor.detach().to(torch.float64)
    rounded = torch.round(exact * factor).to(torch.int64)

    minimum, maximum = DTYPE_LIMITS[dtype]
    actual_minimum = int(rounded.min().item())
    actual_maximum = int(rounded.max().item())
    _require(
        actual_minimum >= minimum and actual_maximum <= maximum,
        f"{name} overflows {dtype}: [{actual_minimum}, {actual_maximum}]",
    )
    worst = max(abs(actual_minimum), abs(actual_maximum))
    if name.endswith("_bias"):
        _require(
            worst <= MAX_SAFE_BIAS_MAGNITUDE,
            f"{name} exceeds the registered safe bias magnitude",
        )
    if name == "psqt_weights":
        _require(
            worst <= MAX_PSQT_MAGNITUDE,
            f"{name} magnitude {worst} exceeds the registered PSQT bound",
        )

    payload = rounded.numpy().astype(NUMPY_DTYPES[dtype], copy=False).tobytes(order="C")
    statistics = {
        "dtype": dtype,
        "elements": int(rounded.numel()),
        "float_max": float(tensor.max().item()),
        "float_min": float(tensor.min().item()),
        "integer_max": actual_maximum,
        "integer_min": actual_minimum,
        "scale": scale_text,
        "scale_denominator": denominator,
        "scale_numerator": numerator,
    }
    return payload, statistics


def _phase_lookup_section() -> tuple[bytes, dict[str, object]]:
    """Serialize the frozen phase lookup and re-check it before it is sealed."""

    values = [int(value) for value in V3_PHASE_LOOKUP]
    _require(
        len(values) == PHASE_LOOKUP_ENTRIES,
        f"phase lookup has {len(values)} entries instead of {PHASE_LOOKUP_ENTRIES}",
    )
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
    payload = struct.pack(f"<{len(values)}b", *values)
    statistics = {
        "dtype": "i8",
        "elements": len(values),
        "exact_constant": True,
        "float_max": float(max(values)),
        "float_min": float(min(values)),
        "integer_max": max(values),
        "integer_min": min(values),
        "scale": "1/1",
        "scale_denominator": 1,
        "scale_numerator": 1,
        "source": "horde_training_models.V3_PHASE_LOOKUP",
    }
    return payload, statistics


def quantized_sections(
    model_state: Mapping[str, object],
) -> tuple[dict[str, bytes], dict[str, dict[str, object]]]:
    """Flatten bucket-major, quantize and bound-check every V3 section."""

    _require(
        set(model_state) == set(TRAINED_STATE_KEYS),
        "checkpoint parameter names do not match the V3 architecture",
    )
    sections: dict[str, bytes] = {}
    statistics: dict[str, dict[str, object]] = {}
    for source in SECTION_SOURCES:
        value = model_state[source.state_key]
        _require(isinstance(value, Tensor), f"parameter {source.state_key} is not a tensor")
        _require(
            tuple(value.shape) == source.state_shape,
            f"parameter {source.state_key} has shape {tuple(value.shape)} "
            f"instead of {source.state_shape}",
        )
        # Bucket-major flattening: a contiguous reshape maps (bucket, lane, k)
        # onto row ``bucket * lanes + lane`` exactly.
        flattened = value.detach().cpu().contiguous().reshape(source.container_shape)
        payload, stats = _quantize(flattened, source.scale, source.dtype, source.name)
        expected = SECTIONS_BY_NAME[source.name].byte_length
        _require(
            len(payload) == expected,
            f"quantized section {source.name} is {len(payload)} bytes instead of {expected}",
        )
        sections[source.name] = payload
        statistics[source.name] = {**stats, "source_shape": list(source.state_shape)}

    payload, stats = _phase_lookup_section()
    _require(
        len(payload) == SECTIONS_BY_NAME["phase_lookup"].byte_length,
        "phase lookup section byte count mismatch",
    )
    sections["phase_lookup"] = payload
    statistics["phase_lookup"] = stats

    _require(
        sum(len(blob) for blob in sections.values()) == PARAMETER_BYTES,
        "quantized parameter byte count drifted from the contract",
    )
    return sections, statistics


def _read_json(path: Path) -> tuple[dict[str, object], bytes]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"training receipt does not exist: {resolved}")
    payload = resolved.read_bytes()
    try:
        root = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExportError(f"cannot decode training receipt: {error}") from error
    _require(isinstance(root, dict), "training receipt root is not an object")
    return root, payload


def _load_checkpoint(path: Path) -> tuple[dict[str, object], str]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"checkpoint does not exist: {resolved}")
    digest = sha256_file(resolved)
    try:
        # V3 training checkpoints carry optimizer, scheduler and RNG state, so
        # the restricted unpickler cannot load them.
        root = torch.load(resolved, map_location="cpu", weights_only=False)
    except (EOFError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise ExportError(f"cannot load checkpoint: {error}") from error
    _require(isinstance(root, dict), "checkpoint root is not an object")
    return root, digest


def _digest(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
        f"{label} is not an uppercase SHA-256",
    )
    _require(value != "0" * 64, f"{label} must be nonzero")
    return str(value)


def _validate_identities(
    checkpoint: Mapping[str, object],
    checkpoint_sha256: str,
    receipt: Mapping[str, object],
    receipt_sha256: str,
    expected_training_structural_sha256: str | None,
) -> dict[str, object]:
    _require(checkpoint.get("schema") == CHECKPOINT_SCHEMA, "checkpoint schema mismatch")
    _require(
        checkpoint.get("architecture") == NETWORK_SCHEMA_NAME,
        "checkpoint architecture is not the V3 network schema",
    )
    _require(
        receipt.get("schema") == TRAINING_RECEIPT_SCHEMA, "training receipt schema mismatch"
    )

    checkpoint_source = checkpoint.get("source")
    _require(isinstance(checkpoint_source, dict), "checkpoint source is missing")
    _require(
        checkpoint_source == receipt.get("source"), "checkpoint/receipt source mismatch"
    )
    _require(checkpoint_source.get("dirty") is False, "dirty training source is forbidden")

    settings = checkpoint.get("settings")
    _require(isinstance(settings, dict), "checkpoint settings are missing")
    architecture_settings = settings.get("architecture")
    _require(
        isinstance(architecture_settings, dict), "checkpoint architecture settings are missing"
    )
    name = architecture_settings.get("name")
    _require(
        isinstance(name, str) and bool(name), "checkpoint architecture name is missing"
    )
    _require(
        architecture_settings.get("schema") == NETWORK_SCHEMA_NAME,
        "checkpoint architecture schema mismatch",
    )
    training_structural = _digest(
        architecture_settings.get("structural_sha256"),
        "checkpoint training structural hash",
    )
    if expected_training_structural_sha256 is not None:
        _require(
            training_structural == expected_training_structural_sha256,
            "checkpoint training structural hash is not the expected registered value",
        )

    receipt_architecture = receipt.get("architecture")
    _require(isinstance(receipt_architecture, dict), "training receipt architecture is missing")
    _require(
        receipt_architecture.get("name") == name,
        "training receipt architecture name mismatch",
    )
    _require(
        receipt_architecture.get("schema") == NETWORK_SCHEMA_NAME,
        "training receipt architecture schema mismatch",
    )
    _require(
        receipt_architecture.get("structural_sha256") == training_structural,
        "training receipt structural hash contradicts the checkpoint",
    )
    _require(
        receipt_architecture.get("serialized_parameter_bytes") == TRAINED_PARAMETER_BYTES,
        "training receipt parameter byte count mismatch",
    )

    artifacts = receipt.get("artifacts")
    _require(isinstance(artifacts, dict), "training receipt artifacts are missing")
    checkpoint_artifact = artifacts.get("checkpoint")
    _require(
        isinstance(checkpoint_artifact, dict), "training receipt checkpoint artifact is missing"
    )
    _require(
        checkpoint_artifact.get("sha256") == checkpoint_sha256,
        "checkpoint SHA-256 contradicts its receipt",
    )

    checkpoint_data = checkpoint.get("data")
    _require(isinstance(checkpoint_data, dict), "checkpoint data identities are missing")
    _require(
        checkpoint_data == receipt.get("data"), "checkpoint/receipt data identities differ"
    )
    train_file = checkpoint_data.get("train_file")
    validation_file = checkpoint_data.get("validation_file")
    _require(
        isinstance(train_file, dict) and isinstance(validation_file, dict),
        "training file identities are missing",
    )

    claims = receipt.get("claims")
    _require(isinstance(claims, dict), "training receipt claims are missing")
    _require(
        claims.get("strength_evidence") is False,
        "integration checkpoint unexpectedly claims strength",
    )

    return {
        "checkpoint_sha256": _digest(checkpoint_sha256, "checkpoint_sha256"),
        "container_schema": CONTAINER_SCHEMA,
        "source_commit": checkpoint_source.get("commit"),
        "source_dirty": False,
        "train_file_sha256": _digest(train_file.get("sha256"), "train_file_sha256"),
        "training_architecture_structural_sha256": training_structural,
        "training_receipt_sha256": _digest(receipt_sha256, "training_receipt_sha256"),
        "validation_file_sha256": _digest(
            validation_file.get("sha256"), "validation_file_sha256"
        ),
        "wdl_calibration_sha256": _digest(
            settings.get("wdl_calibration_sha256"), "wdl_calibration_sha256"
        ),
    }


def export_checkpoint(
    checkpoint_path: Path,
    training_receipt_path: Path,
    output_path: Path,
    export_receipt_path: Path,
    expected_training_structural_sha256: str | None = None,
) -> dict[str, object]:
    output = output_path.expanduser().resolve()
    export_receipt = export_receipt_path.expanduser().resolve()
    _require(output.parent.is_dir(), f"network output parent does not exist: {output.parent}")
    _require(
        export_receipt.parent.is_dir(),
        f"receipt output parent does not exist: {export_receipt.parent}",
    )
    _require(not output.exists(), f"network output already exists: {output}")
    _require(not export_receipt.exists(), f"receipt output already exists: {export_receipt}")

    checkpoint, checkpoint_sha = _load_checkpoint(checkpoint_path)
    training_receipt, training_receipt_bytes = _read_json(training_receipt_path)
    training_receipt_sha = sha256_bytes(training_receipt_bytes)
    provenance = _validate_identities(
        checkpoint,
        checkpoint_sha,
        training_receipt,
        training_receipt_sha,
        expected_training_structural_sha256,
    )

    model_state = checkpoint.get("model_state")
    _require(isinstance(model_state, dict), "checkpoint model state is missing")
    sections, quantization = quantized_sections(model_state)
    container, container_receipt = build_container(
        sections,
        provenance,
        str(provenance["training_architecture_structural_sha256"]),
    )

    implementation = Path(__file__).resolve()
    codec = implementation.with_name("horde_v3_container.py")
    result = {
        "schema": EXPORT_RECEIPT_SCHEMA,
        "container": container_receipt,
        "network_schema": NETWORK_SCHEMA_NAME,
        "network_schema_id": NETWORK_SCHEMA_ID,
        "container_structural_sha256": structural_sha256(),
        "quantization": {
            "rounding": "nearest, ties to even via torch.round",
            "product_precision": "float64",
            "bucket_flattening": "bucket-major; container row = bucket * lanes + lane",
            "dense_scale": DENSE_SCALE,
            "feature_scale": FT_SCALE,
            "hidden_bias_scale": HIDDEN_BIAS_SCALE,
            "output_bias_scale": OUTPUT_BIAS_SCALE,
            "output_weight_scale": (
                f"{OUTPUT_WEIGHT_SCALE_NUMERATOR}/{OUTPUT_WEIGHT_SCALE_DENOMINATOR}"
            ),
            "psqt_scale": PSQT_SCALE,
            "value_semantics": "container value equals 600 times the trained model output",
            "sections": quantization,
        },
        "implementation": {
            "exporter_sha256": sha256_file(implementation),
            "codec_sha256": sha256_file(codec),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "claims": {
            "full_refresh_container": True,
            "incremental_eligible": False,
            "production_dispatch": False,
            "strength_evidence": False,
        },
    }
    receipt_payload = (
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode(
            "ascii"
        )
        + b"\n"
    )
    write_container_exclusive(output, container)
    # The network is already a valid content-addressed artifact.  A receipt
    # failure is surfaced; the network is never removed or rewritten silently.
    with export_receipt.open("xb") as destination:
        destination.write(receipt_payload)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument(
        "--expect-training-structural-sha256",
        default=None,
        help="pin the trainer architecture hash once the V3 trainer registers one",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = export_checkpoint(
        args.checkpoint,
        args.training_receipt,
        args.output,
        args.export_receipt,
        args.expect_training_structural_sha256,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContainerError, ExportError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
