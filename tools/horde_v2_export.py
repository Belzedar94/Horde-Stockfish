#!/usr/bin/env python3
"""Export an authenticated Horde V2 training checkpoint to integer format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Mapping, Sequence

try:
    import torch
    from torch import Tensor
except ImportError as error:  # pragma: no cover - CLI dependency failure
    raise SystemExit("PyTorch is required for Horde V2 checkpoint export") from error

try:
    from .horde_v2_container import (
        CONTAINER_SCHEMA,
        DENSE_SCALE,
        FT_SCALE,
        MAX_SAFE_BIAS_MAGNITUDE,
        SPECS_BY_ARCHITECTURE,
        ContainerError,
        NetworkSpec,
        build_container,
        canonical_json,
        sha256_bytes,
        sha256_file,
        write_container_exclusive,
    )
except ImportError:
    from horde_v2_container import (
        CONTAINER_SCHEMA,
        DENSE_SCALE,
        FT_SCALE,
        MAX_SAFE_BIAS_MAGNITUDE,
        SPECS_BY_ARCHITECTURE,
        ContainerError,
        NetworkSpec,
        build_container,
        canonical_json,
        sha256_bytes,
        sha256_file,
        write_container_exclusive,
    )


CHECKPOINT_SCHEMA = "HORDE_V2_BASE_CHECKPOINT_V1"
TRAINING_RECEIPT_SCHEMA = "HORDE_V2_BASE_TRAINING_V1"
EXPORT_RECEIPT_SCHEMA = "HORDE_V2_INTEGER_CHECKPOINT_EXPORT_V1"


class ExportError(ValueError):
    """Raised when a checkpoint cannot be exported without ambiguity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportError(message)


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
        root = torch.load(resolved, map_location="cpu", weights_only=True)
    except (EOFError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise ExportError(f"cannot load checkpoint: {error}") from error
    _require(isinstance(root, dict), "checkpoint root is not an object")
    return root, digest


def _select_spec(checkpoint: Mapping[str, object]) -> NetworkSpec:
    settings = checkpoint.get("settings")
    _require(isinstance(settings, dict), "checkpoint settings are missing")
    architecture = settings.get("architecture")
    _require(isinstance(architecture, dict), "checkpoint architecture settings are missing")
    name = architecture.get("name")
    _require(isinstance(name, str) and name in SPECS_BY_ARCHITECTURE, "checkpoint architecture is not exportable")
    return SPECS_BY_ARCHITECTURE[name]


def _validate_identities(
    checkpoint: Mapping[str, object],
    checkpoint_sha256: str,
    receipt: Mapping[str, object],
    receipt_sha256: str,
    spec: NetworkSpec,
) -> dict[str, object]:
    _require(checkpoint.get("schema") == CHECKPOINT_SCHEMA, "checkpoint schema mismatch")
    _require(checkpoint.get("architecture") == spec.schema_name, "checkpoint architecture mismatch")
    _require(receipt.get("schema") == TRAINING_RECEIPT_SCHEMA, "training receipt schema mismatch")

    checkpoint_source = checkpoint.get("source")
    receipt_source = receipt.get("source")
    _require(isinstance(checkpoint_source, dict) and checkpoint_source == receipt_source, "checkpoint/receipt source mismatch")
    _require(checkpoint_source.get("dirty") is False, "dirty training source is forbidden")

    settings = checkpoint.get("settings")
    _require(isinstance(settings, dict), "checkpoint settings are missing")
    architecture_settings = settings.get("architecture")
    _require(isinstance(architecture_settings, dict), "checkpoint architecture settings are missing")
    _require(architecture_settings.get("name") == spec.architecture, "checkpoint architecture name mismatch")
    _require(architecture_settings.get("schema") == spec.schema_name, "checkpoint architecture schema mismatch")
    _require(
        architecture_settings.get("structural_sha256") == spec.training_structural_sha256,
        "checkpoint training structural hash mismatch",
    )

    receipt_architecture = receipt.get("architecture")
    _require(isinstance(receipt_architecture, dict), "training receipt architecture is missing")
    _require(receipt_architecture.get("name") == spec.architecture, "training receipt architecture name mismatch")
    _require(receipt_architecture.get("schema") == spec.schema_name, "training receipt architecture schema mismatch")
    _require(
        receipt_architecture.get("structural_sha256") == spec.training_structural_sha256,
        "training receipt structural hash mismatch",
    )
    _require(
        receipt_architecture.get("serialized_parameter_bytes") == spec.parameter_bytes,
        "training receipt parameter byte count mismatch",
    )

    artifacts = receipt.get("artifacts")
    _require(isinstance(artifacts, dict), "training receipt artifacts are missing")
    checkpoint_artifact = artifacts.get("checkpoint")
    _require(isinstance(checkpoint_artifact, dict), "training receipt checkpoint artifact is missing")
    _require(checkpoint_artifact.get("sha256") == checkpoint_sha256, "checkpoint SHA-256 contradicts its receipt")

    checkpoint_data = checkpoint.get("data")
    receipt_data = receipt.get("data")
    _require(isinstance(checkpoint_data, dict) and checkpoint_data == receipt_data, "checkpoint/receipt data identities differ")
    train_file = checkpoint_data.get("train_file")
    validation_file = checkpoint_data.get("validation_file")
    _require(isinstance(train_file, dict) and isinstance(validation_file, dict), "training file identities are missing")
    train_sha = train_file.get("sha256")
    validation_sha = validation_file.get("sha256")
    wdl_sha = settings.get("wdl_calibration_sha256")
    _require(isinstance(train_sha, str) and isinstance(validation_sha, str), "dataset SHA-256 identities are missing")
    _require(isinstance(wdl_sha, str), "WDL calibration SHA-256 is missing")

    claims = receipt.get("claims")
    _require(isinstance(claims, dict), "training receipt claims are missing")
    _require(claims.get("strength_evidence") is False, "integration checkpoint unexpectedly claims strength")

    return {
        "checkpoint_sha256": checkpoint_sha256,
        "container_schema": CONTAINER_SCHEMA,
        "source_commit": checkpoint_source.get("commit"),
        "source_dirty": False,
        "train_file_sha256": train_sha,
        "training_architecture_structural_sha256": spec.training_structural_sha256,
        "training_receipt_sha256": receipt_sha256,
        "validation_file_sha256": validation_sha,
        "wdl_calibration_sha256": wdl_sha,
    }


def _expected_state(spec: NetworkSpec) -> dict[str, tuple[int, ...]]:
    return {section.name: section.shape for section in spec.sections}


def _quantize(tensor: Tensor, scale: int, dtype: str, name: str) -> tuple[bytes, dict[str, object]]:
    _require(tensor.device.type == "cpu", f"{name} is not on CPU")
    _require(tensor.dtype == torch.float32, f"{name} is not float32")
    _require(bool(torch.isfinite(tensor).all()), f"{name} contains a non-finite value")
    rounded = torch.round(tensor.detach() * scale).to(torch.int64)
    limits = {"i8": (-(1 << 7), (1 << 7) - 1), "i16": (-(1 << 15), (1 << 15) - 1), "i32": (-(1 << 31), (1 << 31) - 1)}
    minimum, maximum = limits[dtype]
    actual_minimum = int(rounded.min().item())
    actual_maximum = int(rounded.max().item())
    _require(actual_minimum >= minimum and actual_maximum <= maximum, f"{name} overflows {dtype}")
    if dtype == "i32" and name.endswith("bias"):
        _require(
            max(abs(actual_minimum), abs(actual_maximum)) <= MAX_SAFE_BIAS_MAGNITUDE,
            f"{name} exceeds the registered safe bias magnitude",
        )
    numpy_dtype = {"i8": "<i1", "i16": "<i2", "i32": "<i4"}[dtype]
    payload = rounded.numpy().astype(numpy_dtype, copy=False).tobytes(order="C")
    return payload, {
        "float_min": float(tensor.min().item()),
        "float_max": float(tensor.max().item()),
        "integer_min": actual_minimum,
        "integer_max": actual_maximum,
        "scale": scale,
    }


def _quantized_sections(
    checkpoint: Mapping[str, object], spec: NetworkSpec
) -> tuple[dict[str, bytes], dict[str, dict[str, object]]]:
    model_state = checkpoint.get("model_state")
    _require(isinstance(model_state, dict), "checkpoint model state is missing")
    expected = _expected_state(spec)
    _require(set(model_state) == set(expected), "checkpoint parameter names do not match the architecture")
    sections: dict[str, bytes] = {}
    statistics: dict[str, dict[str, object]] = {}
    for section in spec.sections:
        value = model_state[section.name]
        _require(isinstance(value, Tensor), f"checkpoint parameter {section.name} is not a tensor")
        _require(tuple(value.shape) == section.shape, f"checkpoint parameter {section.name} shape mismatch")
        scale = FT_SCALE if section.name in {spec.first_weight_name, spec.first_bias_name, "global_weights", "global_bias"} or section.name.endswith("bias") else DENSE_SCALE
        payload, stats = _quantize(value.contiguous(), scale, section.dtype, section.name)
        _require(len(payload) == section.byte_length, f"quantized section {section.name} byte count mismatch")
        sections[section.name] = payload
        statistics[section.name] = stats
    return sections, statistics


def export_checkpoint(
    checkpoint_path: Path,
    training_receipt_path: Path,
    output_path: Path,
    export_receipt_path: Path,
) -> dict[str, object]:
    output = output_path.expanduser().resolve()
    export_receipt = export_receipt_path.expanduser().resolve()
    _require(output.parent.is_dir(), f"network output parent does not exist: {output.parent}")
    _require(export_receipt.parent.is_dir(), f"receipt output parent does not exist: {export_receipt.parent}")
    _require(not output.exists(), f"network output already exists: {output}")
    _require(not export_receipt.exists(), f"receipt output already exists: {export_receipt}")

    checkpoint, checkpoint_sha = _load_checkpoint(checkpoint_path)
    training_receipt, training_receipt_bytes = _read_json(training_receipt_path)
    training_receipt_sha = sha256_bytes(training_receipt_bytes)
    spec = _select_spec(checkpoint)
    provenance = _validate_identities(
        checkpoint,
        checkpoint_sha,
        training_receipt,
        training_receipt_sha,
        spec,
    )
    sections, quantization = _quantized_sections(checkpoint, spec)
    container, container_receipt = build_container(spec, sections, provenance)

    implementation = Path(__file__).resolve()
    codec = implementation.with_name("horde_v2_container.py")
    result = {
        "schema": EXPORT_RECEIPT_SCHEMA,
        "container": container_receipt,
        "quantization": {
            "rounding": "nearest, ties to even via torch.round",
            "feature_scale": FT_SCALE,
            "dense_scale": DENSE_SCALE,
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
    receipt_payload = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
    write_container_exclusive(output, container)
    try:
        with export_receipt.open("xb") as destination:
            destination.write(receipt_payload)
    except OSError:
        # The network remains a valid content-addressed artifact.  Do not remove
        # or overwrite it silently; the caller receives the receipt failure.
        raise
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = export_checkpoint(
        args.checkpoint,
        args.training_receipt,
        args.output,
        args.export_receipt,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContainerError, ExportError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
