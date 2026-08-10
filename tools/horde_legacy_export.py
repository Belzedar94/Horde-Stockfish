#!/usr/bin/env python3
"""Export an authenticated fresh Horde legacy checkpoint to legacy NNUE bytes.

The wire format is the frozen Fairy-Stockfish Hordetest format used by Run 6B.
This exporter does not make a network loadable by itself: Horde-Stockfish also
requires the resulting full SHA-256 and byte size to be present in its compiled
legacy artifact registry.
"""

from __future__ import annotations

import argparse
import hashlib
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
    raise SystemExit("PyTorch is required for Horde legacy checkpoint export") from error


CHECKPOINT_SCHEMA = "HORDE_FRESH_LEGACY_CONTROL_CHECKPOINT_V3"
TRAINING_RECEIPT_SCHEMA = "HORDE_FRESH_LEGACY_CONTROL_TRAINING_V3"
ARCHITECTURE_SCHEMA = "HORDETEST_HP_FRESH_CONTROL_V1"
FEATURE_SCHEMA = "HORDETEST_HP_LEGACY_V1"
EXPORT_RECEIPT_SCHEMA = "HORDE_FRESH_LEGACY_NNUE_EXPORT_V1"

FILE_VERSION = 0x7AF32F20
NETWORK_HASH = 0x3C103E72
TRANSFORMER_HASH = 0x5F2348B8
ARCHITECTURE_HASH = 0x633376CA

FEATURE_DIMENSIONS = 896
ACCUMULATOR_LANES = 512
PSQT_BUCKETS = 8
LAYER_STACKS = 8
NETWORK_INPUTS = 2 * ACCUMULATOR_LANES
HIDDEN0_LANES = 16
HIDDEN1_LANES = 32

FT_SCALE = 127.0
PSQT_SCALE = 600.0 * 16.0
HIDDEN_BIAS_SCALE = 127.0 * 64.0
HIDDEN_WEIGHT_SCALE = 64.0
OUTPUT_BIAS_SCALE = 600.0 * 16.0
OUTPUT_WEIGHT_SCALE = OUTPUT_BIAS_SCALE / 127.0

MODEL_SHAPES: dict[str, tuple[int, ...]] = {
    "ft_weights": (FEATURE_DIMENSIONS, ACCUMULATOR_LANES),
    "ft_bias": (ACCUMULATOR_LANES,),
    "psqt_weights": (FEATURE_DIMENSIONS, PSQT_BUCKETS),
    "hidden0_weights": (LAYER_STACKS, HIDDEN0_LANES, NETWORK_INPUTS),
    "hidden0_bias": (LAYER_STACKS, HIDDEN0_LANES),
    "hidden1_weights": (LAYER_STACKS, HIDDEN1_LANES, HIDDEN0_LANES),
    "hidden1_bias": (LAYER_STACKS, HIDDEN1_LANES),
    "output_weights": (LAYER_STACKS, 1, HIDDEN1_LANES),
    "output_bias": (LAYER_STACKS, 1),
}


class LegacyExportError(ValueError):
    """Raised when a checkpoint cannot be exported without ambiguity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LegacyExportError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.upper()
        and all(character in "0123456789ABCDEF" for character in value)
    )


def _read_json(path: Path) -> tuple[dict[str, object], str]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"training receipt does not exist: {resolved}")
    payload = resolved.read_bytes()
    try:
        root = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LegacyExportError(f"cannot decode training receipt: {error}") from error
    _require(isinstance(root, dict), "training receipt root is not an object")
    return root, sha256_bytes(payload)


def _load_checkpoint(path: Path) -> tuple[dict[str, object], str]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"checkpoint does not exist: {resolved}")
    digest = sha256_file(resolved)
    try:
        root = torch.load(resolved, map_location="cpu", weights_only=True)
    except (EOFError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise LegacyExportError(f"cannot load checkpoint: {error}") from error
    _require(isinstance(root, dict), "checkpoint root is not an object")
    return root, digest


def _validate_identities(
    checkpoint: Mapping[str, object],
    checkpoint_sha256: str,
    receipt: Mapping[str, object],
    receipt_sha256: str,
) -> dict[str, object]:
    _require(checkpoint.get("schema") == CHECKPOINT_SCHEMA, "checkpoint schema mismatch")
    _require(checkpoint.get("architecture") == ARCHITECTURE_SCHEMA, "checkpoint architecture mismatch")
    _require(receipt.get("schema") == TRAINING_RECEIPT_SCHEMA, "training receipt schema mismatch")

    checkpoint_source = checkpoint.get("source")
    receipt_source = receipt.get("source")
    _require(
        isinstance(checkpoint_source, dict) and checkpoint_source == receipt_source,
        "checkpoint/receipt source mismatch",
    )
    _require(checkpoint_source.get("dirty") is False, "dirty training source is forbidden")
    source_commit = checkpoint_source.get("commit")
    _require(
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit),
        "training source commit is invalid",
    )

    architecture = receipt.get("architecture")
    _require(isinstance(architecture, dict), "training receipt architecture is missing")
    _require(architecture.get("schema") == ARCHITECTURE_SCHEMA, "training receipt architecture mismatch")
    _require(
        architecture.get("legacy_feature_schema") == FEATURE_SCHEMA,
        "training receipt feature schema mismatch",
    )
    _require(
        architecture.get("serialized_topology")
        == "896 -> 512 shared FT + PSQT; 8 x (1024 -> 16 -> 32 -> 1)",
        "training receipt serialized topology mismatch",
    )
    _require(architecture.get("training_only_factorizer") is False, "factorized legacy checkpoints are forbidden")

    artifacts = receipt.get("artifacts")
    _require(isinstance(artifacts, dict), "training receipt artifacts are missing")
    checkpoint_artifact = artifacts.get("checkpoint")
    _require(isinstance(checkpoint_artifact, dict), "training receipt checkpoint artifact is missing")
    _require(
        checkpoint_artifact.get("sha256") == checkpoint_sha256,
        "checkpoint SHA-256 contradicts its receipt",
    )

    checkpoint_data = checkpoint.get("data")
    receipt_data = receipt.get("data")
    _require(
        isinstance(checkpoint_data, dict) and checkpoint_data == receipt_data,
        "checkpoint/receipt data identities differ",
    )
    train_file = checkpoint_data.get("train_file")
    validation_file = checkpoint_data.get("validation_file")
    _require(isinstance(train_file, dict), "training file identity is missing")
    _require(isinstance(validation_file, dict), "validation file identity is missing")
    train_sha = train_file.get("sha256")
    validation_sha = validation_file.get("sha256")
    _require(_is_sha256(train_sha), "training SHA-256 is missing or invalid")
    _require(
        _is_sha256(validation_sha),
        "validation SHA-256 is missing or invalid",
    )
    wdl_calibration = checkpoint_data.get("wdl_calibration")
    _require(isinstance(wdl_calibration, dict), "WDL calibration identity is missing")
    wdl_sha = wdl_calibration.get("sha256")
    _require(_is_sha256(wdl_sha), "WDL SHA-256 is missing or invalid")

    settings = checkpoint.get("settings")
    run = receipt.get("run")
    labels = receipt.get("labels")
    claims = receipt.get("claims")
    _require(isinstance(settings, dict), "checkpoint settings are missing")
    _require(isinstance(run, dict) and run.get("complete") is True, "training run is incomplete")
    _require(isinstance(labels, dict), "training receipt labels are missing")
    _require(isinstance(claims, dict), "training receipt claims are missing")
    _require(claims.get("production_network") is False, "training receipt unexpectedly claims production")
    _require(claims.get("strength_evidence") is False, "training receipt unexpectedly claims strength")
    seed = settings.get("seed")
    _require(type(seed) is int and seed > 0, "checkpoint seed is invalid")
    _require(run.get("seed") == seed, "checkpoint/receipt seed mismatch")
    _require(settings.get("wdl_calibration_sha256") == wdl_sha, "WDL identity mismatch")
    _require(labels.get("network_to_score") == 600.0, "legacy output scale contract mismatch")

    return {
        "architecture_schema": ARCHITECTURE_SCHEMA,
        "checkpoint_sha256": checkpoint_sha256,
        "feature_schema": FEATURE_SCHEMA,
        "seed": seed,
        "source_commit": source_commit,
        "source_dirty": False,
        "train_file_sha256": train_sha,
        "training_receipt_sha256": receipt_sha256,
        "validation_file_sha256": validation_sha,
        "wdl_calibration_sha256": wdl_sha,
    }


def _model_state(checkpoint: Mapping[str, object]) -> dict[str, Tensor]:
    state = checkpoint.get("model_state")
    _require(isinstance(state, dict), "checkpoint model state is missing")
    _require(set(state) == set(MODEL_SHAPES), "checkpoint parameter names do not match legacy topology")
    result: dict[str, Tensor] = {}
    for name, shape in MODEL_SHAPES.items():
        value = state[name]
        _require(isinstance(value, Tensor), f"checkpoint parameter {name} is not a tensor")
        _require(value.device.type == "cpu", f"checkpoint parameter {name} is not on CPU")
        _require(value.dtype == torch.float32, f"checkpoint parameter {name} is not float32")
        _require(tuple(value.shape) == shape, f"checkpoint parameter {name} shape mismatch")
        _require(bool(torch.isfinite(value).all()), f"checkpoint parameter {name} is non-finite")
        result[name] = value.detach().contiguous()
    return result


def _quantize(
    value: Tensor,
    scale: float,
    dtype: str,
    name: str,
) -> tuple[Tensor, dict[str, object]]:
    rounded = torch.round(value.to(torch.float64) * scale).to(torch.int64)
    limits = {
        "i8": (-(1 << 7), (1 << 7) - 1, torch.int8),
        "i16": (-(1 << 15), (1 << 15) - 1, torch.int16),
        "i32": (-(1 << 31), (1 << 31) - 1, torch.int32),
    }
    minimum, maximum, torch_dtype = limits[dtype]
    actual_minimum = int(rounded.min().item())
    actual_maximum = int(rounded.max().item())
    _require(actual_minimum >= minimum and actual_maximum <= maximum, f"{name} overflows {dtype}")
    quantized = rounded.to(torch_dtype)
    return quantized, {
        "dtype": dtype,
        "float_min": float(value.min().item()),
        "float_max": float(value.max().item()),
        "integer_min": actual_minimum,
        "integer_max": actual_maximum,
        "scale": scale,
    }


def _tensor_bytes(value: Tensor, dtype: str) -> bytes:
    numpy_dtype = {"i8": "<i1", "i16": "<i2", "i32": "<i4"}[dtype]
    return value.cpu().numpy().astype(numpy_dtype, copy=False).tobytes(order="C")


def _dense_bytes(
    weight: Tensor,
    bias: Tensor,
    weight_scale: float,
    bias_scale: float,
    label: str,
) -> tuple[bytes, dict[str, object]]:
    _require(weight.ndim == 2 and bias.ndim == 1, f"{label} dense tensors have invalid ranks")
    _require(weight.shape[0] == bias.shape[0], f"{label} dense output dimensions differ")
    quantized_bias, bias_stats = _quantize(bias, bias_scale, "i32", f"{label}.bias")
    quantized_weight, weight_stats = _quantize(weight, weight_scale, "i8", f"{label}.weight")
    inputs = int(weight.shape[1])
    padded_inputs = (inputs + 31) // 32 * 32
    if padded_inputs != inputs:
        padded = torch.zeros((weight.shape[0], padded_inputs), dtype=torch.int8)
        padded[:, :inputs] = quantized_weight
        quantized_weight = padded
    payload = _tensor_bytes(quantized_bias, "i32") + _tensor_bytes(quantized_weight, "i8")
    return payload, {
        "bias": bias_stats,
        "input_dimensions": inputs,
        "output_dimensions": int(weight.shape[0]),
        "padded_input_dimensions": padded_inputs,
        "weight": weight_stats,
    }


def build_legacy_nnue(
    checkpoint: Mapping[str, object],
    description: str,
) -> tuple[bytes, dict[str, object]]:
    try:
        description_bytes = description.encode("ascii")
    except UnicodeEncodeError as error:
        raise LegacyExportError("network description must be ASCII") from error
    _require(0 < len(description_bytes) <= 1024, "network description length is invalid")
    state = _model_state(checkpoint)

    ft_bias, ft_bias_stats = _quantize(state["ft_bias"], FT_SCALE, "i16", "ft_bias")
    ft_weights, ft_weight_stats = _quantize(
        state["ft_weights"], FT_SCALE, "i16", "ft_weights"
    )
    psqt, psqt_stats = _quantize(
        state["psqt_weights"], PSQT_SCALE, "i32", "psqt_weights"
    )

    payload = bytearray()
    payload.extend(struct.pack("<III", FILE_VERSION, NETWORK_HASH, len(description_bytes)))
    payload.extend(description_bytes)
    payload.extend(struct.pack("<I", TRANSFORMER_HASH))
    payload.extend(_tensor_bytes(ft_bias, "i16"))
    payload.extend(_tensor_bytes(ft_weights, "i16"))
    payload.extend(_tensor_bytes(psqt, "i32"))

    stack_stats: list[dict[str, object]] = []
    for bucket in range(LAYER_STACKS):
        payload.extend(struct.pack("<I", ARCHITECTURE_HASH))
        fc0, fc0_stats = _dense_bytes(
            state["hidden0_weights"][bucket],
            state["hidden0_bias"][bucket],
            HIDDEN_WEIGHT_SCALE,
            HIDDEN_BIAS_SCALE,
            f"stack{bucket}.hidden0",
        )
        fc1, fc1_stats = _dense_bytes(
            state["hidden1_weights"][bucket],
            state["hidden1_bias"][bucket],
            HIDDEN_WEIGHT_SCALE,
            HIDDEN_BIAS_SCALE,
            f"stack{bucket}.hidden1",
        )
        output, output_stats = _dense_bytes(
            state["output_weights"][bucket],
            state["output_bias"][bucket],
            OUTPUT_WEIGHT_SCALE,
            OUTPUT_BIAS_SCALE,
            f"stack{bucket}.output",
        )
        payload.extend(fc0)
        payload.extend(fc1)
        payload.extend(output)
        stack_stats.append(
            {"bucket": bucket, "hidden0": fc0_stats, "hidden1": fc1_stats, "output": output_stats}
        )

    expected_bytes = 1_088_336 + len(description_bytes)
    _require(len(payload) == expected_bytes, f"legacy NNUE byte count mismatch: {len(payload)}")
    return bytes(payload), {
        "feature_transformer": {
            "bias": ft_bias_stats,
            "weights": ft_weight_stats,
            "psqt": psqt_stats,
        },
        "layer_stacks": stack_stats,
        "rounding": "nearest, ties to even via torch.round",
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    created = False
    try:
        with path.open("xb") as destination:
            created = True
            destination.write(payload)
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def export_checkpoint(
    checkpoint_path: Path,
    training_receipt_path: Path,
    output_path: Path,
    export_receipt_path: Path,
    description: str | None = None,
) -> dict[str, object]:
    checkpoint, checkpoint_sha = _load_checkpoint(checkpoint_path)
    training_receipt, training_receipt_sha = _read_json(training_receipt_path)
    provenance = _validate_identities(
        checkpoint,
        checkpoint_sha,
        training_receipt,
        training_receipt_sha,
    )
    if description is None:
        description = (
            "Horde fresh legacy 250k seed "
            f"{provenance['seed']}; checkpoint {checkpoint_sha}; schema {ARCHITECTURE_SCHEMA}"
        )
    network, quantization = build_legacy_nnue(checkpoint, description)
    network_sha = sha256_bytes(network)

    output = output_path.expanduser().resolve()
    export_receipt = export_receipt_path.expanduser().resolve()
    _require(output.parent.is_dir(), f"network output parent does not exist: {output.parent}")
    _require(export_receipt.parent.is_dir(), f"receipt output parent does not exist: {export_receipt.parent}")
    _require(output != export_receipt, "network and receipt outputs must be different files")
    _require(not output.exists(), f"network output already exists: {output}")
    _require(not export_receipt.exists(), f"export receipt already exists: {export_receipt}")

    result = {
        "schema": EXPORT_RECEIPT_SCHEMA,
        "artifact": {
            "description": description,
            "file_name": output.name,
            "file_size": len(network),
            "sha256": network_sha,
        },
        "format": {
            "architecture_hash": f"0x{ARCHITECTURE_HASH:08X}",
            "feature_dimensions": FEATURE_DIMENSIONS,
            "feature_transformer_hash": f"0x{TRANSFORMER_HASH:08X}",
            "layer_stacks": LAYER_STACKS,
            "network_hash": f"0x{NETWORK_HASH:08X}",
            "version": f"0x{FILE_VERSION:08X}",
        },
        "provenance": provenance,
        "quantization": quantization,
        "implementation": {
            "exporter_sha256": sha256_file(Path(__file__).resolve()),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
        "claims": {
            "compiled_registry_required": True,
            "incremental_eligible": True,
            "production_network": False,
            "strength_evidence": False,
        },
    }
    receipt_payload = (
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")
    _write_exclusive(output, network)
    try:
        _write_exclusive(export_receipt, receipt_payload)
    except BaseException:
        # The network was created by this call, so removing it cannot destroy a
        # pre-existing artifact. Keep the two-output operation fail-closed.
        output.unlink(missing_ok=True)
        raise
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--description")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = export_checkpoint(
        args.checkpoint,
        args.training_receipt,
        args.output,
        args.export_receipt,
        args.description,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LegacyExportError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
