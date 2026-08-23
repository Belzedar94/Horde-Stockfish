#!/usr/bin/env python3
"""Import a legacy Horde NNUE container back into a trainer checkpoint.

The exporter is the definition of the wire format, so this tool is written as
its exact inverse and validated against it rather than against a re-reading of
the format: import a container, hand the result straight back to
``horde_legacy_export.build_legacy_nnue`` and require the bytes to return.

What comes out is deliberately not a training receipt. A network that was
trained somewhere else has no run behind it here, so the checkpoint carries the
architecture, the parameters and the provenance of the container it came from,
and nothing that would let it masquerade as a run this repository performed. It
is usable as ``--init`` and it is not resumable, which is the honest shape.

Precision. Quantisation is lossy in one direction only: the container already
holds integers, so dequantising and requantising has to return the same
integers or the round-trip fails. What the float state cannot recover is the
information the original export threw away, and one consequence is recorded
explicitly: an int8 dense weight of -128 dequantises to -2.0, which sits
outside the clipping range the trainer enforces after every optimizer step, so
the first step of a run initialised from such a container moves those weights
to -1.984375. The count is in the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
from typing import Mapping, Sequence

try:
    import torch
    from torch import Tensor
except ImportError as error:  # pragma: no cover - CLI dependency failure
    raise SystemExit("PyTorch is required for Horde legacy container import") from error

sys.path.insert(0, str(Path(__file__).resolve().parent))

import horde_legacy_export as export  # noqa: E402


IMPORT_RECEIPT_SCHEMA = "HORDE_FRESH_LEGACY_NNUE_IMPORT_V1"
CHECKPOINT_PROVENANCE_KEY = "imported_from"

# The trainer's dense clipping bound, restated here so the receipt can report
# how many weights the first optimizer step will move. It is not applied on
# import: the imported network is the container's network, unmodified.
DENSE_HIDDEN_CLIP = 127.0 / 64.0


class LegacyImportError(ValueError):
    """Raised when a container cannot be read back without ambiguity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LegacyImportError(message)


class _Reader:
    """Sequential reader that cannot silently run off the end of a container."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    @property
    def offset(self) -> int:
        return self._offset

    def take(self, count: int, what: str) -> bytes:
        _require(count >= 0, f"{what} has a negative length")
        end = self._offset + count
        _require(
            end <= len(self._payload),
            f"container ends inside {what}: needed {count} bytes at offset {self._offset}",
        )
        chunk = self._payload[self._offset : end]
        self._offset = end
        return chunk

    def uint32(self, what: str) -> int:
        return struct.unpack("<I", self.take(4, what))[0]

    def tensor(self, dtype: str, shape: tuple[int, ...], what: str) -> Tensor:
        numpy_dtype = {"i8": "<i1", "i16": "<i2", "i32": "<i4"}[dtype]
        width = int(numpy_dtype[-1])
        count = 1
        for extent in shape:
            count *= extent
        raw = self.take(count * width, what)
        # frombuffer over a read-only buffer would hand back a non-writable
        # array, and torch warns on that; the copy is a few hundred kilobytes.
        flat = torch.frombuffer(bytearray(raw), dtype={
            "i8": torch.int8,
            "i16": torch.int16,
            "i32": torch.int32,
        }[dtype])
        return flat.reshape(shape)

    def exhausted(self) -> bool:
        return self._offset == len(self._payload)


def _dequantize(value: Tensor, scale: float) -> Tensor:
    return (value.to(torch.float64) / scale).to(torch.float32)


def _integer_stats(value: Tensor, scale: float, dtype: str) -> dict[str, object]:
    return {
        "dtype": dtype,
        "integer_max": int(value.max().item()),
        "integer_min": int(value.min().item()),
        "scale": scale,
    }


def _read_dense(
    reader: _Reader,
    outputs: int,
    inputs: int,
    weight_scale: float,
    bias_scale: float,
    label: str,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    """Read one dense layer, undoing the exporter's input padding."""

    padded_inputs = (inputs + 31) // 32 * 32
    bias = reader.tensor("i32", (outputs,), f"{label}.bias")
    weight = reader.tensor("i8", (outputs, padded_inputs), f"{label}.weight")
    if padded_inputs != inputs:
        tail = weight[:, inputs:]
        # The exporter pads with zeros. A non-zero tail means this is not the
        # layout this tool believes it is, and guessing would corrupt the
        # network silently, so it is refused.
        _require(
            bool((tail == 0).all()),
            f"{label} padding is not zero, so the container layout is not the expected one",
        )
        weight = weight[:, :inputs]
    weight = weight.contiguous()
    return (
        _dequantize(weight, weight_scale),
        _dequantize(bias, bias_scale),
        {
            "bias": _integer_stats(bias, bias_scale, "i32"),
            "input_dimensions": inputs,
            "output_dimensions": outputs,
            "padded_input_dimensions": padded_inputs,
            "weight": _integer_stats(weight, weight_scale, "i8"),
            "weights_at_the_negative_int8_bound": int((weight == -(1 << 7)).sum().item()),
        },
    )


def read_legacy_nnue(payload: bytes) -> tuple[dict[str, Tensor], str, dict[str, object]]:
    """Parse a legacy container into float32 parameters, description and stats."""

    reader = _Reader(payload)
    version = reader.uint32("file version")
    network_hash = reader.uint32("network hash")
    description_length = reader.uint32("description length")
    _require(
        version == export.FILE_VERSION,
        f"file version is 0x{version:08X}, not 0x{export.FILE_VERSION:08X}",
    )
    _require(
        network_hash == export.NETWORK_HASH,
        f"network hash is 0x{network_hash:08X}, not 0x{export.NETWORK_HASH:08X}",
    )
    _require(
        0 < description_length <= 1024,
        f"description length {description_length} is outside the format's range",
    )
    description_bytes = reader.take(description_length, "description")
    try:
        description = description_bytes.decode("ascii")
    except UnicodeDecodeError as error:
        raise LegacyImportError("description is not ASCII") from error

    transformer_hash = reader.uint32("feature transformer hash")
    _require(
        transformer_hash == export.TRANSFORMER_HASH,
        f"transformer hash is 0x{transformer_hash:08X}, not 0x{export.TRANSFORMER_HASH:08X}",
    )
    ft_bias = reader.tensor("i16", (export.ACCUMULATOR_LANES,), "ft_bias")
    ft_weights = reader.tensor(
        "i16", (export.FEATURE_DIMENSIONS, export.ACCUMULATOR_LANES), "ft_weights"
    )
    psqt = reader.tensor(
        "i32", (export.FEATURE_DIMENSIONS, export.PSQT_BUCKETS), "psqt_weights"
    )

    hidden0_weights: list[Tensor] = []
    hidden0_bias: list[Tensor] = []
    hidden1_weights: list[Tensor] = []
    hidden1_bias: list[Tensor] = []
    output_weights: list[Tensor] = []
    output_bias: list[Tensor] = []
    stack_stats: list[dict[str, object]] = []
    for bucket in range(export.LAYER_STACKS):
        architecture_hash = reader.uint32(f"stack{bucket} architecture hash")
        _require(
            architecture_hash == export.ARCHITECTURE_HASH,
            f"stack {bucket} architecture hash is 0x{architecture_hash:08X}, "
            f"not 0x{export.ARCHITECTURE_HASH:08X}",
        )
        weight, bias, fc0_stats = _read_dense(
            reader,
            export.HIDDEN0_LANES,
            export.NETWORK_INPUTS,
            export.HIDDEN_WEIGHT_SCALE,
            export.HIDDEN_BIAS_SCALE,
            f"stack{bucket}.hidden0",
        )
        hidden0_weights.append(weight)
        hidden0_bias.append(bias)
        weight, bias, fc1_stats = _read_dense(
            reader,
            export.HIDDEN1_LANES,
            export.HIDDEN0_LANES,
            export.HIDDEN_WEIGHT_SCALE,
            export.HIDDEN_BIAS_SCALE,
            f"stack{bucket}.hidden1",
        )
        hidden1_weights.append(weight)
        hidden1_bias.append(bias)
        weight, bias, output_stats = _read_dense(
            reader,
            1,
            export.HIDDEN1_LANES,
            export.OUTPUT_WEIGHT_SCALE,
            export.OUTPUT_BIAS_SCALE,
            f"stack{bucket}.output",
        )
        output_weights.append(weight)
        output_bias.append(bias)
        stack_stats.append(
            {
                "bucket": bucket,
                "hidden0": fc0_stats,
                "hidden1": fc1_stats,
                "output": output_stats,
            }
        )

    _require(
        reader.exhausted(),
        f"container has {len(payload) - reader.offset} trailing bytes after the last stack",
    )

    state = {
        "ft_weights": _dequantize(ft_weights, export.FT_SCALE),
        "ft_bias": _dequantize(ft_bias, export.FT_SCALE),
        "psqt_weights": _dequantize(psqt, export.PSQT_SCALE),
        "hidden0_weights": torch.stack(hidden0_weights),
        "hidden0_bias": torch.stack(hidden0_bias),
        "hidden1_weights": torch.stack(hidden1_weights),
        "hidden1_bias": torch.stack(hidden1_bias),
        "output_weights": torch.stack(output_weights),
        "output_bias": torch.stack(output_bias),
    }
    # The exporter's own shape table is the authority, so a layout change there
    # breaks this import loudly instead of producing a plausible wrong network.
    _require(
        set(state) == set(export.MODEL_SHAPES),
        "imported parameter names do not match the legacy topology",
    )
    for name, shape in export.MODEL_SHAPES.items():
        _require(
            tuple(state[name].shape) == shape,
            f"imported parameter {name} has shape {tuple(state[name].shape)}, expected {shape}",
        )
        state[name] = state[name].contiguous()
        _require(
            bool(torch.isfinite(state[name]).all()),
            f"imported parameter {name} is non-finite",
        )

    dequantization = {
        "feature_transformer": {
            "bias": _integer_stats(ft_bias, export.FT_SCALE, "i16"),
            "psqt": _integer_stats(psqt, export.PSQT_SCALE, "i32"),
            "weights": _integer_stats(ft_weights, export.FT_SCALE, "i16"),
        },
        "layer_stacks": stack_stats,
        "method": "integer divided by the exporter's scale, stored as float32",
    }
    return state, description, dequantization


def _repository_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str:
        return subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
        }
    except (OSError, subprocess.SubprocessError) as error:
        raise LegacyImportError(f"cannot read the importer source identity: {error}") from error


def _round_trip(
    checkpoint: Mapping[str, object],
    description: str,
    original: bytes,
) -> dict[str, object]:
    """Re-serialise through the exporter and compare against the container.

    ``build_legacy_nnue`` is the function every production export calls, so the
    bytes compared here are produced by the production path. What is skipped is
    ``_validate_identities``, which authenticates a TRAINING receipt: an
    imported champion has no training run in this repository, and fabricating a
    receipt so the gate would pass is the one thing that would actually
    invalidate this check.
    """

    rebuilt, quantization = export.build_legacy_nnue(checkpoint, description)
    identical = rebuilt == original
    result: dict[str, object] = {
        "byte_identical": identical,
        "method": (
            "horde_legacy_export.build_legacy_nnue on the imported checkpoint, "
            "compared against the source container byte for byte"
        ),
        "rebuilt_bytes": len(rebuilt),
        "rebuilt_sha256": export.sha256_bytes(rebuilt),
        "source_bytes": len(original),
        "source_sha256": export.sha256_bytes(original),
        "quantization": quantization,
    }
    if not identical:
        differences = [
            offset
            for offset in range(min(len(rebuilt), len(original)))
            if rebuilt[offset] != original[offset]
        ]
        result["differing_bytes"] = len(differences)
        result["first_differing_offsets"] = differences[:16]
    return result


def import_container(
    network_path: Path,
    output_path: Path,
    receipt_path: Path,
) -> dict[str, object]:
    network = network_path.expanduser().resolve()
    _require(network.is_file(), f"network does not exist: {network}")
    payload = network.read_bytes()
    network_sha256 = export.sha256_bytes(payload)

    state, description, dequantization = read_legacy_nnue(payload)
    source = _repository_identity()
    imported_from = {
        "bytes": len(payload),
        "description": description,
        "file_name": network.name,
        "format": {
            "architecture_hash": f"0x{export.ARCHITECTURE_HASH:08X}",
            "feature_transformer_hash": f"0x{export.TRANSFORMER_HASH:08X}",
            "network_hash": f"0x{export.NETWORK_HASH:08X}",
            "version": f"0x{export.FILE_VERSION:08X}",
        },
        "sha256": network_sha256,
    }
    checkpoint: dict[str, object] = {
        "schema": export.CHECKPOINT_SCHEMA,
        "architecture": export.ARCHITECTURE_SCHEMA,
        "model_state": state,
        "source": source,
        CHECKPOINT_PROVENANCE_KEY: imported_from,
        # Named rather than merely absent, so a reader who finds no optimizer
        # state knows it was never there instead of wondering what stripped it.
        "not_a_training_run": (
            "this checkpoint was decoded from a network container, so it carries "
            "no optimizer state, no scheduler state, no RNG state and no "
            "progress; it can initialise a run and cannot resume one"
        ),
    }

    round_trip = _round_trip(checkpoint, description, payload)

    output = output_path.expanduser().resolve()
    receipt_output = receipt_path.expanduser().resolve()
    _require(output.parent.is_dir(), f"checkpoint output parent does not exist: {output.parent}")
    _require(
        receipt_output.parent.is_dir(),
        f"receipt output parent does not exist: {receipt_output.parent}",
    )
    _require(output != receipt_output, "checkpoint and receipt outputs must be different files")
    _require(not output.exists(), f"checkpoint output already exists: {output}")
    _require(not receipt_output.exists(), f"import receipt already exists: {receipt_output}")

    torch.save(checkpoint, output)
    try:
        checkpoint_sha256 = export.sha256_file(output)
        result = {
            "schema": IMPORT_RECEIPT_SCHEMA,
            "artifact": {
                "file_name": output.name,
                "file_size": output.stat().st_size,
                "sha256": checkpoint_sha256,
            },
            "claims": {
                "production_network": False,
                "resumable": False,
                "strength_evidence": False,
                "usable_as_trainer_init": True,
            },
            "dequantization": dequantization,
            "first_optimizer_step_will_clip": {
                "dense_hidden_range": [-DENSE_HIDDEN_CLIP, DENSE_HIDDEN_CLIP],
                "note": (
                    "an int8 dense weight of -128 dequantises to -2.0, which the "
                    "trainer clamps to -127/64 after the first optimizer step; the "
                    "import itself changes nothing"
                ),
                "weights_at_the_negative_int8_bound": sum(
                    int(stack[layer]["weights_at_the_negative_int8_bound"])
                    for stack in dequantization["layer_stacks"]
                    for layer in ("hidden0", "hidden1", "output")
                ),
            },
            "implementation": {
                "exporter_sha256": export.sha256_file(
                    Path(export.__file__).resolve()
                ),
                "importer_sha256": export.sha256_file(Path(__file__).resolve()),
                "python": sys.version.split()[0],
                "torch": str(torch.__version__),
            },
            "round_trip": round_trip,
            "source": source,
            "imported_from": imported_from,
        }
        receipt_payload = (
            json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
            + "\n"
        ).encode("ascii")
        with receipt_output.open("xb") as destination:
            destination.write(receipt_payload)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--import-receipt", type=Path, required=True)
    parser.add_argument(
        "--allow-lossy-round-trip",
        action="store_true",
        help=(
            "write the checkpoint even when the container does not return byte "
            "for byte; the receipt records the difference either way"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = import_container(args.network, args.output, args.import_receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    if not receipt["round_trip"]["byte_identical"] and not args.allow_lossy_round_trip:
        print(
            "ERROR: the container did not return byte for byte; see the receipt",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LegacyImportError, export.LegacyExportError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
