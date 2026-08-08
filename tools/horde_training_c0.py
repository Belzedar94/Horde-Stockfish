#!/usr/bin/env python3
"""Prove that splitting absolute G0 lanes is an implementation-only change."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Callable, Mapping, Sequence

try:
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - exercised by the CLI failure path
    raise SystemExit("PyTorch is required for the Horde NNUE C0 control") from error

try:
    from .horde_training_control import (
        DEFAULT_LEARNING_RATE,
        _clip_serialized_dense_weights,
        _make_optimizer,
    )
    from .horde_training_microfit import make_fixture_batch
    from .horde_training_models import C0SingleG0Model, C0SplitG0Model
except ImportError:
    from horde_training_control import (
        DEFAULT_LEARNING_RATE,
        _clip_serialized_dense_weights,
        _make_optimizer,
    )
    from horde_training_microfit import make_fixture_batch
    from horde_training_models import C0SingleG0Model, C0SplitG0Model


SCHEMA = "HORDE_V2_C0_EQUALITY_RECEIPT_V1"
EXPORT_SCHEMA = "HORDE_V2_C0_G0_256_INTEGER_PAYLOAD_V1"
MODEL_SEED = 0x56325F43305F4730
DEFAULT_STEPS = 4
FT_SCALE = 127 * 64
DENSE_SCALE = 64
ACTIVATION_SHIFT = 6
ACTIVATION_MAX = 127
OUTPUT_DIVISOR = 16
PARAMETER_BYTES = 371_016
CANONICAL_NAMES = (
    "global_weights",
    "global_bias",
    "hidden0_weights",
    "hidden0_bias",
    "hidden1_weights",
    "hidden1_bias",
    "output_weights",
    "output_bias",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_parts(model: nn.Module) -> dict[str, tuple[nn.Parameter, ...]]:
    if isinstance(model, C0SingleG0Model):
        global_weights = (model.global_weights,)
        global_bias = (model.global_bias,)
    elif isinstance(model, C0SplitG0Model):
        global_weights = (model.first_weights, model.second_weights)
        global_bias = (model.first_bias, model.second_bias)
    else:  # pragma: no cover - defensive API boundary
        raise TypeError(f"unsupported C0 model: {type(model).__name__}")
    return {
        "global_weights": global_weights,
        "global_bias": global_bias,
        **{name: (getattr(model, name),) for name in CANONICAL_NAMES[2:]},
    }


def _join(name: str, tensors: Sequence[Tensor]) -> Tensor:
    _require(bool(tensors), f"canonical tensor {name} has no parts")
    if len(tensors) == 1:
        return tensors[0]
    dimension = 1 if name == "global_weights" else 0
    return torch.cat(tuple(tensors), dim=dimension)


def _canonical_tensors(
    model: nn.Module,
    select: Callable[[nn.Parameter], Tensor],
) -> dict[str, Tensor]:
    return {
        name: _join(name, tuple(select(parameter) for parameter in parts))
        for name, parts in _canonical_parts(model).items()
    }


def _parameter_tensors(model: nn.Module) -> dict[str, Tensor]:
    return _canonical_tensors(model, lambda parameter: parameter.detach())


def _gradient_tensors(model: nn.Module) -> dict[str, Tensor]:
    def gradient(parameter: nn.Parameter) -> Tensor:
        _require(parameter.grad is not None, "C0 parameter did not receive a gradient")
        return parameter.grad.detach()

    return _canonical_tensors(model, gradient)


def _optimizer_tensors(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for name, parameters in _canonical_parts(model).items():
        states = [optimizer.state[parameter] for parameter in parameters]
        _require(all(state for state in states), f"optimizer state missing for {name}")
        keys = set(states[0])
        _require(
            all(set(state) == keys for state in states),
            f"optimizer state keys differ across {name} slices",
        )
        for key in sorted(keys):
            values = [state[key] for state in states]
            _require(
                all(isinstance(value, Tensor) for value in values),
                f"optimizer state {name}.{key} is not a tensor",
            )
            tensors = [value.detach() for value in values]
            if tensors[0].ndim == 0:
                _require(
                    all(torch.equal(tensors[0], value) for value in tensors[1:]),
                    f"optimizer scalar {name}.{key} differs across slices",
                )
                result[f"{name}.{key}"] = tensors[0]
            else:
                result[f"{name}.{key}"] = _join(name, tensors)
    return result


def _tensor_bytes(tensor: Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous().numpy()
    dtype = value.dtype.newbyteorder("<")
    return value.astype(dtype, copy=False).tobytes(order="C")


def _tensor_map_sha256(tensors: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        digest.update(struct.pack("<H", len(encoded)))
        digest.update(encoded)
        dtype = str(tensor.numpy().dtype).encode("ascii")
        digest.update(struct.pack("<H", len(dtype)))
        digest.update(dtype)
        digest.update(struct.pack("<H", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack("<Q", dimension))
        payload = _tensor_bytes(tensor)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest().upper()


def _assert_tensor_maps_equal(
    expected: Mapping[str, Tensor],
    actual: Mapping[str, Tensor],
    label: str,
) -> None:
    _require(expected.keys() == actual.keys(), f"{label} tensor names differ")
    for name in expected:
        _require(
            torch.equal(expected[name], actual[name]),
            f"{label} differs in canonical tensor {name}",
        )


def _optimizer_contract(optimizer: torch.optim.Optimizer) -> list[dict[str, object]]:
    return [
        {key: value for key, value in group.items() if key != "params"}
        for group in optimizer.param_groups
    ]


def _quantize(value: Tensor, scale: int, dtype: torch.dtype) -> Tensor:
    quantized = torch.round(value.detach().cpu() * scale).to(torch.int64)
    limits = {
        torch.int8: (-(1 << 7), (1 << 7) - 1),
        torch.int16: (-(1 << 15), (1 << 15) - 1),
        torch.int32: (-(1 << 31), (1 << 31) - 1),
    }
    minimum, maximum = limits[dtype]
    _require(
        bool(torch.all((quantized >= minimum) & (quantized <= maximum))),
        f"C0 quantization overflow for {dtype}",
    )
    return quantized.to(dtype)


def _quantized_canonical(model: nn.Module) -> dict[str, Tensor]:
    parameters = _parameter_tensors(model)
    return {
        "global_weights": _quantize(parameters["global_weights"], FT_SCALE, torch.int16),
        "global_bias": _quantize(parameters["global_bias"], FT_SCALE, torch.int32),
        "hidden0_weights": _quantize(
            parameters["hidden0_weights"], DENSE_SCALE, torch.int8
        ),
        "hidden0_bias": _quantize(parameters["hidden0_bias"], FT_SCALE, torch.int32),
        "hidden1_weights": _quantize(
            parameters["hidden1_weights"], DENSE_SCALE, torch.int8
        ),
        "hidden1_bias": _quantize(parameters["hidden1_bias"], FT_SCALE, torch.int32),
        "output_weights": _quantize(
            parameters["output_weights"], DENSE_SCALE, torch.int8
        ),
        "output_bias": _quantize(parameters["output_bias"], FT_SCALE, torch.int32),
    }


def _framed_payload(parameters: Mapping[str, Tensor]) -> tuple[bytes, int]:
    parameter_payload_bytes = sum(len(_tensor_bytes(parameters[name])) for name in CANONICAL_NAMES)
    _require(
        parameter_payload_bytes == PARAMETER_BYTES,
        f"C0 parameter payload is {parameter_payload_bytes} bytes instead of {PARAMETER_BYTES}",
    )
    output = bytearray(EXPORT_SCHEMA.encode("ascii") + b"\0")
    for name in CANONICAL_NAMES:
        encoded = name.encode("ascii")
        payload = _tensor_bytes(parameters[name])
        output.extend(struct.pack("<H", len(encoded)))
        output.extend(encoded)
        output.extend(struct.pack("<Q", len(payload)))
        output.extend(payload)
    return bytes(output), parameter_payload_bytes


def _sparse_affine(
    indices: Tensor,
    offsets: Tensor,
    weights: Tensor,
    bias: Tensor,
) -> Tensor:
    rows: list[Tensor] = []
    for row in range(offsets.numel() - 1):
        begin = int(offsets[row])
        end = int(offsets[row + 1])
        selected = indices[begin:end]
        value = bias.to(torch.int64)
        if selected.numel():
            value = value + weights.index_select(0, selected).to(torch.int64).sum(dim=0)
        rows.append(value)
    return torch.stack(rows)


def _activate(value: Tensor) -> Tensor:
    return torch.clamp(
        torch.bitwise_right_shift(torch.clamp_min(value, 0), ACTIVATION_SHIFT),
        0,
        ACTIVATION_MAX,
    )


def _integer_trace(model: nn.Module, batch: object) -> dict[str, Tensor]:
    indices = getattr(batch, "v2_global").detach().cpu()
    offsets = getattr(batch, "global_offsets").detach().cpu()
    side_to_move = getattr(batch, "side_to_move").detach().cpu()

    if isinstance(model, C0SingleG0Model):
        quantized = _quantized_canonical(model)
        accumulator = _sparse_affine(
            indices,
            offsets,
            quantized["global_weights"],
            quantized["global_bias"],
        )
    elif isinstance(model, C0SplitG0Model):
        first_weights = _quantize(model.first_weights, FT_SCALE, torch.int16)
        first_bias = _quantize(model.first_bias, FT_SCALE, torch.int32)
        second_weights = _quantize(model.second_weights, FT_SCALE, torch.int16)
        second_bias = _quantize(model.second_bias, FT_SCALE, torch.int32)
        accumulator = torch.cat(
            (
                _sparse_affine(indices, offsets, first_weights, first_bias),
                _sparse_affine(indices, offsets, second_weights, second_bias),
            ),
            dim=1,
        )
        quantized = _quantized_canonical(model)
    else:  # pragma: no cover - defensive API boundary
        raise TypeError(f"unsupported C0 model: {type(model).__name__}")

    transformed = _activate(accumulator)
    hidden0_affine = (
        transformed.to(torch.int64)
        @ quantized["hidden0_weights"].to(torch.int64).transpose(0, 1)
        + quantized["hidden0_bias"].to(torch.int64)
    )
    hidden0 = _activate(hidden0_affine)
    hidden1_affine = (
        hidden0
        @ quantized["hidden1_weights"].to(torch.int64).transpose(0, 1)
        + quantized["hidden1_bias"].to(torch.int64)
    )
    hidden1 = _activate(hidden1_affine)
    output_heads = (
        hidden1
        @ quantized["output_weights"].to(torch.int64).transpose(0, 1)
        + quantized["output_bias"].to(torch.int64)
    )
    output_affine = output_heads.gather(1, side_to_move.unsqueeze(1)).squeeze(1)
    pre_rule50 = torch.div(output_affine, OUTPUT_DIVISOR, rounding_mode="trunc")
    value = torch.clamp(pre_rule50, -31_506, 31_506)
    trace = {
        "accumulator": accumulator,
        "transformed": transformed,
        "hidden0_affine": hidden0_affine,
        "hidden0": hidden0,
        "hidden1_affine": hidden1_affine,
        "hidden1": hidden1,
        "output_affine": output_affine,
        "pre_rule50": pre_rule50,
        "value": value,
    }
    for name in ("accumulator", "hidden0_affine", "hidden1_affine", "output_affine"):
        tensor = trace[name]
        _require(
            bool(torch.all((tensor >= -(1 << 31)) & (tensor <= (1 << 31) - 1))),
            f"C0 integer {name} exceeds signed 32-bit bounds",
        )
    return trace


def _run_split(first_lanes: int, steps: int, learning_rate: float) -> dict[str, object]:
    batch, fixture = make_fixture_batch()
    single = C0SingleG0Model(MODEL_SEED)
    split = C0SplitG0Model(single, first_lanes)
    single_optimizer = _make_optimizer(single, learning_rate)
    split_optimizer = _make_optimizer(split, learning_rate)
    _require(
        _optimizer_contract(single_optimizer) == _optimizer_contract(split_optimizer),
        "C0 optimizer hyperparameters differ",
    )
    initial_parameters = {
        name: value.clone() for name, value in _parameter_tensors(single).items()
    }
    _assert_tensor_maps_equal(initial_parameters, _parameter_tensors(split), "initial state")
    initial_output = single(batch)
    _require(torch.equal(initial_output, split(batch)), "initial float forward differs")

    target = batch.scores / 600.0
    step_receipts: list[dict[str, object]] = []
    for step in range(steps):
        single_optimizer.zero_grad(set_to_none=True)
        split_optimizer.zero_grad(set_to_none=True)
        single_output = single(batch)
        split_output = split(batch)
        _require(torch.equal(single_output, split_output), f"float forward differs at step {step}")
        single_loss = torch.mean((single_output - target) ** 2)
        split_loss = torch.mean((split_output - target) ** 2)
        _require(torch.equal(single_loss, split_loss), f"loss differs at step {step}")
        single_loss.backward()
        split_loss.backward()

        single_gradients = _gradient_tensors(single)
        split_gradients = _gradient_tensors(split)
        _assert_tensor_maps_equal(single_gradients, split_gradients, f"gradient step {step}")
        single_optimizer.step()
        split_optimizer.step()
        _clip_serialized_dense_weights(single)
        _clip_serialized_dense_weights(split)

        single_parameters = _parameter_tensors(single)
        split_parameters = _parameter_tensors(split)
        _assert_tensor_maps_equal(single_parameters, split_parameters, f"parameter step {step}")
        single_optimizer_state = _optimizer_tensors(single, single_optimizer)
        split_optimizer_state = _optimizer_tensors(split, split_optimizer)
        _assert_tensor_maps_equal(
            single_optimizer_state,
            split_optimizer_state,
            f"optimizer step {step}",
        )
        step_receipts.append(
            {
                "step": step + 1,
                "loss": float(single_loss.detach()),
                "output_sha256": _tensor_map_sha256({"output": single_output}),
                "gradient_sha256": _tensor_map_sha256(single_gradients),
                "parameter_sha256": _tensor_map_sha256(single_parameters),
                "optimizer_sha256": _tensor_map_sha256(single_optimizer_state),
            }
        )

    single_quantized = _quantized_canonical(single)
    split_quantized = _quantized_canonical(split)
    _assert_tensor_maps_equal(single_quantized, split_quantized, "quantized export")
    single_payload, parameter_bytes = _framed_payload(single_quantized)
    split_payload, split_parameter_bytes = _framed_payload(split_quantized)
    _require(single_payload == split_payload, "reassembled integer payload differs")
    _require(parameter_bytes == split_parameter_bytes, "integer parameter byte counts differ")

    single_trace = _integer_trace(single, batch)
    split_trace = _integer_trace(split, batch)
    _assert_tensor_maps_equal(single_trace, split_trace, "integer forward")
    return {
        "name": f"G0_SPLIT_{first_lanes}_{256 - first_lanes}",
        "first_lanes": first_lanes,
        "second_lanes": 256 - first_lanes,
        "fixture_sha256": fixture["sha256"],
        "sample_count": fixture["sample_count"],
        "initial_parameter_sha256": _tensor_map_sha256(initial_parameters),
        "initial_output_sha256": _tensor_map_sha256({"output": initial_output}),
        "steps": step_receipts,
        "final_parameter_sha256": _tensor_map_sha256(_parameter_tensors(single)),
        "final_optimizer_sha256": _tensor_map_sha256(
            _optimizer_tensors(single, single_optimizer)
        ),
        "integer_export": {
            "schema": EXPORT_SCHEMA,
            "parameter_bytes": parameter_bytes,
            "framed_payload_bytes": len(single_payload),
            "payload_sha256": hashlib.sha256(single_payload).hexdigest().upper(),
            "trace_sha256": _tensor_map_sha256(single_trace),
            "value_sha256": _tensor_map_sha256({"value": single_trace["value"]}),
            "maximum_absolute_affine": {
                name: int(torch.max(torch.abs(single_trace[name])).item())
                for name in (
                    "accumulator",
                    "hidden0_affine",
                    "hidden1_affine",
                    "output_affine",
                )
            },
            "values": single_trace["value"].tolist(),
        },
        "equalities": {
            "float_forward": True,
            "gradients_after_reassembly": True,
            "optimizer_state_after_reassembly": True,
            "parameters_after_reassembly": True,
            "integer_payload_after_reassembly": True,
            "integer_layer_trace": True,
        },
    }


def build_receipt(
    steps: int = DEFAULT_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> dict[str, object]:
    _require(steps > 0, "C0 step count must be positive")
    _require(learning_rate > 0.0, "C0 learning rate must be positive")
    torch.manual_seed(MODEL_SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        _require(torch.get_num_interop_threads() == 1, "PyTorch interop threads are not one")
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False

    variants = [_run_split(width, steps, learning_rate) for width in (64, 128)]
    for field in (
        "initial_parameter_sha256",
        "initial_output_sha256",
        "final_parameter_sha256",
        "final_optimizer_sha256",
    ):
        _require(
            len({variant[field] for variant in variants}) == 1,
            f"C0 split widths disagree in {field}",
        )
    _require(variants[0]["steps"] == variants[1]["steps"], "C0 split step receipts differ")
    for field in ("payload_sha256", "trace_sha256", "value_sha256"):
        _require(
            len({variant["integer_export"][field] for variant in variants}) == 1,
            f"C0 split widths disagree in integer {field}",
        )

    common_fields = (
        "fixture_sha256",
        "sample_count",
        "initial_parameter_sha256",
        "initial_output_sha256",
        "steps",
        "final_parameter_sha256",
        "final_optimizer_sha256",
        "integer_export",
    )
    common = {field: variants[0][field] for field in common_fields}
    for variant in variants:
        for field in common_fields:
            _require(variant[field] == common[field], f"C0 split widths disagree in {field}")
            del variant[field]

    implementation = Path(__file__).resolve()
    models = implementation.with_name("horde_training_models.py")
    return {
        "schema": SCHEMA,
        "purpose": "engineering equality control; no architecture or strength claim",
        "control": {
            "single": "G0_SINGLE_256",
            "splits": [variant["name"] for variant in variants],
            "feature_content": "the same complete absolute G0 rows in every domain",
            "model_seed": MODEL_SEED,
            "steps": steps,
            "optimizer": "torch.optim.RAdam",
            "learning_rate": learning_rate,
        },
        "integer_contract": {
            "schema": EXPORT_SCHEMA,
            "byte_order": "little-endian",
            "types": {
                "feature_weights": "signed int16",
                "feature_and_dense_biases": "signed int32",
                "dense_weights": "signed int8",
                "affine_sums": "signed int64 engineering oracle",
            },
            "rounding": "round to nearest, ties to even via torch.round",
            "ft_scale": FT_SCALE,
            "dense_scale": DENSE_SCALE,
            "activation_shift": ACTIVATION_SHIFT,
            "activation_max": ACTIVATION_MAX,
            "output_divisor": OUTPUT_DIVISOR,
            "rule50_count": 0,
            "parameter_bytes": PARAMETER_BYTES,
            "production_schema": False,
        },
        "implementation": {
            "control_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest().upper(),
            "models_sha256": hashlib.sha256(models.read_bytes()).hexdigest().upper(),
        },
        "common": common,
        "variants": variants,
        "cross_split_exact": True,
        "strength_evidence": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    receipt = build_receipt(args.steps, args.learning_rate)
    payload = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        resolved = args.output.expanduser().resolve()
        _require(resolved.parent.is_dir(), f"output parent does not exist: {resolved.parent}")
        with resolved.open("x", encoding="utf-8", newline="\n") as output:
            output.write(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
