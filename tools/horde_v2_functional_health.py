#!/usr/bin/env python3
"""Measure whether a trained Horde V2 checkpoint still carries feature signal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as functional

try:
    from .horde_training_control import (
        V2_ARCHITECTURES,
        _make_model,
        _rule50_postprocess,
        _torch_v2_batch,
    )
    from .horde_training_decoder import HordeBinV1Dataset, make_sparse_batch
    from .horde_training_models import NNUE_TO_SCORE, _sparse_sum
    from .horde_training_selected_role import SelectedRoleDataset
except ImportError:
    from horde_training_control import (
        V2_ARCHITECTURES,
        _make_model,
        _rule50_postprocess,
        _torch_v2_batch,
    )
    from horde_training_decoder import HordeBinV1Dataset, make_sparse_batch
    from horde_training_models import NNUE_TO_SCORE, _sparse_sum
    from horde_training_selected_role import SelectedRoleDataset


SCHEMA = "HORDE_V2_FUNCTIONAL_HEALTH_RECEIPT_V1"
CONTRACT_SCHEMA = "HORDE_V2_FUNCTIONAL_HEALTH_V1"
CONTRACT_RELATIVE_PATH = Path("schemas/horde-v2-functional-health-v1.json")
CONTRACT_SHA256 = "9BC9F5E90EEA9FD0E34CECCA6DA09E6DB19AF05426B9AEE993BFDA8102F1689E"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WHITE = 0
BLACK = 1
SIDE_NAMES = {WHITE: "white_to_move", BLACK: "black_to_move"}


class FunctionalHealthError(ValueError):
    """Raised when a functional-health input or receipt violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FunctionalHealthError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


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


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"{label} does not exist: {resolved}")
    payload = resolved.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FunctionalHealthError(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(value, dict), f"{label} root is not an object")
    return value, payload


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
    _require(
        len(commit) == 40 and all(character in "0123456789abcdef" for character in commit),
        "functional-health source is not a full Git identity",
    )
    return {"commit": commit, "dirty": dirty}


def load_contract(path: Path | None = None) -> tuple[dict[str, Any], str]:
    resolved = (path or REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH).expanduser().resolve()
    contract, payload = _read_json(resolved, "functional-health contract")
    digest = _sha256_bytes(payload)
    _require(digest == CONTRACT_SHA256, f"functional-health contract SHA-256 mismatch: {digest}")
    _require(contract.get("schema_name") == CONTRACT_SCHEMA, "functional-health schema drifted")

    evaluation = _mapping(contract.get("evaluation"), "functional-health evaluation")
    probe = _mapping(evaluation.get("probe"), "functional-health probe")
    tolerances = _mapping(
        evaluation.get("numeric_tolerances"), "functional-health numeric tolerances"
    )
    gates = _mapping(contract.get("gates"), "functional-health gates")
    _require(probe.get("records") == 4096, "functional-health probe size drifted")
    _require(
        tolerances.get("lane_variance_epsilon") == 1.0e-12
        and tolerances.get("jacobian_absolute_epsilon") == 1.0e-12,
        "functional-health numeric tolerance drifted",
    )
    _require(
        gates.get("minimum_side_records") == 32
        and gates.get("maximum_constant_lane_fraction") == 0.75
        and gates.get("minimum_position_dependent_lane_fraction") == 0.10
        and gates.get("minimum_hidden_interior_sample_fraction") == 0.01
        and gates.get("minimum_within_side_unique_pre_rule50_integer_scores") == 2
        and gates.get("minimum_feature_to_score_jacobian_nonzero_fraction") == 0.001,
        "functional-health gate threshold drifted",
    )
    return contract, digest


def deterministic_probe_indices(record_count: int, probe_count: int) -> tuple[int, ...]:
    """Select one midpoint from every equal-width interval in record order."""

    _require(record_count > 0, "functional-health dataset is empty")
    _require(0 < probe_count <= record_count, "functional-health probe size is invalid")
    indices = tuple(((2 * index + 1) * record_count) // (2 * probe_count) for index in range(probe_count))
    _require(len(set(indices)) == probe_count, "functional-health probe indices are not unique")
    _require(indices[0] >= 0 and indices[-1] < record_count, "functional-health probe escapes dataset")
    return indices


def _indices_sha256(indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(indices)))
    for index in indices:
        digest.update(struct.pack("<Q", index))
    return digest.hexdigest().upper()


def _standard_dataset_identity(dataset: HordeBinV1Dataset, path: Path) -> dict[str, object]:
    manifest = dataset.manifest
    generation = _mapping(manifest.get("generation"), "validation generation")
    return {
        "name": path.name,
        "sha256": dataset.file_sha256,
        "payload_sha256": manifest["payload_sha256"],
        "records": len(dataset),
        "book_sha256": manifest["book_sha256"],
        "seed": generation["seed"],
    }


def _validate_dataset_binding(
    checkpoint: Mapping[str, Any], observed: Mapping[str, object]
) -> None:
    data = _mapping(checkpoint.get("data"), "checkpoint data")
    expected = _mapping(data.get("validation_file"), "checkpoint validation identity")
    _require(dict(expected) == dict(observed), "probe dataset does not match checkpoint validation identity")


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"checkpoint does not exist: {resolved}")
    try:
        checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
    except (EOFError, RuntimeError, ValueError) as error:
        raise FunctionalHealthError(f"cannot load checkpoint: {error}") from error
    _require(isinstance(checkpoint, dict), "checkpoint root is not an object")
    return checkpoint, _sha256_file(resolved)


def _first_domain(
    model: nn.Module, architecture: str, batch: object
) -> tuple[Tensor, Tensor, str]:
    first_domain = str(V2_ARCHITECTURES[architecture]["first_domain"])
    if first_domain == "absolute_nonking":
        indices, offsets = model.absolute_nonking_features(batch)
        weights = model.absolute_nonking_weights
        bias = model.absolute_nonking_bias
    elif first_domain == "royal_rank8":
        indices, offsets = model.royal_rank8_features(batch)
        weights = model.royal_rank8_weights
        bias = model.royal_rank8_bias
    elif first_domain == "royal":
        indices = batch.v2_royal
        offsets = batch.royal_offsets
        weights = model.royal_weights
        bias = model.royal_bias
    else:
        raise FunctionalHealthError(f"unsupported first domain: {first_domain}")
    return _sparse_sum(indices, offsets, weights, bias), offsets, first_domain


def layer_metrics(
    preactivation: Tensor,
    activation: Tensor,
    variance_epsilon: float,
) -> dict[str, object]:
    pre = preactivation.detach().cpu().numpy().astype(np.float64, copy=False)
    active = activation.detach().cpu().numpy().astype(np.float64, copy=False)
    _require(pre.ndim == 2 and active.shape == pre.shape, "activation tensor shape mismatch")
    records, lanes = pre.shape
    _require(records > 0 and lanes > 0, "activation tensor is empty")
    _require(bool(np.isfinite(pre).all()), "preactivation tensor is non-finite")
    _require(bool(np.isfinite(active).all()), "activation tensor is non-finite")

    variances = np.var(active, axis=0)
    pre_means = np.mean(pre, axis=0)
    pre_stds = np.std(pre, axis=0)
    pre_minimums = np.min(pre, axis=0)
    pre_maximums = np.max(pre, axis=0)
    below = np.mean(pre <= 0.0, axis=0)
    interior = np.mean((pre > 0.0) & (pre < 1.0), axis=0)
    above = np.mean(pre >= 1.0, axis=0)
    constant = variances <= variance_epsilon
    position_dependent = ~constant

    per_lane = [
        {
            "lane": lane,
            "pre_mean": float(pre_means[lane]),
            "pre_stddev": float(pre_stds[lane]),
            "pre_minimum": float(pre_minimums[lane]),
            "pre_maximum": float(pre_maximums[lane]),
            "below_or_equal_zero_fraction": float(below[lane]),
            "interior_fraction": float(interior[lane]),
            "above_or_equal_one_fraction": float(above[lane]),
            "activation_variance": float(variances[lane]),
            "position_dependent": bool(position_dependent[lane]),
        }
        for lane in range(lanes)
    ]
    return {
        "records": records,
        "lanes": lanes,
        "preactivation_minimum": float(np.min(pre)),
        "preactivation_maximum": float(np.max(pre)),
        "below_or_equal_zero_fraction": float(np.mean(pre <= 0.0)),
        "interior_sample_fraction": float(np.mean((pre > 0.0) & (pre < 1.0))),
        "above_or_equal_one_fraction": float(np.mean(pre >= 1.0)),
        "constant_lanes": int(np.count_nonzero(constant)),
        "constant_lane_fraction": float(np.mean(constant)),
        "position_dependent_lanes": int(np.count_nonzero(position_dependent)),
        "position_dependent_lane_fraction": float(np.mean(position_dependent)),
        "minimum_activation_variance": float(np.min(variances)),
        "median_activation_variance": float(np.median(variances)),
        "maximum_activation_variance": float(np.max(variances)),
        "per_lane": per_lane,
    }


def _dense_forward(model: nn.Module, transformed: Tensor, side_to_move: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    hidden0_pre = functional.linear(transformed, model.hidden0_weights, model.hidden0_bias)
    hidden0 = torch.clamp(hidden0_pre, 0.0, 1.0)
    hidden1_pre = functional.linear(hidden0, model.hidden1_weights, model.hidden1_bias)
    hidden1 = torch.clamp(hidden1_pre, 0.0, 1.0)
    all_heads = functional.linear(hidden1, model.output_weights, model.output_bias)
    output = all_heads.gather(1, side_to_move.unsqueeze(1)).squeeze(1)
    return hidden0_pre, hidden0, hidden1_pre, hidden1, output


def _score_metrics(pre_rule50: Tensor, post_rule50: Tensor, sides: Tensor) -> dict[str, object]:
    pre = pre_rule50.detach().cpu().numpy().astype(np.float64, copy=False)
    post = post_rule50.detach().cpu().numpy().astype(np.float64, copy=False)
    side_values = sides.detach().cpu().numpy().astype(np.int8, copy=False)
    _require(pre.ndim == 1 and post.shape == pre.shape, "score tensor shape mismatch")
    receipt: dict[str, object] = {}
    for side, name in SIDE_NAMES.items():
        selected = side_values == side
        _require(bool(np.any(selected)), f"functional-health probe has no {name} records")
        values = pre[selected]
        damped = post[selected]
        receipt[name] = {
            "records": int(np.count_nonzero(selected)),
            "pre_rule50_mean_cp": float(np.mean(values)),
            "pre_rule50_stddev_cp": float(np.std(values)),
            "pre_rule50_minimum_cp": float(np.min(values)),
            "pre_rule50_maximum_cp": float(np.max(values)),
            "unique_pre_rule50_integer_scores": int(np.unique(np.trunc(values)).size),
            "unique_post_rule50_integer_scores": int(np.unique(np.trunc(damped)).size),
        }
    return receipt


def _difference_metrics(candidate: Tensor, baseline: Tensor) -> dict[str, object]:
    delta = (candidate - baseline).detach().cpu().numpy().astype(np.float64, copy=False)
    candidate_values = candidate.detach().cpu().numpy()
    baseline_values = baseline.detach().cpu().numpy()
    absolute = np.abs(delta)
    return {
        "mean_absolute_cp": float(np.mean(absolute)),
        "maximum_absolute_cp": float(np.max(absolute)),
        "changed_float_fraction": float(np.mean(delta != 0.0)),
        "changed_integer_fraction": float(
            np.mean(np.trunc(candidate_values) != np.trunc(baseline_values))
        ),
    }


def _group_rotation(side_to_move: Tensor, rule50_count: Tensor) -> Tensor:
    sides = side_to_move.detach().cpu().numpy()
    clocks = rule50_count.detach().cpu().numpy()
    permutation = np.arange(len(sides), dtype=np.int64)
    for side in (WHITE, BLACK):
        for clock in np.unique(clocks[sides == side]):
            indices = np.flatnonzero((sides == side) & (clocks == clock))
            if len(indices) > 1:
                permutation[indices] = np.roll(indices, 1)
    return torch.tensor(permutation, dtype=torch.long, device=side_to_move.device)


def _jacobian_metrics(gradient: Tensor, epsilon: float) -> dict[str, object]:
    values = gradient.detach().cpu().numpy().astype(np.float64, copy=False)
    _require(bool(np.isfinite(values).all()), "feature-to-score Jacobian is non-finite")
    absolute = np.abs(values)
    return {
        "elements": int(values.size),
        "nonzero_fraction": float(np.mean(absolute > epsilon)),
        "mean_absolute_cp_per_activation": float(np.mean(absolute)),
        "rms_cp_per_activation": float(math.sqrt(float(np.mean(values * values)))),
        "maximum_absolute_cp_per_activation": float(np.max(absolute)),
    }


def _gate_receipt(
    contract: Mapping[str, Any],
    layers: Mapping[str, Mapping[str, object]],
    scores: Mapping[str, Mapping[str, object]],
    jacobian: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    gates = _mapping(contract.get("gates"), "functional-health gates")
    max_constant = float(gates["maximum_constant_lane_fraction"])
    min_dependent = float(gates["minimum_position_dependent_lane_fraction"])
    min_interior = float(gates["minimum_hidden_interior_sample_fraction"])
    min_side_records = int(gates["minimum_side_records"])
    min_unique = int(gates["minimum_within_side_unique_pre_rule50_integer_scores"])
    min_jacobian = float(gates["minimum_feature_to_score_jacobian_nonzero_fraction"])

    checks: dict[str, bool] = {}
    for name in ("hidden0", "hidden1"):
        layer = layers[name]
        checks[f"{name}_constant_lane_fraction"] = (
            float(layer["constant_lane_fraction"]) <= max_constant
        )
        checks[f"{name}_position_dependent_lane_fraction"] = (
            float(layer["position_dependent_lane_fraction"]) >= min_dependent
        )
        checks[f"{name}_interior_sample_fraction"] = (
            float(layer["interior_sample_fraction"]) >= min_interior
        )
    for side_name in SIDE_NAMES.values():
        side = scores[side_name]
        checks[f"{side_name}_support"] = int(side["records"]) >= min_side_records
        checks[f"{side_name}_integer_diversity"] = (
            int(side["unique_pre_rule50_integer_scores"]) >= min_unique
        )
    for domain in ("first_domain", "global"):
        checks[f"{domain}_jacobian"] = (
            float(jacobian[domain]["nonzero_fraction"]) >= min_jacobian
        )
    return {
        "thresholds": dict(gates),
        "checks": checks,
        "passed": all(checks.values()),
    }


def analyze(
    checkpoint_path: Path,
    validation_path: Path,
    *,
    selected_role: bool,
    contract_path: Path | None = None,
    probe_records: int | None = None,
    allow_dirty: bool = False,
) -> dict[str, object]:
    contract, contract_sha256 = load_contract(contract_path)
    source = _repository_identity(REPOSITORY_ROOT)
    _require(allow_dirty or not source["dirty"], "functional-health source tree is dirty")
    checkpoint, checkpoint_sha256 = _load_checkpoint(checkpoint_path)
    settings = _mapping(checkpoint.get("settings"), "checkpoint settings")
    architecture_receipt = _mapping(settings.get("architecture"), "checkpoint architecture")
    architecture = str(architecture_receipt.get("name"))
    _require(architecture in V2_ARCHITECTURES, f"unsupported checkpoint architecture: {architecture}")
    supported = _mapping(contract.get("evaluation"), "functional-health evaluation").get(
        "supported_architectures"
    )
    _require(isinstance(supported, list) and architecture in supported, "architecture is outside contract")
    seed = settings.get("seed")
    _require(type(seed) is int and seed > 0, "checkpoint seed is invalid")

    validation_resolved = validation_path.expanduser().resolve()
    dataset_type = SelectedRoleDataset if selected_role else HordeBinV1Dataset
    with dataset_type(validation_resolved) as dataset:
        observed_identity = (
            dataset.identity()
            if selected_role
            else _standard_dataset_identity(dataset, validation_resolved)
        )
        _validate_dataset_binding(checkpoint, observed_identity)
        default_records = int(
            _mapping(
                _mapping(contract.get("evaluation"), "functional-health evaluation").get("probe"),
                "functional-health probe",
            )["records"]
        )
        count = probe_records or default_records
        _require(count == default_records, "production functional-health probe size drifted")
        indices = deterministic_probe_indices(len(dataset), count)
        sparse = make_sparse_batch(tuple(dataset.record(index) for index in indices))

    model = _make_model(architecture, seed)
    state = checkpoint.get("model_state")
    _require(isinstance(state, dict), "checkpoint model state is missing")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise FunctionalHealthError(f"checkpoint model state is incompatible: {error}") from error
    model.eval()
    batch = _torch_v2_batch(sparse, torch.device("cpu"))
    tolerances = _mapping(
        _mapping(contract.get("evaluation"), "functional-health evaluation").get(
            "numeric_tolerances"
        ),
        "functional-health numeric tolerances",
    )
    variance_epsilon = float(tolerances["lane_variance_epsilon"])
    jacobian_epsilon = float(tolerances["jacobian_absolute_epsilon"])

    with torch.no_grad():
        first_pre, _, first_domain = _first_domain(model, architecture, batch)
        global_pre = _sparse_sum(
            batch.v2_global,
            batch.global_offsets,
            model.global_weights,
            model.global_bias,
        )
        first = torch.clamp(first_pre, 0.0, 1.0)
        global_ = torch.clamp(global_pre, 0.0, 1.0)
        transformed = torch.cat((first, global_), dim=1)
        hidden0_pre, hidden0, hidden1_pre, hidden1, output = _dense_forward(
            model, transformed, batch.side_to_move
        )
        pre_rule50 = output * NNUE_TO_SCORE
        post_rule50 = _rule50_postprocess(output, batch.rule50_count)

        permutation = _group_rotation(batch.side_to_move, batch.rule50_count)
        interventions = {
            "zero_first_domain": _difference_metrics(
                _dense_forward(
                    model,
                    torch.cat((torch.zeros_like(first), global_), dim=1),
                    batch.side_to_move,
                )[-1]
                * NNUE_TO_SCORE,
                pre_rule50,
            ),
            "permute_first_domain": _difference_metrics(
                _dense_forward(
                    model,
                    torch.cat((first.index_select(0, permutation), global_), dim=1),
                    batch.side_to_move,
                )[-1]
                * NNUE_TO_SCORE,
                pre_rule50,
            ),
            "zero_global": _difference_metrics(
                _dense_forward(
                    model,
                    torch.cat((first, torch.zeros_like(global_)), dim=1),
                    batch.side_to_move,
                )[-1]
                * NNUE_TO_SCORE,
                pre_rule50,
            ),
            "permute_global": _difference_metrics(
                _dense_forward(
                    model,
                    torch.cat((first, global_.index_select(0, permutation)), dim=1),
                    batch.side_to_move,
                )[-1]
                * NNUE_TO_SCORE,
                pre_rule50,
            ),
        }

    transformed_gradient = transformed.detach().requires_grad_(True)
    gradient_output = _dense_forward(model, transformed_gradient, batch.side_to_move)[-1]
    gradient = torch.autograd.grad(
        (gradient_output * NNUE_TO_SCORE).sum(), transformed_gradient
    )[0]
    first_lanes = first.shape[1]
    jacobian = {
        "first_domain": _jacobian_metrics(gradient[:, :first_lanes], jacobian_epsilon),
        "global": _jacobian_metrics(gradient[:, first_lanes:], jacobian_epsilon),
    }
    layers = {
        "first_domain": layer_metrics(first_pre, first, variance_epsilon),
        "global": layer_metrics(global_pre, global_, variance_epsilon),
        "hidden0": layer_metrics(hidden0_pre, hidden0, variance_epsilon),
        "hidden1": layer_metrics(hidden1_pre, hidden1, variance_epsilon),
    }
    scores = _score_metrics(pre_rule50, post_rule50, batch.side_to_move)
    gates = _gate_receipt(contract, layers, scores, jacobian)
    progress = _mapping(checkpoint.get("progress"), "checkpoint progress")

    return {
        "schema": SCHEMA,
        "contract": {"schema": CONTRACT_SCHEMA, "sha256": contract_sha256},
        "source": source,
        "checkpoint": {
            "name": checkpoint_path.expanduser().resolve().name,
            "sha256": checkpoint_sha256,
            "source": checkpoint.get("source"),
            "architecture": architecture,
            "architecture_schema": architecture_receipt.get("schema"),
            "seed": seed,
            "optimizer_steps": progress.get("optimizer_steps"),
            "samples_consumed": progress.get("samples_consumed"),
        },
        "validation": observed_identity,
        "probe": {
            "selection": "deterministic midpoint stratification",
            "records": len(indices),
            "indices_sha256": _indices_sha256(indices),
            "first_indices": list(indices[:8]),
            "last_indices": list(indices[-8:]),
            "side_counts": {
                name: int(torch.count_nonzero(batch.side_to_move == side).item())
                for side, name in SIDE_NAMES.items()
            },
        },
        "analysis": {
            "first_domain": first_domain,
            "layers": layers,
            "scores": scores,
            "feature_to_score_jacobian": jacobian,
            "interventions": interventions,
        },
        "gates": gates,
        "claims": {
            "functional_health_passed": bool(gates["passed"]),
            "parameter_health_replaced": False,
            "validation_loss_selects_architecture": False,
            "strength_evidence": False,
            "architecture_selected": False,
            "production_network": False,
            "run6b_production_path_changed": False,
        },
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--validation-selected-role", action="store_true")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--probe-records", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_arguments()
    receipt = analyze(
        args.checkpoint,
        args.validation,
        selected_role=args.validation_selected_role,
        contract_path=args.contract,
        probe_records=args.probe_records,
        allow_dirty=args.allow_dirty,
    )
    _write_exclusive(args.output, _canonical_json(receipt))
    print(
        "Horde V2 functional health",
        f"architecture={receipt['checkpoint']['architecture']}",
        f"records={receipt['probe']['records']}",
        f"passed={str(receipt['gates']['passed']).lower()}",
        f"receipt={args.output.expanduser().resolve()}",
    )
    return 2 if args.require_pass and not receipt["gates"]["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
