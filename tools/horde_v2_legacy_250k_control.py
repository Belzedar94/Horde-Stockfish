#!/usr/bin/env python3
"""Qualify the matched 250k legacy H/P control against Horde V2 Rank8.

The frozen C3 receipt already authenticates the three Rank8 runs and
their untouched confirmation-role predictions.  This tool adds the missing
three fresh legacy H/P runs, verifies that they used the same records, recipe,
and paired seeds, evaluates them on the same confirmation role, and emits one
canonical comparison receipt.  Loss direction is diagnostic only: an
equal-time playing-strength gate is still required before selecting V2.
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import nn

try:
    from .horde_training_control import (
        ARCHITECTURE_SCHEMA,
        CHECKPOINT_SCHEMA,
        LEGACY_ARCHITECTURE,
        SCHEMA as LEGACY_TRAINING_SCHEMA,
        _make_model,
        _model_batch,
        _rule50_postprocess,
    )
    from .horde_training_selected_role import SelectedRoleDataset
    from .horde_v2_c2_objective import (
        LOOKUP_SCORE_MAXIMUM,
        LOOKUP_SCORE_MINIMUM,
        build_wdl_lookup,
        float_from_receipt,
        float_receipt,
    )
    from .horde_v2_c2_qualification import evaluate_prediction_scores
    from .horde_v2_c3_confirmation_role import ConfirmationRoleDataset
    from .horde_v2_c3_qualification import (
        SCHEMA as C3_RECEIPT_SCHEMA,
        validate_receipt as validate_c3_receipt,
    )
    from .horde_wdl import load_artifact
except ImportError:
    from horde_training_control import (
        ARCHITECTURE_SCHEMA,
        CHECKPOINT_SCHEMA,
        LEGACY_ARCHITECTURE,
        SCHEMA as LEGACY_TRAINING_SCHEMA,
        _make_model,
        _model_batch,
        _rule50_postprocess,
    )
    from horde_training_selected_role import SelectedRoleDataset
    from horde_v2_c2_objective import (
        LOOKUP_SCORE_MAXIMUM,
        LOOKUP_SCORE_MINIMUM,
        build_wdl_lookup,
        float_from_receipt,
        float_receipt,
    )
    from horde_v2_c2_qualification import evaluate_prediction_scores
    from horde_v2_c3_confirmation_role import ConfirmationRoleDataset
    from horde_v2_c3_qualification import (
        SCHEMA as C3_RECEIPT_SCHEMA,
        validate_receipt as validate_c3_receipt,
    )
    from horde_wdl import load_artifact


SCHEMA = "HORDE_V2_LEGACY_250K_CONTROL_RECEIPT_V1"
RANK8_ARCHITECTURE = "v2-c1-rank8-64x192"
LEGACY_TRAINING_COMMIT = "7c2ab02dbd77d8707b49b1c7038d7d31b869bf94"
EXPECTED_C3_RECEIPT_SHA256 = (
    "0CE7800C1DF65C2395E8507AE65D475C1FD5A088B0F500E769F7D3CF02F6D0C3"
)
EXPECTED_SEEDS = (
    7435908571601354096,
    3557647045056828427,
    4999335725889688378,
)
EXPECTED_RECIPE = {
    "epochs": 8,
    "batch_size": 4096,
    "block_size": 65_536,
    "lambda": 0.6,
    "learning_rate": 0.0015,
    "scheduler_gamma": 0.987,
    "device_type": "cuda",
    "cpu_threads": 1,
    "optimizer_steps": 496,
    "samples_consumed": 2_000_000,
    "dense_learning_rate_multiplier": 0.1,
    "output_learning_rate_multiplier": 0.1,
}
EVALUATION_BATCH_SIZE = 4096


class LegacyControlError(ValueError):
    """Raised when the matched legacy-control evidence is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LegacyControlError(message)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest().upper()


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"{label} does not exist: {resolved}")
    payload = resolved.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LegacyControlError(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(value, dict), f"{label} root is not an object")
    _require(payload == _canonical_json(value), f"{label} is not canonical JSON")
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


def _repository_identity() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]

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

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
        "path": Path(__file__).resolve().relative_to(root).as_posix(),
        "file_sha256": _sha256_file(Path(__file__).resolve()),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
    }


def _identity_matches(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Compare the stable data fields while allowing a rematerialized receipt."""

    required = ("sha256", "payload_sha256", "records")
    optional = ("book_sha256", "seed")
    return all(
        field in observed and field in expected and observed[field] == expected[field]
        for field in required
    ) and all(field not in expected or observed.get(field) == expected[field] for field in optional)


@dataclass(slots=True)
class LegacyRun:
    directory: Path
    seed: int
    checkpoint_sha256: str
    receipt_sha256: str
    stop_validation_loss: float
    model: nn.Module


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise LegacyControlError(f"cannot load legacy checkpoint {path}: {error}") from error
    _require(isinstance(value, dict), f"legacy checkpoint root is invalid: {path}")
    return value


def _prepare_legacy_run(
    directory: Path,
    *,
    expected_training: Mapping[str, Any],
    expected_validation: Mapping[str, Any],
    expected_teacher: Mapping[str, Any],
    expected_wdl: Mapping[str, Any],
) -> LegacyRun:
    resolved = directory.expanduser().resolve()
    _require(resolved.is_dir(), f"legacy run directory does not exist: {resolved}")
    checkpoint_path = resolved / "checkpoint.pt"
    receipt_path = resolved / "receipt.json"
    _require(checkpoint_path.is_file(), f"legacy checkpoint is missing: {checkpoint_path}")
    receipt, receipt_payload = _read_json(receipt_path, "legacy training receipt")
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)

    source = {"commit": LEGACY_TRAINING_COMMIT, "dirty": False}
    _require(
        checkpoint.get("schema") == CHECKPOINT_SCHEMA
        and checkpoint.get("architecture") == ARCHITECTURE_SCHEMA
        and checkpoint.get("source") == source,
        f"legacy checkpoint schema/source drifted: {resolved}",
    )
    settings = _mapping(checkpoint.get("settings"), "legacy checkpoint settings")
    seed = settings.get("seed")
    _require(type(seed) is int and seed in EXPECTED_SEEDS, f"legacy seed is invalid: {resolved}")
    expected_settings = {
        "epochs": EXPECTED_RECIPE["epochs"],
        "batch_size": EXPECTED_RECIPE["batch_size"],
        "block_size": EXPECTED_RECIPE["block_size"],
        "lambda": EXPECTED_RECIPE["lambda"],
        "learning_rate": EXPECTED_RECIPE["learning_rate"],
        "scheduler_gamma": EXPECTED_RECIPE["scheduler_gamma"],
        "device_type": EXPECTED_RECIPE["device_type"],
        "cpu_threads": EXPECTED_RECIPE["cpu_threads"],
        "initialization": "SHA256_NAMED_PARAMETER_SEED_V1",
        "optimizer_learning_rate_multipliers": {
            "dense_trunk": EXPECTED_RECIPE["dense_learning_rate_multiplier"],
            "output": EXPECTED_RECIPE["output_learning_rate_multiplier"],
        },
    }
    _require(
        all(settings.get(key) == value for key, value in expected_settings.items())
        and settings.get("wdl_calibration_sha256") == expected_wdl.get("sha256"),
        f"legacy checkpoint recipe drifted: {resolved}",
    )
    environment = _mapping(checkpoint.get("environment"), "legacy checkpoint environment")
    device = _mapping(environment.get("device"), "legacy checkpoint device")
    _require(
        device.get("type") == "cuda"
        and device.get("name") == "NVIDIA GeForce RTX 3080"
        and device.get("cpu_threads") == 1
        and environment.get("amp") is False
        and environment.get("cuda_matmul_allow_tf32") is False
        and environment.get("cudnn_allow_tf32") is False,
        f"legacy checkpoint environment drifted: {resolved}",
    )
    data = _mapping(checkpoint.get("data"), "legacy checkpoint data")
    observed_training = _mapping(data.get("train_file"), "legacy checkpoint training file")
    observed_validation = _mapping(
        data.get("validation_file"), "legacy checkpoint validation file"
    )
    _require(
        _identity_matches(observed_training, expected_training)
        and _identity_matches(observed_validation, expected_validation)
        and data.get("teacher") == dict(expected_teacher),
        f"legacy checkpoint data identity drifted: {resolved}",
    )
    checkpoint_wdl = _mapping(data.get("wdl_calibration"), "legacy checkpoint WDL")
    _require(
        all(checkpoint_wdl.get(key) == expected_wdl.get(key) for key in (
            "sha256",
            "schema",
            "link_schema",
            "selection_sha256",
            "eligible_records_sha256",
        )),
        f"legacy checkpoint WDL identity drifted: {resolved}",
    )
    progress = _mapping(checkpoint.get("progress"), "legacy checkpoint progress")
    _require(
        progress.get("next_epoch") == EXPECTED_RECIPE["epochs"]
        and progress.get("next_batch") == 0
        and progress.get("optimizer_steps") == EXPECTED_RECIPE["optimizer_steps"]
        and progress.get("samples_consumed") == EXPECTED_RECIPE["samples_consumed"]
        and isinstance(progress.get("epoch_receipts"), list)
        and len(progress["epoch_receipts"]) == EXPECTED_RECIPE["epochs"],
        f"legacy checkpoint is not the final matched exposure: {resolved}",
    )

    _require(
        receipt.get("schema") == LEGACY_TRAINING_SCHEMA
        and receipt.get("source") == source
        and receipt.get("environment") == environment
        and receipt.get("data") == data,
        f"legacy receipt/checkpoint identity differs: {resolved}",
    )
    architecture = _mapping(receipt.get("architecture"), "legacy receipt architecture")
    _require(
        architecture.get("schema") == ARCHITECTURE_SCHEMA
        and architecture.get("legacy_feature_schema") == "HORDETEST_HP_LEGACY_V1"
        and architecture.get("training_only_factorizer") is False,
        f"legacy receipt architecture drifted: {resolved}",
    )
    optimizer = _mapping(receipt.get("optimizer"), "legacy receipt optimizer")
    _require(
        optimizer.get("dense_learning_rate_multiplier")
        == EXPECTED_RECIPE["dense_learning_rate_multiplier"]
        and optimizer.get("output_learning_rate_multiplier")
        == EXPECTED_RECIPE["output_learning_rate_multiplier"],
        f"legacy receipt optimizer drifted: {resolved}",
    )
    run = _mapping(receipt.get("run"), "legacy receipt run")
    _require(
        run.get("seed") == seed
        and run.get("complete") is True
        and run.get("target_epochs") == EXPECTED_RECIPE["epochs"]
        and run.get("target_steps") == EXPECTED_RECIPE["optimizer_steps"]
        and run.get("optimizer_steps") == EXPECTED_RECIPE["optimizer_steps"]
        and run.get("samples_consumed") == EXPECTED_RECIPE["samples_consumed"]
        and run.get("next_epoch") == EXPECTED_RECIPE["epochs"]
        and run.get("next_batch") == 0
        and run.get("resume_checkpoint_sha256") is None,
        f"legacy receipt is not a complete uninterrupted run: {resolved}",
    )
    artifacts = _mapping(receipt.get("artifacts"), "legacy receipt artifacts")
    _require(
        _mapping(artifacts.get("checkpoint"), "legacy checkpoint artifact")
        == {"name": "checkpoint.pt", "sha256": checkpoint_sha256},
        f"legacy receipt does not bind its checkpoint: {resolved}",
    )
    stop_validation = _mapping(run.get("stop_validation"), "legacy stop validation")
    stop_loss = stop_validation.get("composite_loss")
    _require(
        type(stop_loss) is float
        and math.isfinite(stop_loss)
        and stop_validation.get("samples") == expected_validation.get("records"),
        f"legacy stop-validation metric is invalid: {resolved}",
    )
    model_state = checkpoint.get("model_state")
    _require(isinstance(model_state, dict), f"legacy model state is missing: {resolved}")
    model = _make_model(LEGACY_ARCHITECTURE, seed)
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise LegacyControlError(f"legacy model state is incompatible: {resolved}: {error}") from error
    model.eval()
    return LegacyRun(
        directory=resolved,
        seed=seed,
        checkpoint_sha256=checkpoint_sha256,
        receipt_sha256=_sha256_bytes(receipt_payload),
        stop_validation_loss=stop_loss,
        model=model,
    )


def _rank8_runs(c3_receipt: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    values = _mapping(c3_receipt.get("evaluation"), "C3 evaluation").get("architectures")
    _require(isinstance(values, list), "C3 architecture evaluation is invalid")
    rank8 = next(
        (
            value
            for value in values
            if _mapping(value, "C3 architecture summary").get("architecture", {}).get("name")
            == RANK8_ARCHITECTURE
        ),
        None,
    )
    _require(rank8 is not None, "C3 receipt does not contain Rank8")
    runs = _mapping(rank8, "C3 Rank8 summary").get("runs")
    _require(isinstance(runs, list) and len(runs) == 3, "C3 Rank8 run count drifted")
    _require([run.get("seed") for run in runs] == list(EXPECTED_SEEDS), "C3 Rank8 seeds drifted")
    return runs


def _collect_confirmation_predictions(
    dataset: ConfirmationRoleDataset,
    runs: Sequence[LegacyRun],
) -> tuple[array, array, array, dict[int, array]]:
    sides = array("b")
    teacher_scores = array("h")
    results = array("b")
    predictions = {run.seed: array("h") for run in runs}
    device = torch.device("cpu")
    torch.set_num_threads(1)
    for sparse in dataset.batches(EVALUATION_BATCH_SIZE):
        sides.extend(int(value) for value in sparse.side_to_move)
        teacher_scores.extend(int(value) for value in sparse.scores)
        results.extend(int(value) for value in sparse.results)
        batch = _model_batch(LEGACY_ARCHITECTURE, sparse, device)
        with torch.no_grad():
            for run in runs:
                postprocessed = _rule50_postprocess(run.model(batch), batch.rule50_count)
                _require(
                    bool(torch.isfinite(postprocessed).all()),
                    f"legacy seed {run.seed} produced non-finite confirmation scores",
                )
                _require(
                    bool(torch.equal(postprocessed, torch.trunc(postprocessed))),
                    f"legacy seed {run.seed} produced non-integer post-rule50 scores",
                )
                values = postprocessed.to(dtype=torch.int32, device="cpu").tolist()
                _require(
                    all(
                        LOOKUP_SCORE_MINIMUM <= int(value) <= LOOKUP_SCORE_MAXIMUM
                        for value in values
                    ),
                    f"legacy seed {run.seed} escaped the canonical score domain",
                )
                predictions[run.seed].extend(int(value) for value in values)
    _require(len(sides) == len(dataset), "legacy confirmation did not consume every record")
    _require(
        len(teacher_scores) == len(sides)
        and len(results) == len(sides)
        and all(len(values) == len(sides) for values in predictions.values()),
        "legacy confirmation prediction accounting drifted",
    )
    return sides, teacher_scores, results, predictions


def _mean(values: Sequence[float]) -> float:
    _require(bool(values), "cannot average an empty sequence")
    _require(all(math.isfinite(value) for value in values), "cannot average non-finite values")
    return math.fsum(values) / len(values)


def summarize_paired(
    rank8_runs: Sequence[Mapping[str, Any]],
    legacy_runs: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Summarize paired losses; positive deltas favor Rank8."""

    _require(len(rank8_runs) == len(legacy_runs) == 3, "paired control requires three runs per arm")
    rank8_by_seed = {int(run["seed"]): run for run in rank8_runs}
    legacy_by_seed = {int(run["seed"]): run for run in legacy_runs}
    _require(
        tuple(rank8_by_seed) == EXPECTED_SEEDS and set(legacy_by_seed) == set(EXPECTED_SEEDS),
        "paired control seeds are incomplete or reordered",
    )
    pairs: list[dict[str, object]] = []
    validation_deltas: list[float] = []
    confirmation_deltas: list[float] = []
    for seed in EXPECTED_SEEDS:
        rank8 = rank8_by_seed[seed]
        legacy = legacy_by_seed[seed]
        rank8_validation = float_from_receipt(
            rank8["tuning_stop_composite_loss"], f"Rank8 seed {seed} tuning loss"
        )
        legacy_validation = float(legacy["tuning_stop_composite_loss"])
        rank8_confirmation = float_from_receipt(
            _mapping(rank8["evaluation"], f"Rank8 seed {seed} evaluation")[
                "composite_loss_mean_all_records"
            ],
            f"Rank8 seed {seed} confirmation loss",
        )
        legacy_confirmation = float_from_receipt(
            _mapping(legacy["evaluation"], f"legacy seed {seed} evaluation")[
                "composite_loss_mean_all_records"
            ],
            f"legacy seed {seed} confirmation loss",
        )
        validation_delta = legacy_validation - rank8_validation
        confirmation_delta = legacy_confirmation - rank8_confirmation
        validation_deltas.append(validation_delta)
        confirmation_deltas.append(confirmation_delta)
        pairs.append(
            {
                "seed": seed,
                "tuning_validation": {
                    "rank8_composite_loss": float_receipt(rank8_validation),
                    "legacy_composite_loss": float_receipt(legacy_validation),
                    "legacy_minus_rank8": float_receipt(validation_delta),
                    "rank8_lower_loss": validation_delta > 0.0,
                },
                "fresh_confirmation": {
                    "rank8_composite_loss": float_receipt(rank8_confirmation),
                    "legacy_composite_loss": float_receipt(legacy_confirmation),
                    "legacy_minus_rank8": float_receipt(confirmation_delta),
                    "rank8_lower_loss": confirmation_delta > 0.0,
                },
            }
        )
    return {
        "delta_direction": "positive legacy_minus_rank8 favors Rank8",
        "pairs": pairs,
        "three_seed_mean_delta": {
            "tuning_validation_legacy_minus_rank8": float_receipt(_mean(validation_deltas)),
            "fresh_confirmation_legacy_minus_rank8": float_receipt(_mean(confirmation_deltas)),
        },
        "directional_consistency": {
            "rank8_lower_tuning_loss_all_three_seeds": all(value > 0.0 for value in validation_deltas),
            "rank8_lower_confirmation_loss_all_three_seeds": all(
                value > 0.0 for value in confirmation_deltas
            ),
            "rank8_lower_confirmation_loss_seed_count": sum(
                value > 0.0 for value in confirmation_deltas
            ),
        },
    }


def build_receipt(
    c3_path: Path,
    legacy_validation_path: Path,
    confirmation_path: Path,
    wdl_path: Path,
    legacy_directories: Sequence[Path],
    *,
    allow_dirty: bool = False,
) -> dict[str, object]:
    source = _repository_identity()
    _require(allow_dirty or source["dirty"] is False, "control evaluator source tree is dirty")
    c3_receipt, c3_payload = _read_json(c3_path, "C3 representation receipt")
    _require(
        c3_receipt.get("schema") == C3_RECEIPT_SCHEMA
        and _sha256_bytes(c3_payload) == EXPECTED_C3_RECEIPT_SHA256,
        "C3 representation receipt identity drifted",
    )
    try:
        validate_c3_receipt(c3_receipt)
    except ValueError as error:
        raise LegacyControlError(f"C3 representation receipt is invalid: {error}") from error
    rank8 = _rank8_runs(c3_receipt)
    inputs = _mapping(c3_receipt.get("inputs"), "C3 inputs")
    expected_training = _mapping(
        _mapping(inputs.get("constant_baseline"), "C3 constant baseline").get("training_file"),
        "C3 training identity",
    )
    expected_tuning = _mapping(inputs.get("tuning_validation"), "C3 tuning identity")
    expected_confirmation = _mapping(inputs.get("confirmation_role"), "C3 confirmation identity")
    expected_teacher = _mapping(inputs.get("teacher"), "C3 teacher identity")
    expected_wdl = _mapping(inputs.get("wdl_calibration"), "C3 WDL identity")

    with SelectedRoleDataset(legacy_validation_path.expanduser().resolve()) as legacy_validation:
        legacy_validation_identity = legacy_validation.identity()
    _require(
        _identity_matches(legacy_validation_identity, expected_tuning),
        "legacy validation records differ from the C3 tuning role",
    )
    _require(len(legacy_directories) == 3, "exactly three legacy run directories are required")
    legacy = [
        _prepare_legacy_run(
            directory,
            expected_training=expected_training,
            expected_validation=legacy_validation_identity,
            expected_teacher=expected_teacher,
            expected_wdl=expected_wdl,
        )
        for directory in legacy_directories
    ]
    legacy_by_seed = {run.seed: run for run in legacy}
    _require(
        set(legacy_by_seed) == set(EXPECTED_SEEDS) and len(legacy_by_seed) == 3,
        "legacy run matrix is incomplete or duplicated",
    )
    legacy = [legacy_by_seed[seed] for seed in EXPECTED_SEEDS]

    wdl_artifact, parameters, wdl_sha256 = load_artifact(wdl_path.expanduser().resolve())
    _require(
        wdl_sha256 == expected_wdl.get("sha256")
        and wdl_artifact.get("schema") == expected_wdl.get("schema"),
        "WDL calibration identity drifted",
    )
    lookup = build_wdl_lookup(parameters)
    with ConfirmationRoleDataset(confirmation_path.expanduser().resolve()) as confirmation:
        confirmation_identity = confirmation.identity()
        _require(
            confirmation_identity == dict(expected_confirmation),
            "confirmation role differs from the authenticated C3 role",
        )
        sides, teacher_scores, results, predictions = _collect_confirmation_predictions(
            confirmation, legacy
        )

    legacy_receipts: list[dict[str, object]] = []
    for run in legacy:
        evaluation = evaluate_prediction_scores(
            lookup,
            sides,
            teacher_scores,
            results,
            predictions[run.seed],
        )
        legacy_receipts.append(
            {
                "seed": run.seed,
                "directory_name": run.directory.name,
                "checkpoint_sha256": run.checkpoint_sha256,
                "training_receipt_sha256": run.receipt_sha256,
                "tuning_stop_composite_loss": run.stop_validation_loss,
                "evaluation": evaluation,
            }
        )
    comparison = summarize_paired(rank8, legacy_receipts)
    directional = _mapping(comparison.get("directional_consistency"), "directional result")
    checks = {
        "source_clean": source["dirty"] is False,
        "authenticated_c3_rank8_evidence": True,
        "byte_identical_tuning_records": True,
        "same_training_records": True,
        "same_teacher_and_wdl": True,
        "exact_three_paired_seeds": True,
        "matched_optimizer_recipe": True,
        "all_legacy_runs_complete": True,
        "fresh_confirmation_role_authenticated": True,
        "all_confirmation_records_evaluated": True,
    }
    return {
        "schema": SCHEMA,
        "source": source,
        "inputs": {
            "c3_representation_receipt": {
                "name": c3_path.expanduser().resolve().name,
                "sha256": EXPECTED_C3_RECEIPT_SHA256,
                "schema": C3_RECEIPT_SCHEMA,
            },
            "training": dict(expected_training),
            "tuning_validation": {
                "c3": dict(expected_tuning),
                "legacy_rematerialization": legacy_validation_identity,
                "byte_identical_records": True,
            },
            "confirmation_role": dict(expected_confirmation),
            "teacher": dict(expected_teacher),
            "wdl_calibration": dict(expected_wdl),
            "paired_seeds": list(EXPECTED_SEEDS),
            "recipe": dict(EXPECTED_RECIPE),
        },
        "evaluation": {
            "rank8_runs": [dict(run) for run in rank8],
            "legacy_runs": legacy_receipts,
            "paired_comparison": comparison,
        },
        "gates": {"checks": checks, "passed": all(checks.values())},
        "claims": {
            "matched_architecture_control_complete": all(checks.values()),
            "rank8_lower_confirmation_loss_all_three_seeds": directional[
                "rank8_lower_confirmation_loss_all_three_seeds"
            ],
            "loss_selects_architecture": False,
            "best_seed_selected": False,
            "statistical_confidence": False,
            "playing_strength_evidence": False,
            "equal_time_three_control_gate_required": True,
            "production_network": False,
            "run6b_production_path_changed": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("c3_receipt", type=Path)
    parser.add_argument("legacy_validation_receipt", type=Path)
    parser.add_argument("confirmation_receipt", type=Path)
    parser.add_argument("wdl_calibration", type=Path)
    parser.add_argument("--legacy-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_receipt(
        args.c3_receipt,
        args.legacy_validation_receipt,
        args.confirmation_receipt,
        args.wdl_calibration,
        args.legacy_run,
        allow_dirty=args.allow_dirty,
    )
    _write_exclusive(args.output, _canonical_json(receipt))
    comparison = _mapping(
        _mapping(receipt["evaluation"], "control evaluation").get("paired_comparison"),
        "paired comparison",
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output": str(args.output.expanduser().resolve()),
                "passed": receipt["gates"]["passed"],
                "directional_consistency": comparison["directional_consistency"],
                "three_seed_mean_delta": comparison["three_seed_mean_delta"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LegacyControlError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
