#!/usr/bin/env python3
"""Qualify the frozen three-architecture Horde V2 C3 representation matrix."""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import nn

try:
    from . import horde_v2_c2_qualification as c2
    from .horde_training_control import (
        V2_CHECKPOINT_SCHEMA,
        V2_TRAINING_SCHEMA,
        _make_model,
        _rule50_postprocess,
        _torch_v2_batch,
    )
    from .horde_training_decoder import BLACK, WHITE
    from .horde_training_selected_role import SelectedRoleDataset
    from .horde_v2_c3_confirmation_role import (
        ConfirmationRoleDataset,
        CONTRACT_SHA256 as CONFIRMATION_CONTRACT_SHA256,
        CONTRACT_SCHEMA as CONFIRMATION_CONTRACT_SCHEMA,
        SCHEMA as CONFIRMATION_RECEIPT_SCHEMA,
    )
    from .horde_v2_c2_constant_baseline import (
        SCHEMA as CONSTANT_RECEIPT_SCHEMA,
        validate_receipt as validate_constant_receipt,
    )
    from .horde_v2_c2_objective import (
        LAMBDA,
        LOOKUP_SCORE_MAXIMUM,
        LOOKUP_SCORE_MINIMUM,
        OBJECTIVE_SCHEMA,
        build_wdl_lookup,
        float_from_receipt,
        float_receipt,
    )
    from .horde_v2_functional_health import (
        CONTRACT_SCHEMA as HEALTH_CONTRACT_SCHEMA,
        CONTRACT_SHA256 as HEALTH_CONTRACT_SHA256,
        SCHEMA as HEALTH_RECEIPT_SCHEMA,
    )
    from .horde_wdl import LINK_SCHEMA as WDL_LINK_SCHEMA, SCHEMA as WDL_SCHEMA, load_artifact
except ImportError:
    import horde_v2_c2_qualification as c2
    from horde_training_control import (
        V2_CHECKPOINT_SCHEMA,
        V2_TRAINING_SCHEMA,
        _make_model,
        _rule50_postprocess,
        _torch_v2_batch,
    )
    from horde_training_decoder import BLACK, WHITE
    from horde_training_selected_role import SelectedRoleDataset
    from horde_v2_c3_confirmation_role import (
        ConfirmationRoleDataset,
        CONTRACT_SHA256 as CONFIRMATION_CONTRACT_SHA256,
        CONTRACT_SCHEMA as CONFIRMATION_CONTRACT_SCHEMA,
        SCHEMA as CONFIRMATION_RECEIPT_SCHEMA,
    )
    from horde_v2_c2_constant_baseline import (
        SCHEMA as CONSTANT_RECEIPT_SCHEMA,
        validate_receipt as validate_constant_receipt,
    )
    from horde_v2_c2_objective import (
        LAMBDA,
        LOOKUP_SCORE_MAXIMUM,
        LOOKUP_SCORE_MINIMUM,
        OBJECTIVE_SCHEMA,
        build_wdl_lookup,
        float_from_receipt,
        float_receipt,
    )
    from horde_v2_functional_health import (
        CONTRACT_SCHEMA as HEALTH_CONTRACT_SCHEMA,
        CONTRACT_SHA256 as HEALTH_CONTRACT_SHA256,
        SCHEMA as HEALTH_RECEIPT_SCHEMA,
    )
    from horde_wdl import LINK_SCHEMA as WDL_LINK_SCHEMA, SCHEMA as WDL_SCHEMA, load_artifact


SCHEMA = "HORDE_V2_C3_REPRESENTATION_QUALIFICATION_RECEIPT_V1"
CONTRACT_SCHEMA = "HORDE_V2_C3_REPRESENTATION_QUALIFICATION_V1"
CONTRACT_RELATIVE_PATH = Path("schemas/horde-v2-c3-representation-qualification-v1.json")
CONTRACT_SHA256 = "33F48B363AB6B4B20303E586E484DB7F45BF6BDBBFB5DF82C9FB034542A7B7DA"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOL_RELATIVE_PATH = Path("tools/horde_v2_c3_qualification.py")
SIDE_NAMES = {WHITE: "white_to_move", BLACK: "black_to_move"}
EVALUATION_BATCH_SIZE = 4096
EXPORT_RECEIPT_SCHEMA = "HORDE_V2_INTEGER_CHECKPOINT_EXPORT_V1"
EXPORT_CONTAINER_SCHEMA = "HORDE_V2_INTEGER_EXPORT_RECEIPT_V1"
INTEGER_NETWORK_SCHEMA = "HORDE_V2_INTEGER_NETWORK_V1"
CONFIRMATION_VERIFICATION_SCHEMA = "HORDE_V2_C3_CONFIRMATION_ROLE_VERIFICATION_V1"


class C3QualificationError(ValueError):
    """Raised when C3 representation evidence violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C3QualificationError(message)


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
        raise C3QualificationError(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(value, dict), f"{label} root is not an object")
    _require(payload == _canonical_json(value), f"{label} is not canonical JSON")
    return value, payload


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789ABCDEF" for character in value)
    )


def _valid_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value != "0" * 40
        and all(character in "0123456789abcdef" for character in value)
    )


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
    _require(_valid_commit(commit), "C3 qualifier source is not a full Git identity")
    tool = root / TOOL_RELATIVE_PATH
    _require(tool.is_file(), "C3 qualifier source file is missing")
    return {
        "commit": commit,
        "dirty": dirty,
        "path": TOOL_RELATIVE_PATH.as_posix(),
        "file_sha256": _sha256_file(tool),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "torch": torch.__version__,
    }


def load_contract(path: Path | None = None) -> tuple[dict[str, Any], str]:
    resolved = (path or REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH).expanduser().resolve()
    contract, payload = _read_json(resolved, "C3 representation contract")
    digest = _sha256_bytes(payload)
    _require(digest == CONTRACT_SHA256, f"C3 representation contract SHA-256 mismatch: {digest}")
    _require(contract.get("schema_name") == CONTRACT_SCHEMA, "C3 contract schema drifted")
    matrix = _mapping(contract.get("matrix"), "C3 matrix contract")
    architectures = matrix.get("architectures")
    seeds = matrix.get("frozen_seeds")
    training = _mapping(contract.get("training"), "C3 training contract")
    _require(
        isinstance(architectures, list)
        and len(architectures) == 3
        and isinstance(seeds, list)
        and len(seeds) == 3
        and matrix.get("run_count") == 9
        and training.get("epochs") == 8
        and training.get("optimizer_steps") == 496
        and training.get("samples_consumed_per_model") == 2_000_000
        and training.get("batch_size") == 4096
        and training.get("block_size") == 65_536
        and training.get("lambda") == 0.6
        and training.get("learning_rate") == 0.0015
        and _mapping(training.get("optimizer"), "C3 optimizer contract").get(
            "dense_learning_rate_multiplier"
        )
        == 0.1
        and _mapping(training.get("optimizer"), "C3 optimizer contract").get(
            "output_learning_rate_multiplier"
        )
        == 0.1,
        "C3 matrix or qualified recipe drifted",
    )
    return contract, digest


def _architectures(contract: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    matrix = _mapping(contract.get("matrix"), "C3 matrix contract")
    values = matrix.get("architectures")
    _require(isinstance(values, list), "C3 architecture matrix is invalid")
    output: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        architecture = _mapping(value, f"C3 architecture {index}")
        expected_keys = {
            "first_domain",
            "name",
            "parameter_count",
            "schema",
            "serialized_parameter_bytes",
            "structural_sha256",
        }
        _require(
            set(architecture) == expected_keys
            and isinstance(architecture.get("name"), str)
            and isinstance(architecture.get("schema"), str)
            and type(architecture.get("parameter_count")) is int
            and type(architecture.get("serialized_parameter_bytes")) is int
            and _valid_sha256(architecture.get("structural_sha256")),
            f"C3 architecture {index} is invalid",
        )
        output.append(dict(architecture))
    _require(len({item["name"] for item in output}) == 3, "C3 architecture name is duplicated")
    return tuple(output)


def _seeds(contract: Mapping[str, Any]) -> tuple[int, ...]:
    values = _mapping(contract.get("matrix"), "C3 matrix contract").get("frozen_seeds")
    _require(
        isinstance(values, list)
        and len(values) == 3
        and all(type(value) is int and value > 0 for value in values)
        and len(set(values)) == 3,
        "C3 seed matrix is invalid",
    )
    return tuple(values)


@dataclass
class PreparedRun:
    directory: Path
    architecture: dict[str, Any]
    seed: int
    checkpoint_sha256: str
    training_receipt_sha256: str
    network_sha256: str
    export_receipt_sha256: str
    health_receipt_sha256: str
    model_state: Mapping[str, Any]
    stop_validation_loss: float
    model: nn.Module | None = None

    @property
    def key(self) -> tuple[str, int]:
        return str(self.architecture["name"]), self.seed


def _source_clean(value: object, label: str, expected_commit: str) -> dict[str, object]:
    source = _mapping(value, label)
    _require(
        source == {"commit": expected_commit, "dirty": False},
        f"{label} differs from the frozen clean training source",
    )
    return dict(source)


def _prepare_run(
    directory: Path,
    architecture: Mapping[str, Any],
    seeds: Sequence[int],
    training_commit: str,
    tuning_validation_identity: Mapping[str, object],
    expected_training: Mapping[str, object],
    expected_teacher: Mapping[str, object],
    expected_wdl: Mapping[str, object],
) -> PreparedRun:
    resolved = directory.expanduser().resolve()
    _require(resolved.is_dir(), f"C3 run directory does not exist: {resolved}")
    paths = {
        "checkpoint": resolved / "checkpoint.pt",
        "training": resolved / "receipt.json",
        "network": resolved / "network.hsv2",
        "export": resolved / "export-receipt.json",
        "health": resolved / "functional-health.json",
    }
    _require(all(path.is_file() for path in paths.values()), f"C3 run artifacts are incomplete: {resolved}")
    try:
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    except (EOFError, RuntimeError, ValueError) as error:
        raise C3QualificationError(f"cannot load checkpoint {paths['checkpoint']}: {error}") from error
    _require(isinstance(checkpoint, dict), f"checkpoint root is invalid: {paths['checkpoint']}")
    checkpoint_sha256 = _sha256_file(paths["checkpoint"])
    training_receipt, training_payload = _read_json(paths["training"], "C3 training receipt")
    export_receipt, export_payload = _read_json(paths["export"], "C3 export receipt")
    health_receipt, health_payload = _read_json(paths["health"], "C3 functional-health receipt")
    network_sha256 = _sha256_file(paths["network"])
    training_receipt_sha256 = _sha256_bytes(training_payload)

    _require(checkpoint.get("schema") == V2_CHECKPOINT_SCHEMA, "C3 checkpoint schema drifted")
    _require(checkpoint.get("architecture") == architecture["schema"], "C3 checkpoint architecture drifted")
    checkpoint_source = _source_clean(checkpoint.get("source"), "checkpoint source", training_commit)
    settings = _mapping(checkpoint.get("settings"), "C3 checkpoint settings")
    expected_architecture = {
        "name": architecture["name"],
        "schema": architecture["schema"],
        "structural_sha256": architecture["structural_sha256"],
    }
    _require(settings.get("architecture") == expected_architecture, "C3 checkpoint structure drifted")
    seed = settings.get("seed")
    _require(type(seed) is int and seed in seeds, "C3 checkpoint seed is outside the frozen set")
    _require(
        settings.get("epochs") == 8
        and settings.get("batch_size") == 4096
        and settings.get("block_size") == 65_536
        and settings.get("lambda") == LAMBDA
        and settings.get("learning_rate") == 0.0015
        and settings.get("scheduler_gamma") == 0.987
        and settings.get("device_type") == "cuda"
        and settings.get("cpu_threads") == 1
        and settings.get("initialization") == "SHA256_NAMED_PARAMETER_SEED_V1"
        and settings.get("wdl_calibration_sha256") == expected_wdl["sha256"]
        and settings.get("optimizer_learning_rate_multipliers")
        == {"dense_trunk": 0.1, "output": 0.1},
        "C3 checkpoint recipe drifted",
    )
    checkpoint_data = c2._validate_run_data(
        checkpoint.get("data"),
        expected_training=expected_training,
        expected_validation=tuning_validation_identity,
        expected_teacher=expected_teacher,
        expected_wdl=expected_wdl,
        label="C3 checkpoint data",
    )
    progress = _mapping(checkpoint.get("progress"), "C3 checkpoint progress")
    _require(
        progress.get("next_epoch") == 8
        and progress.get("next_batch") == 0
        and progress.get("optimizer_steps") == 496
        and progress.get("samples_consumed") == 2_000_000
        and isinstance(progress.get("epoch_receipts"), list)
        and len(progress["epoch_receipts"]) == 8,
        "C3 checkpoint is not the final frozen exposure",
    )
    state = checkpoint.get("model_state")
    _require(isinstance(state, dict), "C3 checkpoint model state is missing")

    _require(training_receipt.get("schema") == V2_TRAINING_SCHEMA, "C3 training receipt schema drifted")
    receipt_source = _source_clean(training_receipt.get("source"), "training receipt source", training_commit)
    _require(receipt_source == checkpoint_source, "C3 source identities differ")
    receipt_architecture = _mapping(training_receipt.get("architecture"), "C3 training architecture")
    _require(
        receipt_architecture.get("name") == architecture["name"]
        and receipt_architecture.get("schema") == architecture["schema"]
        and receipt_architecture.get("structural_sha256") == architecture["structural_sha256"]
        and receipt_architecture.get("parameter_count") == architecture["parameter_count"]
        and receipt_architecture.get("serialized_parameter_bytes")
        == architecture["serialized_parameter_bytes"],
        "C3 training architecture receipt drifted",
    )
    artifacts = _mapping(training_receipt.get("artifacts"), "C3 training artifacts")
    _require(
        _mapping(artifacts.get("checkpoint"), "C3 checkpoint artifact")
        == {"name": "checkpoint.pt", "sha256": checkpoint_sha256},
        "C3 training receipt does not bind the checkpoint",
    )
    run = _mapping(training_receipt.get("run"), "C3 training run")
    _require(
        run.get("complete") is True
        and run.get("seed") == seed
        and run.get("target_epochs") == 8
        and run.get("next_epoch") == 8
        and run.get("next_batch") == 0
        and run.get("target_steps") == 496
        and run.get("optimizer_steps") == 496
        and run.get("samples_consumed") == 2_000_000
        and isinstance(run.get("epochs_receipt"), list)
        and len(run["epochs_receipt"]) == 8,
        "C3 training receipt is not the final frozen exposure",
    )
    optimizer = _mapping(training_receipt.get("optimizer"), "C3 optimizer receipt")
    _require(
        optimizer.get("dense_learning_rate_multiplier") == 0.1
        and optimizer.get("output_learning_rate_multiplier") == 0.1,
        "C3 training optimizer multipliers drifted",
    )
    environment = _mapping(training_receipt.get("environment"), "C3 environment receipt")
    device = _mapping(environment.get("device"), "C3 device receipt")
    _require(
        device.get("type") == "cuda"
        and device.get("name") == "NVIDIA GeForce RTX 3080"
        and device.get("cpu_threads") == 1
        and environment.get("amp") is False
        and environment.get("cuda_matmul_allow_tf32") is False
        and environment.get("cudnn_allow_tf32") is False,
        "C3 training environment drifted",
    )
    receipt_data = c2._validate_run_data(
        training_receipt.get("data"),
        expected_training=expected_training,
        expected_validation=tuning_validation_identity,
        expected_teacher=expected_teacher,
        expected_wdl=expected_wdl,
        label="C3 training receipt data",
    )
    _require(dict(receipt_data) == dict(checkpoint_data), "C3 checkpoint/receipt data differ")
    stop_validation = _mapping(run.get("stop_validation"), "C3 stop validation")
    stop_loss = stop_validation.get("composite_loss")
    _require(type(stop_loss) is float and math.isfinite(stop_loss), "C3 stop loss is invalid")

    _require(export_receipt.get("schema") == EXPORT_RECEIPT_SCHEMA, "C3 export schema drifted")
    container = _mapping(export_receipt.get("container"), "C3 export container")
    provenance = _mapping(container.get("provenance"), "C3 export provenance")
    _require(
        container.get("schema") == EXPORT_CONTAINER_SCHEMA
        and container.get("container_schema") == INTEGER_NETWORK_SCHEMA
        and container.get("network_schema") == architecture["schema"]
        and container.get("parameter_bytes") == architecture["serialized_parameter_bytes"]
        and container.get("training_architecture_structural_sha256")
        == architecture["structural_sha256"]
        and container.get("file_sha256") == network_sha256
        and container.get("file_bytes") == paths["network"].stat().st_size
        and provenance.get("checkpoint_sha256") == checkpoint_sha256
        and provenance.get("training_receipt_sha256") == training_receipt_sha256
        and provenance.get("source_commit") == training_commit
        and provenance.get("source_dirty") is False
        and provenance.get("train_file_sha256") == expected_training["sha256"]
        and provenance.get("validation_file_sha256") == tuning_validation_identity["sha256"]
        and provenance.get("wdl_calibration_sha256") == expected_wdl["sha256"],
        "C3 export provenance drifted",
    )
    export_claims = _mapping(export_receipt.get("claims"), "C3 export claims")
    _require(
        export_claims.get("full_refresh_container") is True
        and export_claims.get("strength_evidence") is False
        and export_claims.get("production_dispatch") is False,
        "C3 export claims drifted",
    )

    _require(health_receipt.get("schema") == HEALTH_RECEIPT_SCHEMA, "C3 health schema drifted")
    _require(
        health_receipt.get("contract")
        == {"schema": HEALTH_CONTRACT_SCHEMA, "sha256": HEALTH_CONTRACT_SHA256},
        "C3 health contract drifted",
    )
    _source_clean(health_receipt.get("source"), "C3 health source", training_commit)
    health_checkpoint = _mapping(health_receipt.get("checkpoint"), "C3 health checkpoint")
    _require(
        health_checkpoint.get("sha256") == checkpoint_sha256
        and health_checkpoint.get("architecture") == architecture["name"]
        and health_checkpoint.get("architecture_schema") == architecture["schema"]
        and health_checkpoint.get("seed") == seed
        and health_checkpoint.get("optimizer_steps") == 496
        and health_checkpoint.get("samples_consumed") == 2_000_000
        and health_checkpoint.get("source") == checkpoint_source
        and health_receipt.get("validation") == dict(tuning_validation_identity),
        "C3 health receipt does not bind the final checkpoint",
    )
    health_gates = _mapping(health_receipt.get("gates"), "C3 health gates")
    health_checks = _mapping(health_gates.get("checks"), "C3 health checks")
    health_claims = _mapping(health_receipt.get("claims"), "C3 health claims")
    _require(
        health_gates.get("passed") is True
        and bool(health_checks)
        and all(value is True for value in health_checks.values())
        and health_claims.get("functional_health_passed") is True
        and health_claims.get("strength_evidence") is False,
        "C3 functional health did not pass",
    )
    return PreparedRun(
        directory=resolved,
        architecture=dict(architecture),
        seed=seed,
        checkpoint_sha256=checkpoint_sha256,
        training_receipt_sha256=training_receipt_sha256,
        network_sha256=network_sha256,
        export_receipt_sha256=_sha256_bytes(export_payload),
        health_receipt_sha256=_sha256_bytes(health_payload),
        model_state=state,
        stop_validation_loss=stop_loss,
    )


def _load_models(runs: Sequence[PreparedRun]) -> None:
    for run in runs:
        model = _make_model(str(run.architecture["name"]), run.seed)
        try:
            model.load_state_dict(run.model_state, strict=True)
        except RuntimeError as error:
            raise C3QualificationError(
                f"C3 model state is incompatible for {run.architecture['name']} seed {run.seed}: {error}"
            ) from error
        model.eval()
        run.model = model


def _collect_predictions(
    dataset: ConfirmationRoleDataset,
    constants: Mapping[int, int],
    runs: Sequence[PreparedRun],
) -> tuple[array, array, array, array, dict[tuple[str, int], array]]:
    sides = array("b")
    teacher_scores = array("h")
    results = array("b")
    baseline_scores = array("h")
    candidate_scores = {run.key: array("h") for run in runs}
    device = torch.device("cpu")
    for sparse in dataset.batches(EVALUATION_BATCH_SIZE):
        sides.extend(int(value) for value in sparse.side_to_move)
        teacher_scores.extend(int(value) for value in sparse.scores)
        results.extend(int(value) for value in sparse.results)
        baseline_scores.extend(
            c2.rule50_postprocess_constant(constants[int(side)], int(clock))
            for side, clock in zip(sparse.side_to_move, sparse.rule50_count, strict=True)
        )
        batch = _torch_v2_batch(sparse, device)
        with torch.no_grad():
            for run in runs:
                _require(run.model is not None, "C3 model was not loaded after artifact preflight")
                postprocessed = _rule50_postprocess(run.model(batch), batch.rule50_count)
                _require(bool(torch.isfinite(postprocessed).all()), "C3 checkpoint produced non-finite scores")
                _require(
                    bool(torch.equal(postprocessed, torch.trunc(postprocessed))),
                    "C3 post-rule50 scores are not integers",
                )
                values = postprocessed.to(dtype=torch.int32, device="cpu").tolist()
                _require(
                    all(LOOKUP_SCORE_MINIMUM <= int(value) <= LOOKUP_SCORE_MAXIMUM for value in values),
                    "C3 checkpoint score escaped the canonical domain",
                )
                candidate_scores[run.key].extend(int(value) for value in values)
    _require(len(sides) == len(dataset), "C3 evaluation did not consume the full confirmation role")
    _require(
        len(teacher_scores) == len(sides)
        and len(results) == len(sides)
        and len(baseline_scores) == len(sides)
        and all(len(values) == len(sides) for values in candidate_scores.values()),
        "C3 prediction accounting drifted",
    )
    return sides, teacher_scores, results, baseline_scores, candidate_scores


def summarize_architectures(
    architectures: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    baseline_mean: float,
    run_receipts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, object]], list[str]]:
    by_key = {
        (str(run["architecture"]), int(run["seed"])): run
        for run in run_receipts
    }
    summaries: list[dict[str, object]] = []
    for architecture in architectures:
        name = str(architecture["name"])
        ordered = [by_key[(name, seed)] for seed in seeds]
        losses = [
            float_from_receipt(
                _mapping(run["evaluation"], "C3 run evaluation")["composite_loss_mean_all_records"],
                f"C3 {name} seed {run['seed']} loss",
            )
            for run in ordered
        ]
        deltas = [baseline_mean - loss for loss in losses]
        mean_loss = math.fsum(losses) / len(losses)
        mean_delta = baseline_mean - mean_loss
        eligible = all(delta > 0.0 for delta in deltas)
        summaries.append(
            {
                "architecture": dict(architecture),
                "runs": [dict(run) for run in ordered],
                "three_seed_mean_composite_loss": float_receipt(mean_loss),
                "three_seed_mean_delta_constant_minus_checkpoint": float_receipt(mean_delta),
                "all_three_seeds_strictly_better_than_constant": eligible,
                "confirmation_eligible": eligible,
            }
        )
    frontier: list[str] = []
    eligible_summaries = [summary for summary in summaries if summary["confirmation_eligible"]]
    for candidate in eligible_summaries:
        candidate_architecture = _mapping(candidate["architecture"], "C3 candidate architecture")
        candidate_loss = float_from_receipt(
            candidate["three_seed_mean_composite_loss"], "C3 candidate mean loss"
        )
        candidate_bytes = int(candidate_architecture["serialized_parameter_bytes"])
        dominated = False
        for other in eligible_summaries:
            if other is candidate:
                continue
            other_architecture = _mapping(other["architecture"], "C3 other architecture")
            other_loss = float_from_receipt(other["three_seed_mean_composite_loss"], "C3 other mean loss")
            other_bytes = int(other_architecture["serialized_parameter_bytes"])
            if (
                other_loss <= candidate_loss
                and other_bytes <= candidate_bytes
                and (other_loss < candidate_loss or other_bytes < candidate_bytes)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(str(candidate_architecture["name"]))
    return summaries, frontier


def _validate_confirmation_verification(
    path: Path,
    confirmation_identity: Mapping[str, object],
) -> tuple[dict[str, Any], str]:
    verification, payload = _read_json(path, "C3 confirmation verification")
    _require(
        verification.get("schema") == CONFIRMATION_VERIFICATION_SCHEMA
        and verification.get("contract_sha256") == CONFIRMATION_CONTRACT_SHA256
        and verification.get("confirmation_role") == dict(confirmation_identity),
        "C3 confirmation verification identity drifted",
    )
    claims = _mapping(verification.get("claims"), "C3 confirmation verification claims")
    required = (
        "canonical_selection_recomputed",
        "materialized_records_reconstructed",
        "zero_overlap_with_both_excluded_roles",
        "zero_confirmation_duplicates",
        "label_blind_selection",
    )
    _require(all(claims.get(key) is True for key in required), "C3 confirmation verification did not pass")
    _require(
        claims.get("network_inference_performed") is False
        and claims.get("architecture_selected") is False
        and claims.get("strength_evidence") is False,
        "C3 confirmation verification claims drifted",
    )
    return verification, _sha256_bytes(payload)


def build_receipt(
    qualified_recipe_path: Path,
    constant_path: Path,
    tuning_validation_path: Path,
    confirmation_path: Path,
    confirmation_verification_path: Path,
    wdl_path: Path,
    run_directories: Sequence[Path],
    *,
    contract_path: Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, object]:
    contract, contract_sha256 = load_contract(contract_path)
    source = _repository_identity(REPOSITORY_ROOT)
    _require(allow_dirty or source["dirty"] is False, "C3 qualifier source tree is dirty")
    architectures = _architectures(contract)
    seeds = _seeds(contract)
    _require(len(run_directories) == 9, "C3 qualification requires exactly nine run directories")
    dependencies = _mapping(contract.get("dependencies"), "C3 dependency contract")
    training_contract = _mapping(contract.get("training"), "C3 training contract")
    training_commit = str(training_contract["source_commit"])

    qualified_recipe, qualified_recipe_payload = _read_json(
        qualified_recipe_path, "C3 qualified-recipe receipt"
    )
    expected_recipe = _mapping(dependencies.get("qualified_recipe"), "qualified recipe contract")
    _require(
        _sha256_bytes(qualified_recipe_payload) == expected_recipe.get("receipt_sha256")
        and qualified_recipe.get("schema") == expected_recipe.get("receipt_schema"),
        "C3 qualified-recipe receipt drifted",
    )
    try:
        c2.validate_receipt(qualified_recipe)
    except ValueError as error:
        raise C3QualificationError(f"C3 qualified-recipe receipt is invalid: {error}") from error
    _require(
        _mapping(qualified_recipe.get("gates"), "qualified recipe gates").get("passed") is True
        and _mapping(qualified_recipe.get("arm"), "qualified recipe arm").get("name")
        == expected_recipe.get("arm"),
        "C3 recipe was not qualified",
    )

    constant, constant_payload = _read_json(constant_path, "C3 constant-baseline receipt")
    expected_constant = _mapping(dependencies.get("constant_baseline"), "constant contract")
    _require(
        constant.get("schema") == CONSTANT_RECEIPT_SCHEMA
        and constant.get("schema") == expected_constant.get("schema")
        and _sha256_bytes(constant_payload) == expected_constant.get("sha256"),
        "C3 constant-baseline receipt drifted",
    )
    try:
        validate_constant_receipt(constant)
    except ValueError as error:
        raise C3QualificationError(f"C3 constant-baseline receipt is invalid: {error}") from error
    constant_source = _mapping(constant.get("source"), "C3 constant source")
    expected_training = c2._validate_training_identity(
        constant_source.get("training_file"), "C3 training identity"
    )
    expected_teacher = c2._validate_teacher_identity(
        constant_source.get("teacher"), "C3 teacher identity"
    )
    expected_wdl = c2._validate_wdl_identity(
        constant_source.get("wdl_calibration"), "C3 WDL identity"
    )
    expected_training_contract = _mapping(dependencies.get("training_file"), "training file contract")
    _require(
        expected_training["sha256"] == expected_training_contract.get("file_sha256")
        and expected_training["payload_sha256"] == expected_training_contract.get("payload_sha256")
        and expected_training["manifest_sha256"] == expected_training_contract.get("manifest_sha256")
        and expected_training["records"] == expected_training_contract.get("records"),
        "C3 training-file dependency drifted",
    )

    wdl_artifact, parameters, wdl_sha256 = load_artifact(wdl_path.expanduser().resolve())
    expected_wdl_contract = _mapping(dependencies.get("wdl_calibration"), "WDL contract")
    _require(
        wdl_sha256 == expected_wdl["sha256"] == expected_wdl_contract.get("sha256")
        and wdl_artifact.get("schema") == WDL_SCHEMA == expected_wdl_contract.get("schema"),
        "C3 WDL artifact drifted",
    )
    wdl_source = _mapping(wdl_artifact.get("source"), "C3 WDL source")
    _require(
        c2._validate_training_identity(wdl_source.get("training_file"), "C3 WDL training")
        == expected_training
        and c2._validate_teacher_identity(wdl_source.get("teacher"), "C3 WDL teacher")
        == expected_teacher,
        "C3 WDL provenance drifted",
    )
    wdl_selection = _mapping(wdl_artifact.get("selection"), "C3 WDL selection")
    _require(
        wdl_selection.get("selection_sha256") == expected_wdl["selection_sha256"]
        and wdl_selection.get("eligible_records_sha256")
        == expected_wdl["eligible_records_sha256"]
        and _mapping(wdl_artifact.get("link"), "C3 WDL link").get("schema")
        == WDL_LINK_SCHEMA
        == expected_wdl["link_schema"],
        "C3 WDL selection drifted",
    )
    lookup = build_wdl_lookup(parameters)

    with SelectedRoleDataset(tuning_validation_path.expanduser().resolve()) as tuning_validation:
        tuning_validation_identity = c2._validate_validation_identity(
            tuning_validation.identity(), "C3 tuning-validation identity"
        )
        expected_tuning = _mapping(
            dependencies.get("tuning_validation_role"), "tuning-validation contract"
        )
        _require(
            tuning_validation_identity["sha256"] == expected_tuning.get("file_sha256")
            and tuning_validation_identity["records"] == expected_tuning.get("records")
            and tuning_validation_identity["selected_role"]["receipt_sha256"]
            == expected_tuning.get("receipt_sha256")
            and tuning_validation_identity["selected_role"]["candidate_file_sha256"]
            == expected_tuning.get("candidate_file_sha256")
            and c2._teacher_identity_from_manifest(tuning_validation.manifest) == expected_teacher,
            "C3 tuning-validation dependency drifted",
        )

    expected_matrix = {(str(architecture["name"]), seed) for architecture in architectures for seed in seeds}
    prepared: list[PreparedRun] = []
    for path in run_directories:
        resolved = path.expanduser().resolve()
        _require(resolved.is_dir(), f"C3 run directory does not exist: {resolved}")
        checkpoint, _ = c2._load_checkpoint(resolved / "checkpoint.pt")
        settings = _mapping(checkpoint.get("settings"), "C3 routing checkpoint settings")
        routing_architecture = _mapping(settings.get("architecture"), "C3 routing architecture")
        name = routing_architecture.get("name")
        architecture = next(
            (value for value in architectures if value["name"] == name),
            None,
        )
        _require(architecture is not None, f"C3 run uses an unregistered architecture: {name}")
        prepared.append(
            _prepare_run(
                resolved,
                architecture,
                seeds,
                training_commit,
                tuning_validation_identity,
                expected_training,
                expected_teacher,
                expected_wdl,
            )
        )
    observed_matrix = {run.key for run in prepared}
    _require(observed_matrix == expected_matrix and len(prepared) == 9, "C3 run matrix is incomplete or duplicated")

    absolute_name = str(architectures[0]["name"])
    recipe_runs = _mapping(qualified_recipe.get("evaluation"), "qualified recipe evaluation").get("runs")
    _require(isinstance(recipe_runs, list) and len(recipe_runs) == 3, "qualified recipe run evidence drifted")
    recipe_by_seed = {int(run["seed"]): run for run in recipe_runs}
    for run in prepared:
        if run.architecture["name"] != absolute_name:
            continue
        evidence = _mapping(recipe_by_seed.get(run.seed), f"qualified recipe seed {run.seed}")
        _require(
            evidence.get("checkpoint_sha256") == run.checkpoint_sha256
            and evidence.get("training_receipt_sha256") == run.training_receipt_sha256
            and evidence.get("functional_health_receipt_sha256") == run.health_receipt_sha256,
            "C3 absolute run differs from the qualified recipe evidence",
        )

    # The confirmation role is intentionally not opened until every artifact above has passed.
    _load_models(prepared)
    confirmation_resolved = confirmation_path.expanduser().resolve()
    with ConfirmationRoleDataset(confirmation_resolved) as confirmation:
        confirmation_identity = confirmation.identity()
        confirmation_verification, confirmation_verification_sha256 = (
            _validate_confirmation_verification(
                confirmation_verification_path,
                confirmation_identity,
            )
        )
        sides, teacher_scores, results, baseline_scores, predictions = _collect_predictions(
            confirmation,
            c2._constant_inputs(constant),
            prepared,
        )

    baseline_evaluation = c2.evaluate_prediction_scores(
        lookup, sides, teacher_scores, results, baseline_scores
    )
    baseline_mean = float_from_receipt(
        baseline_evaluation["composite_loss_mean_all_records"], "C3 constant mean"
    )
    prepared_by_key = {run.key: run for run in prepared}
    run_receipts: list[dict[str, object]] = []
    for architecture in architectures:
        for seed in seeds:
            run = prepared_by_key[(str(architecture["name"]), seed)]
            evaluation = c2.evaluate_prediction_scores(
                lookup,
                sides,
                teacher_scores,
                results,
                predictions[run.key],
            )
            mean = float_from_receipt(
                evaluation["composite_loss_mean_all_records"],
                f"C3 {architecture['name']} seed {seed} mean",
            )
            delta = baseline_mean - mean
            run_receipts.append(
                {
                    "architecture": architecture["name"],
                    "seed": seed,
                    "directory_name": run.directory.name,
                    "checkpoint_sha256": run.checkpoint_sha256,
                    "training_receipt_sha256": run.training_receipt_sha256,
                    "network_sha256": run.network_sha256,
                    "export_receipt_sha256": run.export_receipt_sha256,
                    "functional_health_receipt_sha256": run.health_receipt_sha256,
                    "tuning_stop_composite_loss": float_receipt(run.stop_validation_loss),
                    "evaluation": evaluation,
                    "paired_delta_constant_minus_checkpoint": float_receipt(delta),
                    "strictly_better_than_constant": delta > 0.0,
                }
            )
    architecture_summaries, pareto_frontier = summarize_architectures(
        architectures,
        seeds,
        baseline_mean,
        run_receipts,
    )
    eligible = [
        str(_mapping(summary["architecture"], "C3 summary architecture")["name"])
        for summary in architecture_summaries
        if summary["confirmation_eligible"]
    ]
    checks = {
        "source_clean": source["dirty"] is False,
        "qualified_recipe_passed": True,
        "exact_three_by_three_matrix": observed_matrix == expected_matrix and len(prepared) == 9,
        "all_final_exposure": True,
        "all_exports_authenticated": True,
        "all_functional_health_pass": True,
        "fresh_role_opened_after_complete_artifact_preflight": True,
        "fresh_role_canonical_verification_passed": True,
        "at_least_one_confirmation_eligible_architecture": bool(eligible),
        "cluster_claim_is_honest": True,
    }
    passed = all(checks.values())
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "contract": {"schema": CONTRACT_SCHEMA, "sha256": contract_sha256},
        "source": source,
        "inputs": {
            "qualified_recipe": {
                "name": qualified_recipe_path.expanduser().resolve().name,
                "sha256": _sha256_bytes(qualified_recipe_payload),
                "schema": qualified_recipe["schema"],
                "arm": expected_recipe["arm"],
            },
            "constant_baseline": {
                "name": constant_path.expanduser().resolve().name,
                "sha256": _sha256_bytes(constant_payload),
                "schema": constant["schema"],
                "training_file": expected_training,
                "constants_cp": {
                    SIDE_NAMES[side]: value
                    for side, value in c2._constant_inputs(constant).items()
                },
            },
            "tuning_validation": tuning_validation_identity,
            "confirmation_role": confirmation_identity,
            "confirmation_verification": {
                "name": confirmation_verification_path.expanduser().resolve().name,
                "sha256": confirmation_verification_sha256,
                "schema": confirmation_verification["schema"],
            },
            "teacher": expected_teacher,
            "wdl_calibration": {
                **expected_wdl,
                "lookup_raw_float32_sha256": lookup.raw_float32_sha256,
                "parameter_float32_sha256": lookup.parameter_float32_sha256,
            },
        },
        "matrix": {
            "architectures": [dict(value) for value in architectures],
            "frozen_seeds": list(seeds),
            "run_count": 9,
            "training_source_commit": training_commit,
            "optimizer_learning_rate_multipliers": {"dense_trunk": 0.1, "output": 0.1},
        },
        "objective": dict(c2.OBJECTIVE_RECEIPT),
        "evaluation": {
            "constant_baseline": baseline_evaluation,
            "architectures": architecture_summaries,
        },
        "diagnostics": {
            "confirmation_eligible_architectures": eligible,
            "loss_size_pareto_frontier": pareto_frontier,
            "loss_selects_architecture": False,
            "fixed_node_strength_is_diagnostic_only": True,
            "equal_time_three_control_gate_required": True,
        },
        "statistics": {
            "unit": "record",
            "sample_identity": "(confirmation_payload_sha256, effective_index)",
            "cluster_identity": None,
            "cluster_identity_reason": "absent from HORDE_BIN_V1",
            "confidence_interval": None,
            "iid_bootstrap": False,
            "game_clustered_claim": False,
            "confirmation_role_status": "fresh and not inspected before complete artifact preflight",
        },
        "gates": {"checks": checks, "passed": passed},
        "claims": {
            "representation_matrix_qualified": passed,
            "architecture_selected": False,
            "best_seed_selected": False,
            "validation_loss_selects_architecture": False,
            "statistical_confidence": False,
            "playing_strength_evidence": False,
            "production_network": False,
            "run6b_production_path_changed": False,
        },
    }
    validate_receipt(receipt)
    return receipt


def validate_receipt(value: object) -> dict[str, object]:
    receipt = _mapping(value, "C3 qualification receipt")
    _require(receipt.get("schema") == SCHEMA, "C3 qualification receipt schema drifted")
    _require(
        receipt.get("contract") == {"schema": CONTRACT_SCHEMA, "sha256": CONTRACT_SHA256},
        "C3 qualification contract receipt drifted",
    )
    inputs = _mapping(receipt.get("inputs"), "C3 qualification inputs")
    confirmation = _mapping(inputs.get("confirmation_role"), "C3 confirmation identity")
    records = confirmation.get("records")
    _require(type(records) is int and records > 0, "C3 confirmation record count is invalid")
    evaluation = _mapping(receipt.get("evaluation"), "C3 qualification evaluation")
    baseline, baseline_mean = c2._validate_evaluation_receipt(
        evaluation.get("constant_baseline"),
        "C3 constant evaluation",
        expected_records=records,
        expected_eligible=None,
    )
    eligible_teacher_scores = int(baseline["eligible_teacher_scores"])
    matrix = _mapping(receipt.get("matrix"), "C3 qualification matrix")
    architectures = matrix.get("architectures")
    seeds = matrix.get("frozen_seeds")
    summaries = evaluation.get("architectures")
    _require(
        isinstance(architectures, list)
        and len(architectures) == 3
        and isinstance(seeds, list)
        and len(seeds) == 3
        and isinstance(summaries, list)
        and len(summaries) == 3,
        "C3 qualification matrix accounting drifted",
    )
    eligible_names: list[str] = []
    flattened_runs: list[Mapping[str, Any]] = []
    for architecture, summary_value in zip(architectures, summaries, strict=True):
        summary = _mapping(summary_value, "C3 architecture summary")
        _require(summary.get("architecture") == architecture, "C3 summary architecture drifted")
        runs = summary.get("runs")
        _require(isinstance(runs, list) and len(runs) == 3, "C3 summary run count drifted")
        losses: list[float] = []
        observed_seeds: list[int] = []
        strict_values: list[bool] = []
        for run_value in runs:
            run = _mapping(run_value, "C3 qualification run")
            _require(run.get("architecture") == architecture["name"], "C3 run architecture drifted")
            observed_seeds.append(int(run.get("seed")))
            candidate, candidate_mean = c2._validate_evaluation_receipt(
                run.get("evaluation"),
                "C3 candidate evaluation",
                expected_records=records,
                expected_eligible=eligible_teacher_scores,
            )
            del candidate
            delta = float_from_receipt(
                run.get("paired_delta_constant_minus_checkpoint"), "C3 paired delta"
            )
            _require(delta == baseline_mean - candidate_mean, "C3 paired delta drifted")
            strict = run.get("strictly_better_than_constant")
            _require(type(strict) is bool and strict is (delta > 0.0), "C3 strict result drifted")
            for field in (
                "checkpoint_sha256",
                "training_receipt_sha256",
                "network_sha256",
                "export_receipt_sha256",
                "functional_health_receipt_sha256",
            ):
                _require(_valid_sha256(run.get(field)), f"C3 run {field} is invalid")
            losses.append(candidate_mean)
            strict_values.append(strict)
            flattened_runs.append(run)
        _require(observed_seeds == seeds, "C3 summary seed order drifted")
        mean_loss = math.fsum(losses) / 3
        _require(
            float_from_receipt(summary.get("three_seed_mean_composite_loss"), "C3 mean loss")
            == mean_loss
            and float_from_receipt(
                summary.get("three_seed_mean_delta_constant_minus_checkpoint"), "C3 mean delta"
            )
            == baseline_mean - mean_loss,
            "C3 architecture mean drifted",
        )
        eligible = all(strict_values)
        _require(
            summary.get("all_three_seeds_strictly_better_than_constant") is eligible
            and summary.get("confirmation_eligible") is eligible,
            "C3 architecture eligibility drifted",
        )
        if eligible:
            eligible_names.append(str(architecture["name"]))
    _require(len(flattened_runs) == 9, "C3 flattened run count drifted")
    diagnostics = _mapping(receipt.get("diagnostics"), "C3 diagnostics")
    _require(
        diagnostics.get("confirmation_eligible_architectures") == eligible_names
        and diagnostics.get("loss_selects_architecture") is False
        and diagnostics.get("fixed_node_strength_is_diagnostic_only") is True
        and diagnostics.get("equal_time_three_control_gate_required") is True,
        "C3 diagnostic claims drifted",
    )
    statistics = _mapping(receipt.get("statistics"), "C3 statistics")
    _require(
        statistics.get("cluster_identity") is None
        and statistics.get("confidence_interval") is None
        and statistics.get("iid_bootstrap") is False
        and statistics.get("game_clustered_claim") is False,
        "C3 statistical claims drifted",
    )
    gates = _mapping(receipt.get("gates"), "C3 gates")
    checks = _mapping(gates.get("checks"), "C3 gate checks")
    _require(
        checks.get("at_least_one_confirmation_eligible_architecture") is bool(eligible_names)
        and gates.get("passed") is all(checks.values()),
        "C3 gate accounting drifted",
    )
    claims = _mapping(receipt.get("claims"), "C3 claims")
    _require(
        claims
        == {
            "representation_matrix_qualified": gates["passed"],
            "architecture_selected": False,
            "best_seed_selected": False,
            "validation_loss_selects_architecture": False,
            "statistical_confidence": False,
            "playing_strength_evidence": False,
            "production_network": False,
            "run6b_production_path_changed": False,
        },
        "C3 qualification claims drifted",
    )
    return dict(receipt)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualified_recipe", type=Path)
    parser.add_argument("constant_baseline", type=Path)
    parser.add_argument("tuning_validation", type=Path)
    parser.add_argument("confirmation_role", type=Path)
    parser.add_argument("confirmation_verification", type=Path)
    parser.add_argument("wdl_calibration", type=Path)
    parser.add_argument("--run-directory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_receipt(
        args.qualified_recipe,
        args.constant_baseline,
        args.tuning_validation,
        args.confirmation_role,
        args.confirmation_verification,
        args.wdl_calibration,
        args.run_directory,
        contract_path=args.contract,
        allow_dirty=args.allow_dirty,
    )
    _write_exclusive(args.output, _canonical_json(receipt))
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "output": str(args.output.expanduser().resolve()),
                "passed": receipt["gates"]["passed"],
                "eligible": receipt["diagnostics"]["confirmation_eligible_architectures"],
                "pareto": receipt["diagnostics"]["loss_size_pareto_frontier"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["gates"]["passed"] or not args.require_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (C3QualificationError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
