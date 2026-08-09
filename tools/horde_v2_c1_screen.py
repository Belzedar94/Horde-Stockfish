#!/usr/bin/env python3
"""Evaluate and screen all nine frozen Horde V2 C1 integer containers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from . import horde_bin_v1 as wire
    from . import horde_v2_c1_campaign as campaign
    from .horde_training_decoder import BLACK, HordeBinV1Dataset, WHITE
    from .horde_v2_container import CONTAINER_SCHEMA, ContainerError, sha256_file
    from .horde_v2_integer_eval import (
        IntegerEvaluationError,
        IntegerNetwork,
        MetricAccumulator,
        loss_arrays,
    )
    from .horde_wdl import CalibrationError, SIDE_NAMES, load_artifact as load_wdl_artifact
except ImportError:
    import horde_bin_v1 as wire
    import horde_v2_c1_campaign as campaign
    from horde_training_decoder import BLACK, HordeBinV1Dataset, WHITE
    from horde_v2_container import CONTAINER_SCHEMA, ContainerError, sha256_file
    from horde_v2_integer_eval import (
        IntegerEvaluationError,
        IntegerNetwork,
        MetricAccumulator,
        loss_arrays,
    )
    from horde_wdl import CalibrationError, SIDE_NAMES, load_artifact as load_wdl_artifact


CONTRACT_SCHEMA = "HORDE_V2_C1_QUANTIZED_SCREEN_V1"
CONTRACT_RELATIVE_PATH = Path("schemas/horde-v2-c1-quantized-screen-v1.json")
CONTRACT_SHA256 = "D7F2A68A19435807FAADE0180968A04ED8C4BDC8D1F2E8AF05FADE989CC4D9DA"
RECEIPT_SCHEMA = "HORDE_V2_C1_QUANTIZED_SCREEN_RECEIPT_V1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ScreenError(ValueError):
    """Raised when C1 screening evidence violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScreenError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


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
        raise ScreenError(f"{label} is invalid JSON: {error}") from error
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


def _finite_float(value: object, label: str) -> float:
    _require(type(value) in (int, float), f"{label} is not numeric")
    converted = float(value)
    _require(math.isfinite(converted), f"{label} is not finite")
    return converted


def _source_identity(root: Path) -> dict[str, object]:
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
        len(commit) == 40 and all(character in "0123456789abcdefABCDEF" for character in commit),
        "screen source is not a full Git identity",
    )
    return {"commit": commit.lower(), "dirty": dirty}


def load_contract(path: Path | None = None) -> tuple[dict[str, Any], str]:
    resolved = (path or REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH).expanduser().resolve()
    contract, payload = _read_json(resolved, "C1 quantized screen contract")
    digest = _sha256_bytes(payload)
    _require(digest == CONTRACT_SHA256, f"C1 screen contract SHA-256 mismatch: {digest}")
    _require(contract.get("schema_name") == CONTRACT_SCHEMA, "C1 screen schema drifted")

    dependencies = _mapping(contract.get("dependencies"), "screen dependencies")
    campaign_dependency = _mapping(
        dependencies.get("campaign_contract"), "campaign contract dependency"
    )
    _require(
        campaign_dependency.get("schema") == campaign.CONTRACT_SCHEMA
        and campaign_dependency.get("sha256") == campaign.CONTRACT_SHA256,
        "screen campaign-contract dependency drifted",
    )
    _require(
        dependencies.get("campaign_plan_schema") == campaign.PLAN_SCHEMA
        and dependencies.get("campaign_verification_schema") == campaign.VERIFICATION_SCHEMA,
        "screen campaign schemas drifted",
    )
    _require(dependencies.get("container_schema") == CONTAINER_SCHEMA, "container schema drifted")
    _require(dependencies.get("dataset_schema") == wire.SCHEMA_NAME, "dataset schema drifted")
    _require(
        dependencies.get("dataset_schema_sha256") == wire.SCHEMA_SHA256,
        "dataset schema hash drifted",
    )
    _require(
        dependencies.get("teacher_network_sha256") == wire.RUN6B_SHA256,
        "screen Run 6B dependency drifted",
    )

    evaluation = _mapping(contract.get("evaluation"), "screen evaluation")
    objective = _mapping(evaluation.get("objective"), "screen objective")
    _require(evaluation.get("dataset_role") == "validation", "screen dataset role drifted")
    _require(evaluation.get("records") == 250_000, "screen validation count drifted")
    _require(evaluation.get("batch_size") == 4096, "screen batch size drifted")
    _require(objective.get("lambda") == 0.6, "screen lambda drifted")
    _require(
        objective.get("calibration_parameters") == "artifact IEEE-754 binary64"
        and objective.get("probability_arithmetic") == "IEEE-754 binary64",
        "screen probability precision drifted",
    )
    health = _mapping(evaluation.get("parameter_health"), "screen parameter health")
    _require(
        health.get("minimum_weight_nonzero_fraction") == 0.01
        and health.get("maximum_weight_dtype_boundary_fraction") == 0.05,
        "screen parameter-health thresholds drifted",
    )

    expected_comparisons = (
        ("rank8_over_absolute", "v2-c1-abs64x192", "v2-c1-rank8-64x192"),
        ("royal32_over_rank8", "v2-c1-rank8-64x192", "v2-64x192"),
        ("royal32_over_absolute", "v2-c1-abs64x192", "v2-64x192"),
    )
    comparisons = contract.get("comparisons")
    _require(isinstance(comparisons, list), "screen comparisons are missing")
    observed_comparisons = tuple(
        (
            comparison.get("id"),
            comparison.get("cheaper"),
            comparison.get("contender"),
        )
        for comparison in comparisons
        if isinstance(comparison, dict)
    )
    _require(observed_comparisons == expected_comparisons, "screen comparisons drifted")

    gates = _mapping(contract.get("gates"), "screen gates")
    _require(gates.get("paired_seed_count") == 3, "screen paired seed count drifted")
    _require(gates.get("last_float_epochs_checked") == 2, "float epoch gate drifted")
    _require(
        gates.get("paired_t_critical_95_df2") == 4.3026527297,
        "paired t critical value drifted",
    )
    nomination = _mapping(contract.get("nomination"), "screen nomination")
    _require(
        nomination.get("predesignated_playing_seed_index") == 0
        and nomination.get("maximum_pairings") == 1,
        "screen nomination policy drifted",
    )
    claims = _mapping(contract.get("claims"), "screen claims")
    _require(
        claims
        == {
            "validation_loss_selects_architecture": False,
            "fixed_node_strength_measured": False,
            "equal_time_strength_measured": False,
            "architecture_selected": False,
            "production_network": False,
            "run6b_production_path_changed": False,
        },
        "screen contract made an unsupported claim",
    )
    return contract, digest


def _white_piece_bin(count: int) -> str:
    for lower, upper in campaign.WHITE_PIECE_BINS:
        if lower <= count <= upper:
            return f"{lower}-{upper}"
    raise ScreenError(f"validation position has {count} White pieces outside C1 bins")


def _read_float_metrics(
    receipt_path: Path,
    metrics_path: Path,
    expected_epochs: int,
) -> dict[str, object]:
    receipt, _ = _read_json(receipt_path, "C1 training receipt")
    run = _mapping(receipt.get("run"), "training run")
    epochs = run.get("epochs_receipt")
    _require(
        isinstance(epochs, list) and len(epochs) == expected_epochs,
        "training epoch metrics are incomplete",
    )
    validation_losses: list[float] = []
    for index, epoch in enumerate(epochs, start=1):
        epoch_value = _mapping(epoch, f"training epoch {index}")
        _require(epoch_value.get("epoch") == index, "training epoch order drifted")
        validation = _mapping(epoch_value.get("validation"), "epoch validation metrics")
        validation_losses.append(
            _finite_float(validation.get("composite_loss"), "epoch validation loss")
        )

    initial_validation = _mapping(run.get("initial_validation"), "initial validation metrics")
    stop_validation = _mapping(run.get("stop_validation"), "stop validation metrics")
    initial_loss = _finite_float(
        initial_validation.get("composite_loss"), "initial validation loss"
    )
    stop_loss = _finite_float(stop_validation.get("composite_loss"), "stop validation loss")
    _require(stop_validation == epochs[-1]["validation"], "stop validation differs from final epoch")
    _require(stop_loss == validation_losses[-1], "final float validation loss drifted")

    _require(metrics_path.is_file(), f"training metrics file is missing: {metrics_path}")
    metric_objects: list[object] = []
    for line_number, raw_line in enumerate(metrics_path.read_bytes().splitlines(), start=1):
        _require(raw_line, f"training metrics line {line_number} is empty")
        try:
            metric_objects.append(json.loads(raw_line.decode("ascii")))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ScreenError(f"training metrics line {line_number} is invalid: {error}") from error
    _require(len(metric_objects) == expected_epochs + 1, "training metrics line count drifted")
    _require(
        metric_objects[0] == {"epoch": 0, "validation": initial_validation},
        "initial validation metrics file entry drifted",
    )
    _require(metric_objects[1:] == epochs, "epoch receipt differs from metrics file")
    return {
        "initial_validation_composite_loss": initial_loss,
        "final_validation_composite_loss": stop_loss,
        "last_two_validation_composite_losses": validation_losses[-2:],
    }


def _validate_file_identity(
    actual: Mapping[str, object], expected: Mapping[str, Any], label: str
) -> None:
    for key in ("name", "sha256", "payload_sha256", "records", "book_sha256", "seed"):
        _require(actual.get(key) == expected.get(key), f"{label} identity field {key} drifted")


def _metric_receipt(
    accumulator: MetricAccumulator, *, allow_empty: bool
) -> dict[str, object]:
    if accumulator.samples:
        return accumulator.receipt()
    _require(allow_empty, "production validation slice is empty")
    return {
        "samples": 0,
        "score_eligible": 0,
        "mate_scores_masked": 0,
        "composite_loss": None,
        "score_half_brier_eligible": None,
        "result_half_brier": None,
        "prediction_mean_wdl": None,
    }


def _paired_interval(values: Sequence[float], critical: float) -> dict[str, float | int]:
    _require(len(values) == 3, "paired comparison does not contain three seeds")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    standard_deviation = math.sqrt(variance)
    half_width = critical * standard_deviation / math.sqrt(len(values))
    return {
        "samples": len(values),
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "half_width_95": half_width,
        "lower_95": mean - half_width,
        "upper_95": mean + half_width,
    }


def compare_architectures(
    comparison: Mapping[str, Any],
    runs_by_architecture: Mapping[str, Mapping[int, Mapping[str, Any]]],
    critical: float,
) -> dict[str, object]:
    cheaper_name = str(comparison["cheaper"])
    contender_name = str(comparison["contender"])
    cheaper_runs = runs_by_architecture.get(cheaper_name)
    contender_runs = runs_by_architecture.get(contender_name)
    _require(cheaper_runs is not None, f"comparison cheaper architecture is missing: {cheaper_name}")
    _require(
        contender_runs is not None,
        f"comparison contender architecture is missing: {contender_name}",
    )
    _require(set(cheaper_runs) == set(contender_runs) == {0, 1, 2}, "paired seeds drifted")

    paired: list[dict[str, object]] = []
    integer_deltas: list[float] = []
    all_float_deltas: list[float] = []
    all_side_deltas: list[float] = []
    health_passed = True
    ranking_not_reversed = True
    for pair_index in range(3):
        cheaper = cheaper_runs[pair_index]
        contender = contender_runs[pair_index]
        cheaper_integer = _mapping(cheaper["integer_validation"], "cheaper integer metrics")
        contender_integer = _mapping(contender["integer_validation"], "contender integer metrics")
        integer_delta = float(
            _mapping(cheaper_integer["overall"], "cheaper overall")["composite_loss"]
            - _mapping(contender_integer["overall"], "contender overall")["composite_loss"]
        )
        integer_deltas.append(integer_delta)

        cheaper_float = _mapping(cheaper["float_validation"], "cheaper float metrics")
        contender_float = _mapping(contender["float_validation"], "contender float metrics")
        cheaper_epochs = cheaper_float["last_two_validation_composite_losses"]
        contender_epochs = contender_float["last_two_validation_composite_losses"]
        _require(
            isinstance(cheaper_epochs, list)
            and isinstance(contender_epochs, list)
            and len(cheaper_epochs) == len(contender_epochs) == 2,
            "last-two-epoch evidence drifted",
        )
        float_deltas = [
            float(cheaper_value - contender_value)
            for cheaper_value, contender_value in zip(
                cheaper_epochs, contender_epochs, strict=True
            )
        ]
        all_float_deltas.extend(float_deltas)
        final_float_delta = float(
            cheaper_float["final_validation_composite_loss"]
            - contender_float["final_validation_composite_loss"]
        )
        ranking_not_reversed = ranking_not_reversed and final_float_delta * integer_delta > 0.0

        side_deltas: dict[str, float] = {}
        for side_name in (SIDE_NAMES[WHITE], SIDE_NAMES[BLACK]):
            cheaper_side = _mapping(
                _mapping(cheaper_integer["side_to_move"], "cheaper side metrics")[side_name],
                "cheaper side slice",
            )
            contender_side = _mapping(
                _mapping(contender_integer["side_to_move"], "contender side metrics")[side_name],
                "contender side slice",
            )
            side_delta = float(cheaper_side["composite_loss"] - contender_side["composite_loss"])
            side_deltas[side_name] = side_delta
            all_side_deltas.append(side_delta)

        pair_health = bool(cheaper["parameter_health"]["passed"]) and bool(
            contender["parameter_health"]["passed"]
        )
        health_passed = health_passed and pair_health
        paired.append(
            {
                "pair_index": pair_index,
                "seed": cheaper["seed"],
                "integer_overall_delta_cheaper_minus_contender": integer_delta,
                "float_last_two_deltas_cheaper_minus_contender": float_deltas,
                "float_final_delta_cheaper_minus_contender": final_float_delta,
                "integer_side_deltas_cheaper_minus_contender": side_deltas,
                "both_parameter_health_passed": pair_health,
            }
        )

    interval = _paired_interval(integer_deltas, critical)
    gates = {
        "integer_overall_positive_every_seed": all(value > 0.0 for value in integer_deltas),
        "paired_integer_lower_95_positive": interval["lower_95"] > 0.0,
        "float_last_two_positive_every_seed": all(value > 0.0 for value in all_float_deltas),
        "integer_side_positive_every_seed": all(value > 0.0 for value in all_side_deltas),
        "integer_final_ranking_not_reversed": ranking_not_reversed,
        "both_architectures_parameter_health_passed": health_passed,
    }
    return {
        "id": comparison["id"],
        "cheaper": cheaper_name,
        "contender": contender_name,
        "paired": paired,
        "paired_integer_overall_interval": interval,
        "gates": gates,
        "passed": all(gates.values()),
    }


def nominate_pairing(
    comparisons: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, object] | None:
    by_id = {str(comparison["id"]): comparison for comparison in comparisons}
    _require(
        set(by_id)
        == {"rank8_over_absolute", "royal32_over_rank8", "royal32_over_absolute"},
        "screen comparison receipt set drifted",
    )
    rank8 = bool(by_id["rank8_over_absolute"]["passed"])
    royal_over_rank8 = bool(by_id["royal32_over_rank8"]["passed"])
    royal_over_absolute = bool(by_id["royal32_over_absolute"]["passed"])

    candidate: str | None = None
    baseline: str | None = None
    reason: str | None = None
    if rank8 and royal_over_rank8 and royal_over_absolute:
        candidate = "v2-64x192"
        baseline = "v2-c1-rank8-64x192"
        reason = "Royal-32 passed both its nearest cheaper control and the absolute control"
    elif rank8:
        candidate = "v2-c1-rank8-64x192"
        baseline = "v2-c1-abs64x192"
        reason = "Rank-8 passed the absolute control while Royal-32 did not clear both controls"
    elif royal_over_rank8 and royal_over_absolute:
        candidate = "v2-64x192"
        baseline = "v2-c1-abs64x192"
        reason = "Royal-32 passed both controls while Rank-8 failed the absolute control"
    if candidate is None or baseline is None or reason is None:
        return None

    selection = _mapping(plan.get("selection"), "campaign selection")
    seed_index = int(selection["predesignated_playing_seed_index"])
    seed = int(selection["predesignated_playing_seed"])
    nomination_contract = _mapping(contract.get("nomination"), "screen nomination contract")
    return {
        "candidate_architecture": candidate,
        "baseline_architecture": baseline,
        "seed_index": seed_index,
        "seed": seed,
        "reason": reason,
        "next_gate": nomination_contract["next_gate"],
    }


def screen_campaign(
    plan_path: Path,
    runs_root: Path,
    validation_path: Path,
    wdl_path: Path,
    *,
    contract_path: Path | None = None,
    campaign_contract_path: Path | None = None,
    _allow_fixture: bool = False,
    _expected_records: int | None = None,
    _source_override: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    contract, contract_sha = load_contract(contract_path)
    plan, plan_payload = _read_json(plan_path, "C1 campaign plan")
    _require(plan_payload == campaign._canonical_json(plan), "C1 campaign plan is not canonical JSON")
    fixture_mode = _mapping(plan.get("claims"), "campaign claims").get("fixture_mode") is True
    _require(_allow_fixture or not fixture_mode, "fixture campaign cannot enter production screen")

    source = (
        dict(_source_override)
        if _source_override is not None
        else _source_identity(REPOSITORY_ROOT)
    )
    plan_source = _mapping(plan.get("source"), "campaign source")
    _require(
        fixture_mode
        or (
            source.get("dirty") is False
            and source.get("commit") == plan_source.get("commit")
        ),
        "screen source must be clean and match the campaign source",
    )

    verification = campaign.verify_campaign(
        plan_path,
        runs_root,
        contract_path=campaign_contract_path,
        _allow_fixture=_allow_fixture,
    )
    verification_claims = _mapping(verification.get("claims"), "campaign verification claims")
    _require(verification_claims.get("nine_runs_complete") is True, "nine C1 runs are incomplete")
    _require(
        verification_claims.get("quantized_containers_authenticated") is True,
        "C1 containers are not authenticated",
    )
    _require(
        verification_claims.get("architecture_selection_eligible") is False,
        "campaign verifier unexpectedly selected an architecture",
    )

    evaluation_contract = _mapping(contract.get("evaluation"), "screen evaluation contract")
    expected_records = _expected_records or int(evaluation_contract["records"])
    _require(
        fixture_mode or expected_records == evaluation_contract["records"],
        "production screen validation count drifted",
    )
    batch_size = int(evaluation_contract["batch_size"])
    configuration = _mapping(plan.get("configuration"), "campaign configuration")
    _require(configuration.get("validation_records") == expected_records, "plan validation count drifted")
    _require(configuration.get("lambda") == 0.6, "plan lambda drifted")

    try:
        wdl_payload, calibration, wdl_sha = load_wdl_artifact(wdl_path)
    except CalibrationError as error:
        raise ScreenError(f"screen WDL calibration is invalid: {error}") from error
    expected_wdl = _mapping(
        _mapping(plan.get("data"), "campaign data").get("wdl_calibration"),
        "campaign WDL identity",
    )
    _require(wdl_sha == expected_wdl.get("sha256"), "screen WDL artifact hash drifted")
    _require(wdl_payload.get("schema") == expected_wdl.get("schema"), "screen WDL schema drifted")

    root = runs_root.expanduser().resolve()
    _require(root.is_dir(), f"runs root does not exist: {root}")
    verification_runs = {
        run["id"]: run for run in verification["runs"] if isinstance(run, dict)
    }
    plan_runs = plan.get("runs")
    _require(isinstance(plan_runs, list) and len(plan_runs) == 9, "screen plan lacks nine runs")
    _require(len(verification_runs) == 9, "screen verification lacks nine runs")

    run_states: dict[str, dict[str, Any]] = {}
    networks: dict[str, IntegerNetwork] = {}
    for planned_run in plan_runs:
        run = _mapping(planned_run, "planned screen run")
        run_id = str(run["id"])
        evidence = _mapping(verification_runs.get(run_id), f"verification evidence for {run_id}")
        run_root = (root / str(run["output_role"])).resolve()
        _require(run_root == root or root in run_root.parents, "screen run path escapes root")
        network = IntegerNetwork.load(run_root / "network.hsv2")
        _require(
            network.container.file_sha256 == evidence.get("network_sha256")
            and network.container.parameter_sha256 == evidence.get("parameter_sha256"),
            f"{run_id} network identity drifted after campaign verification",
        )
        architecture = _mapping(run.get("architecture"), "planned screen architecture")
        _require(
            network.container.spec.architecture == architecture.get("name"),
            f"{run_id} network architecture drifted",
        )
        float_validation = _read_float_metrics(
            run_root / "receipt.json",
            run_root / "metrics.jsonl",
            int(configuration["epochs"]),
        )
        health = network.parameter_health()
        run_states[run_id] = {
            "id": run_id,
            "pair_index": int(run["pair_index"]),
            "seed": int(run["seed"]),
            "architecture": architecture["name"],
            "network_sha256": network.container.file_sha256,
            "parameter_sha256": network.container.parameter_sha256,
            "training_receipt_sha256": evidence["training_receipt_sha256"],
            "checkpoint_sha256": evidence["checkpoint_sha256"],
            "metrics_sha256": evidence["metrics_sha256"],
            "export_receipt_sha256": evidence["export_receipt_sha256"],
            "sample_order_chain_sha256": evidence["sample_order_chain_sha256"],
            "float_validation": float_validation,
            "parameter_health": health,
            "integer_validation": {
                "overall": MetricAccumulator(),
                "side_to_move": {
                    SIDE_NAMES[WHITE]: MetricAccumulator(),
                    SIDE_NAMES[BLACK]: MetricAccumulator(),
                },
                "white_piece_bins": {
                    f"{lower}-{upper}": MetricAccumulator()
                    for lower, upper in campaign.WHITE_PIECE_BINS
                },
            },
        }
        networks[run_id] = network

    expected_validation = _mapping(
        _mapping(plan.get("data"), "campaign data").get("validation_file"),
        "campaign validation identity",
    )
    validation_resolved = validation_path.expanduser().resolve()
    with HordeBinV1Dataset(validation_resolved) as validation:
        _require(len(validation) == expected_records, "screen validation record count drifted")
        validation_identity = {
            "name": validation_resolved.name,
            "sha256": validation.file_sha256,
            "payload_sha256": validation.manifest["payload_sha256"],
            "records": len(validation),
            "book_sha256": validation.manifest["book_sha256"],
            "seed": validation.manifest["generation"]["seed"],
        }
        _validate_file_identity(validation_identity, expected_validation, "validation file")
        for batch in validation.batches(batch_size):
            teacher_scores = np.asarray(batch.scores, dtype=np.int32)
            results = np.asarray(batch.results, dtype=np.int8)
            sides = np.asarray(batch.side_to_move, dtype=np.int8)
            white_counts = np.asarray(batch.white_piece_count, dtype=np.int8)
            for run_id, network in networks.items():
                prediction = network.evaluate(batch)
                terms = loss_arrays(
                    prediction,
                    teacher_scores,
                    results,
                    sides,
                    float(configuration["lambda"]),
                    calibration,
                )
                metrics = run_states[run_id]["integer_validation"]
                metrics["overall"].update(terms)
                for side, side_name in SIDE_NAMES.items():
                    metrics["side_to_move"][side_name].update(terms, sides == side)
                for count in np.unique(white_counts):
                    label = _white_piece_bin(int(count))
                    metrics["white_piece_bins"][label].update(
                        terms, white_counts == count
                    )
    _require(
        sha256_file(validation_resolved) == validation_identity["sha256"],
        "validation file changed during integer screening",
    )
    _require(
        sha256_file(wdl_path.expanduser().resolve()) == wdl_sha,
        "WDL artifact changed during integer screening",
    )

    receipt_runs: list[dict[str, object]] = []
    runs_by_architecture: dict[str, dict[int, Mapping[str, Any]]] = {}
    for planned_run in plan_runs:
        run_id = str(planned_run["id"])
        state = run_states[run_id]
        metrics = state["integer_validation"]
        integer_receipt = {
            "overall": _metric_receipt(metrics["overall"], allow_empty=fixture_mode),
            "side_to_move": {
                name: _metric_receipt(accumulator, allow_empty=fixture_mode)
                for name, accumulator in metrics["side_to_move"].items()
            },
            "white_piece_bins": {
                name: _metric_receipt(accumulator, allow_empty=fixture_mode)
                for name, accumulator in metrics["white_piece_bins"].items()
            },
        }
        state["integer_validation"] = integer_receipt
        public_state = dict(state)
        receipt_runs.append(public_state)
        architecture_runs = runs_by_architecture.setdefault(str(state["architecture"]), {})
        pair_index = int(state["pair_index"])
        _require(pair_index not in architecture_runs, "duplicate architecture seed in screen")
        architecture_runs[pair_index] = public_state

    screen_comparisons = [
        compare_architectures(
            _mapping(comparison, "screen comparison"),
            runs_by_architecture,
            float(_mapping(contract.get("gates"), "screen gates")["paired_t_critical_95_df2"]),
        )
        for comparison in contract["comparisons"]
    ]
    nomination = nominate_pairing(screen_comparisons, plan, contract)
    if fixture_mode:
        nomination = None
    all_health = all(bool(run["parameter_health"]["passed"]) for run in receipt_runs)
    verification_payload = campaign._canonical_json(verification)
    return {
        "schema": RECEIPT_SCHEMA,
        "contract": {
            "path": CONTRACT_RELATIVE_PATH.as_posix(),
            "schema": CONTRACT_SCHEMA,
            "sha256": contract_sha,
        },
        "campaign": {
            "contract_sha256": campaign.CONTRACT_SHA256,
            "plan_sha256": _sha256_bytes(plan_payload),
            "verification_sha256": _sha256_bytes(verification_payload),
            "identity_sha256": plan.get("campaign_identity_sha256"),
            "source": plan_source,
        },
        "screen_environment": {
            "source": source,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "validation": validation_identity,
        "wdl_calibration": {
            "name": wdl_path.expanduser().resolve().name,
            "schema": wdl_payload.get("schema"),
            "sha256": wdl_sha,
        },
        "configuration": {
            "records": expected_records,
            "batch_size": batch_size,
            "lambda": configuration["lambda"],
            "paired_seeds": 3,
        },
        "runs": receipt_runs,
        "comparisons": screen_comparisons,
        "fixed_node_nomination": nomination,
        "claims": {
            "fixture_mode": fixture_mode,
            "nine_integer_containers_evaluated": True,
            "parameter_health_all_passed": all_health,
            "quantized_training_screen_complete": not fixture_mode,
            "fixed_node_pairing_nominated": nomination is not None,
            "architecture_selected": False,
            "fixed_node_strength_evidence": False,
            "equal_time_strength_evidence": False,
            "production_network": False,
            "run6b_production_path_changed": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--wdl-calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--campaign-contract", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = screen_campaign(
        args.plan,
        args.runs_root,
        args.validation,
        args.wdl_calibration,
        contract_path=args.contract,
        campaign_contract_path=args.campaign_contract,
    )
    payload = _canonical_json(result)
    _write_exclusive(args.output, payload)
    print(payload.decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        campaign.CampaignError,
        CalibrationError,
        ContainerError,
        IntegerEvaluationError,
        OSError,
        ScreenError,
        subprocess.SubprocessError,
        wire.FormatError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
