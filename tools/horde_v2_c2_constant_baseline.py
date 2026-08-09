#!/usr/bin/env python3
"""Fit the frozen training-only constant null model for Horde V2 C2."""

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

try:
    from . import horde_bin_v1 as wire
    from .horde_training_decoder import BLACK, HordeBinV1Dataset, WHITE
    from .horde_v2_c2_objective import (
        LAMBDA,
        LOOKUP_SCORE_MAXIMUM,
        LOOKUP_SCORE_MINIMUM,
        OBJECTIVE_SCHEMA,
        aggregate_objective,
        build_wdl_lookup,
        fit_constant_baseline,
        float_from_receipt,
    )
    from .horde_wdl import LINK_SCHEMA, SCHEMA as WDL_SCHEMA, load_artifact
except ImportError:
    import horde_bin_v1 as wire
    from horde_training_decoder import BLACK, HordeBinV1Dataset, WHITE
    from horde_v2_c2_objective import (
        LAMBDA,
        LOOKUP_SCORE_MAXIMUM,
        LOOKUP_SCORE_MINIMUM,
        OBJECTIVE_SCHEMA,
        aggregate_objective,
        build_wdl_lookup,
        fit_constant_baseline,
        float_from_receipt,
    )
    from horde_wdl import LINK_SCHEMA, SCHEMA as WDL_SCHEMA, load_artifact


SCHEMA = "HORDE_V2_C2_CONSTANT_BASELINE_RECEIPT_V1"
CONTRACT_SCHEMA = "HORDE_V2_C2_CONSTANT_BASELINE_V1"
CONTRACT_RELATIVE_PATH = Path("schemas/horde-v2-c2-constant-baseline-v1.json")
CONTRACT_SHA256 = "02890D4C4757B0A7383D8A2BCF4DBCF3BF85DB910FEE119A32C2029BD0CC61CA"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ConstantBaselineError(ValueError):
    """Raised when a constant-baseline input or receipt violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConstantBaselineError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("ascii")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"{label} does not exist: {resolved}")
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ConstantBaselineError(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(value, dict), f"{label} root is not an object")
    return value, raw


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


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
        "constant-baseline source is not a full Git identity",
    )
    return {"commit": commit, "dirty": dirty}


def load_contract(path: Path | None = None) -> tuple[dict[str, Any], str]:
    resolved = (path or REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH).expanduser().resolve()
    contract, raw = _read_json(resolved, "constant-baseline contract")
    digest = _sha256_bytes(raw)
    _require(digest == CONTRACT_SHA256, f"constant-baseline contract SHA-256 mismatch: {digest}")
    _require(contract.get("schema_name") == CONTRACT_SCHEMA, "constant-baseline schema drifted")
    objective = _mapping(contract.get("objective"), "constant-baseline objective")
    fit = _mapping(contract.get("fit"), "constant-baseline fit")
    domain = _mapping(fit.get("constant_domain"), "constant-baseline domain")
    _require(
        objective.get("schema") == OBJECTIVE_SCHEMA
        and objective.get("lambda") == LAMBDA
        and domain.get("minimum") == LOOKUP_SCORE_MINIMUM
        and domain.get("maximum") == LOOKUP_SCORE_MAXIMUM,
        "constant-baseline objective drifted",
    )
    inputs = _mapping(contract.get("inputs"), "constant-baseline inputs")
    _require(
        inputs.get("forbidden")
        == ["validation data", "checkpoints", "functional-health receipts", "test results"],
        "constant-baseline forbidden-input contract drifted",
    )
    _require(
        contract.get("claims")
        == {
            "checkpoint_independent": True,
            "strength_evidence": False,
            "training_split_only": True,
        },
        "constant-baseline claims drifted",
    )
    return contract, digest


def _training_identity(dataset: HordeBinV1Dataset) -> dict[str, object]:
    return {
        "name": dataset.path.name,
        "sha256": dataset.file_sha256,
        "payload_sha256": dataset.manifest["payload_sha256"],
        "manifest_sha256": dataset.manifest_sha256,
        "records": len(dataset),
    }


def build_receipt(
    training_path: Path,
    wdl_path: Path,
    *,
    contract_path: Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, object]:
    _, contract_sha256 = load_contract(contract_path)
    source = _repository_identity(REPOSITORY_ROOT)
    _require(allow_dirty or not source["dirty"], "constant-baseline source tree is dirty")
    try:
        wdl_payload, parameters, wdl_sha256 = load_artifact(wdl_path)
    except ValueError as error:
        raise ConstantBaselineError(f"WDL calibration is invalid: {error}") from error
    wdl_source = _mapping(wdl_payload.get("source"), "WDL source")
    expected_training = _mapping(wdl_source.get("training_file"), "WDL training identity")

    with HordeBinV1Dataset(training_path) as dataset:
        observed_training = _training_identity(dataset)
        _require(
            dict(expected_training) == observed_training,
            "WDL calibration was not fitted from the exact constant-baseline training split",
        )
        lookup = build_wdl_lookup(parameters)
        aggregated = aggregate_objective(dataset, lookup)
    fit = fit_constant_baseline(aggregated, lookup)

    side_fits = _mapping(fit.get("sides"), "constant side fits")
    checks = {
        "source_clean": source["dirty"] is False,
        "both_sides_present": all(count > 0 for count in aggregated.records_by_side),
        "both_sides_have_eligible_teacher_scores": all(
            count > 0 for count in aggregated.eligible_by_side
        ),
        "selected_constants_are_interior": all(
            side_fits[name]["boundary_hit"] is False
            for name in ("white_to_move", "black_to_move")
        ),
        "loss_is_finite": math.isfinite(
            float_from_receipt(fit["composite_loss_mean_all_records"], "constant loss")
        ),
        "exact_moments_match_histograms": all(
            side_fits[name]["exact_integer_audit"]["moment_equals_histogram"] is True
            for name in ("white_to_move", "black_to_move")
        ),
    }
    groups = [
        {
            "side": "white_to_move" if group.side == WHITE else "black_to_move",
            "rule50_count": group.rule50_count,
            "records": group.records,
            "eligible_records": group.eligible_records,
            "result_counts_loss_draw_win": list(group.result_counts),
            "teacher_score_support": len(group.teacher_histogram),
        }
        for group in aggregated.groups
    ]
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "contract": {"schema": CONTRACT_SCHEMA, "sha256": contract_sha256},
        "source": {
            "training_file": observed_training,
            "teacher": wdl_source["teacher"],
            "wdl_calibration": {
                "name": wdl_path.expanduser().resolve().name,
                "sha256": wdl_sha256,
                "schema": WDL_SCHEMA,
                "link_schema": LINK_SCHEMA,
                "selection_sha256": wdl_payload["selection"]["selection_sha256"],
                "eligible_records_sha256": wdl_payload["selection"][
                    "eligible_records_sha256"
                ],
            },
            "software": {
                "commit": source["commit"],
                "dirty": source["dirty"],
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "tool": "tools/horde_v2_c2_constant_baseline.py",
            },
        },
        "objective": {
            "schema": OBJECTIVE_SCHEMA,
            "lambda": LAMBDA,
            "lookup": {
                "score_minimum": LOOKUP_SCORE_MINIMUM,
                "score_maximum": LOOKUP_SCORE_MAXIMUM,
                "layout": "side_to_move, integer_score, loss_draw_win",
                "storage_dtype": "IEEE-754 binary32 little-endian",
                "evaluation_dtype": "IEEE-754 binary64",
                "raw_float32_sha256": lookup.raw_float32_sha256,
                "parameter_float32_sha256": lookup.parameter_float32_sha256,
            },
            "runtime": lookup.runtime,
            "rule50": "integer sign/floor/truncation toward zero",
            "normalization": "all records",
            "mate_policy": "score term excluded; result term retained",
        },
        "aggregation": {
            "total_records": aggregated.total_records,
            "eligible_records": aggregated.eligible_records,
            "mate_records": aggregated.mate_records,
            "records_by_side": {
                "white_to_move": aggregated.records_by_side[WHITE],
                "black_to_move": aggregated.records_by_side[BLACK],
            },
            "eligible_records_by_side": {
                "white_to_move": aggregated.eligible_by_side[WHITE],
                "black_to_move": aggregated.eligible_by_side[BLACK],
            },
            "mate_records_by_side": {
                "white_to_move": aggregated.mate_by_side[WHITE],
                "black_to_move": aggregated.mate_by_side[BLACK],
            },
            "selection_sha256": aggregated.selection_sha256,
            "grouped_histogram_sha256": aggregated.grouped_histogram_sha256,
            "groups": groups,
        },
        "fit": fit,
        "gates": {"checks": checks, "passed": all(checks.values())},
        "claims": {
            "training_split_only": True,
            "checkpoint_independent": True,
            "validation_inspected": False,
            "strength_evidence": False,
        },
    }
    validate_receipt(receipt)
    return receipt


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
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def validate_receipt(value: object) -> dict[str, object]:
    _require(isinstance(value, dict), "constant-baseline receipt root is not an object")
    receipt = value
    _require(
        set(receipt)
        == {"schema", "contract", "source", "objective", "aggregation", "fit", "gates", "claims"},
        "constant-baseline receipt fields are incomplete",
    )
    _require(receipt.get("schema") == SCHEMA, "constant-baseline receipt schema mismatch")
    _require(
        receipt.get("contract") == {"schema": CONTRACT_SCHEMA, "sha256": CONTRACT_SHA256},
        "constant-baseline contract identity mismatch",
    )
    objective = _mapping(receipt.get("objective"), "constant-baseline objective")
    lookup = _mapping(objective.get("lookup"), "constant-baseline lookup")
    runtime = _mapping(objective.get("runtime"), "constant-baseline runtime")
    _require(
        set(objective)
        == {"schema", "lambda", "lookup", "runtime", "rule50", "normalization", "mate_policy"}
        and objective.get("schema") == OBJECTIVE_SCHEMA
        and objective.get("lambda") == LAMBDA
        and objective.get("rule50") == "integer sign/floor/truncation toward zero"
        and objective.get("normalization") == "all records"
        and objective.get("mate_policy") == "score term excluded; result term retained"
        and set(lookup)
        == {
            "score_minimum",
            "score_maximum",
            "layout",
            "storage_dtype",
            "evaluation_dtype",
            "raw_float32_sha256",
            "parameter_float32_sha256",
        }
        and lookup.get("score_minimum") == LOOKUP_SCORE_MINIMUM
        and lookup.get("score_maximum") == LOOKUP_SCORE_MAXIMUM
        and lookup.get("layout") == "side_to_move, integer_score, loss_draw_win"
        and lookup.get("storage_dtype") == "IEEE-754 binary32 little-endian"
        and lookup.get("evaluation_dtype") == "IEEE-754 binary64"
        and _valid_sha256(lookup.get("raw_float32_sha256"))
        and _valid_sha256(lookup.get("parameter_float32_sha256")),
        "constant-baseline objective identity is invalid",
    )
    _require(
        set(runtime)
        == {
            "device",
            "torch_version",
            "numpy_version",
            "torch_threads",
            "torch_interop_threads",
            "deterministic_algorithms",
            "mkldnn_enabled",
        }
        and runtime.get("device") == "cpu"
        and isinstance(runtime.get("torch_version"), str)
        and bool(runtime["torch_version"])
        and isinstance(runtime.get("numpy_version"), str)
        and bool(runtime["numpy_version"])
        and runtime.get("torch_threads") == 1
        and runtime.get("torch_interop_threads") == 1
        and runtime.get("deterministic_algorithms") is True
        and runtime.get("mkldnn_enabled") is False,
        "constant-baseline runtime is not canonical",
    )
    aggregation = _mapping(receipt.get("aggregation"), "constant-baseline aggregation")
    _require(
        set(aggregation)
        == {
            "total_records",
            "eligible_records",
            "mate_records",
            "records_by_side",
            "eligible_records_by_side",
            "mate_records_by_side",
            "selection_sha256",
            "grouped_histogram_sha256",
            "groups",
        },
        "constant-baseline aggregation fields are incomplete",
    )
    total = aggregation.get("total_records")
    eligible = aggregation.get("eligible_records")
    mates = aggregation.get("mate_records")
    _require(
        type(total) is int
        and type(eligible) is int
        and type(mates) is int
        and total > 0
        and eligible > 0
        and total == eligible + mates
        and _valid_sha256(aggregation.get("selection_sha256"))
        and _valid_sha256(aggregation.get("grouped_histogram_sha256")),
        "constant-baseline record accounting is invalid",
    )
    side_names = ("white_to_move", "black_to_move")
    records_by_side = _mapping(
        aggregation.get("records_by_side"), "constant-baseline side records"
    )
    eligible_by_side = _mapping(
        aggregation.get("eligible_records_by_side"), "constant-baseline eligible side records"
    )
    mate_by_side = _mapping(
        aggregation.get("mate_records_by_side"), "constant-baseline mate side records"
    )
    _require(
        all(
            set(mapping) == set(side_names)
            for mapping in (records_by_side, eligible_by_side, mate_by_side)
        )
        and all(
            type(mapping[name]) is int and mapping[name] >= 0
            for mapping in (records_by_side, eligible_by_side, mate_by_side)
            for name in side_names
        )
        and all(records_by_side[name] > 0 and eligible_by_side[name] > 0 for name in side_names)
        and all(
            records_by_side[name] == eligible_by_side[name] + mate_by_side[name]
            for name in side_names
        )
        and sum(records_by_side.values()) == total
        and sum(eligible_by_side.values()) == eligible
        and sum(mate_by_side.values()) == mates,
        "constant-baseline side accounting is invalid",
    )
    groups = aggregation.get("groups")
    _require(isinstance(groups, list) and bool(groups), "constant-baseline groups are missing")
    observed_order: list[tuple[int, int]] = []
    group_records = {name: 0 for name in side_names}
    group_eligible = {name: 0 for name in side_names}
    for group_value in groups:
        group = _mapping(group_value, "constant-baseline group")
        _require(
            set(group)
            == {
                "side",
                "rule50_count",
                "records",
                "eligible_records",
                "result_counts_loss_draw_win",
                "teacher_score_support",
            },
            "constant-baseline group fields are invalid",
        )
        name = group.get("side")
        clock = group.get("rule50_count")
        group_count = group.get("records")
        group_eligible_count = group.get("eligible_records")
        results = group.get("result_counts_loss_draw_win")
        support = group.get("teacher_score_support")
        _require(
            name in side_names
            and type(clock) is int
            and 0 <= clock <= 100
            and type(group_count) is int
            and group_count > 0
            and type(group_eligible_count) is int
            and 0 <= group_eligible_count <= group_count
            and isinstance(results, list)
            and len(results) == 3
            and all(type(count) is int and count >= 0 for count in results)
            and sum(results) == group_count
            and type(support) is int
            and 0 <= support <= group_eligible_count
            and (support > 0) is (group_eligible_count > 0),
            "constant-baseline group accounting is invalid",
        )
        observed_order.append((side_names.index(name), clock))
        group_records[name] += group_count
        group_eligible[name] += group_eligible_count
    _require(
        observed_order == sorted(set(observed_order))
        and group_records == dict(records_by_side)
        and group_eligible == dict(eligible_by_side),
        "constant-baseline group order or totals drifted",
    )

    source = _mapping(receipt.get("source"), "constant-baseline source")
    _require(
        set(source) == {"training_file", "teacher", "wdl_calibration", "software"},
        "constant-baseline source fields are incomplete",
    )
    training = _mapping(source.get("training_file"), "constant-baseline training identity")
    teacher = _mapping(source.get("teacher"), "constant-baseline teacher identity")
    calibration = _mapping(source.get("wdl_calibration"), "constant-baseline WDL identity")
    software = _mapping(source.get("software"), "constant-baseline software identity")
    _require(
        set(training) == {"name", "sha256", "payload_sha256", "manifest_sha256", "records"}
        and isinstance(training.get("name"), str)
        and bool(training["name"])
        and all(
            _valid_sha256(training.get(field))
            for field in ("sha256", "payload_sha256", "manifest_sha256")
        )
        and training.get("records") == total,
        "constant-baseline training identity is invalid",
    )
    _require(
        set(teacher) == {"source_commit", "producer_sha256", "network", "label_contract"}
        and _valid_commit(teacher.get("source_commit"))
        and _valid_sha256(teacher.get("producer_sha256"))
        and teacher.get("network")
        == {"schema": "HORDETEST_HP_LEGACY_V1", "sha256": wire.RUN6B_SHA256}
        and teacher.get("label_contract")
        == {"schema": wire.LABEL_CONTRACT_NAME, "schema_sha256": wire.LABEL_CONTRACT_SHA256},
        "constant-baseline teacher identity is invalid",
    )
    _require(
        set(calibration)
        == {
            "name",
            "sha256",
            "schema",
            "link_schema",
            "selection_sha256",
            "eligible_records_sha256",
        }
        and isinstance(calibration.get("name"), str)
        and bool(calibration["name"])
        and calibration.get("schema") == WDL_SCHEMA
        and calibration.get("link_schema") == LINK_SCHEMA
        and all(
            _valid_sha256(calibration.get(field))
            for field in ("sha256", "selection_sha256", "eligible_records_sha256")
        ),
        "constant-baseline WDL identity is invalid",
    )
    _require(
        set(software) == {"commit", "dirty", "python", "implementation", "tool"}
        and _valid_commit(software.get("commit"))
        and type(software.get("dirty")) is bool
        and all(
            isinstance(software.get(field), str) and bool(software[field])
            for field in ("python", "implementation", "tool")
        )
        and software.get("tool") == "tools/horde_v2_c2_constant_baseline.py",
        "constant-baseline software identity is invalid",
    )

    fit = _mapping(receipt.get("fit"), "constant-baseline fit")
    _require(
        set(fit) == {"sides", "composite_loss_sum", "composite_loss_mean_all_records"},
        "constant-baseline fit fields are incomplete",
    )
    sides = _mapping(fit.get("sides"), "constant-baseline sides")
    _require(set(sides) == set(side_names), "constant-baseline lost one side")
    decoded_side_losses: list[float] = []
    for name in side_names:
        side = _mapping(sides[name], f"{name} constant fit")
        _require(
            set(side)
            == {
                "side",
                "selected_constant_cp",
                "minimizer_count",
                "minimizer_minimum_cp",
                "minimizer_maximum_cp",
                "minimizer_list_sha256",
                "boundary_hit",
                "loss_sum",
                "score_half_brier_sum",
                "result_half_brier_sum",
                "runner_up_loss_sum",
                "runner_up_gap",
                "recordwise_audit",
                "exact_integer_audit",
            }
            and side.get("side") == name,
            f"{name} constant fit fields are invalid",
        )
        constant = side.get("selected_constant_cp")
        minimum = side.get("minimizer_minimum_cp")
        maximum = side.get("minimizer_maximum_cp")
        _require(
            type(constant) is int
            and LOOKUP_SCORE_MINIMUM <= constant <= LOOKUP_SCORE_MAXIMUM
            and type(side.get("minimizer_count")) is int
            and 0 < side["minimizer_count"] <= LOOKUP_SCORE_MAXIMUM - LOOKUP_SCORE_MINIMUM + 1
            and type(minimum) is int
            and type(maximum) is int
            and LOOKUP_SCORE_MINIMUM <= minimum <= constant <= maximum <= LOOKUP_SCORE_MAXIMUM
            and _valid_sha256(side.get("minimizer_list_sha256")),
            f"{name} constant fit is invalid",
        )
        _require(
            type(side.get("boundary_hit")) is bool
            and side["boundary_hit"]
            is (minimum == LOOKUP_SCORE_MINIMUM or maximum == LOOKUP_SCORE_MAXIMUM),
            f"{name} boundary accounting is invalid",
        )
        decoded_values: dict[str, float] = {}
        for field in (
            "loss_sum",
            "score_half_brier_sum",
            "result_half_brier_sum",
            "runner_up_loss_sum",
            "runner_up_gap",
        ):
            decoded = float_from_receipt(side.get(field), f"{name}.{field}")
            _require(decoded >= 0.0, f"{name}.{field} is negative")
            decoded_values[field] = decoded
        _require(
            decoded_values["runner_up_loss_sum"] >= decoded_values["loss_sum"]
            and decoded_values["runner_up_gap"]
            == decoded_values["runner_up_loss_sum"] - decoded_values["loss_sum"],
            f"{name} runner-up accounting is invalid",
        )
        decoded_side_losses.append(decoded_values["loss_sum"])
        audit = _mapping(side.get("recordwise_audit"), f"{name} recordwise audit")
        _require(
            set(audit)
            == {
                "loss_sum",
                "score_half_brier_sum",
                "result_half_brier_sum",
                "absolute_loss_difference",
                "loss_ulp_distance",
            },
            f"{name} recordwise audit fields are invalid",
        )
        reference_loss = float_from_receipt(audit.get("loss_sum"), f"{name} audit loss")
        reference_score = float_from_receipt(
            audit.get("score_half_brier_sum"), f"{name} audit score"
        )
        reference_result = float_from_receipt(
            audit.get("result_half_brier_sum"), f"{name} audit result"
        )
        absolute_difference = float_from_receipt(
            audit.get("absolute_loss_difference"), f"{name} audit difference"
        )
        _require(
            reference_loss >= 0.0
            and reference_score >= 0.0
            and reference_result >= 0.0
            and absolute_difference == abs(decoded_values["loss_sum"] - reference_loss)
            and type(audit.get("loss_ulp_distance")) is int
            and audit["loss_ulp_distance"] >= 0,
            f"{name} recordwise audit is invalid",
        )
        exact = _mapping(side.get("exact_integer_audit"), f"{name} exact integer audit")
        _require(
            set(exact)
            == {
                "float32_common_scale_power",
                "half_brier_denominator_factor",
                "lambda_numerator_score_result",
                "lambda_denominator",
                "moment_equals_histogram",
                "score_numerator_sha256",
                "result_numerator_sha256",
                "composite_numerator_sha256",
                "composite_numerator_bits",
                "lookup_values_used",
            }
            and exact.get("float32_common_scale_power") == 149
            and exact.get("half_brier_denominator_factor") == 2
            and exact.get("lambda_numerator_score_result") == [3, 2]
            and exact.get("lambda_denominator") == 5
            and exact.get("moment_equals_histogram") is True
            and all(
                _valid_sha256(exact.get(field))
                for field in (
                    "score_numerator_sha256",
                    "result_numerator_sha256",
                    "composite_numerator_sha256",
                )
            )
            and type(exact.get("composite_numerator_bits")) is int
            and exact["composite_numerator_bits"] > 0
            and type(exact.get("lookup_values_used")) is int
            and exact["lookup_values_used"] > 0,
            f"{name} exact integer audit is invalid",
        )
    loss_sum = float_from_receipt(fit.get("composite_loss_sum"), "constant composite sum")
    mean = float_from_receipt(
        fit.get("composite_loss_mean_all_records"), "constant composite mean"
    )
    _require(
        loss_sum == math.fsum(decoded_side_losses)
        and mean == loss_sum / total
        and 0.0 <= mean <= 1.0,
        "constant composite loss accounting is invalid",
    )
    gates = _mapping(receipt.get("gates"), "constant-baseline gates")
    checks = _mapping(gates.get("checks"), "constant-baseline checks")
    expected_checks = {
        "source_clean": software["dirty"] is False,
        "both_sides_present": all(records_by_side[name] > 0 for name in side_names),
        "both_sides_have_eligible_teacher_scores": all(
            eligible_by_side[name] > 0 for name in side_names
        ),
        "selected_constants_are_interior": all(
            sides[name]["boundary_hit"] is False for name in side_names
        ),
        "loss_is_finite": math.isfinite(mean),
        "exact_moments_match_histograms": all(
            sides[name]["exact_integer_audit"]["moment_equals_histogram"] is True
            for name in side_names
        ),
    }
    _require(
        set(gates) == {"checks", "passed"}
        and checks == expected_checks
        and type(gates.get("passed")) is bool
        and all(type(check) is bool for check in checks.values())
        and gates["passed"] is all(checks.values()),
        "constant-baseline gate accounting is invalid",
    )
    _require(
        receipt.get("claims")
        == {
            "training_split_only": True,
            "checkpoint_independent": True,
            "validation_inspected": False,
            "strength_evidence": False,
        },
        "constant-baseline receipt claims drifted",
    )
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the frozen training-only constant Horde V2 C2 baseline."
    )
    parser.add_argument("training", type=Path, help="authenticated HORDE_BIN_V1 training split")
    parser.add_argument("wdl_calibration", type=Path, help="training-fitted WDL calibration")
    parser.add_argument("--output", type=Path, required=True, help="new receipt path")
    parser.add_argument("--contract", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_receipt(
        args.training,
        args.wdl_calibration,
        contract_path=args.contract,
        allow_dirty=args.allow_dirty,
    )
    _write_exclusive(args.output, _canonical_json(receipt))
    fit = _mapping(receipt["fit"], "constant fit")
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "output": str(args.output.expanduser().resolve()),
                "loss": fit["composite_loss_mean_all_records"],
                "constants": {
                    name: side["selected_constant_cp"]
                    for name, side in _mapping(fit["sides"], "constant sides").items()
                },
                "passed": receipt["gates"]["passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["gates"]["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ConstantBaselineError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
