#!/usr/bin/env python3
"""Focused tests for the frozen Horde V2 C3 representation qualifier."""

from __future__ import annotations

import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import horde_training_decoder as decoder  # noqa: E402
import horde_v2_c2_objective as objective  # noqa: E402
import horde_v2_c2_qualification as c2  # noqa: E402
import horde_v2_c3_qualification as qualification  # noqa: E402


PARAMETERS = {
    "white_to_move": (0.83, 0.17, -0.72),
    "black_to_move": (1.07, -0.13, -0.91),
}
SIDES = (decoder.WHITE, decoder.WHITE, decoder.BLACK, decoder.BLACK, decoder.WHITE, decoder.BLACK)
TEACHER = (-900, -200, 200, 900, 32_000, -32_000)
RESULTS = (-1, -1, 1, 1, 1, -1)


def _evaluation(predictions: tuple[int, ...]) -> dict[str, object]:
    return c2.evaluate_prediction_scores(
        objective.build_wdl_lookup(PARAMETERS),
        SIDES,
        TEACHER,
        RESULTS,
        predictions,
    )


def _run(
    architecture: str,
    seed: int,
    baseline_mean: float,
    predictions: tuple[int, ...],
) -> dict[str, object]:
    evaluation = _evaluation(predictions)
    candidate_mean = objective.float_from_receipt(
        evaluation["composite_loss_mean_all_records"], "fixture candidate mean"
    )
    delta = baseline_mean - candidate_mean
    return {
        "architecture": architecture,
        "seed": seed,
        "directory_name": architecture,
        "checkpoint_sha256": "A" * 64,
        "training_receipt_sha256": "B" * 64,
        "network_sha256": "C" * 64,
        "export_receipt_sha256": "D" * 64,
        "functional_health_receipt_sha256": "E" * 64,
        "tuning_stop_composite_loss": objective.float_receipt(0.15),
        "evaluation": evaluation,
        "paired_delta_constant_minus_checkpoint": objective.float_receipt(delta),
        "strictly_better_than_constant": delta > 0.0,
    }


def _receipt() -> dict[str, object]:
    contract, _ = qualification.load_contract()
    architectures = contract["matrix"]["architectures"]
    seeds = contract["matrix"]["frozen_seeds"]
    baseline = _evaluation((0, 0, 0, 0, 0, 0))
    baseline_mean = objective.float_from_receipt(
        baseline["composite_loss_mean_all_records"], "fixture baseline mean"
    )
    predictions = {
        architectures[0]["name"]: (-700, -100, 100, 700, 700, -700),
        architectures[1]["name"]: (-900, -200, 200, 900, 900, -900),
        architectures[2]["name"]: (-900, -200, 200, 900, 900, -900),
    }
    runs = [
        _run(architecture["name"], seed, baseline_mean, predictions[architecture["name"]])
        for architecture in architectures
        for seed in seeds
    ]
    summaries, frontier = qualification.summarize_architectures(
        architectures,
        seeds,
        baseline_mean,
        runs,
    )
    eligible = [summary["architecture"]["name"] for summary in summaries]
    checks = {
        "source_clean": True,
        "qualified_recipe_passed": True,
        "exact_three_by_three_matrix": True,
        "all_final_exposure": True,
        "all_exports_authenticated": True,
        "all_functional_health_pass": True,
        "fresh_role_opened_after_complete_artifact_preflight": True,
        "fresh_role_canonical_verification_passed": True,
        "at_least_one_confirmation_eligible_architecture": True,
        "cluster_claim_is_honest": True,
    }
    return {
        "schema": qualification.SCHEMA,
        "contract": {
            "schema": qualification.CONTRACT_SCHEMA,
            "sha256": qualification.CONTRACT_SHA256,
        },
        "source": {
            "commit": "a" * 40,
            "dirty": False,
            "path": "tools/horde_v2_c3_qualification.py",
            "file_sha256": "F" * 64,
            "python": "3.12.0",
            "implementation": "CPython",
            "torch": "fixture",
        },
        "inputs": {
            "confirmation_role": {"records": len(SIDES)},
        },
        "matrix": {
            "architectures": architectures,
            "frozen_seeds": seeds,
            "run_count": 9,
            "training_source_commit": "1" * 40,
            "optimizer_learning_rate_multipliers": {"dense_trunk": 0.1, "output": 0.1},
        },
        "objective": dict(c2.OBJECTIVE_RECEIPT),
        "evaluation": {
            "constant_baseline": baseline,
            "architectures": summaries,
        },
        "diagnostics": {
            "confirmation_eligible_architectures": eligible,
            "loss_size_pareto_frontier": frontier,
            "loss_selects_architecture": False,
            "fixed_node_strength_is_diagnostic_only": True,
            "equal_time_three_control_gate_required": True,
        },
        "statistics": {
            "unit": "record",
            "sample_identity": "fixture",
            "cluster_identity": None,
            "cluster_identity_reason": "absent from HORDE_BIN_V1",
            "confidence_interval": None,
            "iid_bootstrap": False,
            "game_clustered_claim": False,
            "confirmation_role_status": "fresh",
        },
        "gates": {"checks": checks, "passed": True},
        "claims": {
            "representation_matrix_qualified": True,
            "architecture_selected": False,
            "best_seed_selected": False,
            "validation_loss_selects_architecture": False,
            "statistical_confidence": False,
            "playing_strength_evidence": False,
            "production_network": False,
            "run6b_production_path_changed": False,
        },
    }


def _expect_failure(value: dict[str, object], needle: str) -> None:
    try:
        qualification.validate_receipt(value)
    except (qualification.C3QualificationError, c2.QualificationError) as error:
        if needle not in str(error):
            raise AssertionError(f"unexpected C3 qualification error: {error}") from error
    else:
        raise AssertionError(f"C3 qualification accepted tampering: {needle}")


def main() -> int:
    contract, digest = qualification.load_contract()
    if digest != qualification.CONTRACT_SHA256:
        raise AssertionError("C3 qualification contract hash drifted")
    if contract["schema_name"] != qualification.CONTRACT_SCHEMA:
        raise AssertionError("C3 qualification contract schema drifted")

    receipt = _receipt()
    qualification.validate_receipt(receipt)
    frontier = receipt["diagnostics"]["loss_size_pareto_frontier"]
    if frontier != ["v2-c1-abs64x192", "v2-c1-rank8-64x192"]:
        raise AssertionError(f"C3 loss/size Pareto frontier drifted: {frontier}")

    tampered = copy.deepcopy(receipt)
    tampered["evaluation"]["architectures"][0]["runs"][0][
        "paired_delta_constant_minus_checkpoint"
    ] = objective.float_receipt(0.0)
    _expect_failure(tampered, "paired delta drifted")

    tampered = copy.deepcopy(receipt)
    tampered["evaluation"]["architectures"][1]["confirmation_eligible"] = False
    _expect_failure(tampered, "eligibility drifted")

    tampered = copy.deepcopy(receipt)
    tampered["statistics"]["iid_bootstrap"] = True
    _expect_failure(tampered, "statistical claims drifted")

    tampered = copy.deepcopy(receipt)
    tampered["claims"]["architecture_selected"] = True
    _expect_failure(tampered, "claims drifted")

    print(
        "Horde V2 C3 qualification tests passed: exact 3x3 accounting, fresh-loss "
        "deltas, loss/size Pareto diagnostics and non-selection claims"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
