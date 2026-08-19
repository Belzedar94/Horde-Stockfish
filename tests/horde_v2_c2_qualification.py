#!/usr/bin/env python3
"""Focused tests for the frozen Horde V2 C2 qualification gate."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import horde_training_decoder as decoder  # noqa: E402
import horde_training_control as control  # noqa: E402
import horde_v2_c2_objective as objective  # noqa: E402
import horde_v2_c2_qualification as qualification  # noqa: E402


PARAMETERS = {
    "white_to_move": (0.83, 0.17, -0.72),
    "black_to_move": (1.07, -0.13, -0.91),
}


def _training_identity() -> dict[str, object]:
    return {
        "name": "train.bin",
        "sha256": "1" * 64,
        "payload_sha256": "2" * 64,
        "manifest_sha256": "3" * 64,
        "records": 250_000,
    }


def _teacher_identity() -> dict[str, object]:
    return {
        "source_commit": "1" * 40,
        "producer_sha256": "4" * 64,
        "network": {
            "schema": "HORDETEST_HP_LEGACY_V1",
            "sha256": qualification.wire.RUN6B_SHA256,
        },
        "label_contract": {
            "schema": qualification.wire.LABEL_CONTRACT_NAME,
            "schema_sha256": qualification.wire.LABEL_CONTRACT_SHA256,
        },
    }


def _validation_identity() -> dict[str, object]:
    return {
        "name": "selected-records.bin",
        "sha256": "5" * 64,
        "payload_sha256": "5" * 64,
        "book_sha256": "6" * 64,
        "records": 6,
        "seed": "fixture-seed",
        "selected_role": {
            "candidate_file_sha256": "7" * 64,
            "candidate_payload_sha256": "8" * 64,
            "contract_sha256": qualification.SELECTED_ROLE_CONTRACT_SHA256,
            "decision_chain_sha256": "9" * 64,
            "receipt_name": "receipt.json",
            "receipt_sha256": "A" * 64,
            "record_order_sha256": "B" * 64,
            "schema": qualification.SELECTED_ROLE_SCHEMA,
            "selected_index_sha256": "C" * 64,
        },
    }


def _inputs() -> dict[str, object]:
    return {
        "constant_baseline": {
            "name": "c2-constant-baseline-v1.json",
            "sha256": "D" * 64,
            "schema": qualification.CONSTANT_RECEIPT_SCHEMA,
            "training_file": _training_identity(),
            "constants_cp": {"white_to_move": 380, "black_to_move": -314},
        },
        "validation": _validation_identity(),
        "teacher": _teacher_identity(),
        "wdl_calibration": {
            "name": "wdl-calibration.json",
            "sha256": "E" * 64,
            "schema": qualification.WDL_SCHEMA,
            "link_schema": qualification.WDL_LINK_SCHEMA,
            "selection_sha256": "F" * 64,
            "eligible_records_sha256": "0" * 64,
            "lookup_raw_float32_sha256": "1" * 64,
            "parameter_float32_sha256": "2" * 64,
        },
    }


def _run_data() -> dict[str, object]:
    training = _training_identity()
    train_file = {
        "name": training["name"],
        "sha256": training["sha256"],
        "payload_sha256": training["payload_sha256"],
        "book_sha256": "3" * 64,
        "records": training["records"],
        "seed": "fixture-train",
    }
    return {
        "train_file": train_file,
        "decoder": {
            "train": {
                "record_count": training["records"],
                "source": {
                    "file_sha256": training["sha256"],
                    "payload_sha256": training["payload_sha256"],
                    "manifest_sha256": training["manifest_sha256"],
                },
            }
        },
        "teacher": _teacher_identity(),
        "validation_file": _validation_identity(),
        "wdl_calibration": {
            "name": "wdl-calibration.json",
            "sha256": "E" * 64,
            "schema": qualification.WDL_SCHEMA,
            "link_schema": qualification.WDL_LINK_SCHEMA,
            "selection_sha256": "F" * 64,
            "eligible_records_sha256": "0" * 64,
        },
    }


def _evaluation(predictions: tuple[int, ...]) -> dict[str, object]:
    lookup = objective.build_wdl_lookup(PARAMETERS)
    return qualification.evaluate_prediction_scores(
        lookup,
        (decoder.WHITE, decoder.WHITE, decoder.BLACK, decoder.BLACK, decoder.WHITE, decoder.BLACK),
        (-900, -200, 200, 900, 32_000, -32_000),
        (-1, -1, 1, 1, 1, -1),
        predictions,
    )


def test_contract_and_arm_matrix() -> None:
    contract, digest = qualification.load_contract()
    if contract["schema_name"] != qualification.CONTRACT_SCHEMA:
        raise AssertionError("qualification contract schema drifted")
    if digest != qualification.CONTRACT_SHA256:
        raise AssertionError("qualification contract hash drifted")
    for name, multipliers in qualification.ARMS.items():
        if qualification._arm_name(multipliers) != name:
            raise AssertionError("registered optimizer arm did not round-trip")
    try:
        qualification._arm_name({"dense_trunk": 0.1, "output": 1.0})
    except qualification.QualificationError:
        pass
    else:
        raise AssertionError("combined optimizer-factor arm was accepted")
    try:
        qualification._arm_name(
            {"dense_trunk": 0.1, "output": 0.1, "unregistered_factor": 1.0}
        )
    except qualification.QualificationError:
        pass
    else:
        raise AssertionError("optimizer arm accepted an extra factor")
    if qualification.V2_CHECKPOINT_SCHEMA != "HORDE_V2_BASE_CHECKPOINT_V1":
        raise AssertionError("qualification does not use the trainer checkpoint schema")


def test_trainer_emits_registered_arm_identity() -> None:
    common = {
        "seed": qualification.FROZEN_SEEDS[0],
        "epochs": qualification.EPOCHS,
        "lambda_value": qualification.LAMBDA,
        "learning_rate": 0.0015,
        "scheduler_gamma": 0.987,
        "batch_size": qualification.EVALUATION_BATCH_SIZE,
        "block_size": 65_536,
        "cpu_threads": 1,
    }
    for name, multipliers in qualification.ARMS.items():
        args = SimpleNamespace(
            **common,
            dense_learning_rate_multiplier=multipliers["dense_trunk"],
            output_learning_rate_multiplier=multipliers["output"],
        )
        settings = control._training_settings(
            args,
            qualification.torch.device("cuda"),
            "A" * 64,
            architecture=qualification.ARCHITECTURE["name"],
        )
        if settings.get("optimizer_learning_rate_multipliers") != multipliers:
            raise AssertionError(f"trainer did not persist optimizer arm {name}")
        if qualification._arm_name(settings["optimizer_learning_rate_multipliers"]) != name:
            raise AssertionError(f"trainer optimizer arm {name} did not round-trip")


def test_record_order_objective_and_mate_policy() -> None:
    baseline = _evaluation((0, 0, 0, 0, 0, 0))
    candidate = _evaluation((-900, -200, 200, 900, 900, -900))
    baseline_mean = objective.float_from_receipt(
        baseline["composite_loss_mean_all_records"], "baseline mean"
    )
    candidate_mean = objective.float_from_receipt(
        candidate["composite_loss_mean_all_records"], "candidate mean"
    )
    if not candidate_mean < baseline_mean:
        raise AssertionError("teacher-aligned predictions did not beat the zero control")
    if candidate["eligible_teacher_scores"] != 4:
        raise AssertionError("mate-distance labels were not excluded only from the score term")
    if candidate["records"] != 6:
        raise AssertionError("qualification normalization lost mate records")
    if candidate["prediction_and_label_chain_sha256"] == baseline["prediction_and_label_chain_sha256"]:
        raise AssertionError("prediction chain does not bind predictions")

    reordered = qualification.evaluate_prediction_scores(
        objective.build_wdl_lookup(PARAMETERS),
        (decoder.BLACK, decoder.WHITE, decoder.WHITE, decoder.BLACK, decoder.WHITE, decoder.BLACK),
        (-32_000, -900, -200, 200, 32_000, 900),
        (-1, -1, -1, 1, 1, 1),
        (-900, -900, -200, 200, 900, 900),
    )
    reordered_mean = objective.float_from_receipt(
        reordered["composite_loss_mean_all_records"], "reordered mean"
    )
    if reordered_mean != candidate_mean:
        raise AssertionError("record reordering changed the exact objective value")
    if reordered["prediction_and_label_chain_sha256"] == candidate["prediction_and_label_chain_sha256"]:
        raise AssertionError("record-order chain does not bind ordering")


def test_run_data_provenance_bindings() -> None:
    data = _run_data()
    expected_wdl = data["wdl_calibration"]
    qualification._validate_run_data(
        data,
        expected_training=_training_identity(),
        expected_validation=_validation_identity(),
        expected_teacher=_teacher_identity(),
        expected_wdl=expected_wdl,
        label="fixture run",
    )
    tampered = copy.deepcopy(data)
    tampered["train_file"]["payload_sha256"] = "9" * 64
    try:
        qualification._validate_run_data(
            tampered,
            expected_training=_training_identity(),
            expected_validation=_validation_identity(),
            expected_teacher=_teacher_identity(),
            expected_wdl=expected_wdl,
            label="tampered fixture run",
        )
    except qualification.QualificationError:
        pass
    else:
        raise AssertionError("run data accepted another training byte stream")


def _receipt(strict: bool = True) -> dict[str, object]:
    baseline = _evaluation((0, 0, 0, 0, 0, 0))
    candidate = (
        _evaluation((-900, -200, 200, 900, 900, -900))
        if strict
        else copy.deepcopy(baseline)
    )
    baseline_mean = objective.float_from_receipt(
        baseline["composite_loss_mean_all_records"], "baseline mean"
    )
    candidate_mean = objective.float_from_receipt(
        candidate["composite_loss_mean_all_records"], "candidate mean"
    )
    delta = baseline_mean - candidate_mean
    runs = [
        {
            "seed": seed,
            "directory_name": f"seed-{index + 1}",
            "checkpoint_sha256": "A" * 64,
            "training_receipt_sha256": "B" * 64,
            "functional_health_receipt_sha256": "C" * 64,
            "evaluation": copy.deepcopy(candidate),
            "paired_delta_constant_minus_checkpoint": objective.float_receipt(delta),
            "strictly_better_than_constant": delta > 0.0,
        }
        for index, seed in enumerate(qualification.FROZEN_SEEDS)
    ]
    checks = {
        "source_clean": True,
        "exact_three_frozen_seeds": True,
        "one_registered_arm": True,
        "all_final_exposure": True,
        "all_functional_health_pass": True,
        "all_checkpoint_deltas_strictly_positive": strict,
        "cluster_claim_is_honest": True,
    }
    passed = all(checks.values())
    return {
        "schema": qualification.SCHEMA,
        "contract": {
            "schema": qualification.CONTRACT_SCHEMA,
            "sha256": qualification.CONTRACT_SHA256,
        },
        "source": {
            "commit": "a" * 40,
            "dirty": False,
            "python": "3.12.0",
            "implementation": "CPython",
            "torch": "fixture",
            "tool": "tools/horde_v2_c2_qualification.py",
        },
        "inputs": _inputs(),
        "arm": {
            "name": "dense_trunk_0p1",
            "architecture": dict(qualification.ARCHITECTURE),
            "optimizer_learning_rate_multipliers": dict(
                qualification.ARMS["dense_trunk_0p1"]
            ),
            "frozen_seeds": list(qualification.FROZEN_SEEDS),
            "epochs": qualification.EPOCHS,
            "optimizer_steps": qualification.OPTIMIZER_STEPS,
            "samples_consumed_per_seed": qualification.SAMPLES_CONSUMED,
        },
        "objective": dict(qualification.OBJECTIVE_RECEIPT),
        "evaluation": {"constant_baseline": baseline, "runs": runs},
        "statistics": {
            "unit": "record",
            "sample_identity": "(payload_sha256, local_record_index)",
            "cluster_identity": None,
            "cluster_identity_reason": "absent from HORDE_BIN_V1",
            "confidence_interval": None,
            "iid_bootstrap": False,
            "game_clustered_claim": False,
            "selected_role_status": "qualification/tuning; previously inspected",
        },
        "gates": {"checks": checks, "passed": passed},
        "claims": {
            "recipe_qualified": passed,
            "architecture_selected": False,
            "best_seed_or_epoch_selected": False,
            "arm_ranked_against_other_arm": False,
            "statistical_confidence": False,
            "playing_strength_evidence": False,
            "production_network": False,
            "run6b_production_path_changed": False,
        },
    }


def test_receipt_gate_and_tamper_checks() -> None:
    passing = _receipt(True)
    qualification.validate_receipt(passing)
    failing = _receipt(False)
    qualification.validate_receipt(failing)
    if passing["gates"]["passed"] is not True or failing["gates"]["passed"] is not False:
        raise AssertionError("strict three-seed gate accounting drifted")

    tampered = copy.deepcopy(passing)
    tampered["statistics"]["iid_bootstrap"] = True
    try:
        qualification.validate_receipt(tampered)
    except qualification.QualificationError:
        pass
    else:
        raise AssertionError("qualification receipt accepted an IID bootstrap claim")

    tampered = copy.deepcopy(passing)
    tampered["evaluation"]["runs"][0]["paired_delta_constant_minus_checkpoint"] = (
        objective.float_receipt(math.nextafter(0.0, 1.0))
    )
    try:
        qualification.validate_receipt(tampered)
    except qualification.QualificationError:
        pass
    else:
        raise AssertionError("qualification receipt accepted a tampered paired delta")

    tampered = copy.deepcopy(passing)
    tampered["objective"]["lambda"] = 0.5
    try:
        qualification.validate_receipt(tampered)
    except qualification.QualificationError:
        pass
    else:
        raise AssertionError("qualification receipt accepted another objective")

    tampered = copy.deepcopy(passing)
    tampered["inputs"]["constant_baseline"]["training_file"]["payload_sha256"] = (
        "not-a-sha256"
    )
    try:
        qualification.validate_receipt(tampered)
    except qualification.QualificationError:
        pass
    else:
        raise AssertionError("qualification receipt accepted another training identity")

    tampered = copy.deepcopy(passing)
    tampered["evaluation"]["runs"][0]["evaluation"]["records"] += 1
    try:
        qualification.validate_receipt(tampered)
    except qualification.QualificationError:
        pass
    else:
        raise AssertionError("qualification receipt accepted mismatched record accounting")


def main() -> int:
    test_contract_and_arm_matrix()
    test_trainer_emits_registered_arm_identity()
    test_record_order_objective_and_mate_policy()
    test_run_data_provenance_bindings()
    test_receipt_gate_and_tamper_checks()
    print("Horde V2 C2 qualification tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
