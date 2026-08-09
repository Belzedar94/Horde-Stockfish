#!/usr/bin/env python3
"""Fail-closed tests for the frozen Horde V2 C1 campaign."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import horde_training_resume as fixtures  # noqa: E402
import horde_training_control as training  # noqa: E402
import horde_v2_c1_campaign as campaign  # noqa: E402
import horde_v2_container as container  # noqa: E402


TRAIN_RECORDS = 192
VALIDATION_RECORDS = 96
SOURCE = {"commit": "a" * 40, "dirty": False}


def _expect_failure(callable_object: object, needle: str) -> None:
    try:
        callable_object()
    except campaign.CampaignError as error:
        if needle not in str(error):
            raise AssertionError(f"unexpected campaign error: {error}") from error
    else:
        raise AssertionError(f"campaign unexpectedly accepted invalid input: {needle}")


def _write_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    train = root / "train.bin"
    validation = root / "validation.bin"
    split = root / "book-split.json"
    calibration = root / "wdl-calibration.json"
    fixtures._write_dataset(
        train,
        first=0,
        count=TRAIN_RECORDS,
        book_sha256="A" * 64,
        seed=101,
    )
    fixtures._write_dataset(
        validation,
        first=TRAIN_RECORDS,
        count=VALIDATION_RECORDS,
        book_sha256="B" * 64,
        seed=202,
    )
    fixtures._write_split_receipt(
        split,
        train_count=TRAIN_RECORDS,
        validation_count=VALIDATION_RECORDS,
        train_book_sha256="A" * 64,
        validation_book_sha256="B" * 64,
    )
    fixtures._write_wdl_calibration(calibration, train)
    return train, validation, split, calibration


def _plan(
    train: Path,
    validation: Path,
    split: Path,
    calibration: Path,
) -> dict[str, object]:
    return campaign.plan_campaign(
        train,
        validation,
        split,
        calibration,
        _expected_records=(TRAIN_RECORDS, VALIDATION_RECORDS),
        _source_override=SOURCE,
    )


def _environment() -> dict[str, object]:
    return {
        "python": "3.12.0",
        "pytorch": "2.7.1+cu118",
        "platform": "Windows-10-10.0.19045-SP0",
        "device": {
            "type": "cuda",
            "index": 0,
            "name": "NVIDIA GeForce RTX 3080",
            "capability": [8, 6],
            "cuda": "11.8",
            "cudnn": 8700,
            "cpu_threads": 1,
            "deterministic_algorithms": True,
            "mkldnn_enabled": False,
            "cublas_workspace_config": ":4096:8",
        },
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "amp": False,
    }


def _write_completed_runs(root: Path, plan: dict[str, object]) -> None:
    configuration = plan["configuration"]
    for run in plan["runs"]:
        run_root = root / run["output_role"]
        run_root.mkdir(parents=True)
        checkpoint = run_root / "checkpoint.pt"
        metrics = run_root / "metrics.jsonl"
        checkpoint.write_bytes(f"checkpoint:{run['id']}".encode("ascii"))
        architecture_loss = {
            "v2-c1-abs64x192": 0.30,
            "v2-c1-rank8-64x192": 0.20,
            "v2-64x192": 0.10,
        }[run["architecture"]["name"]]
        final_loss = architecture_loss + 0.001 * run["pair_index"]
        initial_validation = {
            "samples": configuration["validation_records"],
            "composite_loss": final_loss + 0.01,
        }
        epoch_receipts = [
            {
                "epoch": epoch + 1,
                "train": {
                    "samples": configuration["training_records"],
                    "composite_loss": final_loss + 0.02 - epoch * 0.001,
                },
                "validation": {
                    "samples": configuration["validation_records"],
                    "composite_loss": final_loss
                    + (configuration["epochs"] - epoch - 1) * 0.001,
                },
            }
            for epoch in range(configuration["epochs"])
        ]
        metric_objects = [
            {"epoch": 0, "validation": initial_validation},
            *epoch_receipts,
        ]
        metrics.write_bytes(
            b"\n".join(
                json.dumps(
                    metric,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("ascii")
                for metric in metric_objects
            )
            + b"\n"
        )
        checkpoint_sha = container.sha256_file(checkpoint)
        metrics_sha = container.sha256_file(metrics)
        sample_order = campaign.sample_order_chain_sha256(
            configuration["training_records"],
            configuration["batch_size"],
            configuration["block_size"],
            run["seed"],
            configuration["epochs"],
            plan["data"]["train_file"]["payload_sha256"],
        )
        receipt = {
            "schema": campaign.TRAINING_RECEIPT_SCHEMA,
            "architecture": {
                "name": run["architecture"]["name"],
                "schema": run["architecture"]["schema"],
                "structural_sha256": run["architecture"]["training_structural_sha256"],
                "serialized_parameter_bytes": run["architecture"][
                    "serialized_parameter_bytes"
                ],
            },
            "source": plan["source"],
            "data": plan["data"],
            "campaign": {
                "schema": "HORDE_V2_C1_TRAINER_BINDING_V1",
                "campaign_plan_sha256": campaign._sha256_bytes(
                    campaign._canonical_json(plan)
                ),
                "parent_contract_sha256": plan["contract"]["sha256"],
                "coverage_addendum_sha256": plan["contract"][
                    "coverage_addendum"
                ]["sha256"],
                "effective_contract_sha256": plan["contract"]["effective_sha256"],
                "campaign_identity_sha256": plan["campaign_identity_sha256"],
                "campaign_run_id": run["id"],
            },
            "environment": _environment(),
            "labels": {"lambda": configuration["lambda"]},
            "optimizer": {
                "name": "torch.optim.RAdam",
                "betas": [0.9, 0.999],
                "epsilon": 1.0e-7,
                "weight_decay": 0.0,
                "foreach": False,
                "base_learning_rate": configuration["learning_rate"],
                "output_learning_rate_multiplier": 0.1,
                "lookahead": False,
                "gradient_centralization": False,
                "scheduler": {
                    "name": "StepLR",
                    "step_size_epochs": 1,
                    "gamma": configuration["scheduler_gamma"],
                },
            },
            "run": {
                "seed": run["seed"],
                "complete": True,
                "target_epochs": configuration["epochs"],
                "target_steps": configuration["optimizer_steps_per_model"],
                "optimizer_steps": configuration["optimizer_steps_per_model"],
                "samples_consumed": configuration["exposures_per_model"],
                "batch_size": configuration["batch_size"],
                "shuffle": {"block_size": configuration["block_size"]},
                "initial_validation": initial_validation,
                "stop_validation": epoch_receipts[-1]["validation"],
                "epochs_receipt": epoch_receipts,
                "sample_order_chain_sha256": sample_order,
            },
            "artifacts": {
                "checkpoint": {"name": "checkpoint.pt", "sha256": checkpoint_sha},
                "metrics": {"name": "metrics.jsonl", "sha256": metrics_sha},
            },
            "claims": {
                "integration_only": True,
                "strength_eligible": False,
                "strength_evidence": False,
                "production_network": False,
            },
        }
        receipt_path = run_root / "receipt.json"
        receipt_path.write_bytes(campaign._canonical_json(receipt))

        spec = container.SPECS_BY_ARCHITECTURE[run["architecture"]["name"]]
        provenance = {
            "checkpoint_sha256": checkpoint_sha,
            "container_schema": container.CONTAINER_SCHEMA,
            "source_commit": plan["source"]["commit"],
            "source_dirty": False,
            "train_file_sha256": plan["data"]["train_file"]["sha256"],
            "training_architecture_structural_sha256": spec.training_structural_sha256,
            "training_receipt_sha256": container.sha256_file(receipt_path),
            "validation_file_sha256": plan["data"]["validation_file"]["sha256"],
            "wdl_calibration_sha256": plan["data"]["wdl_calibration"]["sha256"],
        }
        sections = {section.name: bytes(section.byte_length) for section in spec.sections}
        network, container_receipt = container.build_container(spec, sections, provenance)
        (run_root / "network.hsv2").write_bytes(network)
        export = {
            "schema": campaign.EXPORT_RECEIPT_SCHEMA,
            "container": container_receipt,
            "claims": {
                "full_refresh_container": True,
                "incremental_eligible": False,
                "production_dispatch": False,
                "strength_evidence": False,
            },
        }
        (run_root / "export-receipt.json").write_bytes(campaign._canonical_json(export))


def main() -> int:
    contract, contract_sha = campaign.load_contract()
    if contract["schema_name"] != campaign.CONTRACT_SCHEMA:
        raise AssertionError("campaign contract schema was not loaded")
    if contract_sha != campaign.CONTRACT_SHA256:
        raise AssertionError("campaign contract hash was not frozen")
    addendum, addendum_sha = campaign.load_coverage_addendum()
    if addendum["schema_name"] != campaign.COVERAGE_ADDENDUM_SCHEMA:
        raise AssertionError("coverage addendum schema was not loaded")
    if addendum_sha != campaign.COVERAGE_ADDENDUM_SHA256:
        raise AssertionError("coverage addendum hash was not frozen")
    if campaign.SEED_NAMESPACE != campaign.CONTRACT_SCHEMA:
        raise AssertionError("coverage amendment changed the paired-seed namespace")
    if not campaign._seen_mass_gate(1, 100):
        raise AssertionError("exactly one percent unseen activation should pass")
    if campaign._seen_mass_gate(2, 100):
        raise AssertionError("one activation beyond one percent should fail")

    with tempfile.TemporaryDirectory(prefix="horde-v2-c1-campaign-") as temporary:
        root = Path(temporary)
        train, validation, split, calibration = _write_inputs(root)
        plan = _plan(train, validation, split, calibration)
        if len(plan["runs"]) != 9:
            raise AssertionError("campaign planner did not produce nine runs")
        if len({run["seed"] for run in plan["runs"]}) != 3:
            raise AssertionError("campaign planner did not produce three paired seeds")
        if plan["selection"]["predesignated_playing_seed"] != plan["runs"][0]["seed"]:
            raise AssertionError("campaign playing seed was selected after training")
        if {run["architecture"]["name"] for run in plan["runs"]} != {
            "v2-c1-abs64x192",
            "v2-c1-rank8-64x192",
            "v2-64x192",
        }:
            raise AssertionError("campaign architecture set drifted")
        if plan["claims"] != {
            "fixture_mode": True,
            "campaign_inputs_eligible": False,
            "training_started": False,
            "training_complete": False,
            "architecture_selected": False,
            "strength_evidence": False,
            "production_network": False,
        }:
            raise AssertionError("fixture plan made an unsupported claim")
        if plan["data"]["coverage"]["schema"] != campaign.COVERAGE_SCHEMA:
            raise AssertionError("campaign coverage receipt is missing")
        _expect_failure(
            lambda: campaign._require_production_coverage(plan["data"]["coverage"]),
            "role counts drifted",
        )
        if any("--allow-dirty" in run["training_command"] for run in plan["runs"]):
            raise AssertionError("campaign command permits a dirty source")
        for run in plan["runs"]:
            command = run["training_command"]
            if "--campaign-plan" not in command or "--campaign-run-id" not in command:
                raise AssertionError("campaign run is not bound before epoch-zero validation")
            run_id_index = command.index("--campaign-run-id") + 1
            if command[run_id_index] != run["id"]:
                raise AssertionError("campaign command binds another run id")

        _expect_failure(
            lambda: campaign.plan_campaign(
                train,
                validation,
                split,
                calibration,
                _expected_records=(TRAIN_RECORDS + 1, VALIDATION_RECORDS),
                _source_override=SOURCE,
            ),
            "training record count",
        )

        legacy_split = root / "legacy-split.json"
        legacy_payload = json.loads(split.read_text(encoding="utf-8"))
        legacy_payload["schema"] = "HORDE_TRAINING_BOOK_SPLIT_V1"
        legacy_split.write_text(json.dumps(legacy_payload), encoding="utf-8")
        _expect_failure(
            lambda: _plan(train, validation, legacy_split, calibration),
            "reflection-safe V2 book split",
        )

        overlapping_validation = root / "overlap-validation.bin"
        fixtures._write_dataset(
            overlapping_validation,
            first=0,
            count=VALIDATION_RECORDS,
            book_sha256="B" * 64,
            seed=202,
        )
        _expect_failure(
            lambda: _plan(train, overlapping_validation, split, calibration),
            "roles overlap",
        )

        other_train = root / "other-train.bin"
        other_calibration = root / "other-wdl-calibration.json"
        fixtures._write_dataset(
            other_train,
            first=512,
            count=TRAIN_RECORDS,
            book_sha256="A" * 64,
            seed=303,
        )
        fixtures._write_wdl_calibration(other_calibration, other_train)
        _expect_failure(
            lambda: _plan(train, validation, split, other_calibration),
            "exact training dataset",
        )

        changed_contract = root / "changed-contract.json"
        changed_contract.write_bytes((ROOT / campaign.CONTRACT_RELATIVE_PATH).read_bytes() + b"\n")
        _expect_failure(
            lambda: campaign.load_contract(changed_contract),
            "contract SHA-256",
        )

        changed_addendum = root / "changed-addendum.json"
        changed_addendum.write_bytes(
            (ROOT / campaign.COVERAGE_ADDENDUM_RELATIVE_PATH).read_bytes() + b"\n"
        )
        _expect_failure(
            lambda: campaign.load_coverage_addendum(changed_addendum),
            "addendum SHA-256",
        )

        plan_path = root / "plan.json"
        plan_path.write_bytes(campaign._canonical_json(plan))

        trainer_plan = copy.deepcopy(plan)
        trainer_plan["claims"]["fixture_mode"] = False
        trainer_plan["claims"]["campaign_inputs_eligible"] = True
        trainer_plan_path = root / "trainer-plan.json"
        trainer_plan_path.write_bytes(campaign._canonical_json(trainer_plan))
        first_run = trainer_plan["runs"][0]
        trainer_args = argparse.Namespace(
            campaign_plan=trainer_plan_path,
            campaign_run_id=first_run["id"],
            seed=first_run["seed"],
            epochs=trainer_plan["configuration"]["epochs"],
            batch_size=trainer_plan["configuration"]["batch_size"],
            block_size=trainer_plan["configuration"]["block_size"],
            lambda_value=trainer_plan["configuration"]["lambda"],
            learning_rate=trainer_plan["configuration"]["learning_rate"],
            scheduler_gamma=trainer_plan["configuration"]["scheduler_gamma"],
            device=trainer_plan["configuration"]["device"]["type"],
            cpu_threads=trainer_plan["configuration"]["device"]["cpu_threads"],
        )
        bundle = training._load_campaign_plan(
            trainer_args,
            first_run["architecture"]["name"],
            SOURCE,
        )
        binding = training._finalize_campaign_binding(
            bundle,
            trainer_args,
            first_run["architecture"]["name"],
            trainer_plan["data"],
            TRAIN_RECORDS,
            VALIDATION_RECORDS,
        )
        if binding is None or binding["campaign_run_id"] != first_run["id"]:
            raise AssertionError("trainer did not authenticate its exact campaign run")
        changed_trainer_args = copy.copy(trainer_args)
        changed_trainer_args.seed += 1
        try:
            training._load_campaign_plan(
                changed_trainer_args,
                first_run["architecture"]["name"],
                SOURCE,
            )
        except training.TrainingError as error:
            if "architecture or seed" not in str(error):
                raise AssertionError(f"unexpected trainer binding error: {error}") from error
        else:
            raise AssertionError("trainer accepted a seed outside its campaign run")

        runs_root = root / "runs"
        runs_root.mkdir()
        changed_plan = copy.deepcopy(plan)
        changed_plan["runs"][0]["seed"] += 1
        changed_plan_path = root / "changed-plan.json"
        changed_plan_path.write_bytes(campaign._canonical_json(changed_plan))
        _expect_failure(
            lambda: campaign.verify_campaign(
                changed_plan_path,
                runs_root,
                _allow_fixture=True,
            ),
            "nine-run matrix",
        )
        _write_completed_runs(runs_root, plan)
        verification = campaign.verify_campaign(
            plan_path,
            runs_root,
            _allow_fixture=True,
        )
        if len(verification["runs"]) != 9:
            raise AssertionError("campaign verifier did not authenticate nine runs")
        if verification["claims"]["architecture_selection_eligible"] is not False:
            raise AssertionError("fixture campaign became architecture-selection eligible")
        if verification["claims"]["paired_playing_gate_eligible"] is not False:
            raise AssertionError("authenticated fixtures became playing-gate eligible")
        if len(set(verification["paired_sample_order"].values())) != 3:
            raise AssertionError("paired sample-order receipts are incomplete")

        optimizer_run = plan["runs"][0]
        optimizer_receipt_path = runs_root / optimizer_run["output_role"] / "receipt.json"
        optimizer_receipt_bytes = optimizer_receipt_path.read_bytes()
        optimizer_receipt = json.loads(optimizer_receipt_bytes.decode("ascii"))
        optimizer_receipt["optimizer"]["foreach"] = True
        optimizer_receipt_path.write_bytes(campaign._canonical_json(optimizer_receipt))
        _expect_failure(
            lambda: campaign.verify_campaign(
                plan_path,
                runs_root,
                _allow_fixture=True,
            ),
            "optimizer field foreach",
        )
        optimizer_receipt_path.write_bytes(optimizer_receipt_bytes)

        tampered_run = plan["runs"][1]
        tampered_receipt_path = runs_root / tampered_run["output_role"] / "receipt.json"
        tampered_receipt = json.loads(tampered_receipt_path.read_text(encoding="ascii"))
        tampered_receipt["run"]["sample_order_chain_sha256"] = "F" * 64
        tampered_receipt_path.write_bytes(campaign._canonical_json(tampered_receipt))
        _expect_failure(
            lambda: campaign.verify_campaign(
                plan_path,
                runs_root,
                _allow_fixture=True,
            ),
            "deterministic schedule",
        )

    print(
        "Horde V2 C1 campaign contract passed: "
        "250k/250k production gate, 3 architectures, 3 paired seeds, 9 authenticated runs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
