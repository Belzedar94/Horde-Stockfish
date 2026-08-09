#!/usr/bin/env python3
"""Fail-closed tests for the Horde V2 C1 quantized training screen."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import horde_v2_c1_campaign as campaign  # noqa: E402
import horde_v2_c1_screen as screen  # noqa: E402


FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "horde_v2_c1_campaign_fixture", ROOT / "tests" / "horde_v2_c1_campaign.py"
)
if FIXTURE_SPEC is None or FIXTURE_SPEC.loader is None:
    raise RuntimeError("campaign fixture module could not be loaded")
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixtures)


def _expect_failure(callable_object: object, needle: str) -> None:
    try:
        callable_object()
    except (screen.ScreenError, campaign.CampaignError, ValueError) as error:
        if needle not in str(error):
            raise AssertionError(f"unexpected C1 screen error: {error}") from error
    else:
        raise AssertionError(f"C1 screen accepted invalid evidence: {needle}")


def _run_state(architecture: str, pair_index: int, loss: float) -> dict[str, object]:
    metric = {"composite_loss": loss}
    return {
        "pair_index": pair_index,
        "seed": (101, 202, 303)[pair_index],
        "architecture": architecture,
        "float_validation": {
            "final_validation_composite_loss": loss,
            "last_two_validation_composite_losses": [loss + 0.01, loss],
        },
        "parameter_health": {"passed": True},
        "integer_validation": {
            "overall": metric,
            "side_to_move": {
                "white_to_move": metric,
                "black_to_move": metric,
            },
        },
    }


def _architecture_runs(losses: dict[str, float]) -> dict[str, dict[int, dict[str, object]]]:
    return {
        architecture: {
            pair_index: _run_state(architecture, pair_index, loss + pair_index * 0.001)
            for pair_index in range(3)
        }
        for architecture, loss in losses.items()
    }


def _comparison_receipts(
    contract: dict[str, object], losses: dict[str, float]
) -> list[dict[str, object]]:
    runs = _architecture_runs(losses)
    return [
        screen.compare_architectures(comparison, runs, 4.3026527297)
        for comparison in contract["comparisons"]
    ]


def main() -> int:
    contract, contract_sha = screen.load_contract()
    if contract["schema_name"] != screen.CONTRACT_SCHEMA:
        raise AssertionError("screen contract schema drifted")
    if contract_sha != screen.CONTRACT_SHA256:
        raise AssertionError("screen contract hash was not frozen")

    selection_plan = {
        "selection": {
            "predesignated_playing_seed_index": 0,
            "predesignated_playing_seed": 101,
        }
    }
    all_pass = _comparison_receipts(
        contract,
        {
            "v2-c1-abs64x192": 0.30,
            "v2-c1-rank8-64x192": 0.20,
            "v2-64x192": 0.10,
        },
    )
    if not all(comparison["passed"] for comparison in all_pass):
        raise AssertionError("strict comparison gates rejected a clear paired improvement")
    nomination = screen.nominate_pairing(all_pass, selection_plan, contract)
    if nomination is None or nomination["candidate_architecture"] != "v2-64x192":
        raise AssertionError("Royal-32 was not nominated against a clear control")
    if nomination["baseline_architecture"] != "v2-c1-rank8-64x192":
        raise AssertionError("Royal-32 nomination skipped the nearest cheaper control")

    rank8_only = _comparison_receipts(
        contract,
        {
            "v2-c1-abs64x192": 0.30,
            "v2-c1-rank8-64x192": 0.20,
            "v2-64x192": 0.25,
        },
    )
    rank8_nomination = screen.nominate_pairing(rank8_only, selection_plan, contract)
    if (
        rank8_nomination is None
        or rank8_nomination["candidate_architecture"] != "v2-c1-rank8-64x192"
    ):
        raise AssertionError("Rank-8 was not nominated after passing the absolute control")

    royal_only = _comparison_receipts(
        contract,
        {
            "v2-c1-abs64x192": 0.30,
            "v2-c1-rank8-64x192": 0.40,
            "v2-64x192": 0.20,
        },
    )
    royal_nomination = screen.nominate_pairing(royal_only, selection_plan, contract)
    if (
        royal_nomination is None
        or royal_nomination["baseline_architecture"] != "v2-c1-abs64x192"
    ):
        raise AssertionError("Royal-32 did not fall back to the absolute control")

    no_pass = _comparison_receipts(
        contract,
        {
            "v2-c1-abs64x192": 0.10,
            "v2-c1-rank8-64x192": 0.20,
            "v2-64x192": 0.30,
        },
    )
    if screen.nominate_pairing(no_pass, selection_plan, contract) is not None:
        raise AssertionError("screen nominated an architecture with no passing comparison")

    noisy_runs = _architecture_runs(
        {
            "v2-c1-abs64x192": 0.30,
            "v2-c1-rank8-64x192": 0.20,
            "v2-64x192": 0.10,
        }
    )
    noisy_runs["v2-c1-rank8-64x192"][2] = _run_state(
        "v2-c1-rank8-64x192", 2, 0.31
    )
    noisy_comparison = screen.compare_architectures(
        contract["comparisons"][0], noisy_runs, 4.3026527297
    )
    if noisy_comparison["passed"]:
        raise AssertionError("one losing paired seed passed the strict quantized screen")

    unhealthy_runs = _architecture_runs(
        {
            "v2-c1-abs64x192": 0.30,
            "v2-c1-rank8-64x192": 0.20,
            "v2-64x192": 0.10,
        }
    )
    unhealthy_runs["v2-c1-rank8-64x192"][1]["parameter_health"] = {"passed": False}
    unhealthy_comparison = screen.compare_architectures(
        contract["comparisons"][0], unhealthy_runs, 4.3026527297
    )
    if unhealthy_comparison["passed"]:
        raise AssertionError("unhealthy quantized parameters passed the comparison gate")

    with tempfile.TemporaryDirectory(prefix="horde-v2-c1-screen-") as temporary:
        root = Path(temporary)
        train, validation, split, calibration = fixtures._write_inputs(root)
        plan = fixtures._plan(train, validation, split, calibration)
        plan_path = root / "plan.json"
        plan_path.write_bytes(campaign._canonical_json(plan))
        runs_root = root / "runs"
        runs_root.mkdir()
        fixtures._write_completed_runs(runs_root, plan)

        receipt = screen.screen_campaign(
            plan_path,
            runs_root,
            validation,
            calibration,
            _allow_fixture=True,
            _expected_records=fixtures.VALIDATION_RECORDS,
            _source_override=fixtures.SOURCE,
        )
        if len(receipt["runs"]) != 9:
            raise AssertionError("fixture screen did not evaluate all nine containers")
        if receipt["fixed_node_nomination"] is not None:
            raise AssertionError("all-zero fixture containers produced a nomination")
        if receipt["claims"]["parameter_health_all_passed"] is not False:
            raise AssertionError("all-zero fixture containers passed parameter health")
        if receipt["claims"]["architecture_selected"] is not False:
            raise AssertionError("fixture screen selected a production architecture")
        if receipt["validation"]["records"] != fixtures.VALIDATION_RECORDS:
            raise AssertionError("fixture validation accounting drifted")

        first_run = plan["runs"][0]
        metrics_path = runs_root / first_run["output_role"] / "metrics.jsonl"
        metrics_bytes = metrics_path.read_bytes()
        metrics_path.write_bytes(metrics_bytes + b"{}\n")
        _expect_failure(
            lambda: screen.screen_campaign(
                plan_path,
                runs_root,
                validation,
                calibration,
                _allow_fixture=True,
                _expected_records=fixtures.VALIDATION_RECORDS,
                _source_override=fixtures.SOURCE,
            ),
            "metrics hash drifted",
        )
        metrics_path.write_bytes(metrics_bytes)

        tampered_validation = root / "tampered-validation.bin"
        validation_bytes = validation.read_bytes()
        tampered_validation.write_bytes(validation_bytes[:-1] + bytes([validation_bytes[-1] ^ 1]))
        _expect_failure(
            lambda: screen.screen_campaign(
                plan_path,
                runs_root,
                tampered_validation,
                calibration,
                _allow_fixture=True,
                _expected_records=fixtures.VALIDATION_RECORDS,
                _source_override=fixtures.SOURCE,
            ),
            "payload SHA-256 mismatch",
        )

        changed_contract = root / "changed-screen-contract.json"
        changed_contract.write_bytes((ROOT / screen.CONTRACT_RELATIVE_PATH).read_bytes() + b"\n")
        _expect_failure(
            lambda: screen.load_contract(changed_contract),
            "contract SHA-256 mismatch",
        )

        changed_plan = copy.deepcopy(plan)
        changed_plan["configuration"]["lambda"] = 0.5
        changed_plan_path = root / "changed-plan.json"
        changed_plan_path.write_bytes(campaign._canonical_json(changed_plan))
        _expect_failure(
            lambda: screen.screen_campaign(
                changed_plan_path,
                runs_root,
                validation,
                calibration,
                _allow_fixture=True,
                _expected_records=fixtures.VALIDATION_RECORDS,
                _source_override=fixtures.SOURCE,
            ),
            "configuration",
        )

    print(
        "Horde V2 C1 quantized screen passed: strict paired gates, deterministic "
        "nomination, nine-container fixture evaluation and tamper rejection"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
