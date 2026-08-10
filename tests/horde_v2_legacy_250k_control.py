#!/usr/bin/env python3
"""Focused tests for the frozen Horde V2/legacy 250k control."""

from __future__ import annotations

import copy
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import horde_v2_c2_objective as objective  # noqa: E402
import horde_v2_legacy_250k_control as control  # noqa: E402


def _evaluation(loss: float) -> dict[str, object]:
    return {"composite_loss_mean_all_records": objective.float_receipt(loss)}


def _rank8(seed: int, tuning: float, confirmation: float) -> dict[str, object]:
    return {
        "seed": seed,
        "tuning_stop_composite_loss": objective.float_receipt(tuning),
        "evaluation": _evaluation(confirmation),
    }


def _legacy(seed: int, tuning: float, confirmation: float) -> dict[str, object]:
    return {
        "seed": seed,
        "tuning_stop_composite_loss": tuning,
        "evaluation": _evaluation(confirmation),
    }


def main() -> int:
    stable_identity = {
        "sha256": "A" * 64,
        "payload_sha256": "B" * 64,
        "records": 250_000,
    }
    observed_identity = {
        **stable_identity,
        "book_sha256": "C" * 64,
        "seed": "2026080801",
        "name": "selected-records.bin",
        "selected_role": {"receipt_sha256": "D" * 64},
    }
    if not control._identity_matches(observed_identity, stable_identity):
        raise AssertionError("stable identity rejected additional provenance")
    expected_with_optional = {
        **stable_identity,
        "book_sha256": "C" * 64,
        "seed": "2026080801",
    }
    if not control._identity_matches(observed_identity, expected_with_optional):
        raise AssertionError("stable identity rejected matching optional provenance")
    drifted_identity = {**stable_identity, "sha256": "E" * 64}
    if control._identity_matches(observed_identity, drifted_identity):
        raise AssertionError("stable identity accepted byte drift")
    drifted_optional = {**expected_with_optional, "seed": "2026080803"}
    if control._identity_matches(observed_identity, drifted_optional):
        raise AssertionError("stable identity accepted optional provenance drift")
    expected_teacher = {"network": {"sha256": "F" * 64}, "source_commit": "abc123"}
    observed_teacher = {**expected_teacher, "generation": {"depth": 4}}
    if not control._contains_expected_mapping(observed_teacher, expected_teacher):
        raise AssertionError("teacher identity rejected additional provenance")
    drifted_teacher = {**expected_teacher, "source_commit": "def456"}
    if control._contains_expected_mapping(observed_teacher, drifted_teacher):
        raise AssertionError("teacher identity accepted authenticated-field drift")

    seeds = control.EXPECTED_SEEDS
    rank8 = [
        _rank8(seeds[0], 0.140, 0.141),
        _rank8(seeds[1], 0.150, 0.151),
        _rank8(seeds[2], 0.160, 0.161),
    ]
    legacy = [
        _legacy(seeds[0], 0.145, 0.146),
        _legacy(seeds[1], 0.156, 0.157),
        _legacy(seeds[2], 0.167, 0.168),
    ]
    summary = control.summarize_paired(rank8, legacy)
    direction = summary["directional_consistency"]
    if direction != {
        "rank8_lower_tuning_loss_all_three_seeds": True,
        "rank8_lower_confirmation_loss_all_three_seeds": True,
        "rank8_lower_confirmation_loss_seed_count": 3,
    }:
        raise AssertionError(f"matched-control direction drifted: {direction}")
    confirmation_delta = objective.float_from_receipt(
        summary["three_seed_mean_delta"]["fresh_confirmation_legacy_minus_rank8"],
        "fixture confirmation delta",
    )
    if abs(confirmation_delta - 0.006) > 1.0e-15:
        raise AssertionError(f"matched-control mean delta drifted: {confirmation_delta}")

    mixed = copy.deepcopy(legacy)
    mixed[2]["evaluation"] = _evaluation(0.159)
    mixed_summary = control.summarize_paired(rank8, mixed)
    mixed_direction = mixed_summary["directional_consistency"]
    if mixed_direction["rank8_lower_confirmation_loss_all_three_seeds"] is not False:
        raise AssertionError("mixed confirmation directions were accepted as all-three")
    if mixed_direction["rank8_lower_confirmation_loss_seed_count"] != 2:
        raise AssertionError("mixed confirmation seed count drifted")

    try:
        control.summarize_paired(list(reversed(rank8)), legacy)
    except control.LegacyControlError as error:
        if "reordered" not in str(error):
            raise AssertionError(f"unexpected seed-order error: {error}") from error
    else:
        raise AssertionError("matched control accepted reordered Rank8 seeds")

    print(
        "Horde V2 legacy 250k control tests passed: paired seed accounting, "
        "stable rematerialized identity, delta direction, three-seed consistency "
        "and mixed-direction handling"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
