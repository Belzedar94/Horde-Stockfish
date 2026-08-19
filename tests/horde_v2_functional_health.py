#!/usr/bin/env python3
"""Fail-closed unit tests for the Horde V2 functional-health gate."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import horde_v2_functional_health as health  # noqa: E402


def _healthy_layer() -> dict[str, object]:
    preactivation = torch.tensor(
        [[0.10, 0.20], [0.20, 0.30], [0.30, 0.40], [0.40, 0.50]],
        dtype=torch.float32,
    )
    return health.layer_metrics(preactivation, preactivation, 1.0e-12)


def _collapsed_layer() -> dict[str, object]:
    preactivation = torch.tensor(
        [[-0.25, 1.25], [-0.20, 1.30], [-0.15, 1.35], [-0.10, 1.40]],
        dtype=torch.float32,
    )
    return health.layer_metrics(
        preactivation,
        torch.clamp(preactivation, 0.0, 1.0),
        1.0e-12,
    )


def main() -> int:
    contract, contract_sha = health.load_contract()
    if contract["schema_name"] != health.CONTRACT_SCHEMA:
        raise AssertionError("functional-health contract schema drifted")
    if contract_sha != health.CONTRACT_SHA256:
        raise AssertionError("functional-health contract hash was not frozen")
    if health.deterministic_probe_indices(10, 4) != (1, 3, 6, 8):
        raise AssertionError("deterministic midpoint probe selection drifted")

    healthy = _healthy_layer()
    if healthy["constant_lanes"] != 0 or healthy["interior_sample_fraction"] != 1.0:
        raise AssertionError("healthy position-dependent layer was classified as collapsed")
    collapsed = _collapsed_layer()
    if collapsed["constant_lane_fraction"] != 1.0:
        raise AssertionError("clamped constant layer escaped collapse detection")
    if collapsed["interior_sample_fraction"] != 0.0:
        raise AssertionError("collapsed layer reported interior activation support")
    try:
        health.layer_metrics(
            torch.tensor([[float("nan")]], dtype=torch.float32),
            torch.tensor([[0.0]], dtype=torch.float32),
            1.0e-12,
        )
    except health.FunctionalHealthError as error:
        if "non-finite" not in str(error):
            raise AssertionError(f"unexpected non-finite activation error: {error}") from error
    else:
        raise AssertionError("non-finite activation passed functional health")

    pre_rule50 = torch.tensor(
        [0.1, 1.1] * 16 + [-2.1, -3.1] * 16,
        dtype=torch.float32,
    )
    scores = health._score_metrics(
        pre_rule50,
        pre_rule50,
        torch.tensor([health.WHITE] * 32 + [health.BLACK] * 32),
    )
    if any(
        scores[name]["unique_pre_rule50_integer_scores"] != 2
        for name in health.SIDE_NAMES.values()
    ):
        raise AssertionError("within-side integer-score diversity drifted")

    gradient = torch.ones((4, 2), dtype=torch.float32)
    healthy_jacobian = {
        "first_domain": health._jacobian_metrics(gradient, 1.0e-12),
        "global": health._jacobian_metrics(gradient, 1.0e-12),
    }
    healthy_gate = health._gate_receipt(
        contract,
        {"hidden0": healthy, "hidden1": healthy},
        scores,
        healthy_jacobian,
    )
    if not healthy_gate["passed"]:
        raise AssertionError("healthy functional path failed its gate")

    collapsed_scores = health._score_metrics(
        torch.tensor([1.1] * 32 + [-2.1] * 32),
        torch.tensor([1.1] * 32 + [-2.1] * 32),
        torch.tensor([health.WHITE] * 32 + [health.BLACK] * 32),
    )
    zero_gradient = torch.zeros((4, 2), dtype=torch.float32)
    collapsed_gate = health._gate_receipt(
        contract,
        {"hidden0": collapsed, "hidden1": collapsed},
        collapsed_scores,
        {
            "first_domain": health._jacobian_metrics(zero_gradient, 1.0e-12),
            "global": health._jacobian_metrics(zero_gradient, 1.0e-12),
        },
    )
    if collapsed_gate["passed"]:
        raise AssertionError("constant dead-clamp network passed functional health")
    expected_failures = {
        "hidden0_constant_lane_fraction",
        "hidden0_interior_sample_fraction",
        "hidden1_constant_lane_fraction",
        "hidden1_interior_sample_fraction",
        "white_to_move_integer_diversity",
        "black_to_move_integer_diversity",
        "first_domain_jacobian",
        "global_jacobian",
    }
    observed_failures = {
        name for name, passed in collapsed_gate["checks"].items() if not passed
    }
    if not expected_failures.issubset(observed_failures):
        raise AssertionError("collapsed receipt did not expose every critical failure")

    sides = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    clocks = torch.tensor([0, 0, 2, 2, 3], dtype=torch.long)
    permutation = health._group_rotation(sides, clocks)
    if permutation.tolist() != [1, 0, 3, 2, 4]:
        raise AssertionError("same-side, same-rule50 intervention permutation drifted")
    if not torch.equal(sides.index_select(0, permutation), sides):
        raise AssertionError("intervention permutation crossed side-to-move groups")
    if not torch.equal(clocks.index_select(0, permutation), clocks):
        raise AssertionError("intervention permutation crossed rule50 groups")

    difference = health._difference_metrics(
        torch.tensor([0.0, 2.0]), torch.tensor([0.0, 1.0])
    )
    if difference["changed_integer_fraction"] != 0.5:
        raise AssertionError("intervention integer-difference accounting drifted")

    print(
        "Horde V2 functional-health tests passed: frozen contract, deterministic probe, "
        "dead-clamp rejection, score diversity, Jacobian, and grouped interventions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
