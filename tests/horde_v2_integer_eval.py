#!/usr/bin/env python3
"""Independent batch-parity and metric tests for Horde V2 integer evaluation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import horde_v2_container as container  # noqa: E402
import horde_v2_container_parity as parity  # noqa: E402
import horde_v2_integer_eval as integer_eval  # noqa: E402
from horde_training_decoder import (  # noqa: E402
    TrainingRecord,
    extract_sparse_features,
    make_sparse_batch,
)
from horde_wdl import SIDE_NAMES, probabilities  # noqa: E402


PIECE_CODE = {piece: index for index, piece in enumerate(".PNBRQpnbrqk")}


def _record(fixture: parity.PositionFixture, index: int) -> TrainingRecord:
    board = tuple(PIECE_CODE[piece] for piece in fixture.board)
    return TrainingRecord(
        index=index,
        features=extract_sparse_features(board),
        side_to_move=fixture.side_to_move,
        rule50_count=fixture.rule50,
        game_ply=index,
        score=(-300, -12, 0, 37, 600, 31_507)[index],
        best_move=0,
        played_move=0,
        result=(-1, 0, 1, -1, 0, 1)[index],
        outcome_reason=3,
        board=board,
    )


def _expect_evaluation_failure(callable_object: object, needle: str) -> None:
    try:
        callable_object()
    except integer_eval.IntegerEvaluationError as error:
        if needle not in str(error):
            raise AssertionError(f"unexpected integer evaluation error: {error}") from error
    else:
        raise AssertionError(f"integer evaluator accepted malformed input: {needle}")


def _metric_reference(
    predicted: np.ndarray,
    teacher: np.ndarray,
    results: np.ndarray,
    sides: np.ndarray,
    lambda_value: float,
    calibration: dict[str, tuple[float, float, float]],
) -> tuple[list[float], list[float], list[float], list[bool]]:
    composite: list[float] = []
    score_terms: list[float] = []
    result_terms: list[float] = []
    eligible_values: list[bool] = []
    for prediction_score, teacher_score, result, side in zip(
        predicted, teacher, results, sides, strict=True
    ):
        parameters = calibration[SIDE_NAMES[int(side)]]
        prediction_wdl = probabilities(float(prediction_score), parameters)
        eligible = abs(int(teacher_score)) < integer_eval.MATE_SCORE_THRESHOLD
        teacher_wdl = probabilities(float(teacher_score if eligible else 0), parameters)
        target = (1.0, 0.0, 0.0) if result == -1 else (
            (0.0, 1.0, 0.0) if result == 0 else (0.0, 0.0, 1.0)
        )
        score_term = 0.5 * sum(
            (actual - expected) ** 2
            for actual, expected in zip(prediction_wdl, teacher_wdl, strict=True)
        )
        result_term = 0.5 * sum(
            (actual - expected) ** 2
            for actual, expected in zip(prediction_wdl, target, strict=True)
        )
        composite.append(
            lambda_value * int(eligible) * score_term
            + (1.0 - lambda_value) * result_term
        )
        score_terms.append(score_term)
        result_terms.append(result_term)
        eligible_values.append(eligible)
    return composite, score_terms, result_terms, eligible_values


def main() -> int:
    records = tuple(_record(fixture, index) for index, fixture in enumerate(parity.FIXTURES))
    batch = make_sparse_batch(records)

    for spec in container.SPECS:
        payload, _ = container.build_container(
            spec,
            parity.deterministic_sections(spec),
            parity.deterministic_provenance(spec),
        )
        parsed = container.parse_container(payload)
        network = integer_eval.IntegerNetwork.from_container(parsed)
        observed = network.evaluate(batch)
        parameters = parity.decode_parameters(parsed)
        expected = np.asarray(
            [
                parity.evaluate_fixture(parsed, parameters, fixture)["value"]
                for fixture in parity.FIXTURES
            ],
            dtype=np.int32,
        )
        if not np.array_equal(observed, expected):
            raise AssertionError(
                f"{spec.architecture} vector evaluator differs: "
                f"{observed.tolist()} != {expected.tolist()}"
            )
        health = network.parameter_health()
        if health["passed"] is not True:
            raise AssertionError(f"{spec.architecture} deterministic weights failed health")

    zero_spec = container.SPECS_BY_ARCHITECTURE["v2-c1-abs64x192"]
    zero_payload, _ = container.build_container(
        zero_spec,
        {section.name: bytes(section.byte_length) for section in zero_spec.sections},
        parity.deterministic_provenance(zero_spec),
    )
    zero_health = integer_eval.IntegerNetwork.from_container(
        container.parse_container(zero_payload)
    ).parameter_health()
    if zero_health["passed"] is not False:
        raise AssertionError("all-zero quantized weights passed parameter health")

    saturated_sections = parity.deterministic_sections(zero_spec)
    saturated_sections["hidden0_weights"] = bytes([127]) * next(
        section.byte_length
        for section in zero_spec.sections
        if section.name == "hidden0_weights"
    )
    saturated_payload, _ = container.build_container(
        zero_spec,
        saturated_sections,
        parity.deterministic_provenance(zero_spec),
    )
    saturated_health = integer_eval.IntegerNetwork.from_container(
        container.parse_container(saturated_payload)
    ).parameter_health()
    if saturated_health["passed"] is not False:
        raise AssertionError("dtype-boundary-saturated weights passed parameter health")
    if saturated_health["sections"]["hidden0_weights"]["dtype_boundary_fraction"] != 1.0:
        raise AssertionError("parameter health did not report saturated dense weights")

    valid_network = integer_eval.IntegerNetwork.from_container(
        container.parse_container(
            container.build_container(
                zero_spec,
                parity.deterministic_sections(zero_spec),
                parity.deterministic_provenance(zero_spec),
            )[0]
        )
    )
    malformed_offsets = replace(
        batch,
        global_offsets=(0,) + batch.global_offsets[2:],
    )
    _expect_evaluation_failure(
        lambda: valid_network.evaluate(malformed_offsets),
        "offsets have the wrong shape",
    )
    empty_bag_offsets = list(batch.global_offsets)
    empty_bag_offsets[1] = empty_bag_offsets[0]
    _expect_evaluation_failure(
        lambda: valid_network.evaluate(
            replace(batch, global_offsets=tuple(empty_bag_offsets))
        ),
        "empty or malformed sparse bag",
    )
    out_of_range = replace(
        batch,
        v2_global=(container.GLOBAL_ROWS,) + batch.v2_global[1:],
    )
    _expect_evaluation_failure(
        lambda: valid_network.evaluate(out_of_range),
        "out-of-range sparse row",
    )

    predicted = np.asarray([-600, -1, 0, 1, 600, 31_506], dtype=np.int32)
    teacher = np.asarray(batch.scores, dtype=np.int32)
    results = np.asarray(batch.results, dtype=np.int8)
    sides = np.asarray(batch.side_to_move, dtype=np.int8)
    calibration = {
        "white_to_move": (1.25, -0.15, 0.4),
        "black_to_move": (0.9, 0.2, -0.1),
    }
    terms = integer_eval.loss_arrays(
        predicted,
        teacher,
        results,
        sides,
        0.6,
        calibration,
    )
    reference = _metric_reference(
        predicted,
        teacher,
        results,
        sides,
        0.6,
        calibration,
    )
    for key, expected in zip(
        ("composite", "score_error", "result_error", "score_eligible"),
        reference,
        strict=True,
    ):
        if not np.allclose(terms[key], np.asarray(expected), rtol=0.0, atol=1.0e-15):
            raise AssertionError(f"integer metric term {key} differs from scalar reference")

    metrics = integer_eval.MetricAccumulator()
    metrics.update(terms)
    receipt = metrics.receipt()
    if receipt["samples"] != len(records) or receipt["mate_scores_masked"] != 1:
        raise AssertionError("metric accumulator sample accounting drifted")
    if abs(sum(receipt["prediction_mean_wdl"]) - 1.0) > 1.0e-15:
        raise AssertionError("metric accumulator WDL means do not sum to one")

    print(
        "Horde V2 integer evaluation passed: 3 architectures x 6 positions, "
        "scalar parity, metric parity, parameter-health and malformed-batch rejection"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
