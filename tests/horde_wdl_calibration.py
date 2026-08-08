#!/usr/bin/env python3
"""Focused tests for the frozen Horde Davidson WDL calibration."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import horde_wdl as wdl  # noqa: E402
import horde_training_decoder as decoder  # noqa: E402
import horde_bin_v1 as wire  # noqa: E402


def synthetic_observations(
    parameters: tuple[float, float, float],
) -> tuple[wdl.WeightedObservation, ...]:
    observations: list[tuple[int, int, int]] = []
    for score in (-1800, -1200, -600, 0, 600, 1200, 1800):
        probabilities = wdl.probabilities(score, parameters)
        total = 200_000
        loss = round(total * probabilities[0])
        draw = round(total * probabilities[1])
        win = total - loss - draw
        observations.extend(((score, -1, loss), (score, 0, draw), (score, 1, win)))
    return wdl.aggregate_observations(observations)


def _wire_record(index: int, side: int, result: int) -> bytes:
    board = [0] * 64
    board[0] = 2
    board[57] = 7
    board[60] = 11
    packed_board = bytes(
        board[square] | (board[square + 1] << 4) for square in range(0, 64, 2)
    )
    move = (0 << 6) | 8 if side == decoder.WHITE else (57 << 6) | 42
    reason = 3 if result == 0 else 1
    score = (-900, 0, 900)[result + 1] + (index % 7) - 3
    return packed_board + bytes((side, 0, 64, 0)) + struct.pack(
        "<HHhHHbB", 0, side, score, move, move, result, reason
    )


def _write_wire_dataset(path: Path) -> None:
    payload = b"".join(
        _wire_record(index, side, result)
        for index, (side, result) in enumerate(
            (side, result)
            for side in (decoder.WHITE, decoder.BLACK)
            for result in (-1, 0, 1)
            for _ in range(40)
        )
    )
    record_count = len(payload) // wire.RECORD_SIZE
    manifest = {
        "schema": wire.SCHEMA_NAME,
        "schema_sha256": wire.SCHEMA_SHA256,
        "format_version": wire.FORMAT_VERSION,
        "header_bytes": wire.HEADER_SIZE,
        "record_bytes": wire.RECORD_SIZE,
        "record_count": record_count,
        "byte_order": "little",
        "source_commit": "1" * 40,
        "source_dirty": False,
        "network": {
            "schema": "HORDETEST_HP_LEGACY_V1",
            "sha256": wire.RUN6B_SHA256,
        },
        "book_sha256": "3" * 64,
        "producer_sha256": "2" * 64,
        "payload_sha256": hashlib.sha256(payload).hexdigest().upper(),
        "label_contract": {
            "schema": wire.LABEL_CONTRACT_NAME,
            "schema_sha256": wire.LABEL_CONTRACT_SHA256,
        },
        "generation": {
            "requested_records": record_count,
            "seed": "1",
            "threads": 1,
            "hash_mb": 16,
            "depth": 1,
            "nodes": 0,
            "random_move_min_ply": 1,
            "random_move_max_ply": 1,
            "random_move_count": 0,
            "random_multi_pv": 0,
            "random_multi_pv_diff": 0,
            "write_min_ply": 0,
            "write_max_ply": 1,
            "max_game_ply": 2,
            "opening_count": record_count,
        },
    }
    encoded = json.dumps(
        manifest, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    header = wire.MAGIC + struct.pack("<HHI", 1, wire.HEADER_SIZE, len(encoded)) + encoded
    path.write_bytes(header + bytes(wire.HEADER_SIZE - len(header)) + payload)


def test_probability_contract() -> None:
    parameters = (0.75, -0.2, -0.8)
    previous_win = -1.0
    for score in range(-2400, 2401, 120):
        loss, draw, win = wdl.probabilities(score, parameters)
        if abs(loss + draw + win - 1.0) > 2.0e-15:
            raise AssertionError("Davidson probabilities do not sum to one")
        if not (0.0 < loss < 1.0 and 0.0 < draw < 1.0 and 0.0 < win < 1.0):
            raise AssertionError("Davidson probability left the open unit interval")
        if win <= previous_win:
            raise AssertionError("Davidson win probability is not score-monotone")
        previous_win = win


def test_derivatives() -> None:
    observations = synthetic_observations((0.9, 0.1, -0.6))
    parameters = [0.7, -0.05, -0.4]
    _, gradient, hessian, _ = wdl._statistics(observations, parameters, derivatives=True)
    epsilon = 1.0e-5
    for column in range(3):
        positive = parameters.copy()
        negative = parameters.copy()
        positive[column] += epsilon
        negative[column] -= epsilon
        positive_gradient = wdl._statistics(observations, positive, derivatives=True)[1]
        negative_gradient = wdl._statistics(observations, negative, derivatives=True)[1]
        for row in range(3):
            observed = (positive_gradient[row] - negative_gradient[row]) / (2.0 * epsilon)
            if abs(observed - hessian[row][column]) > 2.0e-8:
                raise AssertionError(
                    f"Davidson Hessian mismatch at ({row}, {column}): "
                    f"{observed} != {hessian[row][column]}"
                )
    if max(abs(value) for value in gradient) <= 1.0e-4:
        raise AssertionError("derivative fixture accidentally starts at its optimum")


def test_recovery_and_determinism() -> None:
    expected = (0.85, 0.15, -0.9)
    observations = synthetic_observations(expected)
    first = wdl.fit_side(observations)
    second = wdl.fit_side(observations)
    if first != second:
        raise AssertionError("Davidson fit changed across identical runs")
    for label, observed, target in zip("ABD", (first.a, first.b, first.d), expected):
        if abs(observed - target) > 2.0e-5:
            raise AssertionError(f"Davidson {label} was not recovered: {observed} != {target}")
    if first.gradient_inf_norm > wdl.DEFAULT_GRADIENT_TOLERANCE:
        raise AssertionError("Davidson optimizer stopped above its gradient tolerance")
    if not (0.0 < first.hessian_condition < wdl.DEFAULT_CONDITION_LIMIT):
        raise AssertionError("Davidson Hessian condition receipt is invalid")


def test_artifact_validation() -> None:
    white = synthetic_observations((0.8, 0.2, -0.7))
    black = synthetic_observations((1.1, -0.1, -1.0))
    counts = tuple(
        tuple(sum(observation.count for observation in side if observation.result == result) for result in (-1, 0, 1))
        for side in (white, black)
    )
    aggregated = wdl.AggregatedLabels(
        by_side=(white, black),
        total_records=sum(sum(side) for side in counts),
        eligible_records=sum(sum(side) for side in counts),
        mate_records_excluded=0,
        class_counts=(counts[0], counts[1]),
        mate_counts=(0, 0),
        selection_sha256="1" * 64,
        eligible_sha256="2" * 64,
    )
    artifact = wdl.build_artifact(
        aggregated,
        {
            "training_file": {
                "name": "fixture.bin",
                "sha256": "3" * 64,
                "payload_sha256": "4" * 64,
                "manifest_sha256": "5" * 64,
                "records": aggregated.total_records,
            },
            "teacher": {
                "source_commit": "1" * 40,
                "producer_sha256": "2" * 64,
                "network": {
                    "schema": "HORDETEST_HP_LEGACY_V1",
                    "sha256": wire.RUN6B_SHA256,
                },
                "label_contract": {
                    "schema": wire.LABEL_CONTRACT_NAME,
                    "schema_sha256": wire.LABEL_CONTRACT_SHA256,
                },
            },
            "software": {
                "commit": "6" * 40,
                "dirty": False,
                "python": "3.12.0",
                "implementation": "CPython",
            },
        },
    )
    decoded = wdl.validate_artifact(artifact)
    if set(decoded) != {"white_to_move", "black_to_move"}:
        raise AssertionError("Davidson artifact lost one side")
    if not wdl.canonical_json(artifact).endswith(b"\n"):
        raise AssertionError("Davidson artifact is not newline terminated")

    tampered = copy.deepcopy(artifact)
    tampered["fit"]["sides"]["white_to_move"]["parameters"]["A"][
        "ieee754_binary64_be"
    ] = "0" * 16
    try:
        wdl.validate_artifact(tampered)
    except wdl.CalibrationError:
        pass
    else:
        raise AssertionError("Davidson artifact accepted mismatched decimal/binary64 parameters")


def test_fail_closed_support() -> None:
    observations = wdl.aggregate_observations(((0, -1, 64), (0, 0, 64), (0, 1, 31)))
    try:
        wdl.fit_side(observations)
    except wdl.CalibrationError:
        pass
    else:
        raise AssertionError("Davidson calibration accepted insufficient class support")


def test_fail_closed_non_positive_slope() -> None:
    observations = wdl.aggregate_observations(
        ((600, -1, 64), (0, 0, 64), (-600, 1, 64))
    )
    try:
        wdl.fit_side(observations)
    except wdl.CalibrationError:
        pass
    else:
        raise AssertionError("Davidson calibration accepted a non-positive score relation")


def test_dataset_aggregation() -> None:
    with tempfile.TemporaryDirectory(prefix="horde-wdl-") as temporary:
        path = Path(temporary) / "train.bin"
        _write_wire_dataset(path)
        with decoder.HordeBinV1Dataset(path) as dataset:
            first = wdl.aggregate_labels(dataset)
        with decoder.HordeBinV1Dataset(path) as dataset:
            second = wdl.aggregate_labels(dataset)
        if first != second:
            raise AssertionError("dataset aggregation is not deterministic")
        if first.total_records != 240 or first.eligible_records != 240:
            raise AssertionError("dataset aggregation record counts changed")
        if first.class_counts != ((40, 40, 40), (40, 40, 40)):
            raise AssertionError(f"dataset class coverage changed: {first.class_counts}")
        if first.mate_records_excluded != 0 or first.mate_counts != (0, 0):
            raise AssertionError("dataset aggregation invented mate labels")
        if len(first.selection_sha256) != 64 or len(first.eligible_sha256) != 64:
            raise AssertionError("dataset aggregation digest is malformed")


def main() -> int:
    test_probability_contract()
    test_derivatives()
    test_recovery_and_determinism()
    test_artifact_validation()
    test_fail_closed_support()
    test_fail_closed_non_positive_slope()
    test_dataset_aggregation()
    print("Horde Davidson WDL calibration tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
