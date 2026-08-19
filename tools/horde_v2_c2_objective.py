#!/usr/bin/env python3
"""Canonical CPU objective and constant null model for Horde V2 C2."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Mapping, Sequence

import numpy as np
import torch

try:
    from . import horde_bin_v1 as wire
    from .horde_training_decoder import BLACK, HordeBinV1Dataset, WHITE
    from .horde_wdl import MATE_SCORE_THRESHOLD, SIDE_NAMES
except ImportError:
    import horde_bin_v1 as wire
    from horde_training_decoder import BLACK, HordeBinV1Dataset, WHITE
    from horde_wdl import MATE_SCORE_THRESHOLD, SIDE_NAMES


OBJECTIVE_SCHEMA = "HORDE_WDL_HALF_BRIER_CANONICAL_CPU_V1"
LOOKUP_SCORE_MINIMUM = -31_506
LOOKUP_SCORE_MAXIMUM = 31_506
LOOKUP_SCORE_COUNT = LOOKUP_SCORE_MAXIMUM - LOOKUP_SCORE_MINIMUM + 1
SCORE_SCALE = 600.0
LAMBDA = 0.6
RESULT_INDEX = {-1: 0, 0: 1, 1: 2}
FLOAT32_COMMON_SCALE_POWER = 149


class C2ObjectiveError(ValueError):
    """Raised when an objective input violates the frozen C2 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C2ObjectiveError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def float_receipt(value: float) -> dict[str, object]:
    _require(math.isfinite(value), "objective value is non-finite")
    return {
        "decimal": format(value, ".17g"),
        "ieee754_binary64_be": struct.pack(">d", value).hex().upper(),
    }


def float_from_receipt(value: object, label: str) -> float:
    _require(isinstance(value, dict), f"{label} is not a float receipt")
    _require(
        set(value) == {"decimal", "ieee754_binary64_be"},
        f"{label} float receipt fields are invalid",
    )
    decimal = value["decimal"]
    bits = value["ieee754_binary64_be"]
    _require(isinstance(decimal, str), f"{label} decimal encoding is invalid")
    _require(isinstance(bits, str) and len(bits) == 16, f"{label} bit encoding is invalid")
    try:
        decoded = struct.unpack(">d", bytes.fromhex(bits))[0]
        parsed = float(decimal)
    except (ValueError, struct.error) as error:
        raise C2ObjectiveError(f"{label} float receipt is invalid") from error
    _require(
        math.isfinite(decoded) and struct.pack(">d", parsed) == struct.pack(">d", decoded),
        f"{label} float encodings differ",
    )
    return decoded


def _configure_cpu_runtime() -> dict[str, object]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.enabled = False
    _require(torch.get_num_threads() == 1, "canonical WDL runtime is not single-threaded")
    _require(
        torch.get_num_interop_threads() == 1,
        "canonical WDL interop runtime is not single-threaded",
    )
    return {
        "device": "cpu",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": bool(getattr(torch.backends.mkldnn, "enabled", False)),
    }


@dataclass(frozen=True, slots=True)
class FrozenWdlLookup:
    values: np.ndarray
    raw_float32_sha256: str
    parameter_float32_sha256: str
    runtime: dict[str, object]

    def probabilities(self, side: int, score: int) -> tuple[float, float, float]:
        _require(side in (WHITE, BLACK), "lookup side is invalid")
        _require(
            LOOKUP_SCORE_MINIMUM <= score <= LOOKUP_SCORE_MAXIMUM,
            "lookup score is outside the canonical domain",
        )
        row = self.values[side, score - LOOKUP_SCORE_MINIMUM]
        return float(row[0]), float(row[1]), float(row[2])


def build_wdl_lookup(
    parameters: Mapping[str, Sequence[float]],
) -> FrozenWdlLookup:
    """Build the exact float32 CPU WDL table used by later canonical evaluation."""

    runtime = _configure_cpu_runtime()
    _require(set(parameters) == set(SIDE_NAMES.values()), "WDL parameter sides are incomplete")
    ordered = [parameters[SIDE_NAMES[side]] for side in (WHITE, BLACK)]
    calibration = torch.tensor(ordered, dtype=torch.float32, device="cpu")
    _require(calibration.shape == (2, 3), "WDL calibration tensor is not 2x3")
    _require(bool(torch.isfinite(calibration).all()), "WDL calibration is non-finite")
    _require(bool(torch.all(calibration[:, 0] > 0.0)), "WDL calibration slope is not positive")

    scores = torch.arange(
        LOOKUP_SCORE_MINIMUM,
        LOOKUP_SCORE_MAXIMUM + 1,
        dtype=torch.float32,
        device="cpu",
    )
    tables = []
    for side in (WHITE, BLACK):
        selected = calibration[side]
        eta = selected[0] * (scores / SCORE_SCALE) + selected[1]
        logits = torch.stack((-eta, torch.full_like(eta, selected[2]), eta), dim=1)
        tables.append(torch.softmax(logits, dim=1))
    table_tensor = torch.stack(tables, dim=0).contiguous()
    _require(
        table_tensor.shape == (2, LOOKUP_SCORE_COUNT, 3),
        "canonical WDL lookup shape drifted",
    )
    _require(bool(torch.isfinite(table_tensor).all()), "canonical WDL lookup is non-finite")

    table32 = np.ascontiguousarray(table_tensor.numpy().astype("<f4", copy=False))
    parameter32 = np.ascontiguousarray(calibration.numpy().astype("<f4", copy=False))
    values = table32.astype(np.float64)
    values.setflags(write=False)
    return FrozenWdlLookup(
        values=values,
        raw_float32_sha256=_sha256_bytes(table32.tobytes(order="C")),
        parameter_float32_sha256=_sha256_bytes(parameter32.tobytes(order="C")),
        runtime=runtime,
    )


def rule50_postprocess_constant(score: int, rule50_count: int) -> int:
    """Apply engine truncation semantics without a floating-point round trip."""

    _require(type(score) is int, "constant score is not an integer")
    _require(
        LOOKUP_SCORE_MINIMUM <= score <= LOOKUP_SCORE_MAXIMUM,
        "constant score is outside the canonical domain",
    )
    _require(type(rule50_count) is int, "rule-50 count is not an integer")
    clock = min(max(rule50_count, 0), 100)
    magnitude = (abs(score) * (100 - clock)) // 100
    damped = -magnitude if score < 0 else magnitude
    return min(max(damped, LOOKUP_SCORE_MINIMUM), LOOKUP_SCORE_MAXIMUM)


@dataclass(frozen=True, slots=True)
class GroupMoments:
    side: int
    rule50_count: int
    records: int
    eligible_records: int
    result_counts: tuple[int, int, int]
    teacher_sum: tuple[float, float, float]
    teacher_squared_norm_sum: float
    teacher_histogram: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class AggregatedObjective:
    groups: tuple[GroupMoments, ...]
    total_records: int
    eligible_records: int
    mate_records: int
    records_by_side: tuple[int, int]
    eligible_by_side: tuple[int, int]
    mate_by_side: tuple[int, int]
    selection_sha256: str
    grouped_histogram_sha256: str


def aggregate_objective(
    dataset: HordeBinV1Dataset,
    lookup: FrozenWdlLookup,
) -> AggregatedObjective:
    """Aggregate deterministic sufficient moments from the authenticated training split."""

    result_counts: dict[tuple[int, int], list[int]] = {}
    teacher_counts: dict[tuple[int, int], dict[int, int]] = {}
    selection_digest = hashlib.sha256()
    records_by_side = [0, 0]
    eligible_by_side = [0, 0]
    mate_by_side = [0, 0]

    for index in range(len(dataset)):
        decoded = wire.validate_record(dataset.raw_record(index), index)
        side = int(decoded["side"])
        score = int(decoded["score"])
        result = int(decoded["result"])
        reason = int(decoded["reason"])
        clock = min(max(int(decoded["rule50"]), 0), 100)
        eligible = abs(score) < MATE_SCORE_THRESHOLD
        key = (side, clock)
        if key not in result_counts:
            result_counts[key] = [0, 0, 0]
            teacher_counts[key] = {}
        result_counts[key][RESULT_INDEX[result]] += 1
        records_by_side[side] += 1
        if eligible:
            teacher_counts[key][score] = teacher_counts[key].get(score, 0) + 1
            eligible_by_side[side] += 1
        else:
            mate_by_side[side] += 1
        selection_digest.update(
            struct.pack("<QBBhbBB", index, side, clock, score, result, reason, int(eligible))
        )

    _require(sum(records_by_side) == len(dataset), "objective record accounting drifted")
    _require(all(count > 0 for count in records_by_side), "objective lacks one side to move")
    _require(
        all(count > 0 for count in eligible_by_side),
        "objective lacks eligible labels on one side",
    )

    groups: list[GroupMoments] = []
    histogram_digest = hashlib.sha256()
    for side, clock in sorted(result_counts):
        results = tuple(result_counts[(side, clock)])
        histogram = tuple(sorted(teacher_counts[(side, clock)].items()))
        records = sum(results)
        eligible = sum(count for _, count in histogram)
        teacher_sum = tuple(
            math.fsum(
                lookup.probabilities(side, score)[lane] * count
                for score, count in histogram
            )
            for lane in range(3)
        )
        squared_norm_sum = math.fsum(
            math.fsum(component * component for component in lookup.probabilities(side, score))
            * count
            for score, count in histogram
        )
        group = GroupMoments(
            side=side,
            rule50_count=clock,
            records=records,
            eligible_records=eligible,
            result_counts=results,
            teacher_sum=teacher_sum,
            teacher_squared_norm_sum=squared_norm_sum,
            teacher_histogram=histogram,
        )
        groups.append(group)
        histogram_digest.update(
            struct.pack(
                "<BBQQQQQ",
                side,
                clock,
                records,
                eligible,
                results[0],
                results[1],
                results[2],
            )
        )
        histogram_digest.update(struct.pack("<Q", len(histogram)))
        for score, count in histogram:
            histogram_digest.update(struct.pack("<hQ", score, count))

    total_eligible = sum(eligible_by_side)
    total_mates = sum(mate_by_side)
    _require(
        total_eligible + total_mates == len(dataset),
        "objective eligibility accounting drifted",
    )
    return AggregatedObjective(
        groups=tuple(groups),
        total_records=len(dataset),
        eligible_records=total_eligible,
        mate_records=total_mates,
        records_by_side=(records_by_side[WHITE], records_by_side[BLACK]),
        eligible_by_side=(eligible_by_side[WHITE], eligible_by_side[BLACK]),
        mate_by_side=(mate_by_side[WHITE], mate_by_side[BLACK]),
        selection_sha256=selection_digest.hexdigest().upper(),
        grouped_histogram_sha256=histogram_digest.hexdigest().upper(),
    )


def _squared_norm(values: Sequence[float]) -> float:
    return math.fsum(value * value for value in values)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def group_loss_from_moments(
    group: GroupMoments,
    prediction: Sequence[float],
    lambda_value: float = LAMBDA,
) -> tuple[float, float, float]:
    _require(lambda_value == LAMBDA, "C2 lambda drifted")
    prediction_norm = _squared_norm(prediction)
    score_sum = 0.5 * math.fsum(
        (
            group.teacher_squared_norm_sum,
            group.eligible_records * prediction_norm,
            -2.0 * _dot(prediction, group.teacher_sum),
        )
    )
    result_sum = 0.5 * math.fsum(
        (
            float(group.records),
            group.records * prediction_norm,
            -2.0 * _dot(prediction, group.result_counts),
        )
    )
    composite = math.fsum((lambda_value * score_sum, (1.0 - lambda_value) * result_sum))
    _require(
        all(math.isfinite(value) for value in (score_sum, result_sum, composite)),
        "objective group loss is non-finite",
    )
    return composite, score_sum, result_sum


def _half_brier(left: Sequence[float], right: Sequence[float]) -> float:
    return 0.5 * math.fsum((a - b) * (a - b) for a, b in zip(left, right))


def group_loss_recordwise(
    group: GroupMoments,
    prediction: Sequence[float],
    lookup: FrozenWdlLookup,
    lambda_value: float = LAMBDA,
) -> tuple[float, float, float]:
    """Independent histogram reference used only to audit a selected constant."""

    _require(lambda_value == LAMBDA, "C2 lambda drifted")
    score_sum = math.fsum(
        count * _half_brier(lookup.probabilities(group.side, score), prediction)
        for score, count in group.teacher_histogram
    )
    result_sum = math.fsum(
        count
        * _half_brier(
            (1.0 if lane == result_index else 0.0 for lane in range(3)),
            prediction,
        )
        for result_index, count in enumerate(group.result_counts)
    )
    composite = math.fsum((lambda_value * score_sum, (1.0 - lambda_value) * result_sum))
    return composite, score_sum, result_sum


def evaluate_side_constant(
    aggregated: AggregatedObjective,
    lookup: FrozenWdlLookup,
    side: int,
    constant: int,
    *,
    recordwise: bool = False,
) -> tuple[float, float, float]:
    _require(side in (WHITE, BLACK), "constant side is invalid")
    evaluator = group_loss_recordwise if recordwise else group_loss_from_moments
    composite: list[float] = []
    score: list[float] = []
    result: list[float] = []
    for group in aggregated.groups:
        if group.side != side:
            continue
        postprocessed = rule50_postprocess_constant(constant, group.rule50_count)
        prediction = lookup.probabilities(side, postprocessed)
        values = (
            evaluator(group, prediction, lookup)
            if recordwise
            else evaluator(group, prediction)
        )
        composite.append(values[0])
        score.append(values[1])
        result.append(values[2])
    _require(bool(composite), "constant side has no objective groups")
    return math.fsum(composite), math.fsum(score), math.fsum(result)


def _ulp_distance(left: float, right: float) -> int:
    _require(left >= 0.0 and right >= 0.0, "ULP audit expects non-negative losses")
    left_bits = struct.unpack(">Q", struct.pack(">d", left))[0]
    right_bits = struct.unpack(">Q", struct.pack(">d", right))[0]
    return abs(left_bits - right_bits)


def _float32_common_numerator(value: float) -> int:
    """Return the exact non-negative float32 value scaled by 2**149."""

    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    sign = bits >> 31
    exponent = (bits >> 23) & 0xFF
    fraction = bits & ((1 << 23) - 1)
    _require(sign == 0 and exponent != 0xFF, "WDL probability is negative or non-finite")
    if exponent == 0:
        return fraction
    return ((1 << 23) | fraction) << (exponent - 1)


def exact_side_audit(
    aggregated: AggregatedObjective,
    lookup: FrozenWdlLookup,
    side: int,
    constant: int,
) -> dict[str, object]:
    """Prove the selected moment loss equals its histogram expansion exactly."""

    scale = 1 << FLOAT32_COMMON_SCALE_POWER
    cache: dict[tuple[int, int], tuple[int, int, int]] = {}

    def exact_probabilities(selected_side: int, score: int) -> tuple[int, int, int]:
        key = (selected_side, score)
        if key not in cache:
            cache[key] = tuple(
                _float32_common_numerator(value)
                for value in lookup.probabilities(selected_side, score)
            )
        return cache[key]

    score_moment_groups: list[int] = []
    score_histogram_groups: list[int] = []
    result_moment_groups: list[int] = []
    result_histogram_groups: list[int] = []
    for group in aggregated.groups:
        if group.side != side:
            continue
        prediction = exact_probabilities(
            side, rule50_postprocess_constant(constant, group.rule50_count)
        )
        prediction_norm = sum(value * value for value in prediction)
        teacher_sum = [0, 0, 0]
        teacher_norm_sum = 0
        histogram_score = 0
        for score, count in group.teacher_histogram:
            teacher = exact_probabilities(side, score)
            for lane in range(3):
                teacher_sum[lane] += count * teacher[lane]
            teacher_norm_sum += count * sum(value * value for value in teacher)
            histogram_score += count * sum(
                (teacher[lane] - prediction[lane]) ** 2 for lane in range(3)
            )
        moment_score = (
            teacher_norm_sum
            + group.eligible_records * prediction_norm
            - 2 * sum(prediction[lane] * teacher_sum[lane] for lane in range(3))
        )
        moment_result = (
            group.records * scale * scale
            + group.records * prediction_norm
            - 2
            * scale
            * sum(
                prediction[lane] * group.result_counts[lane] for lane in range(3)
            )
        )
        histogram_result = sum(
            count
            * sum(
                ((scale if lane == result_index else 0) - prediction[lane]) ** 2
                for lane in range(3)
            )
            for result_index, count in enumerate(group.result_counts)
        )
        score_moment_groups.append(moment_score)
        score_histogram_groups.append(histogram_score)
        result_moment_groups.append(moment_result)
        result_histogram_groups.append(histogram_result)

    _require(bool(score_moment_groups), "exact audit side has no groups")
    score_moment = sum(score_moment_groups)
    score_histogram = sum(score_histogram_groups)
    result_moment = sum(result_moment_groups)
    result_histogram = sum(result_histogram_groups)
    _require(score_moment == score_histogram, "exact score moments differ from histogram")
    _require(result_moment == result_histogram, "exact result moments differ from histogram")
    # Half-Brier contributes a common 1/2; lambda=3/5 contributes the common 1/5.
    composite_numerator = 3 * score_moment + 2 * result_moment
    return {
        "float32_common_scale_power": FLOAT32_COMMON_SCALE_POWER,
        "half_brier_denominator_factor": 2,
        "lambda_numerator_score_result": [3, 2],
        "lambda_denominator": 5,
        "moment_equals_histogram": True,
        "score_numerator_sha256": _sha256_bytes(str(score_moment).encode("ascii")),
        "result_numerator_sha256": _sha256_bytes(str(result_moment).encode("ascii")),
        "composite_numerator_sha256": _sha256_bytes(
            str(composite_numerator).encode("ascii")
        ),
        "composite_numerator_bits": composite_numerator.bit_length(),
        "lookup_values_used": len(cache),
    }


def fit_side_constant(
    aggregated: AggregatedObjective,
    lookup: FrozenWdlLookup,
    side: int,
) -> dict[str, object]:
    losses: list[tuple[float, int]] = []
    for constant in range(LOOKUP_SCORE_MINIMUM, LOOKUP_SCORE_MAXIMUM + 1):
        loss = evaluate_side_constant(aggregated, lookup, side, constant)[0]
        losses.append((loss, constant))
    best_loss = min(loss for loss, _ in losses)
    minimizers = tuple(constant for loss, constant in losses if loss == best_loss)
    _require(bool(minimizers), "constant search found no minimizer")
    selected = min(minimizers, key=lambda constant: (abs(constant), constant))
    runner_up_loss = min((loss for loss, _ in losses if loss > best_loss), default=best_loss)
    moment = evaluate_side_constant(aggregated, lookup, side, selected)
    reference = evaluate_side_constant(
        aggregated, lookup, side, selected, recordwise=True
    )
    exact_audit = exact_side_audit(aggregated, lookup, side, selected)
    return {
        "side": SIDE_NAMES[side],
        "selected_constant_cp": selected,
        "minimizer_count": len(minimizers),
        "minimizer_minimum_cp": min(minimizers),
        "minimizer_maximum_cp": max(minimizers),
        "minimizer_list_sha256": _sha256_bytes(
            b"".join(struct.pack("<i", constant) for constant in minimizers)
        ),
        "boundary_hit": (
            LOOKUP_SCORE_MINIMUM in minimizers or LOOKUP_SCORE_MAXIMUM in minimizers
        ),
        "loss_sum": float_receipt(moment[0]),
        "score_half_brier_sum": float_receipt(moment[1]),
        "result_half_brier_sum": float_receipt(moment[2]),
        "runner_up_loss_sum": float_receipt(runner_up_loss),
        "runner_up_gap": float_receipt(runner_up_loss - best_loss),
        "recordwise_audit": {
            "loss_sum": float_receipt(reference[0]),
            "score_half_brier_sum": float_receipt(reference[1]),
            "result_half_brier_sum": float_receipt(reference[2]),
            "absolute_loss_difference": float_receipt(abs(moment[0] - reference[0])),
            "loss_ulp_distance": _ulp_distance(moment[0], reference[0]),
        },
        "exact_integer_audit": exact_audit,
    }


def fit_constant_baseline(
    aggregated: AggregatedObjective,
    lookup: FrozenWdlLookup,
) -> dict[str, object]:
    sides = [
        fit_side_constant(aggregated, lookup, side) for side in (WHITE, BLACK)
    ]
    total_loss = math.fsum(float_from_receipt(side["loss_sum"], "side loss") for side in sides)
    return {
        "sides": {str(side["side"]): side for side in sides},
        "composite_loss_sum": float_receipt(total_loss),
        "composite_loss_mean_all_records": float_receipt(total_loss / aggregated.total_records),
    }
