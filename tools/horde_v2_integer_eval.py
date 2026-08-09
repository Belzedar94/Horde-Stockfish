#!/usr/bin/env python3
"""Exact vectorized evaluator for authenticated Horde V2 integer containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    from .horde_training_decoder import BLACK, SparseBatch, WHITE
    from .horde_v2_container import (
        ACTIVATION_MAX,
        FIRST_DOMAIN_ABSOLUTE_NONKING,
        FIRST_DOMAIN_ROYAL,
        FIRST_DOMAIN_ROYAL_RANK8,
        FIRST_LANES,
        GLOBAL_LANES,
        OUTPUT_DIVISOR,
        ParsedContainer,
        read_container,
    )
    from .horde_wdl import SIDE_NAMES
except ImportError:
    from horde_training_decoder import BLACK, SparseBatch, WHITE
    from horde_v2_container import (
        ACTIVATION_MAX,
        FIRST_DOMAIN_ABSOLUTE_NONKING,
        FIRST_DOMAIN_ROYAL,
        FIRST_DOMAIN_ROYAL_RANK8,
        FIRST_LANES,
        GLOBAL_LANES,
        OUTPUT_DIVISOR,
        ParsedContainer,
        read_container,
    )
    from horde_wdl import SIDE_NAMES


VALUE_TB_WIN_IN_MAX_PLY = 31_507
NNUE_TO_SCORE = 600.0
MATE_SCORE_THRESHOLD = 31_507


class IntegerEvaluationError(ValueError):
    """Raised when an integer network or evaluation batch violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegerEvaluationError(message)


def _section(container: ParsedContainer, name: str, dtype: str) -> np.ndarray:
    spec = next((item for item in container.spec.sections if item.name == name), None)
    _require(spec is not None, f"container section is not registered: {name}")
    _require(spec.dtype == dtype, f"container section {name} has the wrong dtype")
    numpy_dtype = {"i8": "<i1", "i16": "<i2", "i32": "<i4"}[dtype]
    values = np.frombuffer(container.sections[name], dtype=np.dtype(numpy_dtype))
    _require(values.size == spec.elements, f"container section {name} has the wrong size")
    return values.reshape(spec.shape)


def _validate_offsets(offsets: np.ndarray, index_count: int, batch_size: int, label: str) -> None:
    _require(offsets.shape == (batch_size + 1,), f"{label} offsets have the wrong shape")
    _require(
        int(offsets[0]) == 0
        and int(offsets[-1]) == index_count
        and bool(np.all(offsets[1:] >= offsets[:-1]))
        and bool(np.all(offsets[1:] > offsets[:-1])),
        f"{label} contains an empty or malformed sparse bag",
    )


def _sparse_sum(
    weights: np.ndarray,
    bias: np.ndarray,
    indices: np.ndarray,
    offsets: np.ndarray,
    batch_size: int,
    label: str,
) -> np.ndarray:
    _validate_offsets(offsets, int(indices.size), batch_size, label)
    _require(
        indices.ndim == 1
        and bool(np.all(indices >= 0))
        and bool(np.all(indices < weights.shape[0])),
        f"{label} contains an out-of-range sparse row",
    )
    selected = weights[indices]
    summed = np.add.reduceat(selected, offsets[:-1], axis=0, dtype=np.int64)
    return summed + bias


def _activation(values: np.ndarray) -> np.ndarray:
    positive = np.maximum(values, 0)
    return np.minimum(positive >> 6, ACTIVATION_MAX).astype(np.int64, copy=False)


def _exact_affine(inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray, label: str) -> np.ndarray:
    # Every operand and result is an integer far below 2**53. Float64 BLAS is
    # therefore exact while being substantially faster than NumPy integer GEMM.
    result = inputs.astype(np.float64, copy=False) @ weights.T
    result += bias
    _require(bool(np.all(np.isfinite(result))), f"{label} produced a non-finite affine value")
    _require(
        float(np.max(np.abs(result), initial=0.0)) < float(1 << 53),
        f"{label} exceeded exact float64 integer range",
    )
    integer = result.astype(np.int64)
    _require(
        bool(np.array_equal(result, integer.astype(np.float64))),
        f"{label} float64 affine was not an exact integer",
    )
    return integer


def _trunc_div(values: np.ndarray, divisor: int) -> np.ndarray:
    _require(divisor > 0, "integer divisor must be positive")
    magnitude = np.abs(values) // divisor
    return np.where(values < 0, -magnitude, magnitude)


@dataclass(frozen=True, slots=True)
class IntegerNetwork:
    """Decoded immutable parameters with exact dense-affine caches."""

    container: ParsedContainer
    first_weights: np.ndarray
    first_bias: np.ndarray
    global_weights: np.ndarray
    global_bias: np.ndarray
    hidden0_weights: np.ndarray
    hidden0_bias: np.ndarray
    hidden1_weights: np.ndarray
    hidden1_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray

    @classmethod
    def from_container(cls, container: ParsedContainer) -> "IntegerNetwork":
        spec = container.spec
        return cls(
            container=container,
            first_weights=_section(container, spec.first_weight_name, "i16"),
            first_bias=_section(container, spec.first_bias_name, "i32").astype(np.int64),
            global_weights=_section(container, "global_weights", "i16"),
            global_bias=_section(container, "global_bias", "i32").astype(np.int64),
            hidden0_weights=_section(container, "hidden0_weights", "i8").astype(np.float64),
            hidden0_bias=_section(container, "hidden0_bias", "i32").astype(np.float64),
            hidden1_weights=_section(container, "hidden1_weights", "i8").astype(np.float64),
            hidden1_bias=_section(container, "hidden1_bias", "i32").astype(np.float64),
            output_weights=_section(container, "output_weights", "i8").astype(np.float64),
            output_bias=_section(container, "output_bias", "i32").astype(np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> "IntegerNetwork":
        return cls.from_container(read_container(path))

    def evaluate(self, batch: SparseBatch) -> np.ndarray:
        batch_size = len(batch)
        _require(batch_size > 0, "integer evaluation batch is empty")
        global_rows = np.asarray(batch.v2_global, dtype=np.int64)
        global_offsets = np.asarray(batch.global_offsets, dtype=np.int64)
        royal_rows = np.asarray(batch.v2_royal, dtype=np.int64)
        royal_offsets = np.asarray(batch.royal_offsets, dtype=np.int64)
        side_to_move = np.asarray(batch.side_to_move, dtype=np.int64)
        rule50 = np.asarray(batch.rule50_count, dtype=np.int64)
        _require(
            side_to_move.shape == rule50.shape == (batch_size,)
            and bool(np.all((side_to_move == WHITE) | (side_to_move == BLACK))),
            "integer evaluation side or rule-50 batch is malformed",
        )

        domain = self.container.spec.first_domain_code
        if domain == FIRST_DOMAIN_ROYAL:
            first_rows = royal_rows
            first_offsets = royal_offsets
        elif domain == FIRST_DOMAIN_ROYAL_RANK8:
            rows_per_bucket = 10 * 64
            buckets = royal_rows // rows_per_bucket
            first_rows = (buckets // 4) * rows_per_bucket + royal_rows % rows_per_bucket
            first_offsets = royal_offsets
        elif domain == FIRST_DOMAIN_ABSOLUTE_NONKING:
            keep = global_rows < 10 * 64
            prefix = np.concatenate(
                (np.zeros(1, dtype=np.int64), np.cumsum(keep, dtype=np.int64))
            )
            first_rows = global_rows[keep]
            first_offsets = prefix[global_offsets]
        else:  # pragma: no cover - the container codec rejects this first
            raise IntegerEvaluationError("integer evaluator received an unknown first domain")

        first = _sparse_sum(
            self.first_weights,
            self.first_bias,
            first_rows,
            first_offsets,
            batch_size,
            "first domain",
        )
        global_ = _sparse_sum(
            self.global_weights,
            self.global_bias,
            global_rows,
            global_offsets,
            batch_size,
            "global domain",
        )
        transformed = np.concatenate((_activation(first), _activation(global_)), axis=1)
        _require(
            transformed.shape == (batch_size, FIRST_LANES + GLOBAL_LANES),
            "integer transformed width drifted",
        )
        hidden0 = _activation(
            _exact_affine(transformed, self.hidden0_weights, self.hidden0_bias, "hidden0")
        )
        hidden1 = _activation(
            _exact_affine(hidden0, self.hidden1_weights, self.hidden1_bias, "hidden1")
        )
        heads = _exact_affine(hidden1, self.output_weights, self.output_bias, "output")
        selected = heads[np.arange(batch_size), side_to_move]
        pre_rule50 = _trunc_div(selected, OUTPUT_DIVISOR)
        bounded_rule50 = np.clip(rule50, 0, 100)
        damped = _trunc_div(pre_rule50 * (100 - bounded_rule50), 100)
        return np.clip(
            damped,
            -VALUE_TB_WIN_IN_MAX_PLY + 1,
            VALUE_TB_WIN_IN_MAX_PLY - 1,
        ).astype(np.int32)

    def parameter_health(self) -> dict[str, object]:
        sections: dict[str, object] = {}
        weight_passed = True
        for section in self.container.spec.sections:
            values = _section(self.container, section.name, section.dtype)
            minimum, maximum = {
                "i8": (-(1 << 7), (1 << 7) - 1),
                "i16": (-(1 << 15), (1 << 15) - 1),
                "i32": (-(1 << 31), (1 << 31) - 1),
            }[section.dtype]
            zero_fraction = float(np.count_nonzero(values == 0) / values.size)
            boundary_fraction = float(
                np.count_nonzero((values == minimum) | (values == maximum)) / values.size
            )
            is_weight = section.name.endswith("_weights")
            section_passed = not is_weight or (
                (1.0 - zero_fraction) >= 0.01 and boundary_fraction <= 0.05
            )
            weight_passed = weight_passed and section_passed
            sections[section.name] = {
                "elements": int(values.size),
                "integer_minimum": int(values.min()),
                "integer_maximum": int(values.max()),
                "zero_fraction": zero_fraction,
                "dtype_boundary_fraction": boundary_fraction,
                "weight_health_passed": section_passed,
            }
        return {
            "minimum_weight_nonzero_fraction": 0.01,
            "maximum_weight_dtype_boundary_fraction": 0.05,
            "sections": sections,
            "passed": weight_passed,
        }


def loss_arrays(
    predicted_scores: np.ndarray,
    teacher_scores: np.ndarray,
    results: np.ndarray,
    side_to_move: np.ndarray,
    lambda_value: float,
    calibration: Mapping[str, Sequence[float]],
) -> dict[str, np.ndarray]:
    _require(0.0 <= lambda_value <= 1.0, "metric lambda is outside [0, 1]")
    predicted_scores = np.asarray(predicted_scores, dtype=np.float64)
    teacher_scores = np.asarray(teacher_scores, dtype=np.int64)
    results = np.asarray(results, dtype=np.int64)
    side_to_move = np.asarray(side_to_move, dtype=np.int64)
    count = predicted_scores.size
    _require(
        teacher_scores.shape == results.shape == side_to_move.shape == (count,),
        "metric arrays have different shapes",
    )
    parameters = np.asarray(
        [calibration[SIDE_NAMES[WHITE]], calibration[SIDE_NAMES[BLACK]]],
        dtype=np.float64,
    )
    _require(
        parameters.shape == (2, 3)
        and bool(np.all(np.isfinite(parameters)))
        and bool(np.all(parameters[:, 0] > 0.0)),
        "metric calibration is invalid",
    )

    def probabilities(scores: np.ndarray) -> np.ndarray:
        selected = parameters[side_to_move]
        eta = selected[:, 0] * (scores / NNUE_TO_SCORE) + selected[:, 1]
        logits = np.column_stack((-eta, selected[:, 2], eta))
        logits -= np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(logits)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    prediction = probabilities(predicted_scores)
    eligible = np.abs(teacher_scores) < MATE_SCORE_THRESHOLD
    teacher_input = np.where(eligible, teacher_scores, 0).astype(np.float64)
    teacher = probabilities(teacher_input)
    _require(bool(np.all((results >= -1) & (results <= 1))), "metric result is invalid")
    result_target = np.eye(3, dtype=np.float64)[results + 1]
    score_error = 0.5 * np.sum(np.square(teacher - prediction), axis=1)
    result_error = 0.5 * np.sum(np.square(result_target - prediction), axis=1)
    composite = lambda_value * eligible * score_error + (1.0 - lambda_value) * result_error
    return {
        "composite": composite,
        "score_error": score_error,
        "result_error": result_error,
        "prediction": prediction,
        "score_eligible": eligible,
    }


@dataclass(slots=True)
class MetricAccumulator:
    samples: int = 0
    score_eligible: int = 0
    composite_sum: float = 0.0
    score_sum: float = 0.0
    result_sum: float = 0.0
    prediction_sum_loss: float = 0.0
    prediction_sum_draw: float = 0.0
    prediction_sum_win: float = 0.0

    def update(self, terms: Mapping[str, np.ndarray], mask: np.ndarray | None = None) -> None:
        composite = terms["composite"]
        selected = np.ones(composite.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        _require(selected.shape == composite.shape, "metric slice mask has the wrong shape")
        if not bool(np.any(selected)):
            return
        eligible = terms["score_eligible"][selected]
        prediction = terms["prediction"][selected]
        self.samples += int(np.count_nonzero(selected))
        self.score_eligible += int(np.count_nonzero(eligible))
        self.composite_sum += float(np.sum(composite[selected], dtype=np.float64))
        self.score_sum += float(
            np.sum(terms["score_error"][selected][eligible], dtype=np.float64)
        )
        self.result_sum += float(np.sum(terms["result_error"][selected], dtype=np.float64))
        prediction_sums = np.sum(prediction, axis=0, dtype=np.float64)
        self.prediction_sum_loss += float(prediction_sums[0])
        self.prediction_sum_draw += float(prediction_sums[1])
        self.prediction_sum_win += float(prediction_sums[2])

    def receipt(self) -> dict[str, object]:
        _require(self.samples > 0, "metric accumulator is empty")
        return {
            "samples": self.samples,
            "score_eligible": self.score_eligible,
            "mate_scores_masked": self.samples - self.score_eligible,
            "composite_loss": self.composite_sum / self.samples,
            "score_half_brier_eligible": (
                self.score_sum / self.score_eligible if self.score_eligible else None
            ),
            "result_half_brier": self.result_sum / self.samples,
            "prediction_mean_wdl": [
                self.prediction_sum_loss / self.samples,
                self.prediction_sum_draw / self.samples,
                self.prediction_sum_win / self.samples,
            ],
        }
