#!/usr/bin/env python3
"""Exact vectorized evaluator for authenticated Horde V3 integer containers.

The forward pass is the frozen V3 contract, evaluated entirely in integers::

    acc      = ft_bias + sum of ft_weights over the active V3 rows      (int32)
    act      = clip(max(acc, 0) >> 6, 0, 127)                           (uint8)
    bucket   = phase_lookup[white_piece_count]
    h0_pre   = hidden0_bias[b*16+j] + sum_k hidden0_weights[b*16+j, k] * act[k]
    h0       = clip(max(h0_pre, 0) >> 6, 0, 127)
    h1_pre   = hidden1_bias[b*32+j] + sum_k hidden1_weights[b*32+j, k] * h0[k]
    h1       = clip(max(h1_pre, 0) >> 6, 0, 127)
    out_pre  = output_bias[b*2+stm] + sum_k output_weights[b*2+stm, k] * h1[k]
    psqt_sum = sum over the SAME active rows of psqt_weights[row, b*2+stm]
    value    = trunc_toward_zero(out_pre + psqt_sum, 16)
    damped   = trunc_toward_zero(value * (100 - min(rule50, 100)), 100)
    result   = clip(damped, -31506, 31506)

``out_pre`` and ``psqt_sum`` are both at scale 9600, so their sum divided by 16
is exactly ``600 * v``.  The only floating point anywhere is the float64 GEMM in
``_exact_affine``, whose operands and results are integers far below ``2**53``
and which re-checks that the result really was an exact integer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from . import horde_bin_v1 as wire
    from .horde_training_decoder import BLACK, WHITE, v3_global_rows
    from .horde_v3_container import (
        ACTIVATION_MAX,
        ACTIVATION_MIN,
        DENSE_ACTIVATION_SHIFT,
        FT_ACTIVATION_SHIFT,
        FT_LANES,
        HIDDEN0_LANES,
        HIDDEN1_LANES,
        OUTPUT_DIVISOR,
        OUTPUT_HEADS,
        PHASE_BUCKETS,
        PHASE_LOOKUP_ENTRIES,
        PSQT_COLUMNS,
        SECTIONS,
        SECTIONS_BY_NAME,
        ParsedContainer,
        read_container,
    )
except ImportError:
    import horde_bin_v1 as wire
    from horde_training_decoder import BLACK, WHITE, v3_global_rows
    from horde_v3_container import (
        ACTIVATION_MAX,
        ACTIVATION_MIN,
        DENSE_ACTIVATION_SHIFT,
        FT_ACTIVATION_SHIFT,
        FT_LANES,
        HIDDEN0_LANES,
        HIDDEN1_LANES,
        OUTPUT_DIVISOR,
        OUTPUT_HEADS,
        PHASE_BUCKETS,
        PHASE_LOOKUP_ENTRIES,
        PSQT_COLUMNS,
        SECTIONS,
        SECTIONS_BY_NAME,
        ParsedContainer,
        read_container,
    )


VALUE_TB_WIN_IN_MAX_PLY = 31_507
NNUE_TO_SCORE = 600.0
MAXIMUM_WHITE_PIECE_COUNT = PHASE_LOOKUP_ENTRIES - 1

# Bound the transient gather in the sparse accumulator to roughly 64 MiB.
SPARSE_CHUNK_RECORDS = 256

NUMPY_DTYPES = {"i8": "<i1", "i16": "<i2", "i32": "<i4"}
DTYPE_LIMITS = {
    "i8": (-(1 << 7), (1 << 7) - 1),
    "i16": (-(1 << 15), (1 << 15) - 1),
    "i32": (-(1 << 31), (1 << 31) - 1),
}


class IntegerEvaluationError(ValueError):
    """Raised when an integer network or evaluation batch violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegerEvaluationError(message)


@dataclass(frozen=True, slots=True)
class V3EvaluationBatch:
    """One CSR batch of V3 streams plus the scalars the forward pass needs."""

    v3_rows: np.ndarray
    row_offsets: np.ndarray
    side_to_move: np.ndarray
    white_piece_count: np.ndarray
    rule50_count: np.ndarray
    record_indices: tuple[int, ...] = ()

    def __len__(self) -> int:
        return int(self.row_offsets.size) - 1

    def active_rows(self, position: int) -> tuple[int, ...]:
        start = int(self.row_offsets[position])
        stop = int(self.row_offsets[position + 1])
        return tuple(int(row) for row in self.v3_rows[start:stop])


def _as_array(values: Sequence[int] | np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    _require(array.ndim == 1, f"{label} is not one dimensional")
    return array


def make_batch(
    streams: Sequence[Sequence[int]],
    side_to_move: Sequence[int],
    white_piece_count: Sequence[int],
    rule50_count: Sequence[int],
    record_indices: Sequence[int] = (),
) -> V3EvaluationBatch:
    """Assemble a CSR batch from per-record V3 row lists."""

    count = len(streams)
    _require(count > 0, "V3 evaluation batch is empty")
    rows: list[int] = []
    offsets = [0]
    for position, stream in enumerate(streams):
        _require(len(stream) > 0, f"record {position} activated no V3 rows")
        rows.extend(int(row) for row in stream)
        offsets.append(len(rows))
    return V3EvaluationBatch(
        v3_rows=np.asarray(rows, dtype=np.int64),
        row_offsets=np.asarray(offsets, dtype=np.int64),
        side_to_move=_as_array(side_to_move, "side_to_move"),
        white_piece_count=_as_array(white_piece_count, "white_piece_count"),
        rule50_count=_as_array(rule50_count, "rule50_count"),
        record_indices=tuple(int(index) for index in record_indices),
    )


def batch_from_records(
    payload: bytes, count: int | None = None, start: int = 0
) -> V3EvaluationBatch:
    """Build a batch from a bare stream of HORDE_BIN_V1 48 byte records.

    Every record is re-authenticated with ``horde_bin_v1.validate_record`` and
    its stream is produced by ``horde_training_decoder.v3_global_rows``, so a
    malformed record can never reach the network.
    """

    stride = wire.RECORD_SIZE
    _require(len(payload) % stride == 0, "record stream has a partial record")
    available = len(payload) // stride
    _require(0 <= start < available, f"record start {start} is outside the stream")
    selected = available - start if count is None else int(count)
    _require(
        0 < selected <= available - start,
        f"cannot take {selected} records from offset {start} of {available}",
    )

    streams: list[tuple[int, ...]] = []
    sides: list[int] = []
    white: list[int] = []
    rule50: list[int] = []
    indices: list[int] = []
    for position in range(selected):
        index = start + position
        record = wire.validate_record(payload[index * stride : (index + 1) * stride], index)
        streams.append(v3_global_rows(record["board"]))
        sides.append(int(record["side"]))
        white.append(int(record["white_count"]))
        rule50.append(int(record["rule50"]))
        indices.append(index)
    return make_batch(streams, sides, white, rule50, indices)


def batch_from_file(
    path: Path, count: int | None = None, start: int = 0
) -> V3EvaluationBatch:
    return batch_from_records(Path(path).read_bytes(), count, start)


def _section(container: ParsedContainer, name: str) -> np.ndarray:
    spec = SECTIONS_BY_NAME.get(name)
    _require(spec is not None, f"container section is not registered: {name}")
    values = np.frombuffer(container.sections[name], dtype=np.dtype(NUMPY_DTYPES[spec.dtype]))
    _require(values.size == spec.elements, f"container section {name} has the wrong size")
    return values.reshape(spec.shape)


def _validate_offsets(offsets: np.ndarray, index_count: int, batch_size: int) -> None:
    _require(offsets.shape == (batch_size + 1,), "sparse offsets have the wrong shape")
    _require(
        int(offsets[0]) == 0
        and int(offsets[-1]) == index_count
        and bool(np.all(offsets[1:] > offsets[:-1])),
        "sparse batch contains an empty or malformed bag",
    )


def _sparse_sum(
    weights: np.ndarray, indices: np.ndarray, offsets: np.ndarray, batch_size: int
) -> np.ndarray:
    """Exact int64 row sums over a CSR bag, chunked to bound peak memory."""

    _validate_offsets(offsets, int(indices.size), batch_size)
    _require(
        indices.ndim == 1
        and bool(np.all(indices >= 0))
        and bool(np.all(indices < weights.shape[0])),
        "sparse batch contains an out-of-range V3 row",
    )
    result = np.empty((batch_size, weights.shape[1]), dtype=np.int64)
    start = 0
    while start < batch_size:
        stop = min(start + SPARSE_CHUNK_RECORDS, batch_size)
        low = int(offsets[start])
        high = int(offsets[stop])
        selected = weights[indices[low:high]]
        local = offsets[start:stop] - low
        result[start:stop] = np.add.reduceat(selected, local, axis=0, dtype=np.int64)
        start = stop
    return result


def _activation(values: np.ndarray, shift: int) -> np.ndarray:
    positive = np.maximum(values, 0)
    return np.clip(positive >> shift, ACTIVATION_MIN, ACTIVATION_MAX).astype(
        np.int64, copy=False
    )


def _exact_affine(
    inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray, label: str
) -> np.ndarray:
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


def _gather_bucket_rows(values: np.ndarray, bucket: np.ndarray, lanes: int) -> np.ndarray:
    """Select the ``lanes`` bucket-major columns belonging to each record."""

    columns = bucket[:, None] * lanes + np.arange(lanes, dtype=np.int64)[None, :]
    return np.take_along_axis(values, columns, axis=1)


@dataclass(frozen=True, slots=True)
class IntegerNetworkV3:
    """Decoded immutable V3 parameters with exact dense-affine caches."""

    container: ParsedContainer
    ft_weights: np.ndarray
    ft_bias: np.ndarray
    psqt_weights: np.ndarray
    hidden0_weights: np.ndarray
    hidden0_bias: np.ndarray
    hidden1_weights: np.ndarray
    hidden1_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray
    phase_lookup: np.ndarray

    @classmethod
    def from_container(cls, container: ParsedContainer) -> "IntegerNetworkV3":
        return cls(
            container=container,
            ft_weights=_section(container, "ft_weights"),
            ft_bias=_section(container, "ft_bias").astype(np.int64),
            psqt_weights=_section(container, "psqt_weights").astype(np.int64),
            hidden0_weights=_section(container, "hidden0_weights").astype(np.float64),
            hidden0_bias=_section(container, "hidden0_bias").astype(np.float64),
            hidden1_weights=_section(container, "hidden1_weights").astype(np.float64),
            hidden1_bias=_section(container, "hidden1_bias").astype(np.float64),
            output_weights=_section(container, "output_weights").astype(np.float64),
            output_bias=_section(container, "output_bias").astype(np.float64),
            phase_lookup=_section(container, "phase_lookup").astype(np.int64),
        )

    @classmethod
    def load(cls, path: Path) -> "IntegerNetworkV3":
        return cls.from_container(read_container(path))

    def buckets(self, white_piece_count: np.ndarray) -> np.ndarray:
        _require(
            bool(
                np.all(white_piece_count >= 0)
                and np.all(white_piece_count <= MAXIMUM_WHITE_PIECE_COUNT)
            ),
            "white_piece_count is outside the registered 0..36 domain",
        )
        return self.phase_lookup[white_piece_count]

    def evaluate_layers(self, batch: V3EvaluationBatch) -> dict[str, np.ndarray]:
        """Run the exact integer forward pass and expose every intermediate."""

        batch_size = len(batch)
        _require(batch_size > 0, "integer evaluation batch is empty")
        rows = np.asarray(batch.v3_rows, dtype=np.int64)
        offsets = np.asarray(batch.row_offsets, dtype=np.int64)
        side_to_move = np.asarray(batch.side_to_move, dtype=np.int64)
        white_piece_count = np.asarray(batch.white_piece_count, dtype=np.int64)
        rule50 = np.asarray(batch.rule50_count, dtype=np.int64)
        _require(
            side_to_move.shape
            == rule50.shape
            == white_piece_count.shape
            == (batch_size,),
            "integer evaluation scalar batch is malformed",
        )
        _require(
            bool(np.all((side_to_move == WHITE) | (side_to_move == BLACK))),
            "integer evaluation side to move is malformed",
        )
        _require(bool(np.all(rule50 >= 0)), "integer evaluation rule-50 count is negative")

        bucket = self.buckets(white_piece_count)

        accumulator = _sparse_sum(self.ft_weights, rows, offsets, batch_size) + self.ft_bias
        _require(
            accumulator.shape == (batch_size, FT_LANES), "integer accumulator width drifted"
        )
        activation = _activation(accumulator, FT_ACTIVATION_SHIFT)

        hidden0_all = _exact_affine(
            activation, self.hidden0_weights, self.hidden0_bias, "hidden0"
        )
        hidden0_pre = _gather_bucket_rows(hidden0_all, bucket, HIDDEN0_LANES)
        hidden0 = _activation(hidden0_pre, DENSE_ACTIVATION_SHIFT)

        hidden1_all = _exact_affine(hidden0, self.hidden1_weights, self.hidden1_bias, "hidden1")
        hidden1_pre = _gather_bucket_rows(hidden1_all, bucket, HIDDEN1_LANES)
        hidden1 = _activation(hidden1_pre, DENSE_ACTIVATION_SHIFT)

        heads = _exact_affine(hidden1, self.output_weights, self.output_bias, "output")
        head_index = bucket * OUTPUT_HEADS + side_to_move
        _require(
            bool(np.all(head_index >= 0) and np.all(head_index < PSQT_COLUMNS)),
            "output head index escaped the bucket table",
        )
        output_pre = heads[np.arange(batch_size), head_index]

        psqt_columns = _sparse_sum(self.psqt_weights, rows, offsets, batch_size)
        psqt_sum = psqt_columns[np.arange(batch_size), head_index]

        value = _trunc_div(output_pre + psqt_sum, OUTPUT_DIVISOR)
        damped = _trunc_div(value * (100 - np.clip(rule50, 0, 100)), 100)
        score = np.clip(
            damped, -VALUE_TB_WIN_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1
        ).astype(np.int32)
        return {
            "accumulator": accumulator,
            "activation": activation,
            "bucket": bucket,
            "hidden0_pre": hidden0_pre,
            "hidden0": hidden0,
            "hidden1_pre": hidden1_pre,
            "hidden1": hidden1,
            "output_pre": output_pre,
            "psqt_sum": psqt_sum,
            "value": value,
            "damped": damped,
            "score": score,
        }

    def evaluate(self, batch: V3EvaluationBatch) -> np.ndarray:
        return self.evaluate_layers(batch)["score"]

    def parameter_health(self) -> dict[str, object]:
        sections: dict[str, object] = {}
        weight_passed = True
        for section in SECTIONS:
            values = _section(self.container, section.name)
            minimum, maximum = DTYPE_LIMITS[section.dtype]
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
        lookup = _section(self.container, "phase_lookup")
        sections["phase_lookup"] = {
            **sections["phase_lookup"],
            "buckets_reached": int(np.unique(lookup).size),
            "monotone_nondecreasing": bool(np.all(np.diff(lookup) >= 0)),
        }
        return {
            "minimum_weight_nonzero_fraction": 0.01,
            "maximum_weight_dtype_boundary_fraction": 0.05,
            "phase_buckets": PHASE_BUCKETS,
            "sections": sections,
            "passed": weight_passed,
        }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("network", type=Path)
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args(argv)

    network = IntegerNetworkV3.load(args.network)
    report: dict[str, object] = {
        "file_sha256": network.container.file_sha256,
        "parameter_sha256": network.container.parameter_sha256,
    }
    if args.health:
        report["parameter_health"] = network.parameter_health()
    if args.records is not None:
        batch = batch_from_file(args.records, args.count)
        scores = network.evaluate(batch)
        report["evaluation"] = {
            "records": len(batch),
            "score_minimum": int(scores.min()),
            "score_maximum": int(scores.max()),
            "score_mean": float(scores.mean()),
        }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
