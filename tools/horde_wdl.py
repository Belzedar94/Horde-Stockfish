#!/usr/bin/env python3
"""Deterministic side-specific Davidson calibration for Horde score labels."""

from __future__ import annotations

from array import array
from bisect import bisect_right
from collections import deque
import concurrent.futures
from itertools import islice
import mmap
import os
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable, Sequence

try:
    from . import horde_bin_v1 as wire
    from .horde_training_decoder import BLACK, HordeBinV1Dataset, WHITE
except ImportError:
    import horde_bin_v1 as wire
    from horde_training_decoder import BLACK, HordeBinV1Dataset, WHITE


SCHEMA = "HORDE_WDL_CALIBRATION_V1"

# Provenance. The aggregation pass was sharded after the 50M campaign, and the
# artifact says so rather than leaving a future reader to diff two tools and
# guess whether the difference was semantic. Both digests are order-dependent
# SHA-256 chains, which is exactly what sharding could have broken, so the
# equivalence is evidenced: this pass was re-run on the corpus the 50M
# calibration was fitted from and had to reproduce both digests and both
# counters exactly. Artifacts fitted before the change carry no such block and
# stay valid, because the frozen calibration is the comparability currency of
# the V2 era and nothing here may retire it.
AGGREGATION_IMPLEMENTATION = {
    "pass": (
        "parallel over contiguous record shards; each shard returns its own "
        "byte stream and the parent folds those streams into both digests in "
        "shard order, so each digest sees exactly the bytes the serial pass "
        "fed it"
    ),
    "changed_after_campaign": "horde-v2-rank8-scale-20260810",
    "equivalence_evidence": {
        "method": (
            "re-ran this pass on the corpus the 50M calibration was fitted "
            "from and required its published digests and counters to "
            "reproduce exactly"
        ),
        "published_artifact_selection": {
            "selection_sha256": "9399CBEC6D389F0A4F1051F444486F7EE558CF4789F4BD4F6F5D8E18C1C69AD9",
            "eligible_records_sha256": "7B89800FD307F605980FC74ED74F214C771A79B037636090ACA849C424E031A2",
            "total_records": 50000000,
            "eligible_records": 48378691,
        },
    },
}
LINK_SCHEMA = "DAVIDSON_STM_SOFTMAX_V1"
SCORE_SCALE = 600.0
MATE_SCORE_THRESHOLD = 31_507
DEFAULT_MINIMUM_CLASS_SUPPORT = 32
DEFAULT_GRADIENT_TOLERANCE = 1.0e-11
DEFAULT_MAXIMUM_ITERATIONS = 100
DEFAULT_CONDITION_LIMIT = 1.0e12
SIDE_NAMES = {WHITE: "white_to_move", BLACK: "black_to_move"}
RESULT_INDEX = {-1: 0, 0: 1, 1: 2}
SCORE_DOMAIN = 1 << 16


class CalibrationError(ValueError):
    """Raised when a calibration input or optimum violates the frozen contract."""


@dataclass(frozen=True, slots=True)
class WeightedObservation:
    score: int
    result: int
    count: int


@dataclass(frozen=True, slots=True)
class AggregatedLabels:
    by_side: tuple[tuple[WeightedObservation, ...], tuple[WeightedObservation, ...]]
    total_records: int
    eligible_records: int
    mate_records_excluded: int
    class_counts: tuple[tuple[int, int, int], tuple[int, int, int]]
    mate_counts: tuple[int, int]
    selection_sha256: str
    eligible_sha256: str
    # HORDE_BIN_V1_R2: children excluded because the fit was restricted to
    # parents. Zero for every corpus that has no children, so a plain corpus
    # produces the artifact it always produced.
    parents_only: bool = False
    child_records_excluded: int = 0
    child_counts: tuple[int, int] = (0, 0)


@dataclass(frozen=True, slots=True)
class SideFit:
    a: float
    b: float
    d: float
    observations: int
    class_counts: tuple[int, int, int]
    score_min: int
    score_max: int
    iterations: int
    categorical_nll: float
    mean_brier: float
    mean_half_brier: float
    gradient_inf_norm: float
    hessian_eigenvalues: tuple[float, float, float]
    hessian_condition: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def _float_receipt(value: float) -> dict[str, object]:
    _require(math.isfinite(value), "calibration parameter is non-finite")
    return {
        "decimal": format(value, ".17g"),
        "ieee754_binary64_be": struct.pack(">d", value).hex().upper(),
    }


def _float_from_receipt(value: object, label: str) -> float:
    _require(isinstance(value, dict), f"{label} parameter receipt is not an object")
    _require(
        set(value) == {"decimal", "ieee754_binary64_be"},
        f"{label} parameter receipt fields are invalid",
    )
    decimal = value["decimal"]
    bits = value["ieee754_binary64_be"]
    _require(isinstance(decimal, str), f"{label} decimal parameter is invalid")
    _require(isinstance(bits, str) and len(bits) == 16, f"{label} binary64 parameter is invalid")
    try:
        decoded = struct.unpack(">d", bytes.fromhex(bits))[0]
        parsed = float(decimal)
    except (ValueError, struct.error) as error:
        raise CalibrationError(f"{label} parameter encoding is invalid") from error
    _require(math.isfinite(decoded), f"{label} parameter is non-finite")
    _require(struct.pack(">d", parsed) == struct.pack(">d", decoded), f"{label} encodings differ")
    return decoded


def probabilities(score: float, parameters: Sequence[float]) -> tuple[float, float, float]:
    """Return loss/draw/win probabilities for one post-rule50 score."""

    _require(len(parameters) == 3, "Davidson parameter vector must have length three")
    a, b, d = (float(value) for value in parameters)
    _require(all(math.isfinite(value) for value in (score, a, b, d)), "non-finite WDL input")
    _require(a > 0.0, "Davidson slope A must be positive")
    eta = a * (score / SCORE_SCALE) + b
    maximum = max(-eta, d, eta)
    loss = math.exp(-eta - maximum)
    draw = math.exp(d - maximum)
    win = math.exp(eta - maximum)
    total = loss + draw + win
    return loss / total, draw / total, win / total


# The aggregation pass is per-record independent apart from two order-dependent
# SHA-256 chains. Shards are CONTIGUOUS and each shard returns its own byte
# stream; the parent folds those streams into the digests IN SHARD ORDER, so
# both digests are computed over exactly the bytes the serial pass fed them.
# Everything else the pass accumulates is a sum and does not care about order.
#
# The mmap shard reader is deliberately a local copy rather than an import from
# the selection tool: these are two gate-critical tools and an import edge
# between them would couple their failure modes for thirty lines of code.
WDL_SHARD_RECORDS = 500_000
WDL_WORKERS = max(1, (os.cpu_count() or 2) - 2)
_WDL_STATE: dict[str, Any] = {}


def _wdl_source_table(dataset: Any) -> tuple[tuple[str, int, int], ...] | None:
    """(path, global begin, records) per backing file, or None if unsupported."""

    receipt = getattr(dataset, "receipt", None)
    if isinstance(receipt, dict) and isinstance(receipt.get("chunks"), list):
        root = Path(dataset.path).parent.resolve()
        table = []
        for entry in receipt["chunks"]:
            path = (root / Path(*Path(str(entry["path"])).parts)).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                return None
            table.append((str(path), int(entry["global_begin"]), int(entry["records"])))
        return tuple(table)
    path = getattr(dataset, "path", None)
    manifest = getattr(dataset, "manifest", None)
    if path is not None and isinstance(manifest, dict) and "record_count" in manifest:
        return ((str(Path(path).resolve()), 0, int(manifest["record_count"])),)
    return None


def _wdl_init(table: tuple[tuple[str, int, int], ...]) -> None:
    _WDL_STATE.clear()
    _WDL_STATE["table"] = table
    _WDL_STATE["starts"] = [entry[1] for entry in table]
    _WDL_STATE["slot"] = None
    _WDL_STATE["mapping"] = None
    _WDL_STATE["handle"] = None


def _wdl_close() -> None:
    if _WDL_STATE.get("mapping") is not None:
        _WDL_STATE["mapping"].close()
    if _WDL_STATE.get("handle") is not None:
        _WDL_STATE["handle"].close()
    _WDL_STATE["mapping"] = None
    _WDL_STATE["handle"] = None
    _WDL_STATE["slot"] = None


def _wdl_raw(index: int) -> bytes:
    state = _WDL_STATE
    if state["slot"] is None or not (state["lo"] <= index < state["hi"]):
        slot = bisect_right(state["starts"], index) - 1
        path, begin, records = state["table"][slot]
        _wdl_close()
        handle = open(path, "rb")
        state["handle"] = handle
        state["mapping"] = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        state["slot"] = slot
        state["lo"] = begin
        state["hi"] = begin + records
    offset = wire.HEADER_SIZE + (index - state["lo"]) * wire.RECORD_SIZE
    return state["mapping"][offset : offset + wire.RECORD_SIZE]


def _wdl_shard(job: tuple[int, int, bool]) -> tuple[Any, ...]:
    begin, end, parents_only = job
    histograms: tuple[dict[int, int], dict[int, int]] = ({}, {})
    class_counts = [[0, 0, 0], [0, 0, 0]]
    mate_counts = [0, 0]
    child_counts = [0, 0]
    eligible_records = 0
    selection_stream = bytearray()
    eligible_stream = bytearray()

    for index in range(begin, end):
        decoded = wire.validate_record(_wdl_raw(index), index)
        side = decoded["side"]
        score = decoded["score"]
        result = decoded["result"]
        reason = decoded["reason"]
        is_child = parents_only and decoded["family"] != 0
        eligible = abs(score) < MATE_SCORE_THRESHOLD and not is_child
        encoded = struct.pack(
            "<QBhbBB", index, side, score, result, reason, int(eligible)
        )
        selection_stream += encoded
        if not eligible:
            if is_child:
                child_counts[side] += 1
            else:
                mate_counts[side] += 1
            continue
        result_index = RESULT_INDEX[result]
        key = ((score + 32_768) * 3) + result_index
        histogram = histograms[side]
        histogram[key] = histogram.get(key, 0) + 1
        class_counts[side][result_index] += 1
        eligible_records += 1
        eligible_stream += encoded[:-1]

    return (
        histograms, class_counts, mate_counts, child_counts, eligible_records,
        bytes(selection_stream), bytes(eligible_stream),
    )


def _wdl_shards(total: int, parents_only: bool) -> list[tuple[int, int, bool]]:
    return [
        (begin, min(begin + WDL_SHARD_RECORDS, total), parents_only)
        for begin in range(0, total, WDL_SHARD_RECORDS)
    ]


def aggregate_labels(
    dataset: HordeBinV1Dataset,
    *,
    parents_only: bool = False,
) -> AggregatedLabels:
    """Aggregate sufficient score/result statistics in one bounded-memory pass.

    With ``parents_only`` the fit sees self-play line samples and not expansion
    children. A child inherits its parent's game result, so counting it as an
    independent observation reweights the very score-to-result mapping the fit
    exists to measure, and it reweights it hardest in exactly the tactical
    positions expansion targets.

    A child is excluded the same way a mate score is: it enters the selection
    digest marked ineligible, so the digest still describes every record the
    pass saw and what was decided about it.
    """

    histograms = (array("Q", [0]) * (SCORE_DOMAIN * 3), array("Q", [0]) * (SCORE_DOMAIN * 3))
    class_counts = [[0, 0, 0], [0, 0, 0]]
    mate_counts = [0, 0]
    child_counts = [0, 0]
    selection_digest = hashlib.sha256()
    eligible_digest = hashlib.sha256()
    eligible_records = 0

    table = _wdl_source_table(dataset)
    shards = _wdl_shards(len(dataset), parents_only) if table else []
    workers = min(WDL_WORKERS, len(shards)) if table else 1
    if table is not None and workers > 1:
        pending: deque[Any] = deque()
        remaining = iter(shards)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, initializer=_wdl_init, initargs=(table,)
        ) as pool:
            for shard in islice(remaining, workers * 2):
                pending.append(pool.submit(_wdl_shard, shard))
            while pending:
                (shard_histograms, shard_classes, shard_mates, shard_children,
                 shard_eligible, selection_stream, eligible_stream) = (
                    pending.popleft().result())
                following = next(remaining, None)
                if following is not None:
                    pending.append(pool.submit(_wdl_shard, following))
                selection_digest.update(selection_stream)
                eligible_digest.update(eligible_stream)
                for side in (WHITE, BLACK):
                    target = histograms[side]
                    for key, count in shard_histograms[side].items():
                        target[key] += count
                    for result_index in range(3):
                        class_counts[side][result_index] += (
                            shard_classes[side][result_index])
                    mate_counts[side] += shard_mates[side]
                    child_counts[side] += shard_children[side]
                eligible_records += shard_eligible
        return _aggregated(
            dataset, histograms, class_counts, mate_counts, child_counts,
            eligible_records, selection_digest, eligible_digest, parents_only,
        )

    # One validation per record, not two: reading the family through a second
    # accessor would re-validate all 48 bytes to recover three bits.
    label = dataset.label_with_family if parents_only else None
    for index in range(len(dataset)):
        if label is None:
            side, score, result, reason = dataset.label(index)
            is_child = False
        else:
            side, score, result, reason, family = label(index)
            is_child = family != 0
        eligible = abs(score) < MATE_SCORE_THRESHOLD and not is_child
        encoded = struct.pack("<QBhbBB", index, side, score, result, reason, int(eligible))
        selection_digest.update(encoded)
        if not eligible:
            # A child that also carries a mate score is counted once, as a
            # child, so the two exclusions never double count.
            if is_child:
                child_counts[side] += 1
            else:
                mate_counts[side] += 1
            continue

        result_index = RESULT_INDEX[result]
        histogram_index = ((score + 32_768) * 3) + result_index
        histograms[side][histogram_index] += 1
        class_counts[side][result_index] += 1
        eligible_records += 1
        eligible_digest.update(encoded[:-1])

    return _aggregated(
        dataset, histograms, class_counts, mate_counts, child_counts,
        eligible_records, selection_digest, eligible_digest, parents_only,
    )


def _aggregated(
    dataset: Any,
    histograms: Any,
    class_counts: Any,
    mate_counts: Any,
    child_counts: Any,
    eligible_records: int,
    selection_digest: Any,
    eligible_digest: Any,
    parents_only: bool,
) -> AggregatedLabels:
    by_side: list[tuple[WeightedObservation, ...]] = []
    for side in (WHITE, BLACK):
        observations: list[WeightedObservation] = []
        histogram = histograms[side]
        for encoded_score in range(SCORE_DOMAIN):
            score = encoded_score - 32_768
            base = encoded_score * 3
            for result_index, result in enumerate((-1, 0, 1)):
                count = int(histogram[base + result_index])
                if count:
                    observations.append(WeightedObservation(score, result, count))
        by_side.append(tuple(observations))

    return AggregatedLabels(
        by_side=(by_side[WHITE], by_side[BLACK]),
        total_records=len(dataset),
        eligible_records=eligible_records,
        mate_records_excluded=sum(mate_counts),
        class_counts=(tuple(class_counts[WHITE]), tuple(class_counts[BLACK])),
        mate_counts=(mate_counts[WHITE], mate_counts[BLACK]),
        selection_sha256=selection_digest.hexdigest().upper(),
        eligible_sha256=eligible_digest.hexdigest().upper(),
        parents_only=parents_only,
        child_records_excluded=sum(child_counts),
        child_counts=(child_counts[WHITE], child_counts[BLACK]),
    )


def aggregate_observations(
    observations: Iterable[tuple[int, int, int]],
) -> tuple[WeightedObservation, ...]:
    """Canonicalize synthetic or test observations without dataset framing."""

    counts: dict[tuple[int, int], int] = {}
    for score, result, count in observations:
        _require(-32_768 <= score <= 32_767, "synthetic score is outside int16")
        _require(result in RESULT_INDEX, "synthetic result is outside {-1,0,1}")
        _require(type(count) is int and count > 0, "synthetic observation count is invalid")
        counts[(score, result)] = counts.get((score, result), 0) + count
    return tuple(
        WeightedObservation(score, result, count)
        for (score, result), count in sorted(counts.items())
    )


def _statistics(
    observations: Sequence[WeightedObservation],
    parameters: Sequence[float],
    *,
    derivatives: bool,
) -> tuple[float, tuple[float, float, float], tuple[tuple[float, float, float], ...], float]:
    a, b, d = parameters
    total_count = sum(observation.count for observation in observations)
    _require(total_count > 0, "empty Davidson observation set")

    nll_terms: list[float] = []
    brier_terms: list[float] = []
    gradient = [0.0, 0.0, 0.0]
    hessian = [[0.0, 0.0, 0.0] for _ in range(3)]

    for observation in observations:
        u = observation.score / SCORE_SCALE
        eta = a * u + b
        maximum = max(-eta, d, eta)
        loss_exp = math.exp(-eta - maximum)
        draw_exp = math.exp(d - maximum)
        win_exp = math.exp(eta - maximum)
        denominator = loss_exp + draw_exp + win_exp
        probabilities_ = (
            loss_exp / denominator,
            draw_exp / denominator,
            win_exp / denominator,
        )
        target_index = RESULT_INDEX[observation.result]
        log_partition = maximum + math.log(denominator)
        target_logit = (-eta, d, eta)[target_index]
        nll_terms.append(observation.count * (log_partition - target_logit))
        brier_terms.append(
            observation.count
            * sum(
                (probability - float(index == target_index)) ** 2
                for index, probability in enumerate(probabilities_)
            )
        )
        if not derivatives:
            continue

        loss_probability, draw_probability, win_probability = probabilities_
        expected_sign = win_probability - loss_probability
        target_sign = float(observation.result)
        sign_error = expected_sign - target_sign
        count = float(observation.count)
        gradient[0] += count * u * sign_error
        gradient[1] += count * sign_error
        gradient[2] += count * (draw_probability - float(observation.result == 0))

        sign_variance = win_probability + loss_probability - expected_sign * expected_sign
        draw_variance = draw_probability * (1.0 - draw_probability)
        sign_draw_covariance = -expected_sign * draw_probability
        hessian[0][0] += count * u * u * sign_variance
        hessian[0][1] += count * u * sign_variance
        hessian[1][1] += count * sign_variance
        hessian[0][2] += count * u * sign_draw_covariance
        hessian[1][2] += count * sign_draw_covariance
        hessian[2][2] += count * draw_variance

    inverse_count = 1.0 / total_count
    hessian[1][0] = hessian[0][1]
    hessian[2][0] = hessian[0][2]
    hessian[2][1] = hessian[1][2]
    return (
        math.fsum(nll_terms) * inverse_count,
        tuple(value * inverse_count for value in gradient),
        tuple(tuple(value * inverse_count for value in row) for row in hessian),
        math.fsum(brier_terms) * inverse_count,
    )


def _solve_positive_definite(
    matrix: Sequence[Sequence[float]], vector: Sequence[float]
) -> tuple[float, float, float]:
    _require(len(matrix) == len(vector) == 3, "Newton system is not 3x3")
    m00, m01, m02 = matrix[0]
    _, m11, m12 = matrix[1]
    _, _, m22 = matrix[2]
    _require(m00 > 0.0 and math.isfinite(m00), "Davidson Hessian is not positive definite")
    l00 = math.sqrt(m00)
    l10 = m01 / l00
    l20 = m02 / l00
    diagonal_1 = m11 - l10 * l10
    _require(diagonal_1 > 0.0 and math.isfinite(diagonal_1), "Davidson Hessian is singular")
    l11 = math.sqrt(diagonal_1)
    l21 = (m12 - l20 * l10) / l11
    diagonal_2 = m22 - l20 * l20 - l21 * l21
    _require(diagonal_2 > 0.0 and math.isfinite(diagonal_2), "Davidson Hessian is singular")
    l22 = math.sqrt(diagonal_2)

    y0 = vector[0] / l00
    y1 = (vector[1] - l10 * y0) / l11
    y2 = (vector[2] - l20 * y0 - l21 * y1) / l22
    x2 = y2 / l22
    x1 = (y1 - l21 * x2) / l11
    x0 = (y0 - l10 * x1 - l20 * x2) / l00
    return x0, x1, x2


def _symmetric_eigenvalues(matrix: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    values = [[float(matrix[row][column]) for column in range(3)] for row in range(3)]
    for _ in range(32):
        off_diagonal = ((0, 1), (0, 2), (1, 2))
        p, q = max(off_diagonal, key=lambda pair: abs(values[pair[0]][pair[1]]))
        if abs(values[p][q]) <= 1.0e-18:
            break
        tau = (values[q][q] - values[p][p]) / (2.0 * values[p][q])
        tangent = math.copysign(1.0, tau) / (abs(tau) + math.sqrt(1.0 + tau * tau))
        cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
        sine = tangent * cosine
        app = values[p][p]
        aqq = values[q][q]
        apq = values[p][q]
        values[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        values[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        values[p][q] = values[q][p] = 0.0
        for r in range(3):
            if r in (p, q):
                continue
            arp = values[r][p]
            arq = values[r][q]
            values[r][p] = values[p][r] = cosine * arp - sine * arq
            values[r][q] = values[q][r] = sine * arp + cosine * arq
    return tuple(sorted(values[index][index] for index in range(3)))


def fit_side(
    observations: Sequence[WeightedObservation],
    *,
    minimum_class_support: int = DEFAULT_MINIMUM_CLASS_SUPPORT,
    gradient_tolerance: float = DEFAULT_GRADIENT_TOLERANCE,
    maximum_iterations: int = DEFAULT_MAXIMUM_ITERATIONS,
    condition_limit: float = DEFAULT_CONDITION_LIMIT,
) -> SideFit:
    """Fit one unweighted three-class Davidson link with constrained A > 0."""

    _require(minimum_class_support > 0, "minimum class support must be positive")
    _require(gradient_tolerance > 0.0, "gradient tolerance must be positive")
    _require(maximum_iterations > 0, "maximum iterations must be positive")
    _require(condition_limit > 1.0, "Hessian condition limit must exceed one")
    _require(bool(observations), "empty Davidson observation set")

    class_counts = [0, 0, 0]
    for observation in observations:
        _require(abs(observation.score) < MATE_SCORE_THRESHOLD, "mate score entered calibration")
        _require(observation.result in RESULT_INDEX, "invalid Davidson result")
        _require(observation.count > 0, "non-positive Davidson weight")
        class_counts[RESULT_INDEX[observation.result]] += observation.count
    _require(
        all(count >= minimum_class_support for count in class_counts),
        "Davidson calibration lacks minimum loss/draw/win support",
    )

    loss_count, draw_count, win_count = class_counts
    parameters = [
        1.0,
        0.5 * math.log(win_count / loss_count),
        math.log((2.0 * draw_count) / (win_count + loss_count)),
    ]
    iterations = 0
    converged = False
    for iteration in range(maximum_iterations):
        nll, gradient, hessian, _ = _statistics(observations, parameters, derivatives=True)
        gradient_norm = max(abs(value) for value in gradient)
        if gradient_norm <= gradient_tolerance:
            iterations = iteration
            converged = True
            break
        step_direction = _solve_positive_definite(hessian, gradient)
        directional_derivative = sum(
            gradient[index] * step_direction[index] for index in range(3)
        )
        _require(
            directional_derivative > 0.0 and math.isfinite(directional_derivative),
            "Davidson Newton direction is not descending",
        )
        accepted = False
        step = 1.0
        for _ in range(48):
            candidate = [
                parameters[index] - step * step_direction[index] for index in range(3)
            ]
            if candidate[0] > 1.0e-12 and all(math.isfinite(value) for value in candidate):
                candidate_nll, _, _, _ = _statistics(
                    observations, candidate, derivatives=False
                )
                if candidate_nll <= nll - 1.0e-4 * step * directional_derivative:
                    parameters = candidate
                    accepted = True
                    break
            step *= 0.5
        _require(accepted, "Davidson line search failed; finite optimum is unproven")
        _require(max(abs(value) for value in parameters) < 1.0e6, "Davidson fit is separating")
        iterations = iteration + 1

    final_nll, final_gradient, final_hessian, final_brier = _statistics(
        observations, parameters, derivatives=True
    )
    final_gradient_norm = max(abs(value) for value in final_gradient)
    if final_gradient_norm <= gradient_tolerance:
        converged = True
    _require(converged, "Davidson optimizer did not converge")
    _require(parameters[0] > 0.0, "Davidson slope A is not positive")
    eigenvalues = _symmetric_eigenvalues(final_hessian)
    _require(
        eigenvalues[0] > max(1.0e-15, eigenvalues[-1] / condition_limit),
        "Davidson Hessian is not full-rank positive definite",
    )
    condition = eigenvalues[-1] / eigenvalues[0]
    _require(condition <= condition_limit, "Davidson Hessian condition limit was exceeded")
    total_count = sum(class_counts)
    scores = [observation.score for observation in observations]
    return SideFit(
        a=parameters[0],
        b=parameters[1],
        d=parameters[2],
        observations=total_count,
        class_counts=tuple(class_counts),
        score_min=min(scores),
        score_max=max(scores),
        iterations=iterations,
        categorical_nll=final_nll,
        mean_brier=final_brier,
        mean_half_brier=0.5 * final_brier,
        gradient_inf_norm=final_gradient_norm,
        hessian_eigenvalues=eigenvalues,
        hessian_condition=condition,
    )


def _side_receipt(fit: SideFit) -> dict[str, object]:
    return {
        "parameters": {
            "A": _float_receipt(fit.a),
            "B": _float_receipt(fit.b),
            "D": _float_receipt(fit.d),
        },
        "support": {
            "records": fit.observations,
            "loss_draw_win": list(fit.class_counts),
            "score_min": fit.score_min,
            "score_max": fit.score_max,
        },
        "optimum": {
            "iterations": fit.iterations,
            "categorical_nll": fit.categorical_nll,
            "mean_brier": fit.mean_brier,
            "mean_half_brier": fit.mean_half_brier,
            "gradient_inf_norm": fit.gradient_inf_norm,
            "hessian_eigenvalues": list(fit.hessian_eigenvalues),
            "hessian_condition": fit.hessian_condition,
            "finite": True,
            "positive_slope": True,
            "full_rank_positive_definite_hessian": True,
            "separation_detected": False,
        },
    }


def build_artifact(
    aggregated: AggregatedLabels,
    source: dict[str, object],
    *,
    minimum_class_support: int = DEFAULT_MINIMUM_CLASS_SUPPORT,
    gradient_tolerance: float = DEFAULT_GRADIENT_TOLERANCE,
    maximum_iterations: int = DEFAULT_MAXIMUM_ITERATIONS,
    condition_limit: float = DEFAULT_CONDITION_LIMIT,
) -> dict[str, object]:
    _require(
        minimum_class_support >= DEFAULT_MINIMUM_CLASS_SUPPORT,
        "WDL artifact cannot weaken the minimum class-support gate",
    )
    _require(
        aggregated.total_records
        == aggregated.eligible_records
        + aggregated.mate_records_excluded
        + aggregated.child_records_excluded,
        "WDL aggregation record accounting is inconsistent",
    )
    _require(
        aggregated.child_records_excluded == sum(aggregated.child_counts),
        "WDL aggregation child accounting is inconsistent",
    )
    _require(
        aggregated.parents_only or aggregated.child_records_excluded == 0,
        "WDL aggregation excluded children without being restricted to parents",
    )
    _require(
        aggregated.eligible_records == sum(sum(counts) for counts in aggregated.class_counts),
        "WDL aggregation class accounting is inconsistent",
    )
    _require(
        aggregated.mate_records_excluded == sum(aggregated.mate_counts),
        "WDL aggregation mate accounting is inconsistent",
    )
    _require(
        all(
            tuple(
                sum(observation.count for observation in aggregated.by_side[side] if observation.result == result)
                for result in (-1, 0, 1)
            )
            == aggregated.class_counts[side]
            for side in (WHITE, BLACK)
        ),
        "WDL aggregation histogram contradicts class accounting",
    )
    _require(
        all(
            len(digest) == 64
            and all(character in "0123456789ABCDEF" for character in digest)
            for digest in (aggregated.selection_sha256, aggregated.eligible_sha256)
        ),
        "WDL aggregation digest is not uppercase SHA-256",
    )
    fits = tuple(
        fit_side(
            aggregated.by_side[side],
            minimum_class_support=minimum_class_support,
            gradient_tolerance=gradient_tolerance,
            maximum_iterations=maximum_iterations,
            condition_limit=condition_limit,
        )
        for side in (WHITE, BLACK)
    )
    return {
        "schema": SCHEMA,
        "link": {
            "schema": LINK_SCHEMA,
            "score_input": "post-rule50 Stockfish::Value",
            "score_scale": int(SCORE_SCALE),
            "logits_loss_draw_win": ["-(A * score / 600 + B)", "D", "A * score / 600 + B"],
            "perspective": "side_to_move",
            "side_specific": True,
        },
        "implementation": AGGREGATION_IMPLEMENTATION,
        "source": source,
        "selection": {
            "total_records": aggregated.total_records,
            "eligible_records": aggregated.eligible_records,
            "mate_records_excluded": aggregated.mate_records_excluded,
            "mate_threshold": MATE_SCORE_THRESHOLD,
            "mate_policy": "exclude from calibration and score-derived targets; retain result target",
            "selection_sha256": aggregated.selection_sha256,
            "eligible_records_sha256": aggregated.eligible_sha256,
            "class_counts_by_side": {
                SIDE_NAMES[side]: list(aggregated.class_counts[side])
                for side in (WHITE, BLACK)
            },
            "mate_counts_by_side": {
                SIDE_NAMES[side]: aggregated.mate_counts[side] for side in (WHITE, BLACK)
            },
            # Emitted only for a parents-only fit, so a plain corpus produces
            # the artifact it always produced, byte for byte.
            **(
                {
                    "parents_only": True,
                    "child_records_excluded": aggregated.child_records_excluded,
                    "child_counts_by_side": {
                        SIDE_NAMES[side]: aggregated.child_counts[side]
                        for side in (WHITE, BLACK)
                    },
                }
                if aggregated.parents_only
                else {}
            ),
        },
        "fit": {
            "objective": "unweighted mean categorical negative log likelihood",
            "class_weighting": "none",
            "resampling": "none",
            "side_pooling": "none",
            "algorithm": "full-batch binary64 damped Newton with deterministic Armijo backtracking",
            "minimum_class_support": minimum_class_support,
            "gradient_inf_norm_tolerance": gradient_tolerance,
            "maximum_iterations": maximum_iterations,
            "hessian_condition_limit": condition_limit,
            "sides": {
                SIDE_NAMES[side]: _side_receipt(fits[side]) for side in (WHITE, BLACK)
            },
        },
        "claims": {
            "training_split_only": True,
            "frozen_for_architecture_comparisons": True,
            "strength_evidence": False,
        },
    }


def validate_artifact(payload: object) -> dict[str, tuple[float, float, float]]:
    _require(isinstance(payload, dict), "WDL calibration root is not an object")
    base_fields = {"schema", "link", "source", "selection", "fit", "claims"}
    _require(
        set(payload) in (base_fields, base_fields | {"implementation"}),
        "WDL calibration top-level fields are incomplete",
    )
    _require(payload.get("schema") == SCHEMA, "WDL calibration schema mismatch")
    link = payload.get("link")
    _require(
        isinstance(link, dict)
        and set(link)
        == {
            "schema",
            "score_input",
            "score_scale",
            "logits_loss_draw_win",
            "perspective",
            "side_specific",
        },
        "WDL calibration link is incomplete",
    )
    _require(link.get("schema") == LINK_SCHEMA, "WDL calibration link schema mismatch")
    _require(link.get("score_scale") == int(SCORE_SCALE), "WDL score scale mismatch")
    _require(link.get("perspective") == "side_to_move", "WDL perspective mismatch")
    _require(link.get("side_specific") is True, "WDL side-specific contract is missing")
    _require(
        link.get("score_input") == "post-rule50 Stockfish::Value"
        and link.get("logits_loss_draw_win")
        == ["-(A * score / 600 + B)", "D", "A * score / 600 + B"],
        "WDL link formula mismatch",
    )

    source = payload.get("source")
    _require(
        isinstance(source, dict) and set(source) == {"training_file", "teacher", "software"},
        "WDL calibration source identity is incomplete",
    )
    training_file = source["training_file"]
    teacher = source["teacher"]
    software = source["software"]
    _require(
        isinstance(training_file, dict)
        and set(training_file)
        == {"name", "sha256", "payload_sha256", "manifest_sha256", "records"},
        "WDL training-file identity is incomplete",
    )
    _require(
        all(
            isinstance(training_file.get(field), str)
            and len(training_file[field]) == 64
            and all(character in "0123456789ABCDEF" for character in training_file[field])
            for field in ("sha256", "payload_sha256", "manifest_sha256")
        )
        and type(training_file.get("records")) is int
        and training_file["records"] > 0,
        "WDL training-file identity is invalid",
    )
    _require(
        isinstance(training_file.get("name"), str) and bool(training_file["name"]),
        "WDL training-file name is invalid",
    )
    _require(
        isinstance(teacher, dict)
        and set(teacher) == {"source_commit", "producer_sha256", "network", "label_contract"},
        "WDL teacher identity is incomplete",
    )
    _require(
        isinstance(teacher.get("source_commit"), str)
        and len(teacher["source_commit"]) == 40
        and all(character in "0123456789abcdefABCDEF" for character in teacher["source_commit"])
        and isinstance(teacher.get("producer_sha256"), str)
        and len(teacher["producer_sha256"]) == 64
        and all(character in "0123456789ABCDEF" for character in teacher["producer_sha256"]),
        "WDL teacher hashes are invalid",
    )
    _require(
        teacher.get("network")
        == {"schema": "HORDETEST_HP_LEGACY_V1", "sha256": wire.RUN6B_SHA256},
        "WDL teacher network is not registered Run 6B",
    )
    _require(
        teacher.get("label_contract")
        == {
            "schema": wire.LABEL_CONTRACT_NAME,
            "schema_sha256": wire.LABEL_CONTRACT_SHA256,
        },
        "WDL teacher label contract mismatch",
    )
    _require(
        isinstance(software, dict)
        and set(software) == {"commit", "dirty", "python", "implementation"}
        and software.get("dirty") is False,
        "WDL calibration software identity is dirty or incomplete",
    )
    _require(
        isinstance(software.get("commit"), str)
        and len(software["commit"]) == 40
        and all(character in "0123456789abcdefABCDEF" for character in software["commit"])
        and isinstance(software.get("python"), str)
        and bool(software["python"])
        and isinstance(software.get("implementation"), str)
        and bool(software["implementation"]),
        "WDL calibration software identity is invalid",
    )

    selection = payload.get("selection")
    _require(isinstance(selection, dict), "WDL calibration selection is missing")
    base_fields = {
        "total_records",
        "eligible_records",
        "mate_records_excluded",
        "mate_threshold",
        "mate_policy",
        "selection_sha256",
        "eligible_records_sha256",
        "class_counts_by_side",
        "mate_counts_by_side",
    }
    # The parents-only fields arrived with HORDE_BIN_V1_R2. An artifact fitted
    # before the revision existed carries neither them nor any children, and it
    # stays valid: the frozen calibration is the comparability currency of the
    # V2 era and nothing here may retire it. The three fields travel together,
    # so a partially rewritten artifact still fails closed.
    expansion_fields = {
        "parents_only",
        "child_records_excluded",
        "child_counts_by_side",
    }
    observed = set(selection)
    _require(
        observed == base_fields or observed == base_fields | expansion_fields,
        "WDL calibration selection fields are incomplete",
    )
    if not (observed & expansion_fields):
        selection = {
            **selection,
            "parents_only": False,
            "child_records_excluded": 0,
            "child_counts_by_side": {name: 0 for name in SIDE_NAMES.values()},
        }
    _require(selection.get("mate_threshold") == MATE_SCORE_THRESHOLD, "WDL mate threshold mismatch")
    _require(
        selection.get("mate_policy")
        == "exclude from calibration and score-derived targets; retain result target",
        "WDL mate policy mismatch",
    )
    _require(
        type(selection.get("total_records")) is int
        and type(selection.get("eligible_records")) is int
        and type(selection.get("mate_records_excluded")) is int
        and type(selection.get("child_records_excluded")) is int
        and selection["total_records"]
        == selection["eligible_records"]
        + selection["mate_records_excluded"]
        + selection["child_records_excluded"]
        == training_file["records"],
        "WDL selection record accounting is inconsistent",
    )
    _require(
        type(selection.get("parents_only")) is bool
        and (selection["parents_only"] or selection["child_records_excluded"] == 0),
        "WDL selection excluded children without being restricted to parents",
    )
    child_counts = selection.get("child_counts_by_side")
    _require(
        isinstance(child_counts, dict)
        and set(child_counts) == set(SIDE_NAMES.values())
        and all(type(count) is int and count >= 0 for count in child_counts.values())
        and sum(child_counts.values()) == selection["child_records_excluded"],
        "WDL child-count accounting is inconsistent",
    )
    _require(
        all(
            isinstance(selection.get(field), str)
            and len(selection[field]) == 64
            and all(character in "0123456789ABCDEF" for character in selection[field])
            for field in ("selection_sha256", "eligible_records_sha256")
        ),
        "WDL selection digest is invalid",
    )
    mate_counts = selection.get("mate_counts_by_side")
    _require(
        isinstance(mate_counts, dict)
        and set(mate_counts) == set(SIDE_NAMES.values())
        and all(type(count) is int and count >= 0 for count in mate_counts.values())
        and sum(mate_counts.values()) == selection["mate_records_excluded"],
        "WDL mate-count accounting is inconsistent",
    )
    fit = payload.get("fit")
    _require(
        isinstance(fit, dict)
        and set(fit)
        == {
            "objective",
            "class_weighting",
            "resampling",
            "side_pooling",
            "algorithm",
            "minimum_class_support",
            "gradient_inf_norm_tolerance",
            "maximum_iterations",
            "hessian_condition_limit",
            "sides",
        }
        and fit.get("objective") == "unweighted mean categorical negative log likelihood"
        and fit.get("class_weighting") == "none",
        "WDL fit contract mismatch",
    )
    _require(
        fit.get("resampling") == "none" and fit.get("side_pooling") == "none",
        "WDL fit is weighted or pooled",
    )
    minimum_support = fit.get("minimum_class_support")
    _require(type(minimum_support) is int and minimum_support >= 32, "WDL support gate is too weak")
    gradient_tolerance = fit.get("gradient_inf_norm_tolerance")
    maximum_iterations = fit.get("maximum_iterations")
    condition_limit = fit.get("hessian_condition_limit")
    _require(
        isinstance(gradient_tolerance, float)
        and 0.0 < gradient_tolerance <= DEFAULT_GRADIENT_TOLERANCE
        and type(maximum_iterations) is int
        and maximum_iterations > 0
        and isinstance(condition_limit, float)
        and 1.0 < condition_limit <= DEFAULT_CONDITION_LIMIT,
        "WDL optimizer gates are invalid",
    )
    _require(
        fit.get("algorithm")
        == "full-batch binary64 damped Newton with deterministic Armijo backtracking",
        "WDL optimizer algorithm mismatch",
    )
    sides = fit.get("sides")
    _require(isinstance(sides, dict) and set(sides) == set(SIDE_NAMES.values()), "WDL sides are incomplete")
    decoded: dict[str, tuple[float, float, float]] = {}
    for side_name in SIDE_NAMES.values():
        side = sides[side_name]
        _require(
            isinstance(side, dict) and set(side) == {"parameters", "support", "optimum"},
            f"{side_name} WDL fit is invalid",
        )
        parameters = side.get("parameters")
        support = side.get("support")
        optimum = side.get("optimum")
        _require(isinstance(parameters, dict) and set(parameters) == {"A", "B", "D"}, f"{side_name} parameters are incomplete")
        _require(isinstance(optimum, dict), f"{side_name} optimum receipt is missing")
        _require(
            isinstance(support, dict)
            and set(support) == {"records", "loss_draw_win", "score_min", "score_max"}
            and isinstance(support.get("loss_draw_win"), list)
            and len(support["loss_draw_win"]) == 3
            and all(type(count) is int and count >= minimum_support for count in support["loss_draw_win"])
            and support.get("records") == sum(support["loss_draw_win"]),
            f"{side_name} support receipt is invalid",
        )
        _require(
            type(support.get("score_min")) is int
            and type(support.get("score_max")) is int
            and -MATE_SCORE_THRESHOLD < support["score_min"] <= support["score_max"] < MATE_SCORE_THRESHOLD,
            f"{side_name} score support is invalid",
        )
        _require(
            set(optimum)
            == {
                "iterations",
                "categorical_nll",
                "mean_brier",
                "mean_half_brier",
                "gradient_inf_norm",
                "hessian_eigenvalues",
                "hessian_condition",
                "finite",
                "positive_slope",
                "full_rank_positive_definite_hessian",
                "separation_detected",
            },
            f"{side_name} optimum receipt is incomplete",
        )
        _require(
            optimum.get("finite") is True
            and optimum.get("positive_slope") is True
            and optimum.get("full_rank_positive_definite_hessian") is True
            and optimum.get("separation_detected") is False,
            f"{side_name} optimum did not pass calibration gates",
        )
        numeric_metrics = (
            optimum.get("categorical_nll"),
            optimum.get("mean_brier"),
            optimum.get("mean_half_brier"),
            optimum.get("gradient_inf_norm"),
            optimum.get("hessian_condition"),
        )
        eigenvalues = optimum.get("hessian_eigenvalues")
        _require(
            all(type(value) is float and math.isfinite(value) for value in numeric_metrics)
            and type(optimum.get("iterations")) is int
            and 0 <= optimum["iterations"] <= maximum_iterations
            and optimum["gradient_inf_norm"] <= gradient_tolerance
            and 0.0 < optimum["hessian_condition"] <= condition_limit
            and isinstance(eigenvalues, list)
            and len(eigenvalues) == 3
            and all(type(value) is float and math.isfinite(value) and value > 0.0 for value in eigenvalues)
            and eigenvalues == sorted(eigenvalues),
            f"{side_name} optimum metrics are invalid",
        )
        values = tuple(_float_from_receipt(parameters[name], f"{side_name}.{name}") for name in ("A", "B", "D"))
        _require(values[0] > 0.0, f"{side_name} Davidson slope is not positive")
        decoded[side_name] = values
    class_counts = selection.get("class_counts_by_side")
    _require(
        isinstance(class_counts, dict)
        and set(class_counts) == set(SIDE_NAMES.values())
        and all(
            class_counts[side_name] == sides[side_name]["support"]["loss_draw_win"]
            for side_name in SIDE_NAMES.values()
        ),
        "WDL selection and fit class counts differ",
    )
    _require(
        selection["eligible_records"]
        == sum(sum(class_counts[side_name]) for side_name in SIDE_NAMES.values()),
        "WDL eligible-record accounting differs from class support",
    )
    claims = payload.get("claims")
    _require(
        claims
        == {
            "training_split_only": True,
            "frozen_for_architecture_comparisons": True,
            "strength_evidence": False,
        },
        "WDL calibration claims mismatch",
    )
    return decoded


def load_artifact(path: Path) -> tuple[dict[str, object], dict[str, tuple[float, float, float]], str]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"WDL calibration file does not exist: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise CalibrationError(f"WDL calibration is invalid JSON: {error}") from error
    parameters = validate_artifact(payload)
    return payload, parameters, hashlib.sha256(raw).hexdigest().upper()


def canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")
