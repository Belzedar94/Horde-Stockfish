#!/usr/bin/env python3
"""Focused tests for the frozen Horde V2 C2 constant baseline."""

from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import horde_bin_v1 as wire  # noqa: E402
import horde_training_control as trainer  # noqa: E402
import horde_training_decoder as decoder  # noqa: E402
import horde_v2_c2_constant_baseline as baseline  # noqa: E402
import horde_v2_c2_objective as objective  # noqa: E402
import horde_wdl as wdl  # noqa: E402


PARAMETERS = {
    "white_to_move": (0.83, 0.17, -0.72),
    "black_to_move": (1.07, -0.13, -0.91),
}


def _wire_record(
    index: int,
    side: int,
    result: int,
    *,
    rule50: int,
    score: int,
) -> bytes:
    board = [0] * 64
    board[0] = 2
    board[57] = 7
    board[60] = 11
    packed_board = bytes(
        board[square] | (board[square + 1] << 4) for square in range(0, 64, 2)
    )
    move = (0 << 6) | 8 if side == decoder.WHITE else (57 << 6) | 42
    reason = 3 if result == 0 else 1
    return packed_board + bytes((side, 0, 64, 0)) + struct.pack(
        "<HHhHHbB", rule50, side, score, move, move, result, reason
    )


def _records(sides: tuple[int, ...] = (decoder.WHITE, decoder.BLACK)) -> list[bytes]:
    records: list[bytes] = []
    index = 0
    for side in sides:
        for result in (-1, 0, 1):
            for sample in range(48):
                noise = ((sample * 73 + (result + 1) * 19 + side * 11) % 601) - 300
                score = result * 240 + noise
                clock = (0, 37, 100)[sample % 3]
                records.append(
                    _wire_record(index, side, result, rule50=clock, score=score)
                )
                index += 1
        for sample, result in enumerate((-1, 1)):
            score = -32_000 if result < 0 else 32_000
            records.append(
                _wire_record(index, side, result, rule50=37 * sample, score=score)
            )
            index += 1
    return records


def _write_wire_dataset(path: Path, records: list[bytes]) -> None:
    payload = b"".join(records)
    record_count = len(records)
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
        "network": {"schema": "HORDETEST_HP_LEGACY_V1", "sha256": wire.RUN6B_SHA256},
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


def _write_wdl_artifact(training: Path, output: Path) -> None:
    with decoder.HordeBinV1Dataset(training) as dataset:
        aggregated = wdl.aggregate_labels(dataset)
        training_identity = {
            "name": training.name,
            "sha256": dataset.file_sha256,
            "payload_sha256": dataset.manifest["payload_sha256"],
            "manifest_sha256": dataset.manifest_sha256,
            "records": len(dataset),
        }
        teacher = {
            "source_commit": dataset.manifest["source_commit"],
            "producer_sha256": dataset.manifest["producer_sha256"],
            "network": dataset.manifest["network"],
            "label_contract": dataset.manifest["label_contract"],
        }
    artifact = wdl.build_artifact(
        aggregated,
        {
            "training_file": training_identity,
            "teacher": teacher,
            "software": {
                "commit": "6" * 40,
                "dirty": False,
                "python": "3.12.0",
                "implementation": "CPython",
            },
        },
    )
    output.write_bytes(wdl.canonical_json(artifact))


def test_float32_lookup_matches_trainer() -> None:
    lookup = objective.build_wdl_lookup(PARAMETERS)
    calibration = trainer._torch_calibration(PARAMETERS, torch.device("cpu"))
    scores = torch.arange(
        objective.LOOKUP_SCORE_MINIMUM,
        objective.LOOKUP_SCORE_MAXIMUM + 1,
        dtype=torch.float32,
    )
    for side in (decoder.WHITE, decoder.BLACK):
        sides = torch.full((len(scores),), side, dtype=torch.long)
        expected = trainer._wdl_probabilities(scores, sides, calibration).numpy()
        observed = np.asarray(
            [lookup.probabilities(side, int(score)) for score in scores.tolist()],
            dtype=np.float32,
        )
        if not np.array_equal(observed, expected):
            raise AssertionError("canonical WDL lookup differs from trainer float32 softmax")
    second = objective.build_wdl_lookup(PARAMETERS)
    if lookup.raw_float32_sha256 != second.raw_float32_sha256:
        raise AssertionError("canonical WDL lookup hash is not deterministic")


def test_binary64_mapping_is_not_the_contract() -> None:
    lookup = objective.build_wdl_lookup(PARAMETERS)
    differences = 0
    for side, name in objective.SIDE_NAMES.items():
        for score in (-31_506, -777, 0, 999, 31_506):
            float32 = lookup.probabilities(side, score)
            float64 = wdl.probabilities(score, PARAMETERS[name])
            differences += int(float32 != float64)
    if differences == 0:
        raise AssertionError("float64 WDL helper accidentally became the C2 lookup contract")


def test_rule50_exhaustive() -> None:
    for score in range(objective.LOOKUP_SCORE_MINIMUM, objective.LOOKUP_SCORE_MAXIMUM + 1):
        for clock in range(101):
            expected = math.trunc(score * (100 - clock) / 100)
            observed = objective.rule50_postprocess_constant(score, clock)
            if observed != expected:
                raise AssertionError(
                    f"rule-50 mismatch for score={score}, clock={clock}: {observed} != {expected}"
                )


def test_moments_and_recordwise_reference() -> None:
    lookup = objective.build_wdl_lookup(PARAMETERS)
    with tempfile.TemporaryDirectory(prefix="horde-c2-objective-") as temporary:
        training = Path(temporary) / "train.bin"
        _write_wire_dataset(training, _records())
        with decoder.HordeBinV1Dataset(training) as dataset:
            aggregated = objective.aggregate_objective(dataset, lookup)
    for side in (decoder.WHITE, decoder.BLACK):
        for constant in (-900, 0, 700):
            moments = objective.evaluate_side_constant(aggregated, lookup, side, constant)
            reference = objective.evaluate_side_constant(
                aggregated, lookup, side, constant, recordwise=True
            )
            if abs(moments[0] - reference[0]) > 2.0e-12:
                raise AssertionError(
                    f"moment/reference loss mismatch: {moments[0]} != {reference[0]}"
                )


def test_ties_boundary_and_mate_normalization() -> None:
    lookup = objective.build_wdl_lookup(PARAMETERS)
    prediction = lookup.probabilities(decoder.WHITE, 0)
    tie_group = objective.GroupMoments(
        side=decoder.WHITE,
        rule50_count=100,
        records=3,
        eligible_records=3,
        result_counts=(1, 1, 1),
        teacher_sum=tuple(3.0 * value for value in prediction),
        teacher_squared_norm_sum=3.0 * math.fsum(value * value for value in prediction),
        teacher_histogram=((0, 3),),
    )
    tie_aggregate = objective.AggregatedObjective(
        groups=(tie_group,),
        total_records=3,
        eligible_records=3,
        mate_records=0,
        records_by_side=(3, 0),
        eligible_by_side=(3, 0),
        mate_by_side=(0, 0),
        selection_sha256="1" * 64,
        grouped_histogram_sha256="2" * 64,
    )
    tie = objective.fit_side_constant(tie_aggregate, lookup, decoder.WHITE)
    if tie["selected_constant_cp"] != 0 or tie["minimizer_count"] != objective.LOOKUP_SCORE_COUNT:
        raise AssertionError("constant tie policy did not choose zero deterministically")

    endpoint_prediction = lookup.probabilities(decoder.WHITE, objective.LOOKUP_SCORE_MAXIMUM)
    endpoint_group = objective.GroupMoments(
        side=decoder.WHITE,
        rule50_count=0,
        records=128,
        eligible_records=128,
        result_counts=(0, 0, 128),
        teacher_sum=tuple(128.0 * value for value in endpoint_prediction),
        teacher_squared_norm_sum=128.0
        * math.fsum(value * value for value in endpoint_prediction),
        teacher_histogram=((objective.LOOKUP_SCORE_MAXIMUM, 128),),
    )
    endpoint_aggregate = objective.AggregatedObjective(
        groups=(endpoint_group,),
        total_records=128,
        eligible_records=128,
        mate_records=0,
        records_by_side=(128, 0),
        eligible_by_side=(128, 0),
        mate_by_side=(0, 0),
        selection_sha256="3" * 64,
        grouped_histogram_sha256="4" * 64,
    )
    endpoint = objective.fit_side_constant(endpoint_aggregate, lookup, decoder.WHITE)
    if endpoint["boundary_hit"] is not True:
        raise AssertionError("constant fit failed to report a boundary optimum")

    mate_group = objective.GroupMoments(
        side=decoder.WHITE,
        rule50_count=0,
        records=1,
        eligible_records=0,
        result_counts=(0, 0, 1),
        teacher_sum=(0.0, 0.0, 0.0),
        teacher_squared_norm_sum=0.0,
        teacher_histogram=(),
    )
    composite, score_sum, result_sum = objective.group_loss_from_moments(
        mate_group, prediction
    )
    if score_sum != 0.0 or composite != (1.0 - objective.LAMBDA) * result_sum:
        raise AssertionError("mate label was not excluded only from the score term")


def test_absent_side_is_rejected() -> None:
    lookup = objective.build_wdl_lookup(PARAMETERS)
    with tempfile.TemporaryDirectory(prefix="horde-c2-one-side-") as temporary:
        training = Path(temporary) / "train.bin"
        _write_wire_dataset(training, _records((decoder.WHITE,)))
        with decoder.HordeBinV1Dataset(training) as dataset:
            try:
                objective.aggregate_objective(dataset, lookup)
            except objective.C2ObjectiveError:
                pass
            else:
                raise AssertionError("constant objective accepted a dataset with one side absent")


def test_end_to_end_receipt_and_tamper_checks() -> None:
    with tempfile.TemporaryDirectory(prefix="horde-c2-baseline-") as temporary:
        root = Path(temporary)
        training = root / "train.bin"
        calibration = root / "wdl.json"
        _write_wire_dataset(training, _records())
        _write_wdl_artifact(training, calibration)
        first = baseline.build_receipt(training, calibration, allow_dirty=True)
        second = baseline.build_receipt(training, calibration, allow_dirty=True)
        if first != second:
            raise AssertionError("constant-baseline receipt is not deterministic")
        baseline.validate_receipt(first)
        if first["claims"]["validation_inspected"] is not False:
            raise AssertionError("training-only receipt claims validation access")
        if first["aggregation"]["mate_records"] != 4:
            raise AssertionError("constant-baseline mate accounting changed")
        for side in ("white_to_move", "black_to_move"):
            audit = first["fit"]["sides"][side]["recordwise_audit"]
            difference = objective.float_from_receipt(
                audit["absolute_loss_difference"], f"{side} audit difference"
            )
            if difference > 2.0e-12:
                raise AssertionError("selected constant failed its recordwise audit")

        tampered = copy.deepcopy(first)
        tampered["objective"]["lookup"]["raw_float32_sha256"] = "g" * 64
        try:
            baseline.validate_receipt(tampered)
        except baseline.ConstantBaselineError:
            pass
        else:
            raise AssertionError("constant receipt accepted a malformed lookup hash")

        mismatched = root / "mismatched.bin"
        _write_wire_dataset(mismatched, _records()[:-1])
        try:
            baseline.build_receipt(mismatched, calibration, allow_dirty=True)
        except baseline.ConstantBaselineError:
            pass
        else:
            raise AssertionError("constant baseline accepted the wrong training split")


def test_cli_has_no_validation_or_checkpoint_input() -> None:
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            baseline.parse_args(
                ["train.bin", "wdl.json", "--output", "receipt.json", "--validation", "v.bin"]
            )
        except SystemExit:
            pass
        else:
            raise AssertionError("constant-baseline CLI accepted validation data")
        try:
            baseline.parse_args(
                ["train.bin", "wdl.json", "--output", "receipt.json", "--checkpoint", "x.pt"]
            )
        except SystemExit:
            pass
        else:
            raise AssertionError("constant-baseline CLI accepted a checkpoint")


def main() -> int:
    test_float32_lookup_matches_trainer()
    test_binary64_mapping_is_not_the_contract()
    test_rule50_exhaustive()
    test_moments_and_recordwise_reference()
    test_ties_boundary_and_mate_normalization()
    test_absent_side_is_rejected()
    test_end_to_end_receipt_and_tamper_checks()
    test_cli_has_no_validation_or_checkpoint_input()
    print("Horde V2 C2 constant-baseline tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
