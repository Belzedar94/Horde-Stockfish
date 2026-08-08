#!/usr/bin/env python3
"""Focused tests for the fresh Horde legacy-control reference trainer."""

from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import horde_training_control as control  # noqa: E402
import horde_training_decoder as decoder  # noqa: E402
import horde_training_microfit as microfit  # noqa: E402
import horde_wdl as wdl  # noqa: E402


def calibration(device: str = "cpu") -> control.DavidsonCalibration:
    return control._torch_calibration(
        {
            "white_to_move": (1.0, 0.0, 0.0),
            "black_to_move": (1.0, 0.0, 0.0),
        },
        torch.device(device),
    )


def test_schedule() -> None:
    first = list(control.epoch_batches(103, 13, 31, 0x12345678, 0))
    repeat = list(control.epoch_batches(103, 13, 31, 0x12345678, 0))
    second_epoch = list(control.epoch_batches(103, 13, 31, 0x12345678, 1))
    if first != repeat:
        raise AssertionError("fixed-seed block shuffle is not deterministic")
    if first == second_epoch:
        raise AssertionError("block shuffle did not change between epochs")
    flattened = [index for batch in first for index in batch]
    if sorted(flattened) != list(range(103)):
        raise AssertionError("block shuffle is not a complete permutation")
    if any(not 1 <= len(batch) <= 13 for batch in first):
        raise AssertionError("block shuffle emitted an invalid batch size")
    if control._schedule_sha256(first) != control._schedule_sha256(repeat):
        raise AssertionError("schedule hash changed for identical batches")

    for record_count in range(1, 97):
        for batch_size, block_size in ((1, 1), (7, 9), (13, 31), (32, 32)):
            batches = list(
                control.epoch_batches(
                    record_count,
                    batch_size,
                    max(batch_size, block_size),
                    0xA5A5A5A5,
                    record_count % 5,
                )
            )
            observed = [index for batch in batches for index in batch]
            if sorted(observed) != list(range(record_count)):
                raise AssertionError(
                    f"block shuffle lost a record at n={record_count}, "
                    f"batch={batch_size}, block={block_size}"
                )


def test_mate_mask() -> None:
    batch = control.LegacyBatch(
        legacy_white=torch.empty(0, dtype=torch.long),
        legacy_black=torch.empty(0, dtype=torch.long),
        piece_offsets=torch.zeros(4, dtype=torch.long),
        side_to_move=torch.tensor([0, 1, 0]),
        piece_buckets=torch.zeros(3, dtype=torch.long),
        rule50_count=torch.zeros(3, dtype=torch.long),
        scores=torch.tensor([0.0, 31507.0, -31999.0]),
        results=torch.tensor([0, 1, -1]),
        score_eligible=torch.tensor([True, False, False]),
    )
    output = torch.tensor([0.0, 0.5, -0.5])
    composite, score_error, result_error, prediction = control.loss_terms(
        output, batch, 0.6, calibration()
    )
    if not torch.equal(batch.score_eligible, torch.tensor([True, False, False])):
        raise AssertionError("mate eligibility threshold changed")
    if not torch.allclose(composite[1:], 0.4 * result_error[1:], atol=1.0e-7, rtol=0.0):
        raise AssertionError(f"mate score-derived WDL term was not masked: {composite}")
    if not bool(torch.all(score_error[1:] > 0.0)):
        raise AssertionError("mate fixture does not exercise a non-zero masked score error")
    if not torch.allclose(prediction[0], torch.full((3,), 1.0 / 3.0), atol=1.0e-7, rtol=0.0):
        raise AssertionError("zero-score Davidson prediction changed")
    if not torch.allclose(composite[0], torch.tensor(0.4 / 3.0), atol=1.0e-7, rtol=0.0):
        raise AssertionError("result half-Brier normalization changed")


def test_gradient_path() -> None:
    if sum(code != 0 for code in microfit._fixture_board(52, 0)) != 52:
        raise AssertionError("maximum-capacity Horde fixture changed")
    fixture, _ = microfit.make_fixture_batch()
    batch = control.LegacyBatch(
        legacy_white=fixture.legacy_white,
        legacy_black=fixture.legacy_black,
        piece_offsets=fixture.piece_offsets,
        side_to_move=fixture.side_to_move,
        piece_buckets=fixture.piece_buckets,
        rule50_count=torch.zeros_like(fixture.side_to_move),
        scores=fixture.scores,
        results=torch.round(2.0 * fixture.result_targets - 1.0).to(dtype=torch.long),
        score_eligible=torch.ones_like(fixture.scores, dtype=torch.bool),
    )
    model = control.LegacyHPModel(0xC0FFEE)
    before = control._state_sha256(model)
    optimizer = control._make_optimizer(model, control.DEFAULT_LEARNING_RATE)
    composite, *_ = control.loss_terms(model(batch), batch, 0.6, calibration())
    composite.mean().backward()
    norms = control._gradient_norms(model)
    if set(norms) != {"feature_transformer", "psqt", "dense_trunk", "output"}:
        raise AssertionError(f"gradient domains changed: {norms}")
    optimizer.step()
    control._clip_serialized_dense_weights(model)
    after = control._state_sha256(model)
    if before == after:
        raise AssertionError("optimizer step did not change the model")
    if not control._all_finite(model):
        raise AssertionError("optimizer step produced non-finite parameters")

    if torch.cuda.is_available():
        cuda_model = control.LegacyHPModel(0xC0FFEE).to("cuda")
        cuda_batch = control.LegacyBatch(
            legacy_white=batch.legacy_white.to("cuda"),
            legacy_black=batch.legacy_black.to("cuda"),
            piece_offsets=batch.piece_offsets.to("cuda"),
            side_to_move=batch.side_to_move.to("cuda"),
            piece_buckets=batch.piece_buckets.to("cuda"),
            rule50_count=batch.rule50_count.to("cuda"),
            scores=batch.scores.to("cuda"),
            results=batch.results.to("cuda"),
            score_eligible=batch.score_eligible.to("cuda"),
        )
        with torch.no_grad():
            output = cuda_model(cuda_batch)
        if output.device.type != "cuda" or not bool(torch.isfinite(output).all().cpu()):
            raise AssertionError("legacy control forward is not CUDA-safe")


def test_wdl_label_path() -> None:
    batch = control.LegacyBatch(
        legacy_white=torch.empty(0, dtype=torch.long),
        legacy_black=torch.empty(0, dtype=torch.long),
        piece_offsets=torch.zeros(3, dtype=torch.long),
        side_to_move=torch.tensor([0, 0]),
        piece_buckets=torch.zeros(2, dtype=torch.long),
        rule50_count=torch.tensor([0, 100]),
        scores=torch.tensor([600.0, 600.0]),
        results=torch.tensor([1, 1]),
        score_eligible=torch.tensor([True, True]),
    )
    output = torch.ones(2)
    _, score_error, _, prediction = control.loss_terms(output, batch, 0.6, calibration())
    if score_error[0] != 0.0 or score_error[1] <= 0.0:
        raise AssertionError("stored teacher score was incorrectly reprocessed by rule 50")
    if torch.equal(prediction[0], prediction[1]):
        raise AssertionError("prediction rule-50 postprocessor was bypassed")

    asymmetric = control._torch_calibration(
        {
            "white_to_move": (1.0, 0.5, -0.5),
            "black_to_move": (1.0, -0.5, -0.5),
        },
        torch.device("cpu"),
    )
    scores = torch.zeros(2)
    sides = torch.tensor([0, 1])
    side_predictions = control._wdl_probabilities(scores, sides, asymmetric)
    if side_predictions[0, 2] <= side_predictions[1, 2]:
        raise AssertionError("side-specific Davidson intercepts were pooled")


def test_wdl_tensor_link() -> None:
    parameters = {
        "white_to_move": (0.75, -0.2, -0.8),
        "black_to_move": (1.25, 0.3, -0.4),
    }
    scores = torch.tensor([-1200.0, 0.0, 600.0, 1800.0])
    sides = torch.tensor([0, 1, 0, 1])
    observed = control._wdl_probabilities(
        scores,
        sides,
        control._torch_calibration(parameters, torch.device("cpu")),
    )
    expected = torch.tensor(
        [
            wdl.probabilities(float(score), parameters[wdl.SIDE_NAMES[int(side)]])
            for score, side in zip(scores, sides, strict=True)
        ]
    )
    if not torch.allclose(observed, expected, atol=1.0e-7, rtol=0.0):
        raise AssertionError("trainer Davidson tensor link differs from the frozen scalar contract")


def test_named_initialization() -> None:
    narrow_royal = microfit.HordeV2Model(64, 192, 0xC0FFEE)
    wide_royal = microfit.HordeV2Model(128, 128, 0xC0FFEE)
    for name in (
        "hidden0_weights",
        "hidden0_bias",
        "hidden1_weights",
        "hidden1_bias",
        "output_weights",
        "output_bias",
    ):
        if not torch.equal(getattr(narrow_royal, name), getattr(wide_royal, name)):
            raise AssertionError(f"named initialization changed common V2 parameter {name}")


def test_rule50_postprocessor() -> None:
    output = torch.tensor(
        [
            100.9 / 600.0,
            -100.9 / 600.0,
            101.9 / 600.0,
            -101.9 / 600.0,
            100.9 / 600.0,
        ],
        requires_grad=True,
    )
    rule50 = torch.tensor([50, 50, 33, 33, 100])
    observed = control._rule50_postprocess(output, rule50)
    expected = torch.tensor([50.0, -50.0, 67.0, -67.0, 0.0])
    if not torch.equal(observed, expected):
        raise AssertionError(f"rule-50 integer forward changed: {observed}")
    observed.sum().backward()
    expected_gradient = torch.tensor([300.0, 300.0, 402.0, 402.0, 0.0])
    if not torch.allclose(output.grad, expected_gradient, rtol=0.0, atol=1.0e-4):
        raise AssertionError(f"rule-50 STE gradient changed: {output.grad}")


def test_position_and_model_keys() -> None:
    board = [0] * 64
    board[8] = decoder.WHITE_PAWN
    board[56] = decoder.BLACK_ROOK
    board[60] = decoder.BLACK_KING
    board[63] = decoder.BLACK_ROOK
    physical_board = tuple(board)
    features = decoder.extract_sparse_features(physical_board)
    common = dict(
        index=7,
        features=features,
        side_to_move=decoder.WHITE,
        rule50_count=23,
        game_ply=90,
        score=10,
        best_move=1,
        played_move=1,
        result=0,
        outcome_reason=3,
        board=physical_board,
        ep_square=64,
    )
    kingside = decoder.TrainingRecord(**common, castling_rights=1)
    queenside = decoder.TrainingRecord(**common, castling_rights=2)
    if decoder.physical_position_key(kingside) == decoder.physical_position_key(queenside):
        raise AssertionError("physical key discarded castling rights")
    if decoder.legacy_model_input_key(kingside) != decoder.legacy_model_input_key(queenside):
        raise AssertionError("legacy input key incorrectly includes invisible castling rights")
    changed_rule50 = decoder.TrainingRecord(**{**common, "rule50_count": 24}, castling_rights=1)
    if decoder.physical_position_key(kingside) != decoder.physical_position_key(changed_rule50):
        raise AssertionError("physical key incorrectly includes clock labels")
    if decoder.legacy_model_input_key(kingside) == decoder.legacy_model_input_key(changed_rule50):
        raise AssertionError("legacy evaluator-input key discarded rule50")


def main() -> int:
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    test_schedule()
    test_mate_mask()
    test_gradient_path()
    test_wdl_label_path()
    test_wdl_tensor_link()
    test_named_initialization()
    test_rule50_postprocessor()
    test_position_and_model_keys()
    print("Horde fresh legacy-control trainer tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
