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
        legacy_piece_offsets=torch.zeros(4, dtype=torch.long),
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
        legacy_piece_offsets=fixture.legacy_piece_offsets,
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
            legacy_piece_offsets=batch.legacy_piece_offsets.to("cuda"),
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


def _parameter_ids(group: dict[str, object]) -> set[int]:
    return {id(parameter) for parameter in group["params"]}


def test_optimizer_learning_rate_arms() -> None:
    learning_rate = control.DEFAULT_LEARNING_RATE
    baseline_model = control._make_model("v2-c1-abs64x192", 0xC0FFEE)
    reference_model = control._make_model("v2-c1-abs64x192", 0xC0FFEE)
    reference_output = [reference_model.output_weights, reference_model.output_bias]
    reference_output_ids = {id(parameter) for parameter in reference_output}
    reference_body = [
        parameter
        for parameter in reference_model.parameters()
        if id(parameter) not in reference_output_ids
    ]
    reference = torch.optim.RAdam(
        [
            {"params": reference_body, "lr": learning_rate},
            {"params": reference_output, "lr": learning_rate / 10.0},
        ],
        betas=(0.9, 0.999),
        eps=1.0e-7,
        weight_decay=0.0,
        foreach=False,
    )
    baseline = control._make_optimizer(baseline_model, learning_rate)
    if baseline.state_dict() != reference.state_dict():
        raise AssertionError("baseline optimizer state changed before its first step")
    for (baseline_name, baseline_parameter), (reference_name, reference_parameter) in zip(
        baseline_model.named_parameters(),
        reference_model.named_parameters(),
        strict=True,
    ):
        if baseline_name != reference_name:
            raise AssertionError("baseline optimizer parameter order changed")
        gradient = torch.full_like(baseline_parameter, 0.125)
        baseline_parameter.grad = gradient
        reference_parameter.grad = gradient.clone()
    baseline.step()
    reference.step()
    if control._state_sha256(baseline_model) != control._state_sha256(reference_model):
        raise AssertionError("baseline optimizer step is not bit-identical")

    model = control._make_model("v2-c1-abs64x192", 0xC0FFEE)
    output_ids = {id(model.output_weights), id(model.output_bias)}
    dense_ids = {
        id(getattr(model, name))
        for name in (
            "hidden0_weights",
            "hidden0_bias",
            "hidden1_weights",
            "hidden1_bias",
        )
    }

    baseline = control._make_optimizer(model, learning_rate)
    if len(baseline.param_groups) != 2:
        raise AssertionError("baseline optimizer group count changed")
    if [group["lr"] for group in baseline.param_groups] != [
        learning_rate,
        learning_rate * control.DEFAULT_OUTPUT_LEARNING_RATE_MULTIPLIER,
    ]:
        raise AssertionError("baseline optimizer learning rates changed")
    if _parameter_ids(baseline.param_groups[1]) != output_ids:
        raise AssertionError("baseline output parameter group changed")
    if not dense_ids.issubset(_parameter_ids(baseline.param_groups[0])):
        raise AssertionError("baseline dense parameters left the body group")

    dense_arm = control._make_optimizer(
        model,
        learning_rate,
        dense_learning_rate_multiplier=0.1,
    )
    if len(dense_arm.param_groups) != 3:
        raise AssertionError("dense-rate arm did not isolate the dense trunk")
    if [group["lr"] for group in dense_arm.param_groups] != [
        learning_rate,
        learning_rate * 0.1,
        learning_rate * control.DEFAULT_OUTPUT_LEARNING_RATE_MULTIPLIER,
    ]:
        raise AssertionError("dense-rate arm changed another learning rate")
    if _parameter_ids(dense_arm.param_groups[1]) != dense_ids:
        raise AssertionError("dense-rate arm parameter group is incomplete")
    if _parameter_ids(dense_arm.param_groups[2]) != output_ids:
        raise AssertionError("dense-rate arm changed the output parameter group")

    output_arm = control._make_optimizer(
        model,
        learning_rate,
        output_learning_rate_multiplier=1.0,
    )
    if len(output_arm.param_groups) != 2:
        raise AssertionError("output-rate arm changed the optimizer group topology")
    if [group["lr"] for group in output_arm.param_groups] != [
        learning_rate,
        learning_rate,
    ]:
        raise AssertionError("output-rate arm changed another learning rate")
    if _parameter_ids(output_arm.param_groups[1]) != output_ids:
        raise AssertionError("output-rate arm changed the output parameter group")

    try:
        control._make_optimizer(
            model,
            learning_rate,
            dense_learning_rate_multiplier=0.1,
            output_learning_rate_multiplier=1.0,
        )
    except control.TrainingError:
        pass
    else:
        raise AssertionError("combined optimizer-factor experiment was accepted")

    legacy_model = control._make_model(control.LEGACY_ARCHITECTURE, 0xC0FFEE)
    legacy_dense_ids = {
        id(getattr(legacy_model, name))
        for name in (
            "hidden0_weights",
            "hidden0_bias",
            "hidden1_weights",
            "hidden1_bias",
        )
    }
    legacy_output_ids = {
        id(legacy_model.output_weights),
        id(legacy_model.output_bias),
    }
    legacy_dense_arm = control._make_optimizer(
        legacy_model,
        learning_rate,
        dense_learning_rate_multiplier=0.1,
    )
    if [group["lr"] for group in legacy_dense_arm.param_groups] != [
        learning_rate,
        learning_rate * 0.1,
        learning_rate * control.DEFAULT_OUTPUT_LEARNING_RATE_MULTIPLIER,
    ]:
        raise AssertionError("legacy dense-rate control changed another learning rate")
    if _parameter_ids(legacy_dense_arm.param_groups[1]) != legacy_dense_ids:
        raise AssertionError("legacy dense-rate control did not isolate the dense trunk")
    if _parameter_ids(legacy_dense_arm.param_groups[2]) != legacy_output_ids:
        raise AssertionError("legacy dense-rate control changed the output parameter group")


def test_v2_gradient_path() -> None:
    fixture, _ = microfit.make_fixture_batch()
    batch = control.V2Batch(
        v2_global=fixture.v2_global,
        v2_royal=fixture.v2_royal,
        global_offsets=fixture.global_offsets,
        royal_offsets=fixture.royal_offsets,
        side_to_move=fixture.side_to_move,
        rule50_count=torch.zeros_like(fixture.side_to_move),
        scores=fixture.scores,
        results=torch.round(2.0 * fixture.result_targets - 1.0).to(dtype=torch.long),
        score_eligible=torch.ones_like(fixture.scores, dtype=torch.bool),
    )
    cuda_batch = None
    if torch.cuda.is_available():
        cuda_batch = control.V2Batch(
            **{
                field: getattr(batch, field).to("cuda")
                for field in control.V2Batch.__dataclass_fields__
            }
        )
    expected_architectures = {
        "v2-64x192": (
            "royal",
            64,
            192,
            2_902_344,
            "royal_transformer",
        ),
        "v2-128x128": (
            "royal",
            128,
            128,
            5_433_672,
            "royal_transformer",
        ),
        "v2-c1-abs64x192": (
            "absolute_nonking",
            64,
            192,
            362_824,
            "absolute_nonking_transformer",
        ),
        "v2-c1-rank8-64x192": (
            "royal_rank8",
            64,
            192,
            936_264,
            "royal_rank8_transformer",
        ),
    }
    common_initial_states: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}

    for architecture, (
        first_domain,
        first_lanes,
        global_lanes,
        serialized_bytes,
        first_gradient_group,
    ) in expected_architectures.items():
        structure = control._v2_structure(architecture)
        if structure["structural_sha256"] != control._sha256_bytes(
            control._compact_json(
                {
                    key: value
                    for key, value in structure.items()
                    if key != "structural_sha256"
                }
            )
        ):
            raise AssertionError(f"{architecture} structural hash is not self-consistent")
        domains = structure["domains"]
        if (
            domains[0]["name"] != first_domain
            or domains[0]["lanes"] != first_lanes
            or domains[1]["lanes"] != global_lanes
            or structure["serialized_parameter_bytes"] != serialized_bytes
        ):
            raise AssertionError(f"{architecture} width receipt changed: {structure}")

        model = control._make_model(architecture, 0xC0FFEE)
        if first_domain == "absolute_nonking":
            absolute_indices, absolute_offsets = model.absolute_nonking_features(batch)
            expected_indices = batch.v2_global[
                batch.v2_global < control.V2_ABSOLUTE_NONKING_DIMENSIONS
            ]
            if not torch.equal(absolute_indices, expected_indices):
                raise AssertionError("C1 absolute domain retained a non-physical G0 row")
            if not torch.equal(absolute_offsets, batch.royal_offsets):
                raise AssertionError("C1 absolute domain did not omit exactly the Black king")
        elif first_domain == "royal_rank8":
            rank8_indices, rank8_offsets = model.royal_rank8_features(batch)
            rows_per_bucket = 10 * 64
            expected_indices = (
                torch.div(batch.v2_royal, rows_per_bucket * 4, rounding_mode="floor")
                * rows_per_bucket
                + torch.remainder(batch.v2_royal, rows_per_bucket)
            )
            if not torch.equal(rank8_indices, expected_indices):
                raise AssertionError("C1 R8 projection changed its rank-bucket map")
            if not torch.equal(rank8_offsets, batch.royal_offsets):
                raise AssertionError("C1 R8 projection changed sparse record boundaries")
        before = control._state_sha256(model)
        for name in (
            "global_weights",
            "global_bias",
            "hidden0_weights",
            "hidden0_bias",
            "hidden1_weights",
            "hidden1_bias",
            "output_weights",
            "output_bias",
        ):
            value = getattr(model, name).detach().clone()
            key = (name, tuple(value.shape))
            if key in common_initial_states and not torch.equal(
                common_initial_states[key], value
            ):
                raise AssertionError(f"shared V2 initialization changed for {name}")
            common_initial_states.setdefault(key, value)

        optimizer = control._make_optimizer(model, control.DEFAULT_LEARNING_RATE)
        composite, *_ = control.loss_terms(model(batch), batch, 0.6, calibration())
        composite.mean().backward()
        norms = control._gradient_norms(model)
        expected_groups = {
            first_gradient_group,
            "global_transformer",
            "dense_trunk",
            "output",
        }
        if set(norms) != expected_groups:
            raise AssertionError(f"{architecture} gradient domains changed: {norms}")
        optimizer.step()
        control._clip_serialized_dense_weights(model)
        if control._state_sha256(model) == before:
            raise AssertionError(f"{architecture} optimizer step did not change the model")
        if not control._all_finite(model):
            raise AssertionError(f"{architecture} produced non-finite parameters")
        if cuda_batch is not None:
            cuda_model = control._make_model(architecture, 0xC0FFEE).to("cuda")
            with torch.no_grad():
                cuda_output = cuda_model(cuda_batch)
            if cuda_output.device.type != "cuda" or not bool(
                torch.isfinite(cuda_output).all().cpu()
            ):
                raise AssertionError(f"{architecture} forward is not CUDA-safe")


def test_wdl_label_path() -> None:
    batch = control.LegacyBatch(
        legacy_white=torch.empty(0, dtype=torch.long),
        legacy_black=torch.empty(0, dtype=torch.long),
        legacy_piece_offsets=torch.zeros(3, dtype=torch.long),
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

    contextual_features = decoder.SparseFeatures(
        legacy_white=features.legacy_white,
        legacy_black=features.legacy_black,
        v2_global=features.v2_global + (decoder.V2_GLOBAL_DIMENSIONS,),
        v2_royal=features.v2_royal,
        royal_bucket=features.royal_bucket,
        royal_mirror=features.royal_mirror,
    )
    contextual_record = decoder.TrainingRecord(
        **{**common, "features": contextual_features},
        castling_rights=1,
    )
    sparse = decoder.make_sparse_batch((contextual_record,))
    physical_count = sum(code != 0 for code in physical_board)
    if sparse.legacy_piece_offsets != (0, physical_count):
        raise AssertionError("legacy offsets were coupled to contextual Global rows")
    if sparse.global_offsets != (0, physical_count + 1):
        raise AssertionError("Global offsets did not include the contextual row")
    if sparse.physical_piece_count != (physical_count,) or sparse.white_piece_count != (1,):
        raise AssertionError("physical counts were derived from the complete Global stream")


def main() -> int:
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    test_schedule()
    test_mate_mask()
    test_gradient_path()
    test_optimizer_learning_rate_arms()
    test_v2_gradient_path()
    test_wdl_label_path()
    test_wdl_tensor_link()
    test_named_initialization()
    test_rule50_postprocessor()
    test_position_and_model_keys()
    print("Horde fresh legacy-control trainer tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
