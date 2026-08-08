#!/usr/bin/env python3
"""Run deterministic CPU micro-fits for the Horde NNUE training topologies.

This is an engineering receipt, not a strength-training entry point.  It uses
one frozen synthetic-label corpus to prove that the Horde sparse decoder feeds
every transformer domain, dense trunk, output head, and legacy layer stack.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import platform
from pathlib import Path
import struct
import sys
from typing import Any, Iterable, Sequence

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as functional
except ImportError as error:  # pragma: no cover - exercised by the CLI failure path
    raise SystemExit("PyTorch is required for the Horde NNUE micro-fit receipt") from error

try:
    from .horde_training_decoder import (
        BLACK,
        LEGACY_DIMENSIONS,
        TrainingRecord,
        V2_GLOBAL_DIMENSIONS,
        V2_ROYAL_DIMENSIONS,
        WHITE,
        extract_sparse_features,
        make_sparse_batch,
    )
except ImportError:
    from horde_training_decoder import (
        BLACK,
        LEGACY_DIMENSIONS,
        TrainingRecord,
        V2_GLOBAL_DIMENSIONS,
        V2_ROYAL_DIMENSIONS,
        WHITE,
        extract_sparse_features,
        make_sparse_batch,
    )


SCHEMA = "HORDE_NNUE_MICROFIT_V1"
CORPUS_SCHEMA = "HORDE_NNUE_MICROFIT_CORPUS_V1"
CORPUS_SEED = 0x484F5244455F4D46
MODEL_SEED = 0x56325F4D4943524F
EXPECTED_CORPUS_SHA256 = "0B2844F0FCA016BFA2E0549B38B36C21167944CC68FCAB4EE2D92F2320A552C4"
SAMPLES_PER_BUCKET = 4
LEGACY_ACCUMULATOR_LANES = 512
LEGACY_BUCKETS = 8
HIDDEN0_LANES = 32
HIDDEN1_LANES = 32
LAMBDA = 0.6
NNUE_TO_SCORE = 600.0
EVAL_SIGMOID_SCALE = 410.0
OUTPUT_SIGMOID_SCALE = 361.0


@dataclass(frozen=True, slots=True)
class TorchBatch:
    legacy_white: Tensor
    legacy_black: Tensor
    piece_offsets: Tensor
    v2_global: Tensor
    v2_royal: Tensor
    royal_offsets: Tensor
    side_to_move: Tensor
    piece_buckets: Tensor
    scores: Tensor
    result_targets: Tensor


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return state, value ^ (value >> 31)


def _horde_start_pieces() -> tuple[tuple[int, int], ...]:
    pieces: list[tuple[int, int]] = []
    pieces.extend((square, 1) for square in range(32))
    pieces.extend((square, 1) for square in (33, 34, 37, 38))
    pieces.extend((square, 6) for square in range(48, 56))
    pieces.extend(
        (
            (56, 9),
            (57, 7),
            (58, 8),
            (59, 10),
            (61, 8),
            (62, 7),
            (63, 9),
        )
    )
    _require(len(pieces) == 51, "Horde fixture must expose 51 non-king start pieces")
    return tuple(pieces)


def _fixture_board(piece_count: int, variant: int) -> tuple[int, ...]:
    """Build a deterministic, physically valid board for one legacy bucket."""

    _require(2 <= piece_count <= 52, f"invalid fixture piece count {piece_count}")
    state = (CORPUS_SEED + piece_count * 0x10001 + variant) & ((1 << 64) - 1)
    pool = list(_horde_start_pieces())

    # A deterministic Fisher-Yates permutation gives each bucket different
    # sparse rows without depending on Python's random-module implementation.
    for index in range(len(pool) - 1, 0, -1):
        state, value = _splitmix64(state)
        selected = value % (index + 1)
        pool[index], pool[selected] = pool[selected], pool[index]

    king_squares = (60, 51, 42, 33, 24, 19, 10, 5)
    king_square = king_squares[(piece_count + variant) % len(king_squares)]
    pool = [entry for entry in pool if entry[0] != king_square]

    chosen = pool[: piece_count - 1]
    if not any(1 <= code <= 5 for _, code in chosen):
        replacement = next(entry for entry in pool[piece_count - 1 :] if entry[1] == 1)
        chosen[-1] = replacement

    board = [0] * 64
    board[king_square] = 11
    for square, code in chosen:
        board[square] = code

    # Promote a bounded subset of selected Horde pawns.  This exercises the
    # fixed HN/HB/HR/HQ roles without changing the physical piece count.
    white_pawns = [square for square, code in enumerate(board) if code == 1]
    for promotion_index, square in enumerate(white_pawns[: min(variant, 4)]):
        board[square] = 2 + promotion_index

    if variant & 1:
        mirrored = [0] * 64
        for square, code in enumerate(board):
            if code:
                mirrored[square ^ 7] = code
        board = mirrored

    _require(sum(code != 0 for code in board) == piece_count, "fixture count drift")
    _require(board.count(11) == 1, "fixture does not contain exactly one Royal king")
    _require(any(1 <= code <= 5 for code in board), "fixture contains no Horde piece")
    return tuple(board)


def _sample_score(board: Sequence[int], side_to_move: int, sample_index: int) -> int:
    white_pieces = sum(1 <= code <= 5 for code in board)
    black_non_king = sum(6 <= code <= 10 for code in board)
    white_pawn_progress = sum(square // 8 for square, code in enumerate(board) if code == 1)
    white_promoted = sum(2 <= code <= 5 for code in board)
    state, jitter = _splitmix64(CORPUS_SEED ^ sample_index)
    del state
    white_score = (
        18 * (white_pieces - 12)
        + 3 * white_pawn_progress
        + 28 * white_promoted
        - 22 * black_non_king
        + int(jitter % 121)
        - 60
    )
    score = white_score if side_to_move == WHITE else -white_score
    return max(-1200, min(1200, score))


def make_fixture_batch() -> tuple[TorchBatch, dict[str, object]]:
    records: list[TrainingRecord] = []
    # Counts land in the middle of each Run 6B material bucket.
    piece_counts = (4, 10, 16, 23, 29, 36, 42, 49)
    for bucket, piece_count in enumerate(piece_counts):
        expected_bucket = min((piece_count - 1) * LEGACY_BUCKETS // 52, LEGACY_BUCKETS - 1)
        _require(expected_bucket == bucket, "fixture piece count selected the wrong bucket")
        for variant in range(SAMPLES_PER_BUCKET):
            sample_index = len(records)
            board = _fixture_board(piece_count, variant)
            side_to_move = (bucket + variant) & 1
            score = _sample_score(board, side_to_move, sample_index)
            result = 1 if score >= 180 else -1 if score <= -180 else 0
            records.append(
                TrainingRecord(
                    index=sample_index,
                    features=extract_sparse_features(board),
                    side_to_move=side_to_move,
                    rule50_count=(sample_index * 7) % 101,
                    game_ply=sample_index * 3,
                    score=score,
                    best_move=0,
                    played_move=0,
                    result=result,
                    outcome_reason=0,
                )
            )

    sparse = make_sparse_batch(records)
    piece_counts_observed = [
        sparse.piece_offsets[index + 1] - sparse.piece_offsets[index]
        for index in range(len(sparse))
    ]
    piece_buckets = [
        min((count - 1) * LEGACY_BUCKETS // 52, LEGACY_BUCKETS - 1)
        for count in piece_counts_observed
    ]

    digest = hashlib.sha256()
    for row, record in enumerate(records):
        piece_begin, piece_end = sparse.piece_offsets[row : row + 2]
        royal_begin, royal_end = sparse.royal_offsets[row : row + 2]
        digest.update(
            struct.pack(
                "<IBhhbB",
                record.index,
                record.side_to_move,
                record.score,
                record.game_ply,
                record.result,
                piece_buckets[row],
            )
        )
        for indices in (
            sparse.legacy_white[piece_begin:piece_end],
            sparse.legacy_black[piece_begin:piece_end],
            sparse.v2_global[piece_begin:piece_end],
            sparse.v2_royal[royal_begin:royal_end],
        ):
            digest.update(struct.pack("<I", len(indices)))
            for feature in indices:
                digest.update(struct.pack("<I", feature))

    torch_batch = TorchBatch(
        legacy_white=torch.tensor(sparse.legacy_white, dtype=torch.long),
        legacy_black=torch.tensor(sparse.legacy_black, dtype=torch.long),
        piece_offsets=torch.tensor(sparse.piece_offsets, dtype=torch.long),
        v2_global=torch.tensor(sparse.v2_global, dtype=torch.long),
        v2_royal=torch.tensor(sparse.v2_royal, dtype=torch.long),
        royal_offsets=torch.tensor(sparse.royal_offsets, dtype=torch.long),
        side_to_move=torch.tensor(sparse.side_to_move, dtype=torch.long),
        piece_buckets=torch.tensor(piece_buckets, dtype=torch.long),
        scores=torch.tensor(sparse.scores, dtype=torch.float32),
        result_targets=(torch.tensor(sparse.results, dtype=torch.float32) + 1.0) / 2.0,
    )
    receipt = {
        "schema": CORPUS_SCHEMA,
        "sample_count": len(records),
        "samples_per_legacy_bucket": {
            str(bucket): count for bucket, count in sorted(Counter(piece_buckets).items())
        },
        "side_to_move": {
            "white": sparse.side_to_move.count(WHITE),
            "black": sparse.side_to_move.count(BLACK),
        },
        "result_targets": {
            str(result): count for result, count in sorted(Counter(sparse.results).items())
        },
        "score_minimum": min(sparse.scores),
        "score_maximum": max(sparse.scores),
        "piece_rows": len(sparse.v2_global),
        "royal_rows": len(sparse.v2_royal),
        "sha256": digest.hexdigest().upper(),
    }
    _require(
        receipt["sha256"] == EXPECTED_CORPUS_SHA256,
        f"micro-fit corpus drifted: {receipt['sha256']}",
    )
    return torch_batch, receipt


def _uniform_parameter(shape: Sequence[int], generator: torch.Generator, radius: float) -> nn.Parameter:
    value = torch.empty(tuple(shape), dtype=torch.float32)
    value.uniform_(-radius, radius, generator=generator)
    return nn.Parameter(value)


def _sparse_sum(indices: Tensor, offsets: Tensor, weights: Tensor, bias: Tensor) -> Tensor:
    return functional.embedding_bag(
        indices,
        weights,
        offsets,
        mode="sum",
        include_last_offset=True,
    ) + bias


class LegacyHPModel(nn.Module):
    """Float training form of the serialized Run 6B H/P topology."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.ft_weights = _uniform_parameter(
            (LEGACY_DIMENSIONS, LEGACY_ACCUMULATOR_LANES), generator, 0.012
        )
        self.ft_bias = nn.Parameter(torch.full((LEGACY_ACCUMULATOR_LANES,), 0.45))
        self.psqt_weights = _uniform_parameter((LEGACY_DIMENSIONS, LEGACY_BUCKETS), generator, 0.004)
        self.hidden0_weights = _uniform_parameter(
            (LEGACY_BUCKETS, 16, 2 * LEGACY_ACCUMULATOR_LANES), generator, 0.012
        )
        self.hidden0_bias = nn.Parameter(torch.full((LEGACY_BUCKETS, 16), 0.20))
        self.hidden1_weights = _uniform_parameter((LEGACY_BUCKETS, 32, 16), generator, 0.025)
        self.hidden1_bias = nn.Parameter(torch.full((LEGACY_BUCKETS, 32), 0.20))
        self.output_weights = _uniform_parameter((LEGACY_BUCKETS, 1, 32), generator, 0.020)
        self.output_bias = nn.Parameter(torch.zeros((LEGACY_BUCKETS, 1)))

    def forward(self, batch: TorchBatch) -> Tensor:
        white = _sparse_sum(batch.legacy_white, batch.piece_offsets, self.ft_weights, self.ft_bias)
        black = _sparse_sum(batch.legacy_black, batch.piece_offsets, self.ft_weights, self.ft_bias)
        white_to_move = (batch.side_to_move == WHITE).unsqueeze(1)
        us = torch.where(white_to_move, white, black)
        them = torch.where(white_to_move, black, white)
        transformed = torch.clamp(torch.cat((us, them), dim=1), 0.0, 1.0)

        selected0 = self.hidden0_weights[batch.piece_buckets]
        hidden0 = torch.clamp(
            torch.bmm(selected0, transformed.unsqueeze(2)).squeeze(2)
            + self.hidden0_bias[batch.piece_buckets],
            0.0,
            1.0,
        )
        selected1 = self.hidden1_weights[batch.piece_buckets]
        hidden1 = torch.clamp(
            torch.bmm(selected1, hidden0.unsqueeze(2)).squeeze(2)
            + self.hidden1_bias[batch.piece_buckets],
            0.0,
            1.0,
        )
        positional = (
            torch.bmm(self.output_weights[batch.piece_buckets], hidden1.unsqueeze(2)).squeeze(2)
            + self.output_bias[batch.piece_buckets]
        ).squeeze(1)

        white_psqt = _sparse_sum(
            batch.legacy_white,
            batch.piece_offsets,
            self.psqt_weights,
            self.psqt_weights.new_zeros(LEGACY_BUCKETS),
        )
        black_psqt = _sparse_sum(
            batch.legacy_black,
            batch.piece_offsets,
            self.psqt_weights,
            self.psqt_weights.new_zeros(LEGACY_BUCKETS),
        )
        bucket = batch.piece_buckets.unsqueeze(1)
        white_psqt = white_psqt.gather(1, bucket).squeeze(1)
        black_psqt = black_psqt.gather(1, bucket).squeeze(1)
        psqt = (white_psqt - black_psqt) * (
            (batch.side_to_move == WHITE).to(torch.float32) - 0.5
        )
        return positional + psqt

    def gradient_groups(self) -> dict[str, Iterable[nn.Parameter]]:
        return {
            "feature_transformer": (self.ft_weights, self.ft_bias),
            "psqt": (self.psqt_weights,),
            "dense_trunk": (
                self.hidden0_weights,
                self.hidden0_bias,
                self.hidden1_weights,
                self.hidden1_bias,
            ),
            "output": (self.output_weights, self.output_bias),
        }


class HordeV2Model(nn.Module):
    """No-context V2 base topology with one shared trunk and two STM rows."""

    def __init__(self, royal_lanes: int, global_lanes: int, seed: int) -> None:
        super().__init__()
        self.royal_lanes = royal_lanes
        self.global_lanes = global_lanes
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.royal_weights = _uniform_parameter(
            (V2_ROYAL_DIMENSIONS, royal_lanes), generator, 0.012
        )
        self.royal_bias = nn.Parameter(torch.full((royal_lanes,), 0.45))
        self.global_weights = _uniform_parameter(
            (V2_GLOBAL_DIMENSIONS, global_lanes), generator, 0.012
        )
        self.global_bias = nn.Parameter(torch.full((global_lanes,), 0.45))
        transformed = royal_lanes + global_lanes
        self.hidden0_weights = _uniform_parameter((HIDDEN0_LANES, transformed), generator, 0.018)
        self.hidden0_bias = nn.Parameter(torch.full((HIDDEN0_LANES,), 0.20))
        self.hidden1_weights = _uniform_parameter((HIDDEN1_LANES, HIDDEN0_LANES), generator, 0.025)
        self.hidden1_bias = nn.Parameter(torch.full((HIDDEN1_LANES,), 0.20))
        self.output_weights = _uniform_parameter((2, HIDDEN1_LANES), generator, 0.020)
        self.output_bias = nn.Parameter(torch.zeros(2))

    def forward(self, batch: TorchBatch) -> Tensor:
        royal = _sparse_sum(
            batch.v2_royal, batch.royal_offsets, self.royal_weights, self.royal_bias
        )
        global_ = _sparse_sum(
            batch.v2_global, batch.piece_offsets, self.global_weights, self.global_bias
        )
        transformed = torch.clamp(torch.cat((royal, global_), dim=1), 0.0, 1.0)
        hidden0 = torch.clamp(
            functional.linear(transformed, self.hidden0_weights, self.hidden0_bias),
            0.0,
            1.0,
        )
        hidden1 = torch.clamp(
            functional.linear(hidden0, self.hidden1_weights, self.hidden1_bias),
            0.0,
            1.0,
        )
        all_heads = functional.linear(hidden1, self.output_weights, self.output_bias)
        return all_heads.gather(1, batch.side_to_move.unsqueeze(1)).squeeze(1)

    def gradient_groups(self) -> dict[str, Iterable[nn.Parameter]]:
        return {
            "royal_transformer": (self.royal_weights, self.royal_bias),
            "global_transformer": (self.global_weights, self.global_bias),
            "dense_trunk": (
                self.hidden0_weights,
                self.hidden0_bias,
                self.hidden1_weights,
                self.hidden1_bias,
            ),
            "output": (self.output_weights, self.output_bias),
        }


def _loss(output: Tensor, batch: TorchBatch) -> Tensor:
    prediction = torch.sigmoid(output * NNUE_TO_SCORE / OUTPUT_SIGMOID_SCALE)
    eval_target = torch.sigmoid(batch.scores / EVAL_SIGMOID_SCALE)
    eval_loss = torch.square(eval_target - prediction).mean()
    result_loss = torch.square(batch.result_targets - prediction).mean()
    return LAMBDA * eval_loss + (1.0 - LAMBDA) * result_loss


def _state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        contiguous = value.detach().cpu().contiguous()
        encoded_name = name.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded_name)))
        digest.update(encoded_name)
        digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
        digest.update(struct.pack("<I", contiguous.ndim))
        for dimension in contiguous.shape:
            digest.update(struct.pack("<Q", dimension))
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest().upper()


def _loss_sha256(losses: Sequence[float]) -> str:
    digest = hashlib.sha256()
    for loss in losses:
        digest.update(struct.pack("<d", loss))
    return digest.hexdigest().upper()


def _gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        _require(parameter.grad is not None, "micro-fit parameter did not receive a gradient")
        gradient = parameter.grad.detach().to(torch.float64)
        squared += float(torch.sum(gradient * gradient).item())
    return math.sqrt(squared)


def _fit_once(
    factory: Any,
    batch: TorchBatch,
    steps: int,
    learning_rate: float,
) -> dict[str, object]:
    model: nn.Module = factory()
    initial_sha256 = _state_sha256(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    losses: list[float] = []
    gradient_norms: dict[str, float] | None = None

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        value = _loss(model(batch), batch)
        _require(bool(torch.isfinite(value)), f"micro-fit loss became non-finite at step {step}")
        losses.append(float(value.detach().to(torch.float64).item()))
        value.backward()
        if step == 0:
            gradient_norms = {
                name: _gradient_norm(parameters)
                for name, parameters in model.gradient_groups().items()
            }
            _require(
                all(math.isfinite(norm) and norm > 0.0 for norm in gradient_norms.values()),
                f"micro-fit has a missing gradient domain: {gradient_norms}",
            )
        optimizer.step()

    with torch.no_grad():
        final_loss = _loss(model(batch), batch)
    _require(bool(torch.isfinite(final_loss)), "micro-fit final loss is non-finite")
    losses.append(float(final_loss.to(torch.float64).item()))
    _require(losses[-1] < losses[0], f"micro-fit did not reduce loss: {losses[0]} -> {losses[-1]}")
    _require(all(torch.isfinite(parameter).all() for parameter in model.parameters()),
             "micro-fit produced non-finite parameters")
    _require(gradient_norms is not None, "micro-fit did not execute a training step")

    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_state_sha256": initial_sha256,
        "final_state_sha256": _state_sha256(model),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_reduction": losses[0] - losses[-1],
        "loss_sha256": _loss_sha256(losses),
        "first_step_gradient_l2": gradient_norms,
    }


def _model_receipt(
    name: str,
    topology: dict[str, object],
    factory: Any,
    batch: TorchBatch,
    steps: int,
    learning_rate: float,
) -> dict[str, object]:
    first = _fit_once(factory, batch, steps, learning_rate)
    second = _fit_once(factory, batch, steps, learning_rate)
    _require(first == second, f"{name} repeated CPU micro-fits are not bit-identical")
    return {
        "name": name,
        "topology": topology,
        "repeat_count": 2,
        "repeat_exact": True,
        **first,
    }


def build_receipt(steps: int, learning_rate: float) -> dict[str, object]:
    _require(steps > 0, "micro-fit steps must be positive")
    _require(math.isfinite(learning_rate) and learning_rate > 0.0,
             "micro-fit learning rate must be positive and finite")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    torch.set_float32_matmul_precision("highest")

    batch, corpus = make_fixture_batch()
    models = [
        _model_receipt(
            "fresh-legacy-hp",
            {
                "schema": "HORDETEST_HP_FRESH_CONTROL_V1",
                "feature_dimensions": LEGACY_DIMENSIONS,
                "accumulator_lanes": LEGACY_ACCUMULATOR_LANES,
                "perspectives": 2,
                "psqt_buckets": LEGACY_BUCKETS,
                "dense": [1024, 16, 32, 1],
                "layer_stacks": LEGACY_BUCKETS,
            },
            lambda: LegacyHPModel(MODEL_SEED),
            batch,
            steps,
            learning_rate,
        ),
        _model_receipt(
            "v2-64+192",
            {
                "schema": "V2_BASE_P0_64X192",
                "royal_dimensions": V2_ROYAL_DIMENSIONS,
                "royal_lanes": 64,
                "global_dimensions": V2_GLOBAL_DIMENSIONS,
                "global_lanes": 192,
                "dense": [256, HIDDEN0_LANES, HIDDEN1_LANES, 1],
                "side_to_move_heads": 2,
            },
            lambda: HordeV2Model(64, 192, MODEL_SEED),
            batch,
            steps,
            learning_rate,
        ),
        _model_receipt(
            "v2-128+128",
            {
                "schema": "V2_BASE_P0_128X128",
                "royal_dimensions": V2_ROYAL_DIMENSIONS,
                "royal_lanes": 128,
                "global_dimensions": V2_GLOBAL_DIMENSIONS,
                "global_lanes": 128,
                "dense": [256, HIDDEN0_LANES, HIDDEN1_LANES, 1],
                "side_to_move_heads": 2,
            },
            lambda: HordeV2Model(128, 128, MODEL_SEED),
            batch,
            steps,
            learning_rate,
        ),
    ]

    cuda_device = None
    if torch.cuda.is_available():
        cuda_device = torch.cuda.get_device_name(0)
    implementation_path = Path(__file__).resolve()
    decoder_path = implementation_path.with_name("horde_training_decoder.py")
    return {
        "schema": SCHEMA,
        "purpose": "deterministic gradient-plumbing receipt; no strength claim",
        "implementation": {
            "microfit_sha256": hashlib.sha256(implementation_path.read_bytes()).hexdigest().upper(),
            "decoder_sha256": hashlib.sha256(decoder_path.read_bytes()).hexdigest().upper(),
        },
        "execution": {
            "device": "cpu",
            "threads": 1,
            "interop_threads": 1,
            "deterministic_algorithms": True,
            "mkldnn": False,
            "float32_matmul_precision": "highest",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cuda_available_but_unused": torch.cuda.is_available(),
            "cuda_device": cuda_device,
        },
        "training": {
            "model_seed": MODEL_SEED,
            "steps": steps,
            "optimizer": "SGD",
            "learning_rate": learning_rate,
            "momentum": 0.9,
            "lambda": LAMBDA,
            "nnue_to_score": NNUE_TO_SCORE,
            "eval_sigmoid_scale": EVAL_SIGMOID_SCALE,
            "output_sigmoid_scale": OUTPUT_SIGMOID_SCALE,
            "labels": "frozen synthetic engineering fixture in side-to-move frame",
            "rule50": "not presented to the model; output represents pre-postprocessor value",
        },
        "corpus": corpus,
        "models": models,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_receipt(args.steps, args.learning_rate)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8", newline="\n")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
