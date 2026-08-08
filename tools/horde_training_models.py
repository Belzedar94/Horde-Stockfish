#!/usr/bin/env python3
"""Shared float training models for Horde NNUE controls and V2 ablations."""

from __future__ import annotations

import hashlib
import struct
from typing import Iterable, Protocol, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as functional

try:
    from .horde_training_decoder import (
        LEGACY_DIMENSIONS,
        V2_GLOBAL_DIMENSIONS,
        V2_ROYAL_DIMENSIONS,
        WHITE,
    )
except ImportError:
    from horde_training_decoder import (
        LEGACY_DIMENSIONS,
        V2_GLOBAL_DIMENSIONS,
        V2_ROYAL_DIMENSIONS,
        WHITE,
    )


NAMED_INITIALIZATION_SCHEMA = "SHA256_NAMED_PARAMETER_SEED_V1"
LEGACY_ACCUMULATOR_LANES = 512
LEGACY_BUCKETS = 8
HIDDEN0_LANES = 32
HIDDEN1_LANES = 32
NNUE_TO_SCORE = 600.0


class LegacyModelBatch(Protocol):
    legacy_white: Tensor
    legacy_black: Tensor
    legacy_piece_offsets: Tensor
    side_to_move: Tensor
    piece_buckets: Tensor


class V2ModelBatch(Protocol):
    v2_global: Tensor
    v2_royal: Tensor
    global_offsets: Tensor
    royal_offsets: Tensor
    side_to_move: Tensor


def _named_generator(seed: int, name: str) -> torch.Generator:
    digest = hashlib.sha256()
    digest.update(NAMED_INITIALIZATION_SCHEMA.encode("ascii") + b"\0")
    digest.update(struct.pack("<Q", seed & ((1 << 64) - 1)))
    digest.update(name.encode("utf-8"))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest.digest()[:8], "little"))
    return generator


def _uniform_parameter(
    shape: Sequence[int],
    seed: int,
    name: str,
    radius: float,
) -> nn.Parameter:
    value = torch.empty(tuple(shape), dtype=torch.float32)
    value.uniform_(-radius, radius, generator=_named_generator(seed, name))
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
        self.ft_weights = _uniform_parameter(
            (LEGACY_DIMENSIONS, LEGACY_ACCUMULATOR_LANES), seed, "legacy.ft_weights", 0.012
        )
        self.ft_bias = nn.Parameter(torch.full((LEGACY_ACCUMULATOR_LANES,), 0.45))
        self.psqt_weights = _uniform_parameter(
            (LEGACY_DIMENSIONS, LEGACY_BUCKETS), seed, "legacy.psqt_weights", 0.004
        )
        self.hidden0_weights = _uniform_parameter(
            (LEGACY_BUCKETS, 16, 2 * LEGACY_ACCUMULATOR_LANES),
            seed,
            "legacy.hidden0_weights",
            0.012,
        )
        self.hidden0_bias = nn.Parameter(torch.full((LEGACY_BUCKETS, 16), 0.20))
        self.hidden1_weights = _uniform_parameter(
            (LEGACY_BUCKETS, 32, 16), seed, "legacy.hidden1_weights", 0.025
        )
        self.hidden1_bias = nn.Parameter(torch.full((LEGACY_BUCKETS, 32), 0.20))
        self.output_weights = _uniform_parameter(
            (LEGACY_BUCKETS, 1, 32), seed, "legacy.output_weights", 0.020
        )
        self.output_bias = nn.Parameter(torch.zeros((LEGACY_BUCKETS, 1)))

    def forward(self, batch: LegacyModelBatch) -> Tensor:
        white = _sparse_sum(
            batch.legacy_white,
            batch.legacy_piece_offsets,
            self.ft_weights,
            self.ft_bias,
        )
        black = _sparse_sum(
            batch.legacy_black,
            batch.legacy_piece_offsets,
            self.ft_weights,
            self.ft_bias,
        )
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
            batch.legacy_piece_offsets,
            self.psqt_weights,
            self.psqt_weights.new_zeros(LEGACY_BUCKETS),
        )
        black_psqt = _sparse_sum(
            batch.legacy_black,
            batch.legacy_piece_offsets,
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
        self.royal_weights = _uniform_parameter(
            (V2_ROYAL_DIMENSIONS, royal_lanes), seed, "v2.royal_weights", 0.012
        )
        self.royal_bias = nn.Parameter(torch.full((royal_lanes,), 0.45))
        self.global_weights = _uniform_parameter(
            (V2_GLOBAL_DIMENSIONS, global_lanes), seed, "v2.global_weights", 0.012
        )
        self.global_bias = nn.Parameter(torch.full((global_lanes,), 0.45))
        transformed = royal_lanes + global_lanes
        self.hidden0_weights = _uniform_parameter(
            (HIDDEN0_LANES, transformed), seed, "v2.hidden0_weights", 0.018
        )
        self.hidden0_bias = nn.Parameter(torch.full((HIDDEN0_LANES,), 0.20))
        self.hidden1_weights = _uniform_parameter(
            (HIDDEN1_LANES, HIDDEN0_LANES), seed, "v2.hidden1_weights", 0.025
        )
        self.hidden1_bias = nn.Parameter(torch.full((HIDDEN1_LANES,), 0.20))
        self.output_weights = _uniform_parameter(
            (2, HIDDEN1_LANES), seed, "v2.output_weights", 0.020
        )
        self.output_bias = nn.Parameter(torch.zeros(2))

    def forward(self, batch: V2ModelBatch) -> Tensor:
        royal = _sparse_sum(
            batch.v2_royal,
            batch.royal_offsets,
            self.royal_weights,
            self.royal_bias,
        )
        global_ = _sparse_sum(
            batch.v2_global,
            batch.global_offsets,
            self.global_weights,
            self.global_bias,
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
