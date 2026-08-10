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
        V2_ROYAL_RANK8_DIMENSIONS,
        V2_ROYAL_DIMENSIONS,
        WHITE,
    )
except ImportError:
    from horde_training_decoder import (
        LEGACY_DIMENSIONS,
        V2_GLOBAL_DIMENSIONS,
        V2_ROYAL_RANK8_DIMENSIONS,
        V2_ROYAL_DIMENSIONS,
        WHITE,
    )


NAMED_INITIALIZATION_SCHEMA = "SHA256_NAMED_PARAMETER_SEED_V1"
LEGACY_ACCUMULATOR_LANES = 512
LEGACY_BUCKETS = 8
HIDDEN0_LANES = 32
HIDDEN1_LANES = 32
NNUE_TO_SCORE = 600.0
V2_ABSOLUTE_NONKING_DIMENSIONS = 10 * 64


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


class AbsoluteNonKingV2Model(nn.Module):
    """C1 control with absolute non-king rows instead of Royal-relative rows."""

    def __init__(self, absolute_lanes: int, global_lanes: int, seed: int) -> None:
        super().__init__()
        self.absolute_lanes = absolute_lanes
        self.global_lanes = global_lanes
        self.absolute_nonking_weights = _uniform_parameter(
            (V2_ABSOLUTE_NONKING_DIMENSIONS, absolute_lanes),
            seed,
            "v2.absolute_nonking_weights",
            0.012,
        )
        self.absolute_nonking_bias = nn.Parameter(
            torch.full((absolute_lanes,), 0.45)
        )
        self.global_weights = _uniform_parameter(
            (V2_GLOBAL_DIMENSIONS, global_lanes), seed, "v2.global_weights", 0.012
        )
        self.global_bias = nn.Parameter(torch.full((global_lanes,), 0.45))
        transformed = absolute_lanes + global_lanes
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

    @staticmethod
    def absolute_nonking_features(batch: V2ModelBatch) -> tuple[Tensor, Tensor]:
        """Project G0 onto its ten non-king physical role planes."""

        keep = batch.v2_global < V2_ABSOLUTE_NONKING_DIMENSIONS
        prefix = torch.cat(
            (
                torch.zeros(1, dtype=batch.global_offsets.dtype, device=keep.device),
                torch.cumsum(keep.to(dtype=batch.global_offsets.dtype), dim=0),
            )
        )
        offsets = prefix.index_select(0, batch.global_offsets)
        return batch.v2_global[keep], offsets

    def forward(self, batch: V2ModelBatch) -> Tensor:
        absolute_indices, absolute_offsets = self.absolute_nonking_features(batch)
        absolute = _sparse_sum(
            absolute_indices,
            absolute_offsets,
            self.absolute_nonking_weights,
            self.absolute_nonking_bias,
        )
        global_ = _sparse_sum(
            batch.v2_global,
            batch.global_offsets,
            self.global_weights,
            self.global_bias,
        )
        transformed = torch.clamp(torch.cat((absolute, global_), dim=1), 0.0, 1.0)
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
            "absolute_nonking_transformer": (
                self.absolute_nonking_weights,
                self.absolute_nonking_bias,
            ),
            "global_transformer": (self.global_weights, self.global_bias),
            "dense_trunk": (
                self.hidden0_weights,
                self.hidden0_bias,
                self.hidden1_weights,
                self.hidden1_bias,
            ),
            "output": (self.output_weights, self.output_bias),
        }


class RoyalRank8V2Model(nn.Module):
    """C1 control keyed by Black-king rank and the canonical mirror bit."""

    def __init__(self, royal_lanes: int, global_lanes: int, seed: int) -> None:
        super().__init__()
        self.royal_lanes = royal_lanes
        self.global_lanes = global_lanes
        self.royal_rank8_weights = _uniform_parameter(
            (V2_ROYAL_RANK8_DIMENSIONS, royal_lanes),
            seed,
            "v2.royal_rank8_weights",
            0.012,
        )
        self.royal_rank8_bias = nn.Parameter(torch.full((royal_lanes,), 0.45))
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

    @staticmethod
    def royal_rank8_features(batch: V2ModelBatch) -> tuple[Tensor, Tensor]:
        """Fold each 32-bucket Royal row into its eight-rank bucket."""

        rows_per_bucket = 10 * 64
        bucket = torch.div(batch.v2_royal, rows_per_bucket, rounding_mode="floor")
        within_bucket = torch.remainder(batch.v2_royal, rows_per_bucket)
        rank8 = torch.div(bucket, 4, rounding_mode="floor") * rows_per_bucket + within_bucket
        return rank8, batch.royal_offsets

    def forward(self, batch: V2ModelBatch) -> Tensor:
        rank8_indices, rank8_offsets = self.royal_rank8_features(batch)
        royal = _sparse_sum(
            rank8_indices,
            rank8_offsets,
            self.royal_rank8_weights,
            self.royal_rank8_bias,
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
            "royal_rank8_transformer": (
                self.royal_rank8_weights,
                self.royal_rank8_bias,
            ),
            "global_transformer": (self.global_weights, self.global_bias),
            "dense_trunk": (
                self.hidden0_weights,
                self.hidden0_bias,
                self.hidden1_weights,
                self.hidden1_bias,
            ),
            "output": (self.output_weights, self.output_bias),
        }


class C0SingleG0Model(nn.Module):
    """Engineering control with one absolute G0 transformer of 256 lanes."""

    lanes = 256

    def __init__(self, seed: int) -> None:
        super().__init__()
        self.global_weights = _uniform_parameter(
            (V2_GLOBAL_DIMENSIONS, self.lanes), seed, "v2.global_weights", 0.012
        )
        self.global_bias = nn.Parameter(torch.full((self.lanes,), 0.45))
        self.hidden0_weights = _uniform_parameter(
            (HIDDEN0_LANES, self.lanes), seed, "v2.hidden0_weights", 0.018
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
        transformed = torch.clamp(
            _sparse_sum(
                batch.v2_global,
                batch.global_offsets,
                self.global_weights,
                self.global_bias,
            ),
            0.0,
            1.0,
        )
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


class C0SplitG0Model(nn.Module):
    """The same C0 G0 content split into two independently updated tensors."""

    lanes = 256

    def __init__(self, source: C0SingleG0Model, first_lanes: int) -> None:
        super().__init__()
        if first_lanes not in (64, 128):
            raise ValueError(f"unsupported C0 first-domain width: {first_lanes}")
        self.first_lanes = first_lanes
        self.second_lanes = self.lanes - first_lanes
        self.first_weights = nn.Parameter(
            source.global_weights[:, :first_lanes].detach().clone()
        )
        self.first_bias = nn.Parameter(source.global_bias[:first_lanes].detach().clone())
        self.second_weights = nn.Parameter(
            source.global_weights[:, first_lanes:].detach().clone()
        )
        self.second_bias = nn.Parameter(source.global_bias[first_lanes:].detach().clone())
        for name in (
            "hidden0_weights",
            "hidden0_bias",
            "hidden1_weights",
            "hidden1_bias",
            "output_weights",
            "output_bias",
        ):
            setattr(self, name, nn.Parameter(getattr(source, name).detach().clone()))

    def forward(self, batch: V2ModelBatch) -> Tensor:
        first = _sparse_sum(
            batch.v2_global,
            batch.global_offsets,
            self.first_weights,
            self.first_bias,
        )
        second = _sparse_sum(
            batch.v2_global,
            batch.global_offsets,
            self.second_weights,
            self.second_bias,
        )
        transformed = torch.clamp(torch.cat((first, second), dim=1), 0.0, 1.0)
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
