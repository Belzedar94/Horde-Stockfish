#!/usr/bin/env python3
"""Train Horde NNUE controls from disjoint HORDE_BIN_V1 files.

This is the deterministic reference trainer for the first Horde NNUE training
control.  It deliberately trains the exact serialized Run 6B topology without
the historical training-only layer factorizer.  The same optimizer, schedule,
label policy, and batching contract are also used by the no-context V2 width
survivors so architecture comparisons do not receive hidden training changes.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import struct
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence

# Required by PyTorch for deterministic CUDA matrix multiplications.  It must
# be present before the first cuBLAS handle is created.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - exercised by the CLI failure path
    raise SystemExit("PyTorch is required for Horde NNUE control training") from error

try:
    from .horde_training_decoder import (
        BLACK,
        HordeBinV1Dataset,
        SparseBatch,
        V2_GLOBAL_DIMENSIONS,
        V2_ROYAL_RANK8_DIMENSIONS,
        V2_ROYAL_DIMENSIONS,
        V3_GLOBAL_DIMENSIONS,
        V3_MAX_ACTIVE_ROWS,
        WHITE,
        dataset_receipt,
        make_sparse_batch,
        v3_contextual_rows,
    )
    from .horde_training_models import (
        AbsoluteNonKingV2Model,
        C0SingleG0Model,
        HIDDEN0_LANES,
        HIDDEN1_LANES,
        HordeV2Model,
        HordeV3Model,
        LEGACY_BUCKETS,
        LegacyHPModel,
        NNUE_TO_SCORE,
        RoyalRank8V2Model,
        V2_ABSOLUTE_NONKING_DIMENSIONS,
        v3_phase_bucket,
        v3_structure,
    )
    from .horde_training_chunk_set import HordeChunkSetDataset
    from .horde_training_split_audit import audit_pair
    from .horde_training_selected_role import (
        SelectedRoleDataset,
        SelectedRoleError,
        validate_selected_role_binding,
    )
    from .horde_training_scale_selected_role import (
        CONTRACT_SCHEMA as SCALE_CONTRACT_SCHEMA,
        SCALE_CONTRACTS,
        CONTRACT_SHA256 as SCALE_CONTRACT_SHA256,
        ScaleSelectedRoleDataset,
        ScaleSelectedRoleError,
        load_contract as load_scale_contract,
    )
    from .horde_wdl import (
        CalibrationError,
        LINK_SCHEMA as WDL_LINK_SCHEMA,
        SIDE_NAMES as WDL_SIDE_NAMES,
        load_artifact as load_wdl_artifact,
    )
except ImportError:
    from horde_training_decoder import (
        BLACK,
        HordeBinV1Dataset,
        SparseBatch,
        V2_GLOBAL_DIMENSIONS,
        V2_ROYAL_RANK8_DIMENSIONS,
        V2_ROYAL_DIMENSIONS,
        V3_GLOBAL_DIMENSIONS,
        V3_MAX_ACTIVE_ROWS,
        WHITE,
        dataset_receipt,
        make_sparse_batch,
        v3_contextual_rows,
    )
    from horde_training_models import (
        AbsoluteNonKingV2Model,
        C0SingleG0Model,
        HIDDEN0_LANES,
        HIDDEN1_LANES,
        HordeV2Model,
        HordeV3Model,
        LEGACY_BUCKETS,
        LegacyHPModel,
        NNUE_TO_SCORE,
        RoyalRank8V2Model,
        V2_ABSOLUTE_NONKING_DIMENSIONS,
        v3_phase_bucket,
        v3_structure,
    )
    from horde_training_chunk_set import HordeChunkSetDataset
    from horde_training_split_audit import audit_pair
    from horde_training_selected_role import (
        SelectedRoleDataset,
        SelectedRoleError,
        validate_selected_role_binding,
    )
    from horde_training_scale_selected_role import (
        CONTRACT_SCHEMA as SCALE_CONTRACT_SCHEMA,
        SCALE_CONTRACTS,
        CONTRACT_SHA256 as SCALE_CONTRACT_SHA256,
        ScaleSelectedRoleDataset,
        ScaleSelectedRoleError,
        load_contract as load_scale_contract,
    )
    from horde_wdl import (
        CalibrationError,
        LINK_SCHEMA as WDL_LINK_SCHEMA,
        SIDE_NAMES as WDL_SIDE_NAMES,
        load_artifact as load_wdl_artifact,
    )


SCHEMA = "HORDE_FRESH_LEGACY_CONTROL_TRAINING_V3"
CHECKPOINT_SCHEMA = "HORDE_FRESH_LEGACY_CONTROL_CHECKPOINT_V3"
ARCHITECTURE_SCHEMA = "HORDETEST_HP_FRESH_CONTROL_V1"
LEGACY_ARCHITECTURE = "fresh-legacy-hp"
V2_TRAINING_SCHEMA = "HORDE_V2_BASE_TRAINING_V1"
V2_CHECKPOINT_SCHEMA = "HORDE_V2_BASE_CHECKPOINT_V1"
V2_FEATURE_SCHEMA = "V2_BASE_P0"
V3_TRAINING_SCHEMA = "HORDE_V3_TRAINING_V1"
V3_CHECKPOINT_SCHEMA = "HORDE_V3_CHECKPOINT_V1"
# The container binds this hash.  It is derived from the architecture alone by
# models.v3_structure(); the constant exists so a silent drift fails closed here
# rather than at export time.
V3_STRUCTURAL_SHA256 = (
    "5B49DC20A9EF5424687A649A7E1BEA8665FE96F3AB839A2BA4CEAE8289EC69EC"
)
RANK8_SCALE_ARCHITECTURE = "v2-c1-rank8-64x192"
C1_PLAN_SCHEMA = "HORDE_V2_C1_CAMPAIGN_PLAN_V2"
C1_BINDING_SCHEMA = "HORDE_V2_C1_TRAINER_BINDING_V1"
SCALE_BINDING_SCHEMA = "HORDE_V2_RANK8_SCALE_TRAINER_BINDING_V1"
C1_PARENT_CONTRACT_SCHEMA = "HORDE_V2_C1_CAMPAIGN_V1"
C1_PARENT_CONTRACT_SHA256 = (
    "7B5BDA9DC20AB7CF55DE2964085D2ADBBED83137A3071B418439A5CF7DD939DA"
)
C1_COVERAGE_ADDENDUM_SCHEMA = "HORDE_V2_C1_COVERAGE_ADDENDUM_V1"
C1_COVERAGE_ADDENDUM_SHA256 = (
    "3103951AB8522238C23DBF83A3DEB0073D671024B0054ACBB75E1DC0C7C7D91B"
)
V2_ARCHITECTURES = {
    "v2-64x192": {
        "schema": "V2_BASE_P0_64X192",
        "first_domain": "royal",
        "first_lanes": 64,
        "global_lanes": 192,
        "serialized_parameter_bytes": 2_902_344,
    },
    "v2-128x128": {
        "schema": "V2_BASE_P0_128X128",
        "first_domain": "royal",
        "first_lanes": 128,
        "global_lanes": 128,
        "serialized_parameter_bytes": 5_433_672,
    },
    "v2-c1-abs64x192": {
        "schema": "V2_C1_ABS_NONKING_64X192",
        "first_domain": "absolute_nonking",
        "first_lanes": 64,
        "global_lanes": 192,
        "serialized_parameter_bytes": 362_824,
    },
    "v2-c1-rank8-64x192": {
        "schema": "V2_C1_ROYAL_RANK8_64X192",
        "first_domain": "royal_rank8",
        "first_lanes": 64,
        "global_lanes": 192,
        "serialized_parameter_bytes": 936_264,
    },
}
# Single-domain V2-shaped controls.  They consume the ordinary V2 batch, so they
# reuse the V2 training and checkpoint schemas, but their structure receipt has
# one domain rather than two and therefore its own structural hash.
CONTROL_ARCHITECTURES = {
    "v2-c0-g0single-256": {
        "schema": "G0_SINGLE_256",
        "global_lanes": 256,
        "serialized_parameter_bytes": 371_016,
        "role": "R2 control: one absolute G0 table against the V2 dual domain",
    },
}
# V3 consumes its own batch: the same Global stream extended with the named
# contextual tail, plus a white_piece_count phase bucket per record.
V3_ARCHITECTURES = {
    "v3-g1024-pawn-wpc8": {
        "schema": "V3_G1024_PAWN_WPC8",
        "serialized_parameter_bytes": 2_033_765,
    },
}
ARCHITECTURE_CHOICES = (
    LEGACY_ARCHITECTURE,
    *V2_ARCHITECTURES,
    *CONTROL_ARCHITECTURES,
    *V3_ARCHITECTURES,
)
BOOK_SPLIT_SCHEMAS = {
    "HORDE_TRAINING_BOOK_SPLIT_V1",
    "HORDE_TRAINING_BOOK_SPLIT_V2",
}
MATE_SCORE_THRESHOLD = 31_507  # VALUE_TB_WIN_IN_MAX_PLY at MAX_PLY=246.
DEFAULT_LAMBDA = 0.6
DEFAULT_LEARNING_RATE = 1.5e-3
DEFAULT_SCHEDULER_GAMMA = 0.987
DEFAULT_DENSE_LEARNING_RATE_MULTIPLIER = 1.0
DEFAULT_OUTPUT_LEARNING_RATE_MULTIPLIER = 0.1
MASK_POLICY = "exclude abs(score) >= 31507 from score-derived WDL term; retain result WDL term"
U64_MASK = (1 << 64) - 1


class TrainingError(ValueError):
    """Raised when a training input or requested run violates the contract."""


def _validate_optimizer_learning_rate_multipliers(
    dense: float,
    output: float,
) -> tuple[float, float]:
    _require(
        math.isfinite(dense) and dense > 0.0,
        "dense learning-rate multiplier must be positive",
    )
    _require(
        math.isfinite(output) and output > 0.0,
        "output learning-rate multiplier must be positive",
    )
    _require(
        dense == DEFAULT_DENSE_LEARNING_RATE_MULTIPLIER
        or output == DEFAULT_OUTPUT_LEARNING_RATE_MULTIPLIER,
        "optimizer experiments may change only one learning-rate multiplier",
    )
    return dense, output


def _optimizer_learning_rate_multipliers(
    args: argparse.Namespace,
) -> tuple[float, float]:
    return _validate_optimizer_learning_rate_multipliers(
        float(
            getattr(
                args,
                "dense_learning_rate_multiplier",
                DEFAULT_DENSE_LEARNING_RATE_MULTIPLIER,
            )
        ),
        float(
            getattr(
                args,
                "output_learning_rate_multiplier",
                DEFAULT_OUTPUT_LEARNING_RATE_MULTIPLIER,
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class LegacyBatch:
    legacy_white: Tensor
    legacy_black: Tensor
    legacy_piece_offsets: Tensor
    side_to_move: Tensor
    piece_buckets: Tensor
    rule50_count: Tensor
    scores: Tensor
    results: Tensor
    score_eligible: Tensor


@dataclass(frozen=True, slots=True)
class V2Batch:
    v2_global: Tensor
    v2_royal: Tensor
    global_offsets: Tensor
    royal_offsets: Tensor
    side_to_move: Tensor
    rule50_count: Tensor
    scores: Tensor
    results: Tensor
    score_eligible: Tensor


@dataclass(frozen=True, slots=True)
class V3Batch:
    """One V3 batch: the complete Global stream plus its phase bucket.

    ``v3_global`` carries the immutable G0 rows of each record followed by that
    record's contextual tail, in the single stream ``global_offsets`` frames.
    There is no Royal domain and no second perspective, so the batch is a strict
    subset of V2Batch plus ``phase_buckets``.
    """

    v3_global: Tensor
    global_offsets: Tensor
    side_to_move: Tensor
    phase_buckets: Tensor
    rule50_count: Tensor
    scores: Tensor
    results: Tensor
    score_eligible: Tensor


@dataclass(frozen=True, slots=True)
class DavidsonCalibration:
    parameters: Tensor


@dataclass(slots=True)
class MetricAccumulator:
    samples: int = 0
    score_eligible: int = 0
    composite_sum: float = 0.0
    score_sum: float = 0.0
    result_sum: float = 0.0
    prediction_loss_sum: float = 0.0
    prediction_draw_sum: float = 0.0
    prediction_win_sum: float = 0.0

    @classmethod
    def from_state(cls, state: dict[str, object]) -> MetricAccumulator:
        expected = {
            "samples",
            "score_eligible",
            "composite_sum",
            "score_sum",
            "result_sum",
            "prediction_loss_sum",
            "prediction_draw_sum",
            "prediction_win_sum",
        }
        _require(set(state) == expected, "checkpoint metric state is incomplete")
        return cls(
            samples=int(state["samples"]),
            score_eligible=int(state["score_eligible"]),
            composite_sum=float(state["composite_sum"]),
            score_sum=float(state["score_sum"]),
            result_sum=float(state["result_sum"]),
            prediction_loss_sum=float(state["prediction_loss_sum"]),
            prediction_draw_sum=float(state["prediction_draw_sum"]),
            prediction_win_sum=float(state["prediction_win_sum"]),
        )

    def state(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "score_eligible": self.score_eligible,
            "composite_sum": self.composite_sum,
            "score_sum": self.score_sum,
            "result_sum": self.result_sum,
            "prediction_loss_sum": self.prediction_loss_sum,
            "prediction_draw_sum": self.prediction_draw_sum,
            "prediction_win_sum": self.prediction_win_sum,
        }

    def update(
        self,
        composite: Tensor,
        score_error: Tensor,
        result_error: Tensor,
        prediction: Tensor,
        score_eligible: Tensor,
    ) -> None:
        _require(
            prediction.ndim == 2 and prediction.shape[1] == 3,
            "WDL prediction tensor is not Nx3",
        )
        eligible = score_eligible.to(dtype=score_error.dtype)
        self.samples += int(composite.numel())
        self.score_eligible += int(score_eligible.sum().detach().cpu().item())
        self.composite_sum += float(composite.sum(dtype=torch.float64).detach().cpu().item())
        self.score_sum += float(
            (score_error * eligible).sum(dtype=torch.float64).detach().cpu().item()
        )
        self.result_sum += float(result_error.sum(dtype=torch.float64).detach().cpu().item())
        prediction_sums = prediction.sum(dim=0, dtype=torch.float64).detach().cpu().tolist()
        self.prediction_loss_sum += float(prediction_sums[0])
        self.prediction_draw_sum += float(prediction_sums[1])
        self.prediction_win_sum += float(prediction_sums[2])

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
                self.prediction_loss_sum / self.samples,
                self.prediction_draw_sum / self.samples,
                self.prediction_win_sum / self.samples,
            ],
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingError(message)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _compact_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _architecture_name(args: argparse.Namespace) -> str:
    name = getattr(args, "architecture", LEGACY_ARCHITECTURE)
    _require(name in ARCHITECTURE_CHOICES, f"unknown training architecture: {name}")
    return str(name)


def _architecture_schema(name: str) -> str:
    if name == LEGACY_ARCHITECTURE:
        return ARCHITECTURE_SCHEMA
    for table in (V2_ARCHITECTURES, CONTROL_ARCHITECTURES, V3_ARCHITECTURES):
        if name in table:
            return str(table[name]["schema"])
    raise TrainingError(f"unknown training architecture: {name}")


def _training_schema(name: str) -> str:
    _require(name in ARCHITECTURE_CHOICES, f"unknown training architecture: {name}")
    if name == LEGACY_ARCHITECTURE:
        return SCHEMA
    return V3_TRAINING_SCHEMA if name in V3_ARCHITECTURES else V2_TRAINING_SCHEMA


def _checkpoint_schema(name: str) -> str:
    _require(name in ARCHITECTURE_CHOICES, f"unknown training architecture: {name}")
    if name == LEGACY_ARCHITECTURE:
        return CHECKPOINT_SCHEMA
    return V3_CHECKPOINT_SCHEMA if name in V3_ARCHITECTURES else V2_CHECKPOINT_SCHEMA


def _v2_structure(name: str) -> dict[str, object]:
    config = V2_ARCHITECTURES[name]
    first_domain = str(config["first_domain"])
    first_lanes = int(config["first_lanes"])
    global_lanes = int(config["global_lanes"])
    transformed = first_lanes + global_lanes
    if first_domain == "royal":
        first_domain_receipt: dict[str, object] = {
            "name": "royal",
            "dimensions": V2_ROYAL_DIMENSIONS,
            "lanes": first_lanes,
            "black_king_buckets": 32,
            "horizontal_mirror": "black king canonicalized to files E-H",
            "includes_black_king": False,
        }
    elif first_domain == "royal_rank8":
        first_domain_receipt = {
            "name": "royal_rank8",
            "dimensions": V2_ROYAL_RANK8_DIMENSIONS,
            "lanes": first_lanes,
            "black_king_buckets": 8,
            "bucket_map": "black king rank",
            "horizontal_mirror": "black king canonicalized to files E-H",
            "refresh_key": "black king rank plus horizontal mirror bit",
            "includes_black_king": False,
        }
    elif first_domain == "absolute_nonking":
        first_domain_receipt = {
            "name": "absolute_nonking",
            "dimensions": V2_ABSOLUTE_NONKING_DIMENSIONS,
            "lanes": first_lanes,
            "fixed_roles": 10,
            "orientation": "absolute A1-H8",
            "source_projection": "G0 rows 0-639; Black king row omitted",
            "includes_black_king": False,
        }
    else:  # pragma: no cover - static architecture table
        raise TrainingError(f"unknown V2 first domain: {first_domain}")
    structure: dict[str, object] = {
        "schema": config["schema"],
        "feature_schema": V2_FEATURE_SCHEMA,
        "feature_order": "physical squares A1 through H8",
        "domains": [
            first_domain_receipt,
            {
                "name": "global",
                "dimensions": V2_GLOBAL_DIMENSIONS,
                "lanes": global_lanes,
                "fixed_roles": 11,
                "includes_black_king": True,
            },
        ],
        "transformed_lanes": transformed,
        "dense_lanes": [transformed, HIDDEN0_LANES, HIDDEN1_LANES, 2],
        "side_to_move_heads": {"white": WHITE, "black": BLACK},
        "contextual_features": [],
        "phase_buckets": 1,
        "training_activation": "clamp(x, 0, 1)",
        "network_to_score": NNUE_TO_SCORE,
        "serialized_parameter_bytes": int(config["serialized_parameter_bytes"]),
    }
    return {
        **structure,
        "structural_sha256": _sha256_bytes(_compact_json(structure)),
    }


def _control_structure(name: str) -> dict[str, object]:
    """Structure receipt for a single-domain V2-shaped control.

    Field order follows ``_v2_structure`` so the two remain diffable; the only
    difference is that the domain list holds the Global table alone.
    """

    config = CONTROL_ARCHITECTURES[name]
    global_lanes = int(config["global_lanes"])
    structure: dict[str, object] = {
        "schema": config["schema"],
        "feature_schema": V2_FEATURE_SCHEMA,
        "feature_order": "physical squares A1 through H8",
        "domains": [
            {
                "name": "global",
                "dimensions": V2_GLOBAL_DIMENSIONS,
                "lanes": global_lanes,
                "fixed_roles": 11,
                "includes_black_king": True,
            }
        ],
        "transformed_lanes": global_lanes,
        "dense_lanes": [global_lanes, HIDDEN0_LANES, HIDDEN1_LANES, 2],
        "side_to_move_heads": {"white": WHITE, "black": BLACK},
        "contextual_features": [],
        "phase_buckets": 1,
        "training_activation": "clamp(x, 0, 1)",
        "network_to_score": NNUE_TO_SCORE,
        "serialized_parameter_bytes": int(config["serialized_parameter_bytes"]),
    }
    return {
        **structure,
        "structural_sha256": _sha256_bytes(_compact_json(structure)),
    }


def _v3_structure(name: str) -> dict[str, object]:
    """Canonical V3 receipt, taken from the model module and pinned here."""

    structure = v3_structure()
    config = V3_ARCHITECTURES[name]
    _require(
        structure["structural_sha256"] == V3_STRUCTURAL_SHA256,
        f"V3 structure receipt drifted: {structure['structural_sha256']}",
    )
    _require(
        structure["schema"] == config["schema"]
        and structure["serialized_parameter_bytes"]
        == int(config["serialized_parameter_bytes"]),
        "V3 registry entry disagrees with the canonical structure receipt",
    )
    return structure


def _architecture_structure(name: str) -> dict[str, object]:
    """Dispatch one architecture to its own structure receipt.

    The legacy control keeps its hand-written receipt in ``train``; every other
    registered architecture answers here, so no architecture can silently
    inherit another one's structural hash.
    """

    if name in V3_ARCHITECTURES:
        return _v3_structure(name)
    if name in CONTROL_ARCHITECTURES:
        return _control_structure(name)
    _require(name in V2_ARCHITECTURES, f"unknown training architecture: {name}")
    return _v2_structure(name)


def _make_model(name: str, seed: int) -> nn.Module:
    if name == LEGACY_ARCHITECTURE:
        return LegacyHPModel(seed)
    if name in V3_ARCHITECTURES:
        return HordeV3Model(seed)
    if name in CONTROL_ARCHITECTURES:
        return C0SingleG0Model(seed)
    _require(name in V2_ARCHITECTURES, f"unknown training architecture: {name}")
    config = V2_ARCHITECTURES[name]
    first_lanes = int(config["first_lanes"])
    global_lanes = int(config["global_lanes"])
    if config["first_domain"] == "royal":
        return HordeV2Model(first_lanes, global_lanes, seed)
    if config["first_domain"] == "royal_rank8":
        return RoyalRank8V2Model(first_lanes, global_lanes, seed)
    if config["first_domain"] == "absolute_nonking":
        return AbsoluteNonKingV2Model(first_lanes, global_lanes, seed)
    raise TrainingError(f"unknown V2 first domain: {config['first_domain']}")


def _splitmix64(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & U64_MASK
    value = state
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & U64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & U64_MASK
    return state, (value ^ (value >> 31)) & U64_MASK


def _shuffle(values: list[int], state: int) -> int:
    for index in range(len(values) - 1, 0, -1):
        state, value = _splitmix64(state)
        selected = value % (index + 1)
        values[index], values[selected] = values[selected], values[index]
    return state


def epoch_batches(
    record_count: int,
    batch_size: int,
    block_size: int,
    seed: int,
    epoch: int,
) -> Iterator[tuple[int, ...]]:
    """Yield one deterministic, bounded-memory permutation of all records."""

    _require(record_count > 0, "record count must be positive")
    _require(batch_size > 0, "batch size must be positive")
    _require(block_size >= batch_size, "shuffle block size is smaller than the batch size")
    _require(seed > 0, "training seed must be positive")
    _require(epoch >= 0, "epoch index must be non-negative")

    block_count = (record_count + block_size - 1) // block_size
    blocks = list(range(block_count))
    epoch_state = (
        seed
        ^ 0x484F5244455F4E4E
        ^ (((epoch + 1) * 0xD1342543DE82EF95) & U64_MASK)
    ) & U64_MASK
    epoch_state = _shuffle(blocks, epoch_state)

    pending: list[int] = []
    for block in blocks:
        begin = block * block_size
        end = min(begin + block_size, record_count)
        indices = list(range(begin, end))
        block_state = (
            epoch_state ^ (((block + 1) * 0xA24BAED4963EE407) & U64_MASK)
        ) & U64_MASK
        _shuffle(indices, block_state)
        if pending:
            needed = batch_size - len(pending)
            taken = min(needed, len(indices))
            pending.extend(indices[:taken])
            indices = indices[taken:]
            if len(pending) == batch_size:
                yield tuple(pending)
                pending = []
            elif not indices:
                continue
        full_end = len(indices) - (len(indices) % batch_size)
        for offset in range(0, full_end, batch_size):
            yield tuple(indices[offset : offset + batch_size])
        pending = indices[full_end:]
    if pending:
        yield tuple(pending)


def _schedule_sha256(batches: Sequence[Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(struct.pack("<I", len(batch)))
        for index in batch:
            digest.update(struct.pack("<Q", index))
    return digest.hexdigest().upper()


def _torch_batch(sparse: SparseBatch, device: torch.device) -> LegacyBatch:
    piece_counts = list(sparse.physical_piece_count)
    _require(all(1 <= count <= 52 for count in piece_counts), "invalid legacy piece count")
    piece_buckets = [
        min((count - 1) * LEGACY_BUCKETS // 52, LEGACY_BUCKETS - 1)
        for count in piece_counts
    ]
    scores = torch.tensor(sparse.scores, dtype=torch.float32, device=device)
    return LegacyBatch(
        legacy_white=torch.tensor(sparse.legacy_white, dtype=torch.long, device=device),
        legacy_black=torch.tensor(sparse.legacy_black, dtype=torch.long, device=device),
        legacy_piece_offsets=torch.tensor(
            sparse.legacy_piece_offsets,
            dtype=torch.long,
            device=device,
        ),
        side_to_move=torch.tensor(sparse.side_to_move, dtype=torch.long, device=device),
        piece_buckets=torch.tensor(piece_buckets, dtype=torch.long, device=device),
        rule50_count=torch.tensor(sparse.rule50_count, dtype=torch.long, device=device),
        scores=scores,
        results=torch.tensor(sparse.results, dtype=torch.long, device=device),
        score_eligible=torch.abs(scores) < MATE_SCORE_THRESHOLD,
    )


def _torch_v2_batch(sparse: SparseBatch, device: torch.device) -> V2Batch:
    piece_counts = list(sparse.physical_piece_count)
    global_counts = [
        sparse.global_offsets[index + 1] - sparse.global_offsets[index]
        for index in range(len(sparse))
    ]
    royal_counts = [
        sparse.royal_offsets[index + 1] - sparse.royal_offsets[index]
        for index in range(len(sparse))
    ]
    _require(all(1 <= count <= 52 for count in piece_counts), "invalid V2 piece count")
    _require(
        all(
            global_count >= pieces
            for global_count, pieces in zip(global_counts, piece_counts, strict=True)
        ),
        "V2 Global rows omit a physical G0 row",
    )
    _require(
        all(royal == pieces - 1 for royal, pieces in zip(royal_counts, piece_counts, strict=True)),
        "V2 Royal rows do not exclude exactly the Black king",
    )
    scores = torch.tensor(sparse.scores, dtype=torch.float32, device=device)
    return V2Batch(
        v2_global=torch.tensor(sparse.v2_global, dtype=torch.long, device=device),
        v2_royal=torch.tensor(sparse.v2_royal, dtype=torch.long, device=device),
        global_offsets=torch.tensor(sparse.global_offsets, dtype=torch.long, device=device),
        royal_offsets=torch.tensor(sparse.royal_offsets, dtype=torch.long, device=device),
        side_to_move=torch.tensor(sparse.side_to_move, dtype=torch.long, device=device),
        rule50_count=torch.tensor(sparse.rule50_count, dtype=torch.long, device=device),
        scores=scores,
        results=torch.tensor(sparse.results, dtype=torch.long, device=device),
        score_eligible=torch.abs(scores) < MATE_SCORE_THRESHOLD,
    )


def _torch_v3_batch(sparse: SparseBatch, device: torch.device) -> V3Batch:
    """Build one V3 batch from a sparse batch decoded WITH the contextual tail.

    Fails closed on every stream invariant the model relies on: the immutable
    G0 rows must be present and come first, the contextual tail must stay inside
    its reserved range, rows must be unique within a record, and the record must
    respect the registered active-row bound.
    """

    records = len(sparse)
    piece_counts = list(sparse.physical_piece_count)
    offsets = sparse.global_offsets
    rows = sparse.v2_global
    _require(all(1 <= count <= 52 for count in piece_counts), "invalid V3 piece count")
    _require(
        len(offsets) == records + 1 and offsets[0] == 0 and offsets[-1] == len(rows),
        "V3 Global offsets do not tile the row array",
    )
    for index in range(records):
        begin, end = offsets[index], offsets[index + 1]
        _require(begin <= end, f"record {sparse.record_indices[index]} has inverted V3 offsets")
        pieces = piece_counts[index]
        active = end - begin
        _require(
            active >= pieces,
            f"record {sparse.record_indices[index]} V3 stream omits a physical G0 row",
        )
        _require(
            active <= V3_MAX_ACTIVE_ROWS,
            f"record {sparse.record_indices[index]} activated {active} V3 rows, "
            f"above the registered bound of {V3_MAX_ACTIVE_ROWS}",
        )
        record_rows = rows[begin:end]
        _require(
            len(set(record_rows)) == active,
            f"record {sparse.record_indices[index]} repeats a V3 stream row",
        )
        _require(
            all(0 <= row < V2_GLOBAL_DIMENSIONS for row in record_rows[:pieces]),
            f"record {sparse.record_indices[index]} G0 prefix left the immutable rows",
        )
        _require(
            all(
                V2_GLOBAL_DIMENSIONS <= row < V3_GLOBAL_DIMENSIONS
                for row in record_rows[pieces:]
            ),
            f"record {sparse.record_indices[index]} contextual row escaped "
            f"[{V2_GLOBAL_DIMENSIONS}, {V3_GLOBAL_DIMENSIONS})",
        )
    try:
        phase_buckets = v3_phase_bucket(
            torch.tensor(sparse.white_piece_count, dtype=torch.long, device=device)
        )
    except ValueError as error:
        raise TrainingError(f"V3 phase bucket is undefined: {error}") from error
    scores = torch.tensor(sparse.scores, dtype=torch.float32, device=device)
    return V3Batch(
        v3_global=torch.tensor(rows, dtype=torch.long, device=device),
        global_offsets=torch.tensor(offsets, dtype=torch.long, device=device),
        side_to_move=torch.tensor(sparse.side_to_move, dtype=torch.long, device=device),
        phase_buckets=phase_buckets,
        rule50_count=torch.tensor(sparse.rule50_count, dtype=torch.long, device=device),
        scores=scores,
        results=torch.tensor(sparse.results, dtype=torch.long, device=device),
        score_eligible=torch.abs(scores) < MATE_SCORE_THRESHOLD,
    )


def _model_batch(
    architecture: str,
    sparse: SparseBatch,
    device: torch.device,
) -> LegacyBatch | V2Batch | V3Batch:
    if architecture == LEGACY_ARCHITECTURE:
        return _torch_batch(sparse, device)
    if architecture in V3_ARCHITECTURES:
        return _torch_v3_batch(sparse, device)
    _require(
        architecture in V2_ARCHITECTURES or architecture in CONTROL_ARCHITECTURES,
        f"unknown training architecture: {architecture}",
    )
    return _torch_v2_batch(sparse, device)


def _rule50_postprocess(output: Tensor, rule50_count: Tensor) -> Tensor:
    """Apply the engine's integer rule-50 damping with an STE gradient."""

    pre_float = output * NNUE_TO_SCORE
    pre_integer = torch.trunc(pre_float)
    pre_postprocessor = pre_float + (pre_integer - pre_float).detach()
    rule50 = torch.clamp(rule50_count, 0, 100).to(dtype=pre_float.dtype)
    damped_float = pre_postprocessor * (100.0 - rule50) / 100.0
    damped_integer = torch.trunc(damped_float)
    damped_ste = damped_float + (damped_integer - damped_float).detach()
    return torch.clamp(damped_ste, -31_506.0, 31_506.0)


def _torch_calibration(
    parameters: dict[str, tuple[float, float, float]],
    device: torch.device,
) -> DavidsonCalibration:
    ordered = [parameters[WDL_SIDE_NAMES[side]] for side in (WHITE, BLACK)]
    tensor = torch.tensor(ordered, dtype=torch.float32, device=device)
    _require(tensor.shape == (2, 3), "WDL calibration tensor is not 2x3")
    _require(bool(torch.isfinite(tensor).all().detach().cpu()), "WDL calibration is non-finite")
    _require(
        bool(torch.all(tensor[:, 0] > 0.0).detach().cpu()),
        "WDL calibration slope is not positive",
    )
    return DavidsonCalibration(tensor)


def _wdl_probabilities(
    scores: Tensor,
    side_to_move: Tensor,
    calibration: DavidsonCalibration,
) -> Tensor:
    _require(scores.ndim == 1, "WDL score tensor is not one-dimensional")
    _require(side_to_move.shape == scores.shape, "WDL side tensor shape mismatch")
    selected = calibration.parameters.index_select(0, side_to_move)
    eta = selected[:, 0] * (scores / NNUE_TO_SCORE) + selected[:, 1]
    logits = torch.stack((-eta, selected[:, 2], eta), dim=1)
    return torch.softmax(logits, dim=1)


def loss_terms(
    output: Tensor,
    batch: LegacyBatch | V2Batch | V3Batch,
    lambda_value: float,
    calibration: DavidsonCalibration,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _require(0.0 <= lambda_value <= 1.0, "lambda is outside [0, 1]")
    postprocessed_score = _rule50_postprocess(output, batch.rule50_count)
    prediction = _wdl_probabilities(
        postprocessed_score,
        batch.side_to_move,
        calibration,
    )
    teacher_scores = torch.where(
        batch.score_eligible,
        batch.scores,
        torch.zeros_like(batch.scores),
    )
    teacher = _wdl_probabilities(teacher_scores, batch.side_to_move, calibration)
    result_target = torch.nn.functional.one_hot(batch.results + 1, num_classes=3).to(
        dtype=prediction.dtype
    )
    score_error = 0.5 * torch.sum(torch.square(teacher - prediction), dim=1)
    result_error = 0.5 * torch.sum(torch.square(result_target - prediction), dim=1)
    eligible = batch.score_eligible.to(dtype=score_error.dtype)
    composite = lambda_value * eligible * score_error + (1.0 - lambda_value) * result_error
    return composite, score_error, result_error, prediction


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


def _gradient_norms(model: nn.Module) -> dict[str, float]:
    norms: dict[str, float] = {}
    gradient_groups = getattr(model, "gradient_groups", None)
    _require(callable(gradient_groups), "training model does not expose gradient groups")
    for name, parameters in gradient_groups().items():
        squared = 0.0
        for parameter in parameters:
            _require(parameter.grad is not None, f"{name} parameter has no gradient")
            gradient = parameter.grad.detach().to(dtype=torch.float64)
            squared += float(torch.sum(gradient * gradient).cpu().item())
        norm = math.sqrt(squared)
        _require(math.isfinite(norm) and norm > 0.0, f"{name} gradient is missing: {norm}")
        norms[name] = norm
    return norms


def _clip_serialized_dense_weights(model: nn.Module) -> None:
    with torch.no_grad():
        dense_limit = 127.0 / 64.0
        output_limit = (127.0 * 127.0) / 9600.0
        getattr(model, "hidden0_weights").clamp_(-dense_limit, dense_limit)
        getattr(model, "hidden1_weights").clamp_(-dense_limit, dense_limit)
        getattr(model, "output_weights").clamp_(-output_limit, output_limit)


def _all_finite(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all().detach().cpu()) for parameter in model.parameters())


def _generation_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    ignored = {"requested_records", "seed", "opening_count"}
    return {key: value for key, value in manifest["generation"].items() if key not in ignored}


def _book_split_identity(
    split_receipt_path: Path,
    train_book_sha256: str,
    validation_book_sha256: str,
) -> dict[str, object]:
    split_path = split_receipt_path.expanduser().resolve()
    split_payload = split_path.read_bytes()
    try:
        split = json.loads(split_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingError(f"book split receipt is invalid JSON: {error}") from error
    _require(isinstance(split, dict), "book split receipt root is not an object")
    _require(split.get("schema") in BOOK_SPLIT_SCHEMAS, "book split receipt schema mismatch")
    _require(split.get("disjoint_position_keys") is True, "book split is not position-disjoint")
    _require(split.get("complete_partition") is True, "book split is not a complete partition")
    split_source = split.get("source")
    split_train = split.get("train")
    split_validation = split.get("validation")
    split_assignment = split.get("assignment")
    _require(
        isinstance(split_source, dict)
        and isinstance(split_train, dict)
        and isinstance(split_validation, dict)
        and isinstance(split_assignment, dict),
        "book split receipt sections are missing",
    )
    _require(
        split_train.get("sha256") == train_book_sha256,
        "training book hash does not match the split receipt",
    )
    _require(
        split_validation.get("sha256") == validation_book_sha256,
        "validation book hash does not match the split receipt",
    )
    _require(
        type(split_source.get("records")) is int
        and type(split_train.get("records")) is int
        and type(split_validation.get("records")) is int
        and split_source["records"] == split_train["records"] + split_validation["records"],
        "book split receipt record counts do not form a complete partition",
    )
    return {
        "receipt_name": split_path.name,
        "receipt_sha256": _sha256_bytes(split_payload),
        "schema": split["schema"],
        "source": split_source,
        "assignment": split_assignment,
        "disjoint_position_keys": True,
        "complete_partition": True,
    }


def validate_dataset_pair(
    train_path: Path,
    validation_path: Path,
    train_manifest: dict[str, Any],
    validation_manifest: dict[str, Any],
    split_receipt_path: Path,
) -> dict[str, object]:
    train_resolved = train_path.expanduser().resolve()
    validation_resolved = validation_path.expanduser().resolve()
    _require(train_resolved != validation_resolved, "training and validation paths are identical")

    train_sha = _sha256_file(train_resolved)
    validation_sha = _sha256_file(validation_resolved)
    _require(train_sha != validation_sha, "training and validation files are byte-identical")
    _require(
        train_manifest["book_sha256"] != validation_manifest["book_sha256"],
        "training and validation use the same opening-book hash",
    )
    for field in ("schema", "schema_sha256", "source_commit", "source_dirty", "producer_sha256"):
        _require(
            train_manifest[field] == validation_manifest[field],
            f"training and validation manifest field {field} differs",
        )
    _require(
        train_manifest["network"] == validation_manifest["network"],
        "training and validation use different teacher networks",
    )
    _require(
        train_manifest["label_contract"] == validation_manifest["label_contract"],
        "training and validation use different label contracts",
    )
    _require(
        _generation_contract(train_manifest) == _generation_contract(validation_manifest),
        "training and validation generation settings differ",
    )

    return {
        "train_file": {
            "name": train_resolved.name,
            "sha256": train_sha,
            "payload_sha256": train_manifest["payload_sha256"],
            "records": train_manifest["record_count"],
            "book_sha256": train_manifest["book_sha256"],
            "seed": train_manifest["generation"]["seed"],
        },
        "validation_file": {
            "name": validation_resolved.name,
            "sha256": validation_sha,
            "payload_sha256": validation_manifest["payload_sha256"],
            "records": validation_manifest["record_count"],
            "book_sha256": validation_manifest["book_sha256"],
            "seed": validation_manifest["generation"]["seed"],
        },
        "teacher": {
            "source_commit": train_manifest["source_commit"],
            "producer_sha256": train_manifest["producer_sha256"],
            "network": train_manifest["network"],
            "label_contract": train_manifest["label_contract"],
            "generation": _generation_contract(train_manifest),
        },
        "book_split": _book_split_identity(
            split_receipt_path,
            train_manifest["book_sha256"],
            validation_manifest["book_sha256"],
        ),
    }


def validate_selected_dataset_pair(
    train_path: Path,
    candidate_path: Path,
    train_dataset: HordeBinV1Dataset,
    candidate_dataset: HordeBinV1Dataset,
    validation_dataset: SelectedRoleDataset,
    split_receipt_path: Path,
    *,
    allow_fixture: bool = False,
    source_override: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind an effective selected validation role to its two direct parents."""

    data = validate_dataset_pair(
        train_path,
        candidate_path,
        train_dataset.manifest,
        candidate_dataset.manifest,
        split_receipt_path,
    )
    candidate_identity = data.pop("validation_file")
    selected_identity = validate_selected_role_binding(
        train_dataset,
        candidate_dataset,
        validation_dataset,
        _allow_fixture=allow_fixture,
        _source_override=source_override,
    )
    data["validation_file"] = selected_identity
    data["validation_candidate"] = candidate_identity
    data["validation_selection"] = selected_identity["selected_role"]
    return data


def _scale_chunk_identity(dataset: HordeChunkSetDataset) -> dict[str, object]:
    identity = dataset.identity()
    return {
        "receipt_name": dataset.path.name,
        "receipt_sha256": dataset.receipt_sha256,
        "chunk_set_sha256": dataset.chunk_set_sha256,
        "logical_payload_sha256": dataset.logical_payload_sha256,
        "record_count": len(dataset),
        "book_sha256": dataset.manifest["book_sha256"],
        "role": dataset.receipt["role"],
        "sample_identity": dataset.receipt["identity"]["sample_identity"],
        "chunks": identity["chunks"],
        "manifest": dataset.manifest,
    }


def _scale_training_file_identity(dataset: HordeChunkSetDataset) -> dict[str, object]:
    return {
        "name": dataset.path.name,
        "sha256": dataset.receipt_sha256,
        "payload_sha256": dataset.logical_payload_sha256,
        "manifest_sha256": dataset.manifest_sha256,
        "records": len(dataset),
        "book_sha256": dataset.manifest["book_sha256"],
        "chunk_set_sha256": dataset.chunk_set_sha256,
        "chunks": dataset.identity()["chunks"],
    }


def validate_scale_dataset_pair(
    train_dataset: HordeChunkSetDataset,
    candidate_dataset: HordeChunkSetDataset,
    validation_dataset: ScaleSelectedRoleDataset,
    split_receipt_path: Path,
    *,
    allow_fixture: bool = False,
) -> dict[str, object]:
    """Bind the Rank8 scale train, candidate and selected validation roles."""

    _require(train_dataset.receipt["role"] == "training", "scale training role drifted")
    _require(
        candidate_dataset.receipt["role"] == "validation_candidate",
        "scale validation-candidate role drifted",
    )
    _require(
        train_dataset.chunk_set_sha256 != candidate_dataset.chunk_set_sha256,
        "scale training and candidate chunk sets match",
    )
    # HORDE_PRODUCER_SET_V1: `producer_sha256` is deliberately absent, for the
    # same reason as in the selected-role tool. It is a per-role set digest and
    # two roles can legitimately come from different builds of the generator.
    # Commit, clean-source flag, teacher network and label contract are the real
    # cross-role invariants; a difference in any of those is drift.
    for field in (
        "source_commit",
        "source_dirty",
        "network",
        "label_contract",
    ):
        _require(
            train_dataset.manifest[field] == candidate_dataset.manifest[field],
            f"scale training and candidate field {field} differs",
        )
    _require(
        _generation_contract(train_dataset.manifest)
        == _generation_contract(candidate_dataset.manifest),
        "scale training and candidate generation settings differ",
    )
    _require(
        train_dataset.manifest["book_sha256"]
        != candidate_dataset.manifest["book_sha256"],
        "scale training and candidate use the same opening book",
    )

    receipt = validation_dataset.receipt
    train_reference = _scale_chunk_identity(train_dataset)
    candidate_reference = _scale_chunk_identity(candidate_dataset)
    _require(
        receipt.get("training_reference") == train_reference,
        "scale selected role training reference drifted",
    )
    _require(
        receipt.get("candidate_source") == candidate_reference,
        "scale selected role candidate reference drifted",
    )
    selector_source = receipt.get("selector_source")
    _require(
        isinstance(selector_source, dict)
        and isinstance(selector_source.get("commit"), str)
        and len(selector_source["commit"]) == 40
        and selector_source.get("dirty") is False
        and isinstance(selector_source.get("file_sha256"), str)
        and len(selector_source["file_sha256"]) == 64,
        "scale selector source identity is invalid",
    )
    claims = receipt.get("claims", {})
    _require(
        isinstance(claims, dict)
        and claims.get("bounded_memory_exact_membership") is True
        and (
            (allow_fixture and claims.get("fixture_mode") is True)
            or (not allow_fixture and claims.get("eligible_validation_role") is True)
        ),
        "scale selected role is not training-eligible",
    )
    _require(
        receipt.get("postconditions")
        == {
            "physical_cross_role_overlap_samples": 0,
            "legacy_cross_role_overlap_samples": 0,
            "physical_validation_duplicate_samples": 0,
            "legacy_validation_duplicate_samples": 0,
        },
        "scale selected-role postconditions drifted",
    )

    selected_identity = validation_dataset.identity()
    return {
        "train_file": _scale_training_file_identity(train_dataset),
        "validation_file": selected_identity,
        "validation_candidate": _scale_training_file_identity(candidate_dataset),
        "validation_selection": selected_identity["selected_role"],
        "teacher": {
            "source_commit": train_dataset.manifest["source_commit"],
            "producer_sha256": train_dataset.manifest["producer_sha256"],
            "network": train_dataset.manifest["network"],
            "label_contract": train_dataset.manifest["label_contract"],
            "generation": _generation_contract(train_dataset.manifest),
        },
        "book_split": _book_split_identity(
            split_receipt_path,
            train_dataset.manifest["book_sha256"],
            candidate_dataset.manifest["book_sha256"],
        ),
        "decoder": {
            "train": train_dataset.identity(),
            "validation": validation_dataset.identity(),
            "mode": "authenticated chunk-set random access",
        },
        "overlap_audit": {
            "schema": "HORDE_TRAINING_ROLE_OVERLAP_AUDIT_V1",
            "source": "selected-role exact membership receipt",
            "physical": {"cross_role_overlap_samples": 0},
            "legacy_model_input": {"cross_role_overlap_samples": 0},
            "zero_cross_role_overlap": True,
        },
    }


def _repository_identity(repo_root: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    _require(len(commit) == 40, "trainer source commit is not a full Git identity")
    return {"commit": commit, "dirty": dirty}


def _campaign_identity(plan: Mapping[str, Any]) -> str:
    data = plan.get("data")
    _require(isinstance(data, dict), "campaign plan data is missing")
    payload = {
        "contract": plan.get("contract"),
        "dependencies": plan.get("dependencies"),
        "source": plan.get("source"),
        "train_file": data.get("train_file"),
        "validation_file": data.get("validation_file"),
        "validation_candidate": data.get("validation_candidate"),
        "validation_selection": data.get("validation_selection"),
        "book_split": data.get("book_split"),
        "wdl_calibration": data.get("wdl_calibration"),
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )


def _load_campaign_plan(
    args: argparse.Namespace,
    architecture: str,
    source: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, Any], str] | None:
    plan_argument = getattr(args, "campaign_plan", None)
    run_id = getattr(args, "campaign_run_id", None)
    _require(
        (plan_argument is None) == (run_id is None),
        "--campaign-plan and --campaign-run-id must be supplied together",
    )
    if plan_argument is None:
        return None
    resolved = plan_argument.expanduser().resolve()
    _require(resolved.is_file(), f"campaign plan does not exist: {resolved}")
    payload = resolved.read_bytes()
    try:
        plan = json.loads(payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingError(f"campaign plan is invalid JSON: {error}") from error
    _require(isinstance(plan, dict), "campaign plan root is not an object")
    _require(payload == _canonical_json(plan), "campaign plan is not canonical JSON")
    _require(plan.get("schema") == C1_PLAN_SCHEMA, "campaign plan schema mismatch")
    _require(plan.get("source") == source, "campaign plan source differs from trainer source")
    claims = plan.get("claims")
    _require(
        isinstance(claims, dict)
        and claims.get("fixture_mode") is False
        and claims.get("campaign_inputs_eligible") is True
        and claims.get("training_started") is False,
        "campaign plan is not eligible for a production trainer",
    )
    contract = plan.get("contract")
    _require(isinstance(contract, dict), "campaign contract identity is missing")
    addendum = contract.get("coverage_addendum")
    _require(
        contract.get("schema") == C1_PARENT_CONTRACT_SCHEMA
        and contract.get("sha256") == C1_PARENT_CONTRACT_SHA256
        and isinstance(addendum, dict)
        and addendum.get("schema") == C1_COVERAGE_ADDENDUM_SCHEMA
        and addendum.get("sha256") == C1_COVERAGE_ADDENDUM_SHA256,
        "campaign effective contract identity drifted",
    )
    effective_sha256 = _sha256_bytes(
        json.dumps(
            {
                "coverage_addendum_sha256": C1_COVERAGE_ADDENDUM_SHA256,
                "parent_contract_sha256": C1_PARENT_CONTRACT_SHA256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    _require(
        contract.get("effective_sha256") == effective_sha256,
        "campaign effective contract hash drifted",
    )
    _require(
        plan.get("campaign_identity_sha256") == _campaign_identity(plan),
        "campaign identity hash drifted",
    )
    runs = plan.get("runs")
    _require(isinstance(runs, list), "campaign run matrix is missing")
    selected = [run for run in runs if isinstance(run, dict) and run.get("id") == run_id]
    _require(len(selected) == 1, "campaign run id is absent or ambiguous")
    run = selected[0]
    run_architecture = run.get("architecture")
    _require(
        isinstance(run_architecture, dict)
        and run_architecture.get("name") == architecture
        and run.get("seed") == args.seed,
        "campaign run architecture or seed drifted",
    )
    return plan, run, _sha256_bytes(payload)


def _finalize_campaign_binding(
    bundle: tuple[dict[str, Any], dict[str, Any], str] | None,
    args: argparse.Namespace,
    architecture: str,
    data_receipt: Mapping[str, Any],
    train_records: int,
    validation_records: int,
) -> dict[str, object] | None:
    if bundle is None:
        return None
    plan, run, plan_sha256 = bundle
    planned_data = plan.get("data")
    configuration = plan.get("configuration")
    _require(
        isinstance(planned_data, dict) and isinstance(configuration, dict),
        "campaign plan sections are missing",
    )
    for key in (
        "train_file",
        "validation_file",
        "validation_candidate",
        "validation_selection",
        "teacher",
        "book_split",
        "wdl_calibration",
    ):
        _require(
            data_receipt.get(key) == planned_data.get(key),
            f"campaign trainer data field {key} differs from the plan",
        )
    dense_multiplier, output_multiplier = _optimizer_learning_rate_multipliers(args)
    _require(
        dense_multiplier == DEFAULT_DENSE_LEARNING_RATE_MULTIPLIER
        and output_multiplier == DEFAULT_OUTPUT_LEARNING_RATE_MULTIPLIER,
        "C1 campaign runs require the frozen baseline optimizer recipe",
    )
    _require(
        configuration.get("training_records") == train_records
        and configuration.get("validation_records") == validation_records
        and configuration.get("epochs") == args.epochs
        and configuration.get("batch_size") == args.batch_size
        and configuration.get("block_size") == args.block_size
        and configuration.get("lambda") == args.lambda_value
        and configuration.get("learning_rate") == args.learning_rate
        and configuration.get("scheduler_gamma") == args.scheduler_gamma,
        "campaign trainer recipe differs from the plan",
    )
    device = configuration.get("device")
    _require(
        isinstance(device, dict)
        and device.get("type") == args.device
        and device.get("cpu_threads") == args.cpu_threads,
        "campaign trainer device differs from the plan",
    )
    architecture_receipt = run.get("architecture")
    _require(
        isinstance(architecture_receipt, dict)
        and architecture_receipt.get("name") == architecture,
        "campaign run architecture identity drifted",
    )
    contract = plan["contract"]
    return {
        "schema": C1_BINDING_SCHEMA,
        "campaign_plan_sha256": plan_sha256,
        "parent_contract_sha256": C1_PARENT_CONTRACT_SHA256,
        "coverage_addendum_sha256": C1_COVERAGE_ADDENDUM_SHA256,
        "effective_contract_sha256": contract["effective_sha256"],
        "campaign_identity_sha256": plan["campaign_identity_sha256"],
        "campaign_run_id": run["id"],
    }


def _load_scale_binding(
    args: argparse.Namespace,
    architecture: str,
) -> tuple[dict[str, Any], str] | None:
    argument = getattr(args, "scale_contract", None)
    if argument is None:
        return None
    _require(
        getattr(args, "campaign_plan", None) is None
        and getattr(args, "campaign_run_id", None) is None,
        "--scale-contract cannot be combined with a C1 campaign plan",
    )
    allow_fixture = bool(getattr(args, "scale_contract_fixture", False))
    try:
        contract, digest = load_scale_contract(argument, allow_fixture=allow_fixture)
    except ScaleSelectedRoleError as error:
        raise TrainingError(f"Rank8 scale contract is invalid: {error}") from error
    training = _mapping(contract.get("training"), "scale training contract")
    architecture_contract = _mapping(
        training.get("architecture"), "scale architecture contract"
    )
    # Every scale contract is bound to exactly one architecture by the registry
    # in the selected-role tool. An architecture may never inherit another
    # contract's binding, so the contract's own schema decides what it may train.
    schema_name = str(contract.get("schema_name"))
    _require(schema_name in SCALE_CONTRACTS, f"unknown scale contract schema: {schema_name}")
    bound = str(SCALE_CONTRACTS[schema_name]["architecture"])
    _require(
        architecture == bound
        and architecture_contract.get("name") == architecture
        and architecture_contract.get("schema") == _architecture_schema(architecture),
        f"scale contract {schema_name} binds {bound}, not {architecture}",
    )
    return contract, digest


def _finalize_scale_binding(
    bundle: tuple[dict[str, Any], str] | None,
    args: argparse.Namespace,
    data_receipt: Mapping[str, Any],
    train_records: int,
    validation_records: int,
    target_steps: int,
) -> dict[str, object] | None:
    if bundle is None:
        return None
    contract, contract_sha256 = bundle
    generation = _mapping(contract.get("generation"), "scale generation")
    generation_train = _mapping(generation.get("training"), "scale training generation")
    selection = _mapping(contract.get("validation_selection"), "scale validation selection")
    recipe = _mapping(contract.get("training"), "scale training recipe")
    device = _mapping(recipe.get("device"), "scale training device")
    dense_multiplier, output_multiplier = _optimizer_learning_rate_multipliers(args)
    _require(
        train_records == generation_train.get("records")
        and validation_records == selection.get("target_records"),
        "scale dataset dimensions differ from the contract",
    )
    _require(
        args.seed == recipe.get("seed")
        and args.epochs == recipe.get("epochs")
        and train_records * args.epochs == recipe.get("training_example_exposures")
        and args.batch_size == recipe.get("batch_size")
        and args.block_size == recipe.get("block_size")
        and args.lambda_value == recipe.get("lambda")
        and args.learning_rate == recipe.get("learning_rate")
        and dense_multiplier == recipe.get("dense_learning_rate_multiplier")
        and output_multiplier == recipe.get("output_learning_rate_multiplier")
        and args.scheduler_gamma == recipe.get("scheduler_gamma")
        and target_steps == recipe.get("optimizer_steps"),
        "scale trainer recipe differs from the contract",
    )
    _require(
        recipe.get("optimizer") == "torch.optim.RAdam"
        and device.get("type") == args.device
        and device.get("cpu_threads") == args.cpu_threads,
        "scale optimizer or device differs from the contract",
    )
    if args.device == "cuda" and not bool(getattr(args, "scale_contract_fixture", False)):
        _require(
            torch.cuda.get_device_name(torch.cuda.current_device()) == device.get("expected_name"),
            "scale CUDA device differs from the contract",
        )
    checkpoints = recipe.get("checkpoint_steps")
    _require(
        isinstance(checkpoints, list)
        and all(type(step) is int and 0 < step <= target_steps for step in checkpoints),
        "scale checkpoint schedule is invalid",
    )
    _require(
        args.stop_after_steps is None
        or args.stop_after_steps == target_steps
        or args.stop_after_steps in checkpoints,
        "scale stop step is outside the frozen checkpoint schedule",
    )
    train_file = _mapping(data_receipt.get("train_file"), "scale train identity")
    validation_file = _mapping(
        data_receipt.get("validation_file"), "scale validation identity"
    )
    candidate = _mapping(
        data_receipt.get("validation_candidate"), "scale candidate identity"
    )
    return {
        "schema": SCALE_BINDING_SCHEMA,
        # The contract names itself. Stamping one registered schema on every
        # binding would record the wrong provenance the moment a second scale
        # contract exists, which is exactly what corpus A introduces.
        "contract": {
            "schema": str(contract.get("schema_name")),
            "sha256": contract_sha256,
        },
        "campaign_id": contract["openbench"]["campaign_id"],
        "cohort": contract["openbench"]["cohort"],
        "train_chunk_set_sha256": train_file["chunk_set_sha256"],
        "validation_candidate_chunk_set_sha256": candidate["chunk_set_sha256"],
        "selected_validation_receipt_sha256": validation_file["selected_role"][
            "receipt_sha256"
        ],
        "checkpoint_steps": checkpoints,
    }


def _device_receipt(device: torch.device, cpu_threads: int) -> dict[str, object]:
    receipt: dict[str, object] = {
        "type": device.type,
        "cpu_threads": cpu_threads,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        receipt.update(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            }
        )
    return receipt


def _configure_runtime(seed: int, device_name: str, cpu_threads: int) -> torch.device:
    _require(cpu_threads > 0, "CPU thread count must be positive")
    _require(device_name in ("cpu", "cuda"), "device must be cpu or cuda")
    if device_name == "cuda":
        _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if device_name == "cpu":
        torch.backends.mkldnn.enabled = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    return torch.device(device_name)


def _make_optimizer(
    model: nn.Module,
    learning_rate: float,
    *,
    dense_learning_rate_multiplier: float = DEFAULT_DENSE_LEARNING_RATE_MULTIPLIER,
    output_learning_rate_multiplier: float = DEFAULT_OUTPUT_LEARNING_RATE_MULTIPLIER,
) -> torch.optim.Optimizer:
    _require(math.isfinite(learning_rate) and learning_rate > 0.0, "learning rate must be positive")
    dense_learning_rate_multiplier, output_learning_rate_multiplier = (
        _validate_optimizer_learning_rate_multipliers(
            dense_learning_rate_multiplier,
            output_learning_rate_multiplier,
        )
    )

    output_weights = getattr(model, "output_weights")
    output_bias = getattr(model, "output_bias")
    output_ids = {id(output_weights), id(output_bias)}
    dense = [
        getattr(model, name)
        for name in (
            "hidden0_weights",
            "hidden0_bias",
            "hidden1_weights",
            "hidden1_bias",
        )
    ]
    dense_ids = {id(parameter) for parameter in dense}
    all_parameters = list(model.parameters())

    if dense_learning_rate_multiplier == DEFAULT_DENSE_LEARNING_RATE_MULTIPLIER:
        body = [parameter for parameter in all_parameters if id(parameter) not in output_ids]
        parameter_groups = [
            {"params": body, "lr": learning_rate},
            {
                "params": [output_weights, output_bias],
                "lr": learning_rate * output_learning_rate_multiplier,
            },
        ]
    else:
        body = [
            parameter
            for parameter in all_parameters
            if id(parameter) not in output_ids and id(parameter) not in dense_ids
        ]
        parameter_groups = [
            {"params": body, "lr": learning_rate},
            {
                "params": dense,
                "lr": learning_rate * dense_learning_rate_multiplier,
            },
            {
                "params": [output_weights, output_bias],
                "lr": learning_rate * output_learning_rate_multiplier,
            },
        ]

    grouped_ids = {
        id(parameter)
        for group in parameter_groups
        for parameter in group["params"]
    }
    _require(
        len(grouped_ids) == sum(len(group["params"]) for group in parameter_groups)
        and grouped_ids == {id(parameter) for parameter in all_parameters},
        "optimizer parameter groups are incomplete or overlapping",
    )
    return torch.optim.RAdam(
        parameter_groups,
        betas=(0.9, 0.999),
        eps=1.0e-7,
        weight_decay=0.0,
        foreach=False,
    )


def _dataset_payload_identity(dataset: Any) -> str:
    value = dataset.manifest.get("payload_sha256")
    if value is None:
        value = dataset.manifest.get("logical_payload_sha256")
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value),
        "dataset payload identity is invalid",
    )
    return value.upper()


def _contextual_extractor(architecture: str):
    """Return the named contextual row extractor one architecture requires.

    Only V3 extends the Global stream.  The legacy control, every V2 rung and
    the single-domain controls consume the immutable G0 rows alone, so they must
    never see a contextual tail: routing it to them would change their feature
    content without changing their name.
    """

    _require(architecture in ARCHITECTURE_CHOICES, f"unknown training architecture: {architecture}")
    return v3_contextual_rows if architecture in V3_ARCHITECTURES else None


def _load_sparse_batch(
    architecture: str,
    dataset: Any,
    indices: Sequence[int],
) -> SparseBatch:
    return make_sparse_batch(
        tuple(dataset.record(index) for index in indices),
        contextual=_contextual_extractor(architecture),
    )


def _evaluate(
    model: nn.Module,
    dataset: Any,
    batch_size: int,
    device: torch.device,
    lambda_value: float,
    calibration: DavidsonCalibration,
    architecture: str = LEGACY_ARCHITECTURE,
) -> dict[str, object]:
    metrics = MetricAccumulator()
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(dataset), batch_size):
            indices = tuple(range(begin, min(begin + batch_size, len(dataset))))
            batch = _model_batch(
                architecture,
                _load_sparse_batch(architecture, dataset, indices),
                device,
            )
            composite, score_error, result_error, prediction = loss_terms(
                model(batch), batch, lambda_value, calibration
            )
            metrics.update(
                composite, score_error, result_error, prediction, batch.score_eligible
            )
    return metrics.receipt()


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _save_checkpoint_exclusive(path: Path, checkpoint: object) -> None:
    try:
        with path.open("xb") as checkpoint_file:
            torch.save(checkpoint, checkpoint_file)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _xor_zero_to(limit: int) -> int:
    """Return 0 xor 1 xor ... xor limit for a non-negative limit."""

    if limit < 0:
        return 0
    pattern = (limit, 1, limit + 1, 0)
    return pattern[limit & 3]


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if isinstance(value, Tensor):
                state[key] = value.to(device=device)


def _rng_state(device: torch.device) -> dict[str, object]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng_state(state: dict[str, object], device: torch.device) -> None:
    _require(set(state) == {"torch_cpu", "torch_cuda"}, "checkpoint RNG state is incomplete")
    cpu = state["torch_cpu"]
    cuda = state["torch_cuda"]
    _require(isinstance(cpu, Tensor), "checkpoint CPU RNG state is invalid")
    _require(
        (device.type == "cpu" and cuda is None)
        or (device.type == "cuda" and isinstance(cuda, Tensor)),
        "checkpoint CUDA RNG state contradicts the training device",
    )
    torch.set_rng_state(cpu.cpu())
    if device.type == "cuda":
        assert isinstance(cuda, Tensor)
        torch.cuda.set_rng_state(cuda.cpu(), device=device)


def _sample_chain(previous: bytes, payload_sha256: str, indices: Sequence[int]) -> bytes:
    _require(len(previous) == 32, "sample-order chain state is invalid")
    digest = hashlib.sha256()
    digest.update(previous)
    digest.update(bytes.fromhex(payload_sha256))
    digest.update(struct.pack("<I", len(indices)))
    for index in indices:
        digest.update(struct.pack("<Q", index))
    return digest.digest()


def sample_order_chain_sha256(
    record_count: int,
    batch_size: int,
    block_size: int,
    seed: int,
    epochs: int,
    payload_sha256: str,
) -> str:
    """Reconstruct the complete deterministic training-sample order receipt."""

    _require(epochs > 0, "sample-order epoch count must be positive")
    _require(
        len(payload_sha256) == 64
        and all(character in "0123456789abcdefABCDEF" for character in payload_sha256),
        "sample-order payload identity is invalid",
    )
    chain = bytes(32)
    for epoch in range(epochs):
        for indices in epoch_batches(record_count, batch_size, block_size, seed, epoch):
            chain = _sample_chain(chain, payload_sha256, indices)
    return chain.hex().upper()


def _training_settings(
    args: argparse.Namespace,
    device: torch.device,
    wdl_calibration_sha256: str,
    architecture: str = LEGACY_ARCHITECTURE,
) -> dict[str, object]:
    settings: dict[str, object] = {
        "seed": args.seed,
        "epochs": args.epochs,
        "lambda": args.lambda_value,
        "learning_rate": args.learning_rate,
        "scheduler_gamma": args.scheduler_gamma,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "device_type": device.type,
        "cpu_threads": args.cpu_threads,
        "wdl_calibration_sha256": wdl_calibration_sha256,
        "initialization": "SHA256_NAMED_PARAMETER_SEED_V1",
    }
    dense_multiplier, output_multiplier = _optimizer_learning_rate_multipliers(args)
    if (
        dense_multiplier != DEFAULT_DENSE_LEARNING_RATE_MULTIPLIER
        or output_multiplier != DEFAULT_OUTPUT_LEARNING_RATE_MULTIPLIER
    ):
        settings["optimizer_learning_rate_multipliers"] = {
            "dense_trunk": dense_multiplier,
            "output": output_multiplier,
        }
    if architecture != LEGACY_ARCHITECTURE:
        structure = _architecture_structure(architecture)
        settings["architecture"] = {
            "name": architecture,
            "schema": structure["schema"],
            "structural_sha256": structure["structural_sha256"],
        }
    return settings


def _load_checkpoint(
    path: Path,
    architecture: str = LEGACY_ARCHITECTURE,
) -> tuple[dict[str, object], str]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"resume checkpoint does not exist: {resolved}")
    sha256 = _sha256_file(resolved)
    try:
        checkpoint = torch.load(resolved, map_location="cpu", weights_only=True)
    except (EOFError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise TrainingError(f"cannot load resume checkpoint: {error}") from error
    _require(isinstance(checkpoint, dict), "resume checkpoint root is not an object")
    _require(
        checkpoint.get("schema") == _checkpoint_schema(architecture),
        "resume checkpoint schema mismatch",
    )
    _require(
        checkpoint.get("architecture") == _architecture_schema(architecture),
        "resume checkpoint architecture mismatch",
    )
    return checkpoint, sha256


def train(args: argparse.Namespace) -> dict[str, object]:
    architecture = _architecture_name(args)
    dense_learning_rate_multiplier, output_learning_rate_multiplier = (
        _optimizer_learning_rate_multipliers(args)
    )
    _require(0 < args.seed <= U64_MASK, "training seed must be an unsigned 64-bit value")
    _require(args.epochs > 0, "epoch count must be positive")
    _require(args.batch_size > 0, "batch size must be positive")
    _require(args.block_size >= args.batch_size, "shuffle block size is too small")
    _require(0.0 <= args.lambda_value <= 1.0, "lambda is outside [0, 1]")
    _require(args.learning_rate > 0.0, "learning rate must be positive")
    _require(0.0 < args.scheduler_gamma <= 1.0, "scheduler gamma is outside (0, 1]")
    _require(
        args.stop_after_steps is None or args.stop_after_steps > 0,
        "stop-after-steps must be positive",
    )

    output = args.output.expanduser().resolve()
    _require(output.parent.is_dir(), f"output parent does not exist: {output.parent}")
    _require(not output.exists(), f"output already exists: {output}")

    repo_root = Path(__file__).resolve().parents[1]
    source = _repository_identity(repo_root)
    _require(args.allow_dirty or not source["dirty"], "trainer source tree is dirty")
    campaign_bundle = _load_campaign_plan(args, architecture, source)
    scale_bundle = _load_scale_binding(args, architecture)
    try:
        wdl_payload, wdl_parameters, wdl_sha256 = load_wdl_artifact(args.wdl_calibration)
    except CalibrationError as error:
        raise TrainingError(f"WDL calibration is invalid: {error}") from error
    device = _configure_runtime(args.seed, args.device, args.cpu_threads)
    calibration = _torch_calibration(wdl_parameters, device)
    settings = _training_settings(args, device, wdl_sha256, architecture)
    environment = {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "platform": platform.platform(),
        "device": _device_receipt(device, args.cpu_threads),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ),
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        "amp": False,
    }
    resume_checkpoint: dict[str, object] | None = None
    resume_sha256: str | None = None
    if args.resume is not None:
        resume_checkpoint, resume_sha256 = _load_checkpoint(args.resume, architecture)

    train_path = args.train.expanduser().resolve()
    validation_path = args.validation.expanduser().resolve()
    selected_validation = bool(getattr(args, "validation_selected_role", False))
    candidate_argument = getattr(args, "validation_candidate", None)
    selected_fixture = bool(getattr(args, "selected_role_fixture", False))
    scale_mode = scale_bundle is not None
    if scale_mode:
        _require(
            candidate_argument is not None,
            "Rank8 scale training requires --validation-candidate",
        )
        _require(
            not selected_validation,
            "--scale-contract already selects the scale validation role",
        )
    else:
        _require(
            selected_validation or candidate_argument is None,
            "--validation-candidate requires --validation-selected-role",
        )
        _require(
            not selected_validation or candidate_argument is not None,
            "selected validation requires --validation-candidate",
        )
    with ExitStack() as stack:
        if scale_mode:
            scale_contract_path = args.scale_contract.expanduser().resolve()
            train_dataset = stack.enter_context(
                HordeChunkSetDataset(train_path, scale_contract_path)
            )
            candidate_path = candidate_argument.expanduser().resolve()
            candidate_dataset = stack.enter_context(
                HordeChunkSetDataset(candidate_path, scale_contract_path)
            )
            validation_dataset = stack.enter_context(
                ScaleSelectedRoleDataset(
                    validation_path,
                    scale_contract_path,
                    _allow_fixture=bool(getattr(args, "scale_contract_fixture", False)),
                )
            )
            data_receipt = validate_scale_dataset_pair(
                train_dataset,
                candidate_dataset,
                validation_dataset,
                args.book_split_receipt,
                allow_fixture=bool(getattr(args, "scale_contract_fixture", False)),
            )
        else:
            train_dataset = stack.enter_context(HordeBinV1Dataset(train_path))
        if selected_validation and not scale_mode:
            candidate_path = candidate_argument.expanduser().resolve()
            candidate_dataset = stack.enter_context(HordeBinV1Dataset(candidate_path))
            validation_dataset = stack.enter_context(SelectedRoleDataset(validation_path))
            selected_source_override = getattr(args, "selected_role_source_override", source)
            if campaign_bundle is not None:
                selector_source = validation_dataset.receipt.get("selector_source")
                _require(
                    isinstance(selector_source, dict)
                    and isinstance(selector_source.get("commit"), str)
                    and selector_source.get("dirty") is False,
                    "campaign selected-role source identity is invalid",
                )
                selected_source_override = {
                    "commit": selector_source["commit"],
                    "dirty": False,
                }
            data_receipt = validate_selected_dataset_pair(
                train_path,
                candidate_path,
                train_dataset,
                candidate_dataset,
                validation_dataset,
                args.book_split_receipt,
                allow_fixture=selected_fixture,
                source_override=selected_source_override,
            )
            validation_factory = SelectedRoleDataset
        elif not scale_mode:
            validation_dataset = stack.enter_context(HordeBinV1Dataset(validation_path))
            data_receipt = validate_dataset_pair(
                train_path,
                validation_path,
                train_dataset.manifest,
                validation_dataset.manifest,
                args.book_split_receipt,
            )
            validation_factory = HordeBinV1Dataset
        if not scale_mode:
            data_receipt["decoder"] = {
                "train": dataset_receipt(train_path, args.batch_size),
                "validation": dataset_receipt(
                    validation_path,
                    args.batch_size,
                    dataset_factory=validation_factory,
                ),
            }
            data_receipt["overlap_audit"] = audit_pair(
                train_path,
                validation_path,
                example_limit=8,
                require_zero=True,
                validation_factory=validation_factory,
            )
        _require(
            args.allow_legacy_book_split_v1
            or data_receipt["book_split"]["schema"] == "HORDE_TRAINING_BOOK_SPLIT_V2",
            "legacy V1 book split requires --allow-legacy-book-split-v1",
        )
        wdl_source = wdl_payload["source"]
        _require(isinstance(wdl_source, dict), "WDL calibration source identity is missing")
        wdl_training_file = wdl_source["training_file"]
        wdl_teacher = wdl_source["teacher"]
        _require(
            wdl_training_file["sha256"] == data_receipt["train_file"]["sha256"]
            and wdl_training_file["payload_sha256"]
            == data_receipt["train_file"]["payload_sha256"]
            and wdl_training_file["manifest_sha256"] == train_dataset.manifest_sha256
            and wdl_training_file["records"] == data_receipt["train_file"]["records"],
            "WDL calibration was not fitted from the exact training dataset",
        )
        _require(
            all(data_receipt["teacher"].get(key) == value for key, value in wdl_teacher.items()),
            "WDL calibration teacher identity mismatch",
        )
        data_receipt["wdl_calibration"] = {
            "name": args.wdl_calibration.expanduser().resolve().name,
            "sha256": wdl_sha256,
            "schema": wdl_payload["schema"],
            "link_schema": wdl_payload["link"]["schema"],
            "selection_sha256": wdl_payload["selection"]["selection_sha256"],
            "eligible_records_sha256": wdl_payload["selection"]["eligible_records_sha256"],
        }

        batches_per_epoch = (len(train_dataset) + args.batch_size - 1) // args.batch_size
        target_steps = args.epochs * batches_per_epoch
        campaign_binding = _finalize_campaign_binding(
            campaign_bundle,
            args,
            architecture,
            data_receipt,
            len(train_dataset),
            len(validation_dataset),
        )
        scale_binding = _finalize_scale_binding(
            scale_bundle,
            args,
            data_receipt,
            len(train_dataset),
            len(validation_dataset),
            target_steps,
        )
        _require(
            campaign_binding is None or scale_binding is None,
            "C1 and scale campaign bindings cannot both be active",
        )
        if scale_binding is not None:
            campaign_binding = scale_binding
        stop_at_step = args.stop_after_steps or target_steps
        _require(stop_at_step <= target_steps, "stop-after-steps exceeds the target run")

        model = _make_model(architecture, args.seed).to(device)
        optimizer = _make_optimizer(
            model,
            args.learning_rate,
            dense_learning_rate_multiplier=dense_learning_rate_multiplier,
            output_learning_rate_multiplier=output_learning_rate_multiplier,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=args.scheduler_gamma
        )

        if resume_checkpoint is None:
            initial_state = _state_sha256(model)
            initial_validation = _evaluate(
                model,
                validation_dataset,
                args.batch_size,
                device,
                args.lambda_value,
                calibration,
                architecture,
            )
            gradient_norms: dict[str, float] | None = None
            epoch_receipts: list[dict[str, object]] = []
            next_epoch = 0
            next_batch = 0
            optimizer_steps = 0
            samples_consumed = 0
            train_metrics = MetricAccumulator()
            sample_order_chain = bytes(32)
        else:
            _require(resume_checkpoint.get("source") == source, "resume source identity mismatch")
            _require(
                resume_checkpoint.get("environment") == environment,
                "resume environment identity mismatch",
            )
            _require(resume_checkpoint.get("settings") == settings, "resume settings mismatch")
            _require(resume_checkpoint.get("data") == data_receipt, "resume data identity mismatch")
            _require(
                resume_checkpoint.get("campaign") == campaign_binding,
                "resume campaign binding mismatch",
            )
            model_state = resume_checkpoint.get("model_state")
            optimizer_state = resume_checkpoint.get("optimizer_state")
            scheduler_state = resume_checkpoint.get("scheduler_state")
            progress = resume_checkpoint.get("progress")
            rng_state = resume_checkpoint.get("rng_state")
            _require(isinstance(model_state, dict), "resume model state is invalid")
            _require(isinstance(optimizer_state, dict), "resume optimizer state is invalid")
            _require(isinstance(scheduler_state, dict), "resume scheduler state is invalid")
            _require(isinstance(progress, dict), "resume progress state is invalid")
            _require(isinstance(rng_state, dict), "resume RNG state is invalid")
            model.load_state_dict(model_state, strict=True)
            optimizer.load_state_dict(optimizer_state)
            _optimizer_to_device(optimizer, device)
            scheduler.load_state_dict(scheduler_state)
            _restore_rng_state(rng_state, device)

            expected_progress = {
                "next_epoch",
                "next_batch",
                "optimizer_steps",
                "samples_consumed",
                "sample_order_chain_sha256",
                "in_progress_metrics",
                "initial_state_sha256",
                "initial_validation",
                "first_step_gradient_norms",
                "epoch_receipts",
            }
            _require(set(progress) == expected_progress, "resume progress fields are incomplete")
            next_epoch = int(progress["next_epoch"])
            next_batch = int(progress["next_batch"])
            optimizer_steps = int(progress["optimizer_steps"])
            samples_consumed = int(progress["samples_consumed"])
            chain_text = progress["sample_order_chain_sha256"]
            _require(
                isinstance(chain_text, str) and len(chain_text) == 64,
                "resume sample-order chain is invalid",
            )
            sample_order_chain = bytes.fromhex(chain_text)
            metric_state = progress["in_progress_metrics"]
            _require(isinstance(metric_state, dict), "resume metric state is invalid")
            train_metrics = MetricAccumulator.from_state(metric_state)
            initial_state = str(progress["initial_state_sha256"])
            initial_validation = progress["initial_validation"]
            gradient_value = progress["first_step_gradient_norms"]
            _require(
                gradient_value is None or isinstance(gradient_value, dict),
                "resume gradient receipt is invalid",
            )
            gradient_norms = gradient_value
            epoch_value = progress["epoch_receipts"]
            _require(isinstance(epoch_value, list), "resume epoch receipts are invalid")
            epoch_receipts = epoch_value

            _require(0 <= next_epoch <= args.epochs, "resume epoch cursor is out of range")
            _require(
                0 <= next_batch < batches_per_epoch or (next_epoch == args.epochs and next_batch == 0),
                "resume batch cursor is out of range",
            )
            _require(
                optimizer_steps == next_epoch * batches_per_epoch + next_batch,
                "resume optimizer-step cursor is inconsistent",
            )
            _require(len(epoch_receipts) == next_epoch, "resume epoch receipt count is inconsistent")
            _require(
                (next_batch == 0 and train_metrics.samples == 0)
                or (next_batch > 0 and train_metrics.samples > 0),
                "resume in-progress metric cursor is inconsistent",
            )
            _require(
                optimizer_steps < stop_at_step,
                "resume checkpoint has already reached the requested stop step",
            )

        output.mkdir()
        training_stopped = False
        payload_identity = _dataset_payload_identity(train_dataset)

        for epoch in range(next_epoch, args.epochs):
            start_batch = next_batch if epoch == next_epoch else 0
            if epoch != next_epoch:
                train_metrics = MetricAccumulator()
            schedule_digest = hashlib.sha256()
            schedule_count = 0
            schedule_sum = 0
            schedule_sum_squares = 0
            schedule_xor = 0
            schedule_min = len(train_dataset)
            schedule_max = -1
            model.train()
            batches = epoch_batches(
                len(train_dataset),
                args.batch_size,
                args.block_size,
                args.seed,
                epoch,
            )
            for batch_index, indices in enumerate(batches):
                schedule_digest.update(struct.pack("<I", len(indices)))
                for index in indices:
                    schedule_digest.update(struct.pack("<Q", index))
                    schedule_count += 1
                    schedule_sum += index
                    schedule_sum_squares += index * index
                    schedule_xor ^= index
                    schedule_min = min(schedule_min, index)
                    schedule_max = max(schedule_max, index)
                if batch_index < start_batch:
                    continue

                batch = _model_batch(
                    architecture,
                    _load_sparse_batch(architecture, train_dataset, indices),
                    device,
                )
                optimizer.zero_grad(set_to_none=True)
                composite, score_error, result_error, prediction = loss_terms(
                    model(batch), batch, args.lambda_value, calibration
                )
                loss = composite.mean()
                _require(bool(torch.isfinite(loss).detach().cpu()), "training loss is non-finite")
                loss.backward()
                if optimizer_steps == 0:
                    gradient_norms = _gradient_norms(model)
                optimizer.step()
                _clip_serialized_dense_weights(model)
                _require(_all_finite(model), "model parameters became non-finite")
                train_metrics.update(
                    composite.detach(),
                    score_error.detach(),
                    result_error.detach(),
                    prediction.detach(),
                    batch.score_eligible,
                )
                optimizer_steps += 1
                samples_consumed += len(indices)
                sample_order_chain = _sample_chain(
                    sample_order_chain,
                    payload_identity,
                    indices,
                )
                next_epoch = epoch
                next_batch = batch_index + 1

                if optimizer_steps == stop_at_step and next_batch < batches_per_epoch:
                    training_stopped = True
                    break

            if training_stopped:
                break

            record_count = len(train_dataset)
            expected_sum = record_count * (record_count - 1) // 2
            expected_sum_squares = (
                record_count * (record_count - 1) * (2 * record_count - 1) // 6
            )
            _require(
                schedule_count == record_count
                and schedule_sum == expected_sum
                and schedule_sum_squares == expected_sum_squares
                and schedule_xor == _xor_zero_to(record_count - 1)
                and schedule_min == 0
                and schedule_max == record_count - 1,
                f"epoch {epoch + 1} schedule is not a complete permutation",
            )
            validation_metrics = _evaluate(
                model,
                validation_dataset,
                args.batch_size,
                device,
                args.lambda_value,
                calibration,
                architecture,
            )
            epoch_receipt = {
                "epoch": epoch + 1,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "schedule_sha256": schedule_digest.hexdigest().upper(),
                "state_sha256": _state_sha256(model),
                "train": train_metrics.receipt(),
                "validation": validation_metrics,
            }
            epoch_receipts.append(epoch_receipt)
            scheduler.step()
            next_epoch = epoch + 1
            next_batch = 0
            train_metrics = MetricAccumulator()
            if optimizer_steps == stop_at_step:
                training_stopped = True
                break

        complete = optimizer_steps == target_steps
        _require(
            training_stopped or complete,
            "training ended without reaching its stop or target step",
        )
        _require(gradient_norms is not None, "training did not execute a gradient step")
        stop_validation = _evaluate(
            model,
            validation_dataset,
            args.batch_size,
            device,
            args.lambda_value,
            calibration,
            architecture,
        )

        metrics_lines = [
            _compact_json({"epoch": 0, "validation": initial_validation}),
            *(_compact_json(receipt) for receipt in epoch_receipts),
        ]
        metrics_payload = b"\n".join(metrics_lines) + b"\n"
        metrics_path = output / "metrics.jsonl"
        _write_exclusive(metrics_path, metrics_payload)

        progress = {
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "optimizer_steps": optimizer_steps,
            "samples_consumed": samples_consumed,
            "sample_order_chain_sha256": sample_order_chain.hex().upper(),
            "in_progress_metrics": train_metrics.state(),
            "initial_state_sha256": initial_state,
            "initial_validation": initial_validation,
            "first_step_gradient_norms": gradient_norms,
            "epoch_receipts": epoch_receipts,
        }
        checkpoint = {
            "schema": _checkpoint_schema(architecture),
            "architecture": _architecture_schema(architecture),
            "source": source,
            "environment": environment,
            "model_state": _cpu_tree(model.state_dict()),
            "optimizer_state": _cpu_tree(optimizer.state_dict()),
            "scheduler_state": scheduler.state_dict(),
            "rng_state": _cpu_tree(_rng_state(device)),
            "settings": settings,
            "data": data_receipt,
            "campaign": campaign_binding,
            "progress": progress,
        }
        checkpoint_path = output / "checkpoint.pt"
        _save_checkpoint_exclusive(checkpoint_path, checkpoint)

        if architecture == LEGACY_ARCHITECTURE:
            architecture_receipt: dict[str, object] = {
                "schema": ARCHITECTURE_SCHEMA,
                "legacy_feature_schema": "HORDETEST_HP_LEGACY_V1",
                "serialized_topology": (
                    "896 -> 512 shared FT + PSQT; 8 x (1024 -> 16 -> 32 -> 1)"
                ),
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "training_only_factorizer": False,
                "initialization": "SHA256_NAMED_PARAMETER_SEED_V1",
            }
        else:
            # Every non-legacy architecture answers through the dispatcher, so
            # V3 carries its own canonical receipt and hash instead of the V2 one.
            architecture_receipt = {
                **_architecture_structure(architecture),
                "name": architecture,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "training_only_factorizer": False,
                "initialization": "SHA256_NAMED_PARAMETER_SEED_V1",
                "physical_piece_contract": "all pawns remain PAWN; roles are feature identities",
            }

        receipt = {
            "schema": _training_schema(architecture),
            "architecture": architecture_receipt,
            "source": source,
            "environment": environment,
            "data": data_receipt,
            "campaign": campaign_binding,
            "labels": {
                "score": "exact stored root-search Value from side-to-move perspective",
                "score_teacher_rule50": "already incorporated by search; never reapplied",
                "result": (
                    "terminal game result from side-to-move perspective as "
                    "loss/draw/win one-hot"
                ),
                "objective": {
                    "schema": "HORDE_WDL_HALF_BRIER_V1",
                    "score_term": "0.5 * squared L2(predicted WDL, frozen teacher-score WDL)",
                    "result_term": "0.5 * squared L2(predicted WDL, result one-hot)",
                    "reduction": (
                        "lambda * score_eligible * score_term + (1-lambda) * "
                        "result_term; mean over all records"
                    ),
                    "class_weighting": "none",
                    "resampling": "none",
                },
                "wdl_calibration": {
                    "schema": wdl_payload["schema"],
                    "link_schema": WDL_LINK_SCHEMA,
                    "artifact_sha256": wdl_sha256,
                    "parameter_storage": "IEEE-754 binary64 artifact; float32 training tensor",
                    "side_specific": True,
                    "frozen": True,
                },
                "mate_score_threshold": MATE_SCORE_THRESHOLD,
                "mate_policy": MASK_POLICY,
                "rule50": {
                    "input": "HORDE_BIN_V1 rule50_count",
                    "forward": (
                        "clamp(trunc_toward_zero(trunc_toward_zero(v0) * "
                        "(100 - min(rule50, 100)) / 100), "
                        "-31506, 31506)"
                    ),
                    "v0": "float network output multiplied by 600",
                    "gradient": "straight-through truncation estimator; exact integer forward",
                },
                "lambda": args.lambda_value,
                "network_to_score": NNUE_TO_SCORE,
            },
            "optimizer": {
                "name": "torch.optim.RAdam",
                "betas": [0.9, 0.999],
                "epsilon": 1.0e-7,
                "weight_decay": 0.0,
                "foreach": False,
                "base_learning_rate": args.learning_rate,
                "dense_learning_rate_multiplier": dense_learning_rate_multiplier,
                "output_learning_rate_multiplier": output_learning_rate_multiplier,
                "lookahead": False,
                "gradient_centralization": False,
                "scheduler": {
                    "name": "StepLR",
                    "step_size_epochs": 1,
                    "gamma": args.scheduler_gamma,
                },
                "dense_weight_clipping": {
                    "hidden": [-127.0 / 64.0, 127.0 / 64.0],
                    "output": [-(127.0 * 127.0) / 9600.0, (127.0 * 127.0) / 9600.0],
                },
            },
            "run": {
                "seed": args.seed,
                "target_epochs": args.epochs,
                "target_steps": target_steps,
                "optimizer_steps": optimizer_steps,
                "samples_consumed": samples_consumed,
                "complete": complete,
                "next_epoch": next_epoch,
                "next_batch": next_batch,
                "batch_size": args.batch_size,
                "shuffle": {
                    "schema": "SPLITMIX64_BLOCK_SHUFFLE_V1",
                    "block_size": args.block_size,
                    "complete_permutation_each_epoch": True,
                },
                "sample_order_chain_sha256": sample_order_chain.hex().upper(),
                "initial_state_sha256": initial_state,
                "final_state_sha256": _state_sha256(model),
                "first_step_gradient_norms": gradient_norms,
                "initial_validation": initial_validation,
                "stop_validation": stop_validation,
                "epochs_receipt": epoch_receipts,
                "in_progress_metrics": (
                    train_metrics.receipt() if train_metrics.samples else None
                ),
                "resume_checkpoint_sha256": resume_sha256,
            },
            "artifacts": {
                "checkpoint": {
                    "name": checkpoint_path.name,
                    "sha256": _sha256_file(checkpoint_path),
                },
                "metrics": {
                    "name": metrics_path.name,
                    "sha256": _sha256_bytes(metrics_payload),
                },
            },
            "claims": {
                "purpose": (
                    "rank8-50m-scale-training"
                    if scale_mode
                    else "real-data-integration-canary"
                ),
                "integration_only": not scale_mode,
                "strength_eligible": scale_mode and complete,
                "strength_evidence": False,
                "production_network": False,
            },
        }
        receipt_path = output / "receipt.json"
        _write_exclusive(receipt_path, _canonical_json(receipt))
        return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument(
        "--validation-selected-role",
        action="store_true",
        help="interpret validation as a HORDE_TRAINING_SELECTED_ROLE_V1 receipt",
    )
    parser.add_argument(
        "--validation-candidate",
        type=Path,
        help="direct HORDE_BIN_V1 parent of a selected validation role",
    )
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURE_CHOICES,
        default=LEGACY_ARCHITECTURE,
        help="serialized topology to train; the legacy control remains the default",
    )
    parser.add_argument("--book-split-receipt", type=Path, required=True)
    parser.add_argument("--wdl-calibration", type=Path, required=True)
    parser.add_argument(
        "--campaign-plan",
        type=Path,
        help="authenticated HORDE_V2_C1_CAMPAIGN_PLAN_V2 for a production C1 run",
    )
    parser.add_argument(
        "--campaign-run-id",
        help="exact run id selected from --campaign-plan",
    )
    parser.add_argument(
        "--scale-contract",
        type=Path,
        help=(
            "authenticate train/candidate chunk sets and the selected validation role "
            "against HORDE_V2_RANK8_SCALE_V1"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=65_536)
    parser.add_argument("--lambda", type=float, default=DEFAULT_LAMBDA, dest="lambda_value")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--dense-learning-rate-multiplier",
        type=float,
        default=DEFAULT_DENSE_LEARNING_RATE_MULTIPLIER,
        help="multiplier for hidden0/hidden1 weights and biases",
    )
    parser.add_argument(
        "--output-learning-rate-multiplier",
        type=float,
        default=DEFAULT_OUTPUT_LEARNING_RATE_MULTIPLIER,
        help="multiplier for the output weights and biases",
    )
    parser.add_argument("--scheduler-gamma", type=float, default=DEFAULT_SCHEDULER_GAMMA)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--allow-legacy-book-split-v1", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = train(args)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        RuntimeError,
        SelectedRoleError,
        ScaleSelectedRoleError,
        subprocess.SubprocessError,
        TrainingError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
