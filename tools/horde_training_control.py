#!/usr/bin/env python3
"""Train the fresh legacy H/P control from disjoint HORDE_BIN_V1 files.

This is the deterministic reference trainer for the first Horde NNUE training
control.  It deliberately trains the exact serialized Run 6B topology without
the historical training-only layer factorizer.  The same optimizer, schedule,
label policy, and batching contract can therefore be reused by later
architecture ablations without giving one topology hidden parameters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
from typing import Any, Iterator, Sequence

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
        WHITE,
        make_sparse_batch,
    )
    from .horde_training_microfit import (
        EVAL_SIGMOID_SCALE,
        LEGACY_BUCKETS,
        LegacyHPModel,
        NNUE_TO_SCORE,
        OUTPUT_SIGMOID_SCALE,
    )
except ImportError:
    from horde_training_decoder import (
        BLACK,
        HordeBinV1Dataset,
        SparseBatch,
        WHITE,
        make_sparse_batch,
    )
    from horde_training_microfit import (
        EVAL_SIGMOID_SCALE,
        LEGACY_BUCKETS,
        LegacyHPModel,
        NNUE_TO_SCORE,
        OUTPUT_SIGMOID_SCALE,
    )


SCHEMA = "HORDE_FRESH_LEGACY_CONTROL_TRAINING_V1"
CHECKPOINT_SCHEMA = "HORDE_FRESH_LEGACY_CONTROL_CHECKPOINT_V1"
ARCHITECTURE_SCHEMA = "HORDETEST_HP_FRESH_CONTROL_V1"
BOOK_SPLIT_SCHEMA = "HORDE_TRAINING_BOOK_SPLIT_V1"
MATE_SCORE_THRESHOLD = 31_507  # VALUE_TB_WIN_IN_MAX_PLY at MAX_PLY=246.
DEFAULT_LAMBDA = 0.6
DEFAULT_LEARNING_RATE = 1.5e-3
DEFAULT_SCHEDULER_GAMMA = 0.987
MASK_POLICY = "exclude abs(score) >= 31507 from eval term; retain result term"
U64_MASK = (1 << 64) - 1


class TrainingError(ValueError):
    """Raised when a training input or requested run violates the contract."""


@dataclass(frozen=True, slots=True)
class LegacyBatch:
    legacy_white: Tensor
    legacy_black: Tensor
    piece_offsets: Tensor
    side_to_move: Tensor
    piece_buckets: Tensor
    scores: Tensor
    result_targets: Tensor
    eval_eligible: Tensor


@dataclass(slots=True)
class MetricAccumulator:
    samples: int = 0
    eval_eligible: int = 0
    composite_sum: float = 0.0
    eval_sum: float = 0.0
    result_sum: float = 0.0
    prediction_sum: float = 0.0

    def update(
        self,
        composite: Tensor,
        eval_error: Tensor,
        result_error: Tensor,
        prediction: Tensor,
        eval_eligible: Tensor,
    ) -> None:
        eligible = eval_eligible.to(dtype=eval_error.dtype)
        self.samples += int(composite.numel())
        self.eval_eligible += int(eval_eligible.sum().detach().cpu().item())
        self.composite_sum += float(composite.sum(dtype=torch.float64).detach().cpu().item())
        self.eval_sum += float((eval_error * eligible).sum(dtype=torch.float64).detach().cpu().item())
        self.result_sum += float(result_error.sum(dtype=torch.float64).detach().cpu().item())
        self.prediction_sum += float(prediction.sum(dtype=torch.float64).detach().cpu().item())

    def receipt(self) -> dict[str, object]:
        _require(self.samples > 0, "metric accumulator is empty")
        return {
            "samples": self.samples,
            "eval_eligible": self.eval_eligible,
            "mate_scores_masked": self.samples - self.eval_eligible,
            "composite_loss": self.composite_sum / self.samples,
            "eval_mse_eligible": (
                self.eval_sum / self.eval_eligible if self.eval_eligible else None
            ),
            "result_mse": self.result_sum / self.samples,
            "prediction_mean": self.prediction_sum / self.samples,
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingError(message)


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
    piece_counts = [
        sparse.piece_offsets[index + 1] - sparse.piece_offsets[index]
        for index in range(len(sparse))
    ]
    _require(all(1 <= count <= 52 for count in piece_counts), "invalid legacy piece count")
    piece_buckets = [
        min((count - 1) * LEGACY_BUCKETS // 52, LEGACY_BUCKETS - 1)
        for count in piece_counts
    ]
    scores = torch.tensor(sparse.scores, dtype=torch.float32, device=device)
    return LegacyBatch(
        legacy_white=torch.tensor(sparse.legacy_white, dtype=torch.long, device=device),
        legacy_black=torch.tensor(sparse.legacy_black, dtype=torch.long, device=device),
        piece_offsets=torch.tensor(sparse.piece_offsets, dtype=torch.long, device=device),
        side_to_move=torch.tensor(sparse.side_to_move, dtype=torch.long, device=device),
        piece_buckets=torch.tensor(piece_buckets, dtype=torch.long, device=device),
        scores=scores,
        result_targets=(
            torch.tensor(sparse.results, dtype=torch.float32, device=device) + 1.0
        )
        / 2.0,
        eval_eligible=torch.abs(scores) < MATE_SCORE_THRESHOLD,
    )


def loss_terms(
    output: Tensor,
    batch: LegacyBatch,
    lambda_value: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    _require(0.0 <= lambda_value <= 1.0, "lambda is outside [0, 1]")
    prediction = torch.sigmoid(output * NNUE_TO_SCORE / OUTPUT_SIGMOID_SCALE)
    eval_target = torch.sigmoid(batch.scores / EVAL_SIGMOID_SCALE)
    eval_error = torch.square(eval_target - prediction)
    result_error = torch.square(batch.result_targets - prediction)
    eligible = batch.eval_eligible.to(dtype=eval_error.dtype)
    composite = lambda_value * eligible * eval_error + (1.0 - lambda_value) * result_error
    return composite, eval_error, result_error, prediction


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


def _gradient_norms(model: LegacyHPModel) -> dict[str, float]:
    norms: dict[str, float] = {}
    for name, parameters in model.gradient_groups().items():
        squared = 0.0
        for parameter in parameters:
            _require(parameter.grad is not None, f"{name} parameter has no gradient")
            gradient = parameter.grad.detach().to(dtype=torch.float64)
            squared += float(torch.sum(gradient * gradient).cpu().item())
        norm = math.sqrt(squared)
        _require(math.isfinite(norm) and norm > 0.0, f"{name} gradient is missing: {norm}")
        norms[name] = norm
    return norms


def _clip_serialized_dense_weights(model: LegacyHPModel) -> None:
    with torch.no_grad():
        dense_limit = 127.0 / 64.0
        output_limit = (127.0 * 127.0) / 9600.0
        model.hidden0_weights.clamp_(-dense_limit, dense_limit)
        model.hidden1_weights.clamp_(-dense_limit, dense_limit)
        model.output_weights.clamp_(-output_limit, output_limit)


def _all_finite(model: nn.Module) -> bool:
    return all(bool(torch.isfinite(parameter).all().detach().cpu()) for parameter in model.parameters())


def _generation_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    ignored = {"requested_records", "seed", "opening_count"}
    return {key: value for key, value in manifest["generation"].items() if key not in ignored}


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
        _generation_contract(train_manifest) == _generation_contract(validation_manifest),
        "training and validation generation settings differ",
    )

    split_path = split_receipt_path.expanduser().resolve()
    split_payload = split_path.read_bytes()
    try:
        split = json.loads(split_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingError(f"book split receipt is invalid JSON: {error}") from error
    _require(isinstance(split, dict), "book split receipt root is not an object")
    _require(split.get("schema") == BOOK_SPLIT_SCHEMA, "book split receipt schema mismatch")
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
        split_train.get("sha256") == train_manifest["book_sha256"],
        "training book hash does not match the split receipt",
    )
    _require(
        split_validation.get("sha256") == validation_manifest["book_sha256"],
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
            "generation": _generation_contract(train_manifest),
        },
        "book_split": {
            "receipt_name": split_path.name,
            "receipt_sha256": _sha256_bytes(split_payload),
            "source": split_source,
            "assignment": split_assignment,
            "disjoint_position_keys": True,
            "complete_partition": True,
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
    if device_name == "cpu":
        torch.backends.mkldnn.enabled = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return torch.device(device_name)


def _make_optimizer(
    model: LegacyHPModel,
    learning_rate: float,
) -> torch.optim.Optimizer:
    output_ids = {id(model.output_weights), id(model.output_bias)}
    body = [parameter for parameter in model.parameters() if id(parameter) not in output_ids]
    return torch.optim.RAdam(
        [
            {"params": body, "lr": learning_rate},
            {"params": [model.output_weights, model.output_bias], "lr": learning_rate / 10.0},
        ],
        betas=(0.9, 0.999),
        eps=1.0e-7,
        weight_decay=0.0,
        foreach=False,
    )


def _load_sparse_batch(dataset: HordeBinV1Dataset, indices: Sequence[int]) -> SparseBatch:
    return make_sparse_batch(tuple(dataset.record(index) for index in indices))


def _evaluate(
    model: LegacyHPModel,
    dataset: HordeBinV1Dataset,
    batch_size: int,
    device: torch.device,
    lambda_value: float,
) -> dict[str, object]:
    metrics = MetricAccumulator()
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(dataset), batch_size):
            indices = tuple(range(begin, min(begin + batch_size, len(dataset))))
            batch = _torch_batch(_load_sparse_batch(dataset, indices), device)
            composite, eval_error, result_error, prediction = loss_terms(
                model(batch), batch, lambda_value
            )
            metrics.update(
                composite, eval_error, result_error, prediction, batch.eval_eligible
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


def train(args: argparse.Namespace) -> dict[str, object]:
    _require(args.seed > 0, "training seed must be positive")
    _require(args.epochs > 0, "epoch count must be positive")
    _require(args.batch_size > 0, "batch size must be positive")
    _require(args.block_size >= args.batch_size, "shuffle block size is too small")
    _require(0.0 <= args.lambda_value <= 1.0, "lambda is outside [0, 1]")
    _require(args.learning_rate > 0.0, "learning rate must be positive")
    _require(0.0 < args.scheduler_gamma <= 1.0, "scheduler gamma is outside (0, 1]")

    output = args.output.expanduser().resolve()
    _require(output.parent.is_dir(), f"output parent does not exist: {output.parent}")
    _require(not output.exists(), f"output already exists: {output}")
    output.mkdir()

    repo_root = Path(__file__).resolve().parents[1]
    source = _repository_identity(repo_root)
    _require(args.allow_dirty or not source["dirty"], "trainer source tree is dirty")
    device = _configure_runtime(args.seed, args.device, args.cpu_threads)

    train_path = args.train.expanduser().resolve()
    validation_path = args.validation.expanduser().resolve()
    with HordeBinV1Dataset(train_path) as train_dataset, HordeBinV1Dataset(
        validation_path
    ) as validation_dataset:
        data_receipt = validate_dataset_pair(
            train_path,
            validation_path,
            train_dataset.manifest,
            validation_dataset.manifest,
            args.book_split_receipt,
        )

        model = LegacyHPModel(args.seed).to(device)
        initial_state = _state_sha256(model)
        optimizer = _make_optimizer(model, args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=args.scheduler_gamma
        )

        metrics_path = output / "metrics.jsonl"
        initial_validation = _evaluate(
            model, validation_dataset, args.batch_size, device, args.lambda_value
        )
        metrics_lines = [
            _compact_json({"epoch": 0, "validation": initial_validation})
        ]
        gradient_norms: dict[str, float] | None = None
        epoch_receipts: list[dict[str, object]] = []

        for epoch in range(args.epochs):
            schedule_digest = hashlib.sha256()
            schedule_count = 0
            schedule_sum = 0
            schedule_sum_squares = 0
            schedule_xor = 0
            schedule_min = len(train_dataset)
            schedule_max = -1
            train_metrics = MetricAccumulator()
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
                batch = _torch_batch(_load_sparse_batch(train_dataset, indices), device)
                optimizer.zero_grad(set_to_none=True)
                composite, eval_error, result_error, prediction = loss_terms(
                    model(batch), batch, args.lambda_value
                )
                loss = composite.mean()
                _require(bool(torch.isfinite(loss).detach().cpu()), "training loss is non-finite")
                loss.backward()
                if epoch == 0 and batch_index == 0:
                    gradient_norms = _gradient_norms(model)
                optimizer.step()
                _clip_serialized_dense_weights(model)
                _require(_all_finite(model), "model parameters became non-finite")
                train_metrics.update(
                    composite.detach(),
                    eval_error.detach(),
                    result_error.detach(),
                    prediction.detach(),
                    batch.eval_eligible,
                )

            record_count = len(train_dataset)
            expected_sum = record_count * (record_count - 1) // 2
            expected_sum_squares = record_count * (record_count - 1) * (2 * record_count - 1) // 6
            _require(
                schedule_count == record_count
                and schedule_sum == expected_sum
                and schedule_sum_squares == expected_sum_squares
                and schedule_xor == _xor_zero_to(record_count - 1)
                and schedule_min == 0
                and schedule_max == record_count - 1,
                f"epoch {epoch + 1} schedule is not a complete permutation",
            )
            schedule_sha = schedule_digest.hexdigest().upper()

            validation_metrics = _evaluate(
                model, validation_dataset, args.batch_size, device, args.lambda_value
            )
            epoch_receipt = {
                "epoch": epoch + 1,
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "schedule_sha256": schedule_sha,
                "state_sha256": _state_sha256(model),
                "train": train_metrics.receipt(),
                "validation": validation_metrics,
            }
            epoch_receipts.append(epoch_receipt)
            metrics_lines.append(_compact_json(epoch_receipt))
            scheduler.step()

        _require(gradient_norms is not None, "training did not execute a gradient step")
        metrics_payload = b"\n".join(metrics_lines) + b"\n"
        _write_exclusive(metrics_path, metrics_payload)

        checkpoint = {
            "schema": CHECKPOINT_SCHEMA,
            "architecture": ARCHITECTURE_SCHEMA,
            "source": source,
            "epoch": args.epochs,
            "model_state": _cpu_tree(model.state_dict()),
            "optimizer_state": _cpu_tree(optimizer.state_dict()),
            "scheduler_state": scheduler.state_dict(),
            "settings": {
                "seed": args.seed,
                "lambda": args.lambda_value,
                "learning_rate": args.learning_rate,
                "scheduler_gamma": args.scheduler_gamma,
                "batch_size": args.batch_size,
                "block_size": args.block_size,
            },
            "data": data_receipt,
        }
        checkpoint_path = output / "checkpoint.pt"
        _save_checkpoint_exclusive(checkpoint_path, checkpoint)

        receipt = {
            "schema": SCHEMA,
            "architecture": {
                "schema": ARCHITECTURE_SCHEMA,
                "legacy_feature_schema": "HORDETEST_HP_LEGACY_V1",
                "serialized_topology": "896 -> 512 shared FT + PSQT; 8 x (1024 -> 16 -> 32 -> 1)",
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "training_only_factorizer": False,
            },
            "source": source,
            "environment": {
                "python": platform.python_version(),
                "pytorch": torch.__version__,
                "platform": platform.platform(),
                "device": _device_receipt(device, args.cpu_threads),
            },
            "data": data_receipt,
            "labels": {
                "score": "raw root-search Value from side-to-move perspective",
                "result": "terminal game result from side-to-move perspective mapped to [0, 1]",
                "mate_score_threshold": MATE_SCORE_THRESHOLD,
                "mate_policy": MASK_POLICY,
                "lambda": args.lambda_value,
                "eval_sigmoid_scale": EVAL_SIGMOID_SCALE,
                "network_to_score": NNUE_TO_SCORE,
                "output_sigmoid_scale": OUTPUT_SIGMOID_SCALE,
            },
            "optimizer": {
                "name": "torch.optim.RAdam",
                "betas": [0.9, 0.999],
                "epsilon": 1.0e-7,
                "weight_decay": 0.0,
                "base_learning_rate": args.learning_rate,
                "output_learning_rate_multiplier": 0.1,
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
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "shuffle": {
                    "schema": "SPLITMIX64_BLOCK_SHUFFLE_V1",
                    "block_size": args.block_size,
                    "complete_permutation_each_epoch": True,
                },
                "initial_state_sha256": initial_state,
                "final_state_sha256": _state_sha256(model),
                "first_step_gradient_norms": gradient_norms,
                "initial_validation": initial_validation,
                "epochs_receipt": epoch_receipts,
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
                "integration_only": True,
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
    parser.add_argument("--book-split-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=65_536)
    parser.add_argument("--lambda", type=float, default=DEFAULT_LAMBDA, dest="lambda_value")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--scheduler-gamma", type=float, default=DEFAULT_SCHEDULER_GAMMA)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
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
    except (OSError, RuntimeError, subprocess.SubprocessError, TrainingError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
