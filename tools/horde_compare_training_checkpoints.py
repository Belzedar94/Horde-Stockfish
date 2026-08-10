#!/usr/bin/env python3
"""Compare Horde trainer checkpoints by every semantic field and tensor byte."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import pickle
import struct
import sys
from typing import Any, Sequence

try:
    import torch
    from torch import Tensor
except ImportError as error:  # pragma: no cover - exercised by the CLI failure path
    raise SystemExit("PyTorch is required for Horde checkpoint comparison") from error


class ComparisonError(ValueError):
    """Raised when two checkpoints are not semantically identical."""


def _encode_length(digest: Any, length: int) -> None:
    digest.update(struct.pack("<Q", length))


def _semantic_digest(value: Any, digest: Any, path: str) -> None:
    if value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B\1" if value else b"B\0")
    elif isinstance(value, int):
        encoded = str(value).encode("ascii")
        digest.update(b"I")
        _encode_length(digest, len(encoded))
        digest.update(encoded)
    elif isinstance(value, float):
        digest.update(b"F" + struct.pack("<d", value))
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S")
        _encode_length(digest, len(encoded))
        digest.update(encoded)
    elif isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T")
        dtype = str(tensor.dtype).encode("ascii")
        _encode_length(digest, len(dtype))
        digest.update(dtype)
        _encode_length(digest, tensor.ndim)
        for dimension in tensor.shape:
            _encode_length(digest, dimension)
        raw = tensor.numpy().tobytes(order="C")
        _encode_length(digest, len(raw))
        digest.update(raw)
    elif isinstance(value, dict):
        digest.update(b"D")
        ordered = sorted(value.items(), key=lambda item: (type(item[0]).__name__, repr(item[0])))
        _encode_length(digest, len(ordered))
        for key, item in ordered:
            _semantic_digest(key, digest, f"{path}.<key>")
            _semantic_digest(item, digest, f"{path}.{key}")
    elif isinstance(value, list):
        digest.update(b"L")
        _encode_length(digest, len(value))
        for index, item in enumerate(value):
            _semantic_digest(item, digest, f"{path}[{index}]")
    elif isinstance(value, tuple):
        digest.update(b"U")
        _encode_length(digest, len(value))
        for index, item in enumerate(value):
            _semantic_digest(item, digest, f"{path}[{index}]")
    else:
        raise ComparisonError(f"unsupported checkpoint value at {path}: {type(value).__name__}")


def semantic_sha256(checkpoint: object) -> str:
    digest = hashlib.sha256()
    _semantic_digest(checkpoint, digest, "checkpoint")
    return digest.hexdigest().upper()


def _compare(left: Any, right: Any, path: str) -> None:
    if type(left) is not type(right):
        raise ComparisonError(
            f"type mismatch at {path}: {type(left).__name__} != {type(right).__name__}"
        )
    if isinstance(left, Tensor):
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            raise ComparisonError(
                f"tensor metadata mismatch at {path}: "
                f"{left.dtype}{tuple(left.shape)} != {right.dtype}{tuple(right.shape)}"
            )
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            raise ComparisonError(f"tensor bytes differ at {path}")
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise ComparisonError(f"dictionary keys differ at {path}")
        for key in sorted(left, key=lambda item: (type(item).__name__, repr(item))):
            _compare(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ComparisonError(f"sequence length differs at {path}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare(left_item, right_item, f"{path}[{index}]")
    elif left != right:
        raise ComparisonError(f"value differs at {path}: {left!r} != {right!r}")


def load(path: Path) -> object:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ComparisonError(f"checkpoint does not exist: {resolved}")
    return torch.load(resolved, map_location="cpu", weights_only=True)


def compare_checkpoints(left: object, right: object) -> str:
    """Require exact semantic equality and return the shared canonical digest."""

    _compare(left, right, "checkpoint")
    left_sha = semantic_sha256(left)
    right_sha = semantic_sha256(right)
    if left_sha != right_sha:
        raise ComparisonError("semantic digests differ after exact comparison")
    return left_sha


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    left = load(args.left)
    right = load(args.right)
    semantic_sha = compare_checkpoints(left, right)
    print(f"Horde checkpoints are semantically identical: sha256={semantic_sha}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ComparisonError, OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
