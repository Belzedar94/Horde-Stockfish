#!/usr/bin/env python3
"""Fit the frozen side-specific Horde Davidson WDL calibration artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence

try:
    from .horde_training_chunk_set import HordeChunkSetDataset
    from .horde_training_decoder import HordeBinV1Dataset
    from .horde_wdl import (
        CalibrationError,
        aggregate_labels,
        build_artifact,
        canonical_json,
    )
except ImportError:
    from horde_training_chunk_set import HordeChunkSetDataset
    from horde_training_decoder import HordeBinV1Dataset
    from horde_wdl import CalibrationError, aggregate_labels, build_artifact, canonical_json


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
    if len(commit) != 40:
        raise CalibrationError("calibration source commit is not a full Git identity")
    return {"commit": commit, "dirty": dirty}


def fit(args: argparse.Namespace) -> dict[str, object]:
    train = args.train.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not train.is_file():
        raise CalibrationError(f"training dataset does not exist: {train}")
    if not output.parent.is_dir():
        raise CalibrationError(f"output parent does not exist: {output.parent}")
    if output.exists():
        raise CalibrationError(f"output already exists: {output}")

    repo_root = Path(__file__).resolve().parents[1]
    software = _repository_identity(repo_root)
    if software["dirty"]:
        raise CalibrationError("calibration source worktree is dirty")

    chunk_set = bool(getattr(args, "chunk_set", False))
    contract = getattr(args, "contract", None)
    if chunk_set and contract is None:
        raise CalibrationError("--chunk-set requires --contract")
    if not chunk_set and contract is not None:
        raise CalibrationError("--contract requires --chunk-set")
    dataset_context = (
        HordeChunkSetDataset(train, contract.expanduser().resolve())
        if chunk_set
        else HordeBinV1Dataset(train)
    )
    with dataset_context as dataset:
        aggregated = aggregate_labels(dataset)
        manifest = dataset.manifest
        payload_sha256 = manifest.get("payload_sha256") or manifest.get(
            "logical_payload_sha256"
        )
        if not isinstance(payload_sha256, str) or len(payload_sha256) != 64:
            raise CalibrationError("training payload identity is invalid")
        source = {
            "training_file": {
                "name": train.name,
                "sha256": dataset.file_sha256,
                "payload_sha256": payload_sha256,
                "manifest_sha256": dataset.manifest_sha256,
                "records": len(dataset),
            },
            "teacher": {
                "source_commit": manifest["source_commit"],
                "producer_sha256": manifest["producer_sha256"],
                "network": manifest["network"],
                "label_contract": manifest["label_contract"],
            },
            "software": {
                **software,
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
            },
        }
    artifact = build_artifact(
        aggregated,
        source,
        minimum_class_support=args.minimum_class_support,
    )
    payload = canonical_json(artifact)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument(
        "--chunk-set",
        action="store_true",
        help="interpret train as an authenticated Horde chunk-set receipt",
    )
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-class-support", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    artifact = fit(parse_args(argv))
    print(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CalibrationError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
