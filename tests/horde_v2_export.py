#!/usr/bin/env python3
"""Checkpoint-to-container tests for the registered Horde V2 schemas."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from horde_training_models import (  # noqa: E402
    AbsoluteNonKingV2Model,
    HordeV2Model,
    RoyalRank8V2Model,
)
from horde_v2_container import SPECS, read_container  # noqa: E402
from horde_v2_export import ExportError, export_checkpoint  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def fixture(spec: object) -> tuple[dict[str, object], dict[str, object]]:
    source = {"commit": "1234567890abcdef1234567890abcdef12345678", "dirty": False}
    data = {
        "train_file": {"name": "train.bin", "sha256": "A" * 64},
        "validation_file": {"name": "validation.bin", "sha256": "B" * 64},
    }
    if spec.first_domain_name == "royal":
        model = HordeV2Model(64, 192, 0x56324558)
    elif spec.first_domain_name == "royal_rank8":
        model = RoyalRank8V2Model(64, 192, 0x56324558)
    else:
        model = AbsoluteNonKingV2Model(64, 192, 0x56324558)
    checkpoint: dict[str, object] = {
        "schema": "HORDE_V2_BASE_CHECKPOINT_V1",
        "architecture": spec.schema_name,
        "source": source,
        "settings": {
            "architecture": {
                "name": spec.architecture,
                "schema": spec.schema_name,
                "structural_sha256": spec.training_structural_sha256,
            },
            "wdl_calibration_sha256": "C" * 64,
        },
        "data": data,
        "model_state": model.state_dict(),
    }
    receipt: dict[str, object] = {
        "schema": "HORDE_V2_BASE_TRAINING_V1",
        "architecture": {
            "name": spec.architecture,
            "schema": spec.schema_name,
            "structural_sha256": spec.training_structural_sha256,
            "serialized_parameter_bytes": spec.parameter_bytes,
        },
        "source": source,
        "data": data,
        "claims": {"strength_evidence": False},
        "artifacts": {"checkpoint": {"name": "checkpoint.pt", "sha256": ""}},
    }
    return checkpoint, receipt


def write_fixture(root: Path, spec: object) -> tuple[Path, Path, dict[str, object]]:
    checkpoint, receipt = fixture(spec)
    checkpoint_path = root / f"{spec.architecture}.pt"
    torch.save(checkpoint, checkpoint_path)
    receipt["artifacts"]["checkpoint"]["sha256"] = sha256_file(checkpoint_path)
    receipt_path = root / f"{spec.architecture}.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    return checkpoint_path, receipt_path, receipt


def expect_failure(callable_object: object, needle: str) -> None:
    try:
        callable_object()
    except ExportError as error:
        if needle not in str(error):
            raise AssertionError(f"unexpected export error: {error}") from error
    else:
        raise AssertionError(f"export unexpectedly accepted invalid input: {needle}")


def main() -> int:
    torch.manual_seed(0x56324558)
    torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory(prefix="horde-v2-export-") as temporary:
        root = Path(temporary)
        hashes: dict[str, str] = {}
        for spec in SPECS:
            checkpoint_path, receipt_path, receipt = write_fixture(root, spec)
            outputs: list[bytes] = []
            for repeat in ("a", "b"):
                network_path = root / f"{spec.architecture}-{repeat}.hsv2"
                export_receipt_path = root / f"{spec.architecture}-{repeat}.receipt.json"
                exported = export_checkpoint(
                    checkpoint_path,
                    receipt_path,
                    network_path,
                    export_receipt_path,
                )
                parsed = read_container(network_path)
                assert parsed.spec == spec
                assert parsed.file_sha256 == exported["container"]["file_sha256"]
                assert parsed.parameter_sha256 == exported["container"]["parameter_sha256"]
                assert exported["claims"] == {
                    "full_refresh_container": True,
                    "incremental_eligible": False,
                    "production_dispatch": False,
                    "strength_evidence": False,
                }
                outputs.append(network_path.read_bytes())
            assert outputs[0] == outputs[1]
            hashes[spec.schema_name] = hashlib.sha256(outputs[0]).hexdigest().upper()

            bad_receipt = copy.deepcopy(receipt)
            bad_receipt["artifacts"]["checkpoint"]["sha256"] = "D" * 64
            bad_receipt_path = root / f"{spec.architecture}-bad-receipt.json"
            bad_receipt_path.write_text(json.dumps(bad_receipt), encoding="ascii")
            expect_failure(
                lambda: export_checkpoint(
                    checkpoint_path,
                    bad_receipt_path,
                    root / f"{spec.architecture}-bad.hsv2",
                    root / f"{spec.architecture}-bad.json",
                ),
                "checkpoint SHA-256",
            )

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            dirty = copy.deepcopy(checkpoint)
            dirty["source"]["dirty"] = True
            dirty_path = root / f"{spec.architecture}-dirty.pt"
            torch.save(dirty, dirty_path)
            dirty_receipt = copy.deepcopy(receipt)
            dirty_receipt["source"]["dirty"] = True
            dirty_receipt["artifacts"]["checkpoint"]["sha256"] = sha256_file(dirty_path)
            dirty_receipt_path = root / f"{spec.architecture}-dirty.json"
            dirty_receipt_path.write_text(json.dumps(dirty_receipt), encoding="ascii")
            expect_failure(
                lambda: export_checkpoint(
                    dirty_path,
                    dirty_receipt_path,
                    root / f"{spec.architecture}-dirty.hsv2",
                    root / f"{spec.architecture}-dirty-export.json",
                ),
                "dirty training source",
            )

    print(
        "Horde V2 checkpoint export passed: "
        + ", ".join(f"{schema}={digest[:12]}" for schema, digest in hashes.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
