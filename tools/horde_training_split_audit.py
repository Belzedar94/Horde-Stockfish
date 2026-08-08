#!/usr/bin/env python3
"""Audit zero physical and legacy-input overlap between Horde data roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Callable, Sequence

try:
    from .horde_training_decoder import (
        HordeBinV1Dataset,
        TrainingRecord,
        legacy_model_input_key,
        physical_position_key,
    )
except ImportError:
    from horde_training_decoder import (
        HordeBinV1Dataset,
        TrainingRecord,
        legacy_model_input_key,
        physical_position_key,
    )


SCHEMA = "HORDE_TRAINING_ROLE_OVERLAP_AUDIT_V1"


class AuditError(ValueError):
    """Raised when the dataset roles violate the overlap contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _digest(key: bytes) -> bytes:
    return hashlib.sha256(key).digest()


def _index_validation(
    dataset: HordeBinV1Dataset,
    key: Callable[[TrainingRecord], bytes],
) -> tuple[dict[bytes, int], int]:
    identities: dict[bytes, int] = {}
    duplicates = 0
    for index in range(len(dataset)):
        digest = _digest(key(dataset.record(index)))
        if digest in identities:
            duplicates += 1
        else:
            identities[digest] = index
    return identities, duplicates


def _scan_training(
    dataset: HordeBinV1Dataset,
    validation: dict[bytes, int],
    key: Callable[[TrainingRecord], bytes],
    example_limit: int,
) -> tuple[int, list[dict[str, object]]]:
    overlaps = 0
    examples: list[dict[str, object]] = []
    for index in range(len(dataset)):
        digest = _digest(key(dataset.record(index)))
        validation_index = validation.get(digest)
        if validation_index is None:
            continue
        overlaps += 1
        if len(examples) < example_limit:
            examples.append(
                {
                    "train": [dataset.manifest["payload_sha256"], index],
                    "validation": [validation_index],
                    "key_sha256": digest.hex().upper(),
                }
            )
    return overlaps, examples


def audit_pair(
    train_path: Path,
    validation_path: Path,
    *,
    example_limit: int = 8,
    require_zero: bool = True,
) -> dict[str, object]:
    _require(example_limit >= 0, "example limit must be non-negative")
    train_resolved = train_path.expanduser().resolve()
    validation_resolved = validation_path.expanduser().resolve()
    _require(train_resolved != validation_resolved, "training and validation paths are identical")

    with HordeBinV1Dataset(train_resolved) as train, HordeBinV1Dataset(
        validation_resolved
    ) as validation:
        _require(
            train.manifest["payload_sha256"] != validation.manifest["payload_sha256"],
            "training and validation payload identities are identical",
        )
        _require(
            train.manifest["book_sha256"] != validation.manifest["book_sha256"],
            "training and validation book identities are identical",
        )

        validation_physical, validation_physical_duplicates = _index_validation(
            validation, physical_position_key
        )
        validation_model, validation_model_duplicates = _index_validation(
            validation, legacy_model_input_key
        )
        physical_overlaps, physical_examples = _scan_training(
            train,
            validation_physical,
            physical_position_key,
            example_limit,
        )
        model_overlaps, model_examples = _scan_training(
            train,
            validation_model,
            legacy_model_input_key,
            example_limit,
        )

        # The validation payload is implicit in each validation sample identity
        # and is recorded once here to keep example rows compact.
        for collection in (physical_examples, model_examples):
            for example in collection:
                example["validation"].insert(0, validation.manifest["payload_sha256"])

        receipt = {
            "schema": SCHEMA,
            "roles": {
                "train": {
                    "name": train_resolved.name,
                    "file_sha256": train.file_sha256,
                    "payload_sha256": train.manifest["payload_sha256"],
                    "book_sha256": train.manifest["book_sha256"],
                    "records": len(train),
                },
                "validation": {
                    "name": validation_resolved.name,
                    "file_sha256": validation.file_sha256,
                    "payload_sha256": validation.manifest["payload_sha256"],
                    "book_sha256": validation.manifest["book_sha256"],
                    "records": len(validation),
                },
            },
            "sample_identity": "(payload_sha256, local_record_index)",
            "keys": {
                "physical": (
                    "SHA-256(board, side, castling, en-passant)"
                ),
                "legacy_model_input": (
                    "SHA-256(white rows, black rows, side, material bucket, rule50)"
                ),
                "augmentation_canonicalization": "none",
            },
            "physical": {
                "cross_role_overlap_samples": physical_overlaps,
                "validation_duplicate_samples": validation_physical_duplicates,
                "validation_unique_keys": len(validation_physical),
                "examples": physical_examples,
            },
            "legacy_model_input": {
                "cross_role_overlap_samples": model_overlaps,
                "validation_duplicate_samples": validation_model_duplicates,
                "validation_unique_keys": len(validation_model),
                "examples": model_examples,
            },
            "zero_cross_role_overlap": physical_overlaps == 0 and model_overlaps == 0,
        }
        if require_zero:
            _require(
                receipt["zero_cross_role_overlap"],
                "training and validation roles overlap; inspect the audit receipt",
            )
        return receipt


def _write_exclusive(path: Path, payload: bytes) -> None:
    resolved = path.expanduser().resolve()
    _require(resolved.parent.is_dir(), f"output parent does not exist: {resolved.parent}")
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        resolved.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("validation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--example-limit", type=int, default=8)
    parser.add_argument("--allow-overlap", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = audit_pair(
        args.train,
        args.validation,
        example_limit=args.example_limit,
        require_zero=False,
    )
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if args.output:
        _write_exclusive(args.output, payload)
    else:
        print(payload.decode("utf-8"), end="")
    if not args.allow_overlap and not receipt["zero_cross_role_overlap"]:
        print("ERROR: training and validation roles overlap", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
