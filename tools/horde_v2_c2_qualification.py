#!/usr/bin/env python3
"""Qualify one frozen three-seed Horde V2 C2 optimizer arm."""

from __future__ import annotations

import argparse
from array import array
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
from typing import Any, Mapping, Sequence

import torch
from torch import nn

try:
    from . import horde_bin_v1 as wire
    from .horde_training_control import (
        V2_CHECKPOINT_SCHEMA,
        V2_TRAINING_SCHEMA,
        _make_model,
        _rule50_postprocess,
        _torch_v2_batch,
    )
    from .horde_training_decoder import BLACK, WHITE
    from .horde_training_selected_role import (
        CONTRACT_SHA256 as SELECTED_ROLE_CONTRACT_SHA256,
        SCHEMA as SELECTED_ROLE_SCHEMA,
        SelectedRoleDataset,
    )
    from .horde_v2_c2_constant_baseline import (
        SCHEMA as CONSTANT_RECEIPT_SCHEMA,
        validate_receipt as validate_constant_receipt,
    )
    from .horde_v2_c2_objective import (
        LAMBDA,
        LOOKUP_SCORE_MAXIMUM,
        LOOKUP_SCORE_MINIMUM,
        MATE_SCORE_THRESHOLD,
        OBJECTIVE_SCHEMA,
        FrozenWdlLookup,
        build_wdl_lookup,
        float_from_receipt,
        float_receipt,
        rule50_postprocess_constant,
    )
    from .horde_v2_functional_health import (
        CONTRACT_SCHEMA as HEALTH_CONTRACT_SCHEMA,
        CONTRACT_SHA256 as HEALTH_CONTRACT_SHA256,
        SCHEMA as HEALTH_RECEIPT_SCHEMA,
    )
    from .horde_wdl import LINK_SCHEMA as WDL_LINK_SCHEMA, SCHEMA as WDL_SCHEMA, load_artifact
except ImportError:
    import horde_bin_v1 as wire
    from horde_training_control import (
        V2_CHECKPOINT_SCHEMA,
        V2_TRAINING_SCHEMA,
        _make_model,
        _rule50_postprocess,
        _torch_v2_batch,
    )
    from horde_training_decoder import BLACK, WHITE
    from horde_training_selected_role import (
        CONTRACT_SHA256 as SELECTED_ROLE_CONTRACT_SHA256,
        SCHEMA as SELECTED_ROLE_SCHEMA,
        SelectedRoleDataset,
    )
    from horde_v2_c2_constant_baseline import (
        SCHEMA as CONSTANT_RECEIPT_SCHEMA,
        validate_receipt as validate_constant_receipt,
    )
    from horde_v2_c2_objective import (
        LAMBDA,
        LOOKUP_SCORE_MAXIMUM,
        LOOKUP_SCORE_MINIMUM,
        MATE_SCORE_THRESHOLD,
        OBJECTIVE_SCHEMA,
        FrozenWdlLookup,
        build_wdl_lookup,
        float_from_receipt,
        float_receipt,
        rule50_postprocess_constant,
    )
    from horde_v2_functional_health import (
        CONTRACT_SCHEMA as HEALTH_CONTRACT_SCHEMA,
        CONTRACT_SHA256 as HEALTH_CONTRACT_SHA256,
        SCHEMA as HEALTH_RECEIPT_SCHEMA,
    )
    from horde_wdl import LINK_SCHEMA as WDL_LINK_SCHEMA, SCHEMA as WDL_SCHEMA, load_artifact


SCHEMA = "HORDE_V2_C2_QUALIFICATION_RECEIPT_V1"
CONTRACT_SCHEMA = "HORDE_V2_C2_QUALIFICATION_V1"
CONTRACT_RELATIVE_PATH = Path("schemas/horde-v2-c2-qualification-v1.json")
CONTRACT_SHA256 = "2687A88AA7C923EA8D3EBE4165FDA302B102D85BDAD3A041BBA605A08739ED91"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SIDE_NAMES = {WHITE: "white_to_move", BLACK: "black_to_move"}
FROZEN_SEEDS = (
    7_435_908_571_601_354_096,
    3_557_647_045_056_828_427,
    4_999_335_725_889_688_378,
)
ARCHITECTURE = {
    "name": "v2-c1-abs64x192",
    "schema": "V2_C1_ABS_NONKING_64X192",
    "structural_sha256": "66538DB76A0248662A824545A2ECCD2AE84CD4805DDFCA206B5239FA3BDE45B6",
}
ARMS = {
    "dense_trunk_0p1": {"dense_trunk": 0.1, "output": 0.1},
    "output_1p0": {"dense_trunk": 1.0, "output": 1.0},
}
EPOCHS = 8
OPTIMIZER_STEPS = 496
SAMPLES_CONSUMED = 2_000_000
EVALUATION_BATCH_SIZE = 4096
BINDINGS = {
    "checkpoint_and_training_receipt": "same clean source, complete exposure, architecture, seed, optimizer arm, data receipt and checkpoint SHA-256",
    "functional_health": "exact checkpoint SHA-256, architecture, seed, final exposure and selected validation identity",
    "teacher": "constant baseline, selected validation role and every run use the same source commit, producer, Run 6B network and label contract",
    "training": "constant-baseline training identity exactly matches every checkpoint and training receipt",
    "validation": "every checkpoint, training receipt and functional-health receipt exactly match the authenticated selected-role identity",
    "wdl_calibration": "constant baseline, calibration artifact, every checkpoint and every training receipt use the same artifact identity and lookup bytes",
}
OBJECTIVE_RECEIPT = {
    "schema": OBJECTIVE_SCHEMA,
    "lambda": LAMBDA,
    "evaluation_device": "cpu",
    "checkpoint_forward_dtype": "IEEE-754 binary32",
    "prediction_score": "trunc(output * 600), integer rule-50 damping, clamp [-31506, 31506]",
    "probability_storage": "frozen IEEE-754 binary32 lookup",
    "loss_arithmetic": "IEEE-754 binary64",
    "reduction": "math.fsum in selected-role record order",
    "normalization": "all records",
    "mate_policy": "score term excluded; result term retained",
}


class QualificationError(ValueError):
    """Raised when C2 qualification evidence violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"{label} does not exist: {resolved}")
    payload = resolved.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(value, dict), f"{label} root is not an object")
    _require(payload == _canonical_json(value), f"{label} is not canonical JSON")
    return value, payload


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


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789ABCDEF" for character in value)
    )


def _valid_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _validate_training_identity(value: object, label: str) -> dict[str, object]:
    identity = _mapping(value, label)
    _require(
        set(identity) == {"name", "sha256", "payload_sha256", "manifest_sha256", "records"}
        and _nonempty_string(identity.get("name"))
        and all(
            _valid_sha256(identity.get(field))
            for field in ("sha256", "payload_sha256", "manifest_sha256")
        )
        and type(identity.get("records")) is int
        and identity["records"] > 0,
        f"{label} is invalid",
    )
    return dict(identity)


def _validate_teacher_identity(value: object, label: str) -> dict[str, object]:
    identity = _mapping(value, label)
    _require(
        set(identity) == {"source_commit", "producer_sha256", "network", "label_contract"}
        and _valid_commit(identity.get("source_commit"))
        and _valid_sha256(identity.get("producer_sha256"))
        and identity.get("network")
        == {"schema": "HORDETEST_HP_LEGACY_V1", "sha256": wire.RUN6B_SHA256}
        and identity.get("label_contract")
        == {"schema": wire.LABEL_CONTRACT_NAME, "schema_sha256": wire.LABEL_CONTRACT_SHA256},
        f"{label} is invalid",
    )
    return dict(identity)


def _validate_wdl_identity(value: object, label: str) -> dict[str, object]:
    identity = _mapping(value, label)
    _require(
        set(identity)
        == {
            "name",
            "sha256",
            "schema",
            "link_schema",
            "selection_sha256",
            "eligible_records_sha256",
        }
        and _nonempty_string(identity.get("name"))
        and identity.get("schema") == WDL_SCHEMA
        and identity.get("link_schema") == WDL_LINK_SCHEMA
        and all(
            _valid_sha256(identity.get(field))
            for field in ("sha256", "selection_sha256", "eligible_records_sha256")
        ),
        f"{label} is invalid",
    )
    return dict(identity)


def _validate_validation_identity(value: object, label: str) -> dict[str, object]:
    identity = _mapping(value, label)
    _require(
        set(identity)
        == {
            "name",
            "sha256",
            "payload_sha256",
            "book_sha256",
            "records",
            "seed",
            "selected_role",
        }
        and _nonempty_string(identity.get("name"))
        and all(
            _valid_sha256(identity.get(field))
            for field in ("sha256", "payload_sha256", "book_sha256")
        )
        and identity.get("sha256") == identity.get("payload_sha256")
        and type(identity.get("records")) is int
        and identity["records"] > 0
        and _nonempty_string(identity.get("seed")),
        f"{label} is invalid",
    )
    selected = _mapping(identity.get("selected_role"), f"{label} selected role")
    _require(
        set(selected)
        == {
            "candidate_file_sha256",
            "candidate_payload_sha256",
            "contract_sha256",
            "decision_chain_sha256",
            "receipt_name",
            "receipt_sha256",
            "record_order_sha256",
            "schema",
            "selected_index_sha256",
        }
        and selected.get("schema") == SELECTED_ROLE_SCHEMA
        and selected.get("contract_sha256") == SELECTED_ROLE_CONTRACT_SHA256
        and selected.get("receipt_name") == "receipt.json"
        and all(
            _valid_sha256(selected.get(field))
            for field in (
                "candidate_file_sha256",
                "candidate_payload_sha256",
                "contract_sha256",
                "decision_chain_sha256",
                "receipt_sha256",
                "record_order_sha256",
                "selected_index_sha256",
            )
        ),
        f"{label} selected-role identity is invalid",
    )
    return dict(identity)


def _teacher_identity_from_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    return _validate_teacher_identity(
        {
            "source_commit": manifest.get("source_commit"),
            "producer_sha256": manifest.get("producer_sha256"),
            "network": manifest.get("network"),
            "label_contract": manifest.get("label_contract"),
        },
        "selected validation teacher identity",
    )


def _validate_run_data(
    value: object,
    *,
    expected_training: Mapping[str, object],
    expected_validation: Mapping[str, object],
    expected_teacher: Mapping[str, object],
    expected_wdl: Mapping[str, object],
    label: str,
) -> Mapping[str, Any]:
    data = _mapping(value, label)
    train_file = _mapping(data.get("train_file"), f"{label} training file")
    decoder = _mapping(data.get("decoder"), f"{label} decoder")
    decoder_train = _mapping(decoder.get("train"), f"{label} train decoder")
    decoder_source = _mapping(decoder_train.get("source"), f"{label} train decoder source")
    observed_training = _validate_training_identity(
        {
            "name": train_file.get("name"),
            "sha256": train_file.get("sha256"),
            "payload_sha256": train_file.get("payload_sha256"),
            "manifest_sha256": decoder_source.get("manifest_sha256"),
            "records": train_file.get("records"),
        },
        f"{label} training identity",
    )
    _require(observed_training == dict(expected_training), f"{label} uses another training split")
    _require(
        decoder_source.get("file_sha256") == observed_training["sha256"]
        and decoder_source.get("payload_sha256") == observed_training["payload_sha256"]
        and decoder_train.get("record_count") == observed_training["records"],
        f"{label} decoder does not bind the training split",
    )

    teacher = _mapping(data.get("teacher"), f"{label} teacher")
    observed_teacher = _validate_teacher_identity(
        {
            "source_commit": teacher.get("source_commit"),
            "producer_sha256": teacher.get("producer_sha256"),
            "network": teacher.get("network"),
            "label_contract": teacher.get("label_contract"),
        },
        f"{label} teacher identity",
    )
    _require(observed_teacher == dict(expected_teacher), f"{label} uses another teacher")
    _require(
        _validate_validation_identity(data.get("validation_file"), f"{label} validation identity")
        == dict(expected_validation),
        f"{label} uses another selected validation role",
    )
    _require(
        _validate_wdl_identity(data.get("wdl_calibration"), f"{label} WDL identity")
        == dict(expected_wdl),
        f"{label} uses another WDL calibration",
    )
    return data


def _repository_identity(root: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=all"))
    _require(_valid_commit(commit), "qualification source is not a full Git identity")
    return {
        "commit": commit,
        "dirty": dirty,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "torch": torch.__version__,
        "tool": "tools/horde_v2_c2_qualification.py",
    }


def load_contract(path: Path | None = None) -> tuple[dict[str, Any], str]:
    resolved = (path or REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH).expanduser().resolve()
    contract, payload = _read_json(resolved, "C2 qualification contract")
    digest = _sha256_bytes(payload)
    _require(digest == CONTRACT_SHA256, f"C2 qualification contract SHA-256 mismatch: {digest}")
    _require(contract.get("schema_name") == CONTRACT_SCHEMA, "C2 qualification schema drifted")
    _require(contract.get("bindings") == BINDINGS, "qualification provenance bindings drifted")

    matrix = _mapping(contract.get("matrix"), "C2 qualification matrix")
    _require(matrix.get("architecture") == ARCHITECTURE, "qualification architecture drifted")
    _require(tuple(matrix.get("frozen_seeds", ())) == FROZEN_SEEDS, "qualification seeds drifted")
    _require(matrix.get("allowed_arms") == ARMS, "qualification arm matrix drifted")
    evaluation = _mapping(contract.get("evaluation"), "C2 qualification evaluation")
    _require(
        evaluation.get("objective_schema") == OBJECTIVE_SCHEMA
        and evaluation.get("lambda") == LAMBDA
        and evaluation.get("batch_size") == EVALUATION_BATCH_SIZE,
        "qualification objective drifted",
    )
    gates = _mapping(contract.get("gates"), "C2 qualification gates")
    _require(gates.get("exact_run_count") == 3, "qualification run count drifted")
    _require(
        gates.get("exact_final_exposure")
        == {
            "epochs": EPOCHS,
            "optimizer_steps": OPTIMIZER_STEPS,
            "samples_consumed": SAMPLES_CONSUMED,
        },
        "qualification exposure drifted",
    )
    statistics = _mapping(contract.get("statistics"), "C2 qualification statistics")
    _require(
        statistics
        == {
            "cluster_identity": "absent from HORDE_BIN_V1",
            "confidence_interval": "none",
            "iid_bootstrap": False,
            "sample_identity": "(payload_sha256, local_record_index)",
            "unit": "record",
            "warning": "game_ply and record adjacency must not be used to reconstruct unauthenticated game clusters",
        },
        "qualification statistical claims drifted",
    )
    return contract, digest


def _arm_name(multipliers: Mapping[str, Any]) -> str:
    _require(
        set(multipliers) == {"dense_trunk", "output"},
        "optimizer arm multiplier fields are invalid",
    )
    observed = {
        "dense_trunk": multipliers.get("dense_trunk"),
        "output": multipliers.get("output"),
    }
    for name, expected in ARMS.items():
        if observed == expected:
            return name
    raise QualificationError(f"unregistered C2 optimizer arm: {observed}")


@dataclass(slots=True)
class PreparedRun:
    directory: Path
    seed: int
    arm: str
    multipliers: dict[str, float]
    checkpoint_sha256: str
    checkpoint: dict[str, Any]
    training_receipt_sha256: str
    health_receipt_sha256: str
    model: nn.Module


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], str]:
    _require(path.is_file(), f"checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, RuntimeError, ValueError) as error:
        raise QualificationError(f"cannot load checkpoint {path}: {error}") from error
    _require(isinstance(checkpoint, dict), f"checkpoint root is not an object: {path}")
    return checkpoint, _sha256_file(path)


def _validate_source_clean(source: object, label: str) -> Mapping[str, Any]:
    mapped = _mapping(source, label)
    _require(
        set(mapped) == {"commit", "dirty"} and _valid_commit(mapped.get("commit")),
        f"{label} commit is invalid",
    )
    _require(mapped.get("dirty") is False, f"{label} is dirty")
    return mapped


def _prepare_run(
    directory: Path,
    validation_identity: Mapping[str, object],
    expected_training: Mapping[str, object],
    expected_teacher: Mapping[str, object],
    expected_wdl: Mapping[str, object],
) -> PreparedRun:
    resolved = directory.expanduser().resolve()
    _require(resolved.is_dir(), f"run directory does not exist: {resolved}")
    checkpoint_path = resolved / "checkpoint.pt"
    receipt_path = resolved / "receipt.json"
    health_path = resolved / "functional-health.json"
    checkpoint, checkpoint_sha256 = _load_checkpoint(checkpoint_path)
    receipt, receipt_payload = _read_json(receipt_path, "training receipt")
    health, health_payload = _read_json(health_path, "functional-health receipt")

    _require(checkpoint.get("schema") == V2_CHECKPOINT_SCHEMA, "checkpoint schema drifted")
    checkpoint_source = _validate_source_clean(checkpoint.get("source"), "checkpoint source")
    settings = _mapping(checkpoint.get("settings"), "checkpoint settings")
    architecture = _mapping(settings.get("architecture"), "checkpoint architecture")
    _require(dict(architecture) == ARCHITECTURE, "checkpoint architecture is not the absolute C2 control")
    seed = settings.get("seed")
    _require(type(seed) is int and seed in FROZEN_SEEDS, "checkpoint seed is outside the frozen set")
    multipliers = _mapping(
        settings.get("optimizer_learning_rate_multipliers"),
        "checkpoint optimizer multipliers",
    )
    arm = _arm_name(multipliers)
    _require(
        settings.get("epochs") == EPOCHS
        and settings.get("batch_size") == EVALUATION_BATCH_SIZE
        and settings.get("block_size") == 65_536
        and settings.get("lambda") == LAMBDA
        and settings.get("learning_rate") == 0.0015
        and settings.get("scheduler_gamma") == 0.987
        and settings.get("cpu_threads") == 1
        and settings.get("initialization") == "SHA256_NAMED_PARAMETER_SEED_V1"
        and settings.get("wdl_calibration_sha256") == expected_wdl["sha256"],
        "checkpoint training settings drifted",
    )
    checkpoint_data = _validate_run_data(
        checkpoint.get("data"),
        expected_training=expected_training,
        expected_validation=validation_identity,
        expected_teacher=expected_teacher,
        expected_wdl=expected_wdl,
        label="checkpoint data",
    )
    progress = _mapping(checkpoint.get("progress"), "checkpoint progress")
    _require(
        progress.get("next_epoch") == EPOCHS
        and progress.get("next_batch") == 0
        and progress.get("optimizer_steps") == OPTIMIZER_STEPS
        and progress.get("samples_consumed") == SAMPLES_CONSUMED
        and isinstance(progress.get("epoch_receipts"), list)
        and len(progress["epoch_receipts"]) == EPOCHS,
        "checkpoint is not the complete frozen final exposure",
    )

    _require(receipt.get("schema") == V2_TRAINING_SCHEMA, "training receipt schema drifted")
    receipt_source = _validate_source_clean(receipt.get("source"), "training receipt source")
    _require(
        dict(receipt_source) == dict(checkpoint_source),
        "checkpoint and training receipt source identities differ",
    )
    training_architecture = _mapping(receipt.get("architecture"), "training architecture")
    _require(
        {key: training_architecture.get(key) for key in ARCHITECTURE} == ARCHITECTURE,
        "training receipt architecture drifted",
    )
    artifacts = _mapping(receipt.get("artifacts"), "training artifacts")
    _require(
        _mapping(artifacts.get("checkpoint"), "training checkpoint artifact")
        == {"name": "checkpoint.pt", "sha256": checkpoint_sha256},
        "training receipt does not bind the checkpoint",
    )
    run = _mapping(receipt.get("run"), "training run")
    _require(
        run.get("complete") is True
        and run.get("seed") == seed
        and run.get("target_epochs") == EPOCHS
        and run.get("next_epoch") == EPOCHS
        and run.get("next_batch") == 0
        and run.get("target_steps") == OPTIMIZER_STEPS
        and run.get("optimizer_steps") == OPTIMIZER_STEPS
        and run.get("samples_consumed") == SAMPLES_CONSUMED
        and isinstance(run.get("epochs_receipt"), list)
        and len(run["epochs_receipt"]) == EPOCHS,
        "training receipt is not the complete frozen final exposure",
    )
    optimizer = _mapping(receipt.get("optimizer"), "training optimizer")
    _require(
        optimizer.get("dense_learning_rate_multiplier") == multipliers["dense_trunk"]
        and optimizer.get("output_learning_rate_multiplier") == multipliers["output"],
        "training optimizer multipliers differ from the checkpoint",
    )
    receipt_data = _validate_run_data(
        receipt.get("data"),
        expected_training=expected_training,
        expected_validation=validation_identity,
        expected_teacher=expected_teacher,
        expected_wdl=expected_wdl,
        label="training receipt data",
    )
    _require(
        dict(receipt_data) == dict(checkpoint_data),
        "checkpoint and training receipt data identities differ",
    )

    _require(health.get("schema") == HEALTH_RECEIPT_SCHEMA, "functional-health receipt schema drifted")
    _require(
        health.get("contract")
        == {"schema": HEALTH_CONTRACT_SCHEMA, "sha256": HEALTH_CONTRACT_SHA256},
        "functional-health contract identity drifted",
    )
    _validate_source_clean(health.get("source"), "functional-health source")
    health_checkpoint = _mapping(health.get("checkpoint"), "functional-health checkpoint")
    _require(
        health_checkpoint.get("sha256") == checkpoint_sha256
        and health_checkpoint.get("architecture") == ARCHITECTURE["name"]
        and health_checkpoint.get("architecture_schema") == ARCHITECTURE["schema"]
        and health_checkpoint.get("seed") == seed
        and health_checkpoint.get("optimizer_steps") == OPTIMIZER_STEPS
        and health_checkpoint.get("samples_consumed") == SAMPLES_CONSUMED
        and health_checkpoint.get("source") == dict(checkpoint_source),
        "functional-health receipt does not bind the final checkpoint",
    )
    _require(
        dict(_mapping(health.get("validation"), "functional-health validation"))
        == dict(validation_identity),
        "functional-health validation identity drifted",
    )
    health_gates = _mapping(health.get("gates"), "functional-health gates")
    checks = _mapping(health_gates.get("checks"), "functional-health checks")
    _require(
        health_gates.get("passed") is True
        and bool(checks)
        and all(value is True for value in checks.values()),
        "functional-health receipt did not pass every check",
    )
    health_claims = _mapping(health.get("claims"), "functional-health claims")
    _require(
        health_claims.get("functional_health_passed") is True
        and health_claims.get("strength_evidence") is False,
        "functional-health claims drifted",
    )

    model = _make_model(ARCHITECTURE["name"], seed)
    state = checkpoint.get("model_state")
    _require(isinstance(state, dict), "checkpoint model state is missing")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise QualificationError(f"checkpoint model state is incompatible: {error}") from error
    model.eval()
    return PreparedRun(
        directory=resolved,
        seed=seed,
        arm=arm,
        multipliers={key: float(multipliers[key]) for key in ("dense_trunk", "output")},
        checkpoint_sha256=checkpoint_sha256,
        checkpoint=checkpoint,
        training_receipt_sha256=_sha256_bytes(receipt_payload),
        health_receipt_sha256=_sha256_bytes(health_payload),
        model=model,
    )


def _half_brier(left: Sequence[float], right: Sequence[float]) -> float:
    return 0.5 * math.fsum((a - b) * (a - b) for a, b in zip(left, right, strict=True))


def evaluate_prediction_scores(
    lookup: FrozenWdlLookup,
    sides: Sequence[int],
    teacher_scores: Sequence[int],
    results: Sequence[int],
    predicted_scores: Sequence[int],
) -> dict[str, object]:
    """Evaluate one integer prediction vector with the frozen record-order objective."""

    count = len(sides)
    _require(count > 0, "qualification prediction vector is empty")
    _require(
        len(teacher_scores) == count
        and len(results) == count
        and len(predicted_scores) == count,
        "qualification prediction and label lengths differ",
    )
    composite_values: list[float] = []
    score_values: list[float] = []
    result_values: list[float] = []
    composite_by_side = {WHITE: [], BLACK: []}
    score_by_side = {WHITE: [], BLACK: []}
    result_by_side = {WHITE: [], BLACK: []}
    records_by_side = {WHITE: 0, BLACK: 0}
    eligible_by_side = {WHITE: 0, BLACK: 0}
    unique_by_side = {WHITE: set(), BLACK: set()}
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", count))

    for index in range(count):
        side = int(sides[index])
        teacher_score = int(teacher_scores[index])
        result = int(results[index])
        predicted_score = int(predicted_scores[index])
        _require(side in (WHITE, BLACK), f"invalid side at record {index}")
        _require(result in (-1, 0, 1), f"invalid result at record {index}")
        _require(
            LOOKUP_SCORE_MINIMUM <= predicted_score <= LOOKUP_SCORE_MAXIMUM,
            f"prediction escapes the canonical score domain at record {index}",
        )
        eligible = abs(teacher_score) < MATE_SCORE_THRESHOLD
        if eligible:
            _require(
                LOOKUP_SCORE_MINIMUM <= teacher_score <= LOOKUP_SCORE_MAXIMUM,
                f"eligible teacher score escapes the lookup at record {index}",
            )
        prediction = lookup.probabilities(side, predicted_score)
        score_loss = (
            _half_brier(lookup.probabilities(side, teacher_score), prediction)
            if eligible
            else 0.0
        )
        result_target = tuple(1.0 if lane == result + 1 else 0.0 for lane in range(3))
        result_loss = _half_brier(result_target, prediction)
        composite = math.fsum((LAMBDA * score_loss, (1.0 - LAMBDA) * result_loss))
        _require(
            all(math.isfinite(value) and value >= 0.0 for value in (score_loss, result_loss, composite)),
            f"non-finite qualification loss at record {index}",
        )
        composite_values.append(composite)
        score_values.append(score_loss)
        result_values.append(result_loss)
        composite_by_side[side].append(composite)
        score_by_side[side].append(score_loss)
        result_by_side[side].append(result_loss)
        records_by_side[side] += 1
        eligible_by_side[side] += int(eligible)
        unique_by_side[side].add(predicted_score)
        digest.update(struct.pack("<QBhhbB", index, side, predicted_score, teacher_score, result, int(eligible)))

    _require(all(records_by_side[side] > 0 for side in (WHITE, BLACK)), "qualification role lacks one side to move")
    composite_sum = math.fsum(composite_values)
    score_sum = math.fsum(score_values)
    result_sum = math.fsum(result_values)

    def side_receipt(side: int) -> dict[str, object]:
        records = records_by_side[side]
        return {
            "records": records,
            "eligible_teacher_scores": eligible_by_side[side],
            "unique_predicted_integer_scores": len(unique_by_side[side]),
            "composite_loss_sum": float_receipt(math.fsum(composite_by_side[side])),
            "composite_loss_mean_all_side_records": float_receipt(
                math.fsum(composite_by_side[side]) / records
            ),
            "score_half_brier_sum": float_receipt(math.fsum(score_by_side[side])),
            "result_half_brier_sum": float_receipt(math.fsum(result_by_side[side])),
        }

    return {
        "records": count,
        "eligible_teacher_scores": sum(eligible_by_side.values()),
        "prediction_and_label_chain_sha256": digest.hexdigest().upper(),
        "composite_loss_sum": float_receipt(composite_sum),
        "composite_loss_mean_all_records": float_receipt(composite_sum / count),
        "score_half_brier_sum": float_receipt(score_sum),
        "result_half_brier_sum": float_receipt(result_sum),
        "sides": {SIDE_NAMES[side]: side_receipt(side) for side in (WHITE, BLACK)},
    }


def _collect_predictions(
    dataset: SelectedRoleDataset,
    constants: Mapping[int, int],
    runs: Sequence[PreparedRun],
) -> tuple[array, array, array, array, dict[int, array]]:
    sides = array("b")
    teacher_scores = array("h")
    results = array("b")
    baseline_scores = array("h")
    candidate_scores = {run.seed: array("h") for run in runs}
    device = torch.device("cpu")

    for sparse in dataset.batches(EVALUATION_BATCH_SIZE):
        sides.extend(int(value) for value in sparse.side_to_move)
        teacher_scores.extend(int(value) for value in sparse.scores)
        results.extend(int(value) for value in sparse.results)
        baseline_scores.extend(
            rule50_postprocess_constant(constants[int(side)], int(clock))
            for side, clock in zip(sparse.side_to_move, sparse.rule50_count, strict=True)
        )
        batch = _torch_v2_batch(sparse, device)
        with torch.no_grad():
            for run in runs:
                postprocessed = _rule50_postprocess(run.model(batch), batch.rule50_count)
                _require(bool(torch.isfinite(postprocessed).all()), "checkpoint produced non-finite scores")
                _require(
                    bool(torch.equal(postprocessed, torch.trunc(postprocessed))),
                    "checkpoint post-rule50 scores are not integers",
                )
                values = postprocessed.to(dtype=torch.int32, device="cpu").tolist()
                _require(
                    all(LOOKUP_SCORE_MINIMUM <= int(value) <= LOOKUP_SCORE_MAXIMUM for value in values),
                    "checkpoint score escaped the canonical domain",
                )
                candidate_scores[run.seed].extend(int(value) for value in values)

    _require(len(sides) == len(dataset), "qualification did not consume the full selected role")
    _require(
        len(teacher_scores) == len(sides)
        and len(results) == len(sides)
        and len(baseline_scores) == len(sides)
        and all(len(values) == len(sides) for values in candidate_scores.values()),
        "qualification prediction accounting drifted",
    )
    return sides, teacher_scores, results, baseline_scores, candidate_scores


def _constant_inputs(receipt: Mapping[str, Any]) -> dict[int, int]:
    fit = _mapping(receipt.get("fit"), "constant fit")
    sides = _mapping(fit.get("sides"), "constant sides")
    constants = {
        side: _mapping(sides.get(name), f"constant {name}").get("selected_constant_cp")
        for side, name in SIDE_NAMES.items()
    }
    _require(
        all(type(value) is int for value in constants.values()),
        "constant receipt does not contain two integer constants",
    )
    return {side: int(value) for side, value in constants.items()}


def build_receipt(
    constant_path: Path,
    validation_path: Path,
    wdl_path: Path,
    run_directories: Sequence[Path],
    *,
    contract_path: Path | None = None,
    allow_dirty: bool = False,
) -> dict[str, object]:
    contract, contract_sha256 = load_contract(contract_path)
    source = _repository_identity(REPOSITORY_ROOT)
    _require(allow_dirty or source["dirty"] is False, "qualification source tree is dirty")
    _require(len(run_directories) == 3, "qualification requires exactly three run directories")

    constant, constant_payload = _read_json(constant_path, "constant-baseline receipt")
    _require(constant.get("schema") == CONSTANT_RECEIPT_SCHEMA, "constant receipt schema drifted")
    try:
        validate_constant_receipt(constant)
    except ValueError as error:
        raise QualificationError(f"constant-baseline receipt is invalid: {error}") from error
    constant_source = _mapping(constant.get("source"), "constant source")
    expected_training = _validate_training_identity(
        constant_source.get("training_file"), "constant training identity"
    )
    expected_teacher = _validate_teacher_identity(
        constant_source.get("teacher"), "constant teacher identity"
    )
    expected_wdl = _validate_wdl_identity(
        constant_source.get("wdl_calibration"), "constant WDL identity"
    )
    wdl_artifact, parameters, wdl_sha256 = load_artifact(wdl_path.expanduser().resolve())
    _require(wdl_artifact.get("schema") == WDL_SCHEMA, "WDL artifact schema drifted")
    _require(expected_wdl["sha256"] == wdl_sha256, "constant receipt uses another WDL calibration")
    wdl_source = _mapping(wdl_artifact.get("source"), "WDL source")
    _require(
        _validate_training_identity(wdl_source.get("training_file"), "WDL training identity")
        == expected_training
        and _validate_teacher_identity(wdl_source.get("teacher"), "WDL teacher identity")
        == expected_teacher,
        "WDL artifact provenance differs from the constant baseline",
    )
    wdl_selection = _mapping(wdl_artifact.get("selection"), "WDL selection")
    _require(
        wdl_selection.get("selection_sha256") == expected_wdl["selection_sha256"]
        and wdl_selection.get("eligible_records_sha256")
        == expected_wdl["eligible_records_sha256"]
        and _mapping(wdl_artifact.get("link"), "WDL link").get("schema")
        == expected_wdl["link_schema"],
        "WDL artifact identity differs from the constant baseline",
    )
    lookup = build_wdl_lookup(parameters)
    constant_lookup = _mapping(
        _mapping(constant.get("objective"), "constant objective").get("lookup"),
        "constant lookup",
    )
    _require(
        constant_lookup.get("raw_float32_sha256") == lookup.raw_float32_sha256
        and constant_lookup.get("parameter_float32_sha256") == lookup.parameter_float32_sha256,
        "constant receipt lookup differs from the qualification lookup",
    )

    validation_resolved = validation_path.expanduser().resolve()
    with SelectedRoleDataset(validation_resolved) as dataset:
        validation_identity = _validate_validation_identity(
            dataset.identity(), "qualification validation identity"
        )
        _require(
            dataset.manifest.get("source_dirty") is False
            and _teacher_identity_from_manifest(dataset.manifest) == expected_teacher,
            "qualification validation teacher differs from the constant baseline",
        )
        runs = [
            _prepare_run(
                path,
                validation_identity,
                expected_training,
                expected_teacher,
                expected_wdl,
            )
            for path in run_directories
        ]
        _require(
            tuple(sorted(run.seed for run in runs)) == tuple(sorted(FROZEN_SEEDS)),
            "qualification seed set is incomplete",
        )
        _require(len({run.seed for run in runs}) == 3, "qualification seed is duplicated")
        _require(len({run.arm for run in runs}) == 1, "qualification mixed optimizer arms")
        arm = runs[0].arm
        sides, teacher_scores, results, baseline_scores, candidates = _collect_predictions(
            dataset,
            _constant_inputs(constant),
            runs,
        )

    baseline_evaluation = evaluate_prediction_scores(
        lookup, sides, teacher_scores, results, baseline_scores
    )
    baseline_mean = float_from_receipt(
        baseline_evaluation["composite_loss_mean_all_records"], "baseline validation mean"
    )
    run_receipts: list[dict[str, object]] = []
    deltas: list[float] = []
    for run in sorted(runs, key=lambda item: FROZEN_SEEDS.index(item.seed)):
        evaluation = evaluate_prediction_scores(
            lookup,
            sides,
            teacher_scores,
            results,
            candidates[run.seed],
        )
        candidate_mean = float_from_receipt(
            evaluation["composite_loss_mean_all_records"],
            f"seed {run.seed} validation mean",
        )
        delta = baseline_mean - candidate_mean
        deltas.append(delta)
        run_receipts.append(
            {
                "seed": run.seed,
                "directory_name": run.directory.name,
                "checkpoint_sha256": run.checkpoint_sha256,
                "training_receipt_sha256": run.training_receipt_sha256,
                "functional_health_receipt_sha256": run.health_receipt_sha256,
                "evaluation": evaluation,
                "paired_delta_constant_minus_checkpoint": float_receipt(delta),
                "strictly_better_than_constant": delta > 0.0,
            }
        )

    checks = {
        "source_clean": source["dirty"] is False,
        "exact_three_frozen_seeds": tuple(sorted(run.seed for run in runs))
        == tuple(sorted(FROZEN_SEEDS)),
        "one_registered_arm": len({run.arm for run in runs}) == 1 and arm in ARMS,
        "all_final_exposure": True,
        "all_functional_health_pass": True,
        "all_checkpoint_deltas_strictly_positive": all(delta > 0.0 for delta in deltas),
        "cluster_claim_is_honest": True,
    }
    passed = all(checks.values())
    receipt: dict[str, object] = {
        "schema": SCHEMA,
        "contract": {"schema": CONTRACT_SCHEMA, "sha256": contract_sha256},
        "source": source,
        "inputs": {
            "constant_baseline": {
                "name": constant_path.expanduser().resolve().name,
                "sha256": _sha256_bytes(constant_payload),
                "schema": CONSTANT_RECEIPT_SCHEMA,
                "training_file": expected_training,
                "constants_cp": {
                    SIDE_NAMES[side]: value for side, value in _constant_inputs(constant).items()
                },
            },
            "validation": validation_identity,
            "teacher": expected_teacher,
            "wdl_calibration": {
                "name": wdl_path.expanduser().resolve().name,
                "sha256": wdl_sha256,
                "schema": WDL_SCHEMA,
                "link_schema": expected_wdl["link_schema"],
                "selection_sha256": expected_wdl["selection_sha256"],
                "eligible_records_sha256": expected_wdl["eligible_records_sha256"],
                "lookup_raw_float32_sha256": lookup.raw_float32_sha256,
                "parameter_float32_sha256": lookup.parameter_float32_sha256,
            },
        },
        "arm": {
            "name": arm,
            "architecture": dict(ARCHITECTURE),
            "optimizer_learning_rate_multipliers": dict(ARMS[arm]),
            "frozen_seeds": list(FROZEN_SEEDS),
            "epochs": EPOCHS,
            "optimizer_steps": OPTIMIZER_STEPS,
            "samples_consumed_per_seed": SAMPLES_CONSUMED,
        },
        "objective": dict(OBJECTIVE_RECEIPT),
        "evaluation": {
            "constant_baseline": baseline_evaluation,
            "runs": run_receipts,
        },
        "statistics": {
            "unit": "record",
            "sample_identity": "(payload_sha256, local_record_index)",
            "cluster_identity": None,
            "cluster_identity_reason": "absent from HORDE_BIN_V1",
            "confidence_interval": None,
            "iid_bootstrap": False,
            "game_clustered_claim": False,
            "selected_role_status": "qualification/tuning; previously inspected",
        },
        "gates": {"checks": checks, "passed": passed},
        "claims": {
            "recipe_qualified": passed,
            "architecture_selected": False,
            "best_seed_or_epoch_selected": False,
            "arm_ranked_against_other_arm": False,
            "statistical_confidence": False,
            "playing_strength_evidence": False,
            "production_network": False,
            "run6b_production_path_changed": False,
        },
    }
    validate_receipt(receipt)
    return receipt


def _validate_evaluation_receipt(
    value: object,
    label: str,
    *,
    expected_records: int | None = None,
    expected_eligible: int | None = None,
) -> tuple[Mapping[str, Any], float]:
    evaluation = _mapping(value, label)
    _require(
        set(evaluation)
        == {
            "records",
            "eligible_teacher_scores",
            "prediction_and_label_chain_sha256",
            "composite_loss_sum",
            "composite_loss_mean_all_records",
            "score_half_brier_sum",
            "result_half_brier_sum",
            "sides",
        },
        f"{label} fields are invalid",
    )
    records = evaluation.get("records")
    eligible = evaluation.get("eligible_teacher_scores")
    _require(
        type(records) is int
        and records > 0
        and type(eligible) is int
        and 0 <= eligible <= records
        and (expected_records is None or records == expected_records)
        and (expected_eligible is None or eligible == expected_eligible)
        and _valid_sha256(evaluation.get("prediction_and_label_chain_sha256")),
        f"{label} record accounting is invalid",
    )
    composite_sum = float_from_receipt(evaluation.get("composite_loss_sum"), f"{label} sum")
    composite_mean = float_from_receipt(
        evaluation.get("composite_loss_mean_all_records"), f"{label} mean"
    )
    score_sum = float_from_receipt(
        evaluation.get("score_half_brier_sum"), f"{label} score sum"
    )
    result_sum = float_from_receipt(
        evaluation.get("result_half_brier_sum"), f"{label} result sum"
    )
    _require(
        all(value >= 0.0 for value in (composite_sum, composite_mean, score_sum, result_sum))
        and composite_mean == composite_sum / records,
        f"{label} loss accounting is invalid",
    )

    sides = _mapping(evaluation.get("sides"), f"{label} sides")
    _require(set(sides) == set(SIDE_NAMES.values()), f"{label} side fields are invalid")
    side_records = 0
    side_eligible = 0
    for side_name in SIDE_NAMES.values():
        side = _mapping(sides.get(side_name), f"{label} {side_name}")
        _require(
            set(side)
            == {
                "records",
                "eligible_teacher_scores",
                "unique_predicted_integer_scores",
                "composite_loss_sum",
                "composite_loss_mean_all_side_records",
                "score_half_brier_sum",
                "result_half_brier_sum",
            },
            f"{label} {side_name} fields are invalid",
        )
        count = side.get("records")
        side_score_eligible = side.get("eligible_teacher_scores")
        unique = side.get("unique_predicted_integer_scores")
        _require(
            type(count) is int
            and count > 0
            and type(side_score_eligible) is int
            and 0 <= side_score_eligible <= count
            and type(unique) is int
            and 1 <= unique <= count,
            f"{label} {side_name} record accounting is invalid",
        )
        side_sum = float_from_receipt(
            side.get("composite_loss_sum"), f"{label} {side_name} sum"
        )
        side_mean = float_from_receipt(
            side.get("composite_loss_mean_all_side_records"),
            f"{label} {side_name} mean",
        )
        side_score_sum = float_from_receipt(
            side.get("score_half_brier_sum"), f"{label} {side_name} score sum"
        )
        side_result_sum = float_from_receipt(
            side.get("result_half_brier_sum"), f"{label} {side_name} result sum"
        )
        _require(
            all(
                number >= 0.0
                for number in (side_sum, side_mean, side_score_sum, side_result_sum)
            )
            and side_mean == side_sum / count,
            f"{label} {side_name} loss accounting is invalid",
        )
        side_records += count
        side_eligible += side_score_eligible
    _require(
        side_records == records and side_eligible == eligible,
        f"{label} side totals drifted",
    )
    return evaluation, composite_mean


def validate_receipt(value: object) -> dict[str, object]:
    _require(isinstance(value, dict), "qualification receipt root is not an object")
    receipt = value
    _require(
        set(receipt)
        == {
            "schema",
            "contract",
            "source",
            "inputs",
            "arm",
            "objective",
            "evaluation",
            "statistics",
            "gates",
            "claims",
        },
        "qualification receipt fields are incomplete",
    )
    _require(receipt.get("schema") == SCHEMA, "qualification receipt schema mismatch")
    _require(
        receipt.get("contract") == {"schema": CONTRACT_SCHEMA, "sha256": CONTRACT_SHA256},
        "qualification contract identity mismatch",
    )
    source = _mapping(receipt.get("source"), "qualification source")
    _require(
        set(source)
        == {"commit", "dirty", "python", "implementation", "torch", "tool"}
        and _valid_commit(source.get("commit"))
        and type(source.get("dirty")) is bool
        and all(
            _nonempty_string(source.get(field))
            for field in ("python", "implementation", "torch")
        )
        and source.get("tool") == "tools/horde_v2_c2_qualification.py",
        "qualification source identity is invalid",
    )
    inputs = _mapping(receipt.get("inputs"), "qualification inputs")
    _require(
        set(inputs) == {"constant_baseline", "validation", "teacher", "wdl_calibration"},
        "qualification input fields are invalid",
    )
    constant_input = _mapping(
        inputs.get("constant_baseline"), "qualification constant-baseline input"
    )
    _require(
        set(constant_input)
        == {"name", "sha256", "schema", "training_file", "constants_cp"}
        and _nonempty_string(constant_input.get("name"))
        and _valid_sha256(constant_input.get("sha256"))
        and constant_input.get("schema") == CONSTANT_RECEIPT_SCHEMA,
        "qualification constant-baseline input is invalid",
    )
    _validate_training_identity(
        constant_input.get("training_file"), "qualification training identity"
    )
    constants = _mapping(constant_input.get("constants_cp"), "qualification constants")
    _require(
        set(constants) == set(SIDE_NAMES.values())
        and all(
            type(constants[name]) is int
            and LOOKUP_SCORE_MINIMUM <= constants[name] <= LOOKUP_SCORE_MAXIMUM
            for name in SIDE_NAMES.values()
        ),
        "qualification constant values are invalid",
    )
    validation_identity = _validate_validation_identity(
        inputs.get("validation"), "qualification validation identity"
    )
    _validate_teacher_identity(inputs.get("teacher"), "qualification teacher identity")
    wdl_input = _mapping(inputs.get("wdl_calibration"), "qualification WDL input")
    _require(
        set(wdl_input)
        == {
            "name",
            "sha256",
            "schema",
            "link_schema",
            "selection_sha256",
            "eligible_records_sha256",
            "lookup_raw_float32_sha256",
            "parameter_float32_sha256",
        }
        and _nonempty_string(wdl_input.get("name"))
        and wdl_input.get("schema") == WDL_SCHEMA
        and wdl_input.get("link_schema") == WDL_LINK_SCHEMA
        and all(
            _valid_sha256(wdl_input.get(field))
            for field in (
                "sha256",
                "selection_sha256",
                "eligible_records_sha256",
                "lookup_raw_float32_sha256",
                "parameter_float32_sha256",
            )
        ),
        "qualification WDL input is invalid",
    )
    _require(
        receipt.get("objective") == OBJECTIVE_RECEIPT,
        "qualification objective identity is invalid",
    )
    arm = _mapping(receipt.get("arm"), "qualification arm")
    name = arm.get("name")
    _require(
        name in ARMS
        and arm.get("architecture") == ARCHITECTURE
        and arm.get("optimizer_learning_rate_multipliers") == ARMS[name]
        and arm.get("frozen_seeds") == list(FROZEN_SEEDS)
        and arm.get("epochs") == EPOCHS
        and arm.get("optimizer_steps") == OPTIMIZER_STEPS
        and arm.get("samples_consumed_per_seed") == SAMPLES_CONSUMED,
        "qualification arm identity is invalid",
    )
    statistics = _mapping(receipt.get("statistics"), "qualification statistics")
    _require(
        statistics
        == {
            "unit": "record",
            "sample_identity": "(payload_sha256, local_record_index)",
            "cluster_identity": None,
            "cluster_identity_reason": "absent from HORDE_BIN_V1",
            "confidence_interval": None,
            "iid_bootstrap": False,
            "game_clustered_claim": False,
            "selected_role_status": "qualification/tuning; previously inspected",
        },
        "qualification statistical claims are invalid",
    )
    evaluation = _mapping(receipt.get("evaluation"), "qualification evaluation")
    _require(
        set(evaluation) == {"constant_baseline", "runs"},
        "qualification evaluation fields are invalid",
    )
    baseline, baseline_mean = _validate_evaluation_receipt(
        evaluation.get("constant_baseline"),
        "qualification constant evaluation",
        expected_records=int(validation_identity["records"]),
    )
    runs = evaluation.get("runs")
    _require(isinstance(runs, list) and len(runs) == 3, "qualification does not contain three runs")
    baseline_eligible = int(baseline["eligible_teacher_scores"])
    observed_seeds: list[int] = []
    strict_results: list[bool] = []
    for run_value in runs:
        run = _mapping(run_value, "qualification run")
        seed = run.get("seed")
        _require(type(seed) is int and seed in FROZEN_SEEDS, "qualification run seed is invalid")
        observed_seeds.append(seed)
        _require(
            set(run)
            == {
                "seed",
                "directory_name",
                "checkpoint_sha256",
                "training_receipt_sha256",
                "functional_health_receipt_sha256",
                "evaluation",
                "paired_delta_constant_minus_checkpoint",
                "strictly_better_than_constant",
            }
            and _nonempty_string(run.get("directory_name"))
            and Path(str(run["directory_name"])).name == run["directory_name"]
            and all(separator not in str(run["directory_name"]) for separator in ("/", "\\"))
            and
            all(
                _valid_sha256(run.get(field))
                for field in (
                    "checkpoint_sha256",
                    "training_receipt_sha256",
                    "functional_health_receipt_sha256",
                )
            ),
            "qualification run artifact identity is invalid",
        )
        candidate, candidate_mean = _validate_evaluation_receipt(
            run.get("evaluation"),
            "qualification candidate evaluation",
            expected_records=int(validation_identity["records"]),
            expected_eligible=baseline_eligible,
        )
        delta = float_from_receipt(
            run.get("paired_delta_constant_minus_checkpoint"),
            f"qualification seed {seed} delta",
        )
        _require(delta == baseline_mean - candidate_mean, "qualification paired delta drifted")
        strict = run.get("strictly_better_than_constant")
        _require(type(strict) is bool and strict is (delta > 0.0), "qualification strict result drifted")
        strict_results.append(strict)
    _require(observed_seeds == list(FROZEN_SEEDS), "qualification run order or seed set drifted")
    gates = _mapping(receipt.get("gates"), "qualification gates")
    checks = _mapping(gates.get("checks"), "qualification checks")
    expected_checks = {
        "source_clean": source.get("dirty") is False,
        "exact_three_frozen_seeds": observed_seeds == list(FROZEN_SEEDS),
        "one_registered_arm": name in ARMS,
        "all_final_exposure": True,
        "all_functional_health_pass": True,
        "all_checkpoint_deltas_strictly_positive": all(strict_results),
        "cluster_claim_is_honest": True,
    }
    _require(
        set(gates) == {"checks", "passed"}
        and checks == expected_checks
        and gates.get("passed") is all(expected_checks.values()),
        "qualification gate accounting drifted",
    )
    claims = _mapping(receipt.get("claims"), "qualification claims")
    _require(
        claims
        == {
            "recipe_qualified": gates["passed"],
            "architecture_selected": False,
            "best_seed_or_epoch_selected": False,
            "arm_ranked_against_other_arm": False,
            "statistical_confidence": False,
            "playing_strength_evidence": False,
            "production_network": False,
            "run6b_production_path_changed": False,
        },
        "qualification claims drifted",
    )
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("constant_baseline", type=Path)
    parser.add_argument("validation", type=Path, help="selected-role receipt")
    parser.add_argument("wdl_calibration", type=Path)
    parser.add_argument("--run-directory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--allow-dirty", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_receipt(
        args.constant_baseline,
        args.validation,
        args.wdl_calibration,
        args.run_directory,
        contract_path=args.contract,
        allow_dirty=args.allow_dirty,
    )
    _write_exclusive(args.output, _canonical_json(receipt))
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "arm": receipt["arm"]["name"],
                "output": str(args.output.expanduser().resolve()),
                "passed": receipt["gates"]["passed"],
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["gates"]["passed"] or not args.require_pass else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (QualificationError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
