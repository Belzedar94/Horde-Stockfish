#!/usr/bin/env python3
"""Plan and verify the frozen Horde V2 C1 architecture campaign."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

try:
    from . import horde_bin_v1 as wire
    from .horde_training_control import (
        DEFAULT_LAMBDA,
        DEFAULT_LEARNING_RATE,
        DEFAULT_SCHEDULER_GAMMA,
        MATE_SCORE_THRESHOLD,
        TrainingError,
        V2_ARCHITECTURES,
        sample_order_chain_sha256,
        validate_dataset_pair,
    )
    from .horde_training_decoder import HordeBinV1Dataset
    from .horde_training_split_audit import AuditError, audit_pair
    from .horde_v2_container import (
        ContainerError,
        SPECS_BY_ARCHITECTURE,
        read_container,
        sha256_file,
    )
    from .horde_wdl import CalibrationError, load_artifact as load_wdl_artifact
except ImportError:
    import horde_bin_v1 as wire
    from horde_training_control import (
        DEFAULT_LAMBDA,
        DEFAULT_LEARNING_RATE,
        DEFAULT_SCHEDULER_GAMMA,
        MATE_SCORE_THRESHOLD,
        TrainingError,
        V2_ARCHITECTURES,
        sample_order_chain_sha256,
        validate_dataset_pair,
    )
    from horde_training_decoder import HordeBinV1Dataset
    from horde_training_split_audit import AuditError, audit_pair
    from horde_v2_container import (
        ContainerError,
        SPECS_BY_ARCHITECTURE,
        read_container,
        sha256_file,
    )
    from horde_wdl import CalibrationError, load_artifact as load_wdl_artifact


CONTRACT_SCHEMA = "HORDE_V2_C1_CAMPAIGN_V1"
CONTRACT_RELATIVE_PATH = Path("schemas/horde-v2-c1-campaign-v1.json")
CONTRACT_SHA256 = "7B5BDA9DC20AB7CF55DE2964085D2ADBBED83137A3071B418439A5CF7DD939DA"
PLAN_SCHEMA = "HORDE_V2_C1_CAMPAIGN_PLAN_V1"
VERIFICATION_SCHEMA = "HORDE_V2_C1_CAMPAIGN_VERIFICATION_V1"
TRAINING_RECEIPT_SCHEMA = "HORDE_V2_BASE_TRAINING_V1"
EXPORT_RECEIPT_SCHEMA = "HORDE_V2_INTEGER_CHECKPOINT_EXPORT_V1"
COVERAGE_SCHEMA = "HORDE_V2_C1_DATA_COVERAGE_V1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WHITE_PIECE_BINS = ((1, 4), (5, 8), (9, 16), (17, 24), (25, 30), (31, 36))


class CampaignError(ValueError):
    """Raised when a C1 campaign input or completed run violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


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


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = path.expanduser().resolve()
    _require(resolved.is_file(), f"{label} does not exist: {resolved}")
    payload = resolved.read_bytes()
    try:
        root = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"{label} is invalid JSON: {error}") from error
    _require(isinstance(root, dict), f"{label} root is not an object")
    return root, payload


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, dict), f"{label} is not an object")
    return value


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


def _seed(index: int) -> tuple[int, str]:
    label = f"{CONTRACT_SCHEMA}:seed:{index}"
    digest = hashlib.sha256(label.encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big"), digest.hex().upper()


def _valid_commit(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value != "0" * 40
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


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
    _require(_valid_commit(commit), "campaign source is not a full Git identity")
    return {"commit": commit.lower(), "dirty": dirty}


def load_contract(path: Path | None = None) -> tuple[dict[str, Any], str]:
    contract_path = (path or (REPOSITORY_ROOT / CONTRACT_RELATIVE_PATH)).expanduser().resolve()
    contract, payload = _read_json(contract_path, "C1 campaign contract")
    digest = _sha256_bytes(payload)
    _require(digest == CONTRACT_SHA256, f"C1 campaign contract SHA-256 mismatch: {digest}")
    _require(contract.get("schema_name") == CONTRACT_SCHEMA, "C1 campaign schema mismatch")

    dependencies = _mapping(contract.get("dependencies"), "contract dependencies")
    dataset = _mapping(dependencies.get("dataset"), "contract dataset dependency")
    teacher = _mapping(dependencies.get("teacher"), "contract teacher dependency")
    labels = _mapping(dependencies.get("labels"), "contract label dependency")
    _require(dataset.get("schema") == wire.SCHEMA_NAME, "contract dataset schema drifted")
    _require(dataset.get("schema_sha256") == wire.SCHEMA_SHA256, "contract dataset hash drifted")
    _require(teacher.get("schema") == "HORDETEST_HP_LEGACY_V1", "teacher schema drifted")
    _require(teacher.get("network_sha256") == wire.RUN6B_SHA256, "Run 6B hash drifted")
    _require(labels.get("schema") == wire.LABEL_CONTRACT_NAME, "label schema drifted")
    _require(
        labels.get("schema_sha256") == wire.LABEL_CONTRACT_SHA256,
        "label contract hash drifted",
    )
    _require(
        dependencies.get("book_split_schema") == "HORDE_TRAINING_BOOK_SPLIT_V2",
        "book split schema drifted",
    )

    data_contract = _mapping(contract.get("data"), "contract data section")
    coverage_contract = _mapping(data_contract.get("coverage"), "contract coverage section")
    _require(
        coverage_contract
        == {
            "schema": COVERAGE_SCHEMA,
            "royal_bucket_position_minimums": {"train": 500, "validation": 200},
            "unseen_validation_royal_activation_fraction_maximum_exclusive": 0.001,
            "validation_stm_white_piece_bin_minimum": 1_000,
            "white_piece_bins": [
                f"{minimum}-{maximum}" for minimum, maximum in WHITE_PIECE_BINS
            ],
            "side_result_classes_required": [-1, 0, 1],
        },
        "C1 coverage contract drifted",
    )

    training = _mapping(contract.get("training"), "contract training section")
    architectures = training.get("architectures")
    _require(isinstance(architectures, list), "contract architectures are missing")
    _require(len(architectures) == 3, "C1 must compare exactly three architectures")
    observed_names: list[str] = []
    for item in architectures:
        architecture = _mapping(item, "contract architecture")
        name = architecture.get("name")
        _require(isinstance(name, str), "contract architecture name is invalid")
        _require(name in SPECS_BY_ARCHITECTURE, f"unregistered C1 architecture: {name}")
        _require(name in V2_ARCHITECTURES, f"untrainable C1 architecture: {name}")
        spec = SPECS_BY_ARCHITECTURE[name]
        trainer = V2_ARCHITECTURES[name]
        _require(architecture.get("schema") == spec.schema_name, f"{name} schema drifted")
        _require(
            architecture.get("first_domain") == spec.first_domain_name,
            f"{name} first domain drifted",
        )
        _require(
            architecture.get("serialized_parameter_bytes") == spec.parameter_bytes,
            f"{name} parameter bytes drifted",
        )
        _require(
            architecture.get("training_structural_sha256")
            == spec.training_structural_sha256,
            f"{name} structural hash drifted",
        )
        _require(trainer.get("schema") == spec.schema_name, f"{name} trainer schema drifted")
        _require(
            trainer.get("serialized_parameter_bytes") == spec.parameter_bytes,
            f"{name} trainer parameter bytes drifted",
        )
        observed_names.append(name)
    _require(
        observed_names == ["v2-c1-abs64x192", "v2-c1-rank8-64x192", "v2-64x192"],
        "C1 architecture order drifted",
    )

    paired = _mapping(training.get("paired_seeds"), "contract paired seeds")
    values = paired.get("values")
    expected_seeds = [_seed(index)[0] for index in range(3)]
    _require(values == expected_seeds, "C1 paired seed derivation drifted")
    _require(training.get("run_count") == 9, "C1 run count drifted")
    _require(training.get("epochs") == 8, "C1 epoch count drifted")
    _require(training.get("batch_size") == 4096, "C1 batch size drifted")
    _require(training.get("block_size") == 65_536, "C1 block size drifted")
    _require(training.get("lambda") == DEFAULT_LAMBDA, "C1 lambda drifted")
    _require(
        training.get("learning_rate") == DEFAULT_LEARNING_RATE,
        "C1 learning rate drifted",
    )
    _require(
        training.get("scheduler_gamma") == DEFAULT_SCHEDULER_GAMMA,
        "C1 scheduler gamma drifted",
    )
    optimizer = _mapping(training.get("optimizer"), "contract optimizer")
    _require(
        optimizer
        == {
            "name": "torch.optim.RAdam",
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-7,
            "weight_decay": 0.0,
            "output_learning_rate_multiplier": 0.1,
            "foreach": False,
            "lookahead": False,
            "gradient_centralization": False,
            "scheduler": {
                "name": "StepLR",
                "step_size_epochs": 1,
                "gamma": DEFAULT_SCHEDULER_GAMMA,
            },
        },
        "C1 optimizer contract drifted",
    )
    selection = _mapping(contract.get("selection"), "contract selection section")
    _require(
        selection.get("predesignated_playing_seed_index") == 0
        and selection.get("predesignated_playing_seed") == expected_seeds[0],
        "C1 predesignated playing seed drifted",
    )
    return contract, digest


def _rank8_dependency(contract: Mapping[str, Any]) -> dict[str, object]:
    dependencies = _mapping(contract.get("dependencies"), "contract dependencies")
    frozen = _mapping(dependencies.get("rank8_receipt"), "Rank-8 dependency")
    relative = frozen.get("path")
    expected_sha = frozen.get("sha256")
    _require(isinstance(relative, str), "Rank-8 receipt path is invalid")
    path = (REPOSITORY_ROOT / relative).resolve()
    receipt, payload = _read_json(path, "Rank-8 control receipt")
    digest = _sha256_bytes(payload)
    _require(digest == expected_sha, f"Rank-8 receipt SHA-256 mismatch: {digest}")
    _require(
        receipt.get("schema") == "HORDE_V2_RANK8_CONTROL_RECEIPT_V1",
        "Rank-8 receipt schema mismatch",
    )
    claims = _mapping(receipt.get("claims"), "Rank-8 receipt claims")
    _require(claims.get("incremental_full_refresh_parity") is True, "Rank-8 parity is absent")
    _require(claims.get("run6b_production_path_changed") is False, "Rank-8 changed Run 6B")
    return {"path": relative, "sha256": digest, "schema": receipt["schema"]}


def _validate_wdl(
    path: Path,
    data: Mapping[str, Any],
    train_manifest_sha256: str,
) -> dict[str, object]:
    try:
        payload, _parameters, digest = load_wdl_artifact(path)
    except CalibrationError as error:
        raise CampaignError(f"WDL calibration is invalid: {error}") from error
    source = _mapping(payload.get("source"), "WDL source")
    training_file = _mapping(source.get("training_file"), "WDL training file")
    teacher = _mapping(source.get("teacher"), "WDL teacher")
    expected_train = _mapping(data.get("train_file"), "campaign training file")
    _require(
        training_file.get("sha256") == expected_train.get("sha256")
        and training_file.get("payload_sha256") == expected_train.get("payload_sha256")
        and training_file.get("manifest_sha256") == train_manifest_sha256
        and training_file.get("records") == expected_train.get("records"),
        "WDL calibration was not fitted from the exact training dataset",
    )
    expected_teacher = _mapping(data.get("teacher"), "campaign teacher")
    _require(
        all(expected_teacher.get(key) == value for key, value in teacher.items()),
        "WDL calibration teacher identity mismatch",
    )
    link = _mapping(payload.get("link"), "WDL link")
    selection = _mapping(payload.get("selection"), "WDL selection")
    return {
        "name": path.expanduser().resolve().name,
        "sha256": digest,
        "schema": payload.get("schema"),
        "link_schema": link.get("schema"),
        "selection_sha256": selection.get("selection_sha256"),
        "eligible_records_sha256": selection.get("eligible_records_sha256"),
    }


def _white_piece_bin(count: int) -> str:
    for minimum, maximum in WHITE_PIECE_BINS:
        if minimum <= count <= maximum:
            return f"{minimum}-{maximum}"
    raise CampaignError(f"White piece count is outside the C1 coverage bins: {count}")


def _sorted_counter(counter: Counter[object]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def _dataset_coverage(
    dataset: HordeBinV1Dataset,
) -> tuple[dict[str, object], Counter[int]]:
    side_to_move: Counter[str] = Counter()
    results: dict[str, Counter[int]] = {"white": Counter(), "black": Counter()}
    reasons: Counter[str] = Counter()
    white_piece_counts: Counter[int] = Counter()
    white_pawn_counts: Counter[int] = Counter()
    stm_piece_bins: dict[str, Counter[str]] = {"white": Counter(), "black": Counter()}
    black_nonking_counts: Counter[int] = Counter()
    king_buckets: Counter[int] = Counter()
    king_ranks: Counter[int] = Counter()
    king_files: Counter[int] = Counter()
    royal_rows: Counter[int] = Counter()
    promoted_horde_positions = 0
    castling_positions = 0
    en_passant_positions = 0
    best_played_divergence = 0
    mate_scores_masked = 0
    score_minimum: int | None = None
    score_maximum: int | None = None
    score_sum = 0

    for index in range(len(dataset)):
        record = dataset.record(index)
        side = "white" if record.side_to_move == 0 else "black"
        side_to_move[side] += 1
        results[side][record.result] += 1
        reasons[wire.OUTCOME_NAMES[record.outcome_reason]] += 1
        white_pieces = sum(1 <= code <= 5 for code in record.board)
        white_pawns = sum(code == 1 for code in record.board)
        white_piece_counts[white_pieces] += 1
        white_pawn_counts[white_pawns] += 1
        stm_piece_bins[side][_white_piece_bin(white_pieces)] += 1
        black_nonking_counts[sum(6 <= code <= 10 for code in record.board)] += 1
        king_square = record.board.index(11)
        king_buckets[record.features.royal_bucket] += 1
        king_ranks[king_square // 8] += 1
        king_files[king_square % 8] += 1
        royal_rows.update(record.features.v2_royal)
        promoted_horde_positions += int(any(2 <= code <= 5 for code in record.board))
        castling_positions += int(record.castling_rights != 0)
        en_passant_positions += int(record.ep_square != 64)
        best_played_divergence += int(record.best_move != record.played_move)
        mate_scores_masked += int(abs(record.score) >= MATE_SCORE_THRESHOLD)
        score_minimum = record.score if score_minimum is None else min(score_minimum, record.score)
        score_maximum = record.score if score_maximum is None else max(score_maximum, record.score)
        score_sum += record.score

    record_count = len(dataset)
    return (
        {
            "records": record_count,
            "side_to_move": _sorted_counter(side_to_move),
            "side_result_classes": {
                side: _sorted_counter(counts) for side, counts in results.items()
            },
            "outcome_reasons": _sorted_counter(reasons),
            "white_piece_counts": _sorted_counter(white_piece_counts),
            "white_pawn_counts": _sorted_counter(white_pawn_counts),
            "stm_white_piece_bins": {
                side: _sorted_counter(counts) for side, counts in stm_piece_bins.items()
            },
            "black_nonking_piece_counts": _sorted_counter(black_nonking_counts),
            "royal_bucket_positions": [king_buckets[index] for index in range(32)],
            "black_king_rank_positions": [king_ranks[index] for index in range(8)],
            "black_king_file_positions": [king_files[index] for index in range(8)],
            "royal_row_activations": sum(royal_rows.values()),
            "royal_unique_rows": len(royal_rows),
            "promoted_horde_positions": promoted_horde_positions,
            "castling_positions": castling_positions,
            "en_passant_positions": en_passant_positions,
            "best_played_divergence": best_played_divergence,
            "mate_scores_masked": mate_scores_masked,
            "score": {
                "minimum": score_minimum,
                "maximum": score_maximum,
                "mean": score_sum / record_count,
            },
        },
        royal_rows,
    )


def _coverage_receipt(
    train: HordeBinV1Dataset,
    validation: HordeBinV1Dataset,
) -> dict[str, object]:
    train_summary, train_rows = _dataset_coverage(train)
    validation_summary, validation_rows = _dataset_coverage(validation)
    unseen_activations = sum(
        count for row, count in validation_rows.items() if row not in train_rows
    )
    validation_activations = int(validation_summary["royal_row_activations"])
    unseen_fraction = unseen_activations / validation_activations
    bucket_minimums = {"train": 500, "validation": 200}
    stm_bin_minimum = 1_000
    class_values = {-1, 0, 1}
    bucket_gate = (
        min(train_summary["royal_bucket_positions"]) >= bucket_minimums["train"]
        and min(validation_summary["royal_bucket_positions"])
        >= bucket_minimums["validation"]
    )
    stm_bin_gate = all(
        int(validation_summary["stm_white_piece_bins"].get(side, {}).get(label, 0))
        >= stm_bin_minimum
        for side in ("white", "black")
        for label in (f"{minimum}-{maximum}" for minimum, maximum in WHITE_PIECE_BINS)
    )
    result_gate = all(
        {int(value) for value in summary["side_result_classes"].get(side, {})}
        == class_values
        for summary in (train_summary, validation_summary)
        for side in ("white", "black")
    )
    unseen_gate = unseen_fraction < 0.001
    return {
        "schema": COVERAGE_SCHEMA,
        "train": train_summary,
        "validation": validation_summary,
        "unseen_validation_royal_activations": {
            "count": unseen_activations,
            "total": validation_activations,
            "fraction": unseen_fraction,
            "maximum_exclusive": 0.001,
        },
        "gates": {
            "royal_bucket_position_minimums": bucket_minimums,
            "validation_stm_white_piece_bin_minimum": stm_bin_minimum,
            "side_result_classes_required": [-1, 0, 1],
            "royal_bucket_coverage": bucket_gate,
            "validation_stm_white_piece_bins": stm_bin_gate,
            "side_result_classes": result_gate,
            "unseen_validation_royal_activations": unseen_gate,
            "passed": bucket_gate and stm_bin_gate and result_gate and unseen_gate,
        },
    }


def _validate_coverage_receipt(
    coverage: Mapping[str, Any],
    expected_records: tuple[int, int],
) -> None:
    _require(coverage.get("schema") == COVERAGE_SCHEMA, "C1 coverage schema drifted")
    summaries = {
        "train": _mapping(coverage.get("train"), "C1 training coverage"),
        "validation": _mapping(coverage.get("validation"), "C1 validation coverage"),
    }
    bin_labels = {f"{minimum}-{maximum}" for minimum, maximum in WHITE_PIECE_BINS}
    for (role, summary), expected in zip(summaries.items(), expected_records):
        _require(summary.get("records") == expected, f"C1 {role} coverage count drifted")
        buckets = summary.get("royal_bucket_positions")
        _require(
            isinstance(buckets, list)
            and len(buckets) == 32
            and all(type(count) is int and count >= 0 for count in buckets)
            and sum(buckets) == expected,
            f"C1 {role} Royal-bucket coverage is invalid",
        )
        sides = _mapping(summary.get("side_to_move"), f"C1 {role} STM coverage")
        _require(
            set(sides) == {"black", "white"}
            and all(type(count) is int and count >= 0 for count in sides.values())
            and sum(sides.values()) == expected,
            f"C1 {role} STM coverage is invalid",
        )
        stm_bins = _mapping(
            summary.get("stm_white_piece_bins"),
            f"C1 {role} STM/piece-bin coverage",
        )
        result_classes = _mapping(
            summary.get("side_result_classes"),
            f"C1 {role} side-result coverage",
        )
        for side in ("white", "black"):
            bins = _mapping(stm_bins.get(side), f"C1 {role} {side} piece bins")
            _require(
                set(bins).issubset(bin_labels)
                and all(type(count) is int and count >= 0 for count in bins.values())
                and sum(bins.values()) == sides[side],
                f"C1 {role} {side} piece-bin counts are inconsistent",
            )
            classes = _mapping(
                result_classes.get(side),
                f"C1 {role} {side} result classes",
            )
            _require(
                set(classes).issubset({"-1", "0", "1"})
                and all(type(count) is int and count >= 0 for count in classes.values())
                and sum(classes.values()) == sides[side],
                f"C1 {role} {side} result counts are inconsistent",
            )
        activations = summary.get("royal_row_activations")
        unique_rows = summary.get("royal_unique_rows")
        _require(
            type(activations) is int
            and activations > 0
            and type(unique_rows) is int
            and 0 < unique_rows <= 20_480
            and unique_rows <= activations,
            f"C1 {role} Royal-row coverage is invalid",
        )

    unseen = _mapping(
        coverage.get("unseen_validation_royal_activations"),
        "C1 unseen Royal coverage",
    )
    unseen_count = unseen.get("count")
    unseen_total = unseen.get("total")
    unseen_fraction = unseen.get("fraction")
    _require(
        type(unseen_count) is int
        and type(unseen_total) is int
        and 0 <= unseen_count <= unseen_total
        and unseen_total == summaries["validation"]["royal_row_activations"]
        and isinstance(unseen_fraction, (int, float))
        and math.isfinite(float(unseen_fraction))
        and float(unseen_fraction) == unseen_count / unseen_total
        and unseen.get("maximum_exclusive") == 0.001,
        "C1 unseen Royal-row receipt is inconsistent",
    )
    gates = _mapping(coverage.get("gates"), "C1 coverage gates")
    bucket_minimums = _mapping(
        gates.get("royal_bucket_position_minimums"),
        "C1 Royal-bucket minimums",
    )
    _require(bucket_minimums == {"train": 500, "validation": 200}, "C1 bucket thresholds drifted")
    _require(
        gates.get("validation_stm_white_piece_bin_minimum") == 1_000
        and gates.get("side_result_classes_required") == [-1, 0, 1],
        "C1 coverage thresholds drifted",
    )
    expected_bucket_gate = (
        min(summaries["train"]["royal_bucket_positions"]) >= 500
        and min(summaries["validation"]["royal_bucket_positions"]) >= 200
    )
    expected_stm_gate = all(
        int(summaries["validation"]["stm_white_piece_bins"].get(side, {}).get(label, 0))
        >= 1_000
        for side in ("white", "black")
        for label in bin_labels
    )
    expected_result_gate = all(
        set(summaries[role]["side_result_classes"][side]) == {"-1", "0", "1"}
        for role in ("train", "validation")
        for side in ("white", "black")
    )
    expected_unseen_gate = float(unseen_fraction) < 0.001
    expected_passed = (
        expected_bucket_gate
        and expected_stm_gate
        and expected_result_gate
        and expected_unseen_gate
    )
    _require(
        gates.get("royal_bucket_coverage") is expected_bucket_gate
        and gates.get("validation_stm_white_piece_bins") is expected_stm_gate
        and gates.get("side_result_classes") is expected_result_gate
        and gates.get("unseen_validation_royal_activations") is expected_unseen_gate
        and gates.get("passed") is expected_passed,
        "C1 coverage gate booleans contradict their counts",
    )


def _require_production_coverage(coverage: Mapping[str, Any]) -> None:
    gates = _mapping(coverage.get("gates"), "C1 coverage gates")
    _require(gates.get("royal_bucket_coverage") is True, "C1 Royal buckets lack coverage")
    _require(
        gates.get("validation_stm_white_piece_bins") is True,
        "C1 validation STM/piece bins lack coverage",
    )
    _require(
        gates.get("side_result_classes") is True,
        "C1 side-specific WDL classes lack coverage",
    )
    _require(
        gates.get("unseen_validation_royal_activations") is True,
        "C1 unseen Royal-row activation rate is too high",
    )
    _require(gates.get("passed") is True, "C1 data coverage gate failed")


def _validate_data(
    train_path: Path,
    validation_path: Path,
    split_receipt_path: Path,
    wdl_path: Path,
    expected_records: tuple[int, int],
    *,
    require_production_coverage: bool,
) -> dict[str, Any]:
    train_resolved = train_path.expanduser().resolve()
    validation_resolved = validation_path.expanduser().resolve()
    with HordeBinV1Dataset(train_resolved) as train, HordeBinV1Dataset(
        validation_resolved
    ) as validation:
        _require(len(train) == expected_records[0], "training record count violates C1")
        _require(len(validation) == expected_records[1], "validation record count violates C1")
        try:
            data = validate_dataset_pair(
                train_resolved,
                validation_resolved,
                train.manifest,
                validation.manifest,
                split_receipt_path,
            )
            overlap = audit_pair(
                train_resolved,
                validation_resolved,
                example_limit=8,
                require_zero=True,
            )
        except (TrainingError, AuditError, wire.FormatError) as error:
            raise CampaignError(f"C1 dataset pair is invalid: {error}") from error
        _require(
            data["book_split"]["schema"] == "HORDE_TRAINING_BOOK_SPLIT_V2",
            "C1 requires the reflection-safe V2 book split",
        )
        _require(overlap.get("zero_cross_role_overlap") is True, "C1 roles overlap")
        _require(
            overlap["physical"]["cross_role_overlap_samples"] == 0,
            "physical positions overlap between C1 roles",
        )
        _require(
            overlap["legacy_model_input"]["cross_role_overlap_samples"] == 0,
            "legacy evaluator inputs overlap between C1 roles",
        )
        coverage = _coverage_receipt(train, validation)
        _validate_coverage_receipt(coverage, expected_records)
        if require_production_coverage:
            _require_production_coverage(coverage)
        data["overlap_audit"] = overlap
        data["coverage"] = coverage
        data["wdl_calibration"] = _validate_wdl(
            wdl_path,
            data,
            train.manifest_sha256,
        )
        return data


def _run_plan(
    architecture: Mapping[str, Any],
    seed_index: int,
    seed: int,
    training: Mapping[str, Any],
) -> dict[str, object]:
    name = str(architecture["name"])
    output_role = f"seed-{seed_index + 1:02d}/{name}"
    training_command = [
        "python",
        "tools/horde_training_control.py",
        "{TRAIN_FILE}",
        "{VALIDATION_FILE}",
        "--architecture",
        name,
        "--book-split-receipt",
        "{BOOK_SPLIT_RECEIPT}",
        "--wdl-calibration",
        "{WDL_CALIBRATION}",
        "--output",
        f"{{RUNS_ROOT}}/{output_role}",
        "--seed",
        str(seed),
        "--epochs",
        str(training["epochs"]),
        "--batch-size",
        str(training["batch_size"]),
        "--block-size",
        str(training["block_size"]),
        "--lambda",
        str(training["lambda"]),
        "--learning-rate",
        str(training["learning_rate"]),
        "--scheduler-gamma",
        str(training["scheduler_gamma"]),
        "--device",
        str(training["device"]["type"]),
        "--cpu-threads",
        str(training["device"]["cpu_threads"]),
    ]
    export_command = [
        "python",
        "tools/horde_v2_export.py",
        "--checkpoint",
        f"{{RUNS_ROOT}}/{output_role}/checkpoint.pt",
        "--training-receipt",
        f"{{RUNS_ROOT}}/{output_role}/receipt.json",
        "--output",
        f"{{RUNS_ROOT}}/{output_role}/network.hsv2",
        "--export-receipt",
        f"{{RUNS_ROOT}}/{output_role}/export-receipt.json",
    ]
    seed_value, seed_digest = _seed(seed_index)
    _require(seed_value == seed, "run seed contradicts its derivation")
    return {
        "id": f"c1-s{seed_index + 1:02d}-{name}",
        "pair_index": seed_index,
        "seed": seed,
        "seed_derivation_sha256": seed_digest,
        "architecture": dict(architecture),
        "output_role": output_role,
        "training_command": training_command,
        "export_command": export_command,
    }


def _campaign_identity(plan: Mapping[str, Any]) -> str:
    data = _mapping(plan.get("data"), "campaign data")
    identity_payload = {
        "contract_sha256": _mapping(plan.get("contract"), "campaign contract").get(
            "sha256"
        ),
        "source": plan.get("source"),
        "train_file": data.get("train_file"),
        "validation_file": data.get("validation_file"),
        "book_split": data.get("book_split"),
        "wdl_calibration": data.get("wdl_calibration"),
    }
    return _sha256_bytes(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )


def _validate_plan_against_contract(
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_sha: str,
    *,
    allow_fixture: bool,
) -> None:
    _require(plan.get("schema") == PLAN_SCHEMA, "C1 campaign plan schema mismatch")
    plan_contract = _mapping(plan.get("contract"), "plan contract")
    _require(plan_contract.get("schema") == CONTRACT_SCHEMA, "plan contract schema drifted")
    _require(plan_contract.get("sha256") == contract_sha, "plan contract identity drifted")
    _require(
        plan_contract.get("path") == CONTRACT_RELATIVE_PATH.as_posix(),
        "plan contract path drifted",
    )
    source = _mapping(plan.get("source"), "plan source")
    _require(_valid_commit(source.get("commit")), "plan source commit is invalid")
    _require(type(source.get("dirty")) is bool, "plan source dirty flag is invalid")
    claims = _mapping(plan.get("claims"), "plan claims")
    fixture_mode = claims.get("fixture_mode") is True
    _require(allow_fixture or not fixture_mode, "fixture campaign is forbidden")
    _require(
        claims.get("campaign_inputs_eligible") is (not fixture_mode and not source["dirty"]),
        "plan input-eligibility claim drifted",
    )
    for claim in (
        "training_started",
        "training_complete",
        "architecture_selected",
        "strength_evidence",
        "production_network",
    ):
        _require(claims.get(claim) is False, f"plan claim {claim} is unsupported")

    dependencies = _mapping(plan.get("dependencies"), "plan dependencies")
    _require(
        dependencies.get("rank8_control") == _rank8_dependency(contract),
        "plan Rank-8 dependency drifted",
    )
    _require(dependencies.get("run6b_sha256") == wire.RUN6B_SHA256, "plan Run 6B drifted")
    data = _mapping(plan.get("data"), "plan data")
    train_file = _mapping(data.get("train_file"), "plan training file")
    validation_file = _mapping(data.get("validation_file"), "plan validation file")
    _require(
        train_file.get("sha256") != validation_file.get("sha256"),
        "plan data files are identical",
    )
    _require(
        train_file.get("book_sha256") != validation_file.get("book_sha256"),
        "plan data books are identical",
    )
    teacher = _mapping(data.get("teacher"), "plan teacher")
    network = _mapping(teacher.get("network"), "plan teacher network")
    _require(network.get("sha256") == wire.RUN6B_SHA256, "plan teacher is not Run 6B")
    _require(
        network.get("schema") == "HORDETEST_HP_LEGACY_V1",
        "plan teacher schema drifted",
    )
    split = _mapping(data.get("book_split"), "plan book split")
    _require(split.get("schema") == "HORDE_TRAINING_BOOK_SPLIT_V2", "plan split drifted")
    _require(split.get("disjoint_position_keys") is True, "plan split is not disjoint")
    _require(split.get("complete_partition") is True, "plan split is incomplete")
    overlap = _mapping(data.get("overlap_audit"), "plan overlap audit")
    _require(overlap.get("zero_cross_role_overlap") is True, "plan roles overlap")
    _require(
        _mapping(overlap.get("physical"), "plan physical overlap").get(
            "cross_role_overlap_samples"
        )
        == 0,
        "plan physical roles overlap",
    )
    _require(
        _mapping(overlap.get("legacy_model_input"), "plan legacy overlap").get(
            "cross_role_overlap_samples"
        )
        == 0,
        "plan legacy-input roles overlap",
    )
    coverage = _mapping(data.get("coverage"), "plan coverage receipt")
    coverage_gates = _mapping(coverage.get("gates"), "plan coverage gates")
    coverage_contract = _mapping(contract.get("data"), "contract data")
    frozen_coverage = _mapping(coverage_contract.get("coverage"), "contract coverage")
    _require(coverage.get("schema") == COVERAGE_SCHEMA, "plan coverage schema drifted")
    _require(
        coverage_gates.get("royal_bucket_position_minimums")
        == frozen_coverage.get("royal_bucket_position_minimums")
        and coverage_gates.get("validation_stm_white_piece_bin_minimum")
        == frozen_coverage.get("validation_stm_white_piece_bin_minimum")
        and coverage_gates.get("side_result_classes_required")
        == frozen_coverage.get("side_result_classes_required"),
        "plan coverage thresholds drifted",
    )
    unseen_coverage = _mapping(
        coverage.get("unseen_validation_royal_activations"),
        "plan unseen Royal coverage",
    )
    _require(
        unseen_coverage.get("maximum_exclusive")
        == frozen_coverage.get("unseen_validation_royal_activation_fraction_maximum_exclusive"),
        "plan unseen Royal threshold drifted",
    )
    _validate_coverage_receipt(
        coverage,
        (int(train_file["records"]), int(validation_file["records"])),
    )
    if not fixture_mode:
        _require_production_coverage(coverage)
    wdl = _mapping(data.get("wdl_calibration"), "plan WDL calibration")
    _require(wdl.get("schema") == "HORDE_WDL_CALIBRATION_V1", "plan WDL schema drifted")

    configuration = _mapping(plan.get("configuration"), "plan configuration")
    contract_data = _mapping(contract.get("data"), "contract data")
    contract_training = _mapping(contract.get("training"), "contract training")
    expected_training_records = int(configuration.get("training_records", 0))
    expected_validation_records = int(configuration.get("validation_records", 0))
    _require(expected_training_records > 0, "plan training record count is invalid")
    _require(expected_validation_records > 0, "plan validation record count is invalid")
    _require(
        _mapping(coverage.get("train"), "plan training coverage").get("records")
        == expected_training_records
        and _mapping(coverage.get("validation"), "plan validation coverage").get("records")
        == expected_validation_records,
        "plan coverage record counts drifted",
    )
    if not fixture_mode:
        _require(
            expected_training_records == contract_data.get("training_records"),
            "production plan training record count drifted",
        )
        _require(
            expected_validation_records == contract_data.get("validation_records"),
            "production plan validation record count drifted",
        )
    _require(train_file.get("records") == expected_training_records, "plan train count differs")
    _require(
        validation_file.get("records") == expected_validation_records,
        "plan validation count differs",
    )
    for key in (
        "epochs",
        "batch_size",
        "block_size",
        "lambda",
        "learning_rate",
        "scheduler_gamma",
        "optimizer",
        "device",
    ):
        _require(
            configuration.get(key) == contract_training.get(key),
            f"plan configuration field {key} drifted",
        )
    expected_exposures = expected_training_records * int(contract_training["epochs"])
    _require(
        configuration.get("exposures_per_model") == expected_exposures,
        "plan exposure count drifted",
    )
    if not fixture_mode:
        _require(
            expected_exposures == contract_training.get("training_example_exposures_per_model"),
            "production plan exposure count drifted",
        )
    expected_steps = int(contract_training["epochs"]) * math.ceil(
        expected_training_records / int(contract_training["batch_size"])
    )
    _require(
        configuration.get("optimizer_steps_per_model") == expected_steps,
        "plan optimizer step count drifted",
    )

    architectures = contract_training.get("architectures")
    seeds = _mapping(contract_training.get("paired_seeds"), "contract seeds").get("values")
    _require(isinstance(architectures, list) and isinstance(seeds, list), "contract matrix missing")
    expected_runs = [
        _run_plan(architecture, seed_index, seed, contract_training)
        for seed_index, seed in enumerate(seeds)
        for architecture in architectures
    ]
    _require(plan.get("runs") == expected_runs, "planned nine-run matrix drifted")
    _require(
        plan.get("selection") == contract.get("selection"),
        "plan selection gates drifted",
    )
    _require(
        plan.get("campaign_identity_sha256") == _campaign_identity(plan),
        "campaign identity drifted",
    )


def plan_campaign(
    train_path: Path,
    validation_path: Path,
    split_receipt_path: Path,
    wdl_path: Path,
    *,
    contract_path: Path | None = None,
    _expected_records: tuple[int, int] | None = None,
    _allow_dirty: bool = False,
    _source_override: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    contract, contract_sha = load_contract(contract_path)
    data_contract = _mapping(contract.get("data"), "contract data section")
    production_records = (
        int(data_contract["training_records"]),
        int(data_contract["validation_records"]),
    )
    expected_records = _expected_records or production_records
    _require(
        all(type(value) is int and value > 0 for value in expected_records),
        "expected record counts are invalid",
    )
    fixture_mode = expected_records != production_records
    source = (
        dict(_source_override)
        if _source_override is not None
        else _repository_identity(REPOSITORY_ROOT)
    )
    _require(
        _valid_commit(source.get("commit")) and type(source.get("dirty")) is bool,
        "campaign source override is invalid",
    )
    _require(_allow_dirty or not source["dirty"], "campaign source tree is dirty")
    data = _validate_data(
        train_path,
        validation_path,
        split_receipt_path,
        wdl_path,
        expected_records,
        require_production_coverage=not fixture_mode,
    )
    training = _mapping(contract.get("training"), "contract training section")
    architectures = training["architectures"]
    seeds = training["paired_seeds"]["values"]
    runs = [
        _run_plan(architecture, seed_index, seed, training)
        for seed_index, seed in enumerate(seeds)
        for architecture in architectures
    ]
    _require(len(runs) == training["run_count"], "planned C1 run count drifted")
    exposures = expected_records[0] * int(training["epochs"])
    if not fixture_mode:
        _require(
            exposures == training["training_example_exposures_per_model"],
            "C1 exposure count drifted",
        )

    plan = {
        "schema": PLAN_SCHEMA,
        "campaign_identity_sha256": "",
        "contract": {
            "path": CONTRACT_RELATIVE_PATH.as_posix(),
            "sha256": contract_sha,
            "schema": CONTRACT_SCHEMA,
        },
        "source": source,
        "dependencies": {
            "rank8_control": _rank8_dependency(contract),
            "run6b_sha256": wire.RUN6B_SHA256,
        },
        "data": data,
        "configuration": {
            "training_records": expected_records[0],
            "validation_records": expected_records[1],
            "epochs": training["epochs"],
            "exposures_per_model": exposures,
            "batch_size": training["batch_size"],
            "block_size": training["block_size"],
            "lambda": training["lambda"],
            "learning_rate": training["learning_rate"],
            "scheduler_gamma": training["scheduler_gamma"],
            "optimizer": training["optimizer"],
            "device": training["device"],
            "optimizer_steps_per_model": int(training["epochs"])
            * math.ceil(expected_records[0] / int(training["batch_size"])),
        },
        "runs": runs,
        "selection": contract["selection"],
        "claims": {
            "fixture_mode": fixture_mode,
            "campaign_inputs_eligible": not fixture_mode and not source["dirty"],
            "training_started": False,
            "training_complete": False,
            "architecture_selected": False,
            "strength_evidence": False,
            "production_network": False,
        },
    }
    plan["campaign_identity_sha256"] = _campaign_identity(plan)
    _validate_plan_against_contract(
        plan,
        contract,
        contract_sha,
        allow_fixture=fixture_mode,
    )
    return plan


def _same_identity(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    _require(actual == expected, f"{label} identity differs from the campaign plan")


def _verify_training_receipt(
    receipt: Mapping[str, Any],
    run: Mapping[str, Any],
    plan: Mapping[str, Any],
    run_directory: Path,
) -> dict[str, Any]:
    _require(receipt.get("schema") == TRAINING_RECEIPT_SCHEMA, "training receipt schema mismatch")
    _same_identity(
        _mapping(receipt.get("source"), "training source"),
        _mapping(plan.get("source"), "plan source"),
        "training source",
    )
    architecture = _mapping(receipt.get("architecture"), "training architecture")
    expected_architecture = _mapping(run.get("architecture"), "planned architecture")
    for key in ("name", "schema", "serialized_parameter_bytes", "training_structural_sha256"):
        receipt_key = "structural_sha256" if key == "training_structural_sha256" else key
        _require(
            architecture.get(receipt_key) == expected_architecture.get(key),
            f"{run['id']} architecture field {receipt_key} drifted",
        )

    data = _mapping(receipt.get("data"), "training data")
    expected_data = _mapping(plan.get("data"), "planned data")
    for key in ("train_file", "validation_file", "teacher", "wdl_calibration"):
        _same_identity(
            _mapping(data.get(key), f"training data {key}"),
            _mapping(expected_data.get(key), f"planned data {key}"),
            key,
        )
    book_split = _mapping(data.get("book_split"), "training book split")
    expected_split = _mapping(expected_data.get("book_split"), "planned book split")
    for key in (
        "receipt_sha256",
        "schema",
        "source",
        "assignment",
        "disjoint_position_keys",
        "complete_partition",
    ):
        _require(book_split.get(key) == expected_split.get(key), f"book split field {key} drifted")
    overlap = _mapping(data.get("overlap_audit"), "training overlap audit")
    _require(overlap.get("zero_cross_role_overlap") is True, "training receipt reports overlap")
    _require(
        _mapping(overlap.get("physical"), "physical overlap").get("cross_role_overlap_samples")
        == 0,
        "training receipt reports physical overlap",
    )
    _require(
        _mapping(overlap.get("legacy_model_input"), "legacy overlap").get(
            "cross_role_overlap_samples"
        )
        == 0,
        "training receipt reports legacy-input overlap",
    )

    configuration = _mapping(plan.get("configuration"), "plan configuration")
    run_receipt = _mapping(receipt.get("run"), "training run")
    _require(run_receipt.get("seed") == run.get("seed"), f"{run['id']} seed drifted")
    _require(run_receipt.get("complete") is True, f"{run['id']} is incomplete")
    _require(
        run_receipt.get("target_epochs") == configuration.get("epochs"),
        f"{run['id']} epoch target drifted",
    )
    _require(
        run_receipt.get("target_steps") == configuration.get("optimizer_steps_per_model")
        and run_receipt.get("optimizer_steps") == configuration.get("optimizer_steps_per_model"),
        f"{run['id']} optimizer step count drifted",
    )
    _require(
        run_receipt.get("samples_consumed") == configuration.get("exposures_per_model"),
        f"{run['id']} exposure count drifted",
    )
    _require(
        run_receipt.get("batch_size") == configuration.get("batch_size"),
        f"{run['id']} batch size drifted",
    )
    shuffle = _mapping(run_receipt.get("shuffle"), "training shuffle")
    _require(
        shuffle.get("block_size") == configuration.get("block_size"),
        f"{run['id']} shuffle block size drifted",
    )
    epochs = run_receipt.get("epochs_receipt")
    _require(
        isinstance(epochs, list) and len(epochs) == configuration.get("epochs"),
        f"{run['id']} epoch receipts are incomplete",
    )
    for epoch in epochs:
        epoch_value = _mapping(epoch, "epoch receipt")
        _require(
            _mapping(epoch_value.get("train"), "epoch training metrics").get("samples")
            == configuration.get("training_records"),
            f"{run['id']} epoch training sample count drifted",
        )
        _require(
            _mapping(epoch_value.get("validation"), "epoch validation metrics").get("samples")
            == configuration.get("validation_records"),
            f"{run['id']} epoch validation sample count drifted",
        )

    labels = _mapping(receipt.get("labels"), "training labels")
    optimizer = _mapping(receipt.get("optimizer"), "training optimizer")
    scheduler = _mapping(optimizer.get("scheduler"), "training scheduler")
    expected_optimizer = _mapping(configuration.get("optimizer"), "planned optimizer")
    _require(labels.get("lambda") == configuration.get("lambda"), f"{run['id']} lambda drifted")
    _require(
        optimizer.get("base_learning_rate") == configuration.get("learning_rate"),
        f"{run['id']} learning rate drifted",
    )
    _require(
        scheduler.get("gamma") == configuration.get("scheduler_gamma"),
        f"{run['id']} scheduler gamma drifted",
    )
    for key in (
        "name",
        "betas",
        "epsilon",
        "weight_decay",
        "foreach",
        "output_learning_rate_multiplier",
        "lookahead",
        "gradient_centralization",
    ):
        _require(
            optimizer.get(key) == expected_optimizer.get(key),
            f"{run['id']} optimizer field {key} drifted",
        )
    expected_scheduler = _mapping(expected_optimizer.get("scheduler"), "planned scheduler")
    for key in ("name", "step_size_epochs", "gamma"):
        _require(
            scheduler.get(key) == expected_scheduler.get(key),
            f"{run['id']} scheduler field {key} drifted",
        )

    artifacts = _mapping(receipt.get("artifacts"), "training artifacts")
    artifact_hashes: dict[str, str] = {}
    for role, filename in (("checkpoint", "checkpoint.pt"), ("metrics", "metrics.jsonl")):
        artifact = _mapping(artifacts.get(role), f"training {role} artifact")
        path = run_directory / filename
        _require(path.is_file(), f"{run['id']} {filename} is missing")
        digest = sha256_file(path)
        _require(artifact.get("name") == filename, f"{run['id']} {role} name drifted")
        _require(artifact.get("sha256") == digest, f"{run['id']} {role} hash drifted")
        artifact_hashes[role] = digest

    claims = _mapping(receipt.get("claims"), "training claims")
    _require(claims.get("integration_only") is True, "training receipt left integration scope")
    _require(claims.get("strength_eligible") is False, "training receipt claims strength eligibility")
    _require(claims.get("strength_evidence") is False, "training receipt claims strength")
    _require(claims.get("production_network") is False, "training receipt claims production")
    return {
        "environment": dict(_mapping(receipt.get("environment"), "training environment")),
        "sample_order_chain_sha256": run_receipt.get("sample_order_chain_sha256"),
        "checkpoint_sha256": artifact_hashes["checkpoint"],
        "metrics_sha256": artifact_hashes["metrics"],
    }


def _verify_export(
    run: Mapping[str, Any],
    plan: Mapping[str, Any],
    run_directory: Path,
    training_receipt_path: Path,
    training_evidence: Mapping[str, Any],
) -> dict[str, object]:
    network_path = run_directory / "network.hsv2"
    export_path = run_directory / "export-receipt.json"
    export, export_payload = _read_json(export_path, f"{run['id']} export receipt")
    _require(export.get("schema") == EXPORT_RECEIPT_SCHEMA, "export receipt schema mismatch")
    try:
        parsed = read_container(network_path)
    except ContainerError as error:
        raise CampaignError(f"{run['id']} container is invalid: {error}") from error
    architecture = _mapping(run.get("architecture"), "planned architecture")
    _require(parsed.spec.architecture == architecture.get("name"), "container architecture drifted")
    _require(parsed.spec.schema_name == architecture.get("schema"), "container schema drifted")
    container = _mapping(export.get("container"), "exported container")
    _require(container.get("file_sha256") == parsed.file_sha256, "container file hash drifted")
    _require(
        container.get("parameter_sha256") == parsed.parameter_sha256,
        "container parameter hash drifted",
    )
    provenance = parsed.provenance
    source = _mapping(plan.get("source"), "plan source")
    data = _mapping(plan.get("data"), "plan data")
    _require(provenance.get("source_commit") == source.get("commit"), "container source drifted")
    _require(provenance.get("source_dirty") is False, "container source is dirty")
    _require(
        provenance.get("checkpoint_sha256") == training_evidence.get("checkpoint_sha256"),
        "container checkpoint identity drifted",
    )
    _require(
        provenance.get("training_receipt_sha256") == sha256_file(training_receipt_path),
        "container training receipt identity drifted",
    )
    _require(
        provenance.get("train_file_sha256") == data["train_file"]["sha256"],
        "container training data identity drifted",
    )
    _require(
        provenance.get("validation_file_sha256") == data["validation_file"]["sha256"],
        "container validation data identity drifted",
    )
    _require(
        provenance.get("wdl_calibration_sha256") == data["wdl_calibration"]["sha256"],
        "container WDL identity drifted",
    )
    claims = _mapping(export.get("claims"), "export claims")
    _require(claims.get("strength_evidence") is False, "export receipt claims strength")
    _require(claims.get("production_dispatch") is False, "export receipt claims production")
    return {
        "network_sha256": parsed.file_sha256,
        "parameter_sha256": parsed.parameter_sha256,
        "export_receipt_sha256": _sha256_bytes(export_payload),
    }


def verify_campaign(
    plan_path: Path,
    runs_root: Path,
    *,
    contract_path: Path | None = None,
    _allow_fixture: bool = False,
) -> dict[str, Any]:
    contract, contract_sha = load_contract(contract_path)
    plan, plan_payload = _read_json(plan_path, "C1 campaign plan")
    _require(plan_payload == _canonical_json(plan), "C1 campaign plan is not canonical JSON")
    _validate_plan_against_contract(
        plan,
        contract,
        contract_sha,
        allow_fixture=_allow_fixture,
    )
    plan_claims = _mapping(plan.get("claims"), "plan claims")
    fixture_mode = plan_claims.get("fixture_mode") is True
    _require(_allow_fixture or not fixture_mode, "fixture campaign cannot be verified for selection")
    _require(
        fixture_mode or plan_claims.get("campaign_inputs_eligible") is True,
        "campaign inputs were not eligible",
    )
    source = _mapping(plan.get("source"), "plan source")
    _require(fixture_mode or source.get("dirty") is False, "campaign source is dirty")
    root = runs_root.expanduser().resolve()
    _require(root.is_dir(), f"runs root does not exist: {root}")
    runs = plan.get("runs")
    _require(isinstance(runs, list) and len(runs) == 9, "campaign plan does not contain nine runs")
    configuration = _mapping(plan.get("configuration"), "plan configuration")
    train_file = _mapping(_mapping(plan.get("data"), "plan data").get("train_file"), "plan train file")

    environment: dict[str, Any] | None = None
    sample_orders: dict[int, str] = {}
    expected_sample_orders: dict[int, str] = {}
    evidence: list[dict[str, object]] = []
    for run in runs:
        run_value = _mapping(run, "planned run")
        output_role = run_value.get("output_role")
        _require(isinstance(output_role, str), "planned output role is invalid")
        run_directory = (root / Path(output_role)).resolve()
        _require(
            run_directory == root or root in run_directory.parents,
            "planned output role escapes the runs root",
        )
        receipt_path = run_directory / "receipt.json"
        receipt, receipt_payload = _read_json(receipt_path, f"{run_value['id']} training receipt")
        training_evidence = _verify_training_receipt(
            receipt,
            run_value,
            plan,
            run_directory,
        )
        observed_environment = training_evidence.pop("environment")
        if environment is None:
            environment = dict(observed_environment)
        else:
            _require(observed_environment == environment, "C1 run environments differ")
        pair_index = int(run_value["pair_index"])
        order = training_evidence["sample_order_chain_sha256"]
        _require(isinstance(order, str) and len(order) == 64, "sample-order hash is invalid")
        if pair_index not in expected_sample_orders:
            expected_sample_orders[pair_index] = sample_order_chain_sha256(
                int(configuration["training_records"]),
                int(configuration["batch_size"]),
                int(configuration["block_size"]),
                int(run_value["seed"]),
                int(configuration["epochs"]),
                str(train_file["payload_sha256"]),
            )
        _require(
            order == expected_sample_orders[pair_index],
            f"sample order differs from deterministic schedule for seed {pair_index}",
        )
        previous_order = sample_orders.setdefault(pair_index, order)
        _require(previous_order == order, f"paired sample order differs for seed {pair_index}")
        export_evidence = _verify_export(
            run_value,
            plan,
            run_directory,
            receipt_path,
            training_evidence,
        )
        evidence.append(
            {
                "id": run_value["id"],
                "pair_index": pair_index,
                "architecture": run_value["architecture"]["name"],
                "seed": run_value["seed"],
                "training_receipt_sha256": _sha256_bytes(receipt_payload),
                "sample_order_chain_sha256": order,
                **training_evidence,
                **export_evidence,
            }
        )

    _require(environment is not None, "campaign environment is missing")
    device = _mapping(environment.get("device"), "campaign device")
    expected_device = _mapping(configuration.get("device"), "planned device")
    _require(device.get("type") == expected_device.get("type"), "campaign device type drifted")
    _require(
        fixture_mode or device.get("name") == expected_device.get("expected_name"),
        "campaign did not run on the frozen RTX 3080",
    )
    _require(
        device.get("cpu_threads") == expected_device.get("cpu_threads"),
        "campaign CPU thread count drifted",
    )
    _require(
        device.get("deterministic_algorithms") is True,
        "campaign did not use deterministic algorithms",
    )
    _require(environment.get("amp") is False, "campaign unexpectedly used AMP")
    _require(
        environment.get("cuda_matmul_allow_tf32") is False
        and environment.get("cudnn_allow_tf32") is False,
        "campaign unexpectedly used TF32",
    )
    return {
        "schema": VERIFICATION_SCHEMA,
        "contract_sha256": contract_sha,
        "plan_sha256": _sha256_bytes(plan_payload),
        "campaign_identity_sha256": plan.get("campaign_identity_sha256"),
        "source": source,
        "environment": environment,
        "runs": evidence,
        "paired_sample_order": {
            str(index): digest for index, digest in sorted(sample_orders.items())
        },
        "claims": {
            "fixture_mode": fixture_mode,
            "nine_runs_complete": True,
            "quantized_containers_authenticated": True,
            "training_evidence_complete": not fixture_mode,
            "paired_playing_gate_eligible": False,
            "architecture_selection_eligible": False,
            "architecture_selected": False,
            "strength_evidence": False,
            "production_network": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="validate inputs and write the nine-run plan")
    plan.add_argument("train", type=Path)
    plan.add_argument("validation", type=Path)
    plan.add_argument("--book-split-receipt", type=Path, required=True)
    plan.add_argument("--wdl-calibration", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--contract", type=Path)

    verify = subparsers.add_parser("verify", help="verify all trained and quantized runs")
    verify.add_argument("plan", type=Path)
    verify.add_argument("runs_root", type=Path)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--contract", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        result = plan_campaign(
            args.train,
            args.validation,
            args.book_split_receipt,
            args.wdl_calibration,
            contract_path=args.contract,
        )
    else:
        result = verify_campaign(
            args.plan,
            args.runs_root,
            contract_path=args.contract,
        )
    payload = _canonical_json(result)
    _write_exclusive(args.output, payload)
    print(payload.decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AuditError,
        CalibrationError,
        CampaignError,
        ContainerError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        TrainingError,
        wire.FormatError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
