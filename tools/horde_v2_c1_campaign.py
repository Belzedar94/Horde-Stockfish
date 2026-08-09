#!/usr/bin/env python3
"""Plan and verify the frozen Horde V2 C1 architecture campaign."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
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
        validate_selected_dataset_pair,
    )
    from .horde_training_decoder import HordeBinV1Dataset
    from .horde_training_split_audit import AuditError, audit_pair
    from .horde_training_selected_role import (
        CONTRACT_RELATIVE_PATH as DATA_REPAIR_CONTRACT_RELATIVE_PATH,
        CONTRACT_SCHEMA as DATA_REPAIR_CONTRACT_SCHEMA,
        CONTRACT_SHA256 as DATA_REPAIR_CONTRACT_SHA256,
        SelectedRoleDataset,
        SelectedRoleError,
        load_contract as load_data_repair_contract,
        verify_selected_role,
    )
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
        validate_selected_dataset_pair,
    )
    from horde_training_decoder import HordeBinV1Dataset
    from horde_training_split_audit import AuditError, audit_pair
    from horde_training_selected_role import (
        CONTRACT_RELATIVE_PATH as DATA_REPAIR_CONTRACT_RELATIVE_PATH,
        CONTRACT_SCHEMA as DATA_REPAIR_CONTRACT_SCHEMA,
        CONTRACT_SHA256 as DATA_REPAIR_CONTRACT_SHA256,
        SelectedRoleDataset,
        SelectedRoleError,
        load_contract as load_data_repair_contract,
        verify_selected_role,
    )
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
SEED_NAMESPACE = "HORDE_V2_C1_CAMPAIGN_V1"
COVERAGE_ADDENDUM_SCHEMA = "HORDE_V2_C1_COVERAGE_ADDENDUM_V1"
COVERAGE_ADDENDUM_RELATIVE_PATH = Path(
    "schemas/horde-v2-c1-coverage-addendum-v1.json"
)
COVERAGE_ADDENDUM_SHA256 = (
    "3103951AB8522238C23DBF83A3DEB0073D671024B0054ACBB75E1DC0C7C7D91B"
)
PLAN_SCHEMA = "HORDE_V2_C1_CAMPAIGN_PLAN_V2"
VERIFICATION_SCHEMA = "HORDE_V2_C1_CAMPAIGN_VERIFICATION_V2"
TRAINING_RECEIPT_SCHEMA = "HORDE_V2_BASE_TRAINING_V1"
EXPORT_RECEIPT_SCHEMA = "HORDE_V2_INTEGER_CHECKPOINT_EXPORT_V1"
LEGACY_COVERAGE_SCHEMA = "HORDE_V2_C1_DATA_COVERAGE_V1"
COVERAGE_SCHEMA = "HORDE_V2_C1_DATA_COVERAGE_V2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WHITE_PIECE_BINS = ((1, 4), (5, 8), (9, 16), (17, 24), (25, 30), (31, 36))
TOPOLOGY_SPECS = {
    "absolute_nonking": {"keys": 1, "dimensions": 640},
    "royal_rank8": {"keys": 8, "dimensions": 5_120},
    "royal32": {"keys": 32, "dimensions": 20_480},
}


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
    label = f"{SEED_NAMESPACE}:seed:{index}"
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


def _historical_selector_source(selected: SelectedRoleDataset) -> dict[str, object]:
    source = _mapping(selected.receipt.get("selector_source"), "selector source")
    commit = source.get("commit")
    relative = source.get("path")
    file_sha256 = source.get("file_sha256")
    _require(
        _valid_commit(commit)
        and source.get("dirty") is False
        and relative == "tools/horde_training_selected_role.py"
        and isinstance(file_sha256, str)
        and len(file_sha256) == 64,
        "selected-role historical source identity is invalid",
    )
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(commit), "HEAD"],
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _require(ancestry.returncode == 0, "selected-role source is not an ancestor of C1")
    blob = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout
    _require(
        _sha256_bytes(blob) == file_sha256
        and sha256_file(REPOSITORY_ROOT / str(relative)) == file_sha256,
        "selected-role implementation differs from its authenticated source",
    )
    return {"commit": str(commit).lower(), "dirty": False}


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
            "schema": LEGACY_COVERAGE_SCHEMA,
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


def _effective_contract_sha256(parent_sha256: str, addendum_sha256: str) -> str:
    return _sha256_bytes(
        json.dumps(
            {
                "coverage_addendum_sha256": addendum_sha256,
                "parent_contract_sha256": parent_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )


def load_coverage_addendum(path: Path | None = None) -> tuple[dict[str, Any], str]:
    resolved = (
        path or (REPOSITORY_ROOT / COVERAGE_ADDENDUM_RELATIVE_PATH)
    ).expanduser().resolve()
    addendum, payload = _read_json(resolved, "C1 coverage addendum")
    digest = _sha256_bytes(payload)
    _require(
        digest == COVERAGE_ADDENDUM_SHA256,
        f"C1 coverage addendum SHA-256 mismatch: {digest}",
    )
    _require(
        addendum.get("schema_name") == COVERAGE_ADDENDUM_SCHEMA,
        "C1 coverage addendum schema mismatch",
    )
    _require(
        addendum.get("status") == "registered_before_training",
        "C1 coverage addendum was not registered before training",
    )
    _require(
        addendum.get("parent_campaign")
        == {"schema": CONTRACT_SCHEMA, "sha256": CONTRACT_SHA256},
        "C1 coverage addendum targets another parent campaign",
    )
    _require(
        addendum.get("data_repair")
        == {
            "schema": DATA_REPAIR_CONTRACT_SCHEMA,
            "sha256": DATA_REPAIR_CONTRACT_SHA256,
        },
        "C1 coverage addendum targets another data repair",
    )
    _require(
        addendum.get("replaces_only") == "data.coverage",
        "C1 coverage addendum exceeds its registered scope",
    )
    timing = _mapping(addendum.get("amendment_timing"), "coverage amendment timing")
    _require(
        timing
        == {
            "original_preflight_receipt_preserved": True,
            "trainer_invocations_before_registration": 0,
            "epoch_zero_outputs_inspected": 0,
            "model_losses_inspected": 0,
            "quantized_outputs_inspected": 0,
            "playing_results_inspected": 0,
        },
        "C1 coverage amendment timing drifted",
    )
    effective = _mapping(addendum.get("effective_coverage"), "effective coverage")
    _require(
        effective.get("schema") == COVERAGE_SCHEMA
        and effective.get("exact_record_counts")
        == {"train": 250_000, "validation": 250_000}
        and effective.get("topologies") == TOPOLOGY_SPECS
        and effective.get("minimum_seen_validation_activation_mass")
        == {
            "numerator": 99,
            "denominator": 100,
            "comparison": (
                "unseen_validation_activations * 100 <= "
                "total_validation_activations"
            ),
        }
        and effective.get("validation_stm_white_piece_bin_minimum") == 1_000
        and effective.get("white_piece_bins")
        == [f"{minimum}-{maximum}" for minimum, maximum in WHITE_PIECE_BINS]
        and effective.get("side_result_classes_required") == [-1, 0, 1],
        "C1 effective coverage policy drifted",
    )
    for gate in (
        "zero_physical_cross_role_overlap",
        "zero_legacy_cross_role_overlap",
        "zero_validation_physical_duplicates",
        "zero_validation_legacy_duplicates",
        "all_topology_keys_nonzero_in_both_roles",
        "all_ten_fixed_roles_nonzero_in_both_roles",
        "every_validation_key_has_an_exact_training_row_intersection",
    ):
        _require(effective.get(gate) is True, f"C1 effective gate {gate} drifted")
    _require(
        addendum.get("seed_namespace") == SEED_NAMESPACE,
        "C1 seed namespace drifted",
    )
    return addendum, digest


def _validate_addendum_data_scope(
    data: Mapping[str, Any],
    addendum: Mapping[str, Any],
    *,
    train_manifest_sha256: str,
    candidate_manifest_sha256: str,
) -> None:
    scope = _mapping(addendum.get("applies_only_to"), "coverage addendum data scope")
    expected_train = _mapping(scope.get("training"), "addendum training identity")
    actual_train = _mapping(data.get("train_file"), "campaign training identity")
    _require(
        actual_train.get("sha256") == expected_train.get("file_sha256")
        and actual_train.get("payload_sha256") == expected_train.get("payload_sha256")
        and actual_train.get("records") == expected_train.get("records")
        and train_manifest_sha256 == expected_train.get("manifest_sha256"),
        "coverage addendum does not apply to this training dataset",
    )
    expected_candidate = _mapping(
        scope.get("validation_candidate"), "addendum validation candidate"
    )
    actual_candidate = _mapping(
        data.get("validation_candidate"), "campaign validation candidate"
    )
    _require(
        actual_candidate.get("sha256") == expected_candidate.get("file_sha256")
        and actual_candidate.get("payload_sha256")
        == expected_candidate.get("payload_sha256")
        and actual_candidate.get("records") == expected_candidate.get("records")
        and candidate_manifest_sha256 == expected_candidate.get("manifest_sha256"),
        "coverage addendum does not apply to this validation candidate",
    )
    expected_selected = _mapping(
        scope.get("selected_validation"), "addendum selected validation"
    )
    actual_validation = _mapping(
        data.get("validation_file"), "campaign selected validation"
    )
    actual_selection = _mapping(
        actual_validation.get("selected_role"), "campaign selected-role receipt"
    )
    _require(
        actual_validation.get("sha256") == expected_selected.get("materialized_sha256")
        and actual_validation.get("records") == expected_selected.get("records")
        and actual_selection.get("receipt_sha256")
        == expected_selected.get("receipt_sha256")
        and actual_selection.get("selected_index_sha256")
        == expected_selected.get("selected_index_sha256")
        and actual_selection.get("decision_chain_sha256")
        == expected_selected.get("decision_chain_sha256")
        and actual_selection.get("record_order_sha256")
        == expected_selected.get("record_order_sha256"),
        "coverage addendum does not apply to this selected validation role",
    )
    split = _mapping(data.get("book_split"), "campaign book split")
    wdl = _mapping(data.get("wdl_calibration"), "campaign WDL calibration")
    _require(
        split.get("receipt_sha256") == scope.get("book_split_receipt_sha256"),
        "coverage addendum book split identity drifted",
    )
    _require(
        wdl.get("sha256") == scope.get("wdl_calibration_sha256"),
        "coverage addendum WDL identity drifted",
    )


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


def _feature_rows(record: Any, topology: str) -> tuple[tuple[int, ...], int]:
    if topology == "absolute_nonking":
        return tuple(row for row in record.features.v2_global if row < 640), 0
    if topology == "royal_rank8":
        key = record.features.royal_bucket // 4
        return tuple(key * 640 + row % 640 for row in record.features.v2_royal), key
    _require(topology == "royal32", f"unknown C1 coverage topology: {topology}")
    return record.features.v2_royal, record.features.royal_bucket


def _row_set_sha256(rows: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows):
        digest.update(int(row).to_bytes(4, "little"))
    return digest.hexdigest().upper()


def _position_count_summary(counts: Sequence[int]) -> dict[str, int | float]:
    ordered = sorted(counts)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median: int | float = ordered[middle]
    else:
        total = ordered[middle - 1] + ordered[middle]
        median = total // 2 if total % 2 == 0 else total / 2
    return {"minimum": ordered[0], "median": median, "maximum": ordered[-1]}


def _seen_mass_gate(unseen_activations: int, total_activations: int) -> bool:
    _require(
        type(unseen_activations) is int
        and type(total_activations) is int
        and 0 <= unseen_activations <= total_activations
        and total_activations > 0,
        "C1 seen-mass inputs are invalid",
    )
    return unseen_activations * 100 <= total_activations


def _dataset_coverage(
    dataset: Any,
) -> tuple[dict[str, object], dict[str, dict[str, Any]]]:
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
    topology_support: dict[str, dict[str, Any]] = {}
    for topology, spec in TOPOLOGY_SPECS.items():
        key_count = int(spec["keys"])
        topology_support[topology] = {
            "position_counts": [0] * key_count,
            "row_counts": Counter(),
            "row_counts_by_key": [Counter() for _ in range(key_count)],
            "role_activations": [0] * 10,
            "role_activations_by_key": [[0] * 10 for _ in range(key_count)],
        }
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
        for topology, support in topology_support.items():
            rows, key = _feature_rows(record, topology)
            support["position_counts"][key] += 1
            support["row_counts"].update(rows)
            support["row_counts_by_key"][key].update(rows)
            for row in rows:
                role = row % 640 // 64
                support["role_activations"][role] += 1
                support["role_activations_by_key"][key][role] += 1
        promoted_horde_positions += int(any(2 <= code <= 5 for code in record.board))
        castling_positions += int(record.castling_rights != 0)
        en_passant_positions += int(record.ep_square != 64)
        best_played_divergence += int(record.best_move != record.played_move)
        mate_scores_masked += int(abs(record.score) >= MATE_SCORE_THRESHOLD)
        score_minimum = record.score if score_minimum is None else min(score_minimum, record.score)
        score_maximum = record.score if score_maximum is None else max(score_maximum, record.score)
        score_sum += record.score

    record_count = len(dataset)
    topology_summary: dict[str, object] = {}
    for topology, support in topology_support.items():
        row_counts = support["row_counts"]
        rows_by_key = support["row_counts_by_key"]
        position_counts = support["position_counts"]
        topology_summary[topology] = {
            "position_counts_by_key": position_counts,
            "position_count_summary": _position_count_summary(position_counts),
            "activation_counts_by_key": [sum(rows.values()) for rows in rows_by_key],
            "activation_counts_by_fixed_role": support["role_activations"],
            "activation_counts_by_key_and_fixed_role": support[
                "role_activations_by_key"
            ],
            "total_activations": sum(row_counts.values()),
            "unique_rows": len(row_counts),
            "unique_rows_by_key": [len(rows) for rows in rows_by_key],
            "row_set_sha256": _row_set_sha256(row_counts),
            "row_set_sha256_by_key": [
                _row_set_sha256(rows) for rows in rows_by_key
            ],
        }
    royal_rows = topology_support["royal32"]["row_counts"]
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
            "topology_support": topology_summary,
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
        topology_support,
    )


def _legacy_coverage_receipt(
    train_summary: Mapping[str, Any],
    validation_summary: Mapping[str, Any],
    train_support: Mapping[str, Mapping[str, Any]],
    validation_support: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    train_rows = train_support["royal32"]["row_counts"]
    validation_rows = validation_support["royal32"]["row_counts"]
    unseen_activations = sum(
        count for row, count in validation_rows.items() if row not in train_rows
    )
    total = int(validation_summary["royal_row_activations"])
    fraction = unseen_activations / total
    bucket_gate = (
        min(train_summary["royal_bucket_positions"]) >= 500
        and min(validation_summary["royal_bucket_positions"]) >= 200
    )
    stm_gate = all(
        int(validation_summary["stm_white_piece_bins"].get(side, {}).get(label, 0))
        >= 1_000
        for side in ("white", "black")
        for label in (f"{minimum}-{maximum}" for minimum, maximum in WHITE_PIECE_BINS)
    )
    result_gate = all(
        {int(value) for value in summary["side_result_classes"].get(side, {})}
        == {-1, 0, 1}
        for summary in (train_summary, validation_summary)
        for side in ("white", "black")
    )
    unseen_gate = fraction < 0.001
    return {
        "schema": LEGACY_COVERAGE_SCHEMA,
        "royal_bucket_position_minimums": {"train": 500, "validation": 200},
        "observed_royal_bucket_position_minimums": {
            "train": min(train_summary["royal_bucket_positions"]),
            "validation": min(validation_summary["royal_bucket_positions"]),
        },
        "unseen_validation_royal_activations": {
            "count": unseen_activations,
            "total": total,
            "fraction": fraction,
            "maximum_exclusive": 0.001,
        },
        "gates": {
            "royal_bucket_coverage": bucket_gate,
            "validation_stm_white_piece_bins": stm_gate,
            "side_result_classes": result_gate,
            "unseen_validation_royal_activations": unseen_gate,
            "passed": bucket_gate and stm_gate and result_gate and unseen_gate,
        },
    }


def _coverage_receipt(
    train: Any,
    validation: Any,
    overlap: Mapping[str, Any],
) -> dict[str, object]:
    train_summary, train_support = _dataset_coverage(train)
    validation_summary, validation_support = _dataset_coverage(validation)
    parent_preflight = _legacy_coverage_receipt(
        train_summary,
        validation_summary,
        train_support,
        validation_support,
    )
    stm_gate = all(
        int(validation_summary["stm_white_piece_bins"].get(side, {}).get(label, 0))
        >= 1_000
        for side in ("white", "black")
        for label in (f"{minimum}-{maximum}" for minimum, maximum in WHITE_PIECE_BINS)
    )
    result_gate = all(
        {int(value) for value in summary["side_result_classes"].get(side, {})}
        == {-1, 0, 1}
        for summary in (train_summary, validation_summary)
        for side in ("white", "black")
    )
    training_row_sets = {
        topology: set(train_support[topology]["row_counts"])
        for topology in TOPOLOGY_SPECS
    }
    positions_with_unseen = {topology: 0 for topology in TOPOLOGY_SPECS}
    for index in range(len(validation)):
        record = validation.record(index)
        for topology in TOPOLOGY_SPECS:
            rows, _key = _feature_rows(record, topology)
            positions_with_unseen[topology] += int(
                any(row not in training_row_sets[topology] for row in rows)
            )

    topology_receipts: dict[str, object] = {}
    topology_key_gates: dict[str, bool] = {}
    fixed_role_gates: dict[str, bool] = {}
    intersection_gates: dict[str, bool] = {}
    seen_mass_gates: dict[str, bool] = {}
    for topology, spec in TOPOLOGY_SPECS.items():
        train_topology = train_summary["topology_support"][topology]
        validation_topology = validation_summary["topology_support"][topology]
        train_rows_by_key = train_support[topology]["row_counts_by_key"]
        validation_rows_by_key = validation_support[topology]["row_counts_by_key"]
        keys: list[dict[str, object]] = []
        unseen_activations = 0
        unseen_unique_rows: set[int] = set()
        for key, (train_rows, validation_rows) in enumerate(
            zip(train_rows_by_key, validation_rows_by_key, strict=True)
        ):
            intersection = set(train_rows) & set(validation_rows)
            unseen_rows = set(validation_rows) - set(train_rows)
            key_unseen = sum(validation_rows[row] for row in unseen_rows)
            unseen_activations += key_unseen
            unseen_unique_rows.update(unseen_rows)
            keys.append(
                {
                    "key": key,
                    "train_positions": train_topology["position_counts_by_key"][key],
                    "validation_positions": validation_topology[
                        "position_counts_by_key"
                    ][key],
                    "train_activations": train_topology["activation_counts_by_key"][key],
                    "validation_activations": validation_topology[
                        "activation_counts_by_key"
                    ][key],
                    "train_unique_rows": len(train_rows),
                    "validation_unique_rows": len(validation_rows),
                    "intersection_unique_rows": len(intersection),
                    "unseen_validation_unique_rows": len(unseen_rows),
                    "unseen_validation_activations": key_unseen,
                    "train_row_set_sha256": _row_set_sha256(train_rows),
                    "validation_row_set_sha256": _row_set_sha256(validation_rows),
                }
            )
        total = int(validation_topology["total_activations"])
        topology_key_gates[topology] = all(
            count > 0
            for count in (
                *train_topology["position_counts_by_key"],
                *validation_topology["position_counts_by_key"],
            )
        )
        fixed_role_gates[topology] = all(
            count > 0
            for count in (
                *train_topology["activation_counts_by_fixed_role"],
                *validation_topology["activation_counts_by_fixed_role"],
            )
        )
        intersection_gates[topology] = all(
            item["intersection_unique_rows"] > 0 for item in keys
        )
        seen_mass_gates[topology] = _seen_mass_gate(unseen_activations, total)
        topology_receipts[topology] = {
            "dimensions": spec["dimensions"],
            "key_count": spec["keys"],
            "train": train_topology,
            "validation": validation_topology,
            "keys": keys,
            "unseen_validation": {
                "activation_count": unseen_activations,
                "activation_total": total,
                "activation_fraction": unseen_activations / total,
                "unique_rows": len(unseen_unique_rows),
                "positions_with_any_unseen_row": positions_with_unseen[topology],
                "seen_mass_minimum_numerator": 99,
                "seen_mass_minimum_denominator": 100,
            },
            "gates": {
                "all_keys_nonzero_in_both_roles": topology_key_gates[topology],
                "all_fixed_roles_nonzero_in_both_roles": fixed_role_gates[topology],
                "every_validation_key_has_training_row_intersection": (
                    intersection_gates[topology]
                ),
                "seen_validation_activation_mass_at_least_99_over_100": (
                    seen_mass_gates[topology]
                ),
            },
        }

    physical = _mapping(overlap.get("physical"), "physical overlap")
    legacy = _mapping(overlap.get("legacy_model_input"), "legacy overlap")
    boolean_gates = {
        "exact_record_counts": len(train) == 250_000 and len(validation) == 250_000,
        "zero_physical_cross_role_overlap": physical.get("cross_role_overlap_samples") == 0,
        "zero_legacy_cross_role_overlap": legacy.get("cross_role_overlap_samples") == 0,
        "zero_validation_physical_duplicates": physical.get("validation_duplicate_samples") == 0,
        "zero_validation_legacy_duplicates": legacy.get("validation_duplicate_samples") == 0,
        "all_topology_keys_nonzero_in_both_roles": all(topology_key_gates.values()),
        "all_ten_fixed_roles_nonzero_in_both_roles": all(fixed_role_gates.values()),
        "every_validation_key_has_training_row_intersection": all(
            intersection_gates.values()
        ),
        "all_topologies_seen_validation_activation_mass_at_least_99_over_100": all(
            seen_mass_gates.values()
        ),
        "validation_stm_white_piece_bins": stm_gate,
        "side_result_classes": result_gate,
    }
    return {
        "schema": COVERAGE_SCHEMA,
        "parent_preflight": parent_preflight,
        "train": train_summary,
        "validation": validation_summary,
        "topologies": topology_receipts,
        "gates": {
            **boolean_gates,
            "validation_stm_white_piece_bin_minimum": 1_000,
            "side_result_classes_required": [-1, 0, 1],
            "passed": all(boolean_gates.values()),
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
        support = _mapping(summary.get("topology_support"), f"C1 {role} topologies")
        _require(
            set(support) == set(TOPOLOGY_SPECS),
            f"C1 {role} topology set drifted",
        )

    parent = _mapping(coverage.get("parent_preflight"), "C1 parent preflight")
    _require(parent.get("schema") == LEGACY_COVERAGE_SCHEMA, "parent preflight schema drifted")
    parent_gates = _mapping(parent.get("gates"), "parent preflight gates")
    observed_minimums = _mapping(
        parent.get("observed_royal_bucket_position_minimums"),
        "parent observed bucket minimums",
    )
    parent_unseen = _mapping(
        parent.get("unseen_validation_royal_activations"),
        "parent unseen Royal rows",
    )
    expected_parent_bucket = (
        min(summaries["train"]["royal_bucket_positions"]) >= 500
        and min(summaries["validation"]["royal_bucket_positions"]) >= 200
    )
    expected_parent_unseen = (
        int(parent_unseen.get("count", -1))
        / int(parent_unseen.get("total", 0))
        < 0.001
    )
    _require(
        parent.get("royal_bucket_position_minimums")
        == {"train": 500, "validation": 200}
        and observed_minimums
        == {
            "train": min(summaries["train"]["royal_bucket_positions"]),
            "validation": min(summaries["validation"]["royal_bucket_positions"]),
        }
        and parent_unseen.get("maximum_exclusive") == 0.001
        and parent_unseen.get("total")
        == summaries["validation"]["royal_row_activations"]
        and parent_unseen.get("fraction")
        == parent_unseen.get("count") / parent_unseen.get("total")
        and parent_gates.get("royal_bucket_coverage") is expected_parent_bucket
        and parent_gates.get("unseen_validation_royal_activations")
        is expected_parent_unseen
        and parent_gates.get("passed") is False,
        "failed V1 coverage receipt was not preserved",
    )

    topology_receipts = _mapping(coverage.get("topologies"), "C1 topology receipts")
    _require(set(topology_receipts) == set(TOPOLOGY_SPECS), "C1 topology receipts drifted")
    topology_key_gates: dict[str, bool] = {}
    fixed_role_gates: dict[str, bool] = {}
    intersection_gates: dict[str, bool] = {}
    seen_mass_gates: dict[str, bool] = {}
    for topology, spec in TOPOLOGY_SPECS.items():
        receipt = _mapping(topology_receipts.get(topology), f"C1 {topology} receipt")
        _require(
            receipt.get("dimensions") == spec["dimensions"]
            and receipt.get("key_count") == spec["keys"],
            f"C1 {topology} dimensions drifted",
        )
        role_summaries = {
            role: _mapping(receipt.get(role), f"C1 {topology} {role}")
            for role in ("train", "validation")
        }
        key_count = int(spec["keys"])
        for role, summary in role_summaries.items():
            positions = summary.get("position_counts_by_key")
            key_activations = summary.get("activation_counts_by_key")
            roles = summary.get("activation_counts_by_fixed_role")
            roles_by_key = summary.get("activation_counts_by_key_and_fixed_role")
            expected = expected_records[0 if role == "train" else 1]
            _require(
                isinstance(positions, list)
                and len(positions) == key_count
                and all(type(count) is int and count >= 0 for count in positions)
                and sum(positions) == expected,
                f"C1 {topology} {role} key positions are invalid",
            )
            _require(
                isinstance(key_activations, list)
                and len(key_activations) == key_count
                and all(type(count) is int and count >= 0 for count in key_activations)
                and sum(key_activations) == summary.get("total_activations"),
                f"C1 {topology} {role} key activations are invalid",
            )
            _require(
                isinstance(roles, list)
                and len(roles) == 10
                and all(type(count) is int and count >= 0 for count in roles)
                and sum(roles) == summary.get("total_activations"),
                f"C1 {topology} {role} fixed-role activations are invalid",
            )
            _require(
                isinstance(roles_by_key, list)
                and len(roles_by_key) == key_count
                and all(
                    isinstance(row, list)
                    and len(row) == 10
                    and all(type(count) is int and count >= 0 for count in row)
                    and sum(row) == key_activations[index]
                    for index, row in enumerate(roles_by_key)
                ),
                f"C1 {topology} {role} key-role activations are invalid",
            )
        keys = receipt.get("keys")
        _require(
            isinstance(keys, list)
            and len(keys) == key_count
            and all(isinstance(item, dict) for item in keys)
            and [item["key"] for item in keys] == list(range(key_count)),
            f"C1 {topology} key diagnostics are invalid",
        )
        unseen = _mapping(receipt.get("unseen_validation"), f"C1 {topology} unseen rows")
        unseen_count = unseen.get("activation_count")
        unseen_total = unseen.get("activation_total")
        unseen_fraction = unseen.get("activation_fraction")
        _require(
            type(unseen_count) is int
            and type(unseen_total) is int
            and 0 <= unseen_count <= unseen_total
            and unseen_total == role_summaries["validation"].get("total_activations")
            and isinstance(unseen_fraction, (int, float))
            and math.isfinite(float(unseen_fraction))
            and float(unseen_fraction) == unseen_count / unseen_total
            and unseen.get("seen_mass_minimum_numerator") == 99
            and unseen.get("seen_mass_minimum_denominator") == 100
            and unseen_count
            == sum(int(item["unseen_validation_activations"]) for item in keys),
            f"C1 {topology} unseen-row receipt is inconsistent",
        )
        topology_key_gates[topology] = all(
            count > 0
            for summary in role_summaries.values()
            for count in summary["position_counts_by_key"]
        )
        fixed_role_gates[topology] = all(
            count > 0
            for summary in role_summaries.values()
            for count in summary["activation_counts_by_fixed_role"]
        )
        intersection_gates[topology] = all(
            type(item.get("intersection_unique_rows")) is int
            and item["intersection_unique_rows"] > 0
            for item in keys
        )
        seen_mass_gates[topology] = _seen_mass_gate(unseen_count, unseen_total)
        expected_topology_gates = {
            "all_keys_nonzero_in_both_roles": topology_key_gates[topology],
            "all_fixed_roles_nonzero_in_both_roles": fixed_role_gates[topology],
            "every_validation_key_has_training_row_intersection": intersection_gates[
                topology
            ],
            "seen_validation_activation_mass_at_least_99_over_100": seen_mass_gates[
                topology
            ],
        }
        _require(
            receipt.get("gates") == expected_topology_gates,
            f"C1 {topology} gates contradict their counts",
        )

    gates = _mapping(coverage.get("gates"), "C1 coverage gates")
    _require(
        gates.get("validation_stm_white_piece_bin_minimum") == 1_000
        and gates.get("side_result_classes_required") == [-1, 0, 1],
        "C1 coverage thresholds drifted",
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
    expected_booleans = {
        "exact_record_counts": expected_records == (250_000, 250_000),
        "all_topology_keys_nonzero_in_both_roles": all(topology_key_gates.values()),
        "all_ten_fixed_roles_nonzero_in_both_roles": all(fixed_role_gates.values()),
        "every_validation_key_has_training_row_intersection": all(
            intersection_gates.values()
        ),
        "all_topologies_seen_validation_activation_mass_at_least_99_over_100": all(
            seen_mass_gates.values()
        ),
        "validation_stm_white_piece_bins": expected_stm_gate,
        "side_result_classes": expected_result_gate,
    }
    for name in (
        "zero_physical_cross_role_overlap",
        "zero_legacy_cross_role_overlap",
        "zero_validation_physical_duplicates",
        "zero_validation_legacy_duplicates",
    ):
        _require(type(gates.get(name)) is bool, f"C1 overlap gate {name} is invalid")
        expected_booleans[name] = gates[name]
    _require(
        all(gates.get(name) is value for name, value in expected_booleans.items())
        and gates.get("passed") is all(expected_booleans.values()),
        "C1 coverage gate booleans contradict their counts",
    )


def _require_production_coverage(coverage: Mapping[str, Any]) -> None:
    gates = _mapping(coverage.get("gates"), "C1 coverage gates")
    _require(gates.get("exact_record_counts") is True, "C1 role counts drifted")
    _require(
        gates.get("zero_physical_cross_role_overlap") is True
        and gates.get("zero_legacy_cross_role_overlap") is True,
        "C1 roles overlap",
    )
    _require(
        gates.get("zero_validation_physical_duplicates") is True
        and gates.get("zero_validation_legacy_duplicates") is True,
        "C1 validation contains duplicate evaluator inputs",
    )
    _require(
        gates.get("all_topology_keys_nonzero_in_both_roles") is True,
        "C1 topology keys lack coverage",
    )
    _require(
        gates.get("all_ten_fixed_roles_nonzero_in_both_roles") is True,
        "C1 fixed roles lack coverage",
    )
    _require(
        gates.get("every_validation_key_has_training_row_intersection") is True,
        "C1 topology key lacks exact train/validation row support",
    )
    _require(
        gates.get(
            "all_topologies_seen_validation_activation_mass_at_least_99_over_100"
        )
        is True,
        "C1 unseen first-domain activation rate is too high",
    )
    _require(
        gates.get("validation_stm_white_piece_bins") is True,
        "C1 validation STM/piece bins lack coverage",
    )
    _require(
        gates.get("side_result_classes") is True,
        "C1 side-specific WDL classes lack coverage",
    )
    _require(gates.get("passed") is True, "C1 data coverage gate failed")


def _validate_data(
    train_path: Path,
    validation_path: Path,
    split_receipt_path: Path,
    wdl_path: Path,
    expected_records: tuple[int, int],
    *,
    coverage_addendum: Mapping[str, Any],
    require_production_coverage: bool,
    validation_candidate_path: Path | None,
) -> dict[str, Any]:
    train_resolved = train_path.expanduser().resolve()
    validation_resolved = validation_path.expanduser().resolve()
    _require(
        not require_production_coverage or validation_candidate_path is not None,
        "production C1 requires an authenticated selected validation role",
    )
    with ExitStack() as stack:
        train = stack.enter_context(HordeBinV1Dataset(train_resolved))
        candidate_manifest_sha256: str | None = None
        selection_verification: dict[str, object] | None = None
        if validation_candidate_path is None:
            validation = stack.enter_context(HordeBinV1Dataset(validation_resolved))
            validation_factory = HordeBinV1Dataset
            try:
                data = validate_dataset_pair(
                    train_resolved,
                    validation_resolved,
                    train.manifest,
                    validation.manifest,
                    split_receipt_path,
                )
            except (TrainingError, wire.FormatError) as error:
                raise CampaignError(f"C1 dataset pair is invalid: {error}") from error
        else:
            candidate_resolved = validation_candidate_path.expanduser().resolve()
            candidate = stack.enter_context(HordeBinV1Dataset(candidate_resolved))
            candidate_manifest_sha256 = candidate.manifest_sha256
            validation = stack.enter_context(SelectedRoleDataset(validation_resolved))
            validation_factory = SelectedRoleDataset
            try:
                selector_source = _historical_selector_source(validation)
                selection_verification = verify_selected_role(
                    train_resolved,
                    candidate_resolved,
                    validation_resolved,
                    _source_override=selector_source,
                )
                data = validate_selected_dataset_pair(
                    train_resolved,
                    candidate_resolved,
                    train,
                    candidate,
                    validation,
                    split_receipt_path,
                    source_override=selector_source,
                )
            except (SelectedRoleError, TrainingError, wire.FormatError) as error:
                raise CampaignError(f"C1 selected validation role is invalid: {error}") from error
        _require(len(train) == expected_records[0], "training record count violates C1")
        _require(len(validation) == expected_records[1], "validation record count violates C1")
        try:
            overlap = audit_pair(
                train_resolved,
                validation_resolved,
                example_limit=8,
                require_zero=True,
                validation_factory=validation_factory,
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
        if validation_candidate_path is not None:
            _require(
                overlap["physical"]["validation_duplicate_samples"] == 0,
                "selected validation contains duplicate physical positions",
            )
            _require(
                overlap["legacy_model_input"]["validation_duplicate_samples"] == 0,
                "selected validation contains duplicate legacy inputs",
            )
        coverage = _coverage_receipt(train, validation, overlap)
        _validate_coverage_receipt(coverage, expected_records)
        if require_production_coverage:
            _require_production_coverage(coverage)
        data["overlap_audit"] = overlap
        data["coverage"] = coverage
        if selection_verification is not None:
            data["validation_selection_verification"] = selection_verification
        data["wdl_calibration"] = _validate_wdl(
            wdl_path,
            data,
            train.manifest_sha256,
        )
        if require_production_coverage:
            _require(
                candidate_manifest_sha256 is not None,
                "production validation candidate manifest is absent",
            )
            _validate_addendum_data_scope(
                data,
                coverage_addendum,
                train_manifest_sha256=train.manifest_sha256,
                candidate_manifest_sha256=candidate_manifest_sha256,
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
    run_id = f"c1-s{seed_index + 1:02d}-{name}"
    training_command = [
        "python",
        "tools/horde_training_control.py",
        "{TRAIN_FILE}",
        "{VALIDATION_ROLE}",
        "--validation-selected-role",
        "--validation-candidate",
        "{VALIDATION_CANDIDATE}",
        "--architecture",
        name,
        "--book-split-receipt",
        "{BOOK_SPLIT_RECEIPT}",
        "--wdl-calibration",
        "{WDL_CALIBRATION}",
        "--campaign-plan",
        "{CAMPAIGN_PLAN}",
        "--campaign-run-id",
        run_id,
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
        "id": run_id,
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
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    )


def _validate_plan_against_contract(
    plan: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_sha: str,
    *,
    allow_fixture: bool,
) -> None:
    _coverage_addendum, coverage_addendum_sha = load_coverage_addendum()
    _require(plan.get("schema") == PLAN_SCHEMA, "C1 campaign plan schema mismatch")
    plan_contract = _mapping(plan.get("contract"), "plan contract")
    _require(plan_contract.get("schema") == CONTRACT_SCHEMA, "plan contract schema drifted")
    _require(plan_contract.get("sha256") == contract_sha, "plan contract identity drifted")
    _require(
        plan_contract.get("path") == CONTRACT_RELATIVE_PATH.as_posix(),
        "plan contract path drifted",
    )
    _require(
        plan_contract.get("coverage_addendum")
        == {
            "path": COVERAGE_ADDENDUM_RELATIVE_PATH.as_posix(),
            "sha256": coverage_addendum_sha,
            "schema": COVERAGE_ADDENDUM_SCHEMA,
        }
        and plan_contract.get("effective_sha256")
        == _effective_contract_sha256(contract_sha, coverage_addendum_sha),
        "plan effective contract identity drifted",
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
        dependencies.get("data_repair")
        == {
            "path": DATA_REPAIR_CONTRACT_RELATIVE_PATH.as_posix(),
            "schema": DATA_REPAIR_CONTRACT_SCHEMA,
            "sha256": DATA_REPAIR_CONTRACT_SHA256,
        },
        "plan data-repair dependency drifted",
    )
    _require(
        dependencies.get("coverage_addendum")
        == {
            "path": COVERAGE_ADDENDUM_RELATIVE_PATH.as_posix(),
            "schema": COVERAGE_ADDENDUM_SCHEMA,
            "sha256": coverage_addendum_sha,
        },
        "plan coverage-addendum dependency drifted",
    )
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
    if not fixture_mode:
        selected_identity = _mapping(
            validation_file.get("selected_role"),
            "plan selected validation identity",
        )
        _require(
            selected_identity.get("schema") == "HORDE_TRAINING_SELECTED_ROLE_V1"
            and selected_identity.get("contract_sha256") == DATA_REPAIR_CONTRACT_SHA256,
            "plan validation role is not the registered selected role",
        )
        _mapping(data.get("validation_candidate"), "plan validation candidate")
        _mapping(data.get("validation_selection"), "plan validation selection")
        selection_verification = _mapping(
            data.get("validation_selection_verification"),
            "plan validation selection verification",
        )
        selection_claims = _mapping(
            selection_verification.get("claims"),
            "plan validation selection claims",
        )
        _require(
            selection_claims.get("canonical_selection_recomputed") is True
            and selection_claims.get("materialized_records_reconstructed") is True
            and selection_claims.get("zero_cross_role_overlap") is True
            and selection_claims.get("zero_validation_duplicates") is True
            and selection_claims.get("training_eligible") is True,
            "plan selected validation role lacks independent verification",
        )
        _require(
            _mapping(overlap.get("physical"), "plan physical overlap").get(
                "validation_duplicate_samples"
            )
            == 0
            and _mapping(overlap.get("legacy_model_input"), "plan legacy overlap").get(
                "validation_duplicate_samples"
            )
            == 0,
            "plan selected validation role contains duplicate inputs",
        )
    coverage = _mapping(data.get("coverage"), "plan coverage receipt")
    coverage_gates = _mapping(coverage.get("gates"), "plan coverage gates")
    _require(coverage.get("schema") == COVERAGE_SCHEMA, "plan coverage schema drifted")
    _require(
        coverage_gates.get("validation_stm_white_piece_bin_minimum") == 1_000
        and coverage_gates.get("side_result_classes_required")
        == [-1, 0, 1],
        "plan coverage thresholds drifted",
    )
    _require(
        coverage_gates.get("zero_physical_cross_role_overlap")
        is (_mapping(overlap.get("physical"), "plan physical overlap").get(
            "cross_role_overlap_samples"
        ) == 0)
        and coverage_gates.get("zero_legacy_cross_role_overlap")
        is (_mapping(overlap.get("legacy_model_input"), "plan legacy overlap").get(
            "cross_role_overlap_samples"
        ) == 0)
        and coverage_gates.get("zero_validation_physical_duplicates")
        is (overlap["physical"].get("validation_duplicate_samples") == 0)
        and coverage_gates.get("zero_validation_legacy_duplicates")
        is (overlap["legacy_model_input"].get("validation_duplicate_samples") == 0),
        "plan coverage leakage gates contradict the overlap audit",
    )
    _validate_coverage_receipt(
        coverage,
        (int(train_file["records"]), int(validation_file["records"])),
    )
    if not fixture_mode:
        registered_failure = _mapping(
            _coverage_addendum.get("original_failed_coverage"),
            "registered V1 coverage failure",
        )
        registered_minimums = _mapping(
            registered_failure.get("royal_bucket_position_minimums"),
            "registered V1 bucket failure",
        )
        registered_unseen = _mapping(
            registered_failure.get("unseen_validation_royal_activation_fraction"),
            "registered V1 unseen failure",
        )
        parent = _mapping(coverage.get("parent_preflight"), "plan parent preflight")
        observed = _mapping(
            parent.get("observed_royal_bucket_position_minimums"),
            "plan V1 observed minimums",
        )
        unseen = _mapping(
            parent.get("unseen_validation_royal_activations"),
            "plan V1 unseen activations",
        )
        _require(
            observed.get("train") == registered_minimums.get("observed_train")
            and observed.get("validation")
            == registered_minimums.get("observed_validation")
            and unseen.get("count") == registered_unseen.get("observed_numerator")
            and unseen.get("total") == registered_unseen.get("observed_denominator")
            and parent.get("gates", {}).get("passed") is False,
            "plan does not preserve the registered V1 coverage failure",
        )
        measured = _mapping(
            _coverage_addendum.get("measured_support_before_registration"),
            "registered topology support",
        )
        for topology in TOPOLOGY_SPECS:
            registered = _mapping(measured.get(topology), f"registered {topology} support")
            actual = _mapping(
                _mapping(coverage.get("topologies"), "plan topologies").get(topology),
                f"plan {topology} support",
            )
            actual_unseen = _mapping(
                actual.get("unseen_validation"), f"plan {topology} unseen support"
            )
            _require(
                actual_unseen.get("activation_count")
                == registered.get("unseen_validation_activations")
                and actual_unseen.get("activation_total")
                == registered.get("total_validation_activations"),
                f"plan {topology} support differs from the preregistered measurement",
            )
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
    validation_candidate_path: Path | None = None,
    contract_path: Path | None = None,
    _expected_records: tuple[int, int] | None = None,
    _allow_dirty: bool = False,
    _source_override: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    contract, contract_sha = load_contract(contract_path)
    coverage_addendum, coverage_addendum_sha = load_coverage_addendum()
    data_repair_contract, data_repair_sha = load_data_repair_contract()
    repair_campaign = _mapping(
        data_repair_contract.get("campaign_contract"),
        "data-repair campaign dependency",
    )
    _require(
        repair_campaign.get("schema") == CONTRACT_SCHEMA
        and repair_campaign.get("sha256") == contract_sha,
        "data-repair addendum targets another campaign contract",
    )
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
        coverage_addendum=coverage_addendum,
        require_production_coverage=not fixture_mode,
        validation_candidate_path=validation_candidate_path,
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
            "coverage_addendum": {
                "path": COVERAGE_ADDENDUM_RELATIVE_PATH.as_posix(),
                "sha256": coverage_addendum_sha,
                "schema": COVERAGE_ADDENDUM_SCHEMA,
            },
            "effective_sha256": _effective_contract_sha256(
                contract_sha, coverage_addendum_sha
            ),
        },
        "source": source,
        "dependencies": {
            "data_repair": {
                "path": DATA_REPAIR_CONTRACT_RELATIVE_PATH.as_posix(),
                "schema": DATA_REPAIR_CONTRACT_SCHEMA,
                "sha256": data_repair_sha,
            },
            "coverage_addendum": {
                "path": COVERAGE_ADDENDUM_RELATIVE_PATH.as_posix(),
                "schema": COVERAGE_ADDENDUM_SCHEMA,
                "sha256": coverage_addendum_sha,
            },
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
    plan_contract = _mapping(plan.get("contract"), "plan contract")
    expected_campaign_binding = {
        "schema": "HORDE_V2_C1_TRAINER_BINDING_V1",
        "campaign_plan_sha256": _sha256_bytes(_canonical_json(plan)),
        "parent_contract_sha256": plan_contract.get("sha256"),
        "coverage_addendum_sha256": _mapping(
            plan_contract.get("coverage_addendum"), "plan coverage addendum"
        ).get("sha256"),
        "effective_contract_sha256": plan_contract.get("effective_sha256"),
        "campaign_identity_sha256": plan.get("campaign_identity_sha256"),
        "campaign_run_id": run.get("id"),
    }
    _require(
        receipt.get("campaign") == expected_campaign_binding,
        f"{run['id']} campaign-plan binding drifted",
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
    for key in ("validation_candidate", "validation_selection"):
        if key in expected_data:
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
    train_path: Path | None = None,
    validation_candidate_path: Path | None = None,
    validation_role_path: Path | None = None,
    split_receipt_path: Path | None = None,
    wdl_path: Path | None = None,
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
    selected_role_verification: dict[str, object] | None = None
    if not fixture_mode:
        _require(
            train_path is not None
            and validation_candidate_path is not None
            and validation_role_path is not None
            and split_receipt_path is not None
            and wdl_path is not None,
            (
                "production verification requires train, candidate, selected-role, "
                "book-split and WDL inputs"
            ),
        )
        coverage_addendum, _coverage_addendum_sha = load_coverage_addendum()
        recomputed_data = _validate_data(
            train_path,
            validation_role_path,
            split_receipt_path,
            wdl_path,
            (250_000, 250_000),
            coverage_addendum=coverage_addendum,
            require_production_coverage=True,
            validation_candidate_path=validation_candidate_path,
        )
        _require(
            recomputed_data == _mapping(plan.get("data"), "plan data"),
            "recomputed production data receipt differs from the campaign plan",
        )
        selected_role_verification = dict(
            _mapping(
                recomputed_data.get("validation_selection_verification"),
                "recomputed selected-role verification",
            )
        )
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
        "coverage_addendum_sha256": COVERAGE_ADDENDUM_SHA256,
        "effective_contract_sha256": _effective_contract_sha256(
            contract_sha, COVERAGE_ADDENDUM_SHA256
        ),
        "plan_sha256": _sha256_bytes(plan_payload),
        "campaign_identity_sha256": plan.get("campaign_identity_sha256"),
        "source": source,
        "selected_role_verification": selected_role_verification,
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
    plan.add_argument("validation", type=Path, help="selected validation-role receipt")
    plan.add_argument(
        "--validation-candidate",
        type=Path,
        help="direct HORDE_BIN_V1 parent of the selected validation role",
    )
    plan.add_argument("--book-split-receipt", type=Path, required=True)
    plan.add_argument("--wdl-calibration", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--contract", type=Path)

    verify = subparsers.add_parser("verify", help="verify all trained and quantized runs")
    verify.add_argument("plan", type=Path)
    verify.add_argument("runs_root", type=Path)
    verify.add_argument("--train-file", type=Path)
    verify.add_argument("--validation-candidate", type=Path)
    verify.add_argument("--validation-role", type=Path)
    verify.add_argument("--book-split-receipt", type=Path)
    verify.add_argument("--wdl-calibration", type=Path)
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
            validation_candidate_path=args.validation_candidate,
            contract_path=args.contract,
        )
    else:
        result = verify_campaign(
            args.plan,
            args.runs_root,
            train_path=args.train_file,
            validation_candidate_path=args.validation_candidate,
            validation_role_path=args.validation_role,
            split_receipt_path=args.book_split_receipt,
            wdl_path=args.wdl_calibration,
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
        SelectedRoleError,
        subprocess.SubprocessError,
        TrainingError,
        wire.FormatError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
