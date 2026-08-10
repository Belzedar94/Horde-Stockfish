#!/usr/bin/env python3
"""End-to-end fixture for Rank8 chunk-set WDL and training dispatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
# Keep the production helpers ahead of same-named test modules.  Inserting the
# tests directory last would shadow tools/horde_training_chunk_set.py on Linux
# and Windows CI when horde_fit_wdl imports it as a top-level module.
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

import horde_fit_wdl as fit_wdl  # noqa: E402
import horde_training_chunk_set as chunk_set  # noqa: E402
import horde_training_control as control  # noqa: E402
import horde_training_resume as fixtures  # noqa: E402
import horde_training_scale_selected_role as selector  # noqa: E402


TRAIN_BOOK = "A" * 64
VALIDATION_BOOK = "B" * 64
SOURCE_COMMIT = "1" * 40
TRAIN_RECORDS = 192
CANDIDATE_RECORDS = 128
SELECTED_RECORDS = 96
TRAIN_BASE_SEED = 1000
CANDIDATE_BASE_SEED = 2000
TRAINING_SEED = 7435908571601354096


def _write_contract(path: Path) -> None:
    contract = {
        "schema_name": selector.CONTRACT_SCHEMA,
        "dependencies": {
            "dataset": {
                "schema": "HORDE_BIN_V1",
                "schema_sha256": "B46ADE18AB8954A6AB232593484273E50C12B51550A938763A7A7D94DCCB63E4",
            },
            "teacher": {
                "source_commit": SOURCE_COMMIT,
                "producer_sha256": "2" * 64,
                "network_schema": "HORDETEST_HP_LEGACY_V1",
                "network_sha256": "B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7",
            },
            "labels": {
                "schema": "HORDE_LABEL_CONTRACT_V1",
                "schema_sha256": "C299BA9ECD96DEF24363F8F62A8C67B88241AA860FB0735D4558B8EFEA0DCC22",
            },
            "selected_validation_schema": selector.SCHEMA,
        },
        "openbench": {
            "campaign_id": "fixture-rank8-scale",
            "cohort": "fixture-v3-split",
        },
        "books": {
            "training": {"records": 3, "raw_sha256": TRAIN_BOOK},
            "validation": {"records": 2, "raw_sha256": VALIDATION_BOOK},
        },
        "generation": {
            "common": {
                "hash_mb": 16,
                "depth": 1,
                "nodes": 0,
                "random_move_min_ply": 1,
                "random_move_max_ply": 1,
                "random_move_count": 0,
                "random_multi_pv": 0,
                "random_multi_pv_diff": 0,
                "write_min_ply": 0,
                "write_max_ply": 1,
                "max_game_ply": 2,
            },
            "training": {
                "records": TRAIN_RECORDS,
                "positions_per_chunk": 64,
                "chunk_count": 3,
                "base_seed": TRAIN_BASE_SEED,
            },
            "validation_candidate": {
                "records": CANDIDATE_RECORDS,
                "positions_per_chunk": 64,
                "chunk_count": 2,
                "base_seed": CANDIDATE_BASE_SEED,
            },
        },
        "validation_selection": {
            "target_records": SELECTED_RECORDS,
            "algorithm": selector.ALGORITHM,
            "candidate_order": "chunk index ascending, then local record index ascending",
            "reject_training_physical_key": True,
            "reject_training_legacy_model_input_key": True,
            "reject_selected_physical_duplicate": True,
            "reject_selected_legacy_model_input_duplicate": True,
            "label_blind": True,
            "insufficient_candidate_records_fail_closed": True,
        },
        "training": {
            "architecture": {
                "name": "v2-c1-rank8-64x192",
                "schema": "V2_C1_ROYAL_RANK8_64X192",
                "training_structural_sha256": "3" * 64,
            },
            "seed": TRAINING_SEED,
            "epochs": 1,
            "training_example_exposures": TRAIN_RECORDS,
            "batch_size": 64,
            "optimizer_steps": 3,
            "block_size": 128,
            "lambda": 0.6,
            "learning_rate": 0.0015,
            "dense_learning_rate_multiplier": 0.1,
            "output_learning_rate_multiplier": 0.1,
            "scheduler_gamma": 0.987,
            "optimizer": "torch.optim.RAdam",
            "device": {"type": "cpu", "expected_name": "fixture", "cpu_threads": 1},
            "checkpoint_steps": [1, 2, 3],
            "resume_must_be_bit_exact": True,
        },
    }
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8", newline="\n")


def _assemble_role(
    root: Path,
    contract: Path,
    role: str,
    *,
    first: int,
    chunks: int,
    base_seed: int,
    book_sha256: str,
    opening_count: int,
) -> Path:
    paths: list[Path] = []
    for index in range(chunks):
        path = root / f"{role}-{index}.bin"
        fixtures._write_dataset(
            path,
            first=first + 64 * index,
            count=64,
            book_sha256=book_sha256,
            seed=base_seed + index,
            opening_count=opening_count,
        )
        paths.append(path)
    receipt = root / f"{role}-chunk-set.json"
    chunk_set.assemble_chunk_set(contract, role, receipt, list(reversed(paths)))
    return receipt


def _arguments(
    train: Path,
    selected: Path,
    candidate: Path,
    contract: Path,
    split: Path,
    calibration: Path,
    output: Path,
    *,
    resume: Path | None = None,
    stop_after_steps: int | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        train=train,
        validation=selected,
        validation_candidate=candidate,
        validation_selected_role=False,
        selected_role_fixture=False,
        scale_contract=contract,
        scale_contract_fixture=True,
        campaign_plan=None,
        campaign_run_id=None,
        book_split_receipt=split,
        wdl_calibration=calibration,
        architecture="v2-c1-rank8-64x192",
        output=output,
        seed=TRAINING_SEED,
        epochs=1,
        lambda_value=0.6,
        learning_rate=0.0015,
        dense_learning_rate_multiplier=0.1,
        output_learning_rate_multiplier=0.1,
        scheduler_gamma=0.987,
        batch_size=64,
        block_size=128,
        device="cpu",
        cpu_threads=1,
        resume=resume,
        stop_after_steps=stop_after_steps,
        allow_legacy_book_split_v1=False,
        allow_dirty=True,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="horde-scale-adapter-") as directory:
        root = Path(directory)
        contract = root / "scale.json"
        _write_contract(contract)
        train = _assemble_role(
            root,
            contract,
            "training",
            first=0,
            chunks=3,
            base_seed=TRAIN_BASE_SEED,
            book_sha256=TRAIN_BOOK,
            opening_count=3,
        )
        candidate = _assemble_role(
            root,
            contract,
            "validation_candidate",
            first=TRAIN_RECORDS,
            chunks=2,
            base_seed=CANDIDATE_BASE_SEED,
            book_sha256=VALIDATION_BOOK,
            opening_count=2,
        )
        selected_root = root / "selected"
        selector.create_scale_selected_role(
            train,
            candidate,
            selected_root,
            root / "selection-scratch",
            contract_path=contract,
            _allow_fixture=True,
            _source_override={"commit": SOURCE_COMMIT, "dirty": False},
        )
        selected = selected_root / selector.RECEIPT_FILENAME

        split = root / "book-split.json"
        fixtures._write_split_receipt(
            split,
            train_count=3,
            validation_count=2,
            train_book_sha256=TRAIN_BOOK,
            validation_book_sha256=VALIDATION_BOOK,
        )
        calibration = root / "wdl.json"
        artifact = fit_wdl.fit(
            argparse.Namespace(
                train=train,
                output=calibration,
                minimum_class_support=32,
                chunk_set=True,
                contract=contract,
            )
        )
        if artifact["source"]["training_file"]["name"] != train.name:
            raise AssertionError("WDL calibration lost the chunk-set receipt identity")

        full = root / "full"
        full_receipt = control.train(
            _arguments(train, selected, candidate, contract, split, calibration, full)
        )
        if not full_receipt["run"]["complete"]:
            raise AssertionError("scale fixture did not complete")
        if full_receipt["campaign"]["schema"] != control.SCALE_BINDING_SCHEMA:
            raise AssertionError("scale campaign binding was not emitted")
        if full_receipt["claims"] != {
            "purpose": "rank8-50m-scale-training",
            "integration_only": False,
            "strength_eligible": True,
            "strength_evidence": False,
            "production_network": False,
        }:
            raise AssertionError("scale training claims drifted")
        if full_receipt["data"]["train_file"]["payload_sha256"] != artifact["source"][
            "training_file"
        ]["payload_sha256"]:
            raise AssertionError("trainer and WDL used different logical payloads")

        partial = root / "partial"
        partial_receipt = control.train(
            _arguments(
                train,
                selected,
                candidate,
                contract,
                split,
                calibration,
                partial,
                stop_after_steps=1,
            )
        )
        if partial_receipt["run"]["complete"]:
            raise AssertionError("scale checkpoint fixture did not stop")
        resumed = root / "resumed"
        resumed_receipt = control.train(
            _arguments(
                train,
                selected,
                candidate,
                contract,
                split,
                calibration,
                resumed,
                resume=partial / "checkpoint.pt",
            )
        )
        if resumed_receipt["run"]["final_state_sha256"] != full_receipt["run"][
            "final_state_sha256"
        ]:
            raise AssertionError("scale stop/resume changed the final model state")
        if resumed_receipt["run"]["sample_order_chain_sha256"] != full_receipt["run"][
            "sample_order_chain_sha256"
        ]:
            raise AssertionError("scale stop/resume changed sample order")

    print("scale training adapter tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
