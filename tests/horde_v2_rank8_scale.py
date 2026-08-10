#!/usr/bin/env python3
"""Focused contract checks for the selected Rank-8 scale campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "schemas" / "horde-v2-rank8-scale-v1.json"
SELECTION = ROOT / "docs" / "horde" / "nnue-v2-representation-selection.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))

    if contract["schema_name"] != "HORDE_V2_RANK8_SCALE_V1":
        raise AssertionError("Rank-8 scale schema drifted")
    if selection["schema"] != "HORDE_V2_REPRESENTATION_SELECTION_V1":
        raise AssertionError("Rank-8 selection schema drifted")
    if contract["dependencies"]["architecture_selection"]["sha256"] != _sha256(
        SELECTION
    ):
        raise AssertionError("Rank-8 selection receipt identity drifted")

    selected = selection["selected_architecture"]["name"]
    if selected != contract["training"]["architecture"]["name"]:
        raise AssertionError("scale architecture differs from the selected representation")
    if contract["dependencies"]["architecture_selection"]["selected_architecture"] != selected:
        raise AssertionError("selection dependency names a different architecture")

    generation = contract["generation"]
    train = generation["training"]
    validation = generation["validation_candidate"]
    if train["records"] != train["positions_per_chunk"] * train["chunk_count"]:
        raise AssertionError("training chunks do not sum to 50M")
    if validation["records"] != validation["positions_per_chunk"] * validation["chunk_count"]:
        raise AssertionError("validation chunks do not sum to 1M")
    if train["base_seed"] + train["chunk_count"] - 1 >= validation["base_seed"]:
        raise AssertionError("training and validation seed namespaces overlap")

    recipe = contract["training"]
    expected_steps = (train["records"] + recipe["batch_size"] - 1) // recipe["batch_size"]
    if recipe["optimizer_steps"] != expected_steps:
        raise AssertionError("optimizer-step count does not cover the exact training role")
    if recipe["checkpoint_steps"][-1] != expected_steps:
        raise AssertionError("the final checkpoint does not reach the complete run")
    if recipe["training_example_exposures"] != train["records"] * recipe["epochs"]:
        raise AssertionError("training exposure count drifted")

    if not selection["claims"]["architecture_selected"]:
        raise AssertionError("the representation selection is not explicit")
    for claims in (selection["claims"], contract["claims"]):
        if claims["production_network"] or claims["run6b_production_path_changed"]:
            raise AssertionError("an experimental contract changed the Run 6B production path")

    print("Horde V2 Rank-8 scale contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
