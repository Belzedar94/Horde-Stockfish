#!/usr/bin/env python3
"""Invariants for HORDE_DATA_CAMPAIGN_V1, the declared data campaign.

Data provenance and recipe provenance are independent axes. A contract that
trains a new architecture on an existing corpus declares the campaign that
produced that corpus, and the receipt comparison is made against the declared
campaign. A contract that declares nothing is its own data campaign, which is
the original behaviour byte for byte.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import horde_training_chunk_set as cs  # noqa: E402

FAILURES: list[str] = []
ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "schemas" / "horde-v3-scale-v1.json"
RANK8 = ROOT / "schemas" / "horde-v2-rank8-scale-v1.json"
CORPUS = Path(r"D:/horde-train/train/chunk-set.json")


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def expectation(path: Path, role: str = "training"):
    return cs.load_campaign_expectation(path, role)


def test_axes_are_separate() -> None:
    contract = json.loads(V3.read_text(encoding="utf-8"))
    e = expectation(V3)
    check(
        contract["training"]["architecture"]["name"] == "v3-g1024-pawn-wpc8",
        "the V3 contract must own a V3 recipe",
    )
    check(
        e.data_campaign["contract_schema"] == "HORDE_V2_RANK8_SCALE_V1",
        "the V3 contract must declare the Rank8 data campaign",
    )
    check(
        e.data_campaign["contract_schema"] != contract["schema_name"],
        "this fixture is only meaningful when the two axes actually differ",
    )
    print(
        f"  recipe={contract['training']['architecture']['name']}  "
        f"data={e.data_campaign['campaign_id']} ({e.data_campaign['contract_schema']})"
    )


def test_undeclared_is_its_own_campaign() -> None:
    if not RANK8.is_file():
        print("  Rank8 contract absent, skipping the self-campaign check")
        return
    contract = json.loads(RANK8.read_text(encoding="utf-8"))
    check("data_campaign" not in contract, "the Rank8 contract must not declare one")
    e = expectation(RANK8)
    check(
        e.data_campaign["contract_schema"] == contract["schema_name"]
        and e.data_campaign["contract_name"] == RANK8.name
        and e.data_campaign["campaign_id"] == contract["openbench"]["campaign_id"],
        "a contract with no declaration must be its own data campaign",
    )


def test_matches_the_real_receipt() -> None:
    if not CORPUS.is_file():
        print("  corpus absent, skipping the receipt comparison")
        return
    real = json.loads(CORPUS.read_text(encoding="utf-8"))["campaign"]
    check(
        cs._campaign_section(expectation(V3)) == real,
        "the declared data campaign must reproduce the real chunk receipt exactly",
    )


def _mutated(**overrides) -> Path:
    contract = json.loads(V3.read_text(encoding="utf-8"))
    contract["data_campaign"].update(overrides)
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(contract, handle, indent=2, sort_keys=True)
    handle.close()
    return Path(handle.name)


def test_mismatch_is_rejected() -> None:
    """A contract whose declared campaign does not match the real receipt fails."""

    if not CORPUS.is_file():
        print("  corpus absent, skipping the mismatch rejection")
        return
    real = json.loads(CORPUS.read_text(encoding="utf-8"))["campaign"]
    cases = {
        "a wrong campaign id": {"campaign_id": "horde-not-this-campaign-20260101"},
        "a wrong cohort": {"cohort": "some-other-cohort"},
        "a wrong contract sha": {"contract_sha256": "AB" * 32},
        "a wrong contract name": {"contract_name": "not-the-contract.json"},
    }
    for label, override in cases.items():
        path = _mutated(**override)
        try:
            built = cs._campaign_section(expectation(path))
            check(built != real, f"{label} still reproduced the real receipt")
        finally:
            path.unlink(missing_ok=True)
    # A structurally invalid declaration must be refused outright.
    for label, override in (
        ("an unknown schema", {"contract_schema": "HORDE_NOT_A_SCALE_SCHEMA"}),
        ("a malformed sha", {"contract_sha256": "not-a-sha"}),
        ("an empty campaign id", {"campaign_id": ""}),
    ):
        path = _mutated(**override)
        try:
            expectation(path)
            FAILURES.append(f"{label} was accepted")
        except cs.ChunkSetError:
            pass
        finally:
            path.unlink(missing_ok=True)
    # A declaration missing a field must be refused too.
    contract = json.loads(V3.read_text(encoding="utf-8"))
    contract["data_campaign"].pop("cohort")
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(contract, handle, indent=2, sort_keys=True)
    handle.close()
    try:
        expectation(Path(handle.name))
        FAILURES.append("an incomplete data_campaign was accepted")
    except cs.ChunkSetError:
        pass
    finally:
        Path(handle.name).unlink(missing_ok=True)
    print(f"  rejected {len(cases)} mismatches and 4 malformed declarations")


def main() -> int:
    print("HORDE_DATA_CAMPAIGN_V1 invariants")
    test_axes_are_separate()
    test_undeclared_is_its_own_campaign()
    test_matches_the_real_receipt()
    test_mismatch_is_rejected()
    if FAILURES:
        print(f"\nFAILED with {len(FAILURES)} problems:")
        for failure in FAILURES[:20]:
            print(f"  {failure}")
        return 1
    print("\nall HORDE_DATA_CAMPAIGN_V1 invariants passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
