#!/usr/bin/env python3
"""Invariants for HORDE_DATA_CAMPAIGN_V1, the declared data campaign.

Data provenance and recipe provenance are independent axes. A contract that
trains a new architecture on an existing corpus declares the campaign that
produced that corpus, and the receipt comparison is made against the declared
campaign. A contract that declares nothing is its own data campaign, which is
the original behaviour byte for byte.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import horde_training_chunk_set as cs  # noqa: E402
import horde_training_scale_selected_role as cs_role  # noqa: E402

FAILURES: list[str] = []
ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "schemas" / "horde-v3-scale-v1.json"
RANK8 = ROOT / "schemas" / "horde-v2-rank8-scale-v1.json"
CORPUS = Path(r"D:/horde-train/train/chunk-set.json")
CORPUS_A_FRESH = ROOT / "schemas" / "horde-corpus-a-legacy-scale-v1.json"
# Every lineage contract of the corpus A phase. Each one claims to be the fresh
# legacy contract with the initialization as its only recipe delta, and the
# claim is checked here rather than trusted.
LINEAGE = ("horde-corpus-a-lineage-l1-legacy-scale-v1.json",)


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


def test_the_two_halves_of_the_registry_agree() -> None:
    """Both registries must name the same set of scale contracts.

    The schema names live in the chunk-set reader and the file, SHA-256 and
    architecture live in the selected-role tool, because the second imports the
    first and the dependency cannot run the other way. A contract added to one
    half and not the other passes every contract-level check and then dies at
    the first chunk the trainer opens, minutes into a run, with an error that
    names neither the contract nor the omission.
    """

    names = set(cs.SCALE_SCHEMAS)
    registered = set(cs_role.SCALE_CONTRACTS)
    check(
        names == registered,
        "the scale contract registries disagree: "
        f"only in the chunk-set reader {sorted(names - registered)}, "
        f"only in the selected-role tool {sorted(registered - names)}",
    )
    print(f"  {len(names)} scale contracts registered in both halves")


def test_lineage_contracts_differ_only_by_the_init() -> None:
    """A lineage contract must be the fresh contract plus an initialization.

    The whole discriminator rests on this: if anything other than the init
    moved, a difference in the boards stops being attributable to lineage. The
    claim is therefore checked field by field against the fresh contract rather
    than read off the NOTE.
    """

    if not CORPUS_A_FRESH.is_file():
        print("  corpus A fresh contract absent, skipping the lineage checks")
        return
    fresh_raw = CORPUS_A_FRESH.read_bytes()
    fresh = json.loads(fresh_raw)
    fresh_sha = hashlib.sha256(fresh_raw).hexdigest().upper()
    checked = 0
    for name in LINEAGE:
        path = ROOT / "schemas" / name
        if not path.is_file():
            FAILURES.append(f"{name} is registered in the test but missing on disk")
            continue
        raw = path.read_bytes()
        contract = json.loads(raw)
        digest = hashlib.sha256(raw).hexdigest().upper()

        # Registered, pinned, and pinned to this exact file.
        schema_name = contract["schema_name"]
        registered = cs_role.SCALE_CONTRACTS.get(schema_name)
        if registered is None:
            FAILURES.append(f"{name} declares {schema_name}, which is not registered")
            continue
        check(
            registered["sha256"] == digest,
            f"{name} does not match the SHA-256 pinned for {schema_name}",
        )
        check(
            registered["relative_path"].name == name,
            f"{schema_name} is pinned to a different file than {name}",
        )
        check(
            registered["architecture"] == fresh["training"]["architecture"]["name"],
            f"{name} is bound to an architecture the fresh contract does not train",
        )

        # The recipe: identical field for field, with the init added.
        recipe = dict(contract["training"])
        init = recipe.pop("init", None)
        check(init is not None, f"{name} is a lineage contract with no init")
        check(
            recipe == fresh["training"],
            f"{name} changed the recipe beyond adding the init",
        )
        if isinstance(init, dict):
            check(
                isinstance(init.get("checkpoint_sha256"), str)
                and len(init["checkpoint_sha256"]) == 64
                and init["checkpoint_sha256"] == init["checkpoint_sha256"].upper(),
                f"{name} declares an init without a well-formed checkpoint SHA-256",
            )
            check(
                isinstance(init.get("source"), str) and init["source"],
                f"{name} declares an init without naming its source",
            )

        # Everything the fresh contract says about the data, the gates and the
        # selection is inherited unchanged. Only the identity fields, the
        # declaration of the borrowed corpus and the self-description move.
        allowed_to_differ = {
            "$id",
            "NOTE",
            "data_campaign",
            "purpose",
            "recipe_identity",
            "schema_name",
            "title",
            "training",
        }
        for key in set(fresh) | set(contract):
            if key in allowed_to_differ:
                continue
            check(
                contract.get(key) == fresh.get(key),
                f"{name} changed {key}, which the fresh contract owns",
            )

        # The corpus is declared, not owned, and the self-description points at
        # the contract this one was actually derived from.
        check(
            contract["data_campaign"]["contract_sha256"] == fresh_sha
            and contract["data_campaign"]["contract_schema"] == fresh["schema_name"]
            and contract["data_campaign"]["contract_name"] == CORPUS_A_FRESH.name,
            f"{name} declares a data campaign other than the fresh corpus A contract",
        )
        identity = contract.get("recipe_identity", {})
        check(
            identity.get("identical_to", {}).get("contract_sha256") == fresh_sha
            and identity.get("added_fields") == ["training.init"]
            and identity.get("changed_fields") == [],
            f"{name} misstates its own delta against the fresh contract",
        )

        # Against the real chunks, not only against the fresh contract: the
        # trainer opens these receipts, and a contract that cannot reproduce
        # them fails minutes into a run rather than here.
        for role, receipt_path in (
            ("training", Path(r"D:/horde-train/corpus-a-bulk/bin/chunk-set.json")),
            ("validation_candidate", Path(r"D:/horde-train/corpus-a-val/bin/chunk-set.json")),
        ):
            if not receipt_path.is_file():
                continue
            real = json.loads(receipt_path.read_text(encoding="utf-8"))["campaign"]
            check(
                cs._campaign_section(expectation(path, role)) == real,
                f"{name} does not reproduce the real corpus A {role} chunk receipt",
            )
        checked += 1
    print(f"  {checked} lineage contract(s) differ from the fresh recipe only by the init")


def main() -> int:
    print("HORDE_DATA_CAMPAIGN_V1 invariants")
    test_axes_are_separate()
    test_undeclared_is_its_own_campaign()
    test_matches_the_real_receipt()
    test_mismatch_is_rejected()
    test_the_two_halves_of_the_registry_agree()
    test_lineage_contracts_differ_only_by_the_init()
    if FAILURES:
        print(f"\nFAILED with {len(FAILURES)} problems:")
        for failure in FAILURES[:20]:
            print(f"  {failure}")
        return 1
    print("\nall HORDE_DATA_CAMPAIGN_V1 invariants passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
