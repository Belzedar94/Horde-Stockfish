#!/usr/bin/env python3
"""Deterministic gate for the Horde NNUE V2 C0 split control."""

from __future__ import annotations

import difflib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import horde_training_c0 as c0  # noqa: E402
from horde_training_models import C0SingleG0Model, C0SplitG0Model  # noqa: E402


EXPECTED = {
    "payload_sha256": "0D3DA00CC9E441030DE2BFD9D1B7EE4FA7DB82926BD30EAA3316D3B1F3C58461",
    "trace_sha256": "4E45E11904A624CFB5099B385E79BAED522423293ABB98FA72C20558056CCC69",
    "value_sha256": "6848C7C85CA20C95F9EEB89E7C9F31F5D41F6029683F9CCF96C95A133BF450BC",
}


def _platform_invariant(receipt: dict[str, object]) -> dict[str, object]:
    projected = json.loads(json.dumps(receipt))
    del projected["runtime"]
    for field in (
        "initial_parameter_sha256",
        "initial_output_sha256",
        "steps",
        "final_parameter_sha256",
        "final_optimizer_sha256",
    ):
        del projected["common"][field]
    return projected


def _difference(expected: object, actual: object) -> str:
    return "\n".join(
        difflib.unified_diff(
            json.dumps(expected, indent=2, sort_keys=True).splitlines(),
            json.dumps(actual, indent=2, sort_keys=True).splitlines(),
            fromfile="frozen receipt",
            tofile="runtime receipt",
            lineterm="",
        )
    )


def main() -> int:
    first = c0.build_receipt()
    second = c0.build_receipt()
    if first != second:
        raise AssertionError("C0 receipt is not deterministic within one process")
    frozen = json.loads(
        (REPO_ROOT / "docs" / "horde" / "nnue-v2-c0-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    if _platform_invariant(first) != _platform_invariant(frozen):
        raise AssertionError(
            "C0 platform-invariant receipt changed:\n"
            + _difference(_platform_invariant(frozen), _platform_invariant(first))
        )
    if first["runtime"] == frozen["runtime"] and first != frozen:
        raise AssertionError(
            "C0 frozen receipt changed on its canonical runtime:\n"
            + _difference(frozen, first)
        )
    if (
        first["schema"] != c0.SCHEMA
        or first["cross_split_exact"] is not True
        or first["strength_evidence"] is not False
        or first["integer_contract"]["production_schema"] is not False
        or first["integer_contract"]["parameter_bytes"] != c0.PARAMETER_BYTES
    ):
        raise AssertionError(f"C0 top-level contract changed: {first}")

    variants = first["variants"]
    if [variant["name"] for variant in variants] != [
        "G0_SPLIT_64_192",
        "G0_SPLIT_128_128",
    ]:
        raise AssertionError("C0 variants changed")
    for variant in variants:
        if not all(variant["equalities"].values()):
            raise AssertionError(f"C0 equality failed: {variant['equalities']}")
    common = first["common"]
    for field in ("payload_sha256", "trace_sha256", "value_sha256"):
        if common["integer_export"][field] != EXPECTED[field]:
            raise AssertionError(f"C0 integer {field} changed")

    source = C0SingleG0Model(c0.MODEL_SEED)
    split = C0SplitG0Model(source, 64)
    if (
        split.first_weights.data_ptr() == source.global_weights.data_ptr()
        or split.second_weights.data_ptr() == source.global_weights.data_ptr()
    ):
        raise AssertionError("C0 split tensors alias the single-table control")
    try:
        C0SplitG0Model(source, 32)
    except ValueError as error:
        if "unsupported C0" not in str(error):
            raise
    else:
        raise AssertionError("C0 accepted an unregistered split width")

    print(
        "Horde V2 C0 equality passed: "
        f"payload={EXPECTED['payload_sha256']} trace={EXPECTED['trace_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
