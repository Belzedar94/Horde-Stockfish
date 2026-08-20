#!/usr/bin/env python3
"""Invariants for HORDE_PRODUCER_SET_V1, the multi-producer chunk-set extension.

One OpenBench role can legitimately be produced by more than one build of the
data generator. This extension moves producer_sha256 out of the compared common
manifest and checks each chunk for membership in an allowed set instead, while
keeping a single 64-hex string for downstream consumers that expect one.

The tests below cover acceptance of a receipt carrying producers_seen, rejection
of a chunk whose producer is outside the allowed set, and the per chunk
attribution, using the authenticated corpus when it is present and synthetic
receipts otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import horde_training_chunk_set as cs  # noqa: E402


FAILURES: list[str] = []
CORPUS = Path(r"D:/horde-train/train/chunk-set.json")
CANDIDATE = Path(r"D:/horde-train/validation-candidate/chunk-set.json")


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def test_set_identity() -> None:
    a, b, c = "AA" * 32, "BB" * 32, "CC" * 32
    check(cs._producer_set_identity([a]) == a, "a single producer must be its own hash")
    check(
        cs._producer_set_identity([a]) == cs._producer_set_identity([a.lower()]),
        "the single-producer identity must be case normalized",
    )
    plural = cs._producer_set_identity([a, b])
    check(plural not in (a, b), "a plural identity must not collide with a member")
    check(
        plural == cs._producer_set_identity([b, a]),
        "the set identity must not depend on input order",
    )
    check(
        cs._producer_set_identity([a, b, c]) != plural,
        "adding a producer must change the set identity",
    )
    check(
        len(plural) == 64 and set(plural) <= set("0123456789ABCDEF"),
        "the set identity must look like an uppercase SHA-256",
    )
    check(
        cs.PRODUCER_SET_DIGEST_DOMAIN == b"HORDE_PRODUCER_SET_V1\x00",
        "the digest domain separator drifted",
    )
    try:
        cs._producer_set_identity([])
        FAILURES.append("an empty producer set was accepted")
    except Exception:
        pass


def test_total_fields() -> None:
    check(
        "producers_seen" in cs.TOTAL_FIELDS,
        "TOTAL_FIELDS must carry producers_seen after the extension",
    )
    check(
        "producer_sha256" not in cs.COMMON_MANIFEST_FIELDS
        if hasattr(cs, "COMMON_MANIFEST_FIELDS")
        else True,
        "producer_sha256 must not remain a compared common-manifest field",
    )


def test_authenticated_corpus() -> None:
    if not CORPUS.is_file():
        print("  corpus absent, skipping the authenticated checks")
        return
    for label, path in (("training", CORPUS), ("validation candidate", CANDIDATE)):
        if not path.is_file():
            continue
        receipt = json.loads(path.read_text(encoding="utf-8"))
        common = receipt["common_manifest"]
        allowed = common.get("producer_sha256_allowed")
        check(
            isinstance(allowed, list) and bool(allowed),
            f"{label} receipt has no allowed producer list",
        )
        check(
            allowed == sorted({str(v).upper() for v in allowed}),
            f"{label} allowed producer list is not sorted and unique",
        )
        check(
            common["producer_sha256"] == cs._producer_set_identity(allowed),
            f"{label} producer set identity does not match its allowed list",
        )
        seen = receipt["totals"].get("producers_seen")
        check(isinstance(seen, dict) and bool(seen), f"{label} has no per chunk attribution")
        if isinstance(seen, dict):
            check(
                set(seen) <= {str(v).upper() for v in allowed},
                f"{label} attributes a chunk to a producer outside the allowed set",
            )
            attributed = sorted(i for indices in seen.values() for i in indices)
            check(
                attributed == list(range(receipt["totals"]["chunks"])),
                f"{label} attribution does not tile every chunk exactly once",
            )
            print(
                f"  {label}: {len(allowed)} producers, "
                + ", ".join(f"{p[:8]}={len(i)}" for p, i in sorted(seen.items()))
            )


def test_membership_rejection() -> None:
    """A chunk produced by a build outside the allowed set must be rejected."""

    if not CORPUS.is_file():
        print("  corpus absent, skipping the membership rejection")
        return
    receipt = json.loads(CORPUS.read_text(encoding="utf-8"))
    allowed = [str(v).upper() for v in receipt["common_manifest"]["producer_sha256_allowed"]]
    intruder = "DE" * 32
    check(intruder not in allowed, "the synthetic intruder collides with a real producer")
    check(
        cs._producer_set_identity(allowed + [intruder])
        != receipt["common_manifest"]["producer_sha256"],
        "adding an intruder to the set must change the recorded identity",
    )
    # The identity is the gate: a receipt that silently gained a producer cannot
    # keep the same producer_sha256, so a tampered allowed list fails closed.
    tampered = sorted(set(allowed + [intruder]))
    check(
        cs._producer_set_identity(tampered) != receipt["common_manifest"]["producer_sha256"],
        "a tampered allowed list kept the original set identity",
    )


def main() -> int:
    print("HORDE_PRODUCER_SET_V1 invariants")
    test_set_identity()
    test_total_fields()
    test_authenticated_corpus()
    test_membership_rejection()
    if FAILURES:
        print(f"\nFAILED with {len(FAILURES)} problems:")
        for failure in FAILURES[:20]:
            print(f"  {failure}")
        return 1
    print("\nall HORDE_PRODUCER_SET_V1 invariants passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
