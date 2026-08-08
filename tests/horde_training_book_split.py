#!/usr/bin/env python3
"""Contract tests for the deterministic Horde training-book split."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import horde_split_training_book as splitter  # noqa: E402


SOURCE = """\
8/8/8/8/8/8/P7/4k3 w - - 0 1
8/8/8/8/8/8/1P6/3k4 b - - 0 1
8/8/8/8/8/2P5/8/2k5 w - - 4 7
8/8/8/8/8/3P4/8/1k6 b - - 8 11
8/8/8/8/4P3/8/8/k7 w - - 12 18
8/8/8/5P2/8/8/8/7k b - - 16 25
8/8/6P1/8/8/8/8/6k1 w - - 20 31
8/7P/8/8/8/8/8/5k2 b - - 24 38
"""


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="horde-book-split-") as raw:
        root = Path(raw)
        source = root / "source.epd"
        source.write_text(SOURCE, encoding="ascii", newline="\n")
        train = root / "train.epd"
        validation = root / "validation.epd"
        receipt_path = root / "receipt.json"

        receipt = splitter.write_split(source, train, validation, receipt_path, 3, 1)
        frozen = json.loads(receipt_path.read_text(encoding="utf-8"))
        if frozen != receipt:
            raise AssertionError("serialized split receipt differs from the in-memory receipt")
        if not receipt["disjoint_position_keys"] or not receipt["complete_partition"]:
            raise AssertionError(f"split gates are not closed: {receipt}")
        if receipt["train"]["records"] + receipt["validation"]["records"] != 8:
            raise AssertionError(f"split lost records: {receipt}")

        actual = splitter.split_book(source, 3, 1)
        if actual[0] != train.read_bytes() or actual[1] != validation.read_bytes():
            raise AssertionError("split bytes are not deterministic")
        try:
            splitter.write_split(source, train, root / "other.epd", root / "other.json", 3, 1)
        except splitter.SplitError:
            pass
        else:
            raise AssertionError("splitter overwrote an existing output")

        duplicate = root / "duplicate.epd"
        duplicate.write_text(
            "8/8/8/8/8/8/P7/4k3 w - - 0 1\n"
            "8/8/8/8/8/8/P7/4k3 w - - 99 100\n",
            encoding="ascii",
            newline="\n",
        )
        try:
            splitter.split_book(duplicate, 3, 1)
        except splitter.SplitError as error:
            if "duplicates the physical position" not in str(error):
                raise
        else:
            raise AssertionError("splitter accepted duplicate physical positions")

    print("Horde training-book split tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
