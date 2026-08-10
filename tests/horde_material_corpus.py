#!/usr/bin/env python3
"""Validate Horde winning-material semantics against the pinned scalachess corpus."""

from __future__ import annotations

import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path


CORPUS_SHA256 = "1f01b4fd3ab6066efe5a96a2ae4e0df5074fb99377d7caaa2acba04d453d53cc"
CORPUS_ROWS = 21_996


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    default_name = "stockfish.exe" if os.name == "nt" else "stockfish"
    executable = Path(
        sys.argv[1] if len(sys.argv) > 1 else root / "src" / default_name
    ).resolve()
    corpus = root / "tests" / "data" / "horde_insufficient_material.csv"

    if not executable.is_file():
        raise SystemExit(f"Engine not found: {executable}")
    if not corpus.is_file():
        raise SystemExit(f"Corpus not found: {corpus}")

    payload = corpus.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CORPUS_SHA256:
        raise AssertionError(f"Unexpected corpus SHA-256: {digest}")

    rows = list(csv.reader(payload.decode("utf-8").splitlines()))
    if len(rows) != CORPUS_ROWS:
        raise AssertionError(f"Expected {CORPUS_ROWS} rows, got {len(rows)}")

    commands: list[str] = []
    expected: list[bool] = []
    for line_number, row in enumerate(rows, 1):
        if len(row) < 2:
            raise AssertionError(f"Malformed corpus row {line_number}: {row!r}")
        fen, label = row[:2]
        if fen.split()[1] != "b":
            raise AssertionError(f"Corpus row {line_number} does not query White")
        commands.extend((f"position fen {fen}", "horde-material white"))
        expected.append(label.lower() == "true")
    commands.append("quit")

    completed = subprocess.run(
        [str(executable)],
        input="\n".join(commands) + "\n",
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise AssertionError(
            f"Horde-Stockfish exited with {completed.returncode}.\n{output[-4000:]}"
        )

    actual = [
        line.endswith(" insufficient")
        for line in output.splitlines()
        if line.startswith("horde-material white ")
    ]
    if len(actual) != CORPUS_ROWS:
        raise AssertionError(
            f"Expected {CORPUS_ROWS} engine answers, got {len(actual)}.\n{output[-4000:]}"
        )

    mismatches = [index + 1 for index, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]]
    if mismatches:
        details = []
        for index in mismatches[:20]:
            details.append(f"row {index}: {rows[index - 1]!r}")
        raise AssertionError(
            f"{len(mismatches)} material mismatches. " + "; ".join(details)
        )

    print(
        f"Horde material corpus completed successfully: {CORPUS_ROWS} rows, "
        f"sha256 {CORPUS_SHA256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
