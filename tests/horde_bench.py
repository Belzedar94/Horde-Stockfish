#!/usr/bin/env python3
"""Verify the deterministic Horde-only benchmark receipt."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path


EXPECTED_NODES = 341_503
EXPECTED_BESTMOVES_SHA256 = (
    "9ca26cd22b5b8b9f8939714988e6e8e51fbf53cfff60df895eafc0a8d3df2fcb"
)


def parse_args() -> argparse.Namespace:
    default_name = "stockfish.exe" if os.name == "nt" else "stockfish"
    parser = argparse.ArgumentParser(description="Run the frozen Horde benchmark")
    parser.add_argument("engine", type=Path, nargs="?", default=Path("src") / default_name)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = args.engine.resolve()
    if not engine.is_file():
        raise SystemExit(f"Engine not found: {engine}")
    if args.runs < 1:
        raise SystemExit("--runs must be positive")

    for run in range(1, args.runs + 1):
        completed = subprocess.run(
            [str(engine)],
            input=b"bench 16 1 13 default depth\nquit\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
        output = completed.stdout.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            raise RuntimeError(
                f"benchmark run {run} exited with {completed.returncode}:\n{output[-4000:]}"
            )

        match = re.search(r"Nodes searched\s*:\s*(\d+)", output)
        if not match:
            raise RuntimeError(f"benchmark run {run} emitted no node total")
        nodes = int(match.group(1))
        bestmoves = [line for line in output.splitlines() if line.startswith("bestmove ")]
        if len(bestmoves) != 10:
            raise RuntimeError(
                f"benchmark run {run} emitted {len(bestmoves)} bestmoves, expected 10"
            )
        digest = hashlib.sha256("|".join(bestmoves).encode("ascii")).hexdigest()
        if nodes != EXPECTED_NODES:
            raise RuntimeError(
                f"benchmark run {run} searched {nodes} nodes with bestmove digest {digest}; "
                f"expected {EXPECTED_NODES} nodes"
            )
        if digest != EXPECTED_BESTMOVES_SHA256:
            raise RuntimeError(
                f"benchmark run {run} bestmove digest {digest}, "
                f"expected {EXPECTED_BESTMOVES_SHA256}"
            )
        print(f"Horde benchmark run {run}: nodes={nodes}, bestmoves_sha256={digest}")

    print("Horde deterministic benchmark completed successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
