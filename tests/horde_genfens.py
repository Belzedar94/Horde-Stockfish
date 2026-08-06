#!/usr/bin/env python3
"""Verify deterministic, physical-P Horde opening generation for OpenBench."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile


PREFIX = "info string genfens "
START = "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq -"
OPEN_FLANK = "4k3/pp4q1/3P2p1/8/P3PP2/PPP2r2/PPP5/PPPP4 b - -"


def run(engine: Path, command: str) -> tuple[list[str], str]:
    completed = subprocess.run(
        [str(engine)],
        input=f"{command}\nquit\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"generator exited with {completed.returncode}:\n{completed.stdout[-4000:]}"
        )
    fens = [
        line[len(PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(PREFIX) and "genfens error:" not in line
    ]
    return fens, completed.stdout


def validate_shape(fen: str) -> None:
    fields = fen.split()
    if len(fields) != 6:
        raise AssertionError(f"generator emitted a non-six-field FEN: {fen}")
    board = fields[0]
    if "H" in board or "K" in board:
        raise AssertionError(f"generator leaked a non-physical Horde piece: {fen}")
    if board.count("k") != 1:
        raise AssertionError(f"generator did not preserve one royal king: {fen}")
    if fields[1] not in {"w", "b"}:
        raise AssertionError(f"generator emitted an invalid side to move: {fen}")
    if any(symbol not in "pnbrqkPNBRQ12345678/" for symbol in board):
        raise AssertionError(f"generator emitted an invalid board symbol: {fen}")


def verify_engine_accepts(engine: Path, fens: list[str]) -> None:
    commands = []
    for fen in fens:
        commands.extend((f"position fen {fen}", "go perft 1"))
    commands.append("quit")
    completed = subprocess.run(
        [str(engine)],
        input="\n".join(commands) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"engine rejected a generated FEN:\n{completed.stdout[-4000:]}"
        )
    counts = [
        int(value)
        for value in re.findall(
            r"^Nodes searched:\s*([0-9]+)$",
            completed.stdout,
            flags=re.MULTILINE,
        )
    ]
    if len(counts) != len(fens) or any(count == 0 for count in counts):
        raise AssertionError(
            f"generated FEN validation mismatch: {len(counts)} receipts for "
            f"{len(fens)} positions"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    args = parser.parse_args()
    engine = args.engine.resolve()
    if not engine.is_file():
        raise SystemExit(f"Engine not found: {engine}")

    command = "genfens 64 seed 20260806 book None minplies=6 maxplies=10"
    first, first_output = run(engine, command)
    second, _ = run(engine, command)
    different, _ = run(
        engine,
        "genfens 64 seed 20260807 book None minplies=6 maxplies=10",
    )
    if len(first) != 64:
        raise AssertionError(f"expected 64 generated positions, got {len(first)}")
    if first != second:
        raise AssertionError("same-seed Horde generation is not deterministic")
    if first == different:
        raise AssertionError("different Horde generation seeds produced the same stream")
    if "genfens error:" in first_output:
        raise AssertionError(first_output)
    for fen in first:
        validate_shape(fen)
    verify_engine_accepts(engine, first)

    with tempfile.TemporaryDirectory() as directory:
        book = Path(directory) / "horde.epd"
        book.write_text(f"{START}\n{OPEN_FLANK}\n", encoding="ascii")
        from_book, output = run(
            engine,
            f"genfens 32 seed 94 book {book} minplies=0 maxplies=2",
        )
        if len(from_book) != 32 or "genfens error:" in output:
            raise AssertionError(output)
        for fen in from_book:
            validate_shape(fen)
        verify_engine_accepts(engine, from_book)

    rejected, output = run(
        engine,
        "genfens 1 seed 1 book None unknown-filter",
    )
    if rejected or "unknown genfens argument" not in output:
        raise AssertionError("unknown generator arguments did not fail closed")

    print("Horde genfens: deterministic, physical-P, legal, and fail-closed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
