#!/usr/bin/env python3
"""Targeted UCI regression tests for the fixed Horde rules chassis."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_engine(executable: Path, *commands: str) -> str:
    completed = subprocess.run(
        [str(executable)],
        input="\n".join((*commands, "quit", "")),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise AssertionError(
            f"Horde-Stockfish exited with {completed.returncode}.\n{output}"
        )
    return output


def require(output: str, expected: str, label: str) -> None:
    if expected not in output:
        raise AssertionError(f"{label}: missing {expected!r}.\n{output}")


def reject(output: str, unexpected: str, label: str) -> None:
    if unexpected in output:
        raise AssertionError(f"{label}: found {unexpected!r}.\n{output}")


def main() -> int:
    default_name = "stockfish.exe" if os.name == "nt" else "stockfish"
    executable = Path(sys.argv[1] if len(sys.argv) > 1 else default_name).resolve()
    if not executable.is_file():
        raise SystemExit(f"Engine not found: {executable}")

    output = run_engine(executable, "uci")
    require(output, "id name Horde-Stockfish", "UCI engine name")
    require(
        output,
        "option name UCI_Variant type combo default horde var horde",
        "fixed UCI variant",
    )
    require(
        output,
        "option name SyzygyProbeLimit type spin default 0 min 0 max 0",
        "disabled orthodox tablebases",
    )

    output = run_engine(
        executable,
        "setoption name SyzygyPath value ignored",
        "isready",
    )
    require(output, "Syzygy tablebases are disabled for Horde.", "Syzygy path guard")
    require(output, "readyok", "Syzygy path guard readiness")

    output = run_engine(executable, "position startpos", "d", "go perft 1")
    require(
        output,
        "Fen: rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1",
        "Horde start position",
    )
    require(output, "Nodes searched: 8", "Horde start position perft")

    output = run_engine(
        executable,
        "position fen 1N1N1N1k/N1N1N1N1/1N1N1N1N/N1N1N1N1/1N1N1N1N/N1N1N1N1/1NNN1N1N/N1N1N1N1 w - - 0 1",
        "go perft 1",
    )
    require(output, "Nodes searched: 160", "33-piece Horde move list")

    output = run_engine(
        executable,
        "position fen 4k3/8/8/8/8/8/8/P7 w - - 0 1",
        "go perft 1",
    )
    require(output, "a1a2: 1", "rank-one single step")
    require(output, "a1a3: 1", "rank-one double step")
    require(output, "Nodes searched: 2", "rank-one pawn move count")

    output = run_engine(
        executable,
        "position fen 4k3/8/8/8/8/1p6/8/P7 w - - 0 1 moves a1a3",
        "d",
        "go perft 1",
    )
    require(output, "Fen: 4k3/8/8/8/8/Pp6/8/8 b - - 0 1", "rank-one EP suppression")
    reject(output, "b3a2:", "rank-one EP suppression")

    output = run_engine(
        executable,
        "position fen 4k3/8/8/8/1p6/8/P7/8 w - - 0 1 moves a2a4",
        "d",
        "go perft 1",
    )
    require(output, "Fen: 4k3/8/8/8/Pp6/8/8/8 b - a3 0 1", "ordinary White EP")
    require(output, "b4a3: 1", "ordinary White EP capture")

    output = run_engine(
        executable,
        "position fen 4k3/1p6/8/P7/8/8/8/8 b - - 0 1 moves b7b5",
        "d",
        "go perft 1",
    )
    require(output, "Fen: 4k3/8/8/Pp6/8/8/8/8 w - b6 0 2", "ordinary Black EP")
    require(output, "a5b6: 1", "kingless White EP capture")

    output = run_engine(
        executable,
        "position fen r3k2r/8/8/8/8/8/8/P7 b KQkq - 0 1",
        "d",
        "go perft 1",
    )
    require(output, "Fen: r3k2r/8/8/8/8/8/8/P7 b kq - 0 1", "Black-only castling rights")
    require(output, "e8g8: 1", "Black king-side castling")
    require(output, "e8c8: 1", "Black queen-side castling")

    output = run_engine(
        executable,
        "position fen 4k3/8/8/8/8/8/8/8 b - - 0 1",
        "d",
        "go perft 1",
    )
    require(output, "Horde extinction: yes", "Horde extinction hook")
    require(output, "Nodes searched: 0", "Horde extinction")

    output = run_engine(
        executable,
        "position fen 4k3/8/8/8/8/8/8/Q7 w - - 0 1",
        "d",
    )
    require(output, "White mating material: insufficient", "lone Horde queen material")

    output = run_engine(
        executable,
        "position fen 4k3/7p/8/8/8/8/8/Q7 w - - 0 1",
        "d",
    )
    require(output, "White mating material: sufficient", "Horde queen mating support")

    output = run_engine(
        executable,
        "position fen k7/1Q6/8/8/8/8/8/1R6 b - - 0 1",
        "go depth 1",
    )
    require(output, "score mate 0", "Black checkmate")
    require(output, "bestmove (none)", "Black checkmate best move")

    output = run_engine(
        executable,
        "position fen k1r5/P1P5/8/8/8/8/8/8 w - - 0 1",
        "d",
        "go depth 1",
    )
    require(output, "Horde fortress: yes", "White stalemate fortress hook")
    require(output, "score cp 0", "White stalemate fortress")
    require(output, "bestmove (none)", "White stalemate fortress best move")

    print("Horde rules testing completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
