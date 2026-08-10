#!/usr/bin/env python3
"""Targeted UCI regression tests for the fixed Horde rules chassis."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_engine(executable: Path, *commands: str) -> str:
    # UCI search is asynchronous. A following setoption command is specified to
    # wait for the active search, preventing the final quit from truncating it.
    synchronized: list[str] = []
    for command in commands:
        synchronized.append(command)
        if command == "go" or command.startswith("go "):
            synchronized.append("setoption name Move Overhead value 10")

    completed = subprocess.run(
        [str(executable)],
        input="\n".join((*synchronized, "quit", "")),
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


def run_rejected_position(executable: Path, fen: str, expected: str) -> None:
    completed = subprocess.run(
        [str(executable)],
        input=f"position fen {fen}\n",
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode == 0 or expected not in output:
        raise AssertionError(
            f"Position was not rejected with {expected!r}.\n{output}"
        )


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
        "setoption name UCI_Variant value horde",
        "isready",
    )
    require(output, "readyok", "fixed UCI variant selection")
    reject(output, "was already added", "fixed UCI variant selection")

    output = run_engine(
        executable,
        "setoption name UCI_Chess960 value true",
        "isready",
        "position startpos",
        "go perft 1",
    )
    require(
        output,
        "UCI_Chess960=true is unsupported for Horde.",
        "Chess960 rejection",
    )
    require(output, "readyok", "Chess960 rejection readiness")
    require(output, "Nodes searched: 8", "Chess960 rejection has no effect")

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

    canonical_perfts = (
        ("startpos", 4, 23310),
        (
            "fen 4k3/pp4q1/3P2p1/8/P3PP2/PPP2r2/PPP5/PPPP4 b - - 0 1",
            4,
            56539,
        ),
        (
            "fen k7/5p2/4p2P/3p2P1/2p2P2/1p2P2P/p2P2P1/2P2P2 w - - 0 1",
            4,
            33781,
        ),
        (
            "fen 4k3/7r/8/P7/2p1n2P/3p2P1/1P3P2/PPP1PPP1 w - - 0 1",
            4,
            128809,
        ),
        (
            "fen rnbqkbnr/6p1/2p1Pp1P/P1PPPP2/Pp4PP/1p2PPPP/1P2PPPP/PP1nPPPP b kq a3 0 18",
            4,
            197287,
        ),
    )
    for position, depth, nodes in canonical_perfts:
        output = run_engine(executable, f"position {position}", f"go perft {depth}")
        require(output, f"Nodes searched: {nodes}", f"canonical perft {nodes}")

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
        "position fen r3k2r/8/8/8/8/8/8/P7 b kq - 0 1",
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
        "position fen k7/1Q6/8/8/8/8/8/1R6 b - - 100 1",
        "go depth 1",
    )
    require(output, "score mate 0", "checkmate precedes fifty-move draw")
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

    output = run_engine(
        executable,
        "position fen k7/8/2NN4/8/8/8/8/8 b - - 0 1",
        "go depth 1",
    )
    require(output, "score cp 0", "Black stalemate")
    require(output, "bestmove (none)", "Black stalemate best move")

    # Pinned scalachess hordeClosedPosition fixtures. These exercise the
    # Black-to-move path where every legal Black move must be tested without
    # rebuilding the Position from FEN in the search hot path.
    closed_fortresses = (
        "8/p7/pk6/P7/P7/8/8/8 b - - 0 1",
        "QNBRRBNQ/PPpPPpPP/P1P2PkP/8/8/8/8/8 b - - 0 1",
        "b7/pk6/P7/P7/8/8/8/8 b - - 0 1",
        "8/p7/P7/P7/8/2q5/8/7k b - - 0 1",
        "krb5/pb1p4/P2Pp3/P3Pp2/5Pp1/6Pp/7P/8 b - - 0 1",
    )
    for fen in closed_fortresses:
        output = run_engine(executable, f"position fen {fen}", "d", "go depth 1")
        require(output, "Horde fortress: yes", f"closed fortress hook for {fen}")
        require(output, "score cp 0", f"closed fortress result for {fen}")
        require(output, "bestmove (none)", f"closed fortress best move for {fen}")

    open_fortresses = (
        "k7/p2p4/P2Pp1p1/P3PpPp/5PpP/6Pp/7P/7n b - - 0 1",
        "8/1b5r/1P6/1Pk3q1/1PP5/r1P5/P1P5/2P5 b - - 0 52",
        "8/8/8/7k/7P/7P/8/8 b - - 0 58",
        "8/p7/P7/P7/8/8/8/6qk b - - 0 1",
    )
    for fen in open_fortresses:
        output = run_engine(executable, f"position fen {fen}", "d", "go depth 1")
        require(output, "Horde fortress: no", f"open fortress hook for {fen}")
        reject(output, "bestmove (none)", f"open fortress search for {fen}")

    output = run_engine(
        executable,
        "position fen 4k3/8/8/8/8/8/8/P7 w - - 100 1",
        "go depth 1",
    )
    require(output, "score cp 0", "automatic fifty-move draw")
    require(output, "bestmove (none)", "automatic fifty-move draw best move")

    four_occurrences = "a1a2 h8h7 a2a1 h7h8 " * 3
    output = run_engine(
        executable,
        "position fen 4k2r/8/8/8/8/8/8/R7 w - - 0 1 moves "
        + four_occurrences.strip(),
        "go depth 1",
    )
    reject(output, "bestmove (none)", "fourfold repetition is not automatic")

    repetition_cycle = "a1a2 h8h7 a2a1 h7h8 " * 4
    output = run_engine(
        executable,
        "position fen 4k2r/8/8/8/8/8/8/R7 w - - 0 1 moves "
        + repetition_cycle.strip(),
        "go depth 1",
    )
    require(output, "score cp 0", "automatic fivefold repetition")
    require(output, "bestmove (none)", "fivefold repetition best move")

    output = run_engine(
        executable,
        "position fen r3k3/8/8/8/8/8/8/P7 b - - 0 1",
        "go depth 2",
    )
    require(output, "score mate 1", "last Horde piece capture mate distance")
    require(output, "bestmove a8a1", "last Horde piece capture is never pruned")

    # These searches exercise every incremental legacy-NNUE dirty-piece shape.
    incremental_positions = (
        "4k3/8/8/8/Pp6/8/8/8 b - a3 0 1",  # en passant
        "4k3/P7/8/8/8/8/8/8 w - - 0 1",  # promotion
        "r3k2r/8/8/8/8/8/8/P7 b kq - 0 1",  # castling
    )
    for fen in incremental_positions:
        output = run_engine(executable, f"position fen {fen}", "go depth 3")
        require(output, "bestmove ", f"incremental NNUE search for {fen}")

    output = run_engine(
        executable,
        "position fen 4k3/8/8/8/8/8/8/Q7 w - - 0 1",
        "horde-material white",
        "horde-material black",
    )
    require(output, "horde-material white insufficient", "White material API")
    require(output, "horde-material black sufficient", "Black material API")

    run_rejected_position(
        executable,
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
        "Horde requires no White king and exactly one Black king",
    )
    run_rejected_position(
        executable,
        "8/8/8/8/8/8/8/P7 w - - 0 1",
        "Horde requires no White king and exactly one Black king",
    )
    run_rejected_position(
        executable,
        "4k2k/8/8/8/8/8/8/P7 w - - 0 1",
        "Horde requires no White king and exactly one Black king",
    )
    run_rejected_position(
        executable,
        "4k3/8/8/8/8/8/8/R7 w K - 0 1",
        "White cannot have castling rights in Horde",
    )

    print("Horde rules testing completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
