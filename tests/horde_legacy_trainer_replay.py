#!/usr/bin/env python3
"""Replay pinned Run 6B through the trainer-side HORDE_BIN feature path."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct
import sys
from typing import Sequence

import chess
from chess.variant import HordeBoard


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import horde_nnue_parity as parity  # noqa: E402
import horde_run6b  # noqa: E402
import horde_training_decoder as decoder  # noqa: E402


def physical_board(fen: str) -> tuple[list[int], int]:
    position = HordeBoard(fen)
    board = [0] * 64
    for square, piece in position.piece_map().items():
        if piece.color == chess.WHITE:
            if piece.piece_type == chess.KING:
                raise AssertionError(f"Horde replay encountered a White king: {fen}")
            board[square] = piece.piece_type
        else:
            board[square] = 5 + piece.piece_type
    return board, decoder.WHITE if position.turn == chess.WHITE else decoder.BLACK


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_binary = "stockfish.exe" if sys.platform == "win32" else "stockfish"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=ROOT / "src" / default_binary)
    parser.add_argument(
        "--network",
        type=Path,
        default=ROOT / "networks" / "hordetest_run6b_e37_l06.nnue",
    )
    parser.add_argument("--positions", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x6B37_06)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    engine = args.engine.resolve()
    network_path = args.network.resolve()
    if not engine.is_file() or not network_path.is_file():
        raise AssertionError(f"missing engine or network: {engine}, {network_path}")
    if args.positions <= 0 or args.batch_size <= 0:
        raise AssertionError("positions and batch size must be positive")

    network = horde_run6b.Run6BNetwork.load(network_path)
    positions, coverage = parity.generate_positions(args.positions, args.seed)
    missing = [
        label
        for label in ("capture", "promotion", "en_passant", "castling")
        if coverage[label] == 0
    ]
    if missing:
        raise AssertionError("Run 6B replay corpus lacks " + ", ".join(missing))
    prefix = (
        "uci",
        f"setoption name EvalFile value {network_path}",
        "setoption name Threads value 1",
        "isready",
    )

    digest = hashlib.sha256()
    checked = 0
    for begin in range(0, len(positions), args.batch_size):
        fens = positions[begin : begin + args.batch_size]
        engine_raw = parity.raw_eval_batch(
            engine, prefix, fens, False, ROOT, timeout=120.0
        )
        for offset, (fen, expected) in enumerate(zip(fens, engine_raw, strict=True)):
            board, side_to_move = physical_board(fen)
            features = decoder.extract_sparse_features(board)
            actual = network.evaluate(features, side_to_move)
            observed = (actual.psqt, actual.positional, actual.total)
            if observed != expected:
                index = begin + offset
                raise AssertionError(
                    f"Run 6B trainer replay mismatch at position {index}: "
                    f"trainer={observed}, engine={expected}, fen={fen}"
                )
            digest.update(struct.pack("<iii", *observed))
        checked += len(fens)

    print(
        "Run 6B trainer replay passed: "
        f"positions={checked}, raw_sha256={digest.hexdigest().upper()}, "
        f"description={network.description!r}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, horde_run6b.Run6BError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
