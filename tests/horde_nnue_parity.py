#!/usr/bin/env python3
"""Differential raw-NNUE parity gate for the frozen HordeTest Run 6B net."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import chess
from chess.variant import HordeBoard


ROOT = Path(__file__).resolve().parents[1]
RAW_RE = re.compile(r"^horde-raw-eval (-?\d+) (-?\d+) (-?\d+)$")
EVAL_RE = re.compile(r"^horde-eval-debug eval=(-?\d+)$")

SCRIPTED_GAMES = (
    # A legal ordinary en-passant capture by the Horde side.
    ("h4h5", "a7a5", "b5a6"),
    # A legal Black king-side castle.
    ("h4h5", "g8f6", "a4a5", "g7g6", "d4d5", "f8g7", "e4e5", "e8g8"),
    # A legal white capture-promotion. Promoted pieces remain ordinary pieces.
    ("b5b6", "h7h6", "b6a7", "h6h5", "a7b8q"),
)


def parse_args() -> argparse.Namespace:
    default_name = "stockfish.exe" if os.name == "nt" else "stockfish"
    parser = argparse.ArgumentParser(
        description="Compare Horde-Stockfish raw Run 6B output with pinned Fairy-Stockfish"
    )
    parser.add_argument(
        "--candidate", type=Path, default=ROOT / "src" / default_name
    )
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument(
        "--network",
        type=Path,
        default=ROOT / "networks" / "hordetest_run6b_e37_l06.nnue",
    )
    parser.add_argument(
        "--variant",
        type=Path,
        default=ROOT / "docs" / "horde" / "fixtures" / "variants.ini",
    )
    parser.add_argument("--positions", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=2_500)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x6B37_06)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--allow-incomplete-coverage", action="store_true")
    parser.add_argument("--write-corpus", type=Path)
    return parser.parse_args()


def white_piece_count(board: HordeBoard) -> int:
    return len(board.pieces(chess.PAWN, chess.WHITE)) + sum(
        len(board.pieces(piece_type, chess.WHITE))
        for piece_type in (
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK,
            chess.QUEEN,
        )
    )


def feature_key(board: HordeBoard) -> str:
    """Return the rule-state fields that can affect a feature position."""
    return " ".join(board.fen(en_passant="fen").split()[:4])


def note_transition(board: HordeBoard, move: chess.Move, coverage: Counter[str]) -> None:
    if board.is_capture(move):
        coverage["capture"] += 1
    if board.is_en_passant(move):
        coverage["en_passant"] += 1
    if board.is_castling(move):
        coverage["castling"] += 1
    if move.promotion:
        coverage["promotion"] += 1


def add_position(
    board: HordeBoard,
    positions: list[str],
    seen: set[str],
    coverage: Counter[str],
) -> None:
    if board.is_game_over(claim_draw=True) or white_piece_count(board) == 0:
        return

    key = feature_key(board)
    if key in seen:
        return
    seen.add(key)
    positions.append(board.fen(en_passant="fen"))
    count = white_piece_count(board)
    coverage[f"white_pieces_{count}"] += 1
    if 1 <= count <= 4:
        coverage["white_pieces_1_to_4"] += 1


def add_scripted_games(
    positions: list[str], seen: set[str], coverage: Counter[str]
) -> None:
    for sequence in SCRIPTED_GAMES:
        board = HordeBoard()
        add_position(board, positions, seen, coverage)
        for uci in sequence:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise RuntimeError(f"scripted Horde move is no longer legal: {uci}")
            note_transition(board, move, coverage)
            board.push(move)
            add_position(board, positions, seen, coverage)


def weighted_choice(
    board: HordeBoard, moves: Sequence[chess.Move], rng: random.Random, game_index: int
) -> chess.Move:
    weights: list[float] = []
    capture_mode = game_index % 3 != 0
    promotion_mode = game_index % 5 == 0

    for move in moves:
        weight = 1.0
        is_capture = board.is_capture(move)

        if move.promotion:
            weight *= 1_000.0
        if board.is_en_passant(move):
            weight *= 800.0
        if board.is_castling(move):
            weight *= 120.0

        if board.turn == chess.BLACK:
            if is_capture:
                weight *= 60.0 if capture_mode else 8.0
            elif capture_mode:
                weight *= 0.35
        else:
            if is_capture:
                weight *= 2.5
            if promotion_mode:
                rank_gain = chess.square_rank(move.to_square) - chess.square_rank(
                    move.from_square
                )
                if rank_gain > 0:
                    weight *= 1.0 + rank_gain * 2.0
                if board.piece_type_at(move.from_square) == chess.PAWN:
                    weight *= 2.0

        weights.append(weight)

    return rng.choices(moves, weights=weights, k=1)[0]


def generate_positions(total: int, seed: int) -> tuple[list[str], Counter[str]]:
    if total < 1:
        raise ValueError("--positions must be positive")

    rng = random.Random(seed)
    positions: list[str] = []
    seen: set[str] = set()
    coverage: Counter[str] = Counter()
    add_scripted_games(positions, seen, coverage)

    game_index = 0
    duplicate_streak = 0
    while len(positions) < total:
        board = HordeBoard()
        before = len(positions)
        add_position(board, positions, seen, coverage)

        for _ply in range(500):
            if board.is_game_over(claim_draw=True):
                break
            moves = list(board.legal_moves)
            if not moves:
                break
            move = weighted_choice(board, moves, rng, game_index)
            note_transition(board, move, coverage)
            board.push(move)
            add_position(board, positions, seen, coverage)
            if len(positions) >= total:
                break

        if len(positions) == before:
            duplicate_streak += 1
            if duplicate_streak >= 100:
                raise RuntimeError("position generation stopped making progress")
        else:
            duplicate_streak = 0
        game_index += 1

    return positions[:total], coverage


def to_hordetest_fen(fen: str) -> str:
    fields = fen.split()
    fields[0] = fields[0].replace("P", "H")
    return " ".join(fields)


def network_eval_batch(
    executable: Path,
    prefix: Iterable[str],
    fens: Sequence[str],
    hordetest: bool,
    cwd: Path,
    timeout: float,
) -> tuple[list[tuple[int, int, int]], list[int]]:
    commands = list(prefix)
    for fen in fens:
        commands.append(f"position fen {to_hordetest_fen(fen) if hordetest else fen}")
        commands.append("horde-raw-eval")
        commands.append("horde-eval-debug")
    commands.append("quit")

    completed = subprocess.run(
        [str(executable)],
        cwd=cwd,
        input=("\n".join(commands) + "\n").encode("ascii"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{executable.name} exited with {completed.returncode}:\n{output[-4000:]}"
        )

    raw_values = [
        tuple(map(int, match.groups()))
        for line in output.splitlines()
        if (match := RAW_RE.fullmatch(line.strip()))
    ]
    eval_values = [
        int(match.group(1))
        for line in output.splitlines()
        if (match := EVAL_RE.fullmatch(line.strip()))
    ]
    if len(raw_values) != len(fens) or len(eval_values) != len(fens):
        raise RuntimeError(
            f"{executable.name} returned {len(raw_values)} raw and "
            f"{len(eval_values)} final evaluations for {len(fens)} positions:\n"
            f"{output[-4000:]}"
        )
    return raw_values, eval_values


def raw_eval_batch(
    executable: Path,
    prefix: Iterable[str],
    fens: Sequence[str],
    hordetest: bool,
    cwd: Path,
    timeout: float,
) -> list[tuple[int, int, int]]:
    """Compatibility wrapper for raw-only determinism consumers."""
    return network_eval_batch(executable, prefix, fens, hordetest, cwd, timeout)[0]


def with_rule50(fen: str, count: int) -> str:
    fields = fen.split()
    fields[4] = str(count)
    return " ".join(fields)


def validate_paths(args: argparse.Namespace) -> None:
    for label in ("candidate", "oracle", "network", "variant"):
        path = getattr(args, label).resolve()
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")
        setattr(args, label, path)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")


def require_coverage(coverage: Counter[str]) -> None:
    missing = [
        label
        for label in ("capture", "promotion", "en_passant", "castling")
        if coverage[label] == 0
    ]
    missing.extend(
        f"white_pieces_{count}"
        for count in range(1, 5)
        if coverage[f"white_pieces_{count}"] == 0
    )
    if missing:
        raise RuntimeError("generated corpus lacks required coverage: " + ", ".join(missing))


def corpus_digest(positions: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(positions) + "\n").encode("ascii")).hexdigest()


def main() -> int:
    args = parse_args()
    validate_paths(args)
    positions, coverage = generate_positions(args.positions, args.seed)

    digest = corpus_digest(positions)
    print(
        f"generated {len(positions)} unique reachable Horde positions; "
        f"seed=0x{args.seed:X}; sha256={digest}",
        flush=True,
    )
    print(
        "coverage "
        + " ".join(
            f"{label}={coverage[label]}"
            for label in (
                "capture",
                "promotion",
                "en_passant",
                "castling",
                "white_pieces_1",
                "white_pieces_2",
                "white_pieces_3",
                "white_pieces_4",
            )
        ),
        flush=True,
    )
    if not args.allow_incomplete_coverage:
        require_coverage(coverage)

    if args.write_corpus:
        output_path = args.write_corpus.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(positions) + "\n", encoding="ascii")
        print(f"wrote corpus: {output_path}", flush=True)

    if args.generate_only:
        return 0

    with tempfile.TemporaryDirectory(prefix="horde-nnue-parity-") as temp_name:
        temp = Path(temp_name)
        shutil.copyfile(args.variant, temp / "variants.ini")

        # Fairy-Stockfish resolves EvalFile relative to the executable rather than
        # the process working directory. Use the absolute path; UCI string options
        # preserve the spaces in Windows paths.
        network_option = str(args.network)

        candidate_prefix = (
            "uci",
            f"setoption name EvalFile value {network_option}",
            "setoption name Threads value 1",
            "isready",
        )
        oracle_prefix = (
            "load variants.ini",
            "uci",
            "setoption name UCI_Variant value hordetest",
            f"setoption name EvalFile value {network_option}",
            "setoption name Threads value 1",
            "isready",
        )

        checked = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for start in range(0, len(positions), args.batch_size):
                batch = positions[start : start + args.batch_size]
                candidate_job = pool.submit(
                    network_eval_batch,
                    args.candidate,
                    candidate_prefix,
                    batch,
                    False,
                    temp,
                    args.timeout,
                )
                oracle_job = pool.submit(
                    network_eval_batch,
                    args.oracle,
                    oracle_prefix,
                    batch,
                    True,
                    temp,
                    args.timeout,
                )
                candidate_raw, candidate_eval = candidate_job.result()
                oracle_raw, oracle_eval = oracle_job.result()

                for offset, (candidate, oracle) in enumerate(
                    zip(candidate_raw, oracle_raw, strict=True)
                ):
                    if candidate != oracle:
                        index = start + offset
                        raise RuntimeError(
                            f"raw NNUE mismatch at position {index}: "
                            f"candidate={candidate}, oracle={oracle}, fen={batch[offset]}"
                        )
                for offset, (candidate, oracle) in enumerate(
                    zip(candidate_eval, oracle_eval, strict=True)
                ):
                    if candidate != oracle:
                        index = start + offset
                        raise RuntimeError(
                            f"final NNUE mismatch at position {index}: "
                            f"candidate={candidate}, oracle={oracle}, fen={batch[offset]}"
                        )
                checked += len(batch)
                print(f"parity {checked}/{len(positions)}", flush=True)

            rule50_positions = [with_rule50(positions[0], count) for count in (0, 50, 90, 99)]
            candidate_job = pool.submit(
                network_eval_batch,
                args.candidate,
                candidate_prefix,
                rule50_positions,
                False,
                temp,
                args.timeout,
            )
            oracle_job = pool.submit(
                network_eval_batch,
                args.oracle,
                oracle_prefix,
                rule50_positions,
                True,
                temp,
                args.timeout,
            )
            candidate_raw, candidate_eval = candidate_job.result()
            oracle_raw, oracle_eval = oracle_job.result()
            if candidate_raw != oracle_raw or candidate_eval != oracle_eval:
                raise RuntimeError(
                    "rule50-stratified NNUE mismatch: "
                    f"candidate_raw={candidate_raw}, oracle_raw={oracle_raw}, "
                    f"candidate_eval={candidate_eval}, oracle_eval={oracle_eval}"
                )
            print("rule50 parity 0/50/90/99", flush=True)

    print(
        f"Run 6B raw/final NNUE parity completed successfully: {len(positions)} positions",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
