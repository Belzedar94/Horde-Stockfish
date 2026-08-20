#!/usr/bin/env python3
"""Fail-closed invariants for the V3 contextual White-pawn blocks.

Checks the block layout, the per-block active-row bounds, the exact predicate
definitions on hand built boards, horizontal-reflection equivariance, and the
range and uniqueness guarantees, both on synthetic boards and on authenticated
HORDE_BIN_V1 records when a corpus is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import horde_bin_v1 as wire  # noqa: E402
import horde_training_decoder as dec  # noqa: E402


FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def empty_board() -> list[int]:
    board = [0] * 64
    board[60] = dec.BLACK_KING
    return board


def place(board: list[int], squares: dict[str, int]) -> list[int]:
    for name, code in squares.items():
        file_index = ord(name[0]) - ord("a")
        rank = int(name[1]) - 1
        board[rank * 8 + file_index] = code
    return board


def rows_of(board, name: str) -> list[int]:
    base = dec.V3_BLOCK_BASES[name]
    width = dec.V3_BLOCK_WIDTHS[name]
    return [row - base for row in dec.v3_contextual_rows(board) if base <= row < base + width]


def test_layout() -> None:
    total = sum(dec.V3_BLOCK_WIDTHS[name] for name in dec.V3_BLOCK_ORDER)
    check(total == dec.V3_CONTEXTUAL_ROWS, f"contextual widths sum to {total}, not 192")
    check(dec.V3_GLOBAL_DIMENSIONS == 896, "V3 stream is not 896 rows")
    cursor = dec.V2_GLOBAL_DIMENSIONS
    for name in dec.V3_BLOCK_ORDER:
        check(dec.V3_BLOCK_BASES[name] == cursor, f"block {name} base is not contiguous")
        cursor += dec.V3_BLOCK_WIDTHS[name]
    check(cursor == dec.V3_GLOBAL_DIMENSIONS, "block bases do not tile the stream")
    check(dec.V3_MAX_ACTIVE_ROWS == 100, "documented active-row bound drifted")


def test_predicates() -> None:
    # d4 alone: frontier and rearmost are the same pawn, no neighbours, unblocked.
    board = place(empty_board(), {"d4": dec.WHITE_PAWN})
    d_file = 3
    check(rows_of(board, "frontier_square") == [3 * 8 + d_file], "frontier square wrong for d4")
    check(rows_of(board, "rearmost_square") == [3 * 8 + d_file], "rearmost square wrong for d4")
    check(
        rows_of(board, "file_count") == [d_file * dec.V3_FILE_COUNT_STATES + 0],
        "file count wrong for a single pawn",
    )
    check(rows_of(board, "frontier_blocked") == [], "lone d4 pawn reported as blocked")
    check(rows_of(board, "frontier_supported") == [], "lone d4 pawn reported as supported")
    check(rows_of(board, "frontier_phalanx") == [], "lone d4 pawn reported in a phalanx")

    # Stacked pawns separate frontier from rearmost and set the count state.
    board = place(empty_board(), {"d2": dec.WHITE_PAWN, "d4": dec.WHITE_PAWN, "d6": dec.WHITE_PAWN})
    check(rows_of(board, "frontier_square") == [5 * 8 + d_file], "frontier is not the top pawn")
    check(rows_of(board, "rearmost_square") == [1 * 8 + d_file], "rearmost is not the bottom pawn")
    check(
        rows_of(board, "file_count") == [d_file * dec.V3_FILE_COUNT_STATES + 2],
        "file count state is not count minus one",
    )

    # Blocked by any occupancy, friendly or enemy, pawn or piece.
    for code in (2, 5, 6, 9):
        board = place(empty_board(), {"d4": dec.WHITE_PAWN, "d5": code})
        check(
            rows_of(board, "frontier_blocked") == [d_file],
            f"frontier not blocked by piece code {code}",
        )
    # A White pawn directly in front becomes the frontier itself, so the lower
    # pawn is never the one tested and the file is only blocked if that new
    # frontier is blocked in turn.
    board = place(empty_board(), {"d4": dec.WHITE_PAWN, "d5": dec.WHITE_PAWN})
    check(rows_of(board, "frontier_square") == [4 * 8 + d_file], "stacked frontier is not the top")
    check(rows_of(board, "frontier_blocked") == [], "stacked pawns reported the file as blocked")
    board = place(empty_board(), {"d4": dec.WHITE_PAWN, "d5": dec.WHITE_PAWN, "d6": 9})
    check(rows_of(board, "frontier_blocked") == [d_file], "top pawn of a stack is not blocked")

    # Diagonal support comes from one rank behind, phalanx from the same rank.
    board = place(empty_board(), {"d4": dec.WHITE_PAWN, "c3": dec.WHITE_PAWN})
    check(rows_of(board, "frontier_supported") == [d_file], "c3 does not support d4")
    check(rows_of(board, "frontier_phalanx") == [], "c3 wrongly counted as a d4 phalanx")
    board = place(empty_board(), {"d4": dec.WHITE_PAWN, "e4": dec.WHITE_PAWN})
    phalanx = sorted(rows_of(board, "frontier_phalanx"))
    check(phalanx == [3, 4], f"d4/e4 phalanx reported {phalanx}")
    check(rows_of(board, "frontier_supported") == [], "same-rank neighbour counted as support")
    # A pawn one rank behind on the same file is neither support nor phalanx.
    board = place(empty_board(), {"d4": dec.WHITE_PAWN, "d3": dec.WHITE_PAWN})
    check(rows_of(board, "frontier_supported") == [], "same-file pawn counted as support")
    check(rows_of(board, "frontier_phalanx") == [], "same-file pawn counted as a phalanx")
    # Support is judged on the frontier pawn only, not on rear pawns.
    board = place(empty_board(), {"d6": dec.WHITE_PAWN, "d2": dec.WHITE_PAWN, "c1": dec.WHITE_PAWN})
    check(rows_of(board, "frontier_supported") == [], "support judged on a rear pawn")

    # Edge files must not wrap around the board.
    board = place(empty_board(), {"a4": dec.WHITE_PAWN, "h3": dec.WHITE_PAWN})
    check(rows_of(board, "frontier_phalanx") == [], "a-file and h-file wrapped into a phalanx")
    check(rows_of(board, "frontier_supported") == [], "a-file and h-file wrapped into support")


def test_bounds_and_reflection(trials: int = 4000) -> None:
    rng = random.Random(20260820)
    worst = 0
    for _ in range(trials):
        board = [0] * 64
        board[rng.randrange(56, 64)] = dec.BLACK_KING
        for _ in range(rng.randrange(0, 40)):
            square = rng.randrange(0, 56)
            if board[square] == 0:
                board[square] = dec.WHITE_PAWN
        for _ in range(rng.randrange(0, 8)):
            square = rng.randrange(0, 64)
            if board[square] == 0:
                board[square] = rng.choice((2, 3, 4, 5, 7, 8, 9, 10))
        if sum(1 for code in board if 1 <= code <= 5) > 36:
            continue
        if sum(1 for code in board if 6 <= code <= 11) > 16:
            continue

        rows = dec.v3_contextual_rows(board)
        worst = max(worst, len(rows))
        check(len(rows) <= dec.V3_MAX_ACTIVE_CONTEXTUAL_ROWS, "contextual bound exceeded")
        check(len(set(rows)) == len(rows), "duplicate contextual row")
        for name in dec.V3_BLOCK_ORDER:
            base, width = dec.V3_BLOCK_BASES[name], dec.V3_BLOCK_WIDTHS[name]
            active = [row for row in rows if base <= row < base + width]
            check(len(active) <= 8, f"block {name} exceeded eight active rows")

        reflected = dec.horizontal_reflect_board(board)
        check(
            dec.horizontal_reflect_board(reflected) == tuple(board),
            "reflection is not an involution",
        )
        expected = tuple(sorted(dec.reflect_v3_row(row) for row in rows))
        observed = tuple(sorted(dec.v3_contextual_rows(list(reflected))))
        check(expected == observed, "contextual rows are not reflection equivariant")

        stream = dec.v3_global_rows(board)
        check(len(stream) <= dec.V3_MAX_ACTIVE_ROWS, "V3 stream exceeded its active bound")
        check(
            stream[: len(stream) - len(rows)] == dec.extract_sparse_features(board).v2_global,
            "V3 stream does not begin with the immutable G0 rows",
        )
        reflected_stream = tuple(sorted(dec.reflect_v3_row(row) for row in stream))
        check(
            reflected_stream == tuple(sorted(dec.v3_global_rows(list(reflected)))),
            "V3 stream is not reflection equivariant",
        )
    print(f"  randomized boards: {trials}, maximum contextual rows observed: {worst}")


def test_rejections() -> None:
    for board, reason in (
        ([0] * 63, "short board"),
        ([99] * 64, "invalid piece code"),
    ):
        try:
            dec.v3_contextual_rows(board)
        except wire.FormatError:
            continue
        FAILURES.append(f"contextual extractor accepted a {reason}")
    illegal = empty_board()
    illegal[7 * 8 + 3] = dec.WHITE_PAWN
    try:
        dec.v3_contextual_rows(illegal)
        FAILURES.append("contextual extractor accepted a White pawn on rank 8")
    except wire.FormatError:
        pass
    try:
        dec.reflect_v3_row(dec.V3_GLOBAL_DIMENSIONS)
        FAILURES.append("reflect_v3_row accepted an out-of-stream row")
    except wire.FormatError:
        pass


def test_corpus(path: Path, limit: int) -> None:
    raw = path.read_bytes()
    stride = wire.RECORD_SIZE
    count = min(limit, len(raw) // stride)
    worst = 0
    histogram = {name: 0 for name in dec.V3_BLOCK_ORDER}
    for index in range(count):
        record = wire.validate_record(raw[index * stride : (index + 1) * stride], index)
        board = record["board"]
        rows = dec.v3_contextual_rows(board)
        worst = max(worst, len(rows))
        check(len(rows) <= dec.V3_MAX_ACTIVE_CONTEXTUAL_ROWS, f"record {index} exceeded the bound")
        for name in dec.V3_BLOCK_ORDER:
            base, width = dec.V3_BLOCK_BASES[name], dec.V3_BLOCK_WIDTHS[name]
            histogram[name] += sum(1 for row in rows if base <= row < base + width)
        reflected = dec.horizontal_reflect_board(board)
        check(
            tuple(sorted(dec.reflect_v3_row(row) for row in rows))
            == tuple(sorted(dec.v3_contextual_rows(list(reflected)))),
            f"record {index} broke reflection equivariance",
        )
        check(len(dec.v3_global_rows(board)) <= dec.V3_MAX_ACTIVE_ROWS, f"record {index} stream")
    print(f"  authenticated records: {count}, maximum contextual rows observed: {worst}")
    print("  mean active rows per block: " + ", ".join(
        f"{name}={histogram[name] / max(count, 1):.2f}" for name in dec.V3_BLOCK_ORDER
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--limit", type=int, default=20000)
    args = parser.parse_args(argv)

    print("V3 contextual decoder invariants")
    test_layout()
    test_predicates()
    test_rejections()
    test_bounds_and_reflection()
    if args.records:
        test_corpus(args.records, args.limit)

    if FAILURES:
        print(f"\nFAILED with {len(FAILURES)} problems:")
        for failure in FAILURES[:20]:
            print(f"  {failure}")
        return 1
    print("\nall V3 decoder invariants passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
