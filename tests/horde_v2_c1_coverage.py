#!/usr/bin/env python3
"""Independent feature-index and coverage semantics for the C1 addendum."""

from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import horde_training_decoder as decoder  # noqa: E402
import horde_v2_c1_campaign as campaign  # noqa: E402


class FakeDataset:
    def __init__(self, records: list[SimpleNamespace]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def record(self, index: int) -> SimpleNamespace:
        return self.records[index]


def _board(king_square: int, offset: int, roles: tuple[int, ...] = tuple(range(1, 11))) -> tuple[int, ...]:
    board = [0] * 64
    board[king_square] = 11
    used = {king_square}
    for index, code in enumerate(roles):
        square = (offset + index * 5) % 64
        while square in used:
            square = (square + 1) % 64
        board[square] = code
        used.add(square)
    return tuple(board)


def _record(index: int, king_square: int, offset: int, roles: tuple[int, ...] = tuple(range(1, 11))) -> SimpleNamespace:
    board = _board(king_square, offset, roles)
    return SimpleNamespace(
        board=board,
        features=decoder.extract_sparse_features(board),
        side_to_move=index % 2,
        result=(index // 2) % 3 - 1,
        outcome_reason=3,
        castling_rights=0,
        ep_square=64,
        best_move=0,
        played_move=0,
        score=index - 16,
    )


def _bucket_records(offset: int, roles: tuple[int, ...] = tuple(range(1, 11))) -> list[SimpleNamespace]:
    records: list[SimpleNamespace] = []
    for bucket in range(32):
        rank, canonical_file = divmod(bucket, 4)
        records.append(_record(bucket, rank * 8 + canonical_file + 4, offset, roles))
    return records


def _overlap() -> dict[str, object]:
    return {
        "physical": {
            "cross_role_overlap_samples": 0,
            "validation_duplicate_samples": 0,
        },
        "legacy_model_input": {
            "cross_role_overlap_samples": 0,
            "validation_duplicate_samples": 0,
        },
    }


def _independent_rows(board: tuple[int, ...]) -> tuple[set[int], set[int], set[int]]:
    king_square = board.index(11)
    king_file = king_square & 7
    mirror = king_file <= 3
    canonical_file = king_file ^ 7 if mirror else king_file
    bucket = (king_square >> 3) * 4 + canonical_file - 4
    absolute: set[int] = set()
    rank8: set[int] = set()
    royal32: set[int] = set()
    for square, code in enumerate(board):
        if not code or code == 11:
            continue
        role = code - 1
        absolute.add(role * 64 + square)
        oriented = square ^ 7 if mirror else square
        within = role * 64 + oriented
        royal32.add(bucket * 640 + within)
        rank8.add((bucket // 4) * 640 + within)
    return absolute, rank8, royal32


def main() -> int:
    seen_buckets: set[int] = set()
    seen_ranks: set[int] = set()
    seen_mirrors: set[bool] = set()
    for king_square in range(64):
        record = _record(king_square, king_square, (king_square + 13) % 64)
        expected = _independent_rows(record.board)
        observed = tuple(
            set(campaign._feature_rows(record, topology)[0])
            for topology in ("absolute_nonking", "royal_rank8", "royal32")
        )
        if observed != expected:
            raise AssertionError(f"independent C1 row enumeration differs at {king_square}")
        seen_buckets.add(record.features.royal_bucket)
        seen_ranks.add(record.features.royal_bucket // 4)
        seen_mirrors.add(record.features.royal_mirror)
        reflected_board = decoder.horizontal_reflect_board(record.board)
        reflected = SimpleNamespace(features=decoder.extract_sparse_features(reflected_board))
        for topology in ("royal_rank8", "royal32"):
            if set(campaign._feature_rows(record, topology)[0]) != set(
                campaign._feature_rows(reflected, topology)[0]
            ):
                raise AssertionError(f"{topology} reflection contract drifted")
    if seen_buckets != set(range(32)) or seen_ranks != set(range(8)) or seen_mirrors != {False, True}:
        raise AssertionError("feature-index test did not cover every key and mirror orientation")

    train_records = _bucket_records(0)
    shared_validation = _bucket_records(0)
    shared = campaign._coverage_receipt(
        FakeDataset(train_records), FakeDataset(shared_validation), _overlap()
    )
    campaign._validate_coverage_receipt(shared, (32, 32))
    for topology in campaign.TOPOLOGY_SPECS:
        gates = shared["topologies"][topology]["gates"]
        if not gates["all_keys_nonzero_in_both_roles"]:
            raise AssertionError(f"{topology} rejected complete key coverage")
        if not gates["all_fixed_roles_nonzero_in_both_roles"]:
            raise AssertionError(f"{topology} rejected complete fixed-role coverage")
        if not gates["every_validation_key_has_training_row_intersection"]:
            raise AssertionError(f"{topology} rejected exact row intersections")

    missing_key = campaign._coverage_receipt(
        FakeDataset(train_records), FakeDataset(shared_validation[:-1]), _overlap()
    )
    if missing_key["topologies"]["royal32"]["gates"]["all_keys_nonzero_in_both_roles"]:
        raise AssertionError("Royal-32 accepted a missing validation key")

    roles_without_white_queen = tuple(code for code in range(1, 11) if code != 5)
    missing_role = campaign._coverage_receipt(
        FakeDataset(_bucket_records(0, roles_without_white_queen)),
        FakeDataset(_bucket_records(0, roles_without_white_queen)),
        _overlap(),
    )
    if missing_role["topologies"]["royal32"]["gates"][
        "all_fixed_roles_nonzero_in_both_roles"
    ]:
        raise AssertionError("Royal-32 accepted a missing fixed role")

    disjoint_rows = campaign._coverage_receipt(
        FakeDataset(train_records), FakeDataset(_bucket_records(23)), _overlap()
    )
    if disjoint_rows["topologies"]["royal32"]["gates"][
        "every_validation_key_has_training_row_intersection"
    ]:
        raise AssertionError("Royal-32 accepted a key with no exact row intersection")

    if not campaign._seen_mass_gate(1, 100):
        raise AssertionError("exactly one percent unseen activation should pass")
    if campaign._seen_mass_gate(2, 100):
        raise AssertionError("one activation beyond one percent should fail")

    print(
        "Horde V2 C1 coverage passed: independent ABS/R8/R32 rows, all keys and roles, "
        "per-key intersections and exact 99/100 support boundary"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
