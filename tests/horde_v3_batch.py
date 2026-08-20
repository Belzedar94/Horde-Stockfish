#!/usr/bin/env python3
"""Fail-closed invariants for the V3 training batch ABI.

Round trips a decoded sparse batch through ``_torch_v3_batch`` and checks that
the tensor batch the model consumes still carries exactly what the decoder
produced: offsets that tile the row array, an immutable G0 prefix identical to
the plain V2 Global stream for the same board, a contextual tail identical to
``v3_contextual_rows``, phase buckets from the serialized lookup, and the
unchanged mate-score eligibility rule.  Then it corrupts each of those
invariants in turn and requires the batch builder to refuse the batch.

Runs on one CPU thread and needs no corpus; pass ``--records`` to run the same
checks over authenticated HORDE_BIN_V1 records instead.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import torch  # noqa: E402

import horde_bin_v1 as wire  # noqa: E402
import horde_training_control as control  # noqa: E402
import horde_training_decoder as dec  # noqa: E402
import horde_training_models as models  # noqa: E402


V3_ARCHITECTURE = "v3-g1024-pawn-wpc8"
CONTROL_ARCHITECTURE = "v2-c0-g0single-256"
DEFAULT_CORPUS = Path(r"D:/horde-train/validation-selected/selected-records.bin")
DEVICE = torch.device("cpu")

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def build_records(path: Path | None, count: int) -> list[dec.TrainingRecord]:
    """Decode ``count`` records from the corpus, or from a synthetic payload."""

    if path is not None and path.exists():
        payload = path.read_bytes()
        available = len(payload) // wire.RECORD_SIZE
        count = min(count, available)
        print(f"  source: authenticated corpus {path.name} ({available} records)")
    else:
        payload = dec.synthetic_record_payload(count)
        print(f"  source: deterministic synthetic payload ({count} records)")
    return [
        dec.decode_training_record(
            payload[index * wire.RECORD_SIZE : (index + 1) * wire.RECORD_SIZE],
            index,
        )
        for index in range(count)
    ]


def rejects(sparse: dec.SparseBatch, reason: str) -> None:
    try:
        control._torch_v3_batch(sparse, DEVICE)
    except control.TrainingError:
        return
    except Exception as error:  # noqa: BLE001 - a wrong exception type is a failure
        FAILURES.append(f"{reason}: raised {type(error).__name__} instead of TrainingError")
        return
    FAILURES.append(f"{reason}: the batch builder accepted a corrupt batch")


def corrupt(sparse: dec.SparseBatch, **changes: object) -> dec.SparseBatch:
    return dataclasses.replace(sparse, **changes)


def test_round_trip(records: list[dec.TrainingRecord]) -> dec.SparseBatch:
    sparse = dec.make_sparse_batch(records, contextual=dec.v3_contextual_rows)
    plain = dec.make_sparse_batch(records)
    batch = control._torch_v3_batch(sparse, DEVICE)

    check(
        batch.v3_global.tolist() == list(sparse.v2_global),
        "V3 batch row array is not the decoded stream",
    )
    check(
        batch.global_offsets.tolist() == list(sparse.global_offsets),
        "V3 batch offsets are not the decoded offsets",
    )
    check(
        batch.side_to_move.tolist() == list(sparse.side_to_move),
        "V3 batch side to move drifted",
    )
    check(
        batch.rule50_count.tolist() == list(sparse.rule50_count),
        "V3 batch rule50 counts drifted",
    )
    check(
        batch.results.tolist() == list(sparse.results),
        "V3 batch results drifted",
    )
    check(
        batch.scores.tolist() == [float(score) for score in sparse.scores],
        "V3 batch scores drifted",
    )
    for name in ("v3_global", "global_offsets", "side_to_move", "phase_buckets", "results"):
        check(
            getattr(batch, name).dtype == torch.long,
            f"V3 batch field {name} is not a long tensor",
        )
    check(batch.scores.dtype == torch.float32, "V3 batch scores are not float32")
    check(batch.score_eligible.dtype == torch.bool, "V3 batch eligibility is not boolean")

    offsets = list(sparse.global_offsets)
    check(offsets[0] == 0, "V3 offsets do not start at zero")
    check(offsets[-1] == len(sparse.v2_global), "V3 offsets do not end at the row count")
    check(len(offsets) == len(sparse) + 1, "V3 offsets do not frame every record")
    check(
        all(offsets[index] <= offsets[index + 1] for index in range(len(sparse))),
        "V3 offsets are not monotone",
    )
    tiled: list[int] = []
    for index in range(len(sparse)):
        tiled.extend(sparse.v2_global[offsets[index] : offsets[index + 1]])
    check(tiled == list(sparse.v2_global), "V3 offsets do not tile the row array")

    plain_offsets = list(plain.global_offsets)
    contextual_rows = 0
    for index, record in enumerate(records):
        begin, end = offsets[index], offsets[index + 1]
        pieces = sparse.physical_piece_count[index]
        prefix = sparse.v2_global[begin : begin + pieces]
        tail = sparse.v2_global[begin + pieces : end]
        contextual_rows += len(tail)

        expected_g0 = dec.extract_sparse_features(record.board).v2_global
        plain_g0 = plain.v2_global[plain_offsets[index] : plain_offsets[index + 1]]
        check(prefix == expected_g0, f"record {index} G0 prefix is not the board G0 stream")
        check(prefix == plain_g0, f"record {index} G0 prefix differs from the plain V2 stream")
        check(
            tail == dec.v3_contextual_rows(record.board),
            f"record {index} contextual tail is not v3_contextual_rows",
        )
        check(
            len(set(sparse.v2_global[begin:end])) == end - begin,
            f"record {index} repeats a V3 stream row",
        )
        check(
            end - begin <= dec.V3_MAX_ACTIVE_ROWS,
            f"record {index} exceeded the registered active-row bound",
        )
        check(
            all(row < dec.V2_GLOBAL_DIMENSIONS for row in prefix),
            f"record {index} G0 prefix left the immutable rows",
        )
        check(
            all(
                dec.V2_GLOBAL_DIMENSIONS <= row < dec.V3_GLOBAL_DIMENSIONS
                for row in tail
            ),
            f"record {index} contextual row left its reserved range",
        )

    expected_buckets = [
        models.V3_PHASE_LOOKUP[count] for count in sparse.white_piece_count
    ]
    check(
        batch.phase_buckets.tolist() == expected_buckets,
        "V3 phase buckets do not follow the serialized lookup",
    )
    expected_eligible = [
        abs(score) < control.MATE_SCORE_THRESHOLD for score in sparse.scores
    ]
    check(
        batch.score_eligible.tolist() == expected_eligible,
        "V3 score eligibility does not follow the mate threshold",
    )

    forward = control._make_model(V3_ARCHITECTURE, 0xC0FFEE)
    with torch.no_grad():
        values = forward(batch)
    check(values.shape == (len(sparse),), "V3 model did not accept the batch shape")
    check(bool(torch.isfinite(values).all()), "V3 model produced a non-finite value")

    print(f"  records                : {len(sparse)}")
    print(f"  G0 rows                : {sum(sparse.physical_piece_count)}")
    print(f"  contextual rows        : {contextual_rows}")
    print(f"  buckets present        : {sorted(set(expected_buckets))}")
    print(f"  score eligible         : {sum(expected_eligible)} of {len(sparse)}")
    return sparse


def test_routing(records: list[dec.TrainingRecord]) -> None:
    """Only V3 may extend the Global stream; every other rung sees G0 alone."""

    class Fixture:
        def __init__(self, items: list[dec.TrainingRecord]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def record(self, index: int) -> dec.TrainingRecord:
            return self.items[index]

    dataset = Fixture(records)
    indices = tuple(range(len(records)))
    physical = sum(sum(code != 0 for code in record.board) for record in records)

    v3_sparse = control._load_sparse_batch(V3_ARCHITECTURE, dataset, indices)
    check(
        len(v3_sparse.v2_global) > physical,
        "the V3 architecture did not receive the contextual extractor",
    )
    check(
        sum(v3_sparse.physical_piece_count) == physical,
        "the V3 physical piece count absorbed contextual rows",
    )
    for architecture in (
        control.LEGACY_ARCHITECTURE,
        "v2-64x192",
        "v2-c1-rank8-64x192",
        CONTROL_ARCHITECTURE,
    ):
        sparse = control._load_sparse_batch(architecture, dataset, indices)
        check(
            len(sparse.v2_global) == physical,
            f"{architecture} received a contextual tail it must never see",
        )
    control_batch = control._model_batch(
        CONTROL_ARCHITECTURE,
        control._load_sparse_batch(CONTROL_ARCHITECTURE, dataset, indices),
        DEVICE,
    )
    check(
        isinstance(control_batch, control.V2Batch),
        "the R2 control did not receive an ordinary V2 batch",
    )
    with torch.no_grad():
        values = control._make_model(CONTROL_ARCHITECTURE, 0xC0FFEE)(control_batch)
    check(bool(torch.isfinite(values).all()), "the R2 control produced a non-finite value")
    try:
        control._contextual_extractor("v9-does-not-exist")
        FAILURES.append("the extractor router accepted an unregistered architecture")
    except control.TrainingError:
        pass


def test_rejections(sparse: dec.SparseBatch) -> None:
    offsets = list(sparse.global_offsets)
    rows = list(sparse.v2_global)
    pieces = list(sparse.physical_piece_count)

    tail_record = next(
        (
            index
            for index in range(len(sparse))
            if offsets[index + 1] - offsets[index] > pieces[index]
        ),
        None,
    )
    if tail_record is None:
        FAILURES.append("no fixture record carries a contextual tail to corrupt")
    else:
        out_of_range = list(rows)
        out_of_range[offsets[tail_record + 1] - 1] = dec.V3_GLOBAL_DIMENSIONS
        rejects(
            corrupt(sparse, v2_global=tuple(out_of_range)),
            "contextual row above the stream",
        )

    prefix_escape = list(rows)
    prefix_escape[offsets[0]] = dec.V2_GLOBAL_DIMENSIONS
    rejects(corrupt(sparse, v2_global=tuple(prefix_escape)), "G0 prefix row out of range")

    duplicate_record = next(
        (index for index in range(len(sparse)) if pieces[index] >= 2), None
    )
    if duplicate_record is None:
        FAILURES.append("no fixture record has two G0 rows to duplicate")
    else:
        duplicated = list(rows)
        begin = offsets[duplicate_record]
        duplicated[begin + 1] = duplicated[begin]
        rejects(corrupt(sparse, v2_global=tuple(duplicated)), "duplicate row in a record")

    over_bound = dec.V3_MAX_ACTIVE_ROWS + 1
    single_pieces = pieces[0]
    single_rows = tuple(rows[offsets[0] : offsets[0] + single_pieces]) + tuple(
        range(dec.V2_GLOBAL_DIMENSIONS, dec.V2_GLOBAL_DIMENSIONS + over_bound - single_pieces)
    )
    rejects(
        dec.SparseBatch(
            **{
                **{
                    field: getattr(sparse, field)
                    for field in sparse.__dataclass_fields__
                },
                "record_indices": (sparse.record_indices[0],),
                "physical_boards": (sparse.physical_boards[0],),
                "physical_position_keys": (sparse.physical_position_keys[0],),
                "legacy_model_input_keys": (sparse.legacy_model_input_keys[0],),
                "global_offsets": (0, len(single_rows)),
                "v2_global": single_rows,
                "physical_piece_count": (single_pieces,),
                "white_piece_count": (sparse.white_piece_count[0],),
                "side_to_move": (sparse.side_to_move[0],),
                "rule50_count": (sparse.rule50_count[0],),
                "game_ply": (sparse.game_ply[0],),
                "scores": (sparse.scores[0],),
                "best_moves": (sparse.best_moves[0],),
                "played_moves": (sparse.played_moves[0],),
                "results": (sparse.results[0],),
                "outcome_reasons": (sparse.outcome_reasons[0],),
                "royal_buckets": (sparse.royal_buckets[0],),
                "royal_mirrors": (sparse.royal_mirrors[0],),
            }
        ),
        f"record with {over_bound} active rows",
    )

    missing_g0 = list(pieces)
    missing_g0[0] = offsets[1] - offsets[0] + 1
    rejects(
        corrupt(sparse, physical_piece_count=tuple(missing_g0)),
        "record whose G0 rows are absent",
    )

    out_of_domain = list(sparse.white_piece_count)
    out_of_domain[0] = 37
    rejects(
        corrupt(sparse, white_piece_count=tuple(out_of_domain)),
        "white piece count outside the bucket domain",
    )

    truncated = corrupt(sparse, global_offsets=tuple(offsets[:-1]))
    rejects(truncated, "offsets that do not frame every record")
    shifted = list(offsets)
    shifted[-1] -= 1
    rejects(corrupt(sparse, global_offsets=tuple(shifted)), "offsets that leave a trailing row")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--count", type=int, default=192)
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)

    print("V3 training batch ABI invariants")
    records = build_records(args.records, args.count)
    sparse = test_round_trip(records)
    test_routing(records)
    test_rejections(sparse)

    if FAILURES:
        print(f"\nFAILED with {len(FAILURES)} problems:")
        for failure in FAILURES[:20]:
            print(f"  {failure}")
        return 1
    print("\nall V3 batch ABI invariants passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
