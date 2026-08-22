#!/usr/bin/env python3
"""Batch materialisation must not depend on who materialised it.

The trainer prefetches batches in worker processes. The batch schedule, the
schedule digest and the sample order chain are computed from the index tuples
before anything is materialised, so they cannot move by construction; what
could move is the batch content itself. This pins that it does not, on the
pool path rather than the serial fallback.

The campaign oracle runs the same comparison over 200 real steps of the real
corpus. This is the same property at a second's cost.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import horde_bin_v1 as wire  # noqa: E402
import horde_training_chunk_set as chunk_set  # noqa: E402
import horde_training_control as ctl  # noqa: E402
from horde_training_chunk_set import HordeChunkSetDataset  # noqa: E402


TRAIN_BOOK = "A" * 64
VALIDATION_BOOK = "B" * 64
SOURCE_COMMIT = "1" * 40
PRODUCER_SHA256 = "2" * 64
COMMON = {
    "hash_mb": 16, "depth": 4, "nodes": 0,
    "random_move_min_ply": 1, "random_move_max_ply": 1, "random_move_count": 0,
    "random_multi_pv": 0, "random_multi_pv_diff": 0,
    "write_min_ply": 0, "write_max_ply": 1, "max_game_ply": 2,
}
RECORDS_PER_CHUNK = 64
CHUNKS = 4


def _record(identity: int) -> bytes:
    board = [0] * 64
    side = identity & 1
    result = (-1, 0, 1)[(identity // 2) % 3]
    board[0] = 2
    board[8 + identity % 40] = 4
    board[48 + (identity // 40) % 8] = 9
    board[57] = 7
    board[60] = 11
    packed = bytes(board[s] | (board[s + 1] << 4) for s in range(0, 64, 2))
    move = (0 << 6) | 7 if side == 0 else (57 << 6) | 56
    score = result * 200 + (identity * 137) % 1201 - 600
    reason = 3 if result == 0 else 1
    raw = packed + bytes((side, 0, 64, 0)) + struct.pack(
        "<HHhHHbB", identity % 100, side, score, move, move, result, reason)
    wire.validate_record(raw, identity)
    return raw


def _write_contract(path: Path) -> None:
    total = RECORDS_PER_CHUNK * CHUNKS
    contract = {
        "schema_name": chunk_set.SCALE_SCHEMA,
        "dependencies": {
            "dataset": {"schema": wire.SCHEMA_NAME,
                        "schema_sha256": wire.SCHEMA_SHA256},
            "teacher": {"source_commit": SOURCE_COMMIT,
                        "producer_sha256": PRODUCER_SHA256,
                        "network_schema": "HORDETEST_HP_LEGACY_V1",
                        "network_sha256": wire.RUN6B_SHA256},
            "labels": {"schema": wire.LABEL_CONTRACT_NAME,
                       "schema_sha256": wire.LABEL_CONTRACT_SHA256},
        },
        "openbench": {"campaign_id": "fixture", "cohort": "fixture"},
        "books": {"training": {"records": 3, "raw_sha256": TRAIN_BOOK},
                  "validation": {"records": 2, "raw_sha256": VALIDATION_BOOK}},
        "generation": {
            "common": COMMON,
            "training": {"records": total, "positions_per_chunk": RECORDS_PER_CHUNK,
                         "chunk_count": CHUNKS, "base_seed": 1000},
            "validation_candidate": {"records": RECORDS_PER_CHUNK,
                                     "positions_per_chunk": RECORDS_PER_CHUNK,
                                     "chunk_count": 1, "base_seed": 2000},
        },
    }
    path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_chunk(path: Path, records: list[bytes], seed: int) -> None:
    payload = b"".join(records)
    manifest = {
        "schema": wire.SCHEMA_NAME, "schema_sha256": wire.SCHEMA_SHA256,
        "format_version": wire.FORMAT_VERSION, "header_bytes": wire.HEADER_SIZE,
        "record_bytes": wire.RECORD_SIZE, "record_count": len(records),
        "byte_order": "little", "source_commit": SOURCE_COMMIT,
        "source_dirty": False,
        "network": {"schema": "HORDETEST_HP_LEGACY_V1", "sha256": wire.RUN6B_SHA256},
        "book_sha256": TRAIN_BOOK, "producer_sha256": PRODUCER_SHA256,
        "payload_sha256": hashlib.sha256(payload).hexdigest().upper(),
        "label_contract": {"schema": wire.LABEL_CONTRACT_NAME,
                           "schema_sha256": wire.LABEL_CONTRACT_SHA256},
        "generation": {"requested_records": len(records), "seed": str(seed),
                       "threads": 1, **COMMON, "opening_count": 3},
    }
    encoded = json.dumps(manifest, separators=(",", ":"), ensure_ascii=True,
                         allow_nan=False).encode("utf-8")
    header = wire.MAGIC + struct.pack(
        "<HHI", wire.FORMAT_VERSION, wire.HEADER_SIZE, len(encoded)) + encoded
    header += bytes(wire.HEADER_SIZE - len(header))
    path.write_bytes(header + payload)


def _digest(batch) -> bytes:
    return hashlib.sha256(repr(batch).encode("utf-8")).digest()


def test_prefetched_batches_match_the_serial_path() -> None:
    total = RECORDS_PER_CHUNK * CHUNKS
    batch_size, block_size, seed = 8, 16, 987654321
    with tempfile.TemporaryDirectory(prefix="horde-prefetch-") as directory:
        root = Path(directory)
        contract = root / "campaign.json"
        _write_contract(contract)
        chunks = []
        for index in range(CHUNKS):
            path = root / f"chunk_{index}.bin"
            _write_chunk(
                path,
                [_record(index * RECORDS_PER_CHUNK + i)
                 for i in range(RECORDS_PER_CHUNK)],
                1000 + index,
            )
            chunks.append(path)
        receipt = root / "chunk-set.json"
        chunk_set.assemble_chunk_set(contract, "training", receipt, chunks)

        tuples = list(ctl.epoch_batches(total, batch_size, block_size, seed, 0))
        architecture = ctl.LEGACY_ARCHITECTURE

        with HordeChunkSetDataset(receipt, contract) as dataset:
            serial = [
                ctl._load_sparse_batch(architecture, dataset, indices)
                for indices in tuples
            ]
            prefetch = ctl._BatchPrefetcher(
                architecture, dataset, total, batch_size, block_size, seed, 0, 0,
                workers=3,
            )
            if not prefetch.parallel:
                raise AssertionError("the pool did not engage; nothing was tested")
            try:
                parallel = [prefetch.batch(indices) for indices in tuples]
            finally:
                prefetch.close()

        if len(serial) != len(tuples) or len(parallel) != len(tuples):
            raise AssertionError("batch counts drifted")
        for step, (left, right) in enumerate(zip(serial, parallel)):
            if left != right:
                raise AssertionError(f"batch {step} differs between the paths")
        left_chain = hashlib.sha256()
        right_chain = hashlib.sha256()
        for left, right in zip(serial, parallel):
            left_chain.update(_digest(left))
            right_chain.update(_digest(right))
        if left_chain.digest() != right_chain.digest():
            raise AssertionError("the batch stream digest differs between the paths")


def test_prefetch_refuses_a_schedule_it_was_not_given() -> None:
    """The delivered batch is checked against the tuple the loop is on."""
    total = RECORDS_PER_CHUNK * CHUNKS
    batch_size, block_size, seed = 8, 16, 987654321
    with tempfile.TemporaryDirectory(prefix="horde-prefetch-drift-") as directory:
        root = Path(directory)
        contract = root / "campaign.json"
        _write_contract(contract)
        chunks = []
        for index in range(CHUNKS):
            path = root / f"chunk_{index}.bin"
            _write_chunk(
                path,
                [_record(index * RECORDS_PER_CHUNK + i)
                 for i in range(RECORDS_PER_CHUNK)],
                1000 + index,
            )
            chunks.append(path)
        receipt = root / "chunk-set.json"
        chunk_set.assemble_chunk_set(contract, "training", receipt, chunks)

        with HordeChunkSetDataset(receipt, contract) as dataset:
            prefetch = ctl._BatchPrefetcher(
                ctl.LEGACY_ARCHITECTURE, dataset, total, batch_size, block_size,
                seed, 0, 0, workers=3,
            )
            try:
                wrong = tuple(range(batch_size))
                try:
                    prefetch.batch(wrong)
                except ctl.TrainingError as error:
                    if "does not match the scheduled indices" not in str(error):
                        raise AssertionError(f"unexpected refusal: {error}")
                else:
                    raise AssertionError("the prefetcher accepted a foreign schedule")
            finally:
                prefetch.close()


def main() -> int:
    test_prefetched_batches_match_the_serial_path()
    test_prefetch_refuses_a_schedule_it_was_not_given()
    print("Horde batch prefetch: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
