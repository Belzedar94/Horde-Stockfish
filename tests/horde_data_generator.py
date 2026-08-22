#!/usr/bin/env python3
"""Integration gate for the isolated HORDE_BIN_V1 self-play generator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Sequence


TIMEOUT_SECONDS = 120.0
REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = REPO_ROOT / "schemas" / "horde-bin-v1.schema.json"
LABEL_SCHEMA = REPO_ROOT / "schemas" / "horde-label-contract-v1.json"
CAPABILITY_JSON = (
    '{"schema":"HORDE_BIN_V1","schema_sha256":'
    '"B46ADE18AB8954A6AB232593484273E50C12B51550A938763A7A7D94DCCB63E4",'
    '"revision":"HORDE_BIN_V1_R2","revision_sha256":'
    '"013BF155072149A766B54A391ADBCB3EB1C539F49362EB06CA4E1530AE22B6A6",'
    '"expansion_families":["none","bestmove","promotion","check"],'
    '"label_contract":{"schema":"HORDE_LABEL_CONTRACT_V1","schema_sha256":'
    '"C299BA9ECD96DEF24363F8F62A8C67B88241AA860FB0735D4558B8EFEA0DCC22"},'
    '"write":true,"record_size":48,"header_size":2048}'
)
SCHEMA_SHA256 = "B46ADE18AB8954A6AB232593484273E50C12B51550A938763A7A7D94DCCB63E4"
R2_SCHEMA = REPO_ROOT / "schemas" / "horde-bin-v1-r2.schema.json"
R2_SCHEMA_SHA256 = "013BF155072149A766B54A391ADBCB3EB1C539F49362EB06CA4E1530AE22B6A6"
EXPANSION_RECORDS = 6
# Black to move with two distinct promotion pushes, b2-b1 and b2xa1. The second
# takes White's last piece, so its child is terminal and is skipped; the first
# leaves White a legal move, so its child survives. Nothing here can give check,
# because the Horde has no king.
EXPANSION_BOOK = """k7/8/8/8/8/p7/1p6/P7 b - - 0 1
1k6/8/8/8/8/p7/1p6/P7 b - - 0 1
2k5/8/8/8/8/p7/1p6/P7 b - - 0 1
3k4/8/8/8/8/p7/1p6/P7 b - - 0 1
"""
LABEL_CONTRACT_SHA256 = "C299BA9ECD96DEF24363F8F62A8C67B88241AA860FB0735D4558B8EFEA0DCC22"
RUN6B_SHA256 = "B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7"
TEST_SEED = "horde-bin-v1-wire-test"
TEST_RECORDS = 4
TEST_BOOK = """\
8/8/8/8/8/8/8/kQ6 b - - 0 1
8/8/8/8/8/8/8/6Qk b - - 0 1
kQ6/8/8/8/8/8/8/8 b - - 0 1
6Qk/8/8/8/8/8/8/8 b - - 0 1
"""
RAW_EVAL_RE = re.compile(r"horde-raw-eval (-?\d+) (-?\d+) (-?\d+)")


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise AssertionError(f"{label} does not exist: {resolved}")
    return resolved


def controlled_temp_parent() -> Path:
    configured = Path(tempfile.gettempdir()).resolve()
    if not any(character.isspace() for character in str(configured)):
        return configured
    fallback = Path(configured.anchor) / "horde-sf-test-tmp"
    fallback.mkdir(parents=True, exist_ok=True)
    if any(character.isspace() for character in str(fallback)):
        raise AssertionError(f"controlled test path contains whitespace: {fallback}")
    return fallback


def run_engine(
    binary: Path,
    commands: Sequence[str],
    *,
    expect_success: bool,
    cwd: Path = REPO_ROOT,
) -> str:
    result = subprocess.run(
        [str(binary)],
        input="\n".join((*commands, "")),
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"{binary.name} failed with exit code {result.returncode}:\n{result.stdout}"
        )
    if not expect_success and result.returncode == 0:
        raise AssertionError(f"{binary.name} accepted an invalid command:\n{result.stdout}")
    return result.stdout


def import_tool(module_name: str) -> Any:
    path = REPO_ROOT / "tools" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import tool from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def generator_command(
    *,
    output: Path,
    network: str,
    network_sha256: str,
    producer_sha256: str,
    book: Path,
    book_sha256: str,
    records: int = TEST_RECORDS,
    max_game_ply: int = 4,
    expansion: str = "",
) -> str:
    return (
        "horde_generate_training_data "
        f"threads 1 hash 16 network {network} network_sha256 {network_sha256} "
        f"producer_sha256 {producer_sha256} count {records} seed {TEST_SEED} "
        f"book {book.as_posix()} book_sha256 {book_sha256} out {output.as_posix()} "
        "depth 1 nodes 0 random_move_min_ply 0 random_move_max_ply 0 "
        "random_move_count 0 random_multi_pv 1 random_multi_pv_diff 0 "
        f"write_min_ply 0 write_max_ply 2 max_game_ply {max_game_ply} "
        f"{expansion}set_recommended_uci_options"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--normal-engine", type=Path, required=True)
    parser.add_argument("--network", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    generator = require_file(args.generator, "data-generator binary")
    normal_engine = require_file(args.normal_engine, "normal engine")
    network_path = require_file(args.network, "Run 6B network")
    network_sha = hashlib.sha256(network_path.read_bytes()).hexdigest().upper()
    if network_sha != RUN6B_SHA256:
        raise AssertionError(f"Run 6B SHA-256 mismatch: {network_sha}")
    producer_sha = hashlib.sha256(generator.read_bytes()).hexdigest().upper()
    observed_schema_sha = hashlib.sha256(SCHEMA.read_bytes()).hexdigest().upper()
    if observed_schema_sha != SCHEMA_SHA256:
        raise AssertionError(f"HORDE_BIN_V1 schema SHA-256 mismatch: {observed_schema_sha}")
    observed_r2_sha = hashlib.sha256(R2_SCHEMA.read_bytes()).hexdigest().upper()
    if observed_r2_sha != R2_SCHEMA_SHA256:
        raise AssertionError(f"HORDE_BIN_V1_R2 schema SHA-256 mismatch: {observed_r2_sha}")
    observed_label_sha = hashlib.sha256(LABEL_SCHEMA.read_bytes()).hexdigest().upper()
    if observed_label_sha != LABEL_CONTRACT_SHA256:
        raise AssertionError(
            f"Horde label-contract SHA-256 mismatch: {observed_label_sha}"
        )
    decoder = import_tool("horde_bin_v1")
    training_decoder = import_tool("horde_training_decoder")
    run6b = import_tool("horde_run6b")

    isolation = run_engine(normal_engine, ("horde_data_schema", "quit"), expect_success=True)
    if CAPABILITY_JSON in isolation:
        raise AssertionError("normal playing binary exposes the private DATAGEN capability")

    with tempfile.TemporaryDirectory(prefix="horde-bin-v1-", dir=controlled_temp_parent()) as raw:
        root = Path(raw).resolve()
        book = root / "openings.epd"
        book.write_text(TEST_BOOK, encoding="ascii", newline="\n")
        book_sha = hashlib.sha256(book.read_bytes()).hexdigest().upper()
        relative_network = Path("networks") / network_path.name

        first = root / "first.bin"
        second = root / "second.bin"
        first_command = generator_command(
            output=first,
            network=relative_network.as_posix(),
            network_sha256=network_sha,
            producer_sha256=producer_sha,
            book=book,
            book_sha256=book_sha,
        )
        output = run_engine(
            generator,
            ("horde_data_schema", first_command, "quit"),
            expect_success=True,
        )
        if output.splitlines().count(CAPABILITY_JSON) != 1:
            raise AssertionError(f"schema capability handshake mismatch:\n{output}")
        if output.splitlines().count("INFO: horde_generate_training_data finished.") != 1:
            raise AssertionError(f"completion marker mismatch:\n{output}")

        manifest, coverage = decoder.validate_file(first)
        if coverage["record_count"] != TEST_RECORDS:
            raise AssertionError(f"unexpected record count: {coverage}")
        if manifest["network"]["sha256"] != network_sha:
            raise AssertionError("manifest network identity mismatch")
        if manifest["producer_sha256"] != producer_sha:
            raise AssertionError("manifest producer identity mismatch")
        if manifest["book_sha256"] != book_sha:
            raise AssertionError("manifest book identity mismatch")
        if manifest["label_contract"] != {
            "schema": "HORDE_LABEL_CONTRACT_V1",
            "schema_sha256": LABEL_CONTRACT_SHA256,
        }:
            raise AssertionError("manifest label contract mismatch")
        if manifest["schema"] != "HORDE_BIN_V1" or manifest["schema_sha256"] != SCHEMA_SHA256:
            raise AssertionError("a run without expansion did not keep the V1 identity")
        if any(key.startswith("expand_") for key in manifest["generation"]):
            raise AssertionError("a run without expansion emitted expansion manifest keys")
        if coverage["expansion"] != {
            "ceiling": 0,
            "families": {"none": TEST_RECORDS},
            "children_per_parent": {"0": TEST_RECORDS},
        }:
            raise AssertionError(f"a run without expansion reported children: {coverage}")
        if coverage["outcome_reasons"] != {"extinction": TEST_RECORDS}:
            raise AssertionError(f"forced-extinction fixture was not preserved: {coverage}")
        if coverage["results"] != {"1": TEST_RECORDS}:
            raise AssertionError(f"result perspective mismatch: {coverage}")
        if coverage["sample_flags"].get("best_capture") != TEST_RECORDS:
            raise AssertionError(f"forced captures were not encoded: {coverage}")

        with training_decoder.HordeBinV1Dataset(first) as dataset:
            batches = list(dataset.batches(3))
            if len(dataset) != TEST_RECORDS or [len(batch) for batch in batches] != [3, 1]:
                raise AssertionError("trainer decoder batch framing mismatch")
            for batch in batches:
                legacy_counts = [
                    batch.legacy_piece_offsets[row + 1]
                    - batch.legacy_piece_offsets[row]
                    for row in range(len(batch))
                ]
                global_counts = [
                    batch.global_offsets[row + 1] - batch.global_offsets[row]
                    for row in range(len(batch))
                ]
                royal_counts = [
                    batch.royal_offsets[row + 1] - batch.royal_offsets[row]
                    for row in range(len(batch))
                ]
                if (
                    legacy_counts != [2] * len(batch)
                    or global_counts != [2] * len(batch)
                    or list(batch.physical_piece_count) != [2] * len(batch)
                    or royal_counts != [1] * len(batch)
                ):
                    raise AssertionError("trainer decoder sparse row counts changed")
                if any(index >= training_decoder.LEGACY_DIMENSIONS for index in batch.legacy_white):
                    raise AssertionError("legacy White feature escaped its table")
                if any(index >= training_decoder.LEGACY_DIMENSIONS for index in batch.legacy_black):
                    raise AssertionError("legacy Black feature escaped its table")
                if any(index >= training_decoder.V2_GLOBAL_DIMENSIONS for index in batch.v2_global):
                    raise AssertionError("V2 Global feature escaped its table")
                if any(index >= training_decoder.V2_ROYAL_DIMENSIONS for index in batch.v2_royal):
                    raise AssertionError("V2 Royal feature escaped its table")

            records = [dataset.record(index) for index in range(len(dataset))]
            fens = [training_decoder.training_record_fen(record) for record in records]
            engine_output = run_engine(
                normal_engine,
                (
                    "uci",
                    f"setoption name EvalFile value {network_path}",
                    "setoption name Threads value 1",
                    "isready",
                    *(
                        command
                        for fen in fens
                        for command in (f"position fen {fen}", "horde-raw-eval")
                    ),
                    "quit",
                ),
                expect_success=True,
            )
            engine_raw = [
                tuple(map(int, match.groups()))
                for line in engine_output.splitlines()
                if (match := RAW_EVAL_RE.fullmatch(line.strip()))
            ]
            if len(engine_raw) != len(records):
                raise AssertionError(f"wire replay returned the wrong number of evals:\n{engine_output}")
            replay = run6b.Run6BNetwork.load(network_path)
            replay_digest = hashlib.sha256()
            for record, expected in zip(records, engine_raw, strict=True):
                actual = replay.evaluate(record.features, record.side_to_move)
                observed = (actual.psqt, actual.positional, actual.total)
                if observed != expected:
                    raise AssertionError(
                        f"HORDE_BIN Run 6B replay mismatch at record {record.index}: "
                        f"trainer={observed}, engine={expected}, "
                        f"fen={training_decoder.training_record_fen(record)}"
                    )
                replay_digest.update(bytes.fromhex(record.source_payload_sha256))
                replay_digest.update(record.index.to_bytes(8, "little"))
                for value in observed:
                    replay_digest.update(value.to_bytes(4, "little", signed=True))
            if replay_digest.digest() == bytes(32):
                raise AssertionError("HORDE_BIN Run 6B replay digest is empty")

        sparse_receipt = training_decoder.dataset_receipt(first, batch_size=3)
        single_row_receipt = training_decoder.dataset_receipt(first, batch_size=1)
        if sparse_receipt["schema"] != "HORDE_TRAINING_DECODER_V2":
            raise AssertionError(f"trainer decoder ABI schema mismatch: {sparse_receipt}")
        if sparse_receipt["record_count"] != TEST_RECORDS:
            raise AssertionError(f"trainer decoder record count mismatch: {sparse_receipt}")
        if sparse_receipt["piece_rows"] != 2 * TEST_RECORDS:
            raise AssertionError(f"trainer decoder piece rows mismatch: {sparse_receipt}")
        if sparse_receipt["royal_rows"] != TEST_RECORDS:
            raise AssertionError(f"trainer decoder Royal rows mismatch: {sparse_receipt}")
        if sparse_receipt["row_counts"] != {
            "legacy_physical": 2 * TEST_RECORDS,
            "global_g0": 2 * TEST_RECORDS,
            "global_contextual": {},
            "global_contextual_total": 0,
            "global_complete": 2 * TEST_RECORDS,
            "royal": TEST_RECORDS,
        }:
            raise AssertionError(f"trainer decoder stream counts mismatch: {sparse_receipt}")
        if sparse_receipt["sparse_sha256"] != single_row_receipt["sparse_sha256"]:
            raise AssertionError("trainer decoder receipt depends on batch partitioning")
        if (
            sparse_receipt["feature_stream_sha256"]
            != single_row_receipt["feature_stream_sha256"]
        ):
            raise AssertionError("trainer feature-stream hashes depend on batch partitioning")

        tampered = root / "tampered.bin"
        tampered_payload = bytearray(first.read_bytes())
        tampered_payload[-1] ^= 1
        tampered.write_bytes(tampered_payload)
        try:
            training_decoder.HordeBinV1Dataset(tampered)
        except decoder.FormatError:
            pass
        else:
            raise AssertionError("trainer decoder accepted a tampered payload")

        unbound = root / "unbound-labels.bin"
        unbound_payload = bytearray(first.read_bytes())
        label_offset = unbound_payload.find(LABEL_CONTRACT_SHA256.encode("ascii"))
        if label_offset < 0:
            raise AssertionError("generated manifest does not contain the label-contract hash")
        unbound_payload[label_offset : label_offset + 64] = b"0" * 64
        unbound.write_bytes(unbound_payload)
        try:
            training_decoder.HordeBinV1Dataset(unbound)
        except decoder.FormatError:
            pass
        else:
            raise AssertionError("trainer decoder accepted an unbound label contract")

        second_command = generator_command(
            output=second,
            network=relative_network.as_posix(),
            network_sha256=network_sha,
            producer_sha256=producer_sha,
            book=book,
            book_sha256=book_sha,
        )
        run_engine(generator, (second_command, "quit"), expect_success=True)
        if second.read_bytes() != first.read_bytes():
            raise AssertionError("Threads=1 fixed-seed HORDE_BIN_V1 output is not reproducible")

        # HORDE_BIN_V1_R2 tactical expansion, end to end.
        expansion_book = root / "expansion.epd"
        expansion_book.write_text(EXPANSION_BOOK, encoding="ascii", newline="\n")
        expansion_book_sha = hashlib.sha256(expansion_book.read_bytes()).hexdigest().upper()
        expanded = root / "expanded.bin"
        run_engine(
            generator,
            (
                generator_command(
                    output=expanded,
                    network=relative_network.as_posix(),
                    network_sha256=network_sha,
                    producer_sha256=producer_sha,
                    book=expansion_book,
                    book_sha256=expansion_book_sha,
                    records=EXPANSION_RECORDS,
                    max_game_ply=8,
                    expansion="expand_promo 2 expand_check 2 expand_max_children 5 ",
                ),
                "quit",
            ),
            expect_success=True,
        )

        expanded_manifest, expanded_coverage = decoder.validate_file(expanded)
        if (
            expanded_manifest["schema"] != "HORDE_BIN_V1_R2"
            or expanded_manifest["schema_sha256"] != R2_SCHEMA_SHA256
        ):
            raise AssertionError("an expanded run did not carry the R2 identity")
        generation = expanded_manifest["generation"]
        if [generation.get(key) for key in decoder.EXPANSION_GENERATION_KEYS] != [2, 2, 5]:
            raise AssertionError(f"expansion caps were not declared in the manifest: {generation}")
        if list(generation)[-3:] != decoder.EXPANSION_GENERATION_KEYS:
            raise AssertionError("expansion keys are not appended to the generation object")
        if decoder.manifest_expansion_ceiling(expanded_manifest) != 4:
            raise AssertionError("the declared ceiling is not min(promo + check, max_children)")

        families = expanded_coverage["expansion"]["families"]
        if families.get("promotion", 0) < 1:
            raise AssertionError(f"expansion produced no promotion child: {expanded_coverage}")
        if families.get("bestmove") or families.get("check"):
            raise AssertionError(
                f"expansion produced a family the command did not ask for: {expanded_coverage}"
            )
        if families.get("none", 0) < 1:
            raise AssertionError(f"expansion produced no parents: {expanded_coverage}")
        if sum(families.values()) != EXPANSION_RECORDS:
            raise AssertionError(f"family counts do not cover the payload: {expanded_coverage}")

        # Every child carries the flag and the family together, and follows a
        # parent rather than opening the payload.
        expanded_payload = expanded.read_bytes()
        child_index = None
        for index, raw in enumerate(decoder.iter_records(expanded_payload)):
            record = decoder.validate_record(raw, index)
            is_child = bool(record["flags"] & 0x80)
            if is_child != (record["family"] != 0):
                raise AssertionError(f"record {index} broke the child/family cross-check")
            if is_child and child_index is None:
                if index == 0:
                    raise AssertionError("the payload opens with an expansion child")
                child_index = index
        if child_index is None:
            raise AssertionError("no expansion child reached the payload")

        with training_decoder.HordeBinV1Dataset(expanded) as dataset:
            families_seen = {dataset.record(i).expansion_family for i in range(len(dataset))}
            if 2 not in families_seen or not families_seen <= {0, 2}:
                raise AssertionError(
                    f"the trainer decoder did not expose the expansion family: {families_seen}"
                )

        # Breaking either half of the marking must fail closed, because the flag
        # and the family are cross-checked against each other.
        record_start = decoder.HEADER_SIZE + child_index * decoder.RECORD_SIZE
        for offset, mask in ((35, 0x80), (47, 0x38)):
            broken = bytearray(expanded_payload)
            broken[record_start + offset] ^= mask
            try:
                decoder.validate_record(
                    bytes(broken[record_start : record_start + decoder.RECORD_SIZE]),
                    child_index,
                )
            except decoder.FormatError:
                continue
            raise AssertionError(f"a record with a broken byte {offset} cross-check was accepted")

        inconsistent = root / "inconsistent.bin"
        failed = run_engine(
            generator,
            (
                generator_command(
                    output=inconsistent,
                    network=relative_network.as_posix(),
                    network_sha256=network_sha,
                    producer_sha256=producer_sha,
                    book=expansion_book,
                    book_sha256=expansion_book_sha,
                    expansion="expand_promo 2 ",
                ),
                "quit",
            ),
            expect_success=False,
        )
        if "expand_max_children" not in failed or inconsistent.exists():
            raise AssertionError(f"an inconsistent expansion budget was accepted:\n{failed}")

        bad = root / "bad.bin"
        bad_command = generator_command(
            output=bad,
            network=relative_network.as_posix(),
            network_sha256="0" * 64,
            producer_sha256=producer_sha,
            book=book,
            book_sha256=book_sha,
        )
        failed = run_engine(generator, (bad_command, "quit"), expect_success=False)
        if "registered Run 6B" not in failed or bad.exists():
            raise AssertionError(f"wrong-network failure was not fail-closed:\n{failed}")

        existing = root / "existing.bin"
        existing.write_bytes(b"do-not-overwrite")
        existing_command = generator_command(
            output=existing,
            network=relative_network.as_posix(),
            network_sha256=network_sha,
            producer_sha256=producer_sha,
            book=book,
            book_sha256=book_sha,
        )
        failed = run_engine(generator, (existing_command, "quit"), expect_success=False)
        if "already exists" not in failed or existing.read_bytes() != b"do-not-overwrite":
            raise AssertionError(f"exclusive-output failure was not preserved:\n{failed}")

    print("HORDE_BIN_V1 generator integration passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
