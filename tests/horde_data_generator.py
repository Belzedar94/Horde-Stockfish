#!/usr/bin/env python3
"""Integration gate for the isolated HORDE_BIN_V1 self-play generator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence


TIMEOUT_SECONDS = 120.0
REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = REPO_ROOT / "schemas" / "horde-bin-v1.schema.json"
CAPABILITY_JSON = (
    '{"schema":"HORDE_BIN_V1","schema_sha256":'
    '"B46ADE18AB8954A6AB232593484273E50C12B51550A938763A7A7D94DCCB63E4",'
    '"write":true,"record_size":48,"header_size":2048}'
)
SCHEMA_SHA256 = "B46ADE18AB8954A6AB232593484273E50C12B51550A938763A7A7D94DCCB63E4"
RUN6B_SHA256 = "B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7"
TEST_SEED = "horde-bin-v1-wire-test"
TEST_RECORDS = 4
TEST_BOOK = """\
8/8/8/8/8/8/8/kQ6 b - - 0 1
8/8/8/8/8/8/8/6Qk b - - 0 1
kQ6/8/8/8/8/8/8/8 b - - 0 1
6Qk/8/8/8/8/8/8/8 b - - 0 1
"""


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


def import_decoder() -> Any:
    path = REPO_ROOT / "tools" / "horde_bin_v1.py"
    spec = importlib.util.spec_from_file_location("horde_bin_v1", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import decoder from {path}")
    module = importlib.util.module_from_spec(spec)
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
) -> str:
    return (
        "horde_generate_training_data "
        f"threads 1 hash 16 network {network} network_sha256 {network_sha256} "
        f"producer_sha256 {producer_sha256} count {TEST_RECORDS} seed {TEST_SEED} "
        f"book {book.as_posix()} book_sha256 {book_sha256} out {output.as_posix()} "
        "depth 1 nodes 0 random_move_min_ply 0 random_move_max_ply 0 "
        "random_move_count 0 random_multi_pv 1 random_multi_pv_diff 0 "
        "write_min_ply 0 write_max_ply 2 max_game_ply 4 set_recommended_uci_options"
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
    decoder = import_decoder()

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
        if coverage["outcome_reasons"] != {"extinction": TEST_RECORDS}:
            raise AssertionError(f"forced-extinction fixture was not preserved: {coverage}")
        if coverage["results"] != {"1": TEST_RECORDS}:
            raise AssertionError(f"result perspective mismatch: {coverage}")
        if coverage["sample_flags"].get("best_capture") != TEST_RECORDS:
            raise AssertionError(f"forced captures were not encoded: {coverage}")

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
