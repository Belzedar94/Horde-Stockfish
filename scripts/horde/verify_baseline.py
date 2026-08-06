#!/usr/bin/env python3
"""Verify the frozen HordeTest baseline and optionally probe Fairy-Stockfish."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "docs" / "horde" / "baseline-manifest.json"
NODES_RE = re.compile(r"Nodes searched\s*:\s*([0-9]+)", re.IGNORECASE)


class VerificationError(RuntimeError):
    """Raised when a frozen input or runtime contract does not match."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_local_file(entry: dict[str, object], label: str) -> Path:
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative:
        raise VerificationError(f"{label}: missing repository-relative path")
    path = REPO_ROOT / relative
    if not path.is_file():
        raise VerificationError(f"{label}: file not found: {path}")

    expected_size = entry.get("size")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise VerificationError(
            f"{label}: size {actual_size}, expected {expected_size}: {path}"
        )

    expected_hash = entry.get("sha256")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise VerificationError(
            f"{label}: SHA-256 {actual_hash}, expected {expected_hash}: {path}"
        )

    print(f"ok  {label}: {relative} ({actual_size} bytes, {actual_hash})")
    return path


@dataclass(frozen=True)
class PerftPosition:
    identifier: str
    epd: str
    counts: dict[int, int]


def parse_perft(path: Path) -> list[PerftPosition]:
    positions: list[PerftPosition] = []
    identifier: str | None = None
    epd: str | None = None
    counts: dict[int, int] = {}

    def finish() -> None:
        nonlocal identifier, epd, counts
        if identifier is None and epd is None and not counts:
            return
        if not identifier or not epd or not counts:
            raise VerificationError(f"incomplete perft record in {path}")
        positions.append(PerftPosition(identifier, epd, dict(counts)))
        identifier, epd, counts = None, None, {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            finish()
        elif line.startswith("id "):
            if identifier is not None:
                raise VerificationError(f"duplicate id before record end in {path}")
            identifier = line[3:].strip()
        elif line.startswith("epd "):
            epd = line[4:].strip()
        elif line.startswith("perft "):
            fields = line.split()
            if len(fields) != 3:
                raise VerificationError(f"invalid perft line in {path}: {line}")
            depth, nodes = int(fields[1]), int(fields[2])
            if depth in counts:
                raise VerificationError(f"duplicate perft depth {depth} in {path}")
            counts[depth] = nodes
        else:
            raise VerificationError(f"unknown fixture line in {path}: {line}")
    finish()
    return positions


def horde_to_hordetest(epd: str) -> str:
    fields = epd.split()
    if len(fields) != 4:
        raise VerificationError(f"expected four-field EPD, got: {epd}")
    fields[0] = fields[0].replace("P", "H")
    return " ".join(fields)


def verify_fixture_equivalence(
    standard_path: Path, hordetest_path: Path
) -> list[PerftPosition]:
    standard = parse_perft(standard_path)
    hordetest = parse_perft(hordetest_path)
    if len(standard) != len(hordetest):
        raise VerificationError("standard and HordeTest perft record counts differ")

    for original, encoded in zip(standard, hordetest, strict=True):
        if original.identifier != encoded.identifier:
            raise VerificationError(
                f"perft id mismatch: {original.identifier} != {encoded.identifier}"
            )
        if original.counts != encoded.counts:
            raise VerificationError(
                f"perft counts differ for {original.identifier}: "
                f"{original.counts} != {encoded.counts}"
            )
        expected_epd = horde_to_hordetest(original.epd)
        if encoded.epd != expected_epd:
            raise VerificationError(
                f"HordeTest EPD is not the exact P-to-H mapping for "
                f"{original.identifier}: {encoded.epd} != {expected_epd}"
            )

    print(f"ok  fixture semantics: {len(standard)} P-to-H perft records match")
    return hordetest


class UciProcess:
    def __init__(
        self, executable: Path, arguments: list[str], cwd: Path, timeout: float
    ) -> None:
        self.timeout = timeout
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.transcript: list[str] = []
        self.process = subprocess.Popen(
            [str(executable), *arguments],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def _read_output(self) -> None:
        assert self.process.stdout is not None
        for raw_line in self.process.stdout:
            line = raw_line.rstrip("\r\n")
            self.transcript.append(line)
            self.lines.put(line)
        self.lines.put(None)

    def send(self, command: str) -> None:
        if self.process.poll() is not None:
            raise VerificationError(
                f"engine exited with code {self.process.returncode} before {command!r}"
            )
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def wait_for(
        self, description: str, predicate: Callable[[str], bool]
    ) -> str:
        deadline = time.monotonic() + self.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                tail = "\n".join(self.transcript[-30:])
                raise VerificationError(
                    f"timed out waiting for {description}; output tail:\n{tail}"
                )
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty as exc:
                raise VerificationError(f"timed out waiting for {description}") from exc
            if line is None:
                tail = "\n".join(self.transcript[-30:])
                raise VerificationError(
                    f"engine exited while waiting for {description}; output tail:\n{tail}"
                )
            if predicate(line):
                return line

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.send("quit")
                self.process.wait(timeout=5)
            except (VerificationError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=5)


def full_fen(epd: str) -> str:
    return f"{epd} 0 1"


def probe_engine(
    executable: Path,
    network_path: Path,
    variants_path: Path,
    positions: list[PerftPosition],
    depth: int,
    timeout: float,
) -> None:
    executable = executable.expanduser().resolve()
    if not executable.is_file():
        raise VerificationError(f"engine not found: {executable}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    oracle_hash = manifest["engine"]["historical_oracle_binary"]["sha256"]
    supplied_hash = sha256_file(executable)
    if supplied_hash == oracle_hash:
        raise VerificationError(
            "the supplied executable is the historical oracle binary; "
            "it cannot satisfy the formal baseline gate"
        )

    receipt = manifest["engine"]["formal_build_receipt"]
    expected_binary = receipt["binary"]
    if executable.stat().st_size != expected_binary["size"]:
        raise VerificationError(
            f"formal baseline binary size {executable.stat().st_size}, "
            f"expected {expected_binary['size']}"
        )
    if supplied_hash != expected_binary["sha256"]:
        raise VerificationError(
            f"formal baseline SHA-256 {supplied_hash}, "
            f"expected {expected_binary['sha256']}"
        )

    print(f"info engine: {executable} ({supplied_hash})")
    print(f"ok  formal binary receipt: {receipt['id']}")

    with tempfile.TemporaryDirectory(prefix="horde-baseline-") as temporary:
        runtime_dir = Path(temporary)
        shutil.copyfile(variants_path, runtime_dir / "variants.ini")
        uci = UciProcess(executable, [], runtime_dir, timeout)
        try:
            uci.send("uci")
            uci.wait_for("uciok", lambda line: line.strip() == "uciok")
            uci.send(f"setoption name VariantPath value {(runtime_dir / 'variants.ini').resolve()}")
            uci.send("uci")
            uci.wait_for("uciok after VariantPath", lambda line: line.strip() == "uciok")
            if not any(
                line.startswith("option name UCI_Variant ")
                and " var hordetest" in line
                for line in uci.transcript
            ):
                raise VerificationError(
                    "engine did not advertise hordetest in the UCI_Variant option"
                )
            uci.send("setoption name UCI_Variant value hordetest")
            uci.send(f"setoption name EvalFile value {network_path.resolve()}")
            uci.send("setoption name Threads value 1")
            uci.send("setoption name Hash value 16")
            uci.send("isready")
            uci.wait_for("readyok", lambda line: line.strip() == "readyok")

            for position in positions:
                uci.send(f"position fen {full_fen(position.epd)}")
                for perft_depth, expected_nodes in sorted(position.counts.items()):
                    uci.send(f"go perft {perft_depth}")
                    line = uci.wait_for(
                        f"perft {position.identifier} depth {perft_depth}",
                        lambda output: NODES_RE.search(output) is not None,
                    )
                    match = NODES_RE.search(line)
                    assert match is not None
                    actual_nodes = int(match.group(1))
                    if actual_nodes != expected_nodes:
                        raise VerificationError(
                            f"{position.identifier} depth {perft_depth}: "
                            f"{actual_nodes} nodes, expected {expected_nodes}"
                        )
                    print(
                        f"ok  perft {position.identifier} d{perft_depth}: "
                        f"{actual_nodes}"
                    )

            uci.send("ucinewgame")
            uci.send("isready")
            uci.wait_for("readyok after ucinewgame", lambda line: line.strip() == "readyok")
            uci.send(f"position fen {full_fen(positions[0].epd)}")
            uci.send(f"go depth {depth}")
            bestmove = uci.wait_for(
                f"bestmove at depth {depth}", lambda line: line.startswith("bestmove ")
            )
            move = bestmove.split(maxsplit=2)[1]
            if move in {"(none)", "0000"}:
                raise VerificationError(f"engine returned no legal bestmove: {bestmove}")
            nnue_lines = [
                line
                for line in uci.transcript
                if "NNUE evaluation using" in line and " enabled" in line
            ]
            if not nnue_lines:
                raise VerificationError(
                    "engine searched but did not confirm that NNUE evaluation is enabled"
                )
            if network_path.name not in nnue_lines[-1]:
                raise VerificationError(
                    "engine enabled a different NNUE file: " + nnue_lines[-1]
                )

            if depth == 8:
                expected_search = receipt["depth_8_receipt"]
                info_lines = [
                    line for line in uci.transcript if line.startswith("info depth 8 ")
                ]
                if not info_lines:
                    raise VerificationError("engine emitted no final depth-8 info line")
                final_info = info_lines[-1]
                nodes_match = re.search(r"\bnodes (\d+)\b", final_info)
                if not nodes_match or int(nodes_match.group(1)) != expected_search["nodes"]:
                    raise VerificationError(
                        f"depth-8 node receipt mismatch: {final_info}"
                    )
                if move != expected_search["bestmove"]:
                    raise VerificationError(
                        f"depth-8 bestmove {move}, expected {expected_search['bestmove']}"
                    )
            print(f"ok  NNUE load: {nnue_lines[-1]}")
            print(f"ok  search depth {depth}: {bestmove}")
        finally:
            uci.close()


def load_and_verify_manifest() -> tuple[dict[str, object], dict[str, Path]]:
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read manifest {MANIFEST_PATH}: {exc}") from exc

    if manifest.get("status") != "frozen":
        raise VerificationError("manifest status must be 'frozen'")
    if manifest.get("baseline_id") != "horde-fsf-c19-run6b-e37-l06":
        raise VerificationError("unexpected baseline_id")

    files: dict[str, Path] = {}
    files["network"] = verify_local_file(manifest["network"], "network")
    files["license"] = verify_local_file(
        manifest["network"]["license_notice"], "network license notice"
    )
    files["oracle_patch"] = verify_local_file(
        manifest["engine"]["diagnostic_raw_oracle"]["patch"],
        "diagnostic raw-oracle patch",
    )

    fixtures = manifest["rule_profile"]["local_fixtures"]
    for fixture in fixtures:
        path = verify_local_file(fixture, f"fixture {fixture['path']}")
        files[Path(fixture["path"]).name] = path

    return manifest, files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify frozen HordeTest artifacts and optionally probe an engine."
    )
    parser.add_argument(
        "--engine",
        type=Path,
        help="formally built Fairy-Stockfish executable to probe",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="depth for the final deterministic search probe (default: 8)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="seconds to wait for each UCI response (default: 30)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest, files = load_and_verify_manifest()
        positions = verify_fixture_equivalence(
            files["lichess-horde.perft"], files["hordetest.perft"]
        )
        formal_commit = manifest["engine"]["formal_source"]["commit"]
        print(f"info formal source pin: {formal_commit}")

        if args.engine is not None:
            probe_engine(
                args.engine,
                files["network"],
                files["variants.ini"],
                positions,
                args.depth,
                args.timeout,
            )
        else:
            print("ok  artifact verification complete (engine probe not requested)")
        return 0
    except (VerificationError, KeyError, TypeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
