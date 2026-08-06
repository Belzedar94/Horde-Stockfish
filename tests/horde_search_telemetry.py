#!/usr/bin/env python3
"""Verify the opt-in Horde search telemetry build."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


PREFIX = "info string horde telemetry "
REQUIRED_COUNTERS = {
    "nodes",
    "legalMoves",
    "searchedMoves",
    "failHighs",
    "bestMoveSamples",
    "bestMoveRankSum",
    "nmpConsidered",
    "nmpTried",
    "nmpCutoffs",
    "nmpPawnOnlyBlocked",
    "probCutTried",
    "probCutMoves",
    "probCutCutoffs",
    "lmpTriggered",
    "lmrSearches",
    "lmrReductions",
    "lmrResearches",
    "pvResearches",
    "captureFutilityPrunes",
    "captureSeePrunes",
    "quietHistoryPrunes",
    "quietFutilityPrunes",
    "quietSeePrunes",
    "qNodes",
    "qStandPatCutoffs",
    "qMoveCountPrunes",
    "qFutilityPrunes",
    "qNonCapturePrunes",
    "qSeePrunes",
    "extinctionCapturesSeen",
    "extinctionCapturesSearched",
    "quietPawnPrunes",
    "quietPawnSkipCandidates",
    "fortressSamples",
    "fortressNanoseconds",
}


def read_until(process: subprocess.Popen[str], marker: str) -> list[str]:
    lines = []
    assert process.stdout is not None
    while True:
        line = process.stdout.readline()
        if line == "":
            raise RuntimeError(f"engine closed before {marker!r}:\n" + "".join(lines[-50:]))
        lines.append(line.rstrip("\r\n"))
        if marker in line:
            return lines


def run_search(engine: Path, enabled: bool) -> list[str]:
    process = subprocess.Popen(
        [str(engine)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdin is not None
    try:
        process.stdin.write("uci\n")
        process.stdin.flush()
        output = read_until(process, "uciok")
        if not any("option name HordeSearchTelemetry type check default false" in line for line in output):
            raise AssertionError("instrumented engine did not advertise HordeSearchTelemetry")

        process.stdin.write(
            f"setoption name HordeSearchTelemetry value {'true' if enabled else 'false'}\n"
            "setoption name Threads value 1\n"
            "setoption name Hash value 16\n"
            "isready\n"
        )
        process.stdin.flush()
        output.extend(read_until(process, "readyok"))
        process.stdin.write("position startpos\ngo depth 6\n")
        process.stdin.flush()
        output.extend(read_until(process, "bestmove "))
        process.stdin.write("quit\n")
        process.stdin.flush()
        if process.wait(timeout=30) != 0:
            raise RuntimeError("instrumented engine exited with a failure status")
        return output
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def fixed_result(lines: list[str]) -> tuple[str, int]:
    bestmove = next(line for line in reversed(lines) if line.startswith("bestmove "))
    depth_lines = [line for line in lines if line.startswith("info depth 6 ")]
    if not depth_lines:
        raise AssertionError("depth-six receipt was not emitted")
    match = re.search(r"\bnodes ([0-9]+)\b", depth_lines[-1])
    if not match:
        raise AssertionError("depth-six node receipt was not emitted")
    return bestmove, int(match.group(1))


def verify_enabled(lines: list[str]) -> None:
    telemetry = [line[len(PREFIX) :] for line in lines if line.startswith(PREFIX)]
    summaries = [line for line in telemetry if line.startswith("schema=1 ")]
    cells = [line for line in telemetry if line.startswith("side=")]
    if len(summaries) != 1 or not cells:
        raise AssertionError("telemetry summary or cells were not emitted exactly once")
    summary_nodes = re.search(r"\bnodes=([0-9]+)\b", summaries[0])
    if not summary_nodes or int(summary_nodes.group(1)) <= 0:
        raise AssertionError("telemetry summary contains no searched nodes")
    if not any("side=white" in line for line in cells):
        raise AssertionError("telemetry has no White-side search cells")
    if not any("side=black" in line for line in cells):
        raise AssertionError("telemetry has no Black-side search cells")

    keys = set()
    for token in " ".join(cells).split():
        if "=" in token:
            keys.add(token.split("=", 1)[0])
    missing = REQUIRED_COUNTERS - keys
    if missing:
        raise AssertionError(f"telemetry is missing counters: {sorted(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    args = parser.parse_args()
    engine = args.engine.resolve()
    if not engine.is_file():
        raise SystemExit(f"Engine not found: {engine}")

    disabled_first = run_search(engine, False)
    disabled_second = run_search(engine, False)
    if any(line.startswith(PREFIX) for line in disabled_first + disabled_second):
        raise AssertionError("runtime switch zero emitted Horde telemetry")
    if fixed_result(disabled_first) != fixed_result(disabled_second):
        raise AssertionError("runtime switch zero is not depth-deterministic")

    enabled = run_search(engine, True)
    verify_enabled(enabled)
    print("Horde search telemetry: opt-in, schema-complete, and switch-zero deterministic")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
