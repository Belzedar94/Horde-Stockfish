#!/usr/bin/env python3
"""Build fixtures and verify the fail-closed Horde V2 engine dispatch."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from horde_v2_container_parity import (  # noqa: E402
    FIXTURES,
    deterministic_provenance,
    deterministic_sections,
    expected_trace,
)

sys.path.insert(0, str(REPO_ROOT / "tools"))

from horde_v2_container import SPECS, build_container, parse_container  # noqa: E402


TRACE_RE = re.compile(
    r"^horde-v2-candidate-eval schema=(\S+) file_sha256=([0-9A-F]{64}) "
    r"parameter_sha256=([0-9A-F]{64}) output_affine=(-?\d+) "
    r"pre_rule50=(-?\d+) value=(-?\d+)$"
)
RAW_RE = re.compile(r"^horde-raw-eval (-?\d+) (-?\d+) (-?\d+)$")
FINAL_RE = re.compile(r"^horde-eval-debug eval=(-?\d+)$")
NODES_RE = re.compile(r"^Nodes searched\s*:\s*(\d+)$")

# The low-level parity oracle intentionally includes one board where the side
# not to move can capture the royal piece. Position::set correctly rejects that
# unreachable state, so the engine-facing gate uses only reachable fixtures.
ENGINE_FIXTURES = tuple(fixture for fixture in FIXTURES if fixture.name != "mirror-e4")


def selected_spec(architecture: str):
    for spec in SPECS:
        if spec.architecture == architecture:
            return spec
    choices = ", ".join(spec.architecture for spec in SPECS)
    raise ValueError(f"unknown architecture {architecture!r}; choose one of: {choices}")


def create_fixture(output: Path, architecture: str) -> None:
    spec = selected_spec(architecture)
    payload, receipt = build_container(
        spec,
        deterministic_sections(spec),
        deterministic_provenance(spec),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    parsed = parse_container(payload)
    if parsed.file_sha256 != receipt["file_sha256"]:
        raise AssertionError("fixture receipt does not authenticate the emitted container")
    print(
        f"Horde V2 fixture created: architecture={architecture}, "
        f"bytes={len(payload)}, sha256={parsed.file_sha256}"
    )


def compress_rank(rank: str) -> str:
    fields: list[str] = []
    empty = 0
    for piece in rank:
        if piece == ".":
            empty += 1
        else:
            if empty:
                fields.append(str(empty))
                empty = 0
            fields.append(piece)
    if empty:
        fields.append(str(empty))
    return "".join(fields)


def fixture_fen(board: str, side_to_move: int, rule50: int) -> str:
    if len(board) != 64:
        raise AssertionError("fixture board must contain exactly 64 squares")
    ranks = [board[offset : offset + 8] for offset in range(0, 64, 8)]
    placement = "/".join(compress_rank(rank) for rank in reversed(ranks))
    side = "w" if side_to_move == 0 else "b"
    return f"{placement} {side} - - {rule50} 1"


def run_engine(engine: Path, commands: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(engine)],
        cwd=engine.parent,
        input="\n".join([*commands, "quit", ""]),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def require_success(completed: subprocess.CompletedProcess[str], label: str) -> list[str]:
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout[-8000:]}\n"
            f"stderr:\n{completed.stderr[-8000:]}"
        )
    return [line.strip() for line in completed.stdout.splitlines()]


def verify_full_refresh(engine: Path, network: Path) -> tuple[str, str]:
    expected = expected_trace(network)
    commands = ["uci", f"setoption name EvalFile value {network}", "isready"]
    for fixture in ENGINE_FIXTURES:
        commands.extend(
            (
                f"position fen {fixture_fen(fixture.board, fixture.side_to_move, fixture.rule50)}",
                "eval",
                "horde-raw-eval",
                "horde-eval-debug",
            )
        )
    lines = require_success(run_engine(engine, commands), "candidate full-refresh probe")

    traces = [TRACE_RE.match(line) for line in lines if TRACE_RE.match(line)]
    raws = [RAW_RE.match(line) for line in lines if RAW_RE.match(line)]
    finals = [FINAL_RE.match(line) for line in lines if FINAL_RE.match(line)]
    expected_by_name = {position["name"]: position for position in expected["positions"]}
    expected_positions = [expected_by_name[fixture.name] for fixture in ENGINE_FIXTURES]
    if not (len(traces) == len(raws) == len(finals) == len(expected_positions)):
        raise AssertionError(
            "candidate diagnostics returned an incomplete fixture set: "
            f"traces={len(traces)}, raw={len(raws)}, final={len(finals)}, "
            f"expected={len(expected_positions)}"
        )

    for index, (trace, raw, final, position) in enumerate(
        zip(traces, raws, finals, expected_positions, strict=True)
    ):
        assert trace is not None and raw is not None and final is not None
        actual = {
            "schema": trace.group(1),
            "file_sha256": trace.group(2),
            "parameter_sha256": trace.group(3),
            "output_affine": int(trace.group(4)),
            "pre_rule50": int(trace.group(5)),
            "value": int(trace.group(6)),
        }
        wanted = {
            "schema": expected["network_schema"],
            "file_sha256": expected["file_sha256"],
            "parameter_sha256": expected["parameter_sha256"],
            "output_affine": position["output_affine"],
            "pre_rule50": position["pre_rule50"],
            "value": position["value"],
        }
        if actual != wanted:
            raise AssertionError(
                f"candidate integer mismatch at fixture {index} ({position['name']}): "
                f"actual={actual}, expected={wanted}"
            )
        raw_values = tuple(int(raw.group(item)) for item in range(1, 4))
        if raw_values != (0, position["output_affine"], position["output_affine"]):
            raise AssertionError(f"candidate raw output mismatch at fixture {position['name']}")
        if int(final.group(1)) != position["value"]:
            raise AssertionError(f"candidate final output mismatch at fixture {position['name']}")

    return str(expected["network_schema"]), str(expected["file_sha256"])


def benchmark_receipt(engine: Path, network: Path) -> tuple[int, str]:
    commands = [
        f"setoption name EvalFile value {network}",
        "bench 1 1 1 default depth",
    ]
    completed = run_engine(engine, commands)
    lines = require_success(completed, "candidate depth-1 benchmark")
    lines.extend(line.strip() for line in completed.stderr.splitlines())
    bestmoves = [line for line in lines if line.startswith("bestmove ")]
    nodes = [NODES_RE.match(line) for line in lines if NODES_RE.match(line)]
    if len(nodes) != 1 or not bestmoves:
        raise AssertionError("candidate benchmark did not return nodes and best moves")
    count = int(nodes[0].group(1))
    if count <= 0:
        raise AssertionError("candidate benchmark searched zero nodes")
    digest = hashlib.sha256("\n".join(bestmoves).encode("ascii")).hexdigest().upper()
    return count, digest


def verify_fail_closed(engine: Path, network: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="horde-v2-candidate-corrupt-") as raw:
        corrupted = Path(raw) / "corrupted.hsv2"
        payload = bytearray(network.read_bytes())
        payload[-1] ^= 1
        corrupted.write_bytes(payload)
        completed = run_engine(
            engine,
            [
                f"setoption name EvalFile value {corrupted}",
                "position startpos",
                "horde-eval-debug",
            ],
        )
    if completed.returncode == 0:
        raise AssertionError("candidate engine accepted a corrupted container")
    if "PAYLOAD_MISMATCH" not in completed.stdout + completed.stderr:
        raise AssertionError("candidate engine did not report the corruption class")

    legacy = REPO_ROOT / "networks" / "hordetest_run6b_e37_l06.nnue"
    completed = run_engine(
        engine,
        [
            f"setoption name EvalFile value {legacy}",
            "position startpos",
            "horde-eval-debug",
        ],
    )
    if completed.returncode == 0:
        raise AssertionError("candidate engine fell back to the legacy Run 6B network")
    if "MAGIC_MISMATCH" not in completed.stdout + completed.stderr:
        raise AssertionError("candidate engine did not reject the legacy network as a V2 container")


def verify_engine(engine: Path, network: Path) -> None:
    engine = engine.expanduser().resolve()
    network = network.expanduser().resolve()
    if not engine.is_file():
        raise FileNotFoundError(f"candidate engine does not exist: {engine}")
    if not network.is_file():
        raise FileNotFoundError(f"candidate network does not exist: {network}")

    schema, file_sha256 = verify_full_refresh(engine, network)
    first = benchmark_receipt(engine, network)
    second = benchmark_receipt(engine, network)
    if first != second:
        raise AssertionError(f"candidate benchmark is not deterministic: {first} != {second}")
    verify_fail_closed(engine, network)
    print(
        "Horde V2 candidate engine dispatch passed: "
        f"schema={schema}, sha256={file_sha256}, nodes={first[0]}, "
        f"bestmoves_sha256={first[1]}, fixtures={len(ENGINE_FIXTURES)}, fail_closed=true"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixture = subparsers.add_parser("fixture", help="write one deterministic registered container")
    fixture.add_argument("output", type=Path)
    fixture.add_argument("--architecture", default="v2-c1-rank8-64x192")

    verify = subparsers.add_parser("verify", help="verify a candidate engine and container")
    verify.add_argument("engine", type=Path)
    verify.add_argument("network", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "fixture":
        create_fixture(args.output.expanduser().resolve(), args.architecture)
    else:
        verify_engine(args.engine, args.network)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AssertionError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
