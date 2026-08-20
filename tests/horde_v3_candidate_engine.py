#!/usr/bin/env python3
"""Build fixtures and verify the fail-closed Horde V3 engine dispatch.

Three subcommands:

``fixture``      writes one deterministic, registered ``.hsv3`` container.
``verify``      drives a dispatched engine and checks, in order, that its
                integer outputs equal an independent Python forward, that a
                depth-1 benchmark over en passant, castling and promotion
                positions is deterministic, and that a corrupted, a foreign or
                a missing artifact makes the next evaluation exit
                unsuccessfully instead of falling back.
``build-guard`` checks the build boundary itself: a declared macro that
                contradicts the EVALFILE extension must be a hard error.

The reference forward is ``tools/horde_v3_integer_eval.py``, which decodes the
container independently of the engine and reimplements the frozen V3 contract
in NumPy. Nothing here reads the engine's own arithmetic to decide what the
answer should be.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from horde_v3_parity import board_from_fen, build_fixture_container  # noqa: E402

from horde_training_decoder import v3_global_rows  # noqa: E402
import horde_v3_container as v3  # noqa: E402
import horde_v3_integer_eval as integer_eval  # noqa: E402


TRACE_RE = re.compile(
    r"^horde-v3-candidate-eval schema=(\S+) file_sha256=([0-9A-F]{64}) "
    r"parameter_sha256=([0-9A-F]{64}) output_affine=(-?\d+) "
    r"pre_rule50=(-?\d+) value=(-?\d+)$"
)
RAW_RE = re.compile(r"^horde-raw-eval (-?\d+) (-?\d+) (-?\d+)$")
FINAL_RE = re.compile(r"^horde-eval-debug eval=(-?\d+)$")
NODES_RE = re.compile(r"^Nodes searched\s*:\s*(\d+)$")
VERIFY_RE = re.compile(r"^(?:info string )?NNUE evaluation using .* \[(?P<identity>.*)\]$")
PERFT_MOVE_RE = re.compile(r"^([a-h][1-8][a-h][1-8][nbrq]?): \d+$")

LEGACY_NETWORK = REPO_ROOT / "networks" / "hordetest_run6b_e37_l06.nnue"


class Fixture:
    """One engine-facing position plus the special move it must be able to play."""

    __slots__ = ("name", "fen", "required_move")

    def __init__(self, name: str, fen: str, required_move: str | None = None) -> None:
        self.name = name
        self.fen = fen
        self.required_move = required_move


# Every FEN is a legal Horde position that Position::set accepts. The three
# special move shapes the V3 incremental stack has to survive are named
# explicitly so a position that silently stopped offering them is a failure and
# not a quietly weaker test.
ENGINE_FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        "horde-start",
        "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1",
    ),
    Fixture(
        "horde-start-damped",
        "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP b kq - 37 20",
    ),
    Fixture(
        "en-passant",
        "rnbqkbnr/6p1/2p1Pp1P/P1PPPP2/Pp4PP/1p2PPPP/1P2PPPP/PP1nPPPP b kq a3 0 18",
        required_move="b4a3",
    ),
    Fixture(
        "black-castling",
        "r3k2r/8/8/8/8/8/8/P7 b kq - 0 1",
        required_move="e8g8",
    ),
    Fixture(
        "white-promotion",
        "4k3/P7/8/8/8/8/8/8 w - - 0 1",
        required_move="a7a8q",
    ),
    Fixture(
        "black-promotion",
        "k7/5p2/4p2P/3p2P1/2p2P2/1p2P2P/p2P2P1/2P2P2 b - - 0 1",
        required_move="a2a1q",
    ),
    Fixture("middlegame-b", "4k3/pp4q1/3P2p1/8/P3PP2/PPP2r2/PPP5/PPPP4 b - - 0 1"),
    Fixture("middlegame-w", "4k3/7r/8/P7/2p1n2P/3p2P1/1P3P2/PPP1PPP1 w - - 0 1"),
    Fixture("thin-material", "4k3/8/8/8/3q4/2N5/1P6/R7 w - - 0 1"),
    Fixture("single-queen", "4k3/7p/8/8/8/8/8/Q7 w - - 0 1"),
    Fixture("pawn-chain", "4k3/8/8/4P3/3P4/2P5/1P6/P7 w - - 0 1"),
    Fixture("two-ranks", "rnbqkbnr/pppppppp/8/1PP2PP1/8/8/PPPPPPPP/PPPPPPPP w kq - 0 1"),
    Fixture("three-ranks", "rnbqkbnr/pppppppp/8/1PP2PP1/8/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1"),
)

MINIMUM_PHASE_BUCKETS = 6


# ---------------------------------------------------------------------------
# Fixture container
# ---------------------------------------------------------------------------


def create_fixture(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = build_fixture_container(output)
    parsed = v3.read_container(output)
    if parsed.file_sha256 != receipt["file_sha256"]:
        raise AssertionError("fixture receipt does not authenticate the emitted container")
    if parsed.parameter_sha256 != receipt["parameter_sha256"]:
        raise AssertionError("fixture receipt does not authenticate the emitted parameters")
    print(
        f"Horde V3 fixture created: schema={v3.NETWORK_SCHEMA_NAME}, "
        f"bytes={output.stat().st_size}, sha256={parsed.file_sha256}"
    )


# ---------------------------------------------------------------------------
# Engine driving
# ---------------------------------------------------------------------------


def run_engine(
    engine: Path, commands: list[str], timeout: int = 120
) -> subprocess.CompletedProcess[str]:
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


# ---------------------------------------------------------------------------
# The independent Python forward
# ---------------------------------------------------------------------------


def reference_layers(network_path: Path) -> tuple[dict[str, object], dict[str, str]]:
    """Evaluate every fixture with tools/horde_v3_integer_eval.py."""

    network = integer_eval.IntegerNetworkV3.load(network_path)
    streams: list[tuple[int, ...]] = []
    sides: list[int] = []
    white: list[int] = []
    rule50: list[int] = []
    for fixture in ENGINE_FIXTURES:
        board, side, halfmove = board_from_fen(fixture.fen)
        rows = v3_global_rows(board)
        if not rows:
            raise AssertionError(f"{fixture.name}: the V3 enumerator activated no row")
        streams.append(rows)
        sides.append(side)
        white.append(sum(1 for code in board if code and code <= 5))
        rule50.append(halfmove)

    batch = integer_eval.make_batch(streams, sides, white, rule50)
    layers = network.evaluate_layers(batch)
    identity = {
        "schema": v3.NETWORK_SCHEMA_NAME,
        "file_sha256": network.container.file_sha256,
        "parameter_sha256": network.container.parameter_sha256,
    }
    expected = {
        "buckets": [int(value) for value in layers["bucket"]],
        "output_pre": [int(value) for value in layers["output_pre"]],
        "psqt_sum": [int(value) for value in layers["psqt_sum"]],
        "pre_rule50": [int(value) for value in layers["value"]],
        "score": [int(value) for value in layers["score"]],
        "sides": sides,
        "white_piece_count": white,
        "rule50": rule50,
    }
    return expected, identity


def check_coverage(expected: dict[str, object]) -> tuple[list[int], list[int]]:
    buckets = sorted(set(expected["buckets"]))  # type: ignore[arg-type]
    sides = sorted(set(expected["sides"]))  # type: ignore[arg-type]
    if len(buckets) < MINIMUM_PHASE_BUCKETS:
        raise AssertionError(
            f"the engine fixtures only reach phase buckets {buckets}; at least "
            f"{MINIMUM_PHASE_BUCKETS} distinct buckets are required"
        )
    if sides != [0, 1]:
        raise AssertionError(f"the engine fixtures only reach sides to move {sides}")
    if not any(value > 0 for value in expected["rule50"]):  # type: ignore[union-attr]
        raise AssertionError("no engine fixture exercises the rule-50 damping")
    return buckets, sides


# ---------------------------------------------------------------------------
# a) Integer parity of the dispatched engine
# ---------------------------------------------------------------------------


def verify_full_refresh(
    engine: Path, network: Path, expected: dict[str, object], identity: dict[str, str]
) -> str:
    commands = ["uci", f"setoption name EvalFile value {network}", "isready"]
    for fixture in ENGINE_FIXTURES:
        commands.extend(
            (
                f"position fen {fixture.fen}",
                "eval",
                "horde-raw-eval",
                "horde-eval-debug",
            )
        )
    completed = run_engine(engine, commands)
    lines = require_success(completed, "candidate full-refresh probe")

    traces = [match for match in (TRACE_RE.match(line) for line in lines) if match]
    raws = [match for match in (RAW_RE.match(line) for line in lines) if match]
    finals = [match for match in (FINAL_RE.match(line) for line in lines) if match]
    if not (len(traces) == len(raws) == len(finals) == len(ENGINE_FIXTURES)):
        raise AssertionError(
            "candidate diagnostics returned an incomplete fixture set: "
            f"traces={len(traces)}, raw={len(raws)}, final={len(finals)}, "
            f"expected={len(ENGINE_FIXTURES)}"
        )

    for index, fixture in enumerate(ENGINE_FIXTURES):
        trace, raw, final = traces[index], raws[index], finals[index]
        actual = {
            "schema": trace.group(1),
            "file_sha256": trace.group(2),
            "parameter_sha256": trace.group(3),
            "output_affine": int(trace.group(4)),
            "pre_rule50": int(trace.group(5)),
            "value": int(trace.group(6)),
        }
        wanted = {
            "schema": identity["schema"],
            "file_sha256": identity["file_sha256"],
            "parameter_sha256": identity["parameter_sha256"],
            "output_affine": expected["output_pre"][index],  # type: ignore[index]
            "pre_rule50": expected["pre_rule50"][index],  # type: ignore[index]
            "value": expected["score"][index],  # type: ignore[index]
        }
        if actual != wanted:
            raise AssertionError(
                f"candidate integer mismatch at fixture {index} ({fixture.name}): "
                f"actual={actual}, expected={wanted}"
            )

        raw_values = tuple(int(raw.group(item)) for item in range(1, 4))
        psqt = expected["psqt_sum"][index]  # type: ignore[index]
        affine = expected["output_pre"][index]  # type: ignore[index]
        if raw_values != (psqt, affine, psqt + affine):
            raise AssertionError(
                f"candidate raw output mismatch at fixture {fixture.name}: "
                f"actual={raw_values}, expected={(psqt, affine, psqt + affine)}"
            )
        if int(final.group(1)) != expected["score"][index]:  # type: ignore[index]
            raise AssertionError(f"candidate final output mismatch at fixture {fixture.name}")

    verifications = [
        match.group("identity") for match in (VERIFY_RE.match(line) for line in lines) if match
    ]
    if not verifications:
        raise AssertionError("the candidate engine never reported the active network identity")
    for reported in verifications:
        if identity["file_sha256"] not in reported or identity["schema"] not in reported:
            raise AssertionError(f"the reported network identity is not the loaded one: {reported}")
    return verifications[0]


# ---------------------------------------------------------------------------
# b) The special move shapes really are reachable, then the depth-1 benchmark
# ---------------------------------------------------------------------------


def verify_special_moves(engine: Path, network: Path) -> dict[str, str]:
    required = {
        fixture.name: fixture.required_move
        for fixture in ENGINE_FIXTURES
        if fixture.required_move is not None
    }
    if len(required) < 3:
        raise AssertionError("the fixture set no longer names en passant, castling and promotion")

    commands = [f"setoption name EvalFile value {network}"]
    ordered = [fixture for fixture in ENGINE_FIXTURES if fixture.required_move is not None]
    for fixture in ordered:
        commands.extend((f"position fen {fixture.fen}", "go perft 1"))
    lines = require_success(run_engine(engine, commands), "candidate root move enumeration")

    groups: list[set[str]] = []
    current: set[str] = set()
    for line in lines:
        match = PERFT_MOVE_RE.match(line)
        if match:
            current.add(match.group(1))
        elif line.startswith("Nodes searched") and current:
            groups.append(current)
            current = set()
    if current:
        groups.append(current)
    if len(groups) != len(ordered):
        raise AssertionError(
            f"root move enumeration returned {len(groups)} position(s), expected {len(ordered)}"
        )
    for fixture, moves in zip(ordered, groups):
        if fixture.required_move not in moves:
            raise AssertionError(
                f"{fixture.name}: the required move {fixture.required_move} is not legal, so the "
                "benchmark would not exercise that move shape"
            )
    return required


def benchmark_receipt(engine: Path, network: Path, positions: Path) -> tuple[int, str]:
    commands = [
        f"setoption name EvalFile value {network}",
        f"bench 1 1 1 {positions} depth",
    ]
    completed = run_engine(engine, commands)
    lines = require_success(completed, "candidate depth-1 benchmark")
    lines.extend(line.strip() for line in completed.stderr.splitlines())
    bestmoves = [line for line in lines if line.startswith("bestmove ")]
    nodes = [match for match in (NODES_RE.match(line) for line in lines) if match]
    if len(nodes) != 1 or not bestmoves:
        raise AssertionError("candidate benchmark did not return nodes and best moves")
    if len(bestmoves) != len(ENGINE_FIXTURES):
        raise AssertionError(
            f"candidate benchmark produced {len(bestmoves)} best moves for "
            f"{len(ENGINE_FIXTURES)} positions"
        )
    count = int(nodes[0].group(1))
    if count <= 0:
        raise AssertionError("candidate benchmark searched zero nodes")
    digest = hashlib.sha256("\n".join(bestmoves).encode("ascii")).hexdigest().upper()
    return count, digest


# ---------------------------------------------------------------------------
# c) Fail closed. No fallback of any kind.
# ---------------------------------------------------------------------------


def probe_rejection(
    engine: Path, network: Path, label: str, expected_error: str, allowed_evaluations: int = 0
) -> None:
    completed = run_engine(
        engine,
        [
            f"setoption name EvalFile value {network}",
            "position startpos",
            "horde-eval-debug",
            "eval",
            "go depth 1",
        ],
    )
    output = completed.stdout + completed.stderr
    if completed.returncode == 0:
        raise AssertionError(f"the candidate engine accepted {label}")
    if expected_error not in output:
        raise AssertionError(
            f"the candidate engine did not report {expected_error} for {label}:\n{output[-4000:]}"
        )

    evaluations = [line for line in output.splitlines() if FINAL_RE.match(line.strip())]
    if len(evaluations) != allowed_evaluations:
        raise AssertionError(
            f"{label}: the engine emitted {len(evaluations)} evaluations, "
            f"{allowed_evaluations} were permitted; it fell back instead of failing closed"
        )
    if any(line.strip() == "horde-eval-debug eval=0" for line in output.splitlines()):
        raise AssertionError(f"{label}: the engine fell back to a zero evaluation")
    if any(line.strip().startswith("bestmove") for line in output.splitlines()):
        raise AssertionError(f"{label}: the engine searched with an invalidated network")


def verify_fail_closed(engine: Path, network: Path, expected: dict[str, object]) -> None:
    with tempfile.TemporaryDirectory(prefix="horde-v3-candidate-corrupt-") as raw:
        scratch = Path(raw)

        corrupted = scratch / "corrupted.hsv3"
        payload = bytearray(network.read_bytes())
        payload[-1] ^= 1
        corrupted.write_bytes(payload)
        probe_rejection(engine, corrupted, "a corrupted container", "PAYLOAD_MISMATCH")

        truncated = scratch / "truncated.hsv3"
        truncated.write_bytes(network.read_bytes()[:-64])
        probe_rejection(engine, truncated, "a truncated container", "HEADER_MISMATCH")

        missing = scratch / "absent.hsv3"
        probe_rejection(engine, missing, "a missing container", "OPEN_FAILED")

        if not LEGACY_NETWORK.is_file():
            raise FileNotFoundError(f"the registered Run 6B network is missing: {LEGACY_NETWORK}")
        probe_rejection(engine, LEGACY_NETWORK, "the legacy Run 6B network", "MAGIC_MISMATCH")

        # A network that was valid a moment ago must not survive its
        # replacement by a corrupted one.
        completed = run_engine(
            engine,
            [
                f"setoption name EvalFile value {network}",
                "position startpos",
                "horde-eval-debug",
                f"setoption name EvalFile value {corrupted}",
                "position startpos",
                "horde-eval-debug",
            ],
        )
        output = completed.stdout + completed.stderr
        if completed.returncode == 0:
            raise AssertionError("the candidate engine survived a replacement by a corrupted one")
        evaluations = [
            match.group(1)
            for match in (FINAL_RE.match(line.strip()) for line in output.splitlines())
            if match
        ]
        if len(evaluations) != 1:
            raise AssertionError(
                f"the candidate engine emitted {len(evaluations)} evaluations across a failed "
                "replacement; the previously loaded network was reused"
            )
        if int(evaluations[0]) != expected["score"][0]:  # type: ignore[index]
            raise AssertionError("the pre-replacement evaluation is not the authenticated one")


# ---------------------------------------------------------------------------
# d) The build boundary
# ---------------------------------------------------------------------------


def run_make(make: str, source: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [make, "-s", "-C", str(source), "help", *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )


def verify_build_guard(make: str, source: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="horde-v3-build-guard-") as raw:
        scratch = Path(raw)
        container = scratch / "guard.hsv3"
        create_fixture(container)
        legacy = scratch / "guard.nnue"
        legacy.write_bytes(b"not a container")

        # Make sees POSIX separators on every platform, so the extension
        # filters and the quoted define behave identically under MSYS2.
        hsv3 = container.as_posix()
        hsv2 = (scratch / "guard.hsv2").as_posix()
        nnue = legacy.as_posix()

        must_fail = [
            (
                "a .hsv3 EVALFILE with the V3 dispatch disabled",
                [f"EVALFILE={hsv3}", "HORDE_V3_CANDIDATE=no"],
            ),
            (
                "the V3 dispatch forced onto a .nnue EVALFILE",
                [f"EVALFILE={nnue}", "HORDE_V3_CANDIDATE=yes"],
            ),
            ("the V3 dispatch forced with no EVALFILE", ["EVALFILE=", "HORDE_V3_CANDIDATE=yes"]),
            ("an unknown V3 dispatch value", [f"EVALFILE={hsv3}", "HORDE_V3_CANDIDATE=maybe"]),
            (
                "a .hsv3 EVALFILE claimed by the V2 dispatch",
                [f"EVALFILE={hsv3}", "HORDE_V2_CANDIDATE=yes"],
            ),
            (
                "a .hsv2 EVALFILE claimed by the V3 dispatch",
                [f"EVALFILE={hsv2}", "HORDE_V3_CANDIDATE=yes"],
            ),
            (
                "a .hsv2 EVALFILE with the V2 dispatch disabled",
                [f"EVALFILE={hsv2}", "HORDE_V2_CANDIDATE=no"],
            ),
        ]
        must_pass = [
            ("a .hsv3 EVALFILE resolved automatically", [f"EVALFILE={hsv3}"]),
            (
                "a .hsv3 EVALFILE with the V3 dispatch requested",
                [f"EVALFILE={hsv3}", "HORDE_V3_CANDIDATE=yes"],
            ),
            ("the registered Run 6B build", []),
            ("the registered Run 6B build with the V3 dispatch disabled",
             ["HORDE_V3_CANDIDATE=no"]),
        ]

        for label, arguments in must_fail:
            completed = run_make(make, source, arguments)
            if completed.returncode == 0:
                raise AssertionError(f"the build boundary accepted {label}: make {arguments}")
            print(f"  rejected: {label}")
        for label, arguments in must_pass:
            completed = run_make(make, source, arguments)
            if completed.returncode != 0:
                raise AssertionError(
                    f"the build boundary rejected {label}: make {arguments}\n"
                    f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
                )
            print(f"  accepted: {label}")
    return len(must_fail)


# ---------------------------------------------------------------------------


def verify_engine(engine: Path, network: Path, require_shadow: bool) -> None:
    engine = engine.expanduser().resolve()
    network = network.expanduser().resolve()
    if not engine.is_file():
        raise FileNotFoundError(f"candidate engine does not exist: {engine}")
    if not network.is_file():
        raise FileNotFoundError(f"candidate network does not exist: {network}")

    expected, identity = reference_layers(network)
    buckets, sides = check_coverage(expected)

    reported_identity = verify_full_refresh(engine, network, expected, identity)
    if require_shadow and "shadow" not in reported_identity:
        raise AssertionError(
            "this engine was not built with -DHORDE_V3_CANDIDATE_SHADOW, so the benchmark does "
            f"not compare the search stack against a full refresh: [{reported_identity}]"
        )

    required = verify_special_moves(engine, network)

    with tempfile.TemporaryDirectory(prefix="horde-v3-candidate-bench-") as raw:
        positions = Path(raw) / "positions.fen"
        positions.write_text(
            "\n".join(fixture.fen for fixture in ENGINE_FIXTURES) + "\n", encoding="ascii"
        )
        first = benchmark_receipt(engine, network, positions)
        second = benchmark_receipt(engine, network, positions)
    if first != second:
        raise AssertionError(f"candidate benchmark is not deterministic: {first} != {second}")

    verify_fail_closed(engine, network, expected)

    print(
        "Horde V3 candidate engine dispatch passed: "
        f"schema={identity['schema']}, sha256={identity['file_sha256']}, "
        f"parameter_sha256={identity['parameter_sha256']}, fixtures={len(ENGINE_FIXTURES)}, "
        f"phase_buckets={buckets}, sides={sides}, "
        f"special_moves={sorted(required.values())}, nodes={first[0]}, "
        f"bestmoves_sha256={first[1]}, "
        f"stack_equals_full_refresh={'enforced' if require_shadow else 'not asserted'}, "
        "fail_closed=true"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixture = subparsers.add_parser("fixture", help="write one deterministic registered container")
    fixture.add_argument("output", type=Path)

    verify = subparsers.add_parser("verify", help="verify a candidate engine and container")
    verify.add_argument("engine", type=Path)
    verify.add_argument("network", type=Path)
    verify.add_argument(
        "--no-shadow",
        dest="shadow",
        action="store_false",
        help="the engine was not built with -DHORDE_V3_CANDIDATE_SHADOW",
    )
    verify.set_defaults(shadow=True)

    guard = subparsers.add_parser("build-guard", help="verify the build dispatch boundary")
    guard.add_argument("--source", type=Path, default=REPO_ROOT / "src")
    guard.add_argument("--make", default=os.environ.get("MAKE", "make"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "fixture":
        create_fixture(args.output.expanduser().resolve())
    elif args.command == "build-guard":
        make = shutil.which(args.make) or args.make
        rejected = verify_build_guard(make, args.source.expanduser().resolve())
        print(f"Horde V3 build dispatch boundary passed: {rejected} contradictions rejected")
    else:
        verify_engine(args.engine, args.network, args.shadow)
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
