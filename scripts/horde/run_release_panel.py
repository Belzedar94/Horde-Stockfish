#!/usr/bin/env python3
"""Run the fixed Horde-Stockfish release strength panel.

The runner is intentionally fail-closed: every executable, network and book is
authenticated before play; the three time controls have fixed game counts;
and any abnormal termination invalidates the whole panel. It never performs
early score adjudication.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Iterable


RUN6B_SHA256 = "B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7"
CANDIDATE_NAME = "Horde-Stockfish-release"
BASELINE_NAME = "Fairy-Stockfish-dev"


@dataclass(frozen=True)
class PanelSpec:
    label: str
    tc: str
    hash_mb: int
    games: int
    seed: int


PANEL_SPECS = (
    PanelSpec("VSTC", "2+0.02", 16, 600, 2026081001),
    PanelSpec("STC", "10+0.1", 32, 400, 2026081002),
    PanelSpec("LTC", "30+0.3", 128, 200, 2026081003),
)

FINISHED_RE = re.compile(
    r"^Finished game\s+(\d+)\s+\((.+) vs (.+)\):\s+"
    r"(1-0|0-1|1/2-1/2)\s+\{(.*)\}\s*$"
)
BAD_REASON_TERMS = (
    "abandon",
    "crash",
    "disconnect",
    "failed",
    "illegal",
    "on time",
    "stall",
    "terminated",
    "timed out",
)
HEX_40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
HEX_64_RE = re.compile(r"^[0-9a-fA-F]{64}$")

PRINT_LOCK = threading.Lock()
ACTIVE_LOCK = threading.Lock()
ACTIVE_PROCESSES: set[subprocess.Popen[str]] = set()


def emit(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def logistic_elo(score: float) -> float:
    score = min(max(score, 1e-12), 1.0 - 1e-12)
    return -400.0 * math.log10(1.0 / score - 1.0)


def statistics_from_penta(penta: list[int]) -> dict[str, float]:
    pairs = sum(penta)
    if not pairs:
        return {"elo": 0.0, "ci95": 0.0, "los": 0.5}

    values = [index / 4.0 for index in range(5)]
    mean = sum(value * count for value, count in zip(values, penta)) / pairs
    variance = sum(
        ((value - mean) ** 2) * count for value, count in zip(values, penta)
    ) / pairs
    if variance == 0.0:
        los = 0.5 if mean == 0.5 else float(mean > 0.5)
        return {"elo": logistic_elo(mean), "ci95": 0.0, "los": los}

    standard_error = math.sqrt(variance / pairs)
    z95 = NormalDist().inv_cdf(0.975)
    lower = logistic_elo(mean - z95 * standard_error)
    middle = logistic_elo(mean)
    upper = logistic_elo(mean + z95 * standard_error)
    return {
        "elo": middle,
        "ci95": (upper - lower) / 2.0,
        "los": NormalDist().cdf((mean - 0.5) / standard_error),
    }


def candidate_score(white: str, black: str, result: str) -> float:
    if result == "1/2-1/2":
        return 0.5
    winner = white if result == "1-0" else black
    if winner == CANDIDATE_NAME:
        return 1.0
    if winner == BASELINE_NAME:
        return 0.0
    raise RuntimeError(f"Unknown winner {winner!r}")


class PanelTracker:
    def __init__(self, spec: PanelSpec) -> None:
        self.spec = spec
        self.games: dict[int, float] = {}
        self.anomalies: list[str] = []
        self.status = "PENDING"
        self.lock = threading.Lock()

    def consume(self, line: str) -> None:
        match = FINISHED_RE.match(line.strip())
        if not match:
            return

        game_no = int(match.group(1))
        white, black, result, reason = match.group(2, 3, 4, 5)
        if {white, black} != {CANDIDATE_NAME, BASELINE_NAME}:
            raise RuntimeError(
                f"{self.spec.label} game {game_no} used unexpected engines: "
                f"{white} vs {black}"
            )
        if not 1 <= game_no <= self.spec.games:
            raise RuntimeError(
                f"{self.spec.label} reported out-of-range game {game_no}"
            )

        score = candidate_score(white, black, result)
        anomaly = next(
            (term for term in BAD_REASON_TERMS if term in reason.lower()), None
        )
        with self.lock:
            previous = self.games.get(game_no)
            if previous is not None:
                raise RuntimeError(
                    f"{self.spec.label} reported game {game_no} more than once"
                )
            self.games[game_no] = score
            if anomaly is not None:
                self.anomalies.append(f"game {game_no}: {reason}")
        if anomaly is not None:
            raise RuntimeError(
                f"{self.spec.label} infrastructure defect in game {game_no}: {reason}"
            )

    def set_status(self, status: str) -> None:
        with self.lock:
            self.status = status

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            games = dict(self.games)
            anomalies = list(self.anomalies)
            status = self.status

        wld = [0, 0, 0]
        for score in games.values():
            wld[0 if score == 1.0 else 1 if score == 0.0 else 2] += 1

        penta = [0, 0, 0, 0, 0]
        for first in range(1, self.spec.games + 1, 2):
            second = first + 1
            if first in games and second in games:
                bucket = int(round(2 * (games[first] + games[second])))
                penta[bucket] += 1

        return {
            "label": self.spec.label,
            "tc": self.spec.tc,
            "hash_mb": self.spec.hash_mb,
            "expected_games": self.spec.games,
            "games": len(games),
            "wld": wld,
            "pentanomial": penta,
            "complete_pairs": sum(penta),
            "statistics": statistics_from_penta(penta),
            "anomalies": anomalies,
            "status": status,
        }

    def require_complete(self) -> dict[str, object]:
        snapshot = self.snapshot()
        expected = set(range(1, self.spec.games + 1))
        with self.lock:
            observed = set(self.games)
        missing = sorted(expected - observed)
        if missing:
            preview = ", ".join(str(value) for value in missing[:10])
            raise RuntimeError(
                f"{self.spec.label} expected {self.spec.games} games, missing {preview}"
            )
        if snapshot["anomalies"]:
            raise RuntimeError(
                f"{self.spec.label} infrastructure defect: "
                + "; ".join(snapshot["anomalies"])
            )
        if snapshot["complete_pairs"] != self.spec.games // 2:
            raise RuntimeError(f"{self.spec.label} did not produce complete pairs")
        return snapshot


def path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def checked_sha(value: str) -> str:
    normalized = value.upper()
    if not HEX_64_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected a complete 64-character SHA-256")
    return normalized


def checked_commit(value: str) -> str:
    normalized = value.lower()
    if not HEX_40_RE.fullmatch(normalized):
        raise argparse.ArgumentTypeError("expected a complete 40-character commit")
    return normalized


def count_openings(book: Path) -> int:
    with book.open("r", encoding="utf-8-sig") as handle:
        return sum(
            1
            for line in handle
            if line.strip() and not line.lstrip().startswith(("#", ";"))
        )


def validate_inputs(args: argparse.Namespace) -> dict[str, object]:
    roles = {
        "candidate_executable": (args.candidate, args.candidate_sha256),
        "candidate_network": (args.candidate_network, args.candidate_network_sha256),
        "baseline_executable": (args.baseline, args.baseline_sha256),
        "baseline_network": (args.baseline_network, args.baseline_network_sha256),
        "referee": (args.referee, args.referee_sha256),
        "book": (args.book, args.book_sha256),
    }
    observed: dict[str, dict[str, object]] = {}
    for role, (path, expected_sha) in roles.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role}: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"{role} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        observed[role] = {
            "path": str(path),
            "sha256": actual_sha,
            "size": path.stat().st_size,
        }

    openings = count_openings(args.book)
    required = max(spec.games // 2 for spec in PANEL_SPECS)
    if openings < required:
        raise RuntimeError(
            f"Opening book has {openings} positions; the panel needs at least {required}"
        )
    observed["book"]["opening_count"] = openings
    runner = Path(__file__).resolve()
    observed["runner_script"] = {
        "path": str(runner),
        "sha256": sha256_file(runner),
        "size": runner.stat().st_size,
    }
    return observed


def add_options(engine: list[str], options: Iterable[str]) -> None:
    engine.extend(f"option.{option}" for option in options)


def build_command(
    args: argparse.Namespace,
    spec: PanelSpec,
    pgn_path: Path,
) -> list[str]:
    candidate = [
        f"cmd={args.candidate}",
        f"dir={args.candidate.parent}",
        f"name={CANDIDATE_NAME}",
        "proto=uci",
        f"tc={spec.tc}",
        f"timemargin={args.time_margin_ms}",
        "option.UCI_Variant=horde",
        f"option.EvalFile={args.candidate_network}",
        "option.Threads=1",
        f"option.Hash={spec.hash_mb}",
        f"option.Move Overhead={args.move_overhead_ms}",
    ]
    add_options(candidate, args.candidate_option)

    baseline = [
        f"cmd={args.baseline}",
        f"dir={args.baseline.parent}",
        f"name={BASELINE_NAME}",
        "proto=uci",
        f"tc={spec.tc}",
        f"timemargin={args.time_margin_ms}",
        "option.UCI_Variant=horde",
        f"option.EvalFile={args.baseline_network}",
        "option.Threads=1",
        f"option.Hash={spec.hash_mb}",
        f"option.Move Overhead={args.move_overhead_ms}",
    ]
    add_options(baseline, args.baseline_option)

    return [
        str(args.referee),
        "-repeat",
        "-variant",
        "horde",
        "-concurrency",
        str(args.concurrency_per_tc),
        "-games",
        str(spec.games),
        "-maxmoves",
        "0",
        "-ratinginterval",
        "10",
        "-outcomeinterval",
        "10",
        "-engine",
        *candidate,
        "-engine",
        *baseline,
        "-openings",
        f"file={args.book}",
        "format=epd",
        "order=sequential",
        "start=1",
        "-srand",
        str(spec.seed),
        "-pgnout",
        str(pgn_path),
        "-event",
        f"Horde-Stockfish release panel {spec.label}",
        "-site",
        "local",
    ]


def terminate_active() -> None:
    with ACTIVE_LOCK:
        active = list(ACTIVE_PROCESSES)
    for process in active:
        if process.poll() is None:
            process.terminate()


def format_table(trackers: dict[str, PanelTracker]) -> str:
    rows = [
        f"Release panel status at {utc_now()}",
        "TC    Games       W-L-D          Penta                 Elo +/- CI95      LOS      State",
    ]
    for spec in PANEL_SPECS:
        snapshot = trackers[spec.label].snapshot()
        stats = snapshot["statistics"]
        wld = "-".join(str(value) for value in snapshot["wld"])
        penta = "/".join(str(value) for value in snapshot["pentanomial"])
        rows.append(
            f"{spec.label:<5} {snapshot['games']:>4}/{spec.games:<4} "
            f"{wld:<14} {penta:<21} "
            f"{stats['elo']:+7.2f} +/- {stats['ci95']:<7.2f} "
            f"{100.0 * stats['los']:>6.1f}%  {snapshot['status']}"
        )
    return "\n".join(rows)


def monitor_table(
    trackers: dict[str, PanelTracker], done: threading.Event, interval: float
) -> None:
    emit(format_table(trackers))
    while not done.wait(interval):
        emit(format_table(trackers))


def run_time_control(
    args: argparse.Namespace,
    spec: PanelSpec,
    tracker: PanelTracker,
    abort: threading.Event,
) -> dict[str, object]:
    tc_root = args.output_root / spec.label.lower()
    tc_root.mkdir(parents=True, exist_ok=False)
    pgn_path = tc_root / "games.pgn"
    log_path = tc_root / "cutechess.log"
    command = build_command(args, spec, pgn_path)
    tracker.set_status("RUNNING")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    with ACTIVE_LOCK:
        ACTIVE_PROCESSES.add(process)

    assert process.stdout is not None
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            for raw in process.stdout:
                line = raw.rstrip("\r\n")
                log.write(line + "\n")
                log.flush()
                tracker.consume(line)
                if abort.is_set() and process.poll() is None:
                    process.terminate()
        return_code = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        with ACTIVE_LOCK:
            ACTIVE_PROCESSES.discard(process)

    if abort.is_set():
        raise RuntimeError(f"{spec.label} aborted after another panel failure")
    if return_code != 0:
        raise RuntimeError(f"{spec.label} referee exited with status {return_code}")

    result = tracker.require_complete()
    tracker.set_status("PASS")
    result = tracker.snapshot()
    write_json(tc_root / "result.json", result)
    return result


def make_manifest(
    args: argparse.Namespace,
    assets: dict[str, object],
    commands: dict[str, list[str]],
    started_at: str,
) -> dict[str, object]:
    return {
        "schema": "HORDE_RELEASE_PANEL_V1",
        "started_at": started_at,
        "runner_commit": args.runner_commit,
        "candidate_commit": args.candidate_commit,
        "baseline_commit": args.baseline_commit,
        "referee_commit": args.referee_commit,
        "assets": assets,
        "panel": [asdict(spec) for spec in PANEL_SPECS],
        "commands": commands,
        "contract": {
            "variant": "horde",
            "paired_colors": True,
            "threads_per_engine": 1,
            "move_overhead_ms": args.move_overhead_ms,
            "time_margin_ms": args.time_margin_ms,
            "syzygy": False,
            "classical_adjudication": False,
            "early_score_stop": False,
            "report_interval_seconds": args.report_interval,
        },
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
    }


def run_panel(args: argparse.Namespace) -> int:
    assets = validate_inputs(args)
    commands = {
        spec.label: build_command(
            args, spec, args.output_root / spec.label.lower() / "games.pgn"
        )
        for spec in PANEL_SPECS
    }
    emit("Input authentication: PASS")

    if args.dry_run:
        for spec in PANEL_SPECS:
            emit(f"[{spec.label}] {subprocess.list2cmdline(commands[spec.label])}")
        return 0

    if args.output_root.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)
    started_at = utc_now()
    manifest = make_manifest(args, assets, commands, started_at)
    manifest_path = args.output_root / "manifest.json"
    write_json(manifest_path, manifest)

    trackers = {spec.label: PanelTracker(spec) for spec in PANEL_SPECS}
    done = threading.Event()
    abort = threading.Event()
    monitor = threading.Thread(
        target=monitor_table,
        args=(trackers, done, args.report_interval),
        name="release-panel-monitor",
        daemon=True,
    )
    monitor.start()

    results: dict[str, dict[str, object]] = {}
    error: BaseException | None = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    run_time_control,
                    args,
                    spec,
                    trackers[spec.label],
                    abort,
                ): spec.label
                for spec in PANEL_SPECS
            }
            for future in concurrent.futures.as_completed(futures):
                label = futures[future]
                try:
                    results[label] = future.result()
                except BaseException as exc:  # preserve KeyboardInterrupt receipt
                    if error is None:
                        error = exc
                    trackers[label].set_status("FAIL")
                    abort.set()
                    terminate_active()
    finally:
        done.set()
        monitor.join(timeout=2.0)
        emit(format_table(trackers))
        terminate_active()

    if error is not None:
        invalid = {
            "schema": "HORDE_RELEASE_PANEL_INVALID_V1",
            "started_at": started_at,
            "failed_at": utc_now(),
            "error": f"{type(error).__name__}: {error}",
            "states": {
                label: tracker.snapshot() for label, tracker in trackers.items()
            },
        }
        write_json(args.output_root / "INVALID.json", invalid)
        raise error

    ordered_results = {spec.label: results[spec.label] for spec in PANEL_SPECS}
    output_hashes: dict[str, dict[str, str]] = {}
    for spec in PANEL_SPECS:
        tc_root = args.output_root / spec.label.lower()
        output_hashes[spec.label] = {
            name: sha256_file(tc_root / name)
            for name in ("games.pgn", "cutechess.log", "result.json")
        }

    receipt = {
        "schema": "HORDE_RELEASE_PANEL_RECEIPT_V1",
        "started_at": started_at,
        "completed_at": utc_now(),
        "manifest_sha256": sha256_file(manifest_path),
        "results": ordered_results,
        "output_sha256": output_hashes,
        "gates": {
            "all_games_complete": sum(item["games"] for item in results.values())
            == sum(spec.games for spec in PANEL_SPECS),
            "all_pairs_complete": all(
                item["complete_pairs"] == spec.games // 2
                for spec, item in zip(PANEL_SPECS, ordered_results.values())
            ),
            "zero_infrastructure_defects": all(
                not item["anomalies"] for item in results.values()
            ),
            "passed": True,
        },
    }
    write_json(args.output_root / "panel-receipt.json", receipt)
    emit("Release panel PASS: 600/400/200 games complete with zero defects")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=path_arg)
    parser.add_argument("--candidate-sha256", required=True, type=checked_sha)
    parser.add_argument("--candidate-commit", required=True, type=checked_commit)
    parser.add_argument("--candidate-network", required=True, type=path_arg)
    parser.add_argument("--baseline", required=True, type=path_arg)
    parser.add_argument("--baseline-sha256", required=True, type=checked_sha)
    parser.add_argument("--baseline-commit", required=True, type=checked_commit)
    parser.add_argument("--baseline-network", required=True, type=path_arg)
    parser.add_argument(
        "--baseline-network-sha256", required=True, type=checked_sha
    )
    parser.add_argument("--referee", required=True, type=path_arg)
    parser.add_argument("--referee-sha256", required=True, type=checked_sha)
    parser.add_argument("--referee-commit", required=True, type=checked_commit)
    parser.add_argument("--book", required=True, type=path_arg)
    parser.add_argument("--book-sha256", required=True, type=checked_sha)
    parser.add_argument("--runner-commit", required=True, type=checked_commit)
    parser.add_argument("--output-root", required=True, type=path_arg)
    parser.add_argument("--candidate-option", action="append", default=[])
    parser.add_argument("--baseline-option", action="append", default=[])
    parser.add_argument("--concurrency-per-tc", type=int, default=4)
    parser.add_argument("--move-overhead-ms", type=int, default=50)
    parser.add_argument("--time-margin-ms", type=int, default=250)
    parser.add_argument("--report-interval", type=float, default=300.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.candidate_network_sha256 = RUN6B_SHA256
    if args.concurrency_per_tc <= 0:
        parser.error("--concurrency-per-tc must be positive")
    if args.move_overhead_ms < 0 or args.time_margin_ms < 0:
        parser.error("time margins cannot be negative")
    if args.report_interval <= 0:
        parser.error("--report-interval must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        return run_panel(args)
    except KeyboardInterrupt:
        terminate_active()
        emit("Interrupted; the panel is invalid")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
