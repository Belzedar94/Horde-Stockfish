#!/usr/bin/env python3
"""Compare search throughput of value-identical Horde V2 width builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ORDER_SEED = 0x48563257
BOOTSTRAP_SEED = 0x50414952
BOOTSTRAP_SAMPLES = 20_000


@dataclass(frozen=True)
class SearchReceipt:
    nodes: int
    bestmoves_sha256: str
    root_scores_sha256: str
    root_evals_sha256: str
    root_scores: tuple[str, ...]
    root_evals: tuple[int, ...]
    nps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired, value-identical Horde V2 engine-width benchmarks"
    )
    parser.add_argument(
        "--engine",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="named performance build; provide at least two",
    )
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--depth", type=int, default=13)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_engines(specs: list[str]) -> dict[str, Path]:
    engines: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --engine value: {spec!r}")
        name, raw_path = spec.split("=", 1)
        if not name or name in engines:
            raise ValueError(f"Duplicate or empty engine name: {name!r}")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ValueError(f"Engine not found: {path}")
        engines[name] = path
    if len(engines) < 2:
        raise ValueError("At least two --engine values are required")
    return engines


def run_engine(engine: Path, depth: int, timeout: float) -> SearchReceipt:
    command = (
        f"bench 16 1 1 default eval\nbench 16 1 {depth} default depth\nquit\n"
    ).encode("ascii")
    completed = subprocess.run(
        [str(engine)],
        input=command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    output = completed.stdout.decode("utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{engine.name} exited with {completed.returncode}:\n{output[-4000:]}"
        )

    node_matches = re.findall(r"Nodes searched\s*:\s*(\d+)", output)
    nps_matches = re.findall(r"Nodes/second\s*:\s*(\d+)", output)
    if len(node_matches) < 2 or len(nps_matches) < 2:
        raise RuntimeError(f"{engine.name} emitted an incomplete benchmark receipt")

    root_evals = tuple(
        int(line.split()[1])
        for line in output.splitlines()
        if line.startswith("horde-v2-perf-eval ")
    )
    if len(root_evals) != 10:
        raise RuntimeError(f"{engine.name} emitted {len(root_evals)} root evals; expected 10")

    bestmoves = [line for line in output.splitlines() if line.startswith("bestmove ")]
    if len(bestmoves) != 10:
        raise RuntimeError(f"{engine.name} emitted {len(bestmoves)} best moves; expected 10")

    root_scores: list[str] = []
    current_score: str | None = None
    for line in output.splitlines():
        score_match = re.search(r"^info .*\bscore (cp|mate) (-?\d+)\b", line)
        if score_match:
            node_match = re.search(r"\bnodes (\d+)\b", line)
            if node_match is None:
                raise RuntimeError(f"{engine.name} emitted a root score without a node count")
            padded = f" {line} "
            bound = (
                "lowerbound"
                if " lowerbound " in padded
                else "upperbound"
                if " upperbound " in padded
                else "exact"
            )
            current_score = (
                f"{score_match.group(1)}:{score_match.group(2)}:{bound}:"
                f"nodes={node_match.group(1)}"
            )
        elif line.startswith("bestmove "):
            if current_score is None:
                raise RuntimeError(f"{engine.name} emitted a best move without a root score")
            root_scores.append(f"{current_score}|{line}")
            current_score = None

    if len(root_scores) != 10:
        raise RuntimeError(f"{engine.name} emitted {len(root_scores)} root scores; expected 10")

    return SearchReceipt(
        nodes=int(node_matches[-1]),
        bestmoves_sha256=hashlib.sha256("|".join(bestmoves).encode("ascii")).hexdigest(),
        root_scores_sha256=hashlib.sha256("|".join(root_scores).encode("ascii")).hexdigest(),
        root_evals_sha256=hashlib.sha256(
            "|".join(str(value) for value in root_evals).encode("ascii")
        ).hexdigest(),
        root_scores=tuple(root_scores),
        root_evals=root_evals,
        nps=int(nps_matches[-1]),
    )


def percentile(sorted_values: list[float], probability: float) -> float:
    index = probability * (len(sorted_values) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    fraction = index - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_ratio_interval(numerator: list[int], denominator: list[int]) -> tuple[float, float, float]:
    logs = [math.log(left / right) for left, right in zip(numerator, denominator)]
    estimate = math.exp(statistics.mean(logs))
    if not logs or all(value == logs[0] for value in logs):
        return estimate, estimate, estimate

    rng = random.Random(BOOTSTRAP_SEED)
    bootstrapped: list[float] = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [logs[rng.randrange(len(logs))] for _ in logs]
        bootstrapped.append(math.exp(statistics.mean(sample)))
    bootstrapped.sort()
    return estimate, percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)


def main() -> int:
    args = parse_args()
    engines = parse_engines(args.engine)
    if args.rounds < 2 or args.warmups < 0 or args.depth < 1:
        raise ValueError("rounds must be at least 2; warmups non-negative; depth positive")

    labels = list(engines)
    samples: dict[str, list[int]] = {label: [] for label in labels}
    orders: list[list[str]] = []
    expected_tree: tuple[int, str, str, str] | None = None
    canonical_receipt: SearchReceipt | None = None

    for label, engine in engines.items():
        for _ in range(args.warmups):
            receipt = run_engine(engine, args.depth, args.timeout)
            tree = (
                receipt.nodes,
                receipt.bestmoves_sha256,
                receipt.root_scores_sha256,
                receipt.root_evals_sha256,
            )
            if expected_tree is None:
                expected_tree = tree
                canonical_receipt = receipt
            elif tree != expected_tree:
                raise RuntimeError(f"Warmup search tree differs for {label}: {tree} != {expected_tree}")

    order_rng = random.Random(ORDER_SEED)
    for round_index in range(args.rounds):
        order = labels.copy()
        order_rng.shuffle(order)
        orders.append(order)
        for label in order:
            receipt = run_engine(engines[label], args.depth, args.timeout)
            tree = (
                receipt.nodes,
                receipt.bestmoves_sha256,
                receipt.root_scores_sha256,
                receipt.root_evals_sha256,
            )
            if expected_tree is None:
                expected_tree = tree
                canonical_receipt = receipt
            elif tree != expected_tree:
                raise RuntimeError(
                    f"Search tree differs for {label} in round {round_index + 1}: "
                    f"{tree} != {expected_tree}"
                )
            samples[label].append(receipt.nps)

    medians = {label: statistics.median(values) for label, values in samples.items()}
    fastest = max(labels, key=lambda label: medians[label])
    summaries: dict[str, dict[str, float | int | bool]] = {}
    for label in labels:
        values = samples[label]
        median = medians[label]
        mad = statistics.median(abs(value - median) for value in values)
        estimate, lower, upper = paired_ratio_interval(values, samples[fastest])
        half_width = (upper - lower) / 2.0
        summaries[label] = {
            "median_nps": median,
            "mad_nps": mad,
            "ratio_to_fastest": estimate,
            "ratio_ci95_low": lower,
            "ratio_ci95_high": upper,
            "ratio_ci95_half_width": half_width,
            "precision_gate": half_width <= 0.005,
            "training_speed_gate": lower >= 0.95,
        }

    assert expected_tree is not None and canonical_receipt is not None
    result = {
        "schema": "HORDE_V2_WIDTH_BENCH_V2",
        "source_sha": os.environ.get("GITHUB_SHA", ""),
        "depth": args.depth,
        "rounds": args.rounds,
        "warmups": args.warmups,
        "order_seed": ORDER_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "nodes": expected_tree[0],
        "bestmoves_sha256": expected_tree[1],
        "root_scores_sha256": expected_tree[2],
        "root_evals_sha256": expected_tree[3],
        "root_scores": list(canonical_receipt.root_scores),
        "root_evals": list(canonical_receipt.root_evals),
        "fastest_by_median": fastest,
        "orders": orders,
        "raw_nps": samples,
        "summary": summaries,
    }

    encoded = json.dumps(result, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
