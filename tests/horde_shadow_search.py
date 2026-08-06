#!/usr/bin/env python3
"""Run isolated counterfactual searches for Horde pruning experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


GENFENS_PREFIX = "info string genfens "
EXPERIMENTS = {
    1: "disable_nmp",
    2: "disable_probcut",
    4: "disable_lmp",
    8: "disable_node_futility",
    16: "disable_capture_futility",
    32: "disable_capture_see",
    64: "disable_quiet_history",
    128: "disable_quiet_futility",
    256: "disable_quiet_see",
    512: "disable_qsearch_pruning",
    1024: "disable_lmr",
    2048: "disable_razoring",
}


def generate_positions(engine: Path, count: int, seed: int) -> list[str]:
    completed = subprocess.run(
        [str(engine)],
        input=(
            f"genfens {count} seed {seed} book None minplies=6 maxplies=12\n"
            "quit\n"
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"genfens failed:\n{completed.stdout[-4000:]}")
    positions = [
        line[len(GENFENS_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(GENFENS_PREFIX)
    ]
    if len(positions) != count:
        raise RuntimeError(f"expected {count} shadow positions, got {len(positions)}")
    return positions


class SearchSession:

    def __init__(self, engine: Path, experiment_mask: int):
        self.process = subprocess.Popen(
            [str(engine)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        uci = self.read_until("uciok")
        expected = (
            "option name HordeSearchExperimentMask type spin "
            "default 0 min 0 max 4095"
        )
        if not any(expected in line for line in uci):
            self.close(force=True)
            raise RuntimeError("engine does not expose the shadow-search mask")
        self.send("setoption name Threads value 1")
        self.send("setoption name Hash value 16")
        self.send("setoption name HordeSearchTelemetry value false")
        self.send(f"setoption name HordeSearchExperimentMask value {experiment_mask}")
        self.send("isready")
        self.read_until("readyok")

    def send(self, command: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("engine stdin is unavailable")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def read_until(self, marker: str) -> list[str]:
        if self.process.stdout is None:
            raise RuntimeError("engine stdout is unavailable")
        lines = []
        while True:
            line = self.process.stdout.readline()
            if line == "":
                raise RuntimeError(
                    f"engine closed before {marker!r}:\n" + "\n".join(lines[-50:])
                )
            line = line.rstrip("\r\n")
            lines.append(line)
            if marker in line:
                return lines

    def search(self, fen: str, depth: int) -> dict:
        self.send("ucinewgame")
        self.send("setoption name Clear Hash")
        self.send("isready")
        self.read_until("readyok")
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        lines = self.read_until("bestmove ")
        depth_lines = [line for line in lines if line.startswith(f"info depth {depth} ")]
        if not depth_lines:
            raise RuntimeError(f"engine emitted no depth-{depth} receipt for {fen}")
        receipt = depth_lines[-1]
        score = re.search(r"\bscore (cp|mate) (-?[0-9]+)\b", receipt)
        nodes = re.search(r"\bnodes ([0-9]+)\b", receipt)
        bestmove = next(line.split()[1] for line in reversed(lines) if line.startswith("bestmove "))
        if not score or not nodes:
            raise RuntimeError(f"incomplete search receipt: {receipt}")
        return {
            "bestmove": bestmove,
            "score_type": score.group(1),
            "score": int(score.group(2)),
            "nodes": int(nodes.group(1)),
        }

    def close(self, force: bool = False) -> None:
        if self.process.poll() is not None:
            return
        if force:
            self.process.kill()
        else:
            self.send("quit")
        self.process.wait(timeout=30)

    def __enter__(self) -> "SearchSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close(force=exc is not None)


def run_batch(engine: Path, positions: list[str], depth: int, mask: int) -> list[dict]:
    with SearchSession(engine, mask) as session:
        return [session.search(fen, depth) for fen in positions]


def parse_masks(values: list[int] | None) -> list[int]:
    masks = sorted(EXPERIMENTS if values is None else set(values))
    unknown = [mask for mask in masks if mask not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown single-bit experiment masks: {unknown}")
    return masks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--reference-depth", type=int, default=6)
    parser.add_argument("--masks", type=int, nargs="*")
    args = parser.parse_args()

    engine = args.engine.resolve()
    if not engine.is_file():
        raise SystemExit(f"Engine not found: {engine}")
    if not 1 <= args.positions <= 100000:
        raise ValueError("positions must be between 1 and 100000")
    if not 1 <= args.depth < args.reference_depth <= 64:
        raise ValueError("depths must satisfy 1 <= depth < reference-depth <= 64")
    masks = parse_masks(args.masks)
    positions = generate_positions(engine, args.positions, args.seed)
    positions_sha256 = hashlib.sha256(
        ("\n".join(positions) + "\n").encode("ascii")
    ).hexdigest()

    baseline = run_batch(engine, positions, args.depth, 0)
    baseline_repeat = run_batch(engine, positions, args.depth, 0)
    if baseline != baseline_repeat:
        raise RuntimeError("experiment mask zero is not deterministic")
    reference = run_batch(engine, positions, args.reference_depth, 0)

    experiments = []
    for mask in masks:
        shadow = run_batch(engine, positions, args.depth, mask)
        changed = 0
        reference_aligned = 0
        records = []
        for index, (base, alternative, deeper) in enumerate(
            zip(baseline, shadow, reference, strict=True)
        ):
            differs = (base["bestmove"], base["score_type"], base["score"]) != (
                alternative["bestmove"],
                alternative["score_type"],
                alternative["score"],
            )
            move_changed = base["bestmove"] != alternative["bestmove"]
            aligns = (
                move_changed
                and alternative["bestmove"] == deeper["bestmove"]
                and base["bestmove"] != deeper["bestmove"]
            )
            changed += differs
            reference_aligned += aligns
            if differs:
                records.append(
                    {
                        "position": index,
                        "baseline": base,
                        "shadow": alternative,
                        "reference": deeper,
                        "reference_aligned": aligns,
                    }
                )
        experiments.append(
            {
                "mask": mask,
                "name": EXPERIMENTS[mask],
                "changed": changed,
                "false_prune_candidates": reference_aligned,
                "records": records,
            }
        )

    receipt = {
        "schema": "horde-shadow-search-v1",
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "seed": args.seed,
        "positions": len(positions),
        "positions_sha256": positions_sha256,
        "depth": args.depth,
        "reference_depth": args.reference_depth,
        "experiment_map": {str(bit): name for bit, name in EXPERIMENTS.items()},
        "mask_zero_deterministic": True,
        "experiments": experiments,
    }
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
