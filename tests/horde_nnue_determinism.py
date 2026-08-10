#!/usr/bin/env python3
"""Check raw Run 6B determinism across binaries and thread counts."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from horde_nnue_parity import corpus_digest, generate_positions, raw_eval_batch


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw Horde NNUE values across architectures and thread counts"
    )
    parser.add_argument("--engine", type=Path, action="append", required=True)
    parser.add_argument(
        "--network",
        type=Path,
        default=ROOT / "networks" / "hordetest_run6b_e37_l06.nnue",
    )
    parser.add_argument("--positions", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=2_500)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x6B37_06)
    parser.add_argument("--threads", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engines = [engine.resolve() for engine in args.engine]
    network = args.network.resolve()
    for path in (*engines, network):
        if not path.is_file():
            raise SystemExit(f"file not found: {path}")
    if any(count < 1 for count in args.threads):
        raise SystemExit("thread counts must be positive")

    positions, _coverage = generate_positions(args.positions, args.seed)
    print(
        f"determinism corpus positions={len(positions)} "
        f"seed=0x{args.seed:X} sha256={corpus_digest(positions)}",
        flush=True,
    )

    reference: list[tuple[int, int, int]] | None = None
    reference_label = ""
    with tempfile.TemporaryDirectory(prefix="horde-nnue-determinism-") as temp_name:
        cwd = Path(temp_name)
        for engine in engines:
            for thread_count in args.threads:
                values: list[tuple[int, int, int]] = []
                prefix = (
                    "uci",
                    f"setoption name EvalFile value {network}",
                    f"setoption name Threads value {thread_count}",
                    "isready",
                )
                for start in range(0, len(positions), args.batch_size):
                    batch = positions[start : start + args.batch_size]
                    values.extend(
                        raw_eval_batch(
                            engine,
                            prefix,
                            batch,
                            False,
                            cwd,
                            args.timeout,
                        )
                    )

                label = f"{engine.name}@T{thread_count}"
                if reference is None:
                    reference = values
                    reference_label = label
                elif values != reference:
                    for index, (expected, actual) in enumerate(zip(reference, values, strict=True)):
                        if expected != actual:
                            raise RuntimeError(
                                f"determinism mismatch at position {index}: "
                                f"{reference_label}={expected}, {label}={actual}, "
                                f"fen={positions[index]}"
                            )
                    raise RuntimeError(f"determinism length mismatch for {label}")
                print(f"deterministic {label}: {len(values)} positions", flush=True)

    print("Horde raw NNUE determinism completed successfully", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
