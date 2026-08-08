#!/usr/bin/env python3
"""Independent trainer-side integer receipt for Horde V2_BASE_P0."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Iterable


MASK64 = (1 << 64) - 1
SCALAR_FIXTURE_SEED = 0x4856325F42415345
ROYAL_LANES = 256
GLOBAL_LANES = 256
TRANSFORMED_LANES = ROYAL_LANES + GLOBAL_LANES
HIDDEN0_LANES = 32
HIDDEN1_LANES = 32
ROYAL_DIMENSIONS = 20_480
GLOBAL_DIMENSIONS = 704
PARAMETER_BYTES = 10_865_992
ACTIVATION_SHIFT = 6
ACTIVATION_MAX = 127
OUTPUT_SCALE = 16


class SplitMix64:
    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return value ^ (value >> 31)

    def bounded_signed(self, radius: int) -> int:
        return self.next() % (2 * radius + 1) - radius


def clipped_activation(value: int) -> int:
    return 0 if value <= 0 else min(value >> ACTIVATION_SHIFT, ACTIVATION_MAX)


def trunc_div(numerator: int, denominator: int) -> int:
    magnitude = abs(numerator) // denominator
    return -magnitude if numerator < 0 else magnitude


def start_feature_indices() -> tuple[list[int], list[int]]:
    # Fixed roles: HP,HN,HB,HR,HQ,RP,RN,RB,RR,RQ,RK.
    pieces: list[tuple[int, int]] = []
    pieces.extend((0, square) for square in range(32))
    pieces.extend((0, square) for square in (33, 34, 37, 38))
    pieces.extend((5, square) for square in range(48, 56))
    pieces.extend(
        (
            (8, 56),
            (6, 57),
            (7, 58),
            (9, 59),
            (10, 60),
            (7, 61),
            (6, 62),
            (8, 63),
        )
    )

    global_indices = [role * 64 + square for role, square in pieces]

    # The Black king is on e8: no mirror, canonical Royal bucket 7*4 = 28.
    royal_bucket = 28
    royal_indices = [
        ((royal_bucket * 10 + role) * 64) + square
        for role, square in pieces
        if role != 10
    ]

    assert len(global_indices) == 52
    assert len(royal_indices) == 51
    assert len(set(global_indices)) == len(global_indices)
    assert len(set(royal_indices)) == len(royal_indices)
    assert all(0 <= index < GLOBAL_DIMENSIONS for index in global_indices)
    assert all(0 <= index < ROYAL_DIMENSIONS for index in royal_indices)
    return global_indices, royal_indices


def sparse_transform(
    rng: SplitMix64,
    dimensions: int,
    lanes: int,
    active_indices: Iterable[int],
    biases: list[int],
) -> list[int]:
    active = set(active_indices)
    accumulator = biases.copy()
    for feature in range(dimensions):
        selected = feature in active
        for lane in range(lanes):
            weight = rng.bounded_signed(31)
            if selected:
                accumulator[lane] += weight
    return accumulator


def dense_layer(
    rng: SplitMix64,
    inputs: list[int],
    biases: list[int],
    outputs: int,
) -> tuple[list[int], list[int]]:
    affine: list[int] = []
    activated: list[int] = []
    for output in range(outputs):
        value = biases[output]
        for input_value in inputs:
            value += rng.bounded_signed(7) * input_value
        affine.append(value)
        activated.append(clipped_activation(value))
    return affine, activated


def expected_receipt() -> dict[str, object]:
    rng = SplitMix64(SCALAR_FIXTURE_SEED)
    global_indices, royal_indices = start_feature_indices()

    royal_bias = [rng.bounded_signed(6144) + 4096 for _ in range(ROYAL_LANES)]
    royal_accumulator = sparse_transform(
        rng, ROYAL_DIMENSIONS, ROYAL_LANES, royal_indices, royal_bias
    )

    global_bias = [rng.bounded_signed(6144) + 4096 for _ in range(GLOBAL_LANES)]
    global_accumulator = sparse_transform(
        rng, GLOBAL_DIMENSIONS, GLOBAL_LANES, global_indices, global_bias
    )

    transformed = [clipped_activation(value) for value in royal_accumulator]
    transformed.extend(clipped_activation(value) for value in global_accumulator)
    assert len(transformed) == TRANSFORMED_LANES

    hidden0_bias = [rng.bounded_signed(4096) for _ in range(HIDDEN0_LANES)]
    hidden0_affine, hidden0 = dense_layer(
        rng, transformed, hidden0_bias, HIDDEN0_LANES
    )

    hidden1_bias = [rng.bounded_signed(4096) for _ in range(HIDDEN1_LANES)]
    hidden1_affine, hidden1 = dense_layer(rng, hidden0, hidden1_bias, HIDDEN1_LANES)

    output_bias = [rng.bounded_signed(4096) for _ in range(2)]
    output_affine: list[int] = []
    for head in range(2):
        value = output_bias[head]
        for input_value in hidden1:
            value += rng.bounded_signed(7) * input_value
        output_affine.append(value)

    pre_rule50 = [trunc_div(value, OUTPUT_SCALE) for value in output_affine]
    return {
        "seed": SCALAR_FIXTURE_SEED,
        "parameter_bytes": PARAMETER_BYTES,
        "royal_accumulator": royal_accumulator,
        "global_accumulator": global_accumulator,
        "transformed": transformed,
        "hidden0_affine": hidden0_affine,
        "hidden0": hidden0,
        "hidden1_affine": hidden1_affine,
        "hidden1": hidden1,
        "white_output_affine": output_affine[0],
        "black_output_affine": output_affine[1],
        "white_pre_rule50": pre_rule50[0],
        "black_pre_rule50": pre_rule50[1],
        "white_value": pre_rule50[0],
        "black_value": pre_rule50[1],
    }


def first_difference(expected: object, actual: object) -> str:
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"length {len(expected)} != {len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            if left != right:
                return f"index {index}: {left} != {right}"
        return "unknown list mismatch"
    return f"{expected!r} != {actual!r}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path, help="compiled C++ V2 contract test")
    args = parser.parse_args()

    started = time.perf_counter()
    completed = subprocess.run(
        [str(args.oracle.resolve()), "--scalar-receipt"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(completed.stdout)
    expected = expected_receipt()

    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        raise AssertionError(f"receipt keys differ: missing={missing}, extra={extra}")

    for key, expected_value in expected.items():
        actual_value = actual[key]
        if actual_value != expected_value:
            raise AssertionError(
                f"trainer/C++ mismatch in {key}: "
                f"{first_difference(expected_value, actual_value)}"
            )

    elapsed = time.perf_counter() - started
    print(
        "Horde V2 trainer/C++ integer parity passed: "
        f"{len(expected)} fields, P0={expected['white_pre_rule50']}/"
        f"{expected['black_pre_rule50']}, {elapsed:.2f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
