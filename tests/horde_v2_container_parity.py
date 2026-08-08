#!/usr/bin/env python3
"""Independent integer parity and corruption gates for Horde V2 containers."""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from typing import Callable, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from horde_v2_container import (  # noqa: E402
    DIRECTORY_ENTRY_BYTES,
    DIRECTORY_OFFSET,
    FIRST_DOMAIN_ABSOLUTE_NONKING,
    FIRST_DOMAIN_ROYAL,
    MAX_SAFE_BIAS_MAGNITUDE,
    ContainerError,
    NetworkSpec,
    ParsedContainer,
    SPECS,
    build_container,
    parse_container,
)


TRACE_SCHEMA = "HORDE_V2_FULL_REFRESH_TRACE_V1"
VALUE_TB_WIN_IN_MAX_PLY = 31_507

PIECE_ROLES = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "p": 5,
    "n": 6,
    "b": 7,
    "r": 8,
    "q": 9,
    "k": 10,
}


@dataclass(frozen=True)
class PositionFixture:
    name: str
    board: str
    side_to_move: int
    rule50: int


FIXTURES = (
    PositionFixture(
        "start-white",
        "PPPPPPPP"
        "PPPPPPPP"
        "PPPPPPPP"
        "PPPPPPPP"
        ".PP..PP."
        "........"
        "pppppppp"
        "rnbqkbnr",
        0,
        0,
    ),
    PositionFixture(
        "start-black-r37",
        "PPPPPPPP"
        "PPPPPPPP"
        "PPPPPPPP"
        "PPPPPPPP"
        ".PP..PP."
        "........"
        "pppppppp"
        "rnbqkbnr",
        1,
        37,
    ),
    PositionFixture(
        "minimal-r100",
        "Q......."
        "........"
        "........"
        "........"
        "........"
        "........"
        "........"
        "....k...",
        0,
        100,
    ),
    PositionFixture(
        "mirror-d4",
        "........"
        "P......."
        "........"
        "...k...."
        "........"
        "..N....."
        "........"
        ".......q",
        1,
        7,
    ),
    PositionFixture(
        "mirror-e4",
        "........"
        ".......P"
        "........"
        "....k..."
        "........"
        ".....N.."
        "........"
        "q.......",
        0,
        19,
    ),
    PositionFixture(
        "mixed-promotions",
        "R..Q...."
        "....p..."
        "B......n"
        "....P..."
        ".r...b.."
        "........"
        "...q...."
        "......k.",
        1,
        51,
    ),
)


@dataclass(frozen=True)
class IntegerParameters:
    first_weights: array
    first_bias: array
    global_weights: array
    global_bias: array
    hidden0_weights: array
    hidden0_bias: array
    hidden1_weights: array
    hidden1_bias: array
    output_weights: array
    output_bias: array


def signed_array(payload: bytes, typecode: str, item_size: int) -> array:
    values = array(typecode)
    if values.itemsize != item_size:
        raise AssertionError(
            f"native array({typecode!r}) is {values.itemsize} bytes, expected {item_size}"
        )
    values.frombytes(payload)
    if sys.byteorder != "little" and item_size > 1:
        values.byteswap()
    return values


def encode_signed(values: Iterable[int], typecode: str, item_size: int) -> bytes:
    payload = array(typecode, values)
    if payload.itemsize != item_size:
        raise AssertionError(
            f"native array({typecode!r}) is {payload.itemsize} bytes, expected {item_size}"
        )
    if sys.byteorder != "little" and item_size > 1:
        payload.byteswap()
    return payload.tobytes()


def deterministic_sections(spec: NetworkSpec) -> dict[str, bytes]:
    sections: dict[str, bytes] = {}
    for section in spec.sections:
        salt = spec.schema_id * 17 + section.section_id * 101
        if section.dtype == "i8":
            radius = 7
            width = radius * 2 + 1
            sections[section.name] = bytes(
                (((index * 37 + salt) % width) - radius) & 0xFF
                for index in range(section.elements)
            )
        elif section.dtype == "i16":
            radius = 31
            width = radius * 2 + 1
            sections[section.name] = encode_signed(
                (((index * 97 + salt) % width) - radius for index in range(section.elements)),
                "h",
                2,
            )
        elif section.dtype == "i32":
            radius = 24_000
            width = radius * 2 + 1
            sections[section.name] = encode_signed(
                (((index * 193 + salt) % width) - radius for index in range(section.elements)),
                "i",
                4,
            )
        else:  # pragma: no cover - frozen schema exhaustiveness
            raise AssertionError(f"unhandled dtype {section.dtype}")
    return sections


def deterministic_provenance(spec: NetworkSpec) -> dict[str, object]:
    return {
        "checkpoint_sha256": "11" * 32,
        "container_schema": "HORDE_V2_INTEGER_NETWORK_V1",
        # Uppercase input proves that the canonical codec normalizes the Git ID.
        "source_commit": "ABCDEF0123456789" * 2 + "ABCDEF01",
        "source_dirty": False,
        "train_file_sha256": "22" * 32,
        "training_architecture_structural_sha256": spec.training_structural_sha256,
        "training_receipt_sha256": "33" * 32,
        "validation_file_sha256": "44" * 32,
        "wdl_calibration_sha256": "55" * 32,
    }


def create_synthetic_networks(root: Path) -> list[Path]:
    networks: list[Path] = []
    for spec in SPECS:
        payload, _ = build_container(
            spec,
            deterministic_sections(spec),
            deterministic_provenance(spec),
        )
        path = root / f"{spec.architecture}.hsv2"
        path.write_bytes(payload)
        networks.append(path)
    return networks


def decode_parameters(container: ParsedContainer) -> IntegerParameters:
    spec = container.spec
    return IntegerParameters(
        first_weights=signed_array(container.sections[spec.first_weight_name], "h", 2),
        first_bias=signed_array(container.sections[spec.first_bias_name], "i", 4),
        global_weights=signed_array(container.sections["global_weights"], "h", 2),
        global_bias=signed_array(container.sections["global_bias"], "i", 4),
        hidden0_weights=signed_array(container.sections["hidden0_weights"], "b", 1),
        hidden0_bias=signed_array(container.sections["hidden0_bias"], "i", 4),
        hidden1_weights=signed_array(container.sections["hidden1_weights"], "b", 1),
        hidden1_bias=signed_array(container.sections["hidden1_bias"], "i", 4),
        output_weights=signed_array(container.sections["output_weights"], "b", 1),
        output_bias=signed_array(container.sections["output_bias"], "i", 4),
    )


def sparse_indices(board: str) -> tuple[list[int], list[int], list[int]]:
    if len(board) != 64:
        raise AssertionError(f"fixture board is {len(board)} characters instead of 64")
    pieces: list[tuple[int, int]] = []
    black_kings: list[int] = []
    white_pieces = 0
    black_pieces = 0
    for square, piece in enumerate(board):
        if piece == ".":
            continue
        if piece not in PIECE_ROLES:
            raise AssertionError(f"fixture contains unregistered piece {piece!r}")
        role = PIECE_ROLES[piece]
        pieces.append((role, square))
        if piece.isupper():
            white_pieces += 1
        else:
            black_pieces += 1
        if piece == "k":
            black_kings.append(square)
    if len(black_kings) != 1 or white_pieces > 36 or black_pieces > 16:
        raise AssertionError("fixture violates the fixed Horde physical contract")

    king = black_kings[0]
    mirror = king % 8 <= 3
    canonical_king = king ^ 7 if mirror else king
    bucket = (canonical_king // 8) * 4 + canonical_king % 8 - 4
    global_rows = [role * 64 + square for role, square in pieces]
    royal_rows = [
        ((bucket * 10 + role) * 64) + (square ^ 7 if mirror else square)
        for role, square in pieces
        if role < 10
    ]
    absolute_nonking_rows = [row for row in global_rows if row < 10 * 64]
    return global_rows, royal_rows, absolute_nonking_rows


def add_sparse_rows(
    biases: array,
    weights: array,
    rows: Iterable[int],
    lanes: int,
) -> list[int]:
    accumulator = list(biases)
    for row in rows:
        offset = row * lanes
        for lane in range(lanes):
            accumulator[lane] += weights[offset + lane]
    return accumulator


def activation(values: Iterable[int]) -> list[int]:
    return [0 if value <= 0 else min(value >> 6, 127) for value in values]


def dense_layer(
    inputs: list[int], weights: array, biases: array, outputs: int
) -> tuple[list[int], list[int]]:
    affine: list[int] = []
    for output in range(outputs):
        offset = output * len(inputs)
        value = biases[output]
        for index, input_value in enumerate(inputs):
            value += weights[offset + index] * input_value
        affine.append(value)
    return affine, activation(affine)


def trunc_div(numerator: int, denominator: int) -> int:
    magnitude = abs(numerator) // denominator
    return -magnitude if numerator < 0 else magnitude


def evaluate_fixture(
    container: ParsedContainer,
    parameters: IntegerParameters,
    fixture: PositionFixture,
) -> dict[str, object]:
    global_rows, royal_rows, absolute_rows = sparse_indices(fixture.board)
    if container.spec.first_domain_code == FIRST_DOMAIN_ROYAL:
        first_rows = royal_rows
    elif container.spec.first_domain_code == FIRST_DOMAIN_ABSOLUTE_NONKING:
        first_rows = absolute_rows
    else:  # pragma: no cover - codec already rejects unknown domains
        raise AssertionError("unknown first-domain code")

    first_accumulator = add_sparse_rows(
        parameters.first_bias, parameters.first_weights, first_rows, 64
    )
    global_accumulator = add_sparse_rows(
        parameters.global_bias, parameters.global_weights, global_rows, 192
    )
    transformed = activation(first_accumulator) + activation(global_accumulator)
    hidden0_affine, hidden0 = dense_layer(
        transformed, parameters.hidden0_weights, parameters.hidden0_bias, 32
    )
    hidden1_affine, hidden1 = dense_layer(
        hidden0, parameters.hidden1_weights, parameters.hidden1_bias, 32
    )
    head = fixture.side_to_move
    output_offset = head * 32
    output_affine = parameters.output_bias[head]
    for index, value in enumerate(hidden1):
        output_affine += parameters.output_weights[output_offset + index] * value
    pre_rule50 = trunc_div(output_affine, 16)
    rule50 = min(max(fixture.rule50, 0), 100)
    value = trunc_div(pre_rule50 * (100 - rule50), 100)
    value = min(max(value, -VALUE_TB_WIN_IN_MAX_PLY + 1), VALUE_TB_WIN_IN_MAX_PLY - 1)
    return {
        "name": fixture.name,
        "first_accumulator": first_accumulator,
        "global_accumulator": global_accumulator,
        "transformed": transformed,
        "hidden0_affine": hidden0_affine,
        "hidden0": hidden0,
        "hidden1_affine": hidden1_affine,
        "hidden1": hidden1,
        "output_affine": output_affine,
        "pre_rule50": pre_rule50,
        "value": value,
    }


def expected_trace(path: Path) -> dict[str, object]:
    parsed = parse_container(path.read_bytes())
    parameters = decode_parameters(parsed)
    positions = [evaluate_fixture(parsed, parameters, fixture) for fixture in FIXTURES]
    for key in (
        "first_accumulator",
        "global_accumulator",
        "transformed",
        "hidden0_affine",
        "hidden0",
        "hidden1_affine",
        "hidden1",
    ):
        if positions[0][key] != positions[1][key]:
            raise AssertionError(f"side-to-move changed the shared trunk at {key}")
    if positions[2]["value"] != 0:
        raise AssertionError("rule50=100 did not damp the minimal fixture to zero")
    mirrored_first_equal = (
        positions[3]["first_accumulator"] == positions[4]["first_accumulator"]
    )
    if mirrored_first_equal != (
        parsed.spec.first_domain_code == FIRST_DOMAIN_ROYAL
    ):
        raise AssertionError("mirrored fixtures contradict the registered first domain")
    if positions[3]["global_accumulator"] == positions[4]["global_accumulator"]:
        raise AssertionError("absolute Global domain collapsed mirrored fixtures")

    return {
        "schema": TRACE_SCHEMA,
        "network_schema": parsed.spec.schema_name,
        "file_sha256": parsed.file_sha256,
        "parameter_sha256": parsed.parameter_sha256,
        "positions": positions,
    }


def run_trace(oracle: Path, network: Path) -> dict[str, object]:
    completed = subprocess.run(
        [str(oracle.resolve()), "--trace", str(network.resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    backend = result.pop("backend", None)
    if backend not in {"scalar", "avx2"}:
        raise AssertionError(f"oracle returned unknown backend {backend!r}")
    return result


def trace_digest(trace: Mapping[str, object]) -> str:
    payload = json.dumps(
        trace, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def compare_network(oracles: list[Path], network: Path) -> tuple[str, str]:
    expected = expected_trace(network)
    for oracle in oracles:
        actual = run_trace(oracle, network)
        if actual != expected:
            raise AssertionError(
                f"Python/C++ trace mismatch for {network.name} via {oracle.name}"
            )
    return str(expected["network_schema"]), trace_digest(expected)


def changed(payload: bytes, mutate: Callable[[bytearray], None]) -> bytes:
    result = bytearray(payload)
    mutate(result)
    if result == payload:
        raise AssertionError("corruption helper did not change the container")
    return bytes(result)


def parameter_range_corruption(payload: bytes) -> bytes:
    result = bytearray(payload)
    parameter_offset = struct.unpack_from("<Q", result, 128)[0]
    parameter_bytes = struct.unpack_from("<Q", result, 136)[0]
    bias_entry = DIRECTORY_OFFSET + DIRECTORY_ENTRY_BYTES
    bias_offset = struct.unpack_from("<Q", result, bias_entry + 12)[0]
    bias_bytes = struct.unpack_from("<Q", result, bias_entry + 20)[0]
    struct.pack_into("<i", result, parameter_offset + bias_offset, MAX_SAFE_BIAS_MAGNITUDE + 1)
    bias_payload = result[
        parameter_offset + bias_offset : parameter_offset + bias_offset + bias_bytes
    ]
    result[bias_entry + 28 : bias_entry + 60] = hashlib.sha256(bias_payload).digest()
    parameter_payload = result[parameter_offset : parameter_offset + parameter_bytes]
    result[144:176] = hashlib.sha256(parameter_payload).digest()
    return bytes(result)


def expect_python_rejection(payload: bytes, name: str) -> None:
    try:
        parse_container(payload)
    except ContainerError:
        return
    raise AssertionError(f"Python codec accepted malformed container {name}")


def expect_cpp_rejection(
    oracle: Path, path: Path, expected_error: str, name: str
) -> None:
    completed = subprocess.run(
        [str(oracle.resolve()), "--validate", str(path.resolve())],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        raise AssertionError(f"C++ loader accepted malformed container {name}")
    if expected_error not in completed.stderr:
        raise AssertionError(
            f"C++ loader classified {name} unexpectedly: {completed.stderr.strip()}"
        )


def verify_corruptions(oracle: Path, network: Path, root: Path) -> int:
    payload = network.read_bytes()

    def mutate_u32(offset: int, value: int) -> Callable[[bytearray], None]:
        return lambda data: struct.pack_into("<I", data, offset, value)

    cases: list[tuple[str, bytes, str]] = [
        ("truncated", payload[:1000], "TRUNCATED"),
        ("magic", changed(payload, lambda data: data.__setitem__(0, data[0] ^ 1)), "MAGIC_MISMATCH"),
        ("file-length", changed(payload, mutate_u32(24, len(payload) + 1)), "HEADER_MISMATCH"),
        (
            "offset-overflow",
            changed(payload, lambda data: struct.pack_into("<Q", data, 40, (1 << 64) - 1)),
            "HEADER_MISMATCH",
        ),
        ("schema-id", changed(payload, mutate_u32(16, 0xFFFFFFFF)), "SCHEMA_MISMATCH"),
        ("structure-hash", changed(payload, lambda data: data.__setitem__(48, data[48] ^ 1)), "STRUCTURE_MISMATCH"),
        ("provenance-identity", changed(payload, lambda data: data.__setitem__(272, data[272] ^ 1)), "PROVENANCE_MISMATCH"),
        ("dirty-source", changed(payload, lambda data: data.__setitem__(452, 1)), "PROVENANCE_MISMATCH"),
        ("reserved-header", changed(payload, lambda data: data.__setitem__(532, 1)), "SCHEMA_MISMATCH"),
        ("directory-offset", changed(payload, lambda data: struct.pack_into("<Q", data, DIRECTORY_OFFSET + 12, 1)), "DIRECTORY_MISMATCH"),
        ("payload-byte", changed(payload, lambda data: data.__setitem__(-1, data[-1] ^ 1)), "PAYLOAD_MISMATCH"),
        ("unsafe-bias", parameter_range_corruption(payload), "PARAMETER_RANGE"),
    ]
    for name, corrupted, expected_error in cases:
        expect_python_rejection(corrupted, name)
        path = root / f"malformed-{name}.hsv2"
        path.write_bytes(corrupted)
        expect_cpp_rejection(oracle, path, expected_error, name)
    return len(cases)


def verify_builder_bias_rejection() -> None:
    spec = next(item for item in SPECS if item.first_domain_code == FIRST_DOMAIN_ABSOLUTE_NONKING)
    sections = deterministic_sections(spec)
    bad_bias = bytearray(sections[spec.first_bias_name])
    struct.pack_into("<i", bad_bias, 0, MAX_SAFE_BIAS_MAGNITUDE + 1)
    sections[spec.first_bias_name] = bytes(bad_bias)
    try:
        build_container(spec, sections, deterministic_provenance(spec))
    except ContainerError:
        return
    raise AssertionError("container builder accepted an unsafe bias")


def verify_position_oracles(oracles: Iterable[Path], networks: Iterable[Path]) -> int:
    network_args = [str(path) for path in networks]
    count = 0
    for oracle in oracles:
        completed = subprocess.run(
            [str(oracle), *network_args],
            check=True,
            capture_output=True,
            text=True,
        )
        expected = f"containers={len(network_args)}"
        if expected not in completed.stdout:
            raise AssertionError(
                f"position oracle {oracle} did not report {expected}: {completed.stdout!r}"
            )
        count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle",
        action="append",
        type=Path,
        required=True,
        help="compiled C++ scalar or SIMD container oracle; repeat for each backend",
    )
    parser.add_argument(
        "--network",
        action="append",
        type=Path,
        default=[],
        help="additional trained container to verify after the synthetic fixtures",
    )
    parser.add_argument(
        "--position-oracle",
        action="append",
        type=Path,
        default=[],
        help="real Position stack oracle; repeat for each scalar or SIMD build",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    oracles = [path.expanduser().resolve() for path in args.oracle]
    for oracle in oracles:
        if not oracle.is_file():
            raise FileNotFoundError(f"container oracle does not exist: {oracle}")
    requested_networks = [path.expanduser().resolve() for path in args.network]
    for network in requested_networks:
        if not network.is_file():
            raise FileNotFoundError(f"container network does not exist: {network}")
    position_oracles = [path.expanduser().resolve() for path in args.position_oracle]
    for oracle in position_oracles:
        if not oracle.is_file():
            raise FileNotFoundError(f"position oracle does not exist: {oracle}")

    with tempfile.TemporaryDirectory(prefix="horde-v2-container-") as temporary:
        root = Path(temporary)
        synthetic = create_synthetic_networks(root)
        receipts = [compare_network(oracles, network) for network in synthetic]
        receipts.extend(compare_network(oracles, network) for network in requested_networks)
        position_oracle_count = verify_position_oracles(
            position_oracles, [*synthetic, *requested_networks]
        )
        corruptions = verify_corruptions(oracles[0], synthetic[-1], root)
        verify_builder_bias_rejection()

    print(
        "Horde V2 integer container parity passed: "
        + ", ".join(f"{schema}={digest[:12]}" for schema, digest in receipts)
        + f", oracles={len(oracles)}, position_oracles={position_oracle_count}, "
        + f"corruptions={corruptions}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContainerError, FileNotFoundError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
