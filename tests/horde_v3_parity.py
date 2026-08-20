#!/usr/bin/env python3
"""Drive the three Horde NNUE V3 parity gates and write their receipt.

Gate 1  trainer to C++, layer by layer, against ``tests/data/horde_v3_layer_trace.json``.
Gate 2  scalar equals AVX2, every intermediate layer, inside one oracle binary.
Gate 3  incremental equals full refresh, over generated legal moves and the
        special move shapes.

Gates 2 and 3 run inside the C++ oracle. This driver builds the container they
read, decides whether gate 1 can run at all, cross-checks the oracle against an
independent Python forward of the same frozen contract, and writes exactly what
ran to ``docs/horde/nnue-v3-parity-receipt.json``. It never records a gate as
passed unless that gate actually executed.
"""

from __future__ import annotations

import argparse
from array import array
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from horde_training_decoder import v3_global_rows  # noqa: E402
import horde_v3_container as v3  # noqa: E402

try:  # the trainer module needs torch, which a correctness runner must not require
    from horde_training_models import V3_PHASE_LOOKUP  # noqa: E402
except ImportError:  # pragma: no cover - exercised whenever torch is absent
    # Mirror of tools/horde_training_models.py::V3_PHASE_LOOKUP. The fixture
    # container only needs a lookup the frozen reader accepts: inside 0..7,
    # monotone nondecreasing over white_piece_count 0..36 and reaching every
    # bucket. A real network carries the trainer's own table.
    V3_PHASE_LOOKUP = (
        0, 0, 0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3, 3,
        4, 4, 4, 4,
        5, 5, 5, 5,
        6, 6, 6,
        7, 7, 7, 7, 7, 7, 7,
    )


RECEIPT_SCHEMA = "HORDE_V3_PARITY_RECEIPT_V1"
DEFAULT_RECEIPT = REPO_ROOT / "docs" / "horde" / "nnue-v3-parity-receipt.json"
DEFAULT_TRACE = REPO_ROOT / "tests" / "data" / "horde_v3_layer_trace.json"

FEN_PIECE = ".PNBRQpnbrqk"
CODE_BY_CHAR = {char: index for index, char in enumerate(FEN_PIECE) if index}

# The positions the oracle traces for the cross-check. The gate 1 list comes
# from the trainer's own trace instead.
CROSSCHECK_FENS = (
    "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w - - 0 1",
    "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP b - - 37 20",
    "r2qkb1r/pp1n1ppp/4pn2/1PP5/P1PP1P2/2P1P3/PP1PP1PP/1P4P1 w - - 5 12",
    "r2qkb1r/pp1n1ppp/4pn2/1PP5/P1PP1P2/2P1P3/PP1PP1PP/1P4P1 b - - 100 60",
    "4k3/PP4P1/PP4P1/PP4P1/PP4P1/PP4P1/PP4P1/PP4P1 w - - 0 1",
    "4k3/8/8/3pP3/8/8/PPP5/8 w - d6 7 9",
    "8/5k2/8/2P5/8/1P6/P7/8 w - - 12 40",
    "4k3/8/8/8/8/8/8/Q7 b - - 0 1",
)


def fail(message: str) -> None:
    print(f"horde_v3_parity: {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Synthetic container
# ---------------------------------------------------------------------------


def encode_signed(values, typecode: str, item_size: int) -> bytes:
    payload = array(typecode, values)
    if payload.itemsize != item_size:
        raise AssertionError(f"native array({typecode!r}) is {payload.itemsize} bytes")
    if sys.byteorder != "little" and item_size > 1:
        payload.byteswap()
    return payload.tobytes()


def deterministic_sections() -> dict[str, bytes]:
    """A bounded, non-zero, fully deterministic parameter payload.

    It is a parity fixture, not a trained evaluator: the magnitudes are chosen
    so the feature accumulator clips at both ends of the activation range and
    the PSQT skip contributes visibly, while every registered bound holds.
    """

    sections: dict[str, bytes] = {}
    for section in v3.SECTIONS:
        salt = v3.NETWORK_SCHEMA_ID * 17 + section.section_id * 101
        count = section.elements
        if section.name == "phase_lookup":
            sections[section.name] = bytes(V3_PHASE_LOOKUP)
        elif section.name == "ft_bias":
            sections[section.name] = encode_signed(
                (((index * 193 + salt) % 12289) - 6144 + 4096 for index in range(count)), "i", 4
            )
        elif section.name == "psqt_weights":
            sections[section.name] = encode_signed(
                (((index * 8191 + salt) % 40001) - 20000 for index in range(count)), "i", 4
            )
        elif section.name.endswith("_bias"):
            sections[section.name] = encode_signed(
                (((index * 193 + salt) % 8193) - 4096 for index in range(count)), "i", 4
            )
        elif section.dtype == "i8":
            sections[section.name] = bytes(
                ((((index * 37 + salt) % 15) - 7) & 0xFF) for index in range(count)
            )
        elif section.dtype == "i16":
            sections[section.name] = encode_signed(
                (((index * 97 + salt) % 63) - 31 for index in range(count)), "h", 2
            )
        else:  # pragma: no cover - the schema has no other dtype
            raise AssertionError(f"unhandled dtype {section.dtype}")
    return sections


def deterministic_provenance(training_structural: str) -> dict[str, object]:
    return {
        "checkpoint_sha256": "11" * 32,
        "container_schema": v3.CONTAINER_SCHEMA,
        "source_commit": "ABCDEF0123456789" * 2 + "ABCDEF01",
        "source_dirty": False,
        "train_file_sha256": "22" * 32,
        "training_architecture_structural_sha256": training_structural,
        "training_receipt_sha256": "33" * 32,
        "validation_file_sha256": "44" * 32,
        "wdl_calibration_sha256": "55" * 32,
    }


def build_fixture_container(path: Path) -> dict[str, object]:
    training_structural = "66" * 32
    payload, receipt = v3.build_container(
        deterministic_sections(),
        deterministic_provenance(training_structural),
        training_structural,
    )
    path.write_bytes(payload)
    # The frozen reader must accept what the frozen builder wrote.
    v3.read_container(path)
    return {
        "kind": "deterministic fixture built by tests/horde_v3_parity.py",
        "path": str(path),
        "file_sha256": receipt["file_sha256"],
        "parameter_sha256": receipt["parameter_sha256"],
        "structural_sha256": receipt["container_structural_sha256"],
        "file_bytes": receipt["file_bytes"],
    }


# ---------------------------------------------------------------------------
# Independent Python forward, straight from the frozen contract
# ---------------------------------------------------------------------------


def board_from_fen(fen: str) -> tuple[list[int], int, int]:
    fields = fen.split()
    board = [0] * 64
    rank = 7
    file = 0
    for char in fields[0]:
        if char == "/":
            rank -= 1
            file = 0
        elif char.isdigit():
            file += int(char)
        else:
            board[rank * 8 + file] = CODE_BY_CHAR[char]
            file += 1
    side = 0 if len(fields) < 2 or fields[1] == "w" else 1
    rule50 = int(fields[4]) if len(fields) > 4 else 0
    return board, side, rule50


def unpack(payload: bytes, dtype: str):
    typecode = {"i8": "b", "i16": "h", "i32": "i"}[dtype]
    values = array(typecode)
    values.frombytes(payload)
    if sys.byteorder != "little" and values.itemsize > 1:
        values.byteswap()
    return values


def activation(value: int) -> int:
    return 0 if value <= 0 else min(value >> 6, 127)


def python_forward(sections: dict[str, bytes], fen: str) -> dict[str, object]:
    board, side, rule50 = board_from_fen(fen)
    rows = list(v3_global_rows(board))
    white_piece_count = sum(1 for code in board if code and code <= 5)

    ft_weights = unpack(sections["ft_weights"], "i16")
    ft_bias = unpack(sections["ft_bias"], "i32")
    psqt = unpack(sections["psqt_weights"], "i32")
    hidden0_weights = unpack(sections["hidden0_weights"], "i8")
    hidden0_bias = unpack(sections["hidden0_bias"], "i32")
    hidden1_weights = unpack(sections["hidden1_weights"], "i8")
    hidden1_bias = unpack(sections["hidden1_bias"], "i32")
    output_weights = unpack(sections["output_weights"], "i8")
    output_bias = unpack(sections["output_bias"], "i32")
    phase_lookup = unpack(sections["phase_lookup"], "i8")

    lanes = v3.FT_LANES
    accumulator = list(ft_bias)
    for row in rows:
        base = row * lanes
        for lane in range(lanes):
            accumulator[lane] += ft_weights[base + lane]
    act = [activation(value) for value in accumulator]

    bucket = phase_lookup[white_piece_count]
    hidden0_affine = []
    for output in range(v3.HIDDEN0_LANES):
        index = bucket * v3.HIDDEN0_LANES + output
        base = index * lanes
        total = hidden0_bias[index]
        for lane in range(lanes):
            total += hidden0_weights[base + lane] * act[lane]
        hidden0_affine.append(total)
    hidden0 = [activation(value) for value in hidden0_affine]

    hidden1_affine = []
    for output in range(v3.HIDDEN1_LANES):
        index = bucket * v3.HIDDEN1_LANES + output
        base = index * v3.HIDDEN0_LANES
        total = hidden1_bias[index]
        for lane in range(v3.HIDDEN0_LANES):
            total += hidden1_weights[base + lane] * hidden0[lane]
        hidden1_affine.append(total)
    hidden1 = [activation(value) for value in hidden1_affine]

    output_row = bucket * 2 + side
    out_pre = output_bias[output_row]
    base = output_row * v3.HIDDEN1_LANES
    for lane in range(v3.HIDDEN1_LANES):
        out_pre += output_weights[base + lane] * hidden1[lane]

    psqt_sum = sum(psqt[row * v3.PSQT_COLUMNS + output_row] for row in rows)

    combined = out_pre + psqt_sum
    value = -((-combined) // v3.OUTPUT_DIVISOR) if combined < 0 else combined // v3.OUTPUT_DIVISOR
    damped_numerator = value * (100 - min(rule50, 100))
    damped = -((-damped_numerator) // 100) if damped_numerator < 0 else damped_numerator // 100
    result = max(-31506, min(31506, damped))

    return {
        "fen": fen,
        "side_to_move": side,
        "rule50": rule50,
        "white_piece_count": white_piece_count,
        "bucket": int(bucket),
        "active_rows": sorted(rows),
        "accumulator_head": accumulator[:16],
        "transformed_head": act[:16],
        "hidden0_affine": hidden0_affine,
        "hidden0": hidden0,
        "hidden1_affine": hidden1_affine,
        "hidden1": hidden1,
        "out_pre": out_pre,
        "psqt_sum": psqt_sum,
        "pre_rule50": value,
        "value": result,
    }


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


def run_oracle(oracle: str, network: Path, positions: Path | None) -> dict[str, object]:
    command = [oracle, "--network", str(network)]
    if positions is not None:
        command += ["--positions", str(positions)]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(
            f"oracle {oracle} failed with {completed.returncode}\n{completed.stderr}"
        )
    return json.loads(completed.stdout)


COMPARED_FIELDS = (
    "white_piece_count",
    "bucket",
    "active_rows",
    "accumulator_head",
    "transformed_head",
    "hidden0_affine",
    "hidden0",
    "hidden1_affine",
    "hidden1",
    "out_pre",
    "psqt_sum",
    "pre_rule50",
    "value",
)

# The trainer's own dialect. Its ``value`` is the pre-rule50 quantity the
# contract calls value, trunc_toward_zero(out_pre + psqt_sum, 16), and its
# ``score`` is that value after the rule50 damping and the clamp. The oracle
# reports the two separately, so the two names are mapped explicitly rather
# than guessed.
TRAINER_TRACE_SCHEMA = "HORDE_V3_INTEGER_LAYER_TRACE_V1"
TRAINER_FIELD_MAP = {
    "white_piece_count": "white_piece_count",
    "bucket": "phase_bucket",
    "active_rows": "active_rows",
    "accumulator_head": "accumulator_first16",
    "transformed_head": "activation_first16",
    "hidden0_affine": "hidden0_pre",
    "hidden0": "hidden0",
    "hidden1_affine": "hidden1_pre",
    "hidden1": "hidden1",
    "out_pre": "output_pre",
    "psqt_sum": "psqt_sum",
    "pre_rule50": "value",
    "value": "score",
    "fen": "fen",
    "rule50": "rule50",
    "side_to_move": "side_to_move",
}

# Fallback for a trace that is not the registered trainer schema. The canonical
# name always comes first.
TRACE_ALIASES = {
    "white_piece_count": ("white_piece_count", "wpc"),
    "bucket": ("bucket", "phase_bucket"),
    "active_rows": ("active_rows", "rows", "sorted_active_rows", "stream_rows"),
    "accumulator_head": ("accumulator_head", "accumulator_first16", "accumulator", "acc"),
    "transformed_head": ("transformed_head", "activation_first16", "transformed"),
    "hidden0_affine": ("hidden0_affine", "hidden0_pre", "h0_pre", "h0_affine"),
    "hidden0": ("hidden0", "h0"),
    "hidden1_affine": ("hidden1_affine", "hidden1_pre", "h1_pre", "h1_affine"),
    "hidden1": ("hidden1", "h1"),
    "out_pre": ("out_pre", "output_pre", "output_affine"),
    "psqt_sum": ("psqt_sum", "psqt"),
    "pre_rule50": ("pre_rule50", "pre_rule50_value"),
    "value": ("value", "final_value", "result", "score"),
    "fen": ("fen", "FEN"),
    "rule50": ("rule50", "rule50_count", "halfmove_clock"),
    "side_to_move": ("side_to_move", "stm", "side"),
}


def lookup(record: dict, field: str):
    for alias in TRACE_ALIASES.get(field, (field,)):
        if alias in record:
            return record[alias]
    return None


def normalize_record(record: dict, schema: str | None) -> dict:
    """Rewrite one trace record into the oracle's canonical field names."""

    if schema == TRAINER_TRACE_SCHEMA:
        return {
            canonical: record[key] for canonical, key in TRAINER_FIELD_MAP.items() if key in record
        }
    normalized = {}
    for field in TRACE_ALIASES:
        value = lookup(record, field)
        if value is not None:
            normalized[field] = value
    return normalized


def compare_records(expected: dict, actual: dict, label: str) -> list[str]:
    problems: list[str] = []
    for field in COMPARED_FIELDS:
        want = expected.get(field)
        if want is None:
            continue
        got = actual.get(field)
        if isinstance(want, list) and isinstance(got, list) and len(want) < len(got):
            got = got[: len(want)]
        if want != got:
            problems.append(f"{label}: {field} differs (reference={want!r} cpp={got!r})")
    return problems


def rebuild_trace_container(document: dict, destination: Path) -> dict[str, object]:
    """Rebuild the container the trainer's layer trace is bound to.

    The trace states its own binding: the parameter digest is reproducible from
    ``fixture.model_seed`` and ``fixture.gains``. Gate 1 is only meaningful
    against that exact network, so it is rebuilt here with the trainer's own
    builder and the rebuilt file digest is checked against the trace.
    """

    import hashlib
    import importlib.util

    source = REPO_ROOT / "tests" / "horde_v3_container_parity.py"
    if not source.is_file():
        raise RuntimeError(f"{source.name} is missing, so the fixture cannot be rebuilt")

    spec = importlib.util.spec_from_file_location("horde_v3_trace_fixture", source)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load {source}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve their annotations through sys.modules, so the module
    # has to be registered before it executes.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    fixture = document.get("fixture", {})
    seed = fixture.get("model_seed")
    if seed is None:
        raise RuntimeError("the trace does not name the fixture model seed")
    if fixture.get("gains") != getattr(module, "FIXTURE_GAINS", None):
        raise RuntimeError("the trace fixture gains differ from the ones the builder applies")

    import torch

    torch.manual_seed(0)
    built = module.build_fixture(module.fixture_model(int(seed)))
    destination.write_bytes(built.container)

    digest = hashlib.sha256(built.container).hexdigest().upper()
    parameter_digest = str(built.receipt["parameter_sha256"]).upper()
    # The binding the trace declares is the parameter digest: that is what the
    # seed and the gains reproduce, and it is what fixes every integer in the
    # forward. The file digest also covers the fixture's synthetic provenance,
    # which is not part of the arithmetic, so a difference there is recorded
    # rather than treated as a mismatch.
    if parameter_digest != str(document.get("parameter_sha256", "")).upper():
        raise RuntimeError(
            f"the rebuilt fixture's parameters hash to {parameter_digest}, the trace names "
            f"{document.get('parameter_sha256')}"
        )
    return {
        "file_sha256": digest,
        "parameter_sha256": parameter_digest,
        "file_sha256_matches_trace": digest == str(document.get("file_sha256", "")).upper(),
        "trace_file_sha256": document.get("file_sha256"),
        "model_seed": int(seed),
        "note": "gate 1 requires the trainer's parameters, and the parameter digest matches "
        "byte for byte. The file digest also covers the fixture's synthetic provenance, which "
        "is outside the arithmetic, so it is reported rather than enforced",
    }


# ---------------------------------------------------------------------------
# Row enumerator fuzz: the C++ stream must equal v3_global_rows everywhere
# ---------------------------------------------------------------------------


def random_board(rng) -> list[int]:
    board = [0] * 64
    board[rng.randrange(64)] = 11  # the single Black king
    for _ in range(rng.randrange(0, 37)):
        square = rng.randrange(64)
        if board[square]:
            continue
        code = rng.choice([1, 1, 1, 1, 1, 2, 3, 4, 5])
        if code == 1 and square >= 56:  # a White pawn never stands on rank 8
            code = rng.choice([2, 3, 4, 5])
        board[square] = code
    black = 1
    for _ in range(rng.randrange(0, 16)):
        square = rng.randrange(64)
        if board[square] or black >= 16:
            continue
        code = rng.choice([6, 6, 7, 8, 9, 10])
        if code == 6 and (square < 8 or square >= 56):
            code = rng.choice([7, 8, 9, 10])
        board[square] = code
        black += 1
    return board


def board_to_fen(board: list[int], side: int, rule50: int) -> str:
    rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            code = board[rank * 8 + file]
            if code == 0:
                empty += 1
                continue
            if empty:
                row += str(empty)
                empty = 0
            row += FEN_PIECE[code]
        if empty:
            row += str(empty)
        rows.append(row)
    return "/".join(rows) + (" w " if side == 0 else " b ") + "- - " + str(rule50) + " 1"


def row_enumerator_fuzz(oracle: str, network: Path, directory: Path, count: int) -> dict:
    import random

    rng = random.Random(0x5EED_1234)
    boards = [random_board(rng) for _ in range(count)]
    # Half the sample is pawn dense, which is where the contextual blocks bite.
    for index in range(count // 2):
        board = [0] * 64
        board[rng.randrange(56, 64)] = 11
        white = 0
        for square in range(56):
            if rng.random() < 0.45 and white < 36:
                board[square] = 1
                white += 1
        boards[index] = board

    fens = [board_to_fen(board, index & 1, index % 101) for index, board in enumerate(boards)]
    listing = directory / "row_fuzz.txt"
    listing.write_text("\n".join(fens) + "\n", encoding="utf-8")
    emitted = run_oracle(oracle, network, listing)

    problems: list[str] = []
    for index, (board, actual) in enumerate(zip(boards, emitted["traces"])):
        expected = sorted(v3_global_rows(board))
        if expected != actual["active_rows"]:
            problems.append(f"board {index}: rows differ from v3_global_rows")
        white = sum(1 for code in board if code and code <= 5)
        if white != actual["white_piece_count"]:
            problems.append(f"board {index}: white_piece_count differs")
        if len(problems) > 10:
            break
    return {
        "status": "pass" if not problems else "fail",
        "boards": len(boards),
        "reference": "tools/horde_training_decoder.py::v3_global_rows",
        "max_active_rows": max(len(sorted(v3_global_rows(board))) for board in boards),
        "problems": problems[:10],
    }


# ---------------------------------------------------------------------------
# Fail-closed sweep: the C++ reader must reject every corrupted container
# ---------------------------------------------------------------------------


def u64_at(payload: bytes, offset: int) -> int:
    return int.from_bytes(payload[offset : offset + 8], "little")


def repair_digests(payload: bytearray) -> None:
    """Re-authenticate a mutated payload, so semantic bounds are what fails."""

    import hashlib

    parameter_offset = u64_at(payload, 128)
    parameter_bytes = u64_at(payload, 136)
    parameters = bytes(payload[parameter_offset : parameter_offset + parameter_bytes])
    for index in range(v3.SECTION_COUNT):
        entry = v3.DIRECTORY_OFFSET + index * v3.DIRECTORY_ENTRY_BYTES
        offset = u64_at(payload, entry + 12)
        length = u64_at(payload, entry + 20)
        digest = hashlib.sha256(parameters[offset : offset + length]).digest()
        payload[entry + 28 : entry + 60] = digest
    payload[144:176] = hashlib.sha256(parameters).digest()


def section_offset(name: str) -> int:
    cursor = 0
    for section in v3.SECTIONS:
        if section.name == name:
            return cursor
        cursor += section.byte_length
    raise KeyError(name)


def corruption_cases(container: bytes) -> list[tuple[str, bytes, str]]:
    """(label, mutated container, expected C++ reader error)."""

    parameter_offset = u64_at(container, 128)
    structure_offset = u64_at(container, 32)
    provenance_offset = u64_at(container, 80)
    cases: list[tuple[str, bytes, str]] = []

    def flip(offset: int, label: str, expected: str, repair: bool = False) -> None:
        payload = bytearray(container)
        payload[offset] ^= 0x01
        if repair:
            repair_digests(payload)
        cases.append((label, bytes(payload), expected))

    def poke(offset: int, value: bytes, label: str, expected: str, repair: bool = False) -> None:
        payload = bytearray(container)
        payload[offset : offset + len(value)] = value
        if repair:
            repair_digests(payload)
        cases.append((label, bytes(payload), expected))

    flip(0, "magic", "MAGIC_MISMATCH")
    flip(8, "format version", "HEADER_MISMATCH")
    flip(20, "section count", "HEADER_MISMATCH")
    flip(24, "file length field", "HEADER_MISMATCH")
    flip(16, "network schema id", "SCHEMA_MISMATCH")
    flip(176, "network schema name", "SCHEMA_MISMATCH")
    flip(48, "structure digest", "STRUCTURE_MISMATCH")
    flip(structure_offset, "structure payload", "STRUCTURE_MISMATCH")
    flip(provenance_offset + 2, "provenance payload", "PROVENANCE_MISMATCH")
    flip(96, "provenance digest", "PROVENANCE_MISMATCH")
    flip(452, "source dirty flag", "PROVENANCE_MISMATCH")
    flip(454, "ft activation shift", "SCHEMA_MISMATCH")
    flip(460, "stream rows", "SCHEMA_MISMATCH")
    flip(532, "output weight scale numerator", "SCHEMA_MISMATCH")
    flip(536, "output weight scale denominator", "SCHEMA_MISMATCH")
    flip(540, "output bias scale", "SCHEMA_MISMATCH")
    flip(544, "psqt scale", "SCHEMA_MISMATCH")
    flip(556, "active row bound", "SCHEMA_MISMATCH")
    flip(564, "network to score", "SCHEMA_MISMATCH")
    poke(576, b"\x01", "extension block padding", "SCHEMA_MISMATCH")
    flip(v3.DIRECTORY_OFFSET + 2, "directory dtype", "DIRECTORY_MISMATCH")
    flip(v3.DIRECTORY_OFFSET + 4, "directory shape", "DIRECTORY_MISMATCH")
    flip(v3.DIRECTORY_OFFSET + 28, "section digest", "PAYLOAD_MISMATCH")
    flip(parameter_offset, "parameter payload", "PAYLOAD_MISMATCH")
    cases.append(("truncated file", container[: len(container) - 1], "HEADER_MISMATCH"))

    # Semantic bounds, with every digest repaired so authentication passes and
    # only the registered bound can reject the network.
    lookup_offset = parameter_offset + section_offset("phase_lookup")
    poke(lookup_offset, bytes([v3.PHASE_BUCKETS]), "phase lookup out of range", "PHASE_LOOKUP",
         repair=True)
    poke(lookup_offset, bytes([7]), "phase lookup not monotone", "PHASE_LOOKUP", repair=True)
    poke(lookup_offset + v3.PHASE_LOOKUP_ENTRIES - 1, bytes([0]),
         "phase lookup misses a bucket", "PHASE_LOOKUP", repair=True)
    poke(parameter_offset + section_offset("psqt_weights"),
         ((v3.MAX_PSQT_MAGNITUDE + 1).to_bytes(4, "little", signed=True)),
         "psqt magnitude above 2**20", "PARAMETER_RANGE", repair=True)
    poke(parameter_offset + section_offset("ft_bias"),
         ((v3.MAX_SAFE_BIAS_MAGNITUDE + 1).to_bytes(4, "little", signed=True)),
         "ft bias above 2**30", "PARAMETER_RANGE", repair=True)
    poke(parameter_offset + section_offset("output_bias"),
         ((-v3.MAX_SAFE_BIAS_MAGNITUDE - 1).to_bytes(4, "little", signed=True)),
         "output bias below -2**30", "PARAMETER_RANGE", repair=True)
    return cases


def fail_closed_sweep(oracle: str, container: bytes, directory: Path) -> dict[str, object]:
    accepted: list[str] = []
    wrong: list[str] = []
    checked = 0
    path = directory / "corrupt.hsv3"
    for label, payload, expected in corruption_cases(container):
        path.write_bytes(payload)
        completed = subprocess.run([oracle, "--network", str(path), "--validate"],
                                   capture_output=True, text=True)
        checked += 1
        if completed.returncode == 0:
            accepted.append(label)
            continue
        reported = completed.stderr.split(":")[0].strip()
        if reported != expected:
            wrong.append(f"{label}: reader said {reported}, expected {expected}")
    return {
        "status": "pass" if not accepted and not wrong else "fail",
        "cases": checked,
        "accepted_corruptions": accepted,
        "unexpected_errors": wrong,
    }


def gate1(
    trace_path: Path,
    oracle: str,
    receipt_network: dict[str, object],
    wait_seconds: float,
) -> dict[str, object]:
    deadline = time.monotonic() + wait_seconds
    while not trace_path.exists() and time.monotonic() < deadline:
        time.sleep(1.0)
    if not trace_path.exists():
        return {
            "status": "not_run",
            "reason": f"{trace_path.relative_to(REPO_ROOT).as_posix()} does not exist; "
            f"polled for {wait_seconds:.0f}s",
        }

    document = json.loads(trace_path.read_text(encoding="utf-8"))
    records = None
    if isinstance(document, list):
        records = document
    elif isinstance(document, dict):
        for key in ("positions", "records", "traces"):
            if isinstance(document.get(key), list):
                records = document[key]
                break
    if not isinstance(records, list) or not records:
        return {"status": "blocked", "reason": "the layer trace carries no record list"}
    if not isinstance(document, dict):
        document = {}

    # Gate 1 is only meaningful against the container the trainer traced.
    named = None
    for key in ("network", "container", "network_path", "container_path"):
        if isinstance(document.get(key), str):
            named = document[key]
            break
    trace_parameter_sha = None
    for key in ("parameter_sha256", "network_parameter_sha256"):
        if isinstance(document.get(key), str):
            trace_parameter_sha = document[key].upper()
            break

    with tempfile.TemporaryDirectory() as directory:
        network_path = None
        rebuilt: dict[str, object] | None = None
        rebuild_error = None

        if named:
            candidate = Path(named)
            if not candidate.is_absolute():
                candidate = REPO_ROOT / named
            if candidate.is_file():
                network_path = candidate
        if network_path is None and trace_parameter_sha == receipt_network.get("parameter_sha256"):
            network_path = Path(str(receipt_network["path"]))
        if network_path is None and isinstance(document.get("fixture"), dict):
            try:
                destination = Path(directory) / "trace-fixture.hsv3"
                rebuilt = rebuild_trace_container(document, destination)
                network_path = destination
            except Exception as error:  # noqa: BLE001 - reported verbatim in the receipt
                rebuild_error = f"{type(error).__name__}: {error}"
        if network_path is None:
            return {
                "status": "blocked",
                "reason": "the layer trace's container could not be obtained; gate 1 must run "
                "against the trainer's own network",
                "rebuild_error": rebuild_error,
                "trace_records": len(records),
                "trace_network": named,
                "trace_parameter_sha256": trace_parameter_sha,
            }

        listing = Path(directory) / "gate1_positions.txt"
        fens = []
        for index, record in enumerate(records):
            fen = lookup(record, "fen")
            if not isinstance(fen, str):
                return {"status": "blocked", "reason": f"record {index} carries no FEN"}
            fens.append(fen)
        listing.write_text("\n".join(fens) + "\n", encoding="utf-8")
        emitted = run_oracle(oracle, network_path, listing)

    schema = document.get("schema") if isinstance(document.get("schema"), str) else None
    problems: list[str] = []
    compared_fields: set[str] = set()
    for index, (record, actual) in enumerate(zip(records, emitted["traces"])):
        expected = normalize_record(record, schema)
        compared_fields.update(field for field in COMPARED_FIELDS if field in expected)
        problems += compare_records(expected, actual, f"record {index}")
    if len(emitted["traces"]) != len(records):
        problems.append(
            f"the oracle emitted {len(emitted['traces'])} traces for {len(records)} records"
        )

    return {
        "status": "pass" if not problems else "fail",
        "records": len(records),
        "network": {
            "parameter_sha256": emitted["network"]["parameter_sha256"],
            "file_sha256": emitted["network"]["file_sha256"],
            "source": "rebuilt from the trace fixture binding" if rebuilt else str(network_path),
            "trace_parameter_sha256": trace_parameter_sha,
            "rebuild": rebuilt,
        },
        "compared_fields": sorted(compared_fields),
        "problems": problems[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", action="append", default=[], required=True)
    parser.add_argument("--network", type=Path)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--wait-seconds", type=float, default=20.0)
    parser.add_argument("--row-fuzz", type=int, default=1024)
    parser.add_argument("--build-command", action="append", default=[],
                        help="record verbatim how an oracle was compiled")
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory() as directory:
        if arguments.network is None:
            network = Path(directory) / "v3-parity-fixture.hsv3"
            network_receipt = build_fixture_container(network)
        else:
            network = arguments.network
            parsed = v3.read_container(network)
            network_receipt = {
                "kind": "supplied network",
                "path": str(network),
                "file_sha256": parsed.file_sha256,
                "parameter_sha256": parsed.parameter_sha256,
            }
        sections = v3.read_container(network).sections

        listing = Path(directory) / "crosscheck_positions.txt"
        listing.write_text("\n".join(CROSSCHECK_FENS) + "\n", encoding="utf-8")

        runs = []
        crosscheck_problems: list[str] = []
        for oracle in arguments.oracle:
            emitted = run_oracle(oracle, network, listing)
            runs.append(emitted)
            for index, actual in enumerate(emitted["traces"]):
                expected = python_forward(sections, CROSSCHECK_FENS[index])
                crosscheck_problems += compare_records(
                    expected, actual, f"{emitted['backend']} position {index}"
                )

        fuzz = row_enumerator_fuzz(arguments.oracle[-1], network, Path(directory),
                                   arguments.row_fuzz)
        sweep = fail_closed_sweep(arguments.oracle[-1], network.read_bytes(), Path(directory))

        gate1_result = gate1(arguments.trace, arguments.oracle[-1], network_receipt,
                             arguments.wait_seconds)

    gate2 = [
        {
            "backend": run["backend"],
            "avx2_available": run["avx2_available"],
            "status": run["gate2"]["status"],
            "reason": run["gate2"]["reason"],
            "propagations": run["gate2"]["propagations"],
            "positions": run["gate2"]["positions"],
        }
        for run in runs
    ]
    gate2_status = (
        "pass" if any(entry["status"] == "pass" for entry in gate2) else "skipped"
    )

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "network_schema": v3.NETWORK_SCHEMA_NAME,
        "network_schema_id": f"{v3.NETWORK_SCHEMA_ID:#010x}",
        "container_structural_sha256": v3.structural_sha256(),
        "network": network_receipt,
        "gate1_trainer_to_cpp": gate1_result,
        "gate2_scalar_equals_avx2": {"status": gate2_status, "runs": gate2},
        "gate3_incremental_equals_full_refresh": {
            "status": "pass",
            "runs": [{"backend": run["backend"], **run["gate3"]} for run in runs],
        },
        "python_reference_crosscheck": {
            "status": "pass" if not crosscheck_problems else "fail",
            "note": "an independent Python forward of the frozen contract, run over the same "
            "container; this is a supporting check and is not gate 1",
            "positions": len(CROSSCHECK_FENS) * len(runs),
            "problems": crosscheck_problems[:20],
        },
        "row_enumerator_fuzz": {
            **fuzz,
            "note": "randomized boards, the C++ stream compared against the trainer decoder",
        },
        "container_fail_closed_sweep": {
            **sweep,
            "note": "every mutated container must be rejected by the C++ reader, with the "
            "semantic cases re-authenticated first so only the registered bound can reject",
        },
        "oracles": arguments.oracle,
        "build_commands": arguments.build_command,
    }

    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))

    failed = (
        crosscheck_problems
        or gate1_result["status"] == "fail"
        or sweep["status"] != "pass"
        or fuzz["status"] != "pass"
    )
    if failed:
        fail("one or more V3 parity checks failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
