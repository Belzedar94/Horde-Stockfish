#!/usr/bin/env python3
"""Determinism, gradient reach, bucket lookup and layout checks for HordeV3Model.

Also exercises the R2 control C0SingleG0Model on the same batch, since the
single G0 table at equal lanes is the strength control the V2 ladder never ran.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import torch  # noqa: E402

import horde_bin_v1 as wire  # noqa: E402
import horde_training_decoder as dec  # noqa: E402
import horde_training_models as models  # noqa: E402


FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


@dataclass
class Batch:
    v3_global: torch.Tensor
    v2_global: torch.Tensor
    global_offsets: torch.Tensor
    v3_offsets: torch.Tensor
    side_to_move: torch.Tensor
    phase_buckets: torch.Tensor


def build_batch(records: Path, count: int) -> Batch:
    raw = records.read_bytes()
    stride = wire.RECORD_SIZE
    rows_v3: list[int] = []
    rows_g0: list[int] = []
    off_v3 = [0]
    off_g0 = [0]
    sides: list[int] = []
    wpc: list[int] = []
    for index in range(count):
        record = wire.validate_record(raw[index * stride : (index + 1) * stride], index)
        board = record["board"]
        stream = dec.v3_global_rows(board)
        g0 = dec.extract_sparse_features(board).v2_global
        rows_v3.extend(stream)
        rows_g0.extend(g0)
        off_v3.append(len(rows_v3))
        off_g0.append(len(rows_g0))
        sides.append(record["side"])
        wpc.append(record["white_count"])
    return Batch(
        v3_global=torch.tensor(rows_v3, dtype=torch.long),
        v2_global=torch.tensor(rows_g0, dtype=torch.long),
        global_offsets=torch.tensor(off_g0, dtype=torch.long),
        v3_offsets=torch.tensor(off_v3, dtype=torch.long),
        side_to_move=torch.tensor(sides, dtype=torch.long),
        phase_buckets=models.v3_phase_bucket(torch.tensor(wpc, dtype=torch.long)),
    )


class V3View:
    def __init__(self, batch: Batch) -> None:
        self.v3_global = batch.v3_global
        self.global_offsets = batch.v3_offsets
        self.side_to_move = batch.side_to_move
        self.phase_buckets = batch.phase_buckets


class G0View:
    def __init__(self, batch: Batch) -> None:
        self.v2_global = batch.v2_global
        self.global_offsets = batch.global_offsets
        self.side_to_move = batch.side_to_move


def test_phase_lookup() -> None:
    table = models.V3_PHASE_LOOKUP
    check(len(table) == 37, f"phase lookup has {len(table)} entries, not 37")
    check(min(table) == 0 and max(table) == models.V3_PHASE_BUCKETS - 1, "phase lookup range")
    check(all(b <= a for a, b in zip(table[1:], table[:-1])) is False or True, "")
    check(all(table[i] <= table[i + 1] for i in range(36)), "phase lookup is not monotone")
    check(sorted(set(table)) == list(range(8)), "phase lookup does not use every bucket")
    values = models.v3_phase_bucket(torch.arange(37))
    check(values.tolist() == list(table), "v3_phase_bucket disagrees with the table")
    for bad in (-1, 37):
        try:
            models.v3_phase_bucket(torch.tensor([bad]))
            FAILURES.append(f"phase lookup accepted white_piece_count {bad}")
        except ValueError:
            pass


def test_layout() -> None:
    model = models.HordeV3Model(seed=1)
    shapes = {name: tuple(p.shape) for name, p in model.named_parameters()}
    expected = {
        "ft_weights": (896, 1024),
        "ft_bias": (1024,),
        "psqt_weights": (896, 16),
        "hidden0_weights": (8, 16, 1024),
        "hidden0_bias": (8, 16),
        "hidden1_weights": (8, 32, 16),
        "hidden1_bias": (8, 32),
        "output_weights": (8, 2, 32),
        "output_bias": (8, 2),
    }
    check(shapes == expected, f"V3 parameter shapes drifted: {shapes}")
    floats = sum(p.numel() for p in model.parameters())
    dtype_bytes = {
        "ft_weights": 2, "ft_bias": 4, "psqt_weights": 4,
        "hidden0_weights": 1, "hidden0_bias": 4,
        "hidden1_weights": 1, "hidden1_bias": 4,
        "output_weights": 1, "output_bias": 4,
    }
    serialized = sum(
        int(torch.tensor(shape).prod()) * dtype_bytes[name] for name, shape in expected.items()
    )
    print(f"  V3 float parameters   : {floats}")
    print(f"  V3 serialized bytes   : {serialized} (plus a 37 byte phase lookup section)")
    check(floats == 1_068_944, f"V3 float parameter count is {floats}")
    check(serialized == 2_033_728, f"V3 serialized parameter bytes is {serialized}")


def test_forward(batch: Batch) -> None:
    a = models.HordeV3Model(seed=7435908571601354096)
    b = models.HordeV3Model(seed=7435908571601354096)
    view = V3View(batch)
    with torch.no_grad():
        first, second = a(view), b(view)
    check(torch.equal(first, second), "named initialization is not reproducible")
    check(first.shape == (batch.side_to_move.numel(),), "V3 output shape is wrong")
    check(bool(torch.isfinite(first).all()), "V3 produced a non-finite value")
    with torch.no_grad():
        repeat = a(view)
    check(torch.equal(first, repeat), "V3 forward is not deterministic")
    print(f"  V3 output range       : [{float(first.min()):+.4f}, {float(first.max()):+.4f}]")

    control = models.C0SingleG0Model(seed=7435908571601354096)
    with torch.no_grad():
        out = control(G0View(batch))
    check(out.shape == first.shape, "R2 control output shape is wrong")
    check(bool(torch.isfinite(out).all()), "R2 control produced a non-finite value")


def test_gradients(batch: Batch) -> None:
    model = models.HordeV3Model(seed=11)
    loss = model(V3View(batch)).square().mean()
    loss.backward()
    for group, parameters in model.gradient_groups().items():
        for parameter in parameters:
            check(parameter.grad is not None, f"group {group} has a parameter with no gradient")
            if parameter.grad is not None:
                check(
                    float(parameter.grad.abs().sum()) > 0.0,
                    f"group {group} received an all-zero gradient",
                )
    covered = {id(p) for ps in model.gradient_groups().values() for p in ps}
    check(
        covered == {id(p) for p in model.parameters()},
        "gradient_groups does not cover every V3 parameter",
    )

    control = models.C0SingleG0Model(seed=11)
    control(G0View(batch)).square().mean().backward()
    for name, parameter in control.named_parameters():
        check(
            parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0,
            f"R2 control parameter {name} received no gradient",
        )


def test_bucket_reach(batch: Batch) -> None:
    used = sorted(set(batch.phase_buckets.tolist()))
    print(f"  phase buckets present : {used}")
    check(len(used) == 8, f"batch only reached buckets {used}; widen the sample")
    model = models.HordeV3Model(seed=3)
    model(V3View(batch)).sum().backward()
    reached = (model.hidden0_weights.grad.abs().sum(dim=(1, 2)) > 0).tolist()
    check(all(reached), f"some bucket heads received no gradient: {reached}")
    psqt_columns = (model.psqt_weights.grad.abs().sum(dim=0) > 0).tolist()
    check(
        sum(psqt_columns) >= 9,
        f"only {sum(psqt_columns)} of 16 PSQT columns were reached",
    )


def main() -> int:
    records = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        r"D:/horde-train/validation-selected/selected-records.bin"
    )
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    print("V3 model checks")
    test_phase_lookup()
    test_layout()
    batch = build_batch(records, 3000)
    test_forward(batch)
    test_gradients(batch)
    test_bucket_reach(batch)
    if FAILURES:
        print(f"\nFAILED with {len(FAILURES)} problems:")
        for failure in FAILURES[:20]:
            print(f"  {failure}")
        return 1
    print("\nall V3 model checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
