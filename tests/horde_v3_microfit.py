#!/usr/bin/env python3
"""Deterministic CPU micro-fit for the Horde NNUE V3 topology and the R2 control.

This is an engineering receipt, not a strength-training entry point.  It proves
three things and nothing else: that the V3 decoder path feeds a batch the model
accepts, that two independent runs of the same fit are bit identical, and that
a gradient reaches every named parameter group, every one of the eight phase
bucket heads, and both PSQT columns of every bucket.  It ranks no architecture
and makes no strength claim; the same fit is run for the R2 single G0 control
only so the two share one plumbing receipt, never so they can be compared.

The fixture is drawn from the authenticated corpus when it is present and from
``horde_training_decoder.synthetic_record_payload`` when it is not, so the suite
runs anywhere.  It is balanced by construction: eight records for each of the
sixteen (phase bucket, side to move) cells, which is what makes the per-bucket
PSQT column claim meaningful.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import sys
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import torch  # noqa: E402
from torch import nn  # noqa: E402

import horde_bin_v1 as wire  # noqa: E402
import horde_training_control as control  # noqa: E402
import horde_training_decoder as dec  # noqa: E402
import horde_training_models as models  # noqa: E402


SCHEMA = "HORDE_V3_MICROFIT_V1"
V3_ARCHITECTURE = "v3-g1024-pawn-wpc8"
CONTROL_ARCHITECTURE = "v2-c0-g0single-256"
DEFAULT_CORPUS = Path(r"D:/horde-train/validation-selected/selected-records.bin")
RECEIPT_PATH = Path(__file__).resolve().parents[1] / "docs" / "horde" / "nnue-v3-microfit-receipt.json"

MODEL_SEED = 0x56335F4D4943524F
FIXTURE_SEED = 20260820
RECORDS_PER_CELL = 8
CORPUS_SCAN_LIMIT = 4000
SYNTHETIC_POOL = 2400
STEPS = 24
LEARNING_RATE = 1.5e-2
LAMBDA = control.DEFAULT_LAMBDA
MINIMUM_LOSS_DECREASE = 5.0e-4
# A frozen unit-slope Davidson link.  The production calibration is fitted from
# a specific dataset and would make this receipt depend on the corpus; a micro
# fit only needs a fixed, finite, side-symmetric link.
FIXTURE_CALIBRATION = {
    "white_to_move": (1.0, 0.0, 0.0),
    "black_to_move": (1.0, 0.0, 0.0),
}

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


class FixtureDataset:
    """Minimal random-access dataset so the trainer's own loader can be used."""

    def __init__(self, records: Sequence[dec.TrainingRecord]) -> None:
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def record(self, index: int) -> dec.TrainingRecord:
        return self.records[index]


def _flip_side(record: dec.TrainingRecord) -> dec.TrainingRecord:
    """Re-key one record into the opposite side-to-move frame.

    Score and result are stored from the side-to-move perspective, so both are
    negated with the frame and the ply parity follows the side.  Used only to
    fill a (bucket, side) cell the source could not fill, which on the synthetic
    payload is every black-to-move cell.
    """

    step = 1 if record.side_to_move == dec.WHITE or record.game_ply == 0 else -1
    return dataclasses.replace(
        record,
        side_to_move=1 - record.side_to_move,
        game_ply=record.game_ply + step,
        score=-record.score,
        result=-record.result,
    )


def build_fixture(path: Path | None) -> tuple[list[dec.TrainingRecord], dict[str, object]]:
    """Select eight records for each of the sixteen bucket-by-side cells."""

    if path is not None and path.exists():
        payload = path.read_bytes()
        available = len(payload) // wire.RECORD_SIZE
        scanned = min(available, CORPUS_SCAN_LIMIT)
        source = {
            "kind": "authenticated HORDE_BIN_V1 records",
            "name": path.name,
            "available_records": available,
            "scanned_records": scanned,
        }
    else:
        payload = dec.synthetic_record_payload(SYNTHETIC_POOL, seed=FIXTURE_SEED)
        scanned = SYNTHETIC_POOL
        source = {
            "kind": "deterministic synthetic HORDE_BIN_V1 payload",
            "name": "horde_training_decoder.synthetic_record_payload",
            "available_records": SYNTHETIC_POOL,
            "scanned_records": scanned,
            "seed": FIXTURE_SEED,
        }

    cells: dict[tuple[int, int], list[dec.TrainingRecord]] = {
        (bucket, side): [] for bucket in range(models.V3_PHASE_BUCKETS) for side in (0, 1)
    }
    spare: dict[int, list[dec.TrainingRecord]] = {
        bucket: [] for bucket in range(models.V3_PHASE_BUCKETS)
    }
    for index in range(scanned):
        raw = payload[index * wire.RECORD_SIZE : (index + 1) * wire.RECORD_SIZE]
        board = wire.decode_board(raw)
        white = sum(1 for code in board if 1 <= code <= 5)
        if white > 36:
            continue
        bucket = models.V3_PHASE_LOOKUP[white]
        side = raw[32]
        if len(cells[(bucket, side)]) < RECORDS_PER_CELL:
            cells[(bucket, side)].append(dec.decode_training_record(raw, index))
        elif len(spare[bucket]) < RECORDS_PER_CELL:
            spare[bucket].append(dec.decode_training_record(raw, index))
        if all(len(bank) >= RECORDS_PER_CELL for bank in cells.values()):
            break

    derived = 0
    for (bucket, side), bank in cells.items():
        while len(bank) < RECORDS_PER_CELL:
            pool = cells[(bucket, 1 - side)] + spare[bucket]
            if not pool:
                raise RuntimeError(f"no source record reaches phase bucket {bucket}")
            donor = pool[len(bank) % len(pool)]
            if donor.side_to_move == side:
                bank.append(donor)
            else:
                bank.append(_flip_side(donor))
                derived += 1

    records = [record for key in sorted(cells) for record in cells[key]]
    observed_buckets: dict[str, int] = {}
    observed_sides: dict[str, int] = {}
    for record in records:
        white = sum(1 for code in record.board if 1 <= code <= 5)
        bucket = str(models.V3_PHASE_LOOKUP[white])
        observed_buckets[bucket] = observed_buckets.get(bucket, 0) + 1
        side = str(record.side_to_move)
        observed_sides[side] = observed_sides.get(side, 0) + 1

    digest = hashlib.sha256()
    for record in records:
        digest.update(bytes(record.board))
        digest.update(
            struct.pack(
                "<BHHhbB",
                record.side_to_move,
                record.rule50_count,
                record.game_ply,
                record.score,
                record.result,
                record.outcome_reason,
            )
        )
    receipt = {
        "source": source,
        "records": len(records),
        "records_per_cell": RECORDS_PER_CELL,
        "records_per_phase_bucket": dict(sorted(observed_buckets.items())),
        "records_per_side_to_move": dict(sorted(observed_sides.items())),
        "cells_covered": len(cells),
        "derived_side_records": derived,
        "derived_side_policy": (
            "same board in the opposite side-to-move frame; score, result and "
            "ply parity follow the frame"
        ),
        "sha256": digest.hexdigest().upper(),
    }
    return records, receipt


def _loss_sha256(losses: Sequence[float]) -> str:
    digest = hashlib.sha256()
    for loss in losses:
        digest.update(struct.pack("<d", loss))
    return digest.hexdigest().upper()


def _bucket_reach(model: nn.Module) -> dict[str, list[bool]]:
    """Per-bucket gradient reach for every layer stack of the V3 model."""

    def per_bucket(gradient: torch.Tensor) -> list[bool]:
        flat = gradient.reshape(gradient.shape[0], -1)
        return [bool(value) for value in (flat.abs().sum(dim=1) > 0.0).tolist()]

    psqt = model.psqt_weights.grad.abs().sum(dim=0) > 0.0
    return {
        "hidden0_weights": per_bucket(model.hidden0_weights.grad),
        "hidden0_bias": per_bucket(model.hidden0_bias.grad),
        "hidden1_weights": per_bucket(model.hidden1_weights.grad),
        "hidden1_bias": per_bucket(model.hidden1_bias.grad),
        "output_weights": per_bucket(model.output_weights.grad),
        "output_bias": per_bucket(model.output_bias.grad),
        "psqt_columns": [bool(value) for value in psqt.tolist()],
    }


def fit_once(
    architecture: str,
    batch: Any,
    calibration: control.DavidsonCalibration,
) -> dict[str, object]:
    model = control._make_model(architecture, MODEL_SEED)
    initial_state = control._state_sha256(model)
    optimizer = control._make_optimizer(model, LEARNING_RATE)
    losses: list[float] = []
    gradient_norms: dict[str, float] | None = None
    reach: dict[str, list[bool]] | None = None

    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        composite, *_ = control.loss_terms(model(batch), batch, LAMBDA, calibration)
        loss = composite.mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{architecture} micro-fit loss became non-finite at step {step}")
        losses.append(float(loss.detach().to(torch.float64).item()))
        loss.backward()
        if step == 0:
            # _gradient_norms itself refuses a missing or all-zero group.
            gradient_norms = control._gradient_norms(model)
            if architecture in control.V3_ARCHITECTURES:
                reach = _bucket_reach(model)
        optimizer.step()
        control._clip_serialized_dense_weights(model)
        if not control._all_finite(model):
            raise RuntimeError(f"{architecture} micro-fit produced non-finite parameters")

    with torch.no_grad():
        composite, *_ = control.loss_terms(model(batch), batch, LAMBDA, calibration)
        losses.append(float(composite.mean().to(torch.float64).item()))
    assert gradient_norms is not None

    quarter = max(1, STEPS // 4)
    receipt: dict[str, object] = {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_state_sha256": initial_state,
        "final_state_sha256": control._state_sha256(model),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_reduction": losses[0] - losses[-1],
        "loss_monotone": all(
            losses[index + 1] <= losses[index] for index in range(len(losses) - 1)
        ),
        "head_window_mean_loss": sum(losses[:quarter]) / quarter,
        "tail_window_mean_loss": sum(losses[-quarter:]) / quarter,
        "losses": losses,
        "loss_sha256": _loss_sha256(losses),
        "first_step_gradient_l2": gradient_norms,
        "gradient_groups_exercised": sorted(gradient_norms),
    }
    if reach is not None:
        receipt["bucket_gradient_reach"] = reach
    return receipt


def model_receipt(
    name: str,
    architecture: str,
    batch: Any,
    calibration: control.DavidsonCalibration,
) -> dict[str, object]:
    first = fit_once(architecture, batch, calibration)
    second = fit_once(architecture, batch, calibration)
    check(
        first["final_state_sha256"] == second["final_state_sha256"],
        f"{name} repeated micro-fits do not end in a byte-identical state",
    )
    check(
        first["losses"] == second["losses"],
        f"{name} repeated micro-fits produced different per-step losses",
    )
    check(first == second, f"{name} repeated micro-fit receipts differ")

    check(
        float(first["final_loss"]) < float(first["initial_loss"]),
        f"{name} micro-fit did not reduce the loss: "
        f"{first['initial_loss']} -> {first['final_loss']}",
    )
    check(
        float(first["loss_reduction"]) >= MINIMUM_LOSS_DECREASE,
        f"{name} micro-fit loss reduction {first['loss_reduction']} is below "
        f"{MINIMUM_LOSS_DECREASE}",
    )
    check(
        float(first["tail_window_mean_loss"]) < float(first["head_window_mean_loss"]),
        f"{name} micro-fit tail window did not improve on its head window",
    )
    norms = first["first_step_gradient_l2"]
    assert isinstance(norms, dict)
    check(
        all(math.isfinite(norm) and norm > 0.0 for norm in norms.values()),
        f"{name} micro-fit has a group without a gradient: {norms}",
    )

    reach = first.get("bucket_gradient_reach")
    if reach is not None:
        assert isinstance(reach, dict)
        for layer, flags in reach.items():
            expected = (
                2 * models.V3_PHASE_BUCKETS
                if layer == "psqt_columns"
                else models.V3_PHASE_BUCKETS
            )
            check(len(flags) == expected, f"{name} {layer} reach has {len(flags)} entries")
            missing = [index for index, value in enumerate(flags) if not value]
            check(not missing, f"{name} {layer} received no gradient at {missing}")

    return {
        "name": name,
        "architecture": architecture,
        "schema": control._architecture_schema(architecture),
        "structural_sha256": control._architecture_structure(architecture)["structural_sha256"],
        "training_schema": control._training_schema(architecture),
        "checkpoint_schema": control._checkpoint_schema(architecture),
        "repeat_count": 2,
        "repeat_exact": True,
        **first,
    }


def build_receipt(records_path: Path | None) -> dict[str, object]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    torch.set_float32_matmul_precision("highest")

    device = torch.device("cpu")
    calibration = control._torch_calibration(FIXTURE_CALIBRATION, device)
    records, fixture = build_fixture(records_path)
    dataset = FixtureDataset(records)
    indices = tuple(range(len(records)))

    sparse = {
        architecture: control._load_sparse_batch(architecture, dataset, indices)
        for architecture in (V3_ARCHITECTURE, CONTROL_ARCHITECTURE)
    }
    physical = sum(sparse[V3_ARCHITECTURE].physical_piece_count)
    contextual = len(sparse[V3_ARCHITECTURE].v2_global) - physical
    check(contextual > 0, "the V3 batch carries no contextual rows")
    check(
        len(sparse[CONTROL_ARCHITECTURE].v2_global) == physical,
        "the R2 control batch was built with a contextual tail",
    )
    fixture["g0_rows"] = physical
    fixture["contextual_rows"] = contextual
    batches = {
        architecture: control._model_batch(architecture, sparse[architecture], device)
        for architecture in sparse
    }
    check(
        isinstance(batches[V3_ARCHITECTURE], control.V3Batch),
        "the V3 architecture did not receive a V3Batch",
    )
    check(
        isinstance(batches[CONTROL_ARCHITECTURE], control.V2Batch),
        "the R2 control did not receive a V2Batch",
    )
    observed_buckets = sorted(set(batches[V3_ARCHITECTURE].phase_buckets.tolist()))
    check(
        observed_buckets == list(range(models.V3_PHASE_BUCKETS)),
        f"the fixture only reached phase buckets {observed_buckets}",
    )
    observed_sides = sorted(set(batches[V3_ARCHITECTURE].side_to_move.tolist()))
    check(observed_sides == [0, 1], f"the fixture only reached sides {observed_sides}")

    fitted = [
        model_receipt("v3-g1024", V3_ARCHITECTURE, batches[V3_ARCHITECTURE], calibration),
        model_receipt(
            "r2-g0single-256",
            CONTROL_ARCHITECTURE,
            batches[CONTROL_ARCHITECTURE],
            calibration,
        ),
    ]

    tools = Path(__file__).resolve().parents[1] / "tools"
    return {
        "schema": SCHEMA,
        "purpose": "deterministic data and gradient plumbing receipt; no strength claim",
        "claims": {
            "proves": (
                "the V3 decoder path builds a batch the model accepts, two "
                "independent fits are bit identical, and a gradient reaches every "
                "named group, every phase bucket head and both PSQT columns of "
                "every bucket"
            ),
            "ranks_architectures": False,
            "strength_evidence": False,
            "production_network": False,
            "comparable_across_models": False,
            "labels": (
                "fixture labels under a frozen unit-slope Davidson link; the loss "
                "value carries no meaning beyond decreasing"
            ),
            "float_receipt_scope": "exact only within one pinned runtime",
        },
        "implementation": {
            "microfit_sha256": _sha256(Path(__file__).resolve()),
            "control_sha256": _sha256(tools / "horde_training_control.py"),
            "decoder_sha256": _sha256(tools / "horde_training_decoder.py"),
            "models_sha256": _sha256(tools / "horde_training_models.py"),
        },
        "execution": {
            "device": "cpu",
            "threads": 1,
            "interop_threads": 1,
            "deterministic_algorithms": True,
            "mkldnn": False,
            "float32_matmul_precision": "highest",
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "cuda_available_but_unused": torch.cuda.is_available(),
        },
        "training": {
            "model_seed": MODEL_SEED,
            "fixture_seed": FIXTURE_SEED,
            "steps": STEPS,
            "optimizer": "torch.optim.RAdam",
            "optimizer_source": "horde_training_control._make_optimizer",
            "learning_rate": LEARNING_RATE,
            "learning_rate_note": (
                "micro-fit rate chosen so both topologies move within 24 steps; "
                "not the production schedule"
            ),
            "dense_weight_clipping": True,
            "lambda": LAMBDA,
            "objective": "HORDE_WDL_HALF_BRIER_V1 via horde_training_control.loss_terms",
            "wdl_calibration": {
                "fixture": True,
                "parameters": {
                    side: list(values) for side, values in FIXTURE_CALIBRATION.items()
                },
            },
            "mate_score_threshold": control.MATE_SCORE_THRESHOLD,
            "rule50": "the trainer's integer postprocessor with its truncation STE",
            "initialization": models.NAMED_INITIALIZATION_SCHEMA,
            "minimum_loss_decrease": MINIMUM_LOSS_DECREASE,
        },
        "fixture": fixture,
        "models": fitted,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=RECEIPT_PATH)
    args = parser.parse_args(argv)

    print("V3 micro-fit: deterministic data and gradient plumbing")
    receipt = build_receipt(args.records)
    fixture = receipt["fixture"]
    assert isinstance(fixture, dict)
    print(f"  fixture source         : {fixture['source']['kind']}")
    print(
        f"  fixture records        : {fixture['records']} over "
        f"{fixture['cells_covered']} bucket-by-side cells "
        f"({fixture['derived_side_records']} side-derived)"
    )
    print(f"  G0 / contextual rows   : {fixture['g0_rows']} / {fixture['contextual_rows']}")
    for model in receipt["models"]:
        assert isinstance(model, dict)
        print(
            f"  {model['name']:<16} : loss {model['initial_loss']:.6f} -> "
            f"{model['final_loss']:.6f} (monotone={model['loss_monotone']}), "
            f"groups {model['gradient_groups_exercised']}"
        )
        reach = model.get("bucket_gradient_reach")
        if isinstance(reach, dict):
            print(
                "                     reached "
                + ", ".join(f"{layer}={sum(flags)}/{len(flags)}" for layer, flags in reach.items())
            )
        print(f"                     final state {model['final_state_sha256']}")

    if FAILURES:
        print(f"\nFAILED with {len(FAILURES)} problems:")
        for failure in FAILURES[:20]:
            print(f"  {failure}")
        return 1

    payload = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8", newline="\n")
    print(f"\nreceipt written: {args.output}")
    print("all V3 micro-fit checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
