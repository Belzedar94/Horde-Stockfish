#!/usr/bin/env python3
"""Exercise the authenticated fresh-legacy NNUE exporter contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import horde_legacy_export as exporter  # noqa: E402
import horde_legacy_import as importer  # noqa: E402
import horde_run6b  # noqa: E402


SEED = 7_435_908_571_601_354_096
SOURCE_COMMIT = "7c2ab02dbd77d8707b49b1c7038d7d31b869bf94"
TRAIN_SHA = "0" * 64
VALIDATION_SHA = "1" * 64
WDL_SHA = "2" * 64


def model_state() -> dict[str, torch.Tensor]:
    state = {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in exporter.MODEL_SHAPES.items()
    }
    state["ft_bias"][0] = 0.0
    state["ft_bias"][1] = 2.0 / exporter.FT_SCALE
    state["ft_bias"][2] = -2.0 / exporter.FT_SCALE
    state["ft_weights"][0, 0] = 2.0 / exporter.FT_SCALE
    state["psqt_weights"][0, 0] = 3.0 / exporter.PSQT_SCALE
    state["hidden0_bias"][0, 0] = 4.0 / exporter.HIDDEN_BIAS_SCALE
    state["hidden0_weights"][0, 0, 0] = 5.0 / exporter.HIDDEN_WEIGHT_SCALE
    state["hidden1_bias"][0, 0] = 6.0 / exporter.HIDDEN_BIAS_SCALE
    state["hidden1_weights"][0, 0, 0] = 7.0 / exporter.HIDDEN_WEIGHT_SCALE
    state["output_bias"][0, 0] = 8.0 / exporter.OUTPUT_BIAS_SCALE
    state["output_weights"][0, 0, 0] = 9.0 / exporter.OUTPUT_WEIGHT_SCALE
    return state


def identities() -> dict[str, object]:
    return {
        "train_file": {"name": "train.bin", "sha256": TRAIN_SHA},
        "validation_file": {
            "name": "selected-records.bin",
            "sha256": VALIDATION_SHA,
        },
        "wdl_calibration": {"name": "wdl-calibration.json", "sha256": WDL_SHA},
    }


def checkpoint() -> dict[str, object]:
    return {
        "schema": exporter.CHECKPOINT_SCHEMA,
        "architecture": exporter.ARCHITECTURE_SCHEMA,
        "source": {"commit": SOURCE_COMMIT, "dirty": False},
        "model_state": model_state(),
        "settings": {"seed": SEED, "wdl_calibration_sha256": WDL_SHA},
        "data": identities(),
    }


def receipt(checkpoint_sha256: str) -> dict[str, object]:
    return {
        "schema": exporter.TRAINING_RECEIPT_SCHEMA,
        "source": {"commit": SOURCE_COMMIT, "dirty": False},
        "architecture": {
            "schema": exporter.ARCHITECTURE_SCHEMA,
            "legacy_feature_schema": exporter.FEATURE_SCHEMA,
            "serialized_topology": (
                "896 -> 512 shared FT + PSQT; 8 x (1024 -> 16 -> 32 -> 1)"
            ),
            "training_only_factorizer": False,
        },
        "artifacts": {"checkpoint": {"sha256": checkpoint_sha256}},
        "data": identities(),
        "run": {"complete": True, "seed": SEED},
        "labels": {"network_to_score": 600.0},
        "claims": {"production_network": False, "strength_evidence": False},
    }


def write_inputs(directory: Path, root: dict[str, object] | None = None) -> tuple[Path, Path]:
    checkpoint_path = directory / "checkpoint.pt"
    torch.save(checkpoint() if root is None else root, checkpoint_path)
    checkpoint_sha = exporter.sha256_file(checkpoint_path)
    receipt_path = directory / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt(checkpoint_sha), indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return checkpoint_path, receipt_path


def require_rejected(
    checkpoint_path: Path,
    receipt_path: Path,
    output: Path,
    export_receipt: Path,
    marker: str,
) -> None:
    try:
        exporter.export_checkpoint(checkpoint_path, receipt_path, output, export_receipt)
    except exporter.LegacyExportError as error:
        if marker not in str(error):
            raise AssertionError(f"wrong rejection for {marker!r}: {error}") from error
    else:
        raise AssertionError(f"export unexpectedly accepted {marker}")
    if output.exists() or export_receipt.exists():
        raise AssertionError(f"failed export left a partial output for {marker}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="horde-legacy-export-") as temp_name:
        temp = Path(temp_name)
        checkpoint_path, receipt_path = write_inputs(temp)
        output = temp / "fresh-legacy.nnue"
        export_receipt_path = temp / "export-receipt.json"
        description = "Horde fresh legacy exporter test"
        result = exporter.export_checkpoint(
            checkpoint_path,
            receipt_path,
            output,
            export_receipt_path,
            description,
        )

        expected_size = 1_088_336 + len(description.encode("ascii"))
        if output.stat().st_size != expected_size:
            raise AssertionError("exported legacy byte count changed")
        if result["artifact"]["sha256"] != exporter.sha256_file(output):
            raise AssertionError("export receipt does not bind the NNUE bytes")
        stored_receipt = json.loads(export_receipt_path.read_text(encoding="ascii"))
        if stored_receipt != result:
            raise AssertionError("stored export receipt differs from returned receipt")

        network = horde_run6b.Run6BNetwork.load_registered(
            output,
            result["artifact"]["sha256"],
            expected_size,
            "fresh legacy exporter test",
        )
        if network.description != description:
            raise AssertionError("legacy description did not round-trip")
        if list(network.biases[:3]) != [0, 2, -2]:
            raise AssertionError(f"feature bias quantization changed: {list(network.biases[:3])}")
        if network.weights[0] != 2 or network.psqt_weights[0] != 3:
            raise AssertionError("feature-transformer quantization changed")
        stack = network.layers[0]
        if stack.fc0.biases[0] != 4 or stack.fc0.weights[0] != 5:
            raise AssertionError("hidden0 quantization changed")
        if stack.fc1.biases[0] != 6 or stack.fc1.weights[0] != 7:
            raise AssertionError("hidden1 quantization changed")
        if any(stack.fc1.weights[index] for index in range(16, 32)):
            raise AssertionError("hidden1 padding is not zero")
        if stack.fc2.biases[0] != 8 or stack.fc2.weights[0] != 9:
            raise AssertionError("output quantization changed")

        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        first, _stats = exporter.build_legacy_nnue(loaded, description)
        second, _stats = exporter.build_legacy_nnue(loaded, description)
        if first != second or first != output.read_bytes():
            raise AssertionError("legacy serialization is not deterministic")

        missing_wdl_receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        del missing_wdl_receipt["data"]["wdl_calibration"]
        missing_wdl_path = temp / "missing-wdl.json"
        missing_wdl_path.write_text(json.dumps(missing_wdl_receipt), encoding="ascii")
        require_rejected(
            checkpoint_path,
            missing_wdl_path,
            temp / "missing-wdl.nnue",
            temp / "missing-wdl-export.json",
            "data identities differ",
        )

    with tempfile.TemporaryDirectory(prefix="horde-legacy-export-invalid-") as invalid_name:
        invalid = Path(invalid_name)
        good_directory = invalid / "good"
        bad_directory = invalid / "bad"
        good_directory.mkdir()
        bad_directory.mkdir()
        good_checkpoint, good_receipt = write_inputs(good_directory)

        bad_root = checkpoint()
        bad_state = dict(bad_root["model_state"])
        bad_bias = bad_state["ft_bias"].clone()
        bad_bias[0] = float("nan")
        bad_state["ft_bias"] = bad_bias
        bad_root["model_state"] = bad_state
        bad_checkpoint, bad_receipt = write_inputs(bad_directory, bad_root)
        require_rejected(
            bad_checkpoint,
            bad_receipt,
            invalid / "nonfinite.nnue",
            invalid / "nonfinite-export.json",
            "non-finite",
        )

        overflow_root = checkpoint()
        overflow_root["model_state"]["ft_bias"][0] = 100_000.0
        try:
            exporter.build_legacy_nnue(overflow_root, "overflow test")
        except exporter.LegacyExportError as error:
            if "overflows i16" not in str(error):
                raise AssertionError(f"wrong overflow rejection: {error}") from error
        else:
            raise AssertionError("out-of-range quantization was accepted")

        same_path = invalid / "same-output"
        require_rejected(
            good_checkpoint,
            good_receipt,
            same_path,
            same_path,
            "different files",
        )

    check_import_round_trip()
    print("Horde fresh legacy exporter contract completed successfully")
    return 0


def check_import_round_trip() -> None:
    """The importer must be the exporter's inverse, on real networks.

    A synthetic checkpoint is not enough evidence here. The importer exists to
    read networks this repository did not produce, so the fixture is exported
    first and then the shipped run6b container is imported, because the second
    is the case that actually matters and the only one whose bytes nobody here
    chose.
    """

    with tempfile.TemporaryDirectory(prefix="horde-legacy-import-") as temp_name:
        temp = Path(temp_name)
        description = "Horde fresh legacy importer test"
        original, _stats = exporter.build_legacy_nnue(checkpoint(), description)

        state, read_description, dequantization = importer.read_legacy_nnue(original)
        if read_description != description:
            raise AssertionError("the description did not survive the import")
        rebuilt, _stats = exporter.build_legacy_nnue({"model_state": state}, read_description)
        if rebuilt != original:
            raise AssertionError("a fixture network did not return byte for byte")

        # The parameters, not just the bytes: every value the fixture set is a
        # known integer, so the float that comes back is checkable directly.
        expected = model_state()
        for name in exporter.MODEL_SHAPES:
            if state[name].dtype != torch.float32:
                raise AssertionError(f"imported {name} is not float32")
            if not torch.equal(state[name], expected[name]):
                raise AssertionError(f"imported {name} differs from the exported parameters")

        # Truncation, trailing bytes and a wrong magic are refused rather than
        # read as a shorter or longer network.
        for label, payload, marker in (
            ("a truncated container", original[:-1], "container ends inside"),
            ("a padded container", original + b"\x00", "trailing bytes"),
            ("a wrong file version", b"\x00\x00\x00\x00" + original[4:], "file version"),
        ):
            try:
                importer.read_legacy_nnue(payload)
            except importer.LegacyImportError as error:
                if marker not in str(error):
                    raise AssertionError(
                        f"wrong rejection for {label}: {error}"
                    ) from error
            else:
                raise AssertionError(f"the importer accepted {label}")

        run6b = ROOT / "networks" / "hordetest_run6b_e37_l06.nnue"
        if not run6b.is_file():
            print("  run6b container absent, skipping the champion round-trip")
            return
        payload = run6b.read_bytes()
        state, run6b_description, _dequantization = importer.read_legacy_nnue(payload)
        rebuilt, _stats = exporter.build_legacy_nnue(
            {"model_state": state}, run6b_description
        )
        if exporter.sha256_bytes(rebuilt) != exporter.sha256_bytes(payload):
            raise AssertionError("run6b did not return byte for byte through the importer")

        # And the whole tool, not only its parser: the checkpoint it writes must
        # be loadable and must carry the provenance of the container.
        output = temp / "run6b-import.pt"
        import_receipt = temp / "run6b-import.json"
        result = importer.import_container(run6b, output, import_receipt)
        if result["round_trip"]["byte_identical"] is not True:
            raise AssertionError("the importer reported a lossy round-trip on run6b")
        if result["imported_from"]["sha256"] != exporter.sha256_file(run6b):
            raise AssertionError("the import receipt does not bind the source container")
        loaded = torch.load(output, map_location="cpu", weights_only=True)
        if loaded.get("schema") != exporter.CHECKPOINT_SCHEMA:
            raise AssertionError("the imported checkpoint declares the wrong schema")
        if loaded.get("architecture") != exporter.ARCHITECTURE_SCHEMA:
            raise AssertionError("the imported checkpoint declares the wrong architecture")
        for absent in ("optimizer_state", "scheduler_state", "rng_state", "progress"):
            if absent in loaded:
                raise AssertionError(
                    f"the imported checkpoint carries {absent}, which no container can supply"
                )
        print(f"  run6b round-tripped byte for byte: {result['artifact']['sha256']}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AssertionError,
        OSError,
        RuntimeError,
        exporter.LegacyExportError,
        importer.LegacyImportError,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
