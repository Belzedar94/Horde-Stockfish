#!/usr/bin/env python3
"""Exercise the native release provenance, inventory, and checksum contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "horde"))

from release_manifest import (  # noqa: E402
    CHECKSUM_NAME,
    MANIFEST_NAME,
    ReleaseContractError,
    assemble,
    expected_asset_names,
    sha256,
)
from release_provenance import write_provenance  # noqa: E402


VERSION = "0.1.0"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
SOURCE_DATE_EPOCH = 1_786_291_200


class HordeReleaseContractTests(unittest.TestCase):
    def make_input(self, root: Path, *, omit: str | None = None) -> Path:
        input_root = root / "input"
        input_root.mkdir(parents=True)
        for index, name in enumerate(sorted(expected_asset_names(VERSION))):
            if name == omit:
                continue
            platform = "windows" if name.endswith(".zip") else "linux"
            architecture = "x86-64-bmi2" if "bmi2" in name else "x86-64-avx2"
            asset_dir = input_root / platform / architecture
            asset_dir.mkdir(parents=True)
            asset = asset_dir / name
            asset.write_bytes((f"asset-{index}-{name}\n").encode("ascii"))
            write_provenance(
                asset,
                VERSION,
                COMMIT,
                SOURCE_DATE_EPOCH,
                platform,
                architecture,
                "pinned-test-toolchain",
                ["bash", "scripts/horde/build_native_release.sh"],
            )
        return input_root

    def test_assemble_exact_inventory_and_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = self.make_input(root)
            output = root / "output"
            manifest = assemble(
                input_root, output, VERSION, COMMIT, SOURCE_DATE_EPOCH
            )

            self.assertEqual(manifest["schemaVersion"], 1)
            self.assertEqual(manifest["project"], "Horde-Stockfish")
            self.assertEqual(manifest["commit"], COMMIT)
            self.assertEqual(
                {entry["name"] for entry in manifest["artifacts"]},
                expected_asset_names(VERSION),
            )
            parsed = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(parsed, manifest)

            checksum_lines = (output / CHECKSUM_NAME).read_text(
                encoding="ascii"
            ).splitlines()
            declared = {
                name: digest for digest, name in (line.split("  ", 1) for line in checksum_lines)
            }
            self.assertEqual(
                set(declared), expected_asset_names(VERSION) | {MANIFEST_NAME}
            )
            for name, digest in declared.items():
                self.assertEqual(sha256(output / name), digest)

    def test_manifest_and_checksums_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_input = self.make_input(root / "first")
            second_input = self.make_input(root / "second")
            first_output = root / "first-output"
            second_output = root / "second-output"
            assemble(first_input, first_output, VERSION, COMMIT, SOURCE_DATE_EPOCH)
            assemble(second_input, second_output, VERSION, COMMIT, SOURCE_DATE_EPOCH)
            self.assertEqual(
                (first_output / MANIFEST_NAME).read_bytes(),
                (second_output / MANIFEST_NAME).read_bytes(),
            )
            self.assertEqual(
                (first_output / CHECKSUM_NAME).read_bytes(),
                (second_output / CHECKSUM_NAME).read_bytes(),
            )

    def test_tampered_asset_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = self.make_input(root)
            victim = next(path for path in input_root.rglob("*.zip"))
            victim.write_bytes(victim.read_bytes() + b"tampered")
            output = root / "output"
            with self.assertRaisesRegex(ReleaseContractError, "SHA-256 mismatch"):
                assemble(input_root, output, VERSION, COMMIT, SOURCE_DATE_EPOCH)
            self.assertFalse(output.exists())

    def test_missing_archive_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            omitted = sorted(expected_asset_names(VERSION))[0]
            input_root = self.make_input(root, omit=omitted)
            with self.assertRaisesRegex(ReleaseContractError, "inventory mismatch"):
                assemble(
                    input_root,
                    root / "output",
                    VERSION,
                    COMMIT,
                    SOURCE_DATE_EPOCH,
                )

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = self.make_input(root)
            output = root / "output"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="ascii")
            with self.assertRaisesRegex(ReleaseContractError, "already exists"):
                assemble(input_root, output, VERSION, COMMIT, SOURCE_DATE_EPOCH)
            self.assertEqual(marker.read_text(encoding="ascii"), "keep")


if __name__ == "__main__":
    unittest.main(verbosity=2)
