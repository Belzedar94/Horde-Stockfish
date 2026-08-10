#!/usr/bin/env python3
"""Focused tests for the fixed Horde release-panel runner."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "horde" / "run_release_panel.py"
SPEC = importlib.util.spec_from_file_location("run_release_panel", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
panel = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = panel
SPEC.loader.exec_module(panel)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


class TrackerTests(unittest.TestCase):
    def test_out_of_order_games_form_exact_pairs(self) -> None:
        spec = panel.PanelSpec("TEST", "1+0.01", 16, 4, 1)
        tracker = panel.PanelTracker(spec)
        tracker.set_status("RUNNING")
        tracker.consume(
            "Finished game 2 (Fairy-Stockfish-dev vs Horde-Stockfish-release): "
            "1-0 {Black mates}"
        )
        tracker.consume(
            "Finished game 1 (Horde-Stockfish-release vs Fairy-Stockfish-dev): "
            "1-0 {White mates}"
        )
        tracker.consume(
            "Finished game 4 (Fairy-Stockfish-dev vs Horde-Stockfish-release): "
            "0-1 {Black mates}"
        )
        tracker.consume(
            "Finished game 3 (Horde-Stockfish-release vs Fairy-Stockfish-dev): "
            "1/2-1/2 {Draw by stalemate}"
        )

        snapshot = tracker.require_complete()
        self.assertEqual(snapshot["games"], 4)
        self.assertEqual(snapshot["wld"], [2, 1, 1])
        self.assertEqual(snapshot["pentanomial"], [0, 0, 1, 1, 0])
        self.assertGreater(snapshot["statistics"]["elo"], 0.0)

    def test_infrastructure_reason_invalidates_panel(self) -> None:
        spec = panel.PanelSpec("TEST", "1+0.01", 16, 2, 1)
        tracker = panel.PanelTracker(spec)
        with self.assertRaisesRegex(RuntimeError, "infrastructure defect"):
            tracker.consume(
                "Finished game 1 (Horde-Stockfish-release vs Fairy-Stockfish-dev): "
                "0-1 {White disconnects}"
            )

    def test_missing_game_is_rejected(self) -> None:
        spec = panel.PanelSpec("TEST", "1+0.01", 16, 2, 1)
        tracker = panel.PanelTracker(spec)
        tracker.consume(
            "Finished game 1 (Horde-Stockfish-release vs Fairy-Stockfish-dev): "
            "1/2-1/2 {Draw by stalemate}"
        )
        with self.assertRaisesRegex(RuntimeError, "missing 2"):
            tracker.require_complete()


class ContractTests(unittest.TestCase):
    def test_network_hashes_are_fixed_by_the_runner(self) -> None:
        args = panel.parse_args(
            [
                "--candidate",
                "candidate.exe",
                "--candidate-sha256",
                "0" * 64,
                "--candidate-commit",
                "0" * 40,
                "--candidate-network",
                "Horde_v1.nnue",
                "--baseline",
                "fairy-stockfish.exe",
                "--baseline-sha256",
                "1" * 64,
                "--baseline-commit",
                "1" * 40,
                "--baseline-network",
                "horde-28173ddccabe.nnue",
                "--referee",
                "cutechess-cli.exe",
                "--referee-sha256",
                "2" * 64,
                "--referee-commit",
                "2" * 40,
                "--book",
                "HORDE_openings_v3.epd",
                "--book-sha256",
                "3" * 64,
                "--runner-commit",
                "3" * 40,
                "--output-root",
                "release-panel-output",
            ]
        )
        self.assertEqual(args.candidate_network_sha256, panel.RUN6B_SHA256)
        self.assertEqual(
            args.baseline_network_sha256, panel.OFFICIAL_HORDE_SHA256
        )

    def test_panel_shape_is_frozen(self) -> None:
        self.assertEqual(
            panel.RUN6B_SHA256,
            "B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7",
        )
        self.assertEqual(
            panel.OFFICIAL_HORDE_SHA256,
            "28173DDCCABE12306D02AFA1156DED2B6A69C6A8DB909895DB6E955F8B4AD6A6",
        )
        self.assertEqual(
            [(spec.label, spec.tc, spec.hash_mb, spec.games) for spec in panel.PANEL_SPECS],
            [
                ("VSTC", "2+0.02", 16, 600),
                ("STC", "10+0.1", 32, 400),
                ("LTC", "30+0.3", 128, 200),
            ],
        )

    def test_command_has_pairing_and_no_classical_adjudication(self) -> None:
        args = SimpleNamespace(
            candidate=Path("candidate.exe").resolve(),
            candidate_network=Path("Horde_v1.nnue").resolve(),
            candidate_option=[],
            baseline=Path("fairy-stockfish.exe").resolve(),
            baseline_network=Path("horde-28173ddccabe.nnue").resolve(),
            baseline_option=[],
            referee=Path("cutechess-cli.exe").resolve(),
            book=Path("HORDE_openings_v3.epd").resolve(),
            concurrency_per_tc=4,
            move_overhead_ms=50,
            time_margin_ms=250,
        )
        command = panel.build_command(
            args, panel.PANEL_SPECS[0], Path("games.pgn").resolve()
        )
        self.assertIn("-repeat", command)
        self.assertIn("-maxmoves", command)
        self.assertNotIn("-draw", command)
        self.assertNotIn("-resign", command)
        self.assertNotIn("-recover", command)
        self.assertIn("option.UCI_Variant=horde", command)
        self.assertIn("option.Hash=16", command)
        self.assertEqual(command[command.index("-games") + 1], "600")

    def test_assets_are_authenticated_and_book_cannot_wrap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = {
                "candidate": b"candidate",
                "candidate_network": b"candidate-network",
                "baseline": b"baseline",
                "baseline_network": b"baseline-network",
                "referee": b"referee",
            }
            paths: dict[str, Path] = {}
            for role, content in data.items():
                paths[role] = root / role
                paths[role].write_bytes(content)
            book = root / "book.epd"
            book.write_text(
                "\n".join(f"8/8/8/8/8/8/8/8 w - - 0 1; id {index};" for index in range(300)),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                candidate=paths["candidate"],
                candidate_sha256=digest(data["candidate"]),
                candidate_network=paths["candidate_network"],
                candidate_network_sha256=digest(data["candidate_network"]),
                baseline=paths["baseline"],
                baseline_sha256=digest(data["baseline"]),
                baseline_network=paths["baseline_network"],
                baseline_network_sha256=digest(data["baseline_network"]),
                referee=paths["referee"],
                referee_sha256=digest(data["referee"]),
                book=book,
                book_sha256=panel.sha256_file(book),
            )
            observed = panel.validate_inputs(args)
            self.assertEqual(observed["book"]["opening_count"], 300)
            self.assertEqual(
                observed["runner_script"]["sha256"], panel.sha256_file(MODULE_PATH)
            )

            short_book = root / "short.epd"
            short_book.write_text("position\n" * 299, encoding="utf-8")
            args.book = short_book
            args.book_sha256 = panel.sha256_file(short_book)
            with self.assertRaisesRegex(RuntimeError, "at least 300"):
                panel.validate_inputs(args)


if __name__ == "__main__":
    unittest.main()
