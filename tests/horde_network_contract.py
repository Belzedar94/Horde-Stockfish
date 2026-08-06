#!/usr/bin/env python3
"""Exercise the fail-closed Run 6B EvalFile contract."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path


RUN6B_SHA256 = "b71108587968ac544eb2e62c2333feca880da5aca52866787f1402163444adf7"


def invoke(executable: Path, commands: list[str]) -> subprocess.CompletedProcess[str]:
    synchronized: list[str] = []
    for command in commands:
        synchronized.append(command)
        if command == "go" or command.startswith("go "):
            synchronized.append("setoption name Move Overhead value 10")
    return subprocess.run(
        [str(executable)],
        input="\n".join((*synchronized, "quit", "")),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    default_name = "stockfish.exe" if os.name == "nt" else "stockfish"
    executable = Path(
        sys.argv[1] if len(sys.argv) > 1 else root / "src" / default_name
    ).resolve()
    canonical = root / "networks" / "hordetest_run6b_e37_l06.nnue"

    if not executable.is_file():
        raise SystemExit(f"Engine not found: {executable}")
    payload = canonical.read_bytes()
    if hashlib.sha256(payload).hexdigest() != RUN6B_SHA256:
        raise AssertionError("Canonical Run 6B network digest changed")

    valid = invoke(
        executable,
        [
            f"setoption name EvalFile value {canonical}",
            "position startpos",
            "go depth 1",
        ],
    )
    valid_output = valid.stdout + valid.stderr
    if valid.returncode != 0 or "bestmove " not in valid_output:
        raise AssertionError(f"Canonical network did not search.\n{valid_output}")

    with tempfile.TemporaryDirectory(prefix="horde-network-") as directory:
        mutated = bytearray(payload)
        mutated[-1] ^= 1
        wrong = Path(directory) / "same-size-wrong-sha.nnue"
        wrong.write_bytes(mutated)

        rejected = invoke(
            executable,
            [
                f"setoption name EvalFile value {wrong}",
                "position startpos",
                "go depth 1",
            ],
        )
        rejected_output = rejected.stdout + rejected.stderr
        if rejected.returncode == 0:
            raise AssertionError("A same-size network with the wrong SHA-256 was accepted")
        for marker in (
            "Rejected EvalFile: SHA-256 does not match",
            "Unregistered networks are rejected",
            "Search was not started",
        ):
            if marker not in rejected_output:
                raise AssertionError(f"Missing rejection marker {marker!r}.\n{rejected_output}")

    missing = invoke(
        executable,
        [
            "setoption name EvalFile value missing-run6b.nnue",
            "position startpos",
            "go depth 1",
        ],
    )
    missing_output = missing.stdout + missing.stderr
    if missing.returncode == 0 or "Search was not started" not in missing_output:
        raise AssertionError(f"A missing network did not fail closed.\n{missing_output}")

    print("Horde Run 6B network contract completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
