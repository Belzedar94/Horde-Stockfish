#!/usr/bin/env python3

import argparse
import hashlib
from pathlib import Path
import re
import subprocess


RUN6B_BYTES = 1_088_416
RUN6B_SHA256 = "b71108587968ac544eb2e62c2333feca880da5aca52866787f1402163444adf7"
BASELINE_BENCH = 130_284

POSITIONS = (
    (
        "start",
        "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1",
        4,
        23_310,
    ),
    (
        "open-flank",
        "4k3/pp4q1/3P2p1/8/P3PP2/PPP2r2/PPP5/PPPP4 b - - 0 1",
        4,
        56_539,
    ),
    (
        "en-passant",
        "k7/5p2/4p2P/3p2P1/2p2P2/1p2P2P/p2P2P1/2P2P2 w - - 0 1",
        4,
        33_781,
    ),
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_engine(engine, commands, network=None):
    setup = (
        "uci\n"
        "setoption name UCI_Variant value hordetest\n"
    )
    if network is not None:
        setup += f"setoption name EvalFile value {network.as_posix()}\n"
    setup += "isready\n"

    completed = subprocess.run(
        [str(engine)],
        input=setup + commands + "quit\n",
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode:
        raise RuntimeError(
            f"{engine.name} exited with status {completed.returncode}\n{output}"
        )
    if "readyok" not in output or "variant hordetest" not in output:
        raise RuntimeError(f"Hordetest did not initialize\n{output}")
    return output


def extract(pattern, output, label):
    match = re.search(pattern, output, flags=re.MULTILINE)
    if not match:
        raise AssertionError(f"Missing {label} in engine output")
    return match.group(1).strip()


def read_until(process, prefix):
    lines = []
    while True:
        line = process.stdout.readline()
        if line == "":
            raise RuntimeError(f"Engine closed before {prefix}: {lines[-20:]}")
        line = line.rstrip("\r\n")
        lines.append(line)
        if line.startswith(prefix):
            return lines


def search_receipt(engine, network, fen):
    process = subprocess.Popen(
        [str(engine)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        process.stdin.write("uci\n")
        process.stdin.flush()
        read_until(process, "uciok")

        commands = (
            "setoption name UCI_Variant value hordetest\n"
            f"setoption name EvalFile value {network.as_posix()}\n"
            "setoption name Threads value 1\n"
            "setoption name Hash value 16\n"
            "isready\n"
        )
        process.stdin.write(commands)
        process.stdin.flush()
        read_until(process, "readyok")

        process.stdin.write(f"position fen {fen}\ngo depth 8\n")
        process.stdin.flush()
        lines = read_until(process, "bestmove")
        info = [line for line in lines if line.startswith("info depth 8")][-1]
        bestmove = [line for line in lines if line.startswith("bestmove")][-1]
        match = re.search(
            r"^info depth (\d+) seldepth (\d+).* score (cp|mate) (-?\d+) "
            r"nodes (\d+).* pv (.+)$",
            info,
        )
        if not match:
            raise AssertionError(f"Unexpected depth-8 receipt: {info}")
        return (
            int(match.group(1)),
            int(match.group(2)),
            match.group(3),
            int(match.group(4)),
            int(match.group(5)),
            match.group(6),
            bestmove,
        )
    finally:
        if process.poll() is None:
            process.stdin.write("quit\n")
            process.stdin.flush()
            process.wait(timeout=10)


def verify_bridge(engine, network):
    physical = POSITIONS[0][1]
    legacy = physical.replace("P", "H")

    outputs = []
    for fen in (physical, legacy):
        outputs.append(
            run_engine(
                engine,
                f"position fen {fen}\nd\neval\n",
                network,
            )
        )

    fields = []
    for output in outputs:
        fields.append(
            (
                extract(r"^Fen:\s*(.+)$", output, "FEN"),
                extract(r"^Key:\s*(\S+)$", output, "position key"),
                extract(
                    r"^NNUE evaluation\s+([^\r\n]+)$",
                    output,
                    "NNUE evaluation",
                ),
                extract(
                    r"^Final evaluation\s+([^\r\n]+)$",
                    output,
                    "final evaluation",
                ),
            )
        )

    if fields[0] != fields[1]:
        raise AssertionError(f"Physical P and legacy H diverged: {fields}")
    if fields[0][0] != legacy:
        raise AssertionError(f"Unexpected bridged FEN: {fields[0][0]}")

    promoted = "4k3/pp6/8/8/8/2NBRQ2/PP6/8 w - - 0 1"
    output = run_engine(
        engine,
        f"position fen {promoted}\nd\n",
        network,
    )
    actual = extract(r"^Fen:\s*(.+)$", output, "promoted-piece FEN")
    expected = promoted.replace("P", "H")
    if actual != expected:
        raise AssertionError(f"Promoted pieces changed at the bridge: {actual}")

    print("P/H UCI bridge: exact")

    receipts = [
        search_receipt(engine, network, fen)
        for fen in (physical, legacy)
    ]
    if receipts[0] != receipts[1]:
        raise AssertionError(f"Physical P and legacy H search diverged: {receipts}")
    expected_receipt = (
        8,
        12,
        "cp",
        153,
        1722,
        "a4a5 e7e6 f5e6 d7e6 h4h5 a7a6 a3a4 h7h6",
        "bestmove a4a5 ponder e7e6",
    )
    if receipts[0] != expected_receipt:
        raise AssertionError(f"Depth-8 receipt changed: {receipts[0]}")
    print("P/H depth-8 search: exact (score cp 153, nodes 1722)")


def verify_perfts(engine):
    for name, fen, depth, expected in POSITIONS:
        output = run_engine(
            engine,
            f"position fen {fen}\ngo perft {depth}\n",
        )
        actual = int(
            extract(r"^Nodes searched:\s*([0-9]+)$", output, f"{name} perft")
        )
        if actual != expected:
            raise AssertionError(
                f"{name} perft d{depth}: expected {expected}, got {actual}"
            )
        print(f"{name} d{depth}: {actual}")


def verify_embedded_contract(engine, network):
    commands = (
        "uci\n"
        f"setoption name EvalFile value {network.as_posix()}\n"
        "isready\n"
        "bench\n"
        "quit\n"
    )
    completed = subprocess.run(
        [str(engine)],
        input=commands,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode:
        raise RuntimeError(
            f"{engine.name} exited with status {completed.returncode}\n{output}"
        )
    if not re.search(
        r"^option name UCI_Variant type combo default hordetest(?: |$)",
        output,
        flags=re.MULTILINE,
    ):
        raise AssertionError("Embedded Hordetest is not the UCI default")
    actual = int(
        extract(r"^Nodes searched\s*:\s*([0-9]+)$", output, "baseline bench")
    )
    if actual != BASELINE_BENCH:
        raise AssertionError(
            f"Baseline bench: expected {BASELINE_BENCH}, got {actual}"
        )
    print(f"embedded default bench: {actual}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=Path)
    parser.add_argument("--variant-path", type=Path, default=Path("hordetest.ini"))
    parser.add_argument("--network", type=Path)
    args = parser.parse_args()

    engine = args.engine.resolve()
    variant_path = args.variant_path.resolve()
    if not engine.is_file():
        raise SystemExit(f"Engine not found: {engine}")
    if not variant_path.is_file():
        raise SystemExit(f"Variant file not found: {variant_path}")

    verify_perfts(engine)

    if args.network is not None:
        network = args.network.resolve()
        if network.stat().st_size != RUN6B_BYTES:
            raise SystemExit("Run 6B byte count mismatch")
        if sha256(network) != RUN6B_SHA256:
            raise SystemExit("Run 6B SHA-256 mismatch")
        verify_bridge(engine, network)
        verify_embedded_contract(engine, network)


if __name__ == "__main__":
    main()
