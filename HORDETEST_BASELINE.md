# FSF_HORDETEST_RUN6B_BASELINE_V1

This branch freezes the formal Fairy-Stockfish opponent used to evaluate
Horde-Stockfish.

- Source: `fairy-stockfish/Fairy-Stockfish@c19b5f6c66894fdb0e88d0dd100e3885f744760a`.
- Variant: `hordetest`, embedded from the frozen `hordetest.ini` definition.
- Network: `hordetest_run6b_e37_l06.nnue`, supplied externally.
- Network SHA-256: `b71108587968ac544eb2e62c2333feca880da5aca52866787f1402163444adf7`.
- Network size: 1,088,416 bytes.
- Reference compiler: `g++.exe (Rev5, Built by MSYS2 project) 16.1.0`.
- Reference binary SHA-256: `5a9e0729d9e5dce9166b8bd89f2a0c2c79d6179631d1fbacddf969aa7d45dbda`.
- Reference binary size: 4,530,986 bytes.
- Start-position depth 8: score cp 153, 1,722 nodes, `a4a5 e7e6`.
- Bench: 130,284 nodes with Threads=1 and Hash=16.

The engine-source changes are limited to baseline packaging and the UCI input
boundary. The frozen Hordetest definition is compiled into the executable and
is the default variant on this dedicated branch. Public Horde FENs use physical
white `P` pieces; the legacy evaluator expects those same pawns as feature-role
`H` pieces. `src/uci.cpp` converts uppercase `P` to `H` only when
`UCI_Variant=hordetest`. Move generation, evaluation, and search are unchanged.
Promoted white pieces remain ordinary `N`, `B`, `R`, and `Q` pieces.

Build the baseline with:

```sh
make -C src -j2 build ARCH=x86-64-bmi2 COMP=mingw \
  EXTRALDFLAGS=-Wl,--no-insert-timestamp
```

Validate the bridge, Run 6B identity, and canonical perfts with:

```sh
python tests/hordetest_baseline.py src/stockfish.exe \
  --variant-path hordetest.ini \
  --network /path/to/hordetest_run6b_e37_l06.nnue
```

The OpenBench artifact contains the sole executable. It sets
`UCI_Variant=hordetest` and `EvalFile` to the canonical Run 6B bytes; no sidecar
variant file is required at runtime. `hordetest.ini` remains the reviewable
source receipt for the embedded definition. Standard NNUE, HCE, Syzygy, and
zero-evaluation fallbacks are not valid substitutes for this baseline contract.
