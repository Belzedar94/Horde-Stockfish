# Contributing to Horde-Stockfish

Horde-Stockfish is a specialized engine with strict rule, network and testing
contracts. Contributions are welcome when they preserve those contracts and
keep each experiment easy to interpret.

## Before opening a change

Build the engine from `src`:

```sh
cd src
make -j2 ARCH=x86-64 build
```

Then run the checks relevant to the change. The minimum local smoke set is:

```sh
python scripts/horde/verify_baseline.py
python tests/horde_bench.py src/stockfish --runs 1
```

Rule, evaluation or search changes should also run:

```sh
python tests/horde_rules.py src/stockfish
python tests/horde_material_corpus.py src/stockfish
python tests/horde_network_contract.py src/stockfish
```

The complete required matrix runs in GitHub Actions. See the
[testing and release contract](docs/horde/testing-and-release-contract.md) for
the authoritative gates.

## Issues

Report reproducible Horde-Stockfish defects in this repository's
[issue tracker](https://github.com/Belzedar94/Horde-Stockfish/issues). Include:

- the full engine commit and binary architecture;
- operating system and CPU;
- the network filename and SHA-256;
- the exact UCI commands or GUI configuration;
- the FEN and move sequence when relevant; and
- complete stdout/stderr or PGN evidence.

Security-sensitive reports should not contain credentials, private network
URLs or access tokens.

## Pull requests

Keep each pull request focused on one orthogonal idea. A structural change may
be large, but unrelated search, rule, network or infrastructure changes must
not be accumulated into the same experiment.

Every pull request should describe:

- the hypothesis and intended behavior;
- the exact files and contracts affected;
- the deterministic benchmark before and after the change;
- correctness commands and their results; and
- the OpenBench STC/LTC links for strength-sensitive changes.

Search and move-ordering changes are tested on the project's
[OpenBench instance](https://belzedar.duckdns.org), not on Stockfish Fishtest.
Use the frozen Horde opening book, network and time-control contract recorded in
the test. A passing strength test does not waive correctness, speed or network
compatibility gates.

Rule changes require focused regression cases and comparison with the pinned
scalachess/Fairy-Stockfish authorities. NNUE changes require an explicit schema,
authenticated network identity, Python/native parity, full-refresh/incremental
parity, scalar/SIMD determinism and an equal-time strength gate before they can
replace the production evaluator.

First-time code contributors should add their name to [AUTHORS](AUTHORS).

## Code style

C++ changes must follow [`.clang-format`](.clang-format). Run:

```sh
make -C src format
```

Do not commit generated binaries, local match output, credentials or unrelated
formatting changes.

## License

By contributing, you agree that your contribution is distributed under the
[GNU General Public License version 3](Copying.txt).
