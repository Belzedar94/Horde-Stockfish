# Reproducing the Fairy-Stockfish HordeTest baseline

This recipe recreates the formal Run 6B baseline without relying on an old
opaque executable. It fixes source, rules and network independently so that a
failure can be attributed to the correct layer.

## Frozen inputs

- Fairy-Stockfish source:
  `c19b5f6c66894fdb0e88d0dd100e3885f744760a`
- Custom rule fixture: [`fixtures/variants.ini`](fixtures/variants.ini)
- Network: [`../../networks/hordetest_run6b_e37_l06.nnue`](../../networks/hordetest_run6b_e37_l06.nnue)
- Network SHA-256:
  `b71108587968ac544eb2e62c2333feca880da5aca52866787f1402163444adf7`
- Perft fixture: [`fixtures/hordetest.perft`](fixtures/hordetest.perft)

All machine-readable values are in
[`baseline-manifest.json`](baseline-manifest.json).

## 1. Verify the repository artifacts

From this repository root:

```console
python scripts/horde/verify_baseline.py
```

This step is offline. It reads files and computes hashes; it does not rewrite
the network or fixtures.

## 2. Build the formal engine

Use a separate clean directory:

```console
git clone https://github.com/fairy-stockfish/Fairy-Stockfish.git
cd Fairy-Stockfish
git checkout --detach c19b5f6c66894fdb0e88d0dd100e3885f744760a
git status --short
cd src
make -j profile-build ARCH=x86-64
```

Select an architecture target that the host supports and record it. Do not
claim byte-for-byte reproducibility without also freezing compiler, linker and
build environment. The required source identity is the full commit, not the
example output filename.

## 3. Run the automated rule and NNUE probe

```console
python scripts/horde/verify_baseline.py --engine /absolute/path/to/fairy-stockfish
```

On Windows:

```console
py scripts\horde\verify_baseline.py --engine C:\absolute\path\to\fairy-stockfish.exe
```

The script creates a temporary working directory, copies the frozen fixture to
`variants.ini`, launches the supplied binary as `fairy-stockfish load
variants.ini`, selects `hordetest`, loads the repository network, checks all
perft values and requests `go depth 8`.
Temporary files are removed automatically.

If the supplied binary matches the historical oracle checksum, the script
stops. That binary may be consulted manually to understand old behavior, but
it cannot satisfy the formal-source gate.

## 4. Manual UCI transcript

For manual diagnosis, place a copy of the fixture named `variants.ini` in the
engine's working directory and start the formally built engine from that
directory with:

```console
fairy-stockfish load variants.ini
```

Then send:

```text
uci
setoption name UCI_Variant value hordetest
setoption name EvalFile value /absolute/path/to/hordetest_run6b_e37_l06.nnue
setoption name Threads value 1
setoption name Hash value 64
isready
position fen rnbqkbnr/pppppppp/8/1HH2HH1/HHHHHHHH/HHHHHHHH/HHHHHHHH/HHHHHHHH w kq - 0 1
go depth 8
quit
```

Required observations:

- the engine advertises and accepts `UCI_Variant`;
- `hordetest` is recognized from the local `variants.ini`;
- the NNUE is loaded rather than silently ignored;
- `readyok` arrives;
- search reports nodes and a legal `bestmove`; and
- no assertion, access violation or protocol error occurs.

The network name must keep the `hordetest` prefix. Selecting built-in `horde`
with this network is not the same baseline because white pawns no longer have
the custom `H` feature identity.

## 5. Minimal match configuration

A reproducible match runner must translate these settings without changing
their meaning:

```yaml
variant: hordetest
variants_file: docs/horde/fixtures/variants.ini
engine_source_commit: c19b5f6c66894fdb0e88d0dd100e3885f744760a
engine_options:
  UCI_Variant: hordetest
  EvalFile: networks/hordetest_run6b_e37_l06.nnue
  Threads: 1
  Hash: 512
network_sha256: b71108587968ac544eb2e62c2333feca880da5aca52866787f1402163444adf7
paired_colors: true
```

Time control, openings, concurrency and adjudication are intentionally absent
from this minimal engine definition. They belong to a named experiment and
must be reported with their own hashes and raw totals. Reusing the engine
definition does not make two matches comparable if those experiment settings
differ.

## 6. What each input proves

| Input | What it proves | What it does not prove |
|---|---|---|
| Pinned source | Formal code identity | Compiler or binary identity |
| `variants.ini` | HordeTest feature/rule encoding | Rule correctness by itself |
| Perft fixtures | Required move-tree counts | All terminal/draw semantics |
| Run 6B NNUE | Exact evaluation parameters | Current official network status |
| Depth-8 probe | Basic runtime integration | Playing strength or long-run stability |
| Historical binary | Prior observed behavior | Formal baseline or release provenance |

Follow the full [testing and release contract](testing-and-release-contract.md)
before making a release or strength claim.

## 7. Common failure classifications

- **Unknown variant:** `variants.ini` was not found in the engine working
  directory or the binary lacks custom-variant support.
- **Network ignored:** the filename lost the `hordetest` prefix, `EvalFile` is
  wrong, or the binary is incompatible with the network architecture.
- **Perft differs only in `H` positions:** custom pawn, promotion or en-passant
  semantics are wrong.
- **Perft differs in both encodings:** the underlying Horde rule implementation
  or source pin differs.
- **NPS is zero or no bestmove arrives:** classify protocol, network loading,
  search startup and binary compatibility before attributing it to the GUI.
- **Oracle passes but formal build fails:** the old executable is evidence of a
  behavioral difference, not permission to replace the formal source.
