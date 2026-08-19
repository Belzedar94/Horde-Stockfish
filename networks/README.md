# Horde NNUE network

`hordetest_run6b_e37_l06.nnue` is the immutable HordeTest Run 6B
baseline network.

- Author and attribution: Belzedar
- License: CC0 1.0 Universal
- Size: 1,088,416 bytes
- SHA-256: `b71108587968ac544eb2e62c2333feca880da5aca52866787f1402163444adf7`
- Required runtime variant: `hordetest`

The filename deliberately starts with `hordetest`. Fairy-Stockfish uses the
variant prefix when selecting an NNUE file. Do not rename the tracked source
file, transform its bytes, or substitute a network with the same dimensions.

Horde-Stockfish release packages expose these exact bytes as `Horde_v1.nnue`.
That is a distribution alias only; the tracked source filename and the engine's
default path remain unchanged.

This network is an evaluation artifact, not an engine binary. Its presence does
not change the GPL obligations that apply to Fairy-Stockfish source and binary
distributions.

See [the baseline manifest](../docs/horde/baseline-manifest.json) for the frozen
source, rule and test inputs.
