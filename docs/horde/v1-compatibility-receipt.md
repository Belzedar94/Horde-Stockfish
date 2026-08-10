# V1 compatibility receipt

Date: 2026-08-06

This receipt covers rules and legacy Run 6B compatibility. It is not a strength,
release, or publication receipt.

## Frozen inputs

- Horde-Stockfish source base:
  `official-stockfish/Stockfish@762dd1da9a5db458180b2c5db6c53dc40ec61e1a`.
- Rule authority:
  `lichess-org/scalachess@d5d47c16f65a005ca68e19bab702b02f66dd888c`.
- Move-generation and legacy-format authority:
  `fairy-stockfish/Fairy-Stockfish@c19b5f6c66894fdb0e88d0dd100e3885f744760a`.
- Run 6B SHA-256:
  `B71108587968AC544EB2E62C2333FECA880DA5ACA52866787F1402163444ADF7`.
- Formal Fairy-Stockfish baseline binary SHA-256:
  `5D3B320B9FC8282997243B2FFC340FC5FCB52F88B63D7F4C9D3FF2AFAA2497BC`.

## Differential NNUE gate

The test corpus contains 100,000 unique positions reached exclusively through
legal Horde moves. Generation uses seed `0x6B3706`; the ordered FEN corpus has
SHA-256 `DED6631229A4697DA319A2AFF676902CAAC210F2344EFD708B3E3FF02B47E710`.

Coverage observed in the frozen run:

- 37,263 captures;
- 2,393 promotions;
- 459 en-passant captures;
- 247 castlings;
- 5,423 positions with one White piece;
- 5,494 positions with two White pieces;
- 5,076 positions with three White pieces;
- 5,161 positions with four White pieces.

Horde-Stockfish matched the diagnostic Fairy-Stockfish oracle in all 100,000
positions for each unscaled integer (PSQT, positional output, and their sum)
and for the final scaled evaluation. The final gate reproduces the frozen
HordeTest `H` material value at the evaluator boundary and separately checks
halfmove clocks `0`, `50`, `90`, and `99`. Only the oracle boundary translates
White `P` to HordeTest `H`.

## Incremental and determinism gates

- The search accumulator applies exact dirty-piece deltas for normal moves,
  captures, promotions, en passant, castling, and undo.
- The assertions build compares incremental evaluation with full refresh on
  every evaluated search position.
- Instrumented release builds sample the same comparison every 1,024 calls.
- Generic x86-64 and BMI2 binaries produced identical raw integers on 10,000
  positions at Threads 1, 2, and 4: 60,000 comparisons in total.
- Network changes clear the transposition table, accumulator state, and caches.

## Rules and benchmark gates

- All fixed Horde perfts pass, including depth-4 counts `23310`, `56539`,
  `33781`, `128809`, and `197287`.
- All 21,996 pinned side-specific material rows pass.
- Invalid kings, White castling, Chess960, Syzygy, absent networks, altered
  same-size networks, and unregistered networks fail closed.
- Mate, extinction, both stalemates, fortress, fifty-move, fivefold repetition,
  and last-Horde-piece capture regressions pass.
- The Horde-only benchmark (`bench 16 1 13 default depth`) searches exactly
  315,576 nodes. Three consecutive runs produced the same ten-bestmove digest:
  `FE9A5001C1997125CE34BF0EF119EAB44570F5F363227BD4BAB8E0DB1F4E8592`.

## Gates not claimed here

Linux sanitizer and cross-platform checks must pass on the exact review commit.
Referee validation, OpenBench deployment, DATAGEN, the three-time-control match
panel, the +100 Elo thresholds, and the publication gate remain separate and
mandatory. The repository remains private until those receipts exist.
