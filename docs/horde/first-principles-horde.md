# Horde from first principles

This document defines the game that this project is intended to play. It starts
from observable game rules rather than from inherited Stockfish assumptions.
The normative baseline is the Lichess Horde rules, checked against the pinned
`lichess-org/scalachess` implementation in the
[baseline manifest](baseline-manifest.json).

## 1. Game state

Horde is played on the standard eight-by-eight board. Black starts with the
usual chess army. White starts with 36 pawns and no king:

```text
rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1
```

White occupies every square on ranks one through four, plus `b5`, `c5`, `f5`
and `g5`. Black has both normal castling rights. White has none because white
has no king or rooks.

The two sides therefore have different state invariants:

- Black has one royal king. Black moves are legal only when that king is not
  left in check.
- White has no royal piece. White is not in check and cannot make a move
  illegal by exposing a nonexistent king.
- Every surviving white piece belongs to the Horde, including a piece created
  by promotion.

These facts are not evaluation hints. They are rules that must be represented
correctly in move generation, position validation, terminal detection, FEN
round trips, search and protocol output.

## 2. Legal moves

Except for the Horde-specific pawn rules and the absence of a white king, a
move is legal under the corresponding standard-chess rule.

### 2.1 White pawns

A white Horde pawn:

- moves one rank forward into an empty square;
- captures one square diagonally forward;
- may move two squares from rank one or rank two if both traversed squares are
  empty;
- promotes on rank eight to queen, rook, bishop or knight; and
- may capture en passant when the ordinary Horde en-passant conditions hold.

All four promotion types are required. Omitting knight underpromotion is a
known historical move-generation bug and changes perft results.

The extra rank-one double move is not equivalent to an ordinary rank-two pawn
double move for en passant. The public Lichess rule explicitly says that a
black pawn may not capture a rank-one Horde pawn en passant after that pawn
moves two squares. Other valid en-passant cases remain part of the game.

### 2.2 Black pieces

Black moves by standard chess rules. In particular:

- the black king may not move into check;
- a black move may not expose its own king to check;
- castling is governed by the normal safety and path rules; and
- black pawns promote normally and use only valid en-passant captures.

White pieces attack the black king even though white has no king of its own.

## 3. Terminal results

The result is asymmetric:

- The Horde wins by checkmating the black king.
- Black wins by capturing every white piece. Promoted Horde pieces count, so
  removing the last pawn is not a win if another white piece remains.

Standard repetition and move-count mechanisms remain relevant through the
selected referee profile. Horde also has nontrivial insufficient-material and
fortress cases. A simple material count cannot safely decide all draws. The
pinned `scalachess` Horde implementation is the executable reference for those
edge cases; it includes explicit logic for lone Horde pieces, bishop colors,
smothered-mate resources and unavoidable closed-position stalemates.

This project must not replace those cases with either of these shortcuts:

- “White has any piece, therefore White can mate.”
- “Black has a king, therefore Black can always capture the Horde.”

Both statements fail in reachable edge positions.

## 4. The HordeTest representation

The game rule and the neural-network feature encoding are separate concerns.
The historical Horde NNUE architecture shared piece features across colors. A
black pawn and a white Horde pawn therefore used the same piece-type weights,
even though their roles are radically different.

HordeTest gives white Horde pawns a custom piece identity, `H`, while retaining
pawn movement semantics:

```ini
[hordetest:horde]
customPiece1 = h:fmWfceFifmnD
pawnTypes = ph
doubleStepRegionWhite = *1 *2
startFen = rnbqkbnr/pppppppp/8/1HH2HH1/HHHHHHHH/HHHHHHHH/HHHHHHHH/HHHHHHHH w kq - 0 1
pieceToCharTable = PNBRQH...............Kpnbrqh...............k
```

`H` is a feature-identity distinction, not a new game piece. The standard Horde
FEN and the HordeTest FEN describe the same game state after mechanically
mapping every white `P` to `H`. Legal moves, terminal results and perft counts
must remain identical under that mapping.

The Betza string `fmWfceFifmnD` supplies ordinary forward pawn movement,
diagonal capture, initial multi-step behavior and the no-capture double move.
The inherited Horde rule supplies the special en-passant regions. `pawnTypes =
ph` ensures that both the ordinary pawn and custom Horde pawn participate in
promotion, en passant and irreversible-move accounting.

This encoding exists to make the network input asymmetric. It must never be
used to justify a rule difference between `horde` and `hordetest`.

## 5. Search and evaluation consequences

Several standard-chess intuitions are unsafe in Horde:

1. **King safety has one direction.** Black king safety is terminally
   important. White king safety does not exist.
2. **Material is not symmetric.** A black piece and a white Horde piece do not
   have equal strategic roles merely because their nominal values match.
3. **Promotion pressure dominates many positions.** A single advanced Horde
   pawn may carry mating potential that a generic material scaler misses.
4. **Extinction distance matters.** Black is trying to remove every white
   piece, not merely obtain a material advantage.
5. **Fortresses matter.** Positions with very little material may be drawn for
   reasons that require geometry, bishop color and available self-blocking
   material.
6. **Color mirroring is not a free augmentation.** Mirroring a Horde position
   across colors changes the game semantics.

An evaluation or training pipeline must preserve side identity, custom-piece
identity, promotion outcomes, rule-terminal outcomes and the distinction
between mate and extinction. Data normalization must not silently recolor or
canonicalize positions as if the game were symmetric.

## 6. Executable invariants

The following are baseline invariants, not strength targets:

- Standard and HordeTest start positions contain 36 white Horde pieces and 16
  black pieces.
- The standard and `H`-encoded perft corpora have equal node counts.
- Start-position perft is `8`, `128`, `1274`, `23310` at depths one through
  four.
- The pinned en-passant position reaches `33781` nodes at depth four.
- Full queen, rook, bishop and knight promotions are generated.
- A rank-one double step does not create the forbidden black en-passant capture.
- A valid ordinary en-passant capture remains available.
- Capturing the final white piece is a black win.
- Checkmating the black king is a white win.
- A promoted white piece prevents premature extinction adjudication.
- FEN conversion `P` to `H` and back is lossless for white Horde pieces.

The complete release gates are specified in
[Testing and release contract](testing-and-release-contract.md).

## 7. Normative sources

- [Lichess Horde rules](https://lichess.org/variant/horde)
- [Pinned Lichess Horde implementation](https://github.com/lichess-org/scalachess/blob/d5d47c16f65a005ca68e19bab702b02f66dd888c/core/src/main/scala/variant/Horde.scala)
- [Pinned Lichess perft corpus](https://github.com/lichess-org/scalachess/blob/d5d47c16f65a005ca68e19bab702b02f66dd888c/test-kit/src/test/resources/horde.perft)
- [Pinned Fairy-Stockfish Horde implementation](https://github.com/fairy-stockfish/Fairy-Stockfish/blob/c19b5f6c66894fdb0e88d0dd100e3885f744760a/src/variant.cpp)

When prose, a historical binary and a pinned executable source disagree, the
disagreement is a release blocker. It must be resolved explicitly; it must not
be hidden by choosing whichever output is convenient.
