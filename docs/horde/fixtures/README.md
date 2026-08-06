# Rule fixtures

`lichess-horde.perft` is an exact copy of the Horde perft corpus at
`lichess-org/scalachess@d5d47c16f65a005ca68e19bab702b02f66dd888c`.
The upstream file is MIT-licensed:

<https://github.com/lichess-org/scalachess/blob/d5d47c16f65a005ca68e19bab702b02f66dd888c/test-kit/src/test/resources/horde.perft>

`hordetest.perft` is the mechanically derived HordeTest encoding of the same
positions. Every white `P` is represented as the custom `H` piece. Node counts
must remain identical. The transformation changes NNUE feature identity, not
legal game semantics.

`variants.ini` is the frozen custom-variant definition used by the Run 6B
baseline. Changing any character in it creates a different baseline and
requires a new manifest version.
