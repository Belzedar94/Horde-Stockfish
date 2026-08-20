# Horde NNUE V3 design

Status: engineering proposal for review. Nothing here changes the production
evaluator, and no training run is authorized by this document. The registered
Run 6B `HORDETEST_HP_LEGACY_V1` network remains the default. Every number below
is either read out of an existing authenticated receipt or recomputed from the
authenticated 250,000 position validation role; the recomputation scripts are
described in the appendix so any claim can be reproduced.

This document exists because the V2 program reached a measured dead end. It
states what the evidence says, where the V2 reasoning went wrong, and what a
version 3 has to change in order to be materially stronger rather than
differently shaped.

## 1. What happened, in numbers

### 1.1 The lambda tournament

The 20 August lambda screen trained the fresh legacy H/P control and the
selected V2 Rank-8 topology on the same authenticated 50,000,000 position
corpus distilled from Run 6B at depth 4, with identical labels, calibration,
optimizer, schedule, seed policy and sample order. Only the architecture and
the lambda differ.

Validation metrics after the single 50,000,000 example pass, from
`runs/*/metrics.jsonl` on the frozen 250,000 position role:

| Run | composite loss | result half-Brier | score half-Brier (eligible) |
| --- | ---: | ---: | ---: |
| `legacy-l0p4` | 0.109463 | 0.175077 | 0.011370 |
| `legacy-l0p6` | 0.074853 | 0.174209 | 0.008873 |
| `legacy-l0p8` | 0.040175 | 0.174617 | 0.006760 |
| `rank8-l0p4` | 0.109765 | 0.175755 | 0.011100 |
| `rank8-l0p6` | 0.075564 | 0.175493 | 0.009212 |
| `rank8-l0p8` | 0.041428 | 0.176244 | 0.007954 |

Three facts follow immediately.

1. The legacy topology fits the teacher better than Rank-8 at every lambda, and
   the gap widens as lambda rises. At lambda 0.8 the Rank-8 score half-Brier is
   17.7 percent higher than legacy (0.007954 against 0.006760).
2. The result half-Brier is effectively flat across lambda: 0.1742 to 0.1762
   over a range where the result term weight varies from 0.6 down to 0.2.
   Tripling the weight on the result does not improve result prediction.
3. Rank-8 is worse than legacy on the result term too, at every lambda.

The playing evidence agrees. In the internal very short time control matches
`matches/a1-legacy-l0p6-vs-l0p8` and `matches/a2-rank8-l0p6-vs-l0p8`:

| Match | Games | Result for lambda 0.6 | Elo | LOS |
| --- | ---: | --- | ---: | ---: |
| legacy 0.6 against legacy 0.8 | 80 | 26-52-2 | -117.2 (CI 67.8) | 0.010 percent |
| rank8 0.6 against rank8 0.8 | 120 | 46-74-0 | -82.6 (CI 46.4) | 0.013 percent |

Moving lambda from 0.6 to 0.8 is worth roughly 80 to 120 Elo. Changing the
architecture from legacy to Rank-8 at fixed lambda 0.8 is worth zero: the owner
reports 496-484-20 over 1000 very short time control games, with short and long
time controls also flat.

The single most important line in this document is the ratio between those two
numbers. The label term is worth about a hundred Elo. The architecture choice
under test is worth nothing measurable. V3 must therefore attack the labels
first and the architecture second, and it must not repeat a comparison whose
whole dynamic range is smaller than the noise.

Note on provenance: the 1000 game legacy against Rank-8 tournament is not
present under `D:\horde-train\matches`, which contains only the two lambda
pairings above plus two smoke runs. The tournament receipt should be filed
before the number is cited in any later document.

### 1.2 The teacher ceiling

The frozen WDL calibration `wdl-calibration-50M.json`, fitted on the whole
50,000,000 record training corpus, already reports the answer:

| Side | Records | Mean half-Brier of the fitted link |
| --- | ---: | ---: |
| white to move | 24,179,240 | 0.170013 |
| black to move | 24,199,451 | 0.169719 |

That is what a three parameter monotone function of the depth-4 teacher score
scores against the one hot game result. Recomputed on the exact 250,000
position validation role, with mate records included, the ladder is:

| Predictor of the game result | half-Brier |
| --- | ---: |
| constant per side base rate | 0.261860 |
| frozen parametric link on the teacher score | 0.168504 |
| cross fitted non parametric oracle on (side, score) | 0.166290 |
| best trained network in the screen (`legacy-l0p8`) | 0.174617 |

The trained networks are worse at predicting the game result than a calibrated
lookup on the teacher score they were distilled from. There is no result
information left for the loss to extract that the score does not already carry,
which is exactly why every step toward lambda 1.0 wins. The result term at
depth 4 is not a second source of truth; it is a noisy shadow of the same
source, and mixing it in at weight 0.6 costs about a hundred Elo.

The corpus itself explains why. Results are lopsided and scores are polarized:

- Black wins 61.50 percent of the sampled positions, White 30.20 percent, draws
  8.31 percent. Terminal reasons are extinction 153,744, checkmate 75,490,
  stalemate 15,806, fifty move 3,990, fivefold repetition 594, horde fortress
  376.
- 80.7 percent of records sit at an absolute teacher score above 400. Only 11.2
  percent lie inside the band where an evaluation is actually deciding
  something.
- In the band below -400 the side to move still wins 15.5 percent of the time
  and draws 7.4 percent. Above +400 it still loses 16.4 percent. The depth-4
  teacher declares a decisive position and is wrong about one time in five.

The decisive breakdown is by phase. `white_piece_count` is the Horde phase
variable, and the teacher's explanatory power collapses along it:

| white_piece_count | Records | Share | constant | link | Explained |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-4 | 24,200 | 9.68% | 0.20487 | 0.08955 | 56.3% |
| 5-8 | 34,844 | 13.94% | 0.22085 | 0.11131 | 49.6% |
| 9-12 | 29,491 | 11.80% | 0.24292 | 0.12562 | 48.3% |
| 13-18 | 41,568 | 16.63% | 0.27305 | 0.14664 | 46.3% |
| 19-24 | 45,654 | 18.26% | 0.29105 | 0.18132 | 37.7% |
| 25-30 | 54,342 | 21.74% | 0.28628 | 0.23464 | 18.0% |
| 31-36 | 19,901 | 7.96% | 0.27399 | 0.26388 | 3.7% |

In the opening and early middlegame, which is 29.7 percent of the corpus and
where the Horde's pawn mass actually is, the depth-4 teacher explains between 4
and 18 percent of the result. It is close to blind there. That is the binding
constraint on the whole distillation program, and it is a property of the
teacher, not of the student.

By side to move the picture is symmetric: the link explains 35.6 percent for
white to move and 35.7 percent for black to move. There is no side asymmetry to
exploit in the labels. The asymmetry that matters is phase.

### 1.3 Teacher and student agreement

Running the exact float forward of both frozen checkpoints over the validation
role reproduces the published metrics to within the subsample error
(`rank8-l0p8` score half-Brier 0.007951 against the published 0.007954), so the
breakdown below is trustworthy. MAE is the mean absolute difference between the
student's post rule-50 score and the stored teacher score, over eligible
records.

| white_piece_count | legacy-l0p8 MAE | sign agreement | rank8-l0p8 MAE | sign agreement |
| --- | ---: | ---: | ---: | ---: |
| 1-4 | 338 | 98.9% | 379 | 97.7% |
| 5-8 | 333 | 98.9% | 379 | 98.7% |
| 9-12 | 430 | 96.0% | 493 | 96.1% |
| 13-18 | 589 | 92.7% | 669 | 92.3% |
| 19-24 | 421 | 90.6% | 468 | 90.1% |
| 25-30 | 350 | 91.5% | 370 | 91.0% |
| 31-36 | 185 | 98.2% | 196 | 98.4% |

Three readings.

1. Legacy is closer to the teacher in every single phase bucket. There is no
   region where Rank-8 wins.
2. Neither student is close in absolute terms. A mean error of 350 to 670
   centipawns against the very teacher they were trained to copy, after
   50,000,000 examples, is not a small residual.
3. The two students agree with each other far more than either agrees with the
   teacher. Mean absolute difference between `legacy-l0p8` and `rank8-l0p8` is
   215 centipawns, against 397 and 433 respectively to the teacher. If their
   errors were independent the pairwise gap would be near 590. It is 215. The
   unexplained part of the teacher is a shared floor of these two designs, not
   an artifact of either topology.

Both students carry a systematic pro-Black tilt against the teacher: mean bias
-69 centipawns when White is to move and +73 when Black is to move for
`legacy-l0p8`, and -91 and +97 for `rank8-l0p8`. The Rank-8 tilt is about a
third larger. A network that shades the Horde down by 70 to 100 centipawns
relative to its own teacher will misplay the exact material sacrifices that
Horde play is made of.

Conditioning on the label flags stored in every record exposes where the eval
breaks:

| Subset | Share | legacy-l0p8 MAE | rank8-l0p8 MAE |
| --- | ---: | ---: | ---: |
| all eligible | 100% | 394 | 438 |
| quiet best move | 68.4% | 398 | 446 |
| best move is a capture | 28.6% | 383 | 419 |
| best move is a check | 4.2% | 701 | 793 |
| best move is a promotion | 1.8% | 915 | 1042 |
| royal in check | 4.1% | 723 | 830 |

Captures are not a weak spot, because the generator writes them (the datagen
contract states that capture, check and promotion samples are measured rather
than filtered out). Checks, promotions and positions where the Black king is
already in check are a very large weak spot, at roughly 1.8 to 2.4 times the
average error. Promotion is the central Horde mechanic and it carries the worst
evaluation error in the entire corpus.

### 1.4 Capacity audit

The two serialized layouts, derived exactly and reconciled to the byte against
the shipped files.

Legacy `.nnue`, 1,088,361 bytes, schema `HORDETEST_HP_FRESH_CONTROL_V1`,
topology `896 -> 512 shared FT + PSQT; 8 x (1024 -> 16 -> 32 -> 1)`:

| Section | Shape | dtype | Bytes | Share |
| --- | --- | --- | ---: | ---: |
| feature transformer weights | 896 x 512 | i16 | 917,504 | 84.3% |
| feature transformer bias | 512 | i16 | 1,024 | 0.1% |
| PSQT | 896 x 8 | i32 | 28,672 | 2.6% |
| 8 layer stacks (16x1024, 32x16 padded to 32x32, 1x32, plus hashes) | | i8/i32 | 141,120 | 13.0% |
| file header, 25 byte description, transformer hash | | | 41 | 0.004% |

Those five rows sum to exactly 1,088,361 bytes, so the layout is derived, not
estimated.

Rank-8 `.hsv2`, 941,596 bytes of which 936,264 are parameters, schema
`V2_C1_ROYAL_RANK8_64X192`:

| Section | Shape | dtype | Bytes | Share of parameters |
| --- | --- | --- | ---: | ---: |
| royal rank8 weights | 5,120 x 64 | i16 | 655,360 | 70.0% |
| royal rank8 bias | 64 | i32 | 256 | 0.03% |
| global weights | 704 x 192 | i16 | 270,336 | 28.9% |
| global bias | 192 | i32 | 768 | 0.08% |
| hidden0 | 32 x 256 | i8 | 8,192 | 0.87% |
| hidden1 | 32 x 32 | i8 | 1,024 | 0.11% |
| output | 2 x 32 | i8 | 64 | 0.01% |
| biases | | i32 | 264 | 0.03% |

Side by side, in the quantities that matter for expressivity:

| Quantity | Legacy H/P | V2 Rank-8 | Ratio |
| --- | ---: | ---: | ---: |
| float parameters | 602,248 | 472,450 | 0.78 |
| activated lanes entering the dense trunk | 1,024 | 256 | 0.25 |
| first dense layer MACs per evaluation | 16,384 | 8,192 | 0.50 |
| output conditioning | 8 material buckets plus a bucketed PSQT skip | 1 bucket, 2 STM rows, no skip | |
| parameters in the mixing layers | 142,984 (23.7%) | 9,346 (2.0%) | 0.07 |
| parameters in the largest single table | 458,752 (76.2%) | 327,680 (69.4%) | |

The file sizes are similar, so the two networks look comparable on disk. They
are not comparable in the only sense that matters. V2 Rank-8 has one quarter of
legacy's activated width, one half of its first layer arithmetic, one
fourteenth of its mixing capacity, and no output bucketing at all. It spends 70
percent of its weights on a table which is visited fifty one rows at a time
through sixty four lanes.

The gradient telemetry in the Rank-8 receipt confirms the imbalance directly.
First step gradient norms by group: royal transformer 1.90e-5, global
transformer 4.45e-5, dense trunk 1.18e-3, output 1.85e-2. The largest table in
the network receives the weakest gradient in the network.

The width gate that produced this shape is `nnue-v2-width-receipt.json`:

| Width | Median NPS | Paired 95% CI ratio | Verdict |
| --- | ---: | --- | --- |
| `64+192` | 1,180,947 | 1.0000 | pass |
| `128+128` | 1,168,659 | 0.9876 to 0.9975 | pass |
| `128+256` | 1,051,148 | 0.8856 to 0.8961 | fail |
| `256+256` | 939,137 | 0.7934 to 0.8007 | fail |

That gate is methodologically sound and its measurements are not in dispute.
The problem is ordering. It ran before any evidence existed about how much
capacity the task needs, and its four candidates all sat below legacy's own
activated width. A speed gate applied to a candidate set that never contained a
legacy sized point can only select a small network. V3 reverses the order:
establish the capacity requirement at fixed nodes first, then pay for it or
trade it away with a speed gate that includes the incumbent as a point.

### 1.5 Feature audit

The question is which feature families carry information, measured on the two
axes that matter, both computed on the authenticated validation role with
cross-fitted estimators and explicit null controls.

Axis one, the game result, measured as the half-Brier reduction an additive
correction can buy on top of a perfect reproduction of the teacher score. Null
controls score negative by construction, so any positive number is real.

| Block, incremental over the score | Gain | Share of the score signal |
| --- | ---: | ---: |
| random 400 way one hot (null control) | -0.001855 | |
| random 64 column gaussian (null control) | -0.001487 | |
| full G0 piece square, 704 rows | +0.001204 | +1.26% |
| black king rank, 8 cells | -0.001219 | |
| royal32 king bucket, 32 cells | -0.001145 | |
| white_piece_count, 37 cells | -0.000638 | |
| every pawn structure block together, 240 columns | -0.000136 | |

Incremental over G0 rather than over the score alone:

| Added to G0 | Incremental gain |
| --- | ---: |
| royal32 king bucket | +0.000002 |
| white_piece_count | +0.000332 |
| all pawn structure blocks | +0.000376 |

By phase, incremental over G0:

| white_piece_count | royal32 | pawn blocks |
| --- | ---: | ---: |
| 1-4 | -0.000018 | +0.000894 |
| 5-8 | -0.000001 | +0.001698 |
| 9-12 | +0.000004 | +0.000890 |
| 13-18 | +0.000024 | +0.000439 |
| 19-24 | -0.000000 | +0.000170 |
| 25-30 | -0.000002 | +0.000088 |
| 31-36 | -0.000038 | +0.000021 |

Axis two, the teacher score itself, which is what lambda 0.8 actually
optimizes. Cross fitted ridge, side to move interacted, on 120,000 eligible
validation records, target clipped at plus or minus 3000:

| Linear model | Columns | R squared | MAE |
| --- | ---: | ---: | ---: |
| side to move only | 2 | 0.0096 | 1,269 |
| royal32 king bucket | 66 | 0.0479 | 1,215 |
| white_piece_count | 76 | 0.2443 | 966 |
| G0 piece square | 1,410 | 0.6912 | 657 |
| G0 + royal32 | 1,474 | 0.6906 | 658 |
| G0 + pawn structure blocks | 1,906 | 0.7337 | 597 |
| G0 + white_piece_count | 1,484 | 0.7500 | 574 |
| G0 + pawn + white_piece_count + royal32 | 2,044 | 0.7656 | 553 |

Incremental R squared over G0: royal32 -0.0006, pawn blocks +0.0426,
white_piece_count +0.0589, all together +0.0744.

By phase, incremental R squared over G0:

| white_piece_count | royal32 | pawn blocks |
| --- | ---: | ---: |
| 1-4 | +0.0037 | +0.0086 |
| 5-8 | +0.0009 | +0.0186 |
| 9-12 | +0.0007 | +0.0292 |
| 13-18 | -0.0005 | +0.0306 |
| 19-24 | +0.0003 | +0.0388 |
| 25-30 | +0.0005 | +0.0527 |
| 31-36 | -0.0000 | +0.0126 |

Two independent measurements, on two different targets, give the same ordering.
Royal context adds nothing over the base piece square stream on either axis, in
any phase. Pawn structure and the White piece count both add real, phase graded
signal, with pawn structure peaking exactly in the 19 to 30 piece band that is
40 percent of the corpus and where the teacher is most blind.

For scale: adding those blocks linearly moves the teacher fit MAE from 657 to
553 centipawns, a reduction of 104. The entire nonlinear apparatus of the
trained 472,450 parameter Rank-8 network moves it from 657 to 433, a reduction
of 224. The deferred blocks are worth roughly half of what the whole trained
network buys over a linear probe, and they are currently absent.

A caveat that must be stated plainly: a linear probe understates what a
nonlinear network can recover from G0 alone, so these numbers are not proof
that the current networks fail to recover pawn structure implicitly. They are
strong evidence, not a receipt. The receipt is a trained ablation, which is
rung R5 below.

### 1.6 The working hypothesis, judged

The hypothesis under review was: V2 bought royal structure, which matters
little, paying for it with capacity, and left out all of the pawn structure,
which is what decides Horde games.

**Confirmed, with one important correction.**

Confirmed on royal structure. The Black king moves on 16.5 percent of mainline
moves per the V2 book probe. In this validation role 85.7 percent of positions
have the Black king on rank 7 or rank 8 (150,346 and 63,940 of 250,000), so two
of the eight Rank-8 buckets carry 86 percent of the traffic while the table
holds 5,120 rows. Royal context adds +0.000002 half-Brier over G0 on the result
and -0.0006 R squared on the teacher score. It is not merely small. It is zero.

Confirmed on capacity. Quantified in 1.4: one quarter of legacy's activated
width, one fourteenth of its mixing parameters, no output bucketing, and 70
percent of the weight budget parked in the least useful and least trained
table.

Confirmed on pawn structure. Quantified in 1.5, and the phase profile matches
what `first-principles-horde.md` asserts: promotion pressure, extinction
distance and the shape of the Horde mass are what decide these games. The
structure is dense rather than sparse (mean 17.37 White pawns per position,
54.2 percent of them blocked, 67.6 percent in a phalanx, 63.9 percent
diagonally supported, 85.0 percent of positions with a White pawn on the sixth
rank or beyond, 29.3 percent with one on the seventh), which is precisely why
the boundary oriented per file encoding in the V2 ladder is the right shape and
why it should not have been deferred behind the royal experiments.

The correction concerns how the Rank-8 selection was reached. The decisive C1
comparison was Rank-8 against `V2_C1_ABS_NONKING_64X192`, whose first domain is
the ten absolute non king role planes, 640 rows with no king bucket and no
reflection. Those rows are a strict subset of the Global stream that the same
network already consumes at 192 lanes. The absolute control therefore spends 64
lanes re-encoding information the network already has. Rank-8 beating it by 61
Elo does not show that royal context is useful. It shows that an informative
first domain beats a redundant one, which was never in question. The control
that would have settled the matter is a single G0 table at 256 lanes, which
exists in the codebase as `C0SingleG0Model` but was only ever run as a
numerical equality receipt, never as a strength test. V3 must run it.

## 2. V3 design

Three levers, coupled. Any one of them alone reproduces the V2 outcome.

### 2.1 Lever A: data and labels

The teacher is the binding constraint. Section 1.2 shows the depth-4 teacher
explains 3.7 percent of the result at 31 to 36 White pieces and 18.0 percent at
25 to 30, and section 1.3 shows the students are 350 to 670 centipawns away
from it anyway. Improving the student against that target has a low ceiling.

**A1. A deeper teacher.** Replace the depth 4, nodes 0 generation setting with
a node budget or a materially deeper search, using the winner of the R4 gate as
the teacher network. Spell generates at 10,000 nodes; that is the reference
point for a sane budget. The generation contract fields already exist
(`generation.depth`, `generation.nodes`), so this is a parameter change plus a
new campaign schema, not new code. The cost is throughput and it must be
measured before committing (probe P3 below).

**A2. Port the Spell tactical expansion, after re-diagnosing it for Horde.**
The Spell receipts are in `AUDIT.md` at commit `aeb9e314`: for every written
position, also label the single child after the recorded best move at depth 2,
giving 2,000,000 quiet positions plus 1,000,000 children, measured at +50.74
plus or minus 34 over 400 very short time control games with LOS 99.9 percent.
The follow up round with 5,800,000 children measured +57.9, +31.4 and +86.9
across three time controls. The `+55` figure that circulates is a rounded
retrospective summary and does not appear as a measured value in any receipt;
the measured numbers are +50.74 and +57.9.

Lambda 1.0 is mandatory for the expanded part. Blending it into a lambda 0.75
mixture regressed by 72 Elo at short and 70 at long time control, and lambda 0.9
was also discarded.

The mechanism ports. The diagnosis does not. Spell's gap was that its datagen
skipped captures and in check positions, so the corpus was quiet only. Horde's
datagen explicitly does not filter those. The Horde specific gap, measured in
1.3, is different in shape and just as large: checks at 701 centipawns of error,
promotions at 915, royal in check at 723, against 394 overall. The Horde port
should therefore expand along the axis where Horde is blind, which is forcing
and promoting lines, not the capture axis Spell needed.

Concretely: for each written position, additionally write the position after
the recorded best move, and additionally expand every child reachable by a
promotion or a check when one exists, labelled at the generation depth. Retain
the expanded records with lambda 1.0 and a distinct provenance marker in the
manifest so the mixture ratio is auditable and the expansion can be ablated.

The tooling is not in the Spell repository. `relabel.py`,
`relabel_expand.py`, `uci_probe.py` and `concat_shuffle.py` live in an external
lab folder referenced from `AUDIT.md`. Horde should implement the expansion
inside the existing `horde-stockfish-data-generator` artifact instead, so that
the producer hash keeps binding the labels.

**A3. Cumulative corpus.** The recommendation is to accumulate rather than
replace: generation N trains on generation 1 through N, not on generation N
alone.

Honesty note on the evidence. The specific claim that Spell's generation 2 used
111,000,000 records made of 51,000,000 from generation 1 plus 60,000,000 new is
not written down anywhere in the Spell repository. It is recoverable only from
training logs: the generation 2 only runs consumed 60,000,000 records per epoch
and the `run2f` runs consumed 111,000,000, which is arithmetically consistent.
No document states a measured Elo comparison between the 60,000,000 only and the
111,000,000 nets, and the documented cause of the generation 2 gate failure is
something else entirely, namely an inverted scaling direction between label and
network output worth -32 Elo at very short time control.

What is documented, in `AUDIT.md` around 25 July, is the same failure family
from a different experiment: continuing training on raw datagen alone was
crushed at -224 Elo with a 0.0 percent gate, described as catastrophic
forgetting of mined tactical knowledge, and a mixture of mined and raw data
still regressed by 51 Elo. That is a genuine receipt for "new data alone
destroys prior knowledge, and mixing damps but does not cure". It supports
accumulation. It is not the 51 plus 60 equals 111 receipt, and this document
does not claim it is.

**A4. The reinforcement loop, later.** Generation 2 self play from the V3
champion is a follow up, gated on A1 through A3 landing and on the Spell
scaling lesson being encoded as a forward parity test before the first
generation 2 run rather than after.

### 2.2 Lever B: architecture

Inherited from V2 without change: the fixed White and Horde frame with no White
king feature and exactly one required Black king, forbidden vertical flipping,
horizontal reflection allowed as specified, positive output meaning good for
the side to move, the deterministic integer contract, the authenticated
container with fail closed dispatch, and the accumulator stack contract driven
by the engine's `DirtyPiece` trace. That machinery is correct, it has receipts,
and V3 extends its schema rather than replacing it.

**B1. One primary domain at legacy activated width.**

Drop the separate royal transformer. The base V3 network has a single sparse
transformer over the fixed frame:

```text
rows   = 704 G0 physical piece square
       + 56  frontier White pawn square, ranks 1 to 7
       + 56  rearmost White pawn square, ranks 1 to 7
       + 56  White pawn count by file, states 1 to 7
       + 8   frontmost White pawn blocked, per file
       + 8   frontier pawn diagonally supported, per file
       + 8   frontier pawn in a same rank phalanx, per file
       = 896 rows
lanes  = 1024
```

The row budget lands on 896, the same as the legacy feature dimension, by
coincidence rather than design; the point is that the contextual blocks cost
192 rows on top of G0, which is 27 percent more rows and no extra activated
lanes.

The width choice is not arbitrary. Legacy reaches 1,024 activated lanes by
running two 512 lane perspectives over a shared table. The Horde frame forbids
the color flip, so V3 reaches the same activated width with one 1,024 lane
table. The arithmetic is identical:

| Operation | Legacy, 2 x 512 | V3, 1 x 1024 |
| --- | ---: | ---: |
| full refresh lane operations | 52 rows x 512 x 2 = 53,248 | 52 rows x 1024 = 53,248 |
| quiet move lane operations | 2 x 2 x 512 = 2,048 | 2 x 1024 = 2,048 |
| gathers per refresh | 2 | 1 |
| first dense layer MACs | 16 x 1024 = 16,384 | 16 x 1024 = 16,384 |

V3 does strictly less work per refresh than legacy because it runs one gather
instead of two, and identical work per incremental update, before the
contextual blocks. Those blocks add bounded row operations: at most 8 active
rows per block, and the V2 design already bounds a worst case bundled update
across two changed files at 16 transformer row operations.

The costs that do rise are table size and accumulator frame size. The
transformer table is 896 x 1024 x 2 bytes, or 1,835,008 bytes, against legacy's
917,504. The accumulator frame is 1,024 i32 lanes, 4,096 bytes, against
legacy's 2,048 bytes of i16 accumulators. Both are cache pressure, both are
measurable, and probe P1 measures them.

**B2. Output buckets on `white_piece_count`.**

The single largest linear block gain measured anywhere in section 1.5 is
`white_piece_count`, at +0.0589 R squared over G0 against the teacher score.
Legacy's current bucket is on total material and conflates a 30 pawn Horde
against 4 Black pieces with a 10 pawn Horde against 15. Its occupancy is also
badly skewed on this corpus: bucket 7 holds 3,034 of 250,000 records.

Base V3 uses 8 buckets from an exact serialized lookup on `white_piece_count`
in 0 to 36, chosen for balanced occupancy on the training split rather than by
an arithmetic formula, times 2 side to move rows, for 16 heads. Occupancy of
`white_piece_count` on the validation role is well spread from 1 to 32 and thin
only at 33 to 36 (3,735, 1,622, 436 and 62 records), so the top bucket must
absorb the tail.

A two dimensional grid on `white_piece_count` by Black material, in the style
of Spell's 4 by 4 material by potions grid, is registered as a later rung, not
as part of the base.

**B3. Dense trunk matched to legacy, then widened as a rung.**

Base V3 uses `1024 -> 16 -> 32 -> 1` per bucket, which is exactly legacy's
dense shape, plus a bucketed PSQT skip of 896 x 8. This keeps the base V3
comparison against legacy a clean test of the feature and width changes with
the dense arithmetic held constant. Widening the first hidden layer from 16 to
32 doubles the dense MACs and is a separate rung with its own speed gate.

Base V3 parameter budget:

| Section | Shape | dtype | Bytes |
| --- | --- | --- | ---: |
| transformer weights | 896 x 1024 | i16 | 1,835,008 |
| transformer bias | 1024 | i32 | 4,096 |
| PSQT | 896 x 8 | i32 | 28,672 |
| 8 buckets x (16x1024, 32x16, 2x32 and biases) | | i8/i32 | 137,280 |
| total parameters | | | 2,005,056 |

That is 1.91 MiB, against legacy's 1.04 MiB and Rank-8's 0.89 MiB. The
activated width matches legacy, the mixing capacity matches legacy at about
143,000 parameters against Rank-8's 9,300, and no weight is spent on a table
that measures zero.

**B4. Royal context returns only if it earns a rung.**

The Black king remains a role in G0, exactly as it is today. The Rank-8 royal
domain is retired from the base. If it is ever revisited it must first beat a
single G0 table at equal activated lanes, which is the control that was never
run.

**B5. Integer contract.**

Extend `HORDE_V2_INTEGER_NETWORK_V1` rather than replacing it. Register schema
`0x00020001` as `V3_G1024_PAWN_WPC8`. Keep the frozen conversion unchanged:
round to nearest with ties to even, feature transform scale 8,128, dense weight
scale 64, signed i16 feature weights, signed i8 dense weights, signed i32
biases and accumulators, both activations computing `clip(max(affine, 0) >> 6,
0, 127)`, selected output divided by 16 with truncation toward zero, then the
versioned rule-50 postprocessor, then the tablebase safe clamp, every bias
bounded to magnitude 2^30. New sections are needed for the contextual block
ranges, their structural hashes, and the `white_piece_count` bucket lookup
table, which must be serialized explicitly and never inferred.

The contextual blocks require the invalidation discipline the V2 design already
specifies: every accumulator frame retains the per file categorical codes and
the frontier and rearmost bitboards, an incremental update recomputes only the
candidate files, the candidate set is derived from every physically changed
square rather than from the nominal move source and destination, and undo
restores the saved source frame rather than inferring old contextual roles.

### 2.3 Lever C: protocol

The V2 receipt discipline is retained in full: same split, labels, optimizer,
schedule, filters, three seeds, seed one predesignated for any playing gate
before metrics exist, manifests binding trainer commit, dataset hashes,
structural schema hash, seed, validation metrics and refresh rates.

Two changes to the order of gates, both forced by section 1.4.

First, fixed node before equal time, always, and the incumbent is a point in
every speed comparison. The V2 width gate compared four V2 candidates against
each other and never against legacy, so a candidate could pass the speed gate
while being slower than the network it was supposed to replace.

Second, no speed gate runs before a capacity gate has established what the task
needs. The ladder below runs capacity at fixed nodes first.

The rungs:

| Rung | Question | Design | Gate |
| --- | --- | --- | --- |
| R0 | Does the shipped 50M champion beat Run 6B at all? | `legacy-l0p8` and the pending `l1p0` against Run 6B | three time controls, this is the fork in section 3 |
| R1 | Is lambda 1.0 better than 0.8? | existing `l1p0` runs, both architectures | very short time control pairing, then three time controls |
| R2 | Single G0 table against the V2 dual domain at equal lanes | `G0_SINGLE_256` against `rank8-64x192`, three seeds | fixed node, the control C1 never ran |
| R3 | Does activated width buy strength? | one G0 table at 256, 512 and 1024 lanes, dense held at 16/32/1, one bucket | fixed node first, then measure NPS |
| R4 | Do output buckets on `white_piece_count` buy strength? | best R3 width, 1 bucket against 8 buckets | fixed node, then equal time |
| R5 | Do the pawn blocks buy strength? | best R4 point, with and without the 192 contextual rows | fixed node, then equal time |
| R6 | Assemble and gate against Run 6B | base V3 against legacy at the same corpus | three time controls under the release contract |

Each rung changes one named thing. A network differing in more than its named
rung is not a valid ablation, and two individually losing blocks are combined
only after both individual receipts exist.

Lambda policy: every rung from R2 onward runs at the best lambda from R1, and
the expanded records from A2, when they exist, always enter at lambda 1.0
regardless of the base lambda.

## 3. Decision forks

**Fork 1, the primary one. Does the 50M champion beat Run 6B?**

This is R0 and it must run before anything else, because it decides which lever
goes first.

If the champion does not beat Run 6B, the binding constraint is the labels and
lever A goes first. A student trained on 50,000,000 depth-4 labels that cannot
beat the very network that produced them has exhausted what those labels
contain, and the next 50,000,000 of the same will not help. In that branch the
order is A1 and A2, regenerate, then B.

If the champion beats Run 6B comfortably, the labels still have room, the
binding constraint is capacity, and lever B goes first at the existing corpus,
which is much cheaper because no regeneration is needed.

The evidence in section 1.2 leans toward the first branch, but leaning is not
measuring. R0 decides.

**Fork 2. Does lambda 1.0 beat 0.8?**

Two lambda 1.0 trainings are running. If 1.0 wins, the result term is confirmed
as pure noise at depth 4 and it should be dropped from the objective entirely
for the current teacher, which also removes a term from every future comparison.
If 0.8 wins, a small result weight is still buying calibration and the schedule
should stay mixed.

**Fork 3. What does a deeper teacher cost?**

Unmeasured. If generation throughput at the new depth or node budget makes a
50,000,000 corpus cost more than a few days on the available fleet, the answer
may be a smaller but much better labelled corpus rather than the same size at
greater depth. Probe P3 decides.

**Fork 4. Does 1,024 activated lanes hold NPS?**

Unmeasured, and unmeasurable today. Section 2.2 argues the incremental
arithmetic is identical to legacy and the refresh is cheaper by one gather, but
the table is twice as large and the accumulator frame is twice as large, so
cache behaviour could still cost real NPS. Probe P1 decides. If 1,024 fails,
512 lanes is the fallback and it still doubles Rank-8.

**Fork 5. Where do the expanded records come from?**

Implementing the expansion inside the data generator keeps the producer hash
binding the labels but requires engine work. An external relabelling tool, in
the style of Spell's lab scripts, is faster to build but breaks the provenance
chain that every Horde receipt currently depends on. The recommendation is the
in engine route; the fork is open because it changes the schedule materially.

## 4. Cost, schedule and parallelism

Measured facts: one 50,000,000 record training pass is 12,208 optimizer steps
at batch 4,096 on the RTX 3080, single CPU thread for the loader, deterministic
algorithms enabled, no AMP, no TF32. The owner reports 3.5 to 4 hours per cell
today. There is one GPU, so trainings serialize.

| Phase | Work | GPU hours | Wall time with one GPU | Parallel with |
| --- | --- | ---: | --- | --- |
| R0 | no training, matches only | 0 | hours | anything |
| R1 | no new training, the two `l1p0` runs are in flight | 0 | in progress | anything |
| R2 | 3 seeds of `G0_SINGLE_256` | 11 to 12 | half a day | match play |
| R3 | 3 widths x 3 seeds | 32 to 36 | one and a half days | match play |
| R4 | 2 bucket variants x 3 seeds | 21 to 24 | one day | match play |
| R5 | 2 feature variants x 3 seeds | 21 to 24 | one day | match play |
| R6 | 3 seeds of base V3 | 11 to 12 | half a day | match play |
| A1 and A2 regeneration | 50,000,000 records at the new teacher setting | 0 | unmeasured, see P3 | all training |

Total training for the B ladder is roughly 100 to 110 GPU hours, four to five
days of continuous GPU time, if no rung fails and forces a repeat. Budget seven
to eight days.

What parallelizes: data generation runs on CPU and the fleet while the GPU
trains, so lever A regeneration should start as soon as fork 1 and fork 3 are
resolved and should overlap the entire B ladder. Match play for a finished rung
overlaps the training of the next. What does not parallelize: rungs, because
each one selects the base of the next, and trainings, because there is one GPU.

The cheapest possible schedule is therefore: run R0 and R1 immediately since
they need no new training, start A1 and A2 generation the moment fork 1 is
decided, and run the B ladder against the existing 50,000,000 corpus while the
new corpus is being generated. The B ladder answers architecture questions that
do not depend on which corpus is used, so its conclusions carry over.

## 5. Speed probes, to be run when the machine is idle

No timing measurement in this document was taken today, and none should be
until the two lambda 1.0 trainings and the three match runners have finished.
The following are specified and ready.

**P1, activated width against NPS.** Extend the existing width matrix with the
V3 points and add the incumbent. Build the four AVX2 binaries sequentially on
one runner exactly as the V2 gate did, randomize paired run order under a frozen
seed, retain every raw sample, and require the headline NPS ratio interval to
have a half width no larger than 0.5 percent.

```console
python tests/horde_v2_engine_widths.py --widths 64+192,256,512,1024 \
  --include-incumbent run6b --payload PERF_COMMON_V1 --seed <frozen> \
  --output docs/horde/nnue-v3-width-receipt.json
```

Accept the timings only after every width returns identical per position root
evaluations, root scores, node counts, best moves and trace hashes.

**P2, contextual block update cost.** Measure the incremental update cost of
the 192 contextual rows in isolation: frame copy, block recomputation on a
single changed file, worst case across two changed files, a Black king move, a
castling, and a promotion. Report average and maximum removed and added rows per
block, which the V2 telemetry contract already requires.

**P3, generation throughput at the new teacher.** Measure records per hour per
core at the current depth 4 setting and at each candidate setting, on one
worker, and extrapolate the fleet cost of 50,000,000 records.

```console
horde-stockfish-data-generator --depth 4 --nodes 0 --records 250000 ...
horde-stockfish-data-generator --depth 8 --nodes 0 --records 250000 ...
horde-stockfish-data-generator --depth 0 --nodes 10000 --records 250000 ...
```

**P4, accumulator frame pressure.** Measure search NPS with the V3 frame at 512
and 1,024 lanes against legacy at realistic search depth, since the frame stack
grows from 2,048 to 4,096 bytes per ply.

## 6. Open receipt discrepancies

Raised rather than papered over.

1. **The 1000 game legacy against Rank-8 tournament has no receipt on disk.**
   `D:\horde-train\matches` holds only the two lambda pairings and two smoke
   runs. The 496-484-20 result and the short and long time control results are
   quoted from the owner. They should be filed before being cited further.

2. **The registered campaign schema does not describe the campaign that ran.**
   `schemas/horde-v2-rank8-scale-v1.json` in the repository, SHA-256
   `B8A8512D...`, names teacher source commit `491e5227` and producer
   `50B79890...`. The shipped 50,000,000 corpus binds instead to
   `horde-v2-rank8-scale-v1.LOCAL.json`, SHA-256 `EFD8CF7A...`, marked
   `local_staging_not_registered`, which corrects the teacher commit to
   `bea049e7`. The divergence is documented inside the local file, so it is
   deliberate, but the repository's registered schema currently describes a run
   that never happened.

3. **The producer that generated the training corpus is not in either allowed
   list.** The training chunk set's common manifest names producer
   `EEBD8977...`. The local contract allows `4F20645A...` and `F29CD034...` for
   training, and `F6D545E5...` for the validation candidate. The validation
   candidate matches. The training producer does not appear in either the
   repository contract or the local contract. This may be benign, since the
   chunk set carries its own `producer_sha256_allowed` list, but the campaign
   verifier should be run against the local contract to confirm which identity
   is authoritative.

4. **Two different byte counts circulate for the Rank-8 network.** 936,264 is
   the parameter payload and 941,596 is the container file, the difference
   being the 2,048 byte header plus 2,594 bytes of section structure plus 690
   bytes of provenance. Both are correct; they are not interchangeable.

5. **The C1 selection compared Rank-8 against a redundant control.** Detailed
   in section 1.6. This is not a bookkeeping error, it is an inference error,
   and it is the reason the Rank-8 topology was carried to a 50,000,000 record
   scale run.

## 7. Appendix: reproducing the numbers

Every recomputed figure comes from four read only scripts run against
`D:\horde-train\validation-selected\selected-records.bin`, whose SHA-256
`DA97A6C8E392214100B28C2457EC2430316118EEA1932DFAA07DC152B3B28F85` matches the
selected role receipt, and against the frozen calibration
`wdl-calibration-50M.json`, SHA-256
`3FF1C4F5ACF3D096B8DF6DD2DD3CC06474F9F779435B4FB83D36FBDB87045486`, which
matches the identity recorded in every training receipt.

- corpus shape, result distribution, teacher ceiling by phase and by side, and
  pawn structure density;
- honest cross fitted block value on the result axis, with null controls;
- side to move interacted linear recoverability of the teacher score, by block
  and by phase;
- teacher and student agreement, running the exact float forward of the frozen
  checkpoints, validated against the published `metrics.jsonl` to within
  subsample error.

The forward reimplementation reproduces `rank8-l0p8` score half-Brier as
0.007951 against the published 0.007954 and `legacy-l0p8` as 0.006744 against
0.006760, on a 40,000 record subsample of the same role.
