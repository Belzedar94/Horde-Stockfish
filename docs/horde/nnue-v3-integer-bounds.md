# Horde NNUE V3 integer bounds and output scale

Status: engineering analysis. Two questions had to be answered before a V3
container schema could be registered. The first closes cleanly. The second does
not, and it blocks the container work until the owner rules on it.

## 1. Signed 32-bit bounds at 1024 lanes

The V2 contract fixes signed `int16` feature weights, signed `int8` dense
weights, signed `int32` biases and accumulators, a maximum bias magnitude of
`2^30`, and non-saturating accumulation. V3 widens the transformer from 256
activated lanes to 1024 and lengthens the active row list, so every accumulator
bound has to be re-derived.

Active row bound. G0 contributes one row per physical piece, at most 52. The six
contextual blocks are each keyed on the frontier pawn of a file, so each
contributes at most one row per file, giving at most 8 rows per block and 48 in
total. The V3 stream therefore activates at most 100 rows. Measured on 20,000
authenticated validation records the observed maximum is 45 contextual rows, and
on 4,000 randomized boards it is 42, so the analytic bound of 48 holds with
margin and is not tight by accident.

| Sum | Worst case | Limit | Headroom |
| --- | ---: | ---: | ---: |
| feature accumulator: bias + 100 rows of `int16` | 2^30 + 100 x 32,767 = 1,077,018,524 | 2,147,483,647 | 1.99x |
| hidden0 affine: bias + 1024 x `int8` x `uint8` | 2^30 + 1024 x 127 x 127 = 1,090,257,920 | 2,147,483,647 | 1.97x |
| hidden1 affine: bias + 16 terms | 2^30 + 16 x 127 x 127 = 1,073,999,888 | 2,147,483,647 | 2.00x |
| output affine: bias + 32 terms | 2^30 + 32 x 127 x 127 = 1,074,257,952 | 2,147,483,647 | 2.00x |
| output plus PSQT skip: 100 PSQT rows | above + 100 x 2^20 = 1,179,115,552 | 2,147,483,647 | 1.82x |

Every sum stays inside signed 32-bit range without saturation, so the frozen
accumulator and bias types carry over to 1024 lanes unchanged. The widening does
not consume the existing safety factor: the binding term is the `2^30` bias
bound, not the widened dot product, which is why the headroom barely moves.

One new bound is required. V2 has no PSQT skip and therefore no bound on PSQT
magnitudes. V3 restores the skip, so the contract must add:

```text
maximum_psqt_magnitude = 2^20
```

At the legacy PSQT scale of 9,600 that admits float PSQT values up to 109.2,
against a largest observed legacy magnitude of 18,317, which is 1.9 units. The
bound is therefore not restrictive in practice and keeps the final sum inside
range with a factor of 1.82.

## 2. The output scale does not close, and V2 has a defect

The V3 container was meant to inherit the frozen V2 integer scales unchanged.
It cannot, because those scales are internally inconsistent, and the
inconsistency is a live defect in the shipped V2 export path rather than a V3
design question.

The trainer defines `NNUE_TO_SCORE = 600` and applies `trunc(v * 600)` inside
its rule-50 postprocessor, so the trainer's loss models an engine whose value is
`600 * v`.

The legacy exporter honours that exactly:

```text
OUTPUT_BIAS_SCALE   = 600 * 16 = 9600
OUTPUT_WEIGHT_SCALE = 9600 / 127
output_int          = sum(round(w * 9600/127) * round(127 * h)) + round(b * 9600)
                    = 9600 * v
value               = output_int / OutputScale(16) = 600 * v
```

`src/nnue/nnue_architecture.h` states the intent in the source: the comment
reads that 1.0 should equal `600 * OutputScale`.

The V2 container exporter does not. In `tools/horde_v2_export.py` the section
scale rule assigns `DENSE_SCALE` to every non-bias dense section and `FT_SCALE`
to every bias, so the output layer is quantized at 64 and 8,128 rather than at
9,600/127 and 9,600:

```text
output_int = sum(round(w * 64) * round(127 * h)) + round(b * 8128) = 8128 * v
value      = output_int / OUTPUT_DIVISOR(16) = 508 * v
```

Every exported V2 network therefore evaluates at `508 * v` where the trainer
optimized `600 * v`, a uniform compression of all evaluations toward zero by a
factor of 0.84667.

Three independent lines of evidence agree.

1. **Algebra.** 8,128 / 16 = 508, against `NNUE_TO_SCORE` of 600.
2. **Measurement.** Evaluating the shipped `rank8-l0p8.hsv2` with the
   repository's own integer evaluator on 256 authenticated validation records
   and comparing against the float forward of the matching checkpoint gives a
   median ratio of 0.8493, against the predicted 0.84667. The residual is
   quantization and the two truncation stages.
3. **The trainer's own weight clipping.** Every V2 receipt records
   `dense_weight_clipping.hidden = 1.984375` and
   `dense_weight_clipping.output = 1.6801041666666667`. The first is exactly
   `127 / 64`, the clip implied by `DENSE_SCALE`. The second is exactly
   `127^2 / 9600`, the clip implied by the legacy output scale. The trainer was
   written for the 9,600 output scale; only the V2 exporter uses 64.

The third point is what makes this a defect rather than a design choice. The
trainer and the exporter disagree, and the trainer is the side that matches the
incumbent and the engine comment.

### What it contaminates

- **Contaminated.** Any match between a `.hsv2` engine and a `.nnue` engine. The
  V2 arm plays with evaluations compressed by 15.3 percent and the legacy arm
  does not. This is exactly the cross architecture comparison that produced the
  reported statistical tie, so that tie is not a clean architecture result.
- **Not contaminated.** Every validation metric in `metrics.jsonl`, because the
  trainer evaluates the float model before export. The lambda screen loss
  ordering stands.
- **Not contaminated.** Matches with `.hsv2` on both sides, including the Rank-8
  lambda pairing and the C1 selection against the absolute control, because both
  arms are compressed identically.
- **Partly affected.** The startpos anchor readings. `rank8-l0p8` at +75 is a
  compressed number whose uncompressed value is near +89; `legacy-l0p8` at +112
  is uncompressed. Both remain wrong in sign by thousands of centipawns, so the
  anchor conclusion is unchanged.

A uniform monotone rescaling of the evaluation is not neutral inside a search.
It shifts every margin that is expressed in evaluation units, including futility
and razoring margins, aspiration windows and the mate and tablebase thresholds.

That is no longer an argument from first principles. Re-exporting `rank8-l0p8`
at the corrected scale, with **identical weights and only the output
quantization changed**, moved the tree at a fixed depth 20 from **603,716 nodes
to 1,316,193 nodes**, a factor of 2.18. The same network, searching the same
position to the same depth, now explores a tree more than twice the size. The
scale constant is interacting strongly with pruning and reduction, exactly where
the margins are denominated in evaluation units.

Two consequences follow.

- A fixed-depth node count is not a scale-invariant measurement. Any comparison
  that holds depth constant across a scale change is comparing two different
  amounts of work and must be discarded.
- The search-side effect is large enough that it cannot be reasoned about. It
  has to be played. `scaleab` pairs the same network at 600 against 508 and will
  put the number in Elo; its receipt slot is below and stays empty until it
  lands.

| Receipt | Question | Status |
| --- | --- | --- |
| `scaleab` | same network, 600 against 508, what is the scale worth in Elo | in flight, not yet filed |

Until that lands, the honest statement is direction and mechanism only: the
defect changes the search materially, it is the same failure family as the
inverted label scaling that cost 32 Elo in the Spell generation 2 gate, and its
own magnitude is unmeasured.

### The decision this blocks

V3 cannot both inherit the frozen scales and be self consistent. Three options,
none of which should be taken without a ruling:

1. **Fix the output scale for V3 only.** Register schema `0x00020001` with
   `output_weight_scale = 9600/127` and `output_bias_scale = 9600`, matching the
   trainer, the legacy exporter and the engine comment. V3 then evaluates at
   `600 * v`. V2 containers keep their current behaviour and their receipts stay
   valid as historical artifacts.
2. **Fix V2 as well and re-export.** Correct `_quantized_sections`, re-export
   every `.hsv2`, and rerun any cross architecture match. This invalidates the
   existing V2 container receipts and the C1 selection evidence, and it costs
   match time.
3. **Keep 508 deliberately** and change `NNUE_TO_SCORE` to 508 so the trainer
   models what the engine executes. This is self consistent but changes the
   meaning of every stored loss and of the frozen WDL calibration, whose SHA-256
   is bound into every comparable recipe.

### The ruling

Option 1 is adopted. Schema `0x00020001` is registered in
`tools/horde_v3_container.py` with `OUTPUT_WEIGHT_SCALE = 9600/127` and
`OUTPUT_BIAS_SCALE = 9600`, so a V3 container evaluates at `600 * v`, aligned
with `NNUE_TO_SCORE`, with the legacy exporter and with the WDL link. The
descriptor records `value_semantics` explicitly so the property is readable
rather than implied, and the reader re-checks the two scales out of the header
extension block.

Option 2 is prepared and not executed. The exporter patch, the new schema ids
the corrected structural hash forces, the re-export procedure and the list of
evidence that must be rerun are written up in
[v2-output-scale-repair.md](v2-output-scale-repair.md). Nothing has been applied
and no existing artifact has been rewritten. The Rank-8 lambda comparison in
flight uses two exports of the old scale on both arms, so it is clean as a
lambda comparison and is not touched.

Option 3 is discarded. The WDL link was fitted against teacher scores on the 600
scale, so moving the trainer to 508 would misalign the label and the prediction
rather than align the prediction and the engine.

V3 therefore never carries the defect, and the V3 parity gates measure a network
whose value is `600 * v` by construction.

## 3. What was verified

| Item | Status |
| --- | --- |
| V3 active row bound of 100, analytic and measured | closes |
| feature accumulator, hidden0, hidden1, output `int32` bounds at 1024 lanes | closes |
| PSQT skip bound, new constant `2^20` required | closes, needs registering |
| output scale consistency with `NNUE_TO_SCORE` | does not close, blocks the container |
