# V2 output scale repair, handed off for execution

Status: handoff delivered. The patch and its operating instructions are written
to `D:\horde-train\patches\v2-export-scale-fix.patch` and
`v2-export-scale-fix-README.md` for the campaign agent to apply in its own
staging tree. Nothing is applied here and no existing `.hsv2` artifact has been
rewritten.

The defect and its three independent proofs are recorded in
[nnue-v3-integer-bounds.md](nnue-v3-integer-bounds.md). In one line: the V2
container exporter quantizes the output layer at the dense scale, so every
exported V2 network evaluates at `508 * v` where the trainer optimized
`600 * v`, a uniform compression of 0.84667.

## Scope for the Rank-8 lambda rerun

The owner's decision is that the rerun plays on corrected exports, with **both**
arms re-exported. That keeps the comparison fair and also makes it true. Mixing
a corrected arm against an uncorrected one would compare an architecture against
a scale defect, so it is forbidden.

The repair never edits a container in place. It writes new files beside the old
ones and the originals are kept, because they are the artifacts the historical
matches actually played and deleting them would make those receipts
unreproducible.

## The patch

`tools/horde_v2_export.py` picks one scale per section from a single
expression. The output layer needs the same treatment the legacy exporter
already gives it.

```diff
@@ tools/horde_v2_export.py
+V2_OUTPUT_BIAS_SCALE = 600 * 16
+V2_OUTPUT_WEIGHT_SCALE = (600 * 16) / 127
+
+
+def _section_scale(section_name: str, spec: NetworkSpec) -> float:
+    """Return the quantization scale for one section.
+
+    The output layer is scaled so that the container value equals
+    ``NNUE_TO_SCORE * v``. The legacy exporter has always done this; the
+    original V2 rule reused the dense and feature scales here and produced
+    ``508 * v`` instead of ``600 * v``.
+    """
+
+    if section_name == "output_weights":
+        return V2_OUTPUT_WEIGHT_SCALE
+    if section_name == "output_bias":
+        return V2_OUTPUT_BIAS_SCALE
+    if section_name in {spec.first_weight_name, spec.first_bias_name,
+                        "global_weights", "global_bias"}:
+        return FT_SCALE
+    if section_name.endswith("bias"):
+        return FT_SCALE
+    return DENSE_SCALE
+
+
 def _quantized_sections(
     checkpoint: Mapping[str, object], spec: NetworkSpec
 ) -> tuple[dict[str, bytes], dict[str, dict[str, object]]]:
@@
-        scale = FT_SCALE if section.name in {spec.first_weight_name, spec.first_bias_name, "global_weights", "global_bias"} or section.name.endswith("bias") else DENSE_SCALE
+        scale = _section_scale(section.name, spec)
         payload, stats = _quantize(value.contiguous(), scale, section.dtype, section.name)
```

`tools/horde_v2_container.py` is deliberately **not** touched. Adding the two
output scales to the descriptor would change `structural_sha256`, which would
force new schema ids, a C++ reader change and an engine rebuild. Stamping a
marker into the reserved fixed-header region is not available either: the V2
reader rejects any container whose header bytes 532 through 640 are nonzero.

Keeping the codec untouched means a corrected container is a drop-in for every
existing reader and needs no rebuild, at the cost that it is distinguishable
from an uncorrected one only by its file SHA-256 and by the export receipt,
which now carries `output_scale_repair: "V2_OUTPUT_SCALE_REPAIR_V1"` and the two
output scales. Record the container SHA-256 of both arms in the match log for
any run that uses re-exported networks.

## Re-export procedure

Nothing is done in place. For each affected checkpoint:

1. Confirm no live process holds the existing artifact. The current referee and
   trainers are Python processes; check before starting and abort if a match is
   running against the file being replaced.
2. Re-export beside the original, never over it:

   ```console
   python tools/horde_v2_export.py CHECKPOINT.pt TRAINING-RECEIPT.json \
     --output EXPORTS/rank8-l0p8.s600.hsv2 \
     --export-receipt EXPORTS/rank8-l0p8.s600.export.json
   ```

3. Verify the corrected scale with the same measurement that found the defect:
   evaluate the new container and the float checkpoint over at least 256
   authenticated validation records and require the median ratio of integer to
   `trunc(v * 600)` to be within 0.02 of 1.000. The uncorrected containers
   measure 0.8493 on this test, so it separates the two populations cleanly.
4. Record both file SHA-256 values, old and new, in the re-export receipt, and
   record the measured ratio for each.
5. Do not delete the old containers. They are the artifacts the historical
   matches actually played, and deleting them would make those receipts
   unreproducible.

## What has to be rerun afterwards, and what does not

| Evidence | Status after repair |
| --- | --- |
| every `metrics.jsonl` validation number | unaffected, the trainer scores the float model before export |
| Rank-8 lambda pairing, `.hsv2` on both arms | unaffected, both arms compressed identically |
| the Rank-8 lambda rerun | plays on corrected exports, both arms re-exported |
| C1 selection, Rank-8 against the absolute control | unaffected, both arms `.hsv2` |
| any match of a `.hsv2` engine against a `.nnue` engine | must be rerun before it is cited as architecture evidence |
| the reported legacy against Rank-8 statistical tie | must be rerun; this is the one that matters |
| startpos anchor evaluations for `rank8-l0p8` | should be re-read from a corrected container; the conclusion does not change, since the error there is thousands of centipawns |

## Ordering against the V3 work

V3 does not wait on this. Schema `0x00020001` is registered with the corrected
output scale from the start, so the V3 container never carries the defect and
the V3 parity gates measure a network whose value is `600 * v` by construction.
The V2 repair only affects how earlier V2 artifacts are compared with legacy
ones, which is a question about historical evidence rather than about the V3
build.
