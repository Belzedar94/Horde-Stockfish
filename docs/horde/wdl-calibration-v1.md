# Horde WDL calibration V1

`HORDE_WDL_CALIBRATION_V1` converts a post-rule-50 Horde search score into
loss, draw, and win probabilities. The artifact is fitted once from the
training split and then frozen unchanged across Fresh H/P and every V2
architecture or seed in the comparison campaign. Validation labels never
participate in the fit.

## Link

White-to-move and Black-to-move records are fitted independently. For side
`s`, score `x`, and `u = x / 600`:

```text
eta = A_s * u + B_s
logits(loss, draw, win) = (-eta, D_s, eta)
probabilities = softmax(logits)
```

`A_s` must be positive. The fitter minimizes unweighted mean categorical
negative log likelihood. It does not use class weights, result resampling,
pseudocounts, or side pooling.

Only ordinary scores with `abs(score) < 31507` enter calibration. Scores at or
beyond that threshold are mate-distance labels: they remain available to the
result objective but never produce a score-derived WDL target. The stored
teacher score already includes search's rule-50 behavior and is therefore
passed directly to the frozen link. A network prediction first passes through
`HORDE_RULE50_LINEAR_V1` and then through this link.

## Fit gates

Each side must contain at least 32 eligible loss, draw, and win records. The
binary64 full-batch Newton solve is accepted only when all of the following are
true:

- the gradient infinity norm reaches the registered tolerance;
- `A_s > 0`;
- every parameter and metric is finite;
- the final Hessian is full-rank and positive definite;
- its registered condition-number limit is respected;
- no separating or unbounded solution is detected.

The artifact stores the three parameters both as 17-digit decimal strings and
as exact big-endian IEEE-754 binary64 bits. It also binds the complete training
file, payload, manifest, teacher producer, Run 6B network, label contract,
record-selection digest, class support, optimizer gates, NLL, Brier metrics,
gradient, and Hessian diagnostics. Consumers reject any dirty software receipt,
unknown schema, inconsistent count, altered float encoding, or weakened gate.

## Command

Run from a clean checkout after generating the authenticated training split:

```text
python tools/horde_fit_wdl.py TRAIN.bin --output horde_wdl_calibration_v1.json
```

The output path is exclusive and the canonical JSON is deterministic for the
same dataset, source commit, Python runtime, and fitter contract. The resulting
file SHA-256 is the calibration identity consumed by trainers and checkpoints.

Calibration quality is not strength evidence and cannot promote a network.
