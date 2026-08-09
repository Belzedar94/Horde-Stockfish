# Horde NNUE V2 functional health

## Purpose

Validation loss and nonzero serialized parameters do not prove that a trained
network still uses its position features. A hard-clamped dense path can become
constant while every transformer has nonzero gradients and healthy-looking
weights. Architecture comparison and playing tests are invalid in that state.

`HORDE_V2_FUNCTIONAL_HEALTH_V1` is an additional fail-closed gate. It does not
replace parameter health, integer parity, NPS measurement, or playing tests.
It changes no network parameter and adds no inference cost.

## C1 seed-1 diagnosis

The first two authenticated C1 checkpoints exposed the same functional
collapse:

| Architecture | Checkpoint SHA-256 | Hidden0 constant lanes | Hidden1 constant lanes | Pre-rule50 integer scores per side | Feature Jacobian |
| --- | --- | ---: | ---: | ---: | ---: |
| `v2-c1-abs64x192` | `7B7A6BFB826161EE711BBDBBE4B8EB2A9D5B464164929487E0850ABE14D9FC16` | 32/32 | 32/32 | 1 | 0 |
| `v2-c1-rank8-64x192` | `19F435D8F3F17507E4DFE8584B21E96E2C035016631244BBA5DADEAB46BD25A9` | 32/32 | 32/32 | 1 | 0 |
| `v2-64x192` | `0EE35534175CB968EAF1A365401A8388CCDE4D3458340AC28194C4ADE088635B` | 32/32 | 32/32 | 1 | 0 |

Both first-domain accumulators remained position-dependent and unsaturated.
The rank-8, absolute, and full Royal activations differed materially, but
hidden0 mapped every validation probe position to the same 0/1 vector. Hidden1
was likewise constant. The surviving evaluator was therefore one constant per
side to move, with rule-50 damping applied afterward. All three architectures
finished at the exact same validation composite loss,
`0.16662875441138447`, despite distinct parameter and state hashes.

The exact equality of aggregate validation metrics is a consequence of the
integer-forward training objective: float outputs differing by less than one
centipawn can occupy the same truncated score bin while the straight-through
gradient still updates distinct states. The checkpoints are not duplicates,
and this evidence does not identify an encoder-dispatch failure.

C1 remains useful only as a three-seed characterization of recipe stability.
No C1 architecture may be nominated from validation loss unless every paired
checkpoint first passes functional health. The legacy Run 6B production path
is unchanged.

## Frozen probe

The tool selects 4,096 deterministic midpoints across the complete validation
record order and verifies that the selected dataset identity exactly matches
the checkpoint receipt. It reports:

- unclipped and clipped first-domain and Global accumulators;
- hidden0 and hidden1 preactivation ranges, clamp occupancy, per-lane variance,
  and constant-lane counts;
- pre-rule50 and post-rule50 output diversity for each side to move;
- the exact float score Jacobian with respect to both feature domains;
- zero and same-side/same-rule50 permutation interventions for each domain.

The gate rejects excessive constant lanes, missing interior clamp support,
missing within-side integer diversity, insufficient side support, or a dead
feature-to-score Jacobian. Thresholds and numeric tolerances are frozen in
`schemas/horde-v2-functional-health-v1.json`.

Example for an authenticated selected validation role:

```console
python tools/horde_v2_functional_health.py \
  path/to/checkpoint.pt \
  path/to/selected-role/receipt.json \
  --validation-selected-role \
  --output path/to/functional-health.json \
  --require-pass
```

The receipt is always written before `--require-pass` returns a failing exit
status, preserving the diagnostic evidence.

## C2 recipe qualification

C2 must qualify the training recipe on the cheapest absolute control before
comparing representations. Each arm uses all three frozen seeds and changes
one factor only.

The first arm reduces only the learning rate of `hidden0` and `hidden1`,
including their biases, to 0.1 times the base rate. Transformer and output
learning rates, data, sample order, objective, batch size, epochs, optimizer,
initialization, widths, and quantization remain fixed. This targets the shared
dense collapse without adding parameters or engine work.

A separate arm may change only the output learning-rate multiplier from 0.1 to
1.0. The two changes must not be combined in one experiment. Batch size,
clamp-gradient semantics, objective weighting, feature widths, and architecture
remain later orthogonal questions.

The reference trainer exposes the two arms explicitly:

```text
--dense-learning-rate-multiplier 0.1
--output-learning-rate-multiplier 1.0
```

The first command leaves the output multiplier at its frozen 0.1 default. The
second leaves the dense multiplier at its frozen 1.0 default. Non-default values
for both controls in the same run fail closed, and the frozen C1 campaign rejects
either override. With both defaults, optimizer grouping and the first RAdam step
remain bit-identical to the pre-C2 trainer.

An arm is recipe-qualified only if all three seeds pass functional health and
beat the exact side-to-move plus rule-50 constant baseline on held-out data.
Only then may absolute, rank-8 Royal, and full Royal be compared under the same
qualified recipe and dataset.

### Frozen constant baseline

`HORDE_V2_C2_CONSTANT_BASELINE_V1` fits exactly two integer values: one
pre-rule50 score for White to move and one for Black to move. It exhaustively
enumerates every integer in `[-31506, 31506]`; ties are exact binary64 ties and
are resolved by `(abs(score), score)`. The fit accepts only the authenticated
training split and its training-fitted WDL calibration. Its command line has no
validation or checkpoint input:

```console
python tools/horde_v2_c2_constant_baseline.py \
  path/to/train.bin \
  path/to/wdl-calibration.json \
  --output path/to/c2-constant-baseline.json
```

The fitter first casts the side-specific calibration to float32 and builds the
complete CPU `torch.softmax` lookup used by the trainer for both sides and all
63,013 integer scores. The receipt hashes the raw little-endian float32 table.
Objective evaluation then casts those probabilities to binary64, applies the
rule-50 transform with integer truncation semantics, and evaluates frozen
sufficient moments in deterministic order. Mate-distance teacher scores are
excluded only from the score-derived term; their result targets remain, and the
composite mean is normalized by every record.

The independent histogram reference is evaluated at each selected constant and
its binary64 difference from the moment evaluator is recorded. A second audit
scales every exact float32 probability by `2^149` and requires the moment and
histogram integer numerators to be identical, with no tolerance. Minima,
complete tie-set identity, runner-up gap, boundary contact, dataset identity,
calibration identity, and software identity are all receipted. The frozen
contract lives at `schemas/horde-v2-c2-constant-baseline-v1.json`.

This artifact is a null model, not validation evidence. A later qualification
command must re-evaluate both the frozen null and each complete three-seed arm
on the same selected role. It may not refit the constants, replace a seed, or
select a best epoch after seeing validation.

### Qualification statistics

`HORDE_BIN_V1` stores `game_ply` but no authenticated game or opening identity.
The generator's in-memory opening index is not serialized into a training
record. Record adjacency and `game_ply` therefore cannot be used to reconstruct
clusters after the fact. C2 reports exact paired per-record objective deltas and
makes no game-clustered confidence claim; an IID bootstrap would be equally
misleading and is forbidden.

The selected validation role has already participated in engineering and
functional-health inspection, so it is a qualification/tuning role rather than
fresh confirmation evidence. An optimizer arm qualifies only when all three
frozen final checkpoints pass functional health and each has strictly lower
canonical loss than the frozen constant on the same record order. There is no
post-hoc numeric margin, best-seed replacement, best-epoch selection, or
winner selection between two qualifying arms on this role. A later label-blind
role is required for confirmatory ranking.

Qualification is fail-closed on provenance as well as loss. The frozen
constant, all three checkpoints, their training receipts, their
functional-health receipts, the selected validation role, the teacher identity
and the WDL lookup must agree exactly. A clean but unrelated receipt or a run
trained on another byte stream is not admissible evidence.
