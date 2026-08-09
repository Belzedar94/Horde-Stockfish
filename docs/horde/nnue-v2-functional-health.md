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

An arm is recipe-qualified only if all three seeds pass functional health and
beat the exact side-to-move plus rule-50 constant baseline on held-out data.
Only then may absolute, rank-8 Royal, and full Royal be compared under the same
qualified recipe and dataset.
