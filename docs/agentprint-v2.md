# AgentPrint schema v2 extension contract

The public `python -m agentprint prepare|train|evaluate --config ...` commands
dispatch by `schema_version`. Genuine v1 configurations still use the existing
strict schema, cache fingerprints, models and checkpoint format. The new
`configs/agentprint/common-v2.yaml` is a complete v2 base configuration.

## Configuration

V2 retains the existing `task`, `transform`, `mode`, `seed`, `paths`, and
`preprocessing` fields. This extension classifies complete Episodes into the five
Scenario classes; `task` must be `scenario`. The base MTAM remains 1800 slots,
1024-sample shards and uint64. Its preprocessing fingerprint is unchanged.

Additional required mappings:

| Mapping | Fields |
| --- | --- |
| `feature` | `version: 1`, `iat_enabled: bool`, `fit_artifact: str`, `iat_cache_root: str` |
| `model` | `version: 1`, `variant: single/replace/prepend`, `width`, `dropout_2d`, `dropout_1d` |
| `training` | Existing fields plus `min_learning_rate` and `numerical_threads` |

Disabled IAT requires both feature references to be empty strings. Enabled IAT
requires nonempty references; its fit identity and supplemental cache identity
will be resolved by the fitting/cache tickets. `paths.cache_root` always names
the original MTAM cache. Runtime CLI overrides remain restricted to the existing
three root paths and device/resume/evaluation options.

The approved new anchor is width 2, dropouts 0.05/0.1, Adam learning rate 0.001,
weight decay 0.001, batch 200, `lr_decay: 0.5`, `min_learning_rate: 0.000001`,
workers 0 and numerical threads 4. Learning rate at zero-based epoch e is
`max(min_learning_rate, learning_rate * lr_decay ** (e / 30))`.

The trainer supports smoke and formal stopping. Formal **train execution** requires
seed 42, the exact approved anchor above, and `training.epochs: 200`. Parsing,
fitting and preparation do not authorize or execute a formal training run. Enabled
IAT and multikernel execution are implemented by their separate tickets. A smoke
result explicitly records `converged: false` and `reason: smoke_budget_exhausted`.

Formal stopping reuses `ConvergencePolicy`: the first 30 epochs establish the best
validation Macro-F1 anchor, followed by at least 30 additional epochs. Convergence
requires 20 epochs without a cumulative significant gain of at least 0.001 and
less than 1% relative loss improvement between the latest two 10-epoch windows.
A zero earlier-window mean gives zero relative improvement. The earliest stop is
60 total epochs; the cap is 200. At the cap, a run without both plateau conditions
reports `epoch_cap_reached` with `converged: false`. The significant-gain threshold
affects only stopping: every actual validation F1 improvement updates best.pt;
equal scores retain the earlier epoch. This interface does not grant a budget to
run formal training during smoke acceptance.

`agentprint.v2_protocol.stop_status(rows, mode, epochs)` exposes this assessment
for complete zero-based synthetic histories, following the existing stopping-rule
test precedent. `learning_rate(epoch, settings)` applies the v2 schedule, without
calling the legacy convergence learning-rate function. `protocol.json` records the
policy and schedule; `convergence.json`, `last.pt` and `result.json` record the
committed stopping outcome. Metric rows also record per-epoch elapsed seconds.

T03 additionally executes `variant: replace` at width 2, with disabled IAT and
either transform. It replaces the whole original first Conv2d/BatchNorm2d/ReLU
unit, preserves the 1800 input time slots through the replacement, then retains
the subsequent RF frontend, pooling, reshape and one-dimensional classifier.
It has 3,544,771 parameters with 2 input channels and 3,552,403 with 10 input
channels, at width 2 and the approved dropouts. These are actual parameter
counts, not a claim of equal capacity to the single model. Ten-channel model
checks use synthetic tensors; enabled-IAT loading remains a separate extension.

T04 additionally executes `variant: prepend` at width 2, with disabled IAT and
either transform. The shared multikernel unit projects its 64 fused channels
back to the original 2 or 10 input channels, then feeds the complete original
RF frontend and classifier: `CNN(F(x))`, with no residual. Prepend and replacement
are independent definitions and checkpoints. Prepend has 3,543,113 parameters
with 2 input channels and 3,560,497 with 10 input channels at the approved
width and dropouts. Ten-channel checks remain synthetic model-interface checks
until enabled-IAT loading is implemented.

## Public extension seams

`agentprint.v2.build_model(model_definition, in_channels=2)` accepts 2 or 10
input channels and returns a five-logit PyTorch model for tensors shaped
`[batch, channels, 2, 1800]`. Definitions use the exact `model` mapping above.
Unknown fields and unsupported variants fail. At original RF width/dropouts,
the 2-channel single variant preserves the original initialization and output.
The optional `in_channels` argument of `tuning.build_model` defaults to 2,
preserving every existing tuning call.

`agentprint.multikernel.MultiKernelUnit(in_channels, out_channels)` is the shared
unit for both multikernel integrations. Inputs have 2 or 10 channels; outputs
may have 64 (replacement) or 2/10 (prepend) channels. It preserves direction
height and time length: `[N, in_channels, 2, T]` becomes
`[N, out_channels, 2, T]`. Three Conv2d/BatchNorm2d/ReLU branches use kernels
`(3, 3)`, `(3, 5)`, `(3, 7)`, padding `(1, 1)`, `(1, 2)`, `(1, 3)`, stride 1,
and channels 22/21/21. Concatenation feeds a 64-to-output 1x1
Conv2d/BatchNorm2d/ReLU projection. Convolutions include biases, as in RF.
The constructor applies RF's Conv2d and BatchNorm2d initialization; the model's
Conv1d layers keep PyTorch's default initialization. All weights are created
from scratch. The unit has no residual and performs no cropping.

`v2.loader(config, split, shuffle=False, generator=None)` returns
`(samples, batches)`. Samples expose `records`, `metadata`, length and indexed
float32 transformed inputs, matching the existing loader. The loader ticket can
join supplemental IAT samples here without changing the CLI/trainer contract.

`v2.feature_definition(config)` declares extension/base versions, channel order,
shape and dtype. Transform is part of experiment identity, and applies after
uint64-to-float32 conversion. `v2.experiment_identity(config, train_samples,
val_samples)` combines definitions, label map, seed, training settings, both
base cache metadata identities and explicit `fit_fingerprint`/`iat_cache` slots.
Those two slots are null when disabled. File-system roots do not substitute for
artifact identities. Later tickets should fill these slots from verified
artifacts rather than fingerprinting reference path strings.

## Checkpoints, evaluation and recovery

Format-2 checkpoints include `config`, `identity`, `experiment_id` (SHA256 of
the identity), `model_definition`, `feature_definition`, model state and actual
parameter count. Complete-epoch `last.pt` additionally stores Adam state,
global step, all metric rows, best checkpoint pointer and Python/NumPy/Torch/
CUDA/shuffle-generator state. Best snapshots remain immutable and the existing
publication and ahead-log recovery helpers are reused. Recovery checks contiguous
history, saved stopping status, and exact agreement between checkpoint rows and
the committed metrics log after ahead/partial rows have been archived. A run
already stopped by convergence or budget does not execute another epoch.

Evaluation validates the saved definition/identity, reconstructs from the saved
model definition and verifies the validation cache metadata. Incompatible CLI
feature/model/transform settings are rejected. The saved training protocol
governs evaluation; runtime roots may relocate artifacts. Selection artifacts
pair raw and log1p candidates by the entire saved experiment identity **excluding
only transform**. Feature/model definitions, fit/cache identities, manifest,
training settings, mode and seed must all agree. Each transform's latest explicitly
evaluated checkpoint supplies its saved best validation Macro-F1; equal transform
scores select raw. An artifact is complete only after both candidates exist.
Candidate fingerprints are checked before aggregation. Explicit synthetic test evaluation remains
available through `--split test`; training loads train and val only.

Resume requires an existing run with a committed format-2 `last.pt` and an exact
experiment identity. V1 checkpoint loading and recovery continue through the
unchanged v1 command path.
