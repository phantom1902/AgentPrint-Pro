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

T01 executes only `mode: smoke`, disabled IAT, `variant: single`, raw or log1p.
Known future combinations parse but fail explicitly before preparing caches or
creating a training run. Formal stopping, enabled IAT and multikernel execution
are implemented by the subsequent tickets. A smoke result explicitly records
`converged: false` and `reason: smoke_budget_exhausted`.

## Public extension seams

`agentprint.v2.build_model(model_definition, in_channels=2)` accepts 2 or 10
input channels and returns a five-logit PyTorch model for tensors shaped
`[batch, channels, 2, 1800]`. Definitions use the exact `model` mapping above.
Unknown fields and unsupported variants fail. At original RF width/dropouts,
the 2-channel single variant preserves the original initialization and output.
The optional `in_channels` argument of `tuning.build_model` defaults to 2,
preserving every existing tuning call.

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
publication and ahead-log recovery helpers are reused.

Evaluation validates the saved definition/identity, reconstructs from the saved
model definition and verifies the validation cache metadata. Incompatible CLI
feature/model/transform settings are rejected. The saved training protocol
governs evaluation; runtime roots may relocate artifacts. Selection artifacts
are temporarily keyed by the entire experiment identity, including feature/model,
fit/cache, transform and training settings. Thus future variants cannot enter an
old raw/log1p-only cohort accidentally. This tracer does not yet aggregate raw
and log1p candidates: T06 must exclude transform from the shared cohort identity
and compare the two transforms only within the same feature/model/fit/cache/
training cohort. Explicit synthetic test evaluation remains
available through `--split test`; training loads train and val only.

Resume requires an existing run with a committed format-2 `last.pt` and an exact
experiment identity. V1 checkpoint loading and recovery continue through the
unchanged v1 command path.
