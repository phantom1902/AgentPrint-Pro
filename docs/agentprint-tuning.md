# Scenario/raw hyperparameter search

The tuning entry point is independent of the fixed 30-epoch protocol and the preserved convergence experiments. It reads only `scenario/train` and `scenario/val` from the existing MTAM cache. It never creates or reads a test loader. Results are exploratory validation results, not an estimate on a fresh held-out test set.

The session agent selects explicit parameter points using all completed history. `optim-agent==0.1.1` records each point with `study.ask(params)` and records the measured result with `study.tell`. Selection is sequential; a Linux file lock prevents concurrent study processes. The package is installed in a data-disk virtual environment using system site packages; existing PyTorch packages are unchanged.

## Parameters

| Parameter | Range | Original point | Meaning |
|---|---|---|---|
| learning_rate | 1e-5 to .005 | .0005 | Adam initial rate |
| decay_per_30 | .1 to 1 | .2 | Multiply learning rate by this factor every 30 epochs |
| weight_decay | 0 to .01 | .001 | Adam coupled L2 regularization |
| batch_size | 32, 64, 128, 200, 256 | 200 | Samples per update; final incomplete batch is retained |
| dropout_2d | 0 to .3 | .1 | Dropout after both 2D pooling blocks |
| dropout_1d | 0 to .5 | .3 | Dropout after both 1D pooling blocks |
| width | .5, 1, 2 | 1 | Scale all hidden convolution channels; inputs and five outputs do not scale |

The original point has 891,855 trainable parameters. Half width has 226,287 and double width has 3,540,879. Width=1 uses the original model constructor and preserves its initialization exactly, including RF's final convolution/BatchNorm/ReLU. Topology, input statistics, ordinary cross entropy, deterministic FP32 execution and DataLoader workers=0 are fixed.

Learning rate is `max(1e-6, learning_rate * decay_per_30 ** (epoch / 30))`, with zero-based epoch. Each trial starts from scratch. Minimum 60 and maximum 200 epochs; after the first 30-epoch anchor period, stop only when Macro-F1 has not increased by at least .001 for 20 epochs and adjacent 10-epoch mean validation loss windows improve by less than 1%. Small F1 gains still update best checkpoints without resetting the significant-gain patience counter.

The approved limits are 17 total evaluations and 3600 seconds of study wall time, whichever comes first. The intended allocation is one baseline, up to 12 new seed-42 configurations, and four repeats for the two leading configurations with seeds 43 and 44. Leave time for repeats rather than exhaust the search allocation. Final comparison uses three-seed mean best validation Macro-F1. A deadline-ended trial reports its latest valid intermediate and is `pruned`; it is not ranked as a completed trial. A crash is `failed` and is not assigned an invented objective.

## Execution

Run in the Git checkout on the data disk. Use Git's `info/exclude` to ignore `/.optim-agent-runs/`. Save the explicit seven-parameter JSON before invoking:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/root/autodl-tmp/ant-workspace/envs/optim-agent/bin/python -m agentprint.tuning \
  --study-dir .optim-agent-runs/scenario-raw-20261030 \
  --trial-name trial-00 \
  --params .optim-agent-runs/scenario-raw-20261030/params/00.json \
  --rationale baseline-parity --phase baseline --seed 42 \
  --cache-root /root/autodl-tmp/ant-workspace/data/processed/agentprint-full-v2
```

Subsequent calls use a new trial name and explicit params JSON, `--phase search --seed 42`, or `--phase repeat --seed 43` / `44`. The first baseline creates the study deadline; it is never reset on resume. Exact invocation and proposal rationale are saved in each trial's `proposal.json`. Re-running a completed point is a no-op. Interrupted trials resume complete epochs with Adam and RNG state, while ahead log tails are archived. Cache manifests and trial identities must match.

## Artifacts and verification

- Study: `protocol.json`, `study.json` with real ask/tell outcomes, explicit `params/*.json`.
- Trial: proposal, identity/environment, metrics, atomic complete-epoch `last.pt`, committed immutable best snapshots, published `best.pt`, result and best validation per-class/confusion metrics.
- Unit/integration checks: original-model initialization/output parity; all width variants and backward; bounds; joint stopping; real CPU evaluator; idempotence; interrupted model/Adam/RNG recovery; deadline pruning without fabricated scores.
- Full-data baseline audit: all 90 epochs match the preserved original convergence experiment on nine deterministic fields.
- Final artifact audit: recompute first-stop prefixes, selected checkpoint score/epoch, learning rates, optimizer step counts, ask/tell correspondence, three-seed selection; reload finalist checkpoints and independently evaluate validation metrics.

The final report must distinguish the best seed-42 observation from the selected three-seed mean. Search on 202 validation episodes can overfit validation; no global optimum or fresh-test performance is claimed.

## Full-cache IAT replace/log1p tuning (v2)

`python -m agentprint.iat_tuning` is the separate v2 runner. It does not change the historical scenario/raw runner above. It requires `optim-agent==0.1.1`, full-scope IAT fit/train/validation caches, and a v2 source config. Each candidate fixes replace/log1p, width 2, batch 200, Adam, unweighted cross-entropy, workers 0, numerical threads 4, deterministic FP32 with AMP/TF32 disabled, and a 200-epoch cap. The shared formal joint convergence policy requires at least 60 epochs. Only `learning_rate`, `weight_decay`, `lr_decay`, `dropout_2d`, and `dropout_1d` are optimizer parameters.

The inclusive bounds are respectively `[0.00001, 0.005]`, `[0, 0.01]`, `[0.1, 1]`, `[0, 0.3]`, and `[0, 0.5]`. The required first baseline JSON is:

```json
{"learning_rate": 0.001, "weight_decay": 0.001, "lr_decay": 0.5, "dropout_2d": 0.05, "dropout_1d": 0.1}
```

Example on the approved data disk:

```bash
/root/autodl-tmp/ant-workspace/envs/optim-agent/bin/python -m agentprint.iat_tuning \
  --study-dir /root/autodl-tmp/ant-workspace/.optim-agent-runs/replace-log1p-20261102 \
  --trial-name trial-00 --params /root/autodl-tmp/ant-workspace/.optim-agent-runs/replace-log1p-20261102/params/00.json \
  --phase baseline --seed 42 --rationale baseline-parity \
  --config configs/agentprint/replace-log1p-tuning-20261102.yaml --device cuda:0
```

The first baseline fixes a single 3600-second deadline. Proposal files reserve evaluation slots, including failed and pruned evaluations. At most 17 evaluations are admitted: one baseline, 12 search points at seed 42, and four repeats at seeds 43/44. A repeat must match a completed new seed-42 configuration. The session agent should select each next point from completed history and reserve repeats for the leading two new configurations. The runner rejects duplicate point/seed pairs and requires pending trials to be resumed before new proposals. A process-wide study lock prevents concurrent launches. New trials after the deadline are rejected.

Each trial has isolated `config.yaml`, `proposal.json`, `ask.json`, command records, logs, `runs/`, and `result.json`. The proposal records the exact resolved configuration, full train/validation manifest and cache metadata, IAT fit fingerprint, feature/model versions, parameters, seed, rationale, command, and output path before training. The baseline protocol freezes those workload identities; changes fail closed. The source fit/cache roots are preserved, and outputs may not overlap existing source data, caches, fit artifacts, or run roots. The runner only constructs train/validation loaders and invokes public `python -m agentprint train`, then `evaluate --split val` on the immutable validation-best snapshot. Completed objectives must match that independently reloaded validation score.

Both subprocesses share the remaining study deadline. A timeout kills the public subprocess and records `pruned`. Its intermediate score comes exclusively from the latest complete epoch in atomic `last.pt`, never from an ahead metrics-log tail. With no committed epoch the pruning score is null. The last checkpoint and immutable best snapshots remain available. Pruned intermediate values are reported to optim-agent but are excluded from completed rankings. Exceptions are `failed` with no objective. A timeout during final evaluation remains pruned even if training had finished.

Reinvoke the same trial name/config/params to resume a committed epoch or reconcile a saved result after a crash before `tell`; finalized matching invocations are idempotent. An interrupted first epoch without `last.pt` cannot resume and is recorded failed rather than pretending an epoch committed. Optim-agent 0.1.1 persists only terminal trials, so `proposal.json` and `ask.json` preserve the pending ask boundary across process interruption. Never edit the study protocol to reset its deadline. Checkpoints remain ignored under `.optim-agent-runs`; commit only light result summaries.
