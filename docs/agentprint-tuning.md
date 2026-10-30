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
