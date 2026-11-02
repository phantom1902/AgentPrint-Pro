# Replace Multikernel log1p Optimization Design

## Goal

Find well-performing training hyperparameters for the AgentPrint IAT-enabled replace-multikernel Scenario classifier with log1p inputs, using only the official train and validation splits and the optim-agent ask/tell study.

## Objective and experiment boundary

- Maximize best validation Macro-F1. Select each trial checkpoint by validation Macro-F1, breaking ties toward the earlier epoch.
- Never load or evaluate the real test split during search, finalist selection, or reporting.
- Reuse the completed full-scope IAT fit and train/validation supplemental cache. Require their existing identities and the base manifest IDs to match the completed six-configuration experiment.
- Train each trial from scratch with five-class Scenario labels, width 2, batch size 200, seed 42 during search, Adam, unweighted cross-entropy, FP32, AMP/TF32 disabled, workers 0, numerical threads 4, and deterministic execution.
- Use the existing LR formula `max(1e-6, learning_rate * lr_decay ** (epoch / 30))` and approved joint stopping policy: at least 60 and at most 200 epochs; stop only after 20 epochs without a validation Macro-F1 gain of at least 0.001 and less than 1% relative improvement between adjacent 10-epoch mean validation-loss windows.

## Search parameters

Only these five values vary. Their legal ranges follow the existing tuning implementation:

| Parameter | Range | Baseline |
|---|---:|---:|
| `learning_rate` | 0.00001 to 0.005 | 0.001 |
| `weight_decay` | 0 to 0.01 | 0.001 |
| `lr_decay` | 0.1 to 1.0 | 0.5 |
| `dropout_2d` | 0 to 0.3 | 0.05 |
| `dropout_1d` | 0 to 0.5 | 0.1 |

`width=2`, `batch_size=200`, and search `seed=42` are fixed. The model variant is fixed to `replace`, and transform to `log1p`. No architecture, cache, feature, loss, data split, batch, or seed changes are candidates.

## Budget and selection

- At most 17 evaluations and 3600 seconds of wall time, whichever comes first.
- Intended allocation: one baseline, up to 12 new seed-42 configurations, and up to four finalist repeats (the two leading new configurations at seeds 43 and 44).
- Trials are sequential. Use completed outcomes to choose each next point; do not launch a fixed grid or concurrent proposals.
- A study-deadline interruption is recorded as pruned with its latest valid validation score when one exists. Crashes are failed and have no objective value.
- Compare the two replicated finalists by their three-seed mean best validation Macro-F1. If time prevents both replications, report the best seed-42 observation as exploratory and make no replicated-winner claim.

## Implementation boundary

Add an explicit v2 `tuning` mode that uses full-scope caches and the same convergence policy while bypassing only the formal mode's fixed-anchor parameter guard. Existing `formal` and `smoke` semantics remain unchanged. Add a separate IAT/v2 optim-agent entry point that creates isolated trial configs and run roots, invokes the public train/evaluate commands, and records measured outcomes with sequential ask/tell. Study files and model outputs remain under `/root/autodl-tmp/ant-workspace` on the data disk; repository study metadata is kept under ignored `.optim-agent-runs`.

Before each expensive command, persist its parameter set, seed, rationale, command, and output location. Resume only from the same trial identity and the existing complete-epoch recovery boundary. Reject changed manifests, fit fingerprints, or cache identities. Never overwrite the previous six-configuration results.

## Verification

- Unit tests verify tuning-mode config acceptance, full-cache scope, formal-anchor isolation, convergence behavior, parameter bounds, and trial identity.
- Public command integration tests use synthetic full-scope IAT artifacts to train/evaluate a short tuning-mode run and verify checkpoint reconstruction and train/validation-only loading.
- Existing formal-mode tests continue to reject altered anchor parameters; smoke tests retain their fixed-budget, non-converged behavior.
- Before search, validate both supplied manifests, fit/cache fingerprint, CUDA/FP32 settings, available disk space, and the default baseline config.
- Audit ask/tell records against proposal and result files, confirm trial count and elapsed time stayed within budget, independently reload/re-evaluate completed finalist checkpoints on validation, and confirm no test access.
