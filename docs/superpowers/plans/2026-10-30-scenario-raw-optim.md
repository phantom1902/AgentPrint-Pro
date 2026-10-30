# Scenario raw optim-agent experiment plan

> Execute inline in the current session, using writing-plans, TDD and verification-before-completion. The user approved the experiment design and budget in this conversation.

**Goal:** Find the best measured scenario/raw configuration within 17 trials and 3600 seconds of study wall time, whichever is reached first.

**Architecture:** An independent tuning module reuses the immutable MTAM cache, RF topology, metric implementation and checkpoint helpers. The session agent proposes explicit configurations sequentially through optim-agent ask/tell; no automatic sampler chooses points. Production training and the previous experiments remain unchanged.

**Tech stack:** Python 3.12, PyTorch FP32, optim-agent 0.1.1 in a data-disk venv with system site packages.

## Approved constraints

- Five-class scenario, raw inputs, official train/val only; no test loading or evaluation.
- 2697 training and 202 validation episodes; compare exact manifest identities across trials.
- One baseline + up to 12 new configurations at seed 42; two leading configurations receive seed 43 and 44 repeats (17 total). Final selection uses their three-seed mean best validation Macro-F1. Accuracy is descriptive.
- Each trial starts from scratch, max 200 epochs, minimum 60; after that require both stale Macro-F1 for 20 epochs at delta 0.001 and relative mean validation loss improvement below 1% over adjacent 10-epoch windows. The anchor begins at the best of the first 30 epochs, matching the prior continuation protocol.
- Search learning_rate [1e-5, 0.005], weight_decay [0, 0.01], decay_per_30 [0.1, 1], batch_size in {32,64,128,200,256}, dropout_2d [0,0.3], dropout_1d [0,0.5], width in {0.5,1,2}.
- Learning rate is max(1e-6, initial_lr * decay_per_30 ** (epoch/30)); minimum does not change.
- Default point: lr=.0005, weight_decay=.001, decay_per_30=.2, batch=200, dropout_2d=.1, dropout_1d=.3, width=1.
- Fixed topology and unweighted cross entropy, FP32, deterministic execution, workers=0. Only hidden channels scale; input/output class counts remain fixed.
- `.optim-agent-runs/` is ignored using Git info/exclude; all remote artifacts and dependency environment reside on /root/autodl-tmp.
- Deadline checks occur within batches; a deadline-ended trial is pruned with the latest valid intermediate result and excluded from the complete-trial ranking. Crashes are failed, not given fabricated scores.
- Each trial saves params/command before evaluation, identity and environment, per-epoch metrics, best model and complete last state. Completed results are reconciled with ask/tell after interruptions; partially committed epochs resume from last state and RNG.

## Tasks

### 1. Validate model and stopping behavior

- [x] Write `tests/test_agentprint_tuning.py` before `agentprint/tuning.py`; require default initialization/output parity, 0.5/2 width forward/backward, invalid bounds rejection, and plateau/cap behavior.
- [x] Run `/root/autodl-tmp/ant-workspace/envs/optim-agent/bin/python -m unittest discover -s tests -p test_agentprint_tuning.py -v`; first fail on absent tuning module, then pass after implementation.

### 2. Implement and verify one-trial evaluator

- [x] Add `evaluate_trial(params, root, cache_root, seed, device, deadline)` and CLI accepting explicit params/phase. Use existing cache and metric code; record only train and val identities.
- [x] Add a real CPU synthetic-cache integration test for result/checkpoint consistency, idempotent recovery, interrupted epoch recovery and deadline pruning.
- [x] Run focused new tests and existing model/convergence tests; compile and git diff --check; commit verified source and sync the commit to the remote branch.

### 3. Run informed sequential search

- [x] Establish same-evaluator baseline, compare its first 30 metrics and final best with the preserved baseline/continuation as a reproducibility audit.
- [x] Read full history and choose one bounded new point at a time; use ask/tell, save rationale and output directory before GPU evaluation. No speculative concurrent trials.
- [x] Reserve four repeat slots, compare top-two three-seed means, and report all measured outcomes including failures/pruning.
- [x] Audit best artifacts and study JSON, back up locally with hashes, commit lightweight results and exact reproduction command. Do not claim global optimality.

## Outcome

Completed 12 trials in 3532.15 seconds: one baseline, seven new configurations, four repeats. All complete, zero failed/pruned. The reduced search count reserved the fixed wall-time budget for replication. Selected three-seed mean validation Macro-F1 0.785323 ± 0.016825. See `docs/experiments/2026-10-30-scenario-raw-tuning/README.md` for measured results and reproduction.
