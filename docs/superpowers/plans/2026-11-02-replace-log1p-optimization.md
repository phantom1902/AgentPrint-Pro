# Replace Multikernel log1p Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tune the replace-multikernel log1p Scenario model through sequential optim-agent trials on the existing full IAT train/validation cache.

**Architecture:** Add an explicit v2 tuning mode that shares the full-cache identities, model, public train/evaluate commands, complete-epoch recovery, and joint convergence protocol. Add a separate IAT tuning entry point that validates fixed workload and bounded parameters, creates isolated trial configs, and records sequential ask/tell results.

**Tech Stack:** Python 3.12, PyTorch CUDA FP32, unittest, PyYAML, optim-agent 0.1.1, existing AgentPrint v2 CLI.

## Global Constraints

- Objective is maximum best validation Macro-F1; real test data is never loaded.
- Fixed variant `replace`, transform `log1p`, width 2, batch size 200; search seeds use 42.
- Search ranges: learning rate `[0.00001, 0.005]`; weight decay `[0, 0.01]`; LR decay `[0.1, 1.0]`; 2D dropout `[0, 0.3]`; 1D dropout `[0, 0.5]`.
- Adam, unweighted cross-entropy, FP32, AMP/TF32 disabled, workers 0, numerical threads 4, deterministic execution.
- At least 60 and at most 200 epochs. Stop only when the existing joint Macro-F1/loss plateau policy passes.
- At most 17 trials and 3600 seconds wall time. Intended allocation is one baseline, up to 12 new seed-42 trials, and four finalist repeats with seeds 43/44.
- Trial artifacts live under ignored `.optim-agent-runs`; all remote outputs stay under `/root/autodl-tmp/ant-workspace`.
- Existing formal/smoke semantics and the previous six-config results remain unchanged.

---

### Task 1: Add Isolated V2 Tuning Mode

**Files:**
- Modify: `agentprint/v2.py`
- Modify: `agentprint/v2_protocol.py`
- Modify: `agentprint/training.py`
- Test: `tests/test_agentprint_v2.py`
- Test: `tests/test_agentprint_v2_protocol.py`

**Interfaces:**
- `validate_config` accepts `mode='tuning'` as a third v2 mode and still checks all other schema rules.
- `stop_status(rows, 'tuning', 200)` uses the formal joint convergence policy and 200-epoch cap; smoke remains budget-only and never converged.
- Formal anchor validation applies only to `mode='formal'`; a tuning run records `formal_experiment=false`.
- Tuning-mode training requires all five training classes, as formal training does.

- [ ] **Step 1: Write tests for tuning-mode configuration and convergence**

Add tests asserting a tuning config with non-anchor candidate values loads, a formal config with the same changed values is rejected, 60-row tuning history can meet the joint stop rule, and smoke history still reports `converged=false`.

- [ ] **Step 2: Verify the tests fail for the missing mode**

Run: `PYTHONPATH=tests:. python -m unittest test_agentprint_v2 test_agentprint_v2_protocol -v`

Expected: tuning-mode config is rejected and `stop_status` raises for mode `tuning`.

- [ ] **Step 3: Implement the minimal mode branch**

Extend v2 mode validation to `('smoke', 'formal', 'tuning')`; keep `validate_formal` strict for formal only; allow tuning in `stop_status`; only apply the smoke budget override to smoke; make missing-class validation apply to formal and tuning.

- [ ] **Step 4: Run the targeted tests**

Run: `PYTHONPATH=tests:. python -m unittest test_agentprint_v2 test_agentprint_v2_protocol -v`

Expected: all targeted tests pass and existing formal-anchor/smoke assertions remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add agentprint/v2.py agentprint/v2_protocol.py agentprint/training.py tests/test_agentprint_v2.py tests/test_agentprint_v2_protocol.py
git commit -m "feat(agentprint): isolate v2 hyperparameter tuning mode"
```

### Task 2: Add Sequential IAT Optim-Agent Trial Runner

**Files:**
- Create: `agentprint/iat_tuning.py`
- Test: `tests/test_agentprint_iat_tuning.py`
- Modify: `docs/agentprint-tuning.md`

**Interfaces:**
- `validate_iat_params(params)` requires exactly `learning_rate`, `weight_decay`, `lr_decay`, `dropout_2d`, `dropout_1d`; fixed width, batch, variant, transform, and seed are supplied by the protocol, not optimizer parameters.
- CLI accepts `--study-dir`, `--trial-name`, `--params`, `--phase`, `--seed`, `--config`, and `--device`.
- One invocation records a proposal through `study.ask`, runs public `agentprint train` and `agentprint evaluate --split val`, then records the measured best validation Macro-F1 through `study.tell`.
- Baseline is exactly `{learning_rate: 0.001, weight_decay: 0.001, lr_decay: 0.5, dropout_2d: 0.05, dropout_1d: 0.1}` at seed 42.
- Completed trial identity includes full train/validation manifest IDs, IAT fit fingerprint, both IAT cache fingerprints, model/feature identity, config, parameters, and seed.

- [ ] **Step 1: Write parameter/protocol tests**

Test legal endpoints, reject unknown/missing/non-finite/out-of-range values, and assert candidate config construction preserves the full fit/cache roots while fixing replace/log1p/width2/batch200/200 epochs.

- [ ] **Step 2: Verify the new tests fail before implementation**

Run: `PYTHONPATH=tests:. python -m unittest test_agentprint_iat_tuning -v`

Expected: import fails because `agentprint.iat_tuning` does not exist.

- [ ] **Step 3: Implement the isolated tuning CLI**

Build each trial's config under its own output directory. Persist `proposal.json` before expensive work. Create/resume the JSON-backed optim-agent study sequentially. Reject duplicate trial/seed combinations, manifest drift, changed baseline protocol, and exhausted 17-trial/12-search/4-repeat counts. Set the first-baseline 3600-second deadline once and never reset it. Mark exceptions `failed`; if the study deadline expires, report the latest committed validation score and mark the trial `pruned`.

- [ ] **Step 4: Verify isolated synthetic/public behavior**

Run: `PYTHONPATH=tests:. python -m unittest test_agentprint_iat_tuning -v`

Expected: parameter bounds, fixed workload, study identity, baseline rule, and trial-limit tests pass without loading real test data.

- [ ] **Step 5: Commit**

```bash
git add agentprint/iat_tuning.py tests/test_agentprint_iat_tuning.py docs/agentprint-tuning.md
git commit -m "feat(agentprint): tune replace log1p with sequential ask tell"
```

### Task 3: Verify the Integrated Tuning Contract

**Files:**
- Modify: `tests/test_agentprint_iat_integration.py`
- Modify: `docs/experiments/2026-11-02-replace-log1p-optimization/README.md`

**Interfaces:**
- Existing synthetic full-scope fit/cache fixtures feed a short tuning-mode public train/evaluate integration test.
- The study report records exact command, commit, environment, every ask/tell state, completed/pruned/failed count, elapsed budget, and validation-only outcomes.

- [ ] **Step 1: Add a public CLI integration assertion**

Use a synthetic full-scope fixture and a two-epoch tuning test override to prove train/evaluate reconstruct the replace model from its checkpoint while formal and smoke tests retain their existing behavior. Production search parameters still require a 200-epoch cap.

- [ ] **Step 2: Run targeted v2/IAT tests**

Run: `PYTHONPATH=tests:. python -m unittest test_agentprint_v2 test_agentprint_v2_protocol test_agentprint_iat_tuning test_agentprint_iat_integration -v`

Expected: all targeted tests pass; no test cache or checkpoint is used as a substitute for measured study values.

- [ ] **Step 3: Run local targeted regression and static checks**

Run: `PYTHONPATH=tests:. python -m unittest test_agentprint_v2 test_agentprint_v2_protocol test_agentprint_iat_fit test_agentprint_iat_cache test_agentprint_iat_integration -v`

Expected: existing formal/smoke/cache behavior remains green.

- [ ] **Step 4: Commit**

```bash
git add tests/test_agentprint_iat_integration.py docs/experiments/2026-11-02-replace-log1p-optimization/README.md
git commit -m "test(agentprint): cover isolated IAT tuning workflow"
```

### Task 4: Run the Approved Sequential Search

**Files:**
- Create: `configs/agentprint/replace-log1p-tuning-20261102.yaml`
- Create/update: `docs/experiments/2026-11-02-replace-log1p-optimization/`
- Remote artifacts: `/root/autodl-tmp/ant-workspace/.optim-agent-runs/replace-log1p-20261102/`

**Interfaces:**
- All trials use existing full IAT train/validation cache and the selected replace/log1p model.
- First point is the stated baseline. Each subsequent seed-42 point is selected after reading all completed study results.
- Repeat the leading two new configurations at seeds 43 and 44 when the approved count/time budget permits.

- [ ] **Step 1: Verify server and data/cache identities**

Check GPU, CUDA/PyTorch, memory/disk, clean committed code, 2697/202 manifest IDs, full fit fingerprint, IAT cache fingerprints, and optim-agent 0.1.1. Fail closed if any identity differs.

- [ ] **Step 2: Run and record the baseline**

Use seed42, fixed anchor params, full train/validation, public train/evaluate, and complete-epoch recovery. Verify the measured score against its own reloaded validation checkpoint.

- [ ] **Step 3: Search sequentially**

Propose one bounded point at a time from completed history, persist its JSON before execution, and use ask/tell. Stop at 17 trials or 3600 seconds. Do not parallelize, read test, or alter data/model/batch/epoch rules.

- [ ] **Step 4: Repeat finalists and audit**

Use seeds43/44 for each of the top two new completed configurations if budget permits. Re-evaluate each completed best checkpoint on validation and audit all results against the study records.

- [ ] **Step 5: Report and commit evidence**

Report baseline, best seed42 result, finalist three-seed means when complete, exact params, completed/pruned/failed trials, wall time, and reproduction command. State clearly if no replicated winner is established. Commit only light result summaries; keep checkpoints and cache artifacts on the data disk.
