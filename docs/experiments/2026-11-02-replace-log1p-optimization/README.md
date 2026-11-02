# Replace-multikernel log1p parameter exploration

The approved study maximizes best validation Macro-F1 with at most 17 evaluations
and 3600 seconds of study wall time. It fixes the five-class Scenario task,
replace model, log1p transform, width 2, batch 200 and search seed 42. Each trial
starts from scratch and uses the existing joint stopping rule (60–200 epochs).
Search varies learning rate, weight decay, LR decay and both dropout values.
No real test split is loaded.

Design and implementation requirements are recorded in
`../../superpowers/specs/2026-11-02-replace-log1p-optimization-design.md` and
`../../superpowers/plans/2026-11-02-replace-log1p-optimization.md`.

Remote source and every generated artifact are under
`/root/autodl-tmp/ant-workspace`. Study proposals, checkpoints, outcomes and
ask/tell history live in `.optim-agent-runs/replace-log1p-20261102` in the
isolated remote Git checkout. Existing formal outputs are preserved.

`preflight.py CONFIG` verifies the full train/validation workload against the
saved formal baseline identity (allowing only the mode to change), counts,
CUDA, optim-agent version and free disk before starting the study clock.

The preserved formal reference has validation Macro-F1 0.8342599314377898,
best zero-based epoch 77, and stopped after 108 epochs. This study first
re-runs that parameter point in tuning mode. Final evidence will distinguish
the best seed-42 observation from a replicated finalist comparison.

## Verification before search

- Source revision: `36a899c` on `codex/replace-multikernel-log1p-optim`.
- Local v2 configuration/protocol regression: 19 tests passed.
- Remote v2/protocol/trial/public-IAT integration: 34 tests passed in 170.347 s.
- Remote IAT fit/cache regression: 14 tests passed in 58.701 s.
- Final runner revision: 14 tests passed in 0.823 s, including real Linux
  parent-SIGKILL and deadline lifecycle checks. No skips on Linux.
- Independent review approved the implementation after adding parent-death
  protection and child-held study lock. Validation-score agreement uses an
  absolute tolerance of 1e-12 with zero relative tolerance.
- [Preflight evidence](preflight.json) verifies 2697/202 samples, matching
  formal identities and available CUDA. It was recorded immediately before
  the runner-only lifecycle fix; fit/cache/model/training behavior is identical.

The first baseline was launched as a detached process with the committed source.
Its complete-epoch metrics are compared with the preserved formal reference;
the study and all trial commands retain the original 3600-second deadline.

## Results

The study stopped at the approved 3600-second deadline. Seven evaluations were
proposed: six completed and converged, one was pruned, and none failed. Terminal
prune bookkeeping completed 0.325 seconds after the deadline; the training child
was terminated by the original subprocess timeout. GPU usage returned to zero.
The 17-evaluation cap was not reached. The intended 12-new-point allocation could
not fit this runtime; three new points were evaluated sequentially, followed by
finalist repeats. No extra trials or budget extensions were started.

| Trial | Configuration change from anchor | Seed | Epochs | Best validation Macro-F1 | State |
|---|---|---:|---:|---:|---|
| 00 | Exact anchor | 42 | 108 | 0.834260 | Complete, converged |
| 01 | Initial LR 0.0005 | 42 | 69 | 0.819053 | Complete, converged |
| 02 | Dropout 2D/1D 0.10/0.25 | 42 | 106 | 0.830897 | Complete, converged |
| 03 | Weight decay 0.003, LR decay 0.7 | 42 | 85 | 0.809966 | Complete, converged |
| 04 | Repeat trial 02 | 43 | 84 | 0.817404 | Complete, converged |
| 05 | Repeat trial 01 | 43 | 71 | 0.800917 | Complete, converged |
| 06 | Repeat trial 02 | 44 | 91 | 0.827516 observed | Pruned, not converged |

Trial 06 has no completed objective. Its latest committed validation Macro-F1 is
0.820378; optim-agent records that intermediate with state `pruned`. The observed
best 0.827516 was not re-evaluated as a completed trial and is excluded from
finalist means. Trial 01's seed 44 repeat was not started because the budget was
exhausted. Thus no complete three-seed finalist comparison or replicated winner
is established. Completed seeds 42/43 average 0.824150 for stronger dropout and
0.809985 for the lower learning rate; these are explicitly two-seed summaries.

The best seed-42 result remains the anchor: validation Macro-F1 0.8342599314 and
Accuracy 0.8415841584. Its parameters are:

```json
{"learning_rate": 0.001, "weight_decay": 0.001, "lr_decay": 0.5,
 "dropout_2d": 0.05, "dropout_1d": 0.1}
```

The model stays replace/log1p, width 2, batch 200, seed 42, FP32 with the approved
joint stopping rule. This is the best observed point in this bounded study,
not a global optimum. No new configuration showed an improvement over the
anchor on seed 42. The anchor has not been repeated at seeds 43/44 in this study,
so two-seed candidate means do not establish statistical superiority/inferiority
to a replicated baseline.

Every completed trial reloaded its immutable validation-best checkpoint through
the public validation command and verified the score against training history.
The final audit checked ask/tell correspondence, frozen trial identities, every
learning-rate value, 14 Adam steps per epoch, committed history, first stopping
prefix, convergence, and validation sample count. No real test data was loaded.
All 108 baseline epochs exactly match the original formal experiment on nine
deterministic fields; see [baseline parity](baseline-parity.json).

Full machine-readable evidence: [results](results.json). Remote checkpoints,
proposal rationales, commands, logs, configs and study JSON are preserved under
`/root/autodl-tmp/ant-workspace/.optim-agent-runs/replace-log1p-20261102`.
The winning checkpoint is the `best.pt` in
`trial-00/runs/v2-65f4bf386ae5-20261102T093500Z-30924347/`.

## Reproduction

Run from the committed remote checkout
`/root/autodl-tmp/ant-workspace/src/AgentPrint-Pro-IAT-Optim` at `36a899c`.
For a fresh baseline study, use a new study directory, write the anchor JSON
above to `params/00.json`, then execute:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
/root/autodl-tmp/ant-workspace/envs/optim-agent/bin/python -u -m agentprint.iat_tuning \
  --study-dir /root/autodl-tmp/ant-workspace/.optim-agent-runs/NEW-STUDY \
  --trial-name trial-00 \
  --params /root/autodl-tmp/ant-workspace/.optim-agent-runs/NEW-STUDY/params/00.json \
  --phase baseline --seed 42 --device cuda \
  --config configs/agentprint/replace-log1p-tuning-20261102.yaml \
  --rationale baseline-reproduction
```

This command is documentation only; a new budgeted study has not been started.
