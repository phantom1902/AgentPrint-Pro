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
