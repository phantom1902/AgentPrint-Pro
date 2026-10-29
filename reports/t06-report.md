# T06 training recovery implementation

Branch: `codex/ant-t06`, based on T01 `13264f0`. Scope: `training.py` loop,
new `recovery.py`, public-command recovery tests, recovery documentation.
No installs, SSH, GPU execution, real-data acceptance, push or root merge.

## Red / green evidence

All commands used
`D:\AllenLeeSpace\studyziliao\paper_code\.agentprint-worktrees\.venv\Scripts\python.exe`
with `PYTHONUTF8=1`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` from this worktree.

- RED: `python -m unittest discover -s tests -p test_agentprint_recovery.py -v`
  failed 3 tests in 45.761s: no `last.pt`, recovery explicitly unimplemented.
- Intermediate verification found a negative transform test reached T01's
  intentional raw-only gate before recovery; changed the isolated T06 test to
  supported preprocessing/seed incompatibility cases. T05 owns removing that
  support gate. Recovery interruption and archival tests already passed.
- GREEN: same focused command, 4 tests passed in 210.865s. Real CLI subprocess
  termination after epoch 0, then resume; exact model tensor and visible metric
  equality with uninterrupted CPU training, except elapsed time. Completed-run
  resume adds no epochs. An authentic earlier `last.pt` replay verifies archived
  newer/partial JSONL rows, repair of damaged `best.pt`, and effective TensorBoard
  epochs `[0,1,2]`. Another small synthetic CPU run verifies global step 50 after
  resume and final step 54. Negative CLI cases reject seed/preprocessing/training,
  label map, model/preprocessing versions and train/val manifest identities;
  relocating cache/output roots is accepted and recorded.
- `python -m compileall -q agentprint tests/test_agentprint_recovery.py`: PASS.
- `python -m mypy --check-untyped-defs --follow-imports=silent --ignore-missing-imports agentprint/training.py agentprint/recovery.py`:
  PASS, 2 files. The body check initially caught the optional missing-package
  version value; the annotation was corrected and the same command rerun green.
- `git diff --check`: PASS.

These are synthetic correctness tests. They do not consume the four real CPU
acceptance configurations or constitute formal training results.

## Contract and artifacts

- New unique run IDs retain task/transform/seed/UTC/random suffix and use
  `mkdir(exist_ok=False)`; resume explicitly names an existing run directory.
- Atomic `last.pt` commit includes model, Adam, epoch/global step, best
  score/epoch/immutable relative snapshot pointer, Python/NumPy/PyTorch/CUDA RNG
  and explicit shuffle-generator state. Target epochs and schedule are unchanged.
- Best snapshots have unique filenames in `checkpoints/`; they precede the
  last commit. Convenience `best.pt` and validation JSON/CSVs are published after
  last and repaired from its pointer on resume. An interrupted candidate cannot
  overwrite the previously committed best snapshot.
- Logs beyond last are archived, including duplicate/partial lines. Epoch and
  global-step TensorBoard streams have independent restart/purge positions and
  retain their historical event files. `training_steps.jsonl` records every 50
  global training steps; `metrics.jsonl` and TensorBoard include epoch losses,
  accuracy/F1, applicable fixed47 F1, LR/time/counts/global step.
- Accepted launch history preserves configuration/paths/device, Git commit/dirty
  state and environment package versions. Latest merged config stays readable
  as `resolved_config.yaml`. FP32/determinism/TF32 settings are recorded. OOM or
  nonfinite loss still fail explicitly without adaptive parameter changes.
- Human-facing behavior documented in `docs/agentprint-recovery.md`.

## Required root integration with T05

Keep T05's `supported`, `evaluate`, `validate_training`, `protocol_metadata`,
formal gate/selection helpers, and metrics/config changes. Apply the T06 train
loop and recovery module, then:

1. Keep the optional `loader(..., generator=None)` argument and pass it to
   DataLoader. Retain T05's empty-train case hook invoking `validate_training`
   before the generic no-valid-samples error.
2. Invoke `validate_training(config, train_samples)` immediately after the train
   loader. Retain T05's `protocol.json` writing hook using `protocol_metadata`.
3. Use task-specific `getRF(K)` in train/evaluate; isolated T06 intentionally
   retains T01 `getRF(5)` and supported scenario/raw boundary.
4. The best checkpoint retains original top-level `config`, `model`, `epoch`,
   `best_macro_f1`, label/version and manifest keys expected by T05; nested
   `validation_result` supports repair and does not replace those keys.
5. Root's preprocessing version v2 is inherited through imported constants;
   this branch used T01 v1 fixtures only, without changing constants.

No unresolved specification conflict. Cross-device continuation is allowed and
recorded without bitwise guarantees. Concurrent writers to the same resume run
are not coordinated. Recovery requires at least one fully committed epoch.

Commit: see the commit containing this report (`git log -1 --oneline`).
