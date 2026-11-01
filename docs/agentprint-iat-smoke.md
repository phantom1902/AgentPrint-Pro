# Bounded IAT acceptance entry

`python -m agentprint.iat_smoke` implements the approved real-data smoke budget.
Run it only from a clean, committed implementation after T07 integration. It
reads an existing Scenario MTAM cache, links its selected arrays without copying
or rebuilding them, and never opens the test split. Linux uses symbolic links;
Windows uses hard links on the same volume, without requiring symlink privileges.

Example on the designated server data disk (the base config must reference the
existing full MTAM cache and original raw data):

```bash
python -m agentprint.iat_smoke \
  --config /root/autodl-tmp/ant-workspace/acceptance-base.yaml \
  --workspace-root /root/autodl-tmp/ant-workspace \
  --work-root /root/autodl-tmp/ant-workspace/acceptance/iat-20261101
```

The work root must be new, outside Git, and inside the specified data workspace.
The caller must set `TMPDIR` on that data disk before starting Python; the entry
also confines its runtime temporary/cache directories to the new work root.
There is no automatic overwrite or retry of a failed acceptance directory.
`--select-only` publishes the selection, linked base view, and six configurations
without fitting or training. This provides a cheap, explicit inspection mode.

Selection uses existing valid rows only, at most four train and two validation
Episodes per class, preferring smaller PCAPs then lexicographic sample identity.
Missing message annotations do not exclude an Episode. The selected source
identities and base arrays are validated through the public IAT interface before
fitting. The original cache index and sources remain available in the view;
selected manifests have their own fingerprints and counts.

The normal entry runs one shared train-only fit and one train/val supplemental
prepare, followed by six serial two-epoch seed-42 runs: single/replace/prepend
crossed with raw/log1p, always with IAT enabled. Configurations come from the
committed common-v2 optimization anchor (batch200, width2, FP32, four numerical
threads, zero loader workers). CUDA is used if available, otherwise CPU. No
resource failure reduces the budget, changes the model, or skips a configuration.

Each best checkpoint is reloaded for validation evaluation; each completed run is
resumed without adding epochs. SHA256 checks confirm that last/best checkpoints,
metrics and result remain unchanged by this completed-run resume. Actual
interrupted-epoch equivalence is separately covered by T07 integration tests.

Inspect `selection.json`, six `configs/*.yaml`, `commands.log`, `fit.json`,
`iat-cache/scenario/{train,val}/metadata.json`, and `summary.json`. The summary
records the committed Git revision, hardware, fit/cache identities, parameter
counts, run locations, validation metrics, resume checks, and explicit failures.
A failed run stops immediately and preserves its evidence. These smoke scores
provide execution acceptance only; they do not rank methods or show convergence.
Full IAT preprocessing, formal training and real test evaluation remain outside
this entry's scope.
