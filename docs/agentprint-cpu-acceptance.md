# T07 real ANT CPU acceptance helper

`tools/agentprint_cpu_acceptance.py` is an independent acceptance utility. It does
not add flags to `python -m agentprint`, edit original annotations, change quality
rules, modify the production loader, install dependencies, or access SSH.

Run only after integration tests and both model-head CE/backward checks have
passed on the same checkout. Keep server environment changes and workloads
serial. Check the existing checkout and preserve user changes before syncing.

## Server command

From `/root/autodl-tmp/ant-workspace/src/AgentPrint-Pro`, after checking out the
accepted development commit, use a fresh workspace outside the checkout:

```sh
/root/miniconda3/bin/python tools/agentprint_cpu_acceptance.py run \
  --data-root /root/autodl-tmp/ant-workspace/data/raw/ANT \
  --work-root /root/autodl-tmp/ant-workspace/acceptance/t07-COMMIT \
  --cpu-threads 2
```

Replace `COMMIT` with the actual short SHA. The script records the full SHA and
checkout status, interpreter/package versions and import checks, CPU thread
settings, exact public-command argv/cwd, stdout/stderr and wall times. It performs
no package installation; a failed import leaves acceptance incomplete. Device is
always CPU and subprocesses hide CUDA devices. OMP/MKL threads are recorded and
kept identical for all four configurations.

The script selects official train/val only, creates individual-file symlinks to
original captures, complete `messages.jsonl` and endpoint metadata, and generates
four ordinary smoke YAML overlays: one epoch, batch four, seed 42. It performs
four public `prepare` calls, checks both task caches before any training, then
serial `train` and validation-only `evaluate` for scenario/raw, scenario/log1p,
primitive/raw and primitive/log1p. The transforms share each task's fixed cache.

## Subset policy and limits

Selection considers Episodes with existing nonempty full annotations, preserving
all original lines and boundaries. It first maximizes feasible scenario coverage
using minimum annotation costs to reserve budget for remaining classes. Within
those constraints it prefers smaller PCAPs, with relative path breaking ties,
and fills further whole Episodes round-robin. It uses no model scores. This
sampling constraint does not alter production scenario-quality eligibility.

Train uses at most 20 Episodes and 64 physical annotation rows; val uses at most
10 Episodes and 32 rows. Therefore valid primitive counts cannot exceed 64/32;
valid counts can be lower after unchanged quality exclusions. Five-scenario
coverage is attempted whenever feasible under those whole-annotation budgets.
The selected Episode count can be far below 20/10. Actual support, exclusions,
missing scenarios, file identities and limitations are recorded. If either task
has zero valid train or val samples, the helper stops before training and reports
incomplete acceptance. It never deletes annotations to force a budget or selects
only a convenient prefix of a capture. No derived manifest or cache rewrites are
needed; full interval/provenance records and uint64 arrays come directly from
public preparation.

`selection.json` records PCAP relative path/size/mtime, full small-file SHA256,
original root, selected sources, split summaries and a selection identity. As in
the production cache contract, it cannot detect a PCAP content replacement that
preserves both size and mtime. No payload or annotation message body is copied
into metadata.

## Inspection and verification

`select` creates the selection, links and configurations without imports or
training. A subsequent `run` can execute that selected workspace. `prepare`
creates a fresh selection and prepares both tasks only, saving
`prepared_not_accepted`; this diagnostic execution does not count as acceptance.
An execution attempt is never silently repeated in its existing directory. Keep
failed artifacts and use a fresh path for another complete attempt; recovery of
an individual public command remains available through the regular CLI.

```sh
/root/miniconda3/bin/python tools/agentprint_cpu_acceptance.py select --work-root /root/autodl-tmp/ant-workspace/acceptance/t07-selection
/root/miniconda3/bin/python tools/agentprint_cpu_acceptance.py verify-cache --work-root /root/autodl-tmp/ant-workspace/acceptance/t07-COMMIT
/root/miniconda3/bin/python tools/agentprint_cpu_acceptance.py verify --work-root /root/autodl-tmp/ant-workspace/acceptance/t07-COMMIT
/root/miniconda3/bin/python -m unittest discover -s tests -p test_agentprint_cpu_acceptance.py -v
```

Expected workspace artifacts are `selection.json`, `selected-data/`,
`configs/{scenario,primitive}_{raw,log1p}.yaml`, `environment.json`,
`execution.json`, `commands.jsonl`, per-command logs, `cache/`,
`cache-verification.json`, `run-results.json`, `runs/`,
`artifact-verification.json` and `summary.json`. The final status is
`cpu_acceptance_passed` only after checking all four runs. Failure gives nonzero
exit and `incomplete` with a reason, retaining generated evidence.

Verification reads manifests and their fingerprints/counts/support, selected
source references, full integer-nanosecond intervals, label maps, exclusion
reasons, uint64 shard shapes/indices/statistics, best/last checkpoint state and
manifest references, one committed epoch, runtime/config records, fixed-class
CSV shapes/counts, and TensorBoard epoch scalars. Each run has training and
`evaluation-val` summary/per-class/confusion artifacts; comparison artifacts must
contain both transforms for both tasks. Training-step JSONL appears only at step
50, so it is legitimately absent from these bounded runs.

This proves only a bounded CPU execution path, not model performance, full-data
coverage, completion of full preprocessing, or formal GPU experiments. Keep NPY,
checkpoints and TensorBoard outside Git. Later evidence curation can retain
selected lightweight JSON/YAML/CSV, environment and command records.

Tests use tiny synthetic captures and cannot substitute for these real server
runs. On Windows without symlink privileges, four symlink-dependent tests skip;
run the complete focused suite on Linux before accepting the helper.
