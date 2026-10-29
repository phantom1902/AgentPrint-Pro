# Full-cache delivery verification

After both public `prepare` commands have finished, run the separate audit utility
from the checkout containing the verifier. The normal `agentprint` CLI retains
only `prepare`, `train`, and `evaluate`.

```bash
python -m agentprint.fullcache \
  --data-root /root/autodl-tmp/ant-workspace/data/raw/ANT \
  --cache-root /root/autodl-tmp/ant-workspace/data/processed/agentprint-full-v2 \
  --runtime-record /root/autodl-tmp/ant-workspace/full-preparation/t08-2b9e2c4/execution.json \
  --output /root/autodl-tmp/ant-workspace/full-preparation/t08-2b9e2c4/coverage.json
```

The raw and cache roots are read-only; output must be outside both. An incomplete
index, task, split, or completion journal fails with exit 1 and `verified: false`.
The utility does not resume preparation. Use the original public `prepare`
commands and compatible configuration to resume its committed shards.

Verification requires both tasks, all official train/val/test datasets, fixed
label mapping and current format/preprocessing/recovery identifiers. It checks
current source inventory against each cached contract: every capture's relative
path, size and nanosecond mtime, and annotation/endpoint metadata SHA256. It does
not read or hash PCAP payload; replacement preserving capture size and mtime
cannot be detected under this existing identity policy.

Manifests must exactly match ordered commit records, final source cursors, stored
counts, fixed-class supports and reason counts. Scenario records cover every
capture exactly once. Primitive records cover every annotation row exactly once,
or one explicit missing/empty-annotation exclusion for the corresponding episode.
Small annotation metadata is reread to verify row identity, label and exact
timestamp; message bodies never enter the report. Valid rows must retain coherent
intervals, inferred endpoint evidence and capture provenance. Each shard must be
referenced exactly once with every array row covered exactly once; all uint64
`[n,2,2,1800]` values are read to reconcile packet/IP-byte totals, shapes and bytes.
This validates artifact consistency, without independently reinterpreting raw
traffic or performing model evaluation on test data.

`datasets[task/split]` reports valid/support, excluded, all exclusion reasons,
coverage and excluded fraction, including every predetermined class with zero
support. `classes.__unknown__` explicitly counts unavailable or unrecognized
labels; `excluded_labels` records those labels without assigning a fixed class.
Coverage is valid/(valid+excluded), and is null for a zero denominator. Primitive
missing/empty-log exclusions count episodes, not an invented number of absent
annotation rows. The coverage definition is recorded in the JSON. Scenario
eligibility is independent of whether primitive annotations are present.

Exit 0 means a consistent delivered cache, even when some fixed classes have no
training support. Check `formal_training[task].eligible` independently for each
task. Missing training classes are listed explicitly, and block that task's
formal training; another task's cache and quality report remain deliverable. The
existing formal training command enforces its own support gate. This utility
never starts training or changes the fixed class set.

Optional `--runtime-record` preserves a separate preparation execution record
and its SHA256 in `preparation_execution`. Accepted statuses are
`prepared_pending_verification`, `verified`, or `complete`, with successful command
records for both tasks; `running` and failed/incomplete commands are rejected.
If present, its `cache_root` must match. Without this record, preparation execution
is explicitly null; preparation duration or source revision is never inferred
from file timestamps. `verification_execution` separately records verifier Git
SHA, Python/NumPy/platform and audit duration; `storage` records cache/NPY bytes
and free disk at verification. Keep the root execution record's actual preparation
SHA, environment, start/end times and disk measurements for reproducibility.

Acceptance tests use regular synthetic files, both public preparation commands,
and subprocess invocation of this utility:

```bash
python -m unittest discover -s tests -p test_agentprint_fullcache.py -v
```
