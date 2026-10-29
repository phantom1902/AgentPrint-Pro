# AgentPrint ANT scenario tracer (T01)

This independent module implements the supported clean scenario/raw path. It
does not modify the existing risk baseline suite. The classifier is the actual
RF reference CNN, adapted only at the first Conv2d input channels (2) and the
last class count (5 or 47); its reshape, final BN/ReLU and initialization remain
the reference implementation. Public `agentprint.model.getRF(5)` and `getRF(47)`
accept `[B,2,2,1800]` and return class scores for ordinary unweighted cross entropy.
The 47-class model contract does not imply primitive preparation is implemented.

## Reproducible synthetic CPU demonstration

Use Python with `requirements-agentprint.txt` and `scapy` installed. Use a fresh
directory outside the source checkout; preparation refuses existing cache
outputs until T04 adds compatible shard recovery. These are illustrative shell
commands; substitute absolute paths appropriate to the machine.

```text
python tests/make_agentprint_fixture.py /tmp/agentprint-demo/data
python -m agentprint prepare --config configs/agentprint/smoke_scenario_raw.yaml --data-root /tmp/agentprint-demo/data --cache-root /tmp/agentprint-demo/cache
python -m agentprint train --config configs/agentprint/smoke_scenario_raw.yaml --cache-root /tmp/agentprint-demo/cache --output-root /tmp/agentprint-demo/runs --device cpu
python -m agentprint evaluate --config configs/agentprint/smoke_scenario_raw.yaml --cache-root /tmp/agentprint-demo/cache --checkpoint /tmp/agentprint-demo/runs/RUN_ID/best.pt --device cpu --split val
python -m unittest discover -s tests -p test_agentprint.py -v
```

The fixture has an 1800-second capture, one unique RFC1918 initial SYN source,
and two peers. Participation counts are 4, 3 and 1. Expected outgoing packet
counts are one in bins 0, 1 and 1799; incoming count is one in bin 900. Each
packet has IP total length 40, independently of its 54-byte Ethernet frame.
All other cells are zero. These synthetic results establish functionality,
not traffic-classification quality. Real CPU acceptance is a later ticket.

## Current support boundary

Supported input layout is `SPLIT/ENGINE/SCENARIO/EPISODE/network.pcap`, preserving
official `train`, `val`, `test` identities and the fixed sorted five-class map.
Classic PCAP microsecond/nanosecond, either byte order, untagged Ethernet and
complete unfragmented IPv4/TCP are supported. Non-IP frames affect the capture
interval and are counted in quality but do not enter MTAM. All observed TCP
packets, control packets and retransmissions remain included. Other runtime
connections are included; no per-connection filtering is used.

The unique RFC1918 SYN-without-ACK source must also be the unique IP packet
participation leader. Ties, no candidate, multiple candidates, unsupported IP
or transport, fragmentation, unreadable headers, frame truncation and damaged
record streams exclude the whole episode with a reason. No prefix cache is
produced for an excluded episode. IPv6, non-TCP and partial IP packet support
belong to T03. The inference evidence is not ground-truth identity metadata.

`prepare` currently accepts scenario only. `train` and `evaluate` accept
scenario/raw only. Primitive annotations, full parse quality, cache recovery,
four-configuration protocol and training recovery/TensorBoard logging are
explicitly left to T02–T06. `--resume` currently fails explicitly. The T01
training implementation saves a validation-Macro-F1 `best.pt`, one JSONL row
per epoch and validation CSV/JSON outputs. Strict `>` replacement preserves
the earlier epoch on ties. Evaluation uses only classes with true support
for its main macro average; fixed model heads and support arrays do not shrink.

## Stable extension contracts

YAML uses `extends: relative/path.yaml` for recursive common+overlay merging.
Only these fields are valid; every field is required after merging:

| Group | Fields |
| --- | --- |
| Top-level | `schema_version` (1), `task` (scenario/primitive), `transform` (raw/log1p), `mode` (formal/smoke), `seed` |
| `paths` | `data_root`, `cache_root`, `output_root` |
| `preprocessing` | `bins` (1800), `shard_size` (1024), `splits` (unique official names) |
| `training` | `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `optimizer` (adam), `lr_decay`, `num_workers` (0) |

Runtime CLI options are `--config`, data/cache/output root overrides, device,
training `--resume`, and evaluation checkpoint plus val/test split. No experiment
parameter flags are accepted. A merged configuration is written in each unique
run directory. Config supports both task/transform names for subsequent tickets;
the command support checks state which paths are implemented.

`parsing.iter_records` streams `(integer_time_ns, raw_ethernet)`;
`decode_packet` returns a decoded `Packet` or None for non-IP;
`scan_episode` returns the complete capture interval, endpoint set and evidence,
and quality; expected exclusions use `EpisodeExcluded(reason, detail, **context)`.
`direction` returns 0 (outgoing), 1 (incoming), or None. `preparation` composes
discovery, sample construction and cache writing; it retains no whole packet
list. `build_mtam` uses the complete interval with exact integer bin arithmetic.
T02 should add its segment builder alongside `scenario_samples`, keeping the
sample-and-array iterator seam to `cache.write_split`.

The cache contains `label_map.json`, `index.json`, and per-task/split
`valid.jsonl`, `excluded.jsonl`, `metadata.json`, `shards/00000.npy` etc. Shards
are uint64 `[n,2,2,1800]`, 1<=n<=1024, and every row is a complete sample. The
first statistic is packet count, the second IP bytes; directions are outgoing,
incoming. Index source entries contain PCAP relative path, size and mtime_ns.
PCAP payload content is never persisted in metadata. The index currently has
empty annotation metadata because T01 does not consume annotations.

Valid records use the complete specified field set: sample_id, task, split,
engine, scenario, episode_id, label_name/id, pcap_relpath, annotation_relpath/row,
start_ns/end_ns/duration_ns, runtime_endpoints, endpoint_source, packet_count,
ip_bytes, shard_relpath, row_index, quality. Annotation fields are null for
scenario rows; packet_count/ip_bytes are `[outgoing,incoming]` totals. Exclusions
retain available source and label fields plus reason/detail and known timing;
they never contain shard locations. Relative source paths resolve against data
root; shard paths resolve against cache root. Sample IDs include task and full
official episode identity so engine/split collisions cannot alias.

Manifest metadata identifies format/preprocessing versions, preprocessing
config fingerprint, label map ID and manifest ID, split support, exclusions,
and completion state. The index tracks source inventory identity and per-split
metadata. Fingerprints exclude transforms/training settings and storage roots
from the preprocessing contract. T04 should extend commit/recovery mechanics
around these whole-shard files, not introduce per-episode MTAM copies.

`CachedSamples` reads shards with mmap and converts each selected row to float32
(raw or log1p). `metrics` owns fixed-label summaries and exports.
`training.evaluate_model` is the common scoring seam; `train` and `evaluate`
compose it with model/dataset/checkpoint artifacts. T05 should add task/transform
protocol checks and formal coverage rules; T06 owns latest-state recovery and
TensorBoard rather than changing the RF model or cache representation.

Reference source: https://github.com/robust-fingerprinting/RF/blob/master/RF/models/RF.py
(UTF-8/LF SHA256 `4cf9ff1a71a3601aeab61756327467add251475fa3bfd9c19eeb43e658effa19`).
This is a source-traceable RF adaptation, not a claim of exact AgentPrint replication.
