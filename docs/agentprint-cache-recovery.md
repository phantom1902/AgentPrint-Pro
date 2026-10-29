# T04 resumable preparation

`prepare` automatically resumes a compatible task cache; raw/log1p share the
same integer statistics. Reuse requires matching format/preprocessing/recovery
versions, label map, preprocessing configuration and source inventory. Captures
use relative path, size and nanosecond mtime, never full content hashing. This
cannot detect capture replacement preserving both size and mtime. Sibling
`messages.jsonl` and JSON/JSONL/YAML endpoint metadata use SHA256 content hashes;
only paths and hashes are stored, without their contents.

The index retains independent `tasks[task]` compatibility contracts and merges
`datasets[task/split]`, allowing both tasks under one cache root. Existing T01
caches without recovery commits are explicitly incompatible.

Each task/split stores immutable `commits/NNNNN.json` records beside bounded
uint64 `[n,2,2,1800]` NPY shards (`n <= 1024`). The array is flushed and renamed
before its atomic JSON completion record; only that record publishes the shard.
Each commit includes valid/excluded metadata and a split-local source cursor.
Recovery preserves committed arrays and removes unpublished shard files.
Manifests and completed split metadata are atomically reconstructed from the
commit records. A final commit also records trailing exclusions or empty splits,
including when the previous commit contained exactly 1024 valid samples.

`resumed_samples(data_root, sources, generator, task, cursor)` wraps the existing
`generator(data_root, sources)` pair iterator. Its cursor has `source_index`
and `sample_offset`, where offset counts valid AND excluded rows. Scenario
advances directly to the next episode after its one record. Other task generators
can replay only the current episode and skip already committed rows; completed
source prefixes are not regenerated. It yields `(record, array_or_none, cursor)`
for the callable factory accepted by `cache.write_split`. No segment PCAP or
per-episode MTAM artifacts are introduced. Unexpected exceptions propagate while
completed commits remain recoverable; expected exclusions remain ordinary rows.

Acceptance: `python -m unittest discover -s tests -p test_agentprint_cache_recovery.py -v`
uses the public commands, terminates a prepare subprocess after the first full
1024-sample shard, adds an unpublished broken shard, and resumes. Recovered
manifests and arrays match an uninterrupted preparation; the recovered cache
then supports public CPU training and evaluation. Other cases cover raw/log1p
reuse and explicit incompatibility for changed configuration, versions, PCAP
mtime/source list, annotations and endpoint metadata.
