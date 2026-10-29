# AgentPrint annotated primitive classification (T02)

T02 enables the independent primitive/raw `prepare`, `train`, and `evaluate`
path. It shares the parser, endpoint inference, MTAM definition and RF model
with scenario classification. Samples, label support, model weights and result
directories remain task-specific. The output head is always 47 classes:
N001 through N047 map to 0 through 46, including when a split omits classes.

Use `configs/agentprint/smoke_primitive_raw.yaml` for a deliberately small CPU
run and the same public commands as the scenario demonstration. Override
data/cache/output roots and device through the supported CLI options. The
smoke overlay uses one epoch and batch size 4; it does not limit the number of
raw Episodes. Select a small input data root explicitly. Synthetic acceptance
tests are available through:

```text
python -m unittest discover -s tests -p test_agentprint_primitive.py -v
```

Each Episode consumes its complete sibling `messages.jsonl`; missing logs are
reported as primitive exclusions. `messages.jsonl.incomplete` is never used.
Scenario eligibility is independent of those annotations. Each JSON object
uses `start_time`, an ISO timestamp with explicit timezone and at most nine
fractional digits. Conversion uses integer nanoseconds without a floating point
timestamp. The current label comes from `primitive_label_group_id`, falling
back to `macro_group_id` if the first field is absent. When both exist they
must agree. Historical `original_macro_group_id` and message text are ignored.

Records are sorted by exact start while preserving their one-based original
JSONL row identity. Every known start supplies a boundary, including starts
with invalid labels. Each interval ends at the next distinct start; the last
ends at the complete capture end. All rows in a duplicate-start group and its
interval are excluded. Out-of-capture intervals, invalid timestamps, labels,
zero duration and intervals without usable runtime packets receive explicit
exclusions. No offset, clipping, minimum duration, or minimum packet threshold
is introduced. Internal endpoints belong to the next interval. The capture-end
packet enters only its actual valid final interval, never an earlier interval
when the final one is excluded.

One full capture scan establishes shared endpoint evidence and full bounds.
One additional stream aggregates all valid intervals, with each interval split
into 1800 bins using its complete annotation bounds. MTAMs for the Episode are
held until this stream completes, so a damaged stream cannot publish a prefix
of valid primitive samples. Packet data and message bodies never enter the
manifests. Small complete annotation files receive SHA256 source identities;
PCAP files retain the T01 path/size/mtime identity contract. T04 extends cache
recovery and inventory orchestration around these sample generators.

The raw path uses ordinary unweighted cross entropy. Evaluation reports true
class support for all 47 classes, the macro average over classes with true
support, and the fixed-47 auxiliary macro average. The full four-configuration
raw/log1p selection protocol belongs to T05; training recovery and TensorBoard
belong to T06. This ticket establishes functionality, not classification quality
or completion of real-data CPU acceptance.
