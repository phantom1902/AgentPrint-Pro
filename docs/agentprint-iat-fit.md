# Shared IAT fitting

Run `python -m agentprint fit-bins --config <v2-config.yaml>` explicitly before
IAT preparation. The configuration must enable IAT and name `feature.fit_artifact`.
V1 and disabled-IAT configurations reject this command. T02 supplies fitting;
IAT-enabled prepare/train remain blocked until the supplemental-cache integration.

Fitting requires a completed base Scenario cache. Only valid `scenario/train`
manifest rows select captures; fitting never discovers captures or reads val/test.
A selected cache view can publish selected valid rows and corresponding metadata
while retaining the original source inventory and references to original arrays.
The command checks PCAP path/size/mtime and annotation/endpoint metadata hashes
against that inventory. It does not regenerate MTAM or change eligibility.

`agentprint.iat.effective_episode(data_root, record)` is the shared feature seam.
It uses the base record's full capture interval and runtime endpoints, and the
existing effective-packet decoder/direction rule. `EpisodeIAT.timestamps_ns`
(int64) and `directions` (uint8) contain stably sorted effective packets, including
the first packet. `intervals_ns` is uint64 with one fewer entry. Its matching
`current_times_ns` and `current_directions` views attribute each interval to the
current packet. `quality.timestamp_inversions` counts adjacent effective-record
timestamp decreases before sorting. Memory/sort failures stop processing.

Each Episode samples at most 4096 positive intervals uniformly without replacement
using NumPy PCG64 and the first 128 SHA256 bits of `42:<sample_id>`, interpreted
big-endian. Smaller Episodes use all positive intervals. Zero-only Episodes stay
in classification but do not participate in the fit. Participating Episodes each
have equal total weight. An exact integer common denominator determines the six
empirical inverse-CDF quantiles at 1/7 through 6/7 without interpolation. Duplicate
observed boundaries remain. Zero belongs to bucket 0; for positive intervals use
`1 + searchsorted(boundaries_ns, interval_ns, side='right')`.

The completed JSON artifact records `kind: agentprint-iat-fit`, `format_version: 1`,
`scope: smoke|full`, `rules`, `train_manifest`, `sources`, `base_cache_identity`,
`boundaries_ns`, per-Episode summaries, statistics and a SHA256 `fingerprint` of
all other fields. `load_fit_artifact(path)` validates this contract. Model,
transform, training seed and artifact path do not enter the fit identity.
`fit_sample_positive_bucket_counts` and `empty_positive_buckets` describe the
fitting sample; downstream feature generation must count every effective interval.

`<artifact>.progress/identity.json` binds the fit identity. Atomic per-Episode JSON
commits under `episodes/` retain at most 4096 sampled integer intervals, quality
counts and a commit checksum. Successful Episodes are reused after interruption;
source/rule/scope changes and corrupt commits are rejected. No packet payloads or
full packet sequences are persisted. An entirely zero-only fit pool fails without
publishing fabricated boundaries.
