# Supplemental IAT preparation and ten-channel loading

The v2 public pipeline supports enabled IAT with the existing single and
replacement model factories. First prepare the original Scenario MTAM with an
IAT-disabled configuration, then enable IAT and run explicit `fit-bins`,
`prepare`, `train`, and `evaluate`. Enabled preparation never rebuilds MTAM or
fits boundaries implicitly. A missing/incompatible base cache or fit fails.

`feature.iat_cache_root` must be separate from the base cache, original data,
fit artifact, and fit progress (including ancestor/descendant overlaps).
The fit must match the selected train manifest, source inventory and base
metadata exactly. `mode: smoke` requires a smoke fit; formal mode requires a
full fit. Selecting train/val does not read test captures.

Each `scenario/<split>` supplemental dataset contains `identity.json`,
`commits/*.json`, `shards/*.npy`, `valid.jsonl`, and `metadata.json`. Raw arrays
are uint64 `[N,8,2,1800]`, with 256 samples per shard and an optional short final
shard. Every effective interval is counted, including those not sampled during
fitting. Zero uses bucket 0; positive intervals use the shared six boundaries
with equality assigned right. Current-packet direction and its original capture
slot determine attribution; the first packet contributes no interval.

Atomic array publication precedes the atomic JSON shard commit. Only committed
shards count during recovery; orphan NPY and temporary NPY files are discarded.
Each commit binds the exact sample IDs, fit/base/source identity and array SHA256.
Publication of the final manifest and complete metadata is repeatable after a
crash. Changed identities, corrupt commits and corrupt committed arrays fail
instead of being silently replaced. Only aggregates and quality counts persist.

`v2.loader` retains the base manifest order and joins IAT by `sample_id`, so the
base's 1024-sample layout need not match the supplemental layout. Inputs convert
from uint64 to float32 and then apply raw or log1p across all ten channels.
`samples.metadata` remains the original base metadata; `samples.iat_metadata`
and `samples.fit_fingerprint` expose separately verified supplemental identities.

Experiment/checkpoint identity stores those verified identities, never reference
path strings. Evaluation reconstructs the saved model and verifies both saved
train/validation cache identities and the fit, including when explicitly
evaluating a synthetic test split. A test supplemental cache must independently
match that fit and its own base manifest/sources. Artifact paths can relocate
when the content identities remain unchanged.
