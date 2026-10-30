# Scenario/raw optimization, round 2

Continue the user-approved optim-agent experiment protocol in a separate study. The user requested another round informed by the completed measurements. Carry forward the approved maximum of 17 evaluations or 3600 seconds of study wall time, whichever comes first; do not modify or extend round 1's deadline.

## Fixed protocol

- Scenario classification, five classes, raw MTAM inputs; identical official train/validation manifests. Never read test.
- Explicit sequential optim-agent ask/tell proposals chosen from all completed results. Existing tuning implementation remains unchanged.
- From scratch, seed 42 search, deterministic FP32, Adam, ordinary CE, workers 0, numerical-library threads 4.
- At least 60 and at most 200 epochs; preserve the previously approved joint F1/loss plateau rule and minimum learning rate of 1e-6.
- Baseline uses the original default point through the existing CLI. Re-run its 90-epoch trajectory and compare against round 1 to verify the environment before using historical incumbent scores.
- Retain round 1's winner as incumbent: width 2, learning_rate .001, decay_per_30 .5, weight_decay .001, batch_size 200, dropout_2d .05, dropout_1d .1. Its seed 42/43/44 scores are .8025304272460534, .7689083471226328, .7845288179544058; mean .785322530774364.
- Reserve four slots and sufficient wall time to repeat the two best new completed configurations under seeds 43/44. Compare their three-seed means with the historical incumbent only after baseline/environment/cache parity is established. If full replication cannot finish, explicitly report incomplete selection and retain the incumbent recommendation.
- A deadline-ended trial is pruned, a crash failed. Do not claim convergence for a cap-ended run. Preserve all trial artifacts on /root/autodl-tmp and back up final lightweight records and best models with SHA256 verification.

## Evidence and next proposals

Round 1 favors higher capacity and a slower learning-rate schedule. The decay .5 point also leads decay .2 in three-seed mean, supporting local exploration. Weight decay and batch conclusions have only one seed per comparison. Do not infer causal certainty or global optimality from these measurements.

Hold width at 2 initially and probe learning_rate around .0007-.002, decay_per_30 around .35-.75, dropout_2d around 0-.1, and dropout_1d around .05-.2. These remain inside the approved legal parameter bounds. Batch 200 and weight decay .001 are the initial anchors; change one only if completed evidence warrants it. No widening beyond 2, new loss, data transformation, or split changes in this round.

First new point: learning_rate .0015 with all other incumbent parameters unchanged. This tests whether a moderately higher starting rate helps before changing decay or regularization. Each later point must use completed history, rather than a pre-launched grid.

## Execution and acceptance

- [x] Record Git branch and preserve source/configuration identity.
- [x] Establish baseline and validate round-1 trajectory/environment/cache parity.
- [x] Sequentially run informed new configurations inside the count/time limit.
- [x] Complete top-two new-config seed 43/44 replication when the budget permits.
- [x] Independently audit checkpoints, metrics, stopping, ask/tell, and final selection against incumbent.
- [x] Back up results with hashes; commit the report and reproduction parameters; confirm training has stopped.

Operational scripts may reuse the prior round's orchestration with a new study/development directory. Production code and old artifacts are preserved. No additional GPU work is authorized after this round's deadline.

## Outcome

12 complete trials in 3324.08 seconds; no failed/pruned. Both challengers completed three seeds. Retained round-1 incumbent (.785323 ± .016825), ahead of new challengers (.770611 ± .006130 and .756170 ± .033291). Source, environment and cache parity passed; 9 checkpoints independently re-evaluated; related 14 tests passed. See `docs/experiments/2026-10-30-scenario-raw-tuning-round2/README.md`.
