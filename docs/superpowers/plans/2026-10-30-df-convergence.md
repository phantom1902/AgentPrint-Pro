# DF convergence experiment

Goal: fulfill the user's request to train DF to the previously approved strict validation plateau, keeping DF defaults and the old 30-epoch run intact.

The original DF checkpoint contains only the best model (epoch 27), without final model/optimizer/RNG. Reconstruct the first 30 epochs from seed 42 using the same cached inputs, model, Adamax .002, batch128, drop_last, and numerical environment. Verify every recorded training loss and validation metric, plus the selected checkpoint tensors, against the original baseline before any extension. If replay differs, stop rather than call it exact continuation.

Use the existing ConvergencePolicy: baseline30, at least30 additional epochs, F1 patience20 at delta .001, adjacent10-epoch mean validation CE improvement below1%, total cap200. Keep original validation-accuracy checkpoint selection (latest tie); F1 determines stopping only. The original DF has a fixed .002 LR: do not introduce AgentPrint's LR schedule. Test is evaluated only after final checkpoint selection.

Files: independent `tools/df_convergence.py`, tests in `tests/test_df_convergence.py`, results under `docs/experiments/`. Reuse existing WFlib loaders/model/optimizer and recovery helpers. Save immutable best snapshots and atomic last model/Adamax/RNG/history so interrupted extensions resume complete epochs without losing optimizer state. No production baseline file changes.

GPU replay audit: the first-epoch training loss failed exact comparison (.0013 difference); a one-epoch run through the original native entry point also did not reproduce the original metric. No extension was performed from this failed replay. The user was informed that the old checkpoint cannot support exact continuation. Fulfill the existing request with a separately identified `--fresh` deterministic training experiment instead: disable TF32 and cuDNN benchmarking, enable deterministic algorithms and CUBLAS workspace configuration; retain model/optimizer/LR/batch/splits. In this mode the first30 are a new baseline, `baseline_parity_passed=false`, and parent metrics are a historical comparison only. Never label it as exact continuation of the old30.

- [ ] Write real CPU integration tests for baseline replay/checkpoint parity, parent preservation, cap reporting, idempotence, input identity rejection, and interrupted optimizer/RNG continuation. Observe failure before implementation.
- [ ] Implement the independent runner, run new tests and existing convergence rule tests, commit and sync the source.
- [ ] Run GPU replay and extension on the data disk; independently verify first-stop rule, checkpoint selection and metrics. Report cap as not converged.
- [ ] Back up and verify model/metrics/state, commit results and reproduction command, confirm GPU stopped.
