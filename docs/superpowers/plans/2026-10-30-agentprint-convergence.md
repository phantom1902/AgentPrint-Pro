# AgentPrint convergence continuation implementation plan

User-approved extension of the four completed 30-epoch runs. Execute inline; no additional agents are needed.

## Approved protocol
- Preserve all baseline runs and caches. Resume model, Adam, shuffle generator and Python/NumPy/Torch/CUDA RNG states from each epoch-30 last.pt.
- Keep seed42, batch200, FP32, unweighted CE, deterministic algorithms, DataLoader workers0 and serial GPU execution.
- Zero-based epoch e uses max(1e-6, 5e-4 * 0.2 ** (e / 30)); extending total epochs must not stretch the original schedule.
- At least 30 additional epochs. Significant Macro-F1 improvement means >=0.001 above the last qualifying best; cumulative subthreshold improvements can qualify. Start continuation patience at epoch30 with the baseline best score.
- Stop only when 20 epochs have passed without qualifying improvement AND relative improvement of the last10 mean validation losses over the preceding10 is <0.01. Negative improvement satisfies loss plateau.
- Maximum total200 epochs, including baseline30; a cap without the plateau conditions is NOT convergence.
- Best checkpoint selection continues strict maximum validation Macro-F1 with earlier-epoch tie handling, independently of the early-stop threshold.
- Train/val only; do not inspect or evaluate test data for this extension.

## Implementation and checks
1. Add tests/test_agentprint_convergence.py: pure stopping rules (minimum budget, both conditions, cumulative gains, exact boundaries, cap, finite metrics) and unchanged exponential learning rate. Run red.
2. Add agentprint/convergence.py: frozen policy, pure assessment, separate continuation runner/CLI. Reuse existing model, loaders, evaluation, atomic checkpoint/publication and RNG recovery. Existing public prepare/train/evaluate and formal configuration validation stay unchanged.
3. Add CPU integration tests with a real synthetic public baseline: parent hashes unchanged, optimizer/step continuity, best checkpoint retention, train/val artifacts, completed-run no-op, interrupted continuation matches uninterrupted tensors. Run red before implementing runner.
4. Run new tests, complete unittest suite, compileall and git diff --check. Commit only intended files.
5. Sync commit to the authorized GPU instance using a Git bundle. Run four continuations serially in a fresh data-disk output directory. Save runner/command logs and periodically verify real epoch progression.
6. Verify each stop reason by recomputing policy from epoch records, inspect selected checkpoints and finite metrics, preserve baseline hashes, select each task transform by validation only. Back up results locally. Report reached plateau separately from cap or failure.
