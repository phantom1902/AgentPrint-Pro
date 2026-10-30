# Validation-plateau continuation

This is a separately recorded extension experiment, not the fixed 30-epoch formal protocol. The original prepare/train/evaluate commands, configuration rules and baseline artifacts remain unchanged.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python -m agentprint.convergence --parent-run /data/formal-run   --output-root /data/convergence-experiment --device cuda:0
```

The parent must have 30 committed epochs. The destination is a new sibling experiment with its own lineage, checkpoints and logs; rerun the same command to recover. Existing run names are retained within the new output root. Model, optimizer and all saved RNG states resume from last.pt, not best.pt. The baseline best remains a candidate for final selection.

## User-approved stopping rule

At least 30 additional epochs are required. The Macro-F1 significance anchor starts at the baseline best; a >=0.001 absolute improvement updates the anchor and resets patience. Smaller gains can accumulate until this threshold is reached. The checkpoint best is independently updated on ANY strictly higher Macro-F1, with earlier-epoch tie handling.

After the minimum budget, stop only when BOTH apply:
- 20 epochs without a qualifying Macro-F1 gain.
- Relative validation loss improvement <1% between the previous10 and recent10 epoch means, calculated (previous-recent)/previous. Worsening loss satisfies this plateau condition. With zero previous loss, nonnegative losses cannot improve and relative improvement is reported as0.

The hard cap is200 total epochs including baseline30. Hitting the cap without both plateau conditions returns converged=false and reason=epoch_cap_reached. Validation plateau is an operational stopping criterion, not proof of mathematical convergence or optimal generalization.

Learning rate is max(1e-6,5e-4*0.2**(zero_based_epoch/30)); changing the cap does not stretch the original schedule. Seed42, FP32, batch200, deterministic settings, Adam/weight decay and unweighted CE remain unchanged. Runs execute serially. The extension reads train/val only, never test.

## Artifacts and recovery

lineage.json hashes parent last.pt, best snapshot and metrics, and locks the policy. last.pt atomically commits each epoch, optimizer, RNG and policy assessment; logs ahead of this point are archived on recovery. convergence.json records both plateau tests, total/additional epochs and the stop reason. protocol.json marks formal_experiment=false and experiment_kind=convergence_extension. metrics.jsonl includes the original30 rows plus continuation rows; TensorBoard contains continuation epochs. validation-best contains the best validation metrics/per-class/confusion matrix, and best.pt/last.pt remain reloadable.

Tests: python -m unittest discover -s tests -p 'test_agentprint_convergence*.py' -v.
