# Training history and recovery

Start a run with `python -m agentprint train --config CONFIG --device cpu`.
Each run gets a new directory named with task, transform, seed, UTC timestamp,
and random suffix. Existing directories are never reused for a new experiment.

Continue that same experiment with
`python -m agentprint train --config CONFIG --device cpu --resume RUN_DIRECTORY`.
The target is the configuration's original total epoch count. A completed run
can be resumed safely without adding epochs. A run interrupted before its first
complete epoch has no recovery point; start a new run in that case.

Recovery requires the same configuration except storage paths, and the same
label map, train/validation manifest IDs, preprocessing version and model
implementation version. This includes the original optimizer settings, seed,
batch size, epoch target and learning-rate schedule. Device and storage roots
may change. The explicit run directory remains the destination even if the
output root override changes. `runtime_history.jsonl` retains each accepted
launch's resolved configuration, device, package/platform versions, Git commit
and working-tree state. `resolved_config.yaml` shows the latest accepted config.
Identical CPU execution is covered by the synthetic recovery test; changing
devices carries no bitwise reproducibility guarantee.

`last.pt` commits a complete epoch's model, Adam state, epoch, global training
step, best score/epoch/snapshot pointer, and Python/NumPy/PyTorch/CUDA RNG states,
including the explicit DataLoader shuffle generator. Validation Macro-F1 uses
strict improvement, preserving the earlier epoch on ties. Immutable best
snapshots live under `checkpoints/`. The convenient `best.pt`, `summary.json`,
`per_class.csv` and `confusion_matrix.csv` represent that committed best; resume
repairs these views from the pointer stored in `last.pt`.

Epoch metrics are appended to `metrics.jsonl`, including train/validation loss,
validation accuracy and primary Macro-F1, applicable fixed-47 Macro-F1,
learning rate, elapsed time, sample counts and global step. Every 50 global
training batches, `training_steps.jsonl` records the batch's training loss.
Both streams are available in TensorBoard beneath `tensorboard/RUN_ID/`, with
separate `epochs` and `steps` streams to preserve their different step units.
Point TensorBoard's log directory at the run's `tensorboard` directory.

The atomic replacement of `last.pt` is the commit point. A crash may leave newer
metrics and candidate snapshots. Resume moves metric rows beyond the committed
epoch/step, duplicate rows and partial trailing rows to a unique `archive/`
directory before continuing. Original TensorBoard events remain on disk; restart
markers purge their uncommitted tail from the effective display. A best candidate
is written before the commit, so interruption between writes cannot destroy the
previously committed best. JSON/CSV views can be repaired by another resume.

Execution uses FP32, unweighted cross entropy, deterministic algorithms,
`num_workers=0`, disabled TF32 and disabled cuDNN benchmarking. OOM, unsupported
deterministic execution and nonfinite loss stop execution. The command never
changes batch size, precision or experiment settings automatically.
