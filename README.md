# ANT: Traffic Collection and Risk / Scenario Benchmark

[English](README.md) | [简体中文](README.zh-CN.md)

This repository connects two stages of the ANT workflow: **collecting agent traffic** and **evaluating risk and scenario classifiers on the released fixed splits**. The experiment suite provides preprocessing, training, prediction, episode aggregation, and metric reporting for **binary risk classification and five-class scenario classification across 13 baselines**. It uses `Benchmark/data/ANT/{train,val,test}` as its only task data input. The models are the locally adapted implementations used in the project; see [SOURCES.md](SOURCES.md).

The experiment component remains a minimal, independently usable suite. It retains the original model modules and excludes primitive classification, ablations, hyperparameter search, historical results, feature caches, data, and weights. The newly included [collection sources](data_collection/README.md) document the upstream collection stage, but **the supplied collection archive is incomplete and cannot run collection end to end**. Collection is not required to train or evaluate on the released data. Run benchmark commands from this repository's root.

Original project code is **GPL-3.0-only**; the authors' rights in the separately released ANT data and original annotations are **CC-BY-4.0**. Third-party implementations retain their own terms, with unresolved permissions explicitly listed in [licensing scope](LICENSING.md) and [third-party licenses](THIRD_PARTY_LICENSES.md).

## Release contents

| Component | Location | Purpose |
| --- | --- | --- |
| Collection sources and import audit | `data_collection/` | Batch orchestration, session export, task index, and diagnostic tools; incomplete runtime |
| Released data and fixed splits | [Hugging Face dataset](https://huggingface.co/datasets/benzenenene/ANT), placed under `Benchmark/data/ANT/{train,val,test}` | Read each episode's `network.pcap` and directory labels |
| Label definitions | `configs/scenario.json`, `suite.py::task_label` | Define the risk and scenario labels |
| Models and PCAP adapters | `Benchmark/`, `engines.py` | Build model inputs from released PCAPs and train the models |
| Main-experiment configurations | `configs/{risk,scenario}/`, `Benchmark/all_default_configs/` | Specify model parameters and training budgets |
| Evaluation and aggregation | `suite.py`, `Benchmark/evaluation.py` | Select aggregation on validation data and report test results |
| Run entry points | `suite.py`, `run_all.py` | Run one model or all 26 model–task combinations |
| Dependencies, provenance, and validation records | `requirements-*.txt`, `SOURCES.md`, `VALIDATION.md` | Document installation, implementation sources, and verification scope |
| Three initial checkpoints or instructions for obtaining them | `WEIGHTS.json`, `verify_weights.py` | Supply separately; weights are not included in the code package |

**The suite supports retraining from the released data; it does not guarantee an exact reproduction of the paper's tables.** Historical checkpoints, resumed training states, and some configuration snapshots are incomplete. Known differences are documented under [Relationship to historical results](#relationship-to-historical-results).

## Dataset release

The ANT dataset is hosted on Hugging Face: [benzenenene/ANT](https://huggingface.co/datasets/benzenenene/ANT).

The publishing account was newly registered specifically for this submission. It is not linked to any author identity and has no prior contribution history unrelated to this submission.

Obtain the released `train`, `val`, and `test` splits from the dataset repository and place them under `Benchmark/data/ANT/`, preserving the hierarchy below. You can also keep the data elsewhere and pass that ANT directory to `suite.py` with `--data`. Use the released fixed splits without repartitioning them. The data license and its scope are described in [LICENSING.md](LICENSING.md).

## Data layout and labels

Preserve the following hierarchy rather than placing all PCAPs directly under each split:

```text
Benchmark/data/ANT/
├── train/<engine>/<family>/Axxxx/network.pcap
├── val/<engine>/<family>/Axxxx/network.pcap
└── test/<engine>/<family>/Axxxx/network.pcap
```

- **Risk:** `attack_code_generation` and `attack_execution` are labeled `malicious`; the other included families are labeled `benign`.
- **Scenario:** the first two digits after `A` in `Axxxx` identify the task. Tasks map to `artifact_processing`, `general_knowledge`, `scientific_evidence`, `business_services`, or `security_testing`. The exact task mapping is in `configs/scenario.json`.
- The suite does not read session logs, prompts, private annotations, or flow-layer annotations.
- It uses the released fixed splits without randomly splitting the data again. Duplicate `engine/episode` identities and duplicate resolved PCAP paths, including those shared across splits, are rejected. Every split must contain all classes. This check does not deduplicate packet contents.
- The current data directory contains **2,703 training**, **202 validation**, and **209 test** episodes. If the data release changes, use the generated `dataset_counts.json` as the count for that run.

## From collection to the released benchmark

There are two entry paths:

1. **Run the main experiments:** obtain the released fixed splits, set up the model environments below, provide required initial weights, and run `suite.py`. No collection scripts or agent accounts are needed.
2. **Inspect or extend collection:** use [data_collection/README.md](data_collection/README.md) for the delivered sources, missing components, dependency boundaries, and diagnostics. Its Python requirements are separate from the model environments.

The imported archive contains batch launchers and session utilities, but its original 521-entry manifest references **480 absent files and 13 empty scripts**. The runner, network environment helpers, eight mock servers, and task helpers are not a complete runnable dependency set. Empty scripts, private development notes, macOS metadata, and an invalid nested ZIP were excluded. [IMPORT_REPORT.json](data_collection/IMPORT_REPORT.json) records this gap; the collection manifest describes only files actually delivered here.

Collection launchers write raw episodes under paths such as `data_collection/episodes/Agent/<engine>/Axxxx/`. These are not automatically the released train/validation/test splits. Raw outputs can include sessions, prompts, logs, and workspaces; classification reads only the released `network.pcap` files and directory labels. A release must supply the original episode selection, family labels, and fixed split assignments to produce the hierarchy above. No export/split-construction tool is provided in this import, and a new random split would define a different experiment. The broad collection task index is not a replacement for the classification mapping in `configs/scenario.json`.

## Environments

Use two Python environments on Linux. The versions below were observed in the local environments that passed the small-sample checks. Installation has not been reproduced on a clean machine; GPU drivers and CUDA build tools must be compatible with the selected PyTorch version.

The main environment uses Python 3.12 and supports the 12 models other than NetMamba:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-main.txt
```

TrafficFormer's PCAP conversion also requires the system `tshark` executable on PATH. A CPU can run FlowLens and small-sample checks; use CUDA for the full deep-learning experiments.

NetMamba uses a separate environment with Python 3.9, PyTorch 2.0.1 / CUDA 11.7, and torchvision 0.15.2. It requires a C++ toolchain, the CUDA toolkit including `nvcc`, and two CUDA extensions. In the following commands, `python` refers to this separate environment:

```bash
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements-netmamba.txt
python -m pip install --no-build-isolation --no-deps causal-conv1d==1.1.0
MAMBA_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps ./Benchmark/2024-ICNP-NetMamba/mamba-1p1p1
```

Keep the NetMamba and mamba source code included in the suite. The package does not include machine-specific `.so` files. The checks used an existing compatible local extension, so the build steps above have not been revalidated. If `selective_scan_cuda` is missing, install the extension before running NetMamba; a CPU random forest is not a substitute for this model.

## Initial weights

By default, create a `weights/` directory at the suite root and provide these three files:

```text
weights/yatc-pretrained.pth
weights/netmamba-pretrained.pth
weights/trafficformer-pretrained.bin
```

[WEIGHTS.json](WEIGHTS.json) records the source paths, SHA-256 hashes, and known upstream download locations for the files used to initialize the experiments. These filenames are standardized names for the suite; they do not imply that all three files are unmodified official pretrained checkpoints.

```bash
python verify_weights.py --weights ./weights
```

For TrafficFormer, the actual starting file was the local `finetuned_model.bin`. Its SHA-256 differs from that of the local official pretrained file, `nomoe_bertflow_pre-trained_model.bin-120000`. Their equivalence has not been established. To match the historical starting conditions, supply the actual file separately and document how it was produced; using the official pretrained file instead changes those conditions. The other 10 models require no external checkpoints. TFusion learns its encoder from benign flows in the training split.

## Running experiments

First check the data and generate manifests for both tasks. The data may reside outside the suite; point `--data` to the actual ANT directory.

```bash
python suite.py check --data ./Benchmark/data/ANT
python suite.py prepare --data ./Benchmark/data/ANT --output ./outputs
```

Examples for individual models:

```bash
python suite.py run --task risk --model FlowLens --device cpu
python suite.py run --task scenario --model DF --device cuda:0
python suite.py run --task scenario --model YaTC --device cuda:0 --weights ./weights
```

To run all 26 model–task combinations, provide the Python executable for each environment. Runs execute sequentially to avoid competing for GPU memory. Completed runs are skipped, and execution stops on failure.

```bash
python run_all.py \
  --python-main /path/to/main-env/bin/python \
  --python-netmamba /path/to/netmamba-env/bin/python \
  --data ./Benchmark/data/ANT --weights ./weights \
  --output ./outputs --device cuda:0
```

Supported model names are `FlowLens YaTC NetMamba TFusion TrafficFormer DF AWF TikTok VarCNN NetCLR TMWF ARES BAPM`. Select a subset with, for example, `run_all.py --models DF AWF --tasks risk`. For an individual NetMamba run, invoke `suite.py` with the Python executable from its own environment.

Add `--prepare-only` to build inputs without training; pretrained weights are not required for this step. Remove that option and keep the other arguments to train afterward. For a small-sample check, explicitly add `--epochs 1 --batch-size 8`. These overrides are not the main-experiment configuration. For TFusion, `--epochs` controls encoder training; it has no effect on FlowLens.

The single-model entry point does not overwrite completed or failed runs in the same output directory. Use a new `--output` when rerunning an experiment or retrying after a failure. Keep results from different datasets, parameters, and model versions separate. Skipping completed runs in `run_all.py` is intended only for continuing the same experiment batch.

```bash
python suite.py summarize --output ./outputs
```

This produces `outputs/results.csv` with the status and available metrics for all 26 combinations. Unrun entries remain empty; historical results are not inserted.

## Evaluation protocol and outputs

FlowLens, YaTC, NetMamba, TFusion, and TrafficFormer first produce flow-level predictions. The suite selects one of `mean_soft`, `majority_vote`, `confidence_weighted_mean`, and `max_confidence` using only **validation episode accuracy**. Ties follow the order listed here. The selected rule is then applied to test predictions. Entropy-based confidence is `1 − H(p)/log(C)`; if every confidence weight is zero, aggregation falls back to the mean.

The eight WF models take episodes directly as input and do not use flow aggregation. Their inputs include all parseable protocols, matching this project's all-protocol main-experiment entry point. NetCLR uses the supervised classification procedure in the local shared WF runner; the suite does not add the original method's contrastive pretraining stage.

Main outputs for each run:

```text
outputs/<risk|scenario>/<model>/
├── suite_status.json           # Arguments, Python version, and run status
├── resolved_config.json        # Resolved config for the four integrated models
├── native/ or algorithm_output/ # Model checkpoints and native outputs
├── val/, test/                 # Flow predictions and their episode mappings
├── aggregation.json            # Validation-based rule selection for flow models
├── episode/test/               # Final episode predictions and evaluation_metrics.json
└── coverage.json               # Evaluated counts and missing episode IDs
```

Preprocessing can filter out short or unparseable flows. Metrics are computed over episodes for which predictions were produced. Check `coverage.json` when reporting results, and report subset coverage accurately. Global counts are in `dataset_counts.json`; model-specific preprocessing details remain in each run directory.

Random seeds follow the existing implementations: the controller, WF models, and YaTC default to 42; native NetMamba training uses 0; TrafficFormer uses 7 for training and 10 for conversion; TFusion uses 777 for contrastive batches and 42 for its RF; FlowLens sets `random_state` to 42 in its configuration. `suite.py --seed` overrides only the controller, WF, YaTC, and TFusion RF seeds, not all native settings. CUDA operations are not guaranteed to be bitwise deterministic.

## Relationship to historical results

1. `configs/origins.json` records source descriptions and hashes for the frozen configurations. Internal run paths are withheld in the anonymous release. The risk configurations for TFusion and TrafficFormer were reconstructed from best-result parameters and local defaults; unrecorded parameters cannot all be confirmed to match the historical runs.
2. Historical TFusion scenario runs reused an existing benign-flow encoder. This suite retrains that encoder from the released training split using its recorded parameters, then trains an RF with 20 trees and a maximum depth of 50. The risk task uses its own tuned parameters. A checkpoint from another task is not substituted for the historical encoder.
3. The local `--resume` path in the historical NetMamba scenario configuration has been removed. The suite starts a new fine-tuning run from the supplied initial weights while retaining the recorded explicit learning rate and other saved parameters.
4. NetMamba and TrafficFormer retain their native stopping mechanism with a validation threshold of 0.99, so the epoch setting is an upper limit. Some native logs include test metrics recorded after the stopping decision. Test results do not participate in this suite's aggregation-rule selection.
5. YaTC preserves the actual adapter setting `zero_ip_addresses=false`; NetMamba uses `remove_ip=true`. The suite does not silently change preprocessing to match a uniform description in the paper.
6. Runs use the current released directory's fixed splits, totaling 3,114 episodes. The suite neither fills in missing episodes nor substitutes historical table denominators for the current manifest.

See [VALIDATION.md](VALIDATION.md) for completed checks and their limits, and [SOURCES.md](SOURCES.md) for provenance and changes made when extracting the suite. These supplementary records are currently in Chinese.

See [ANONYMITY.md](ANONYMITY.md) for the anonymous release audit and the checks to repeat before sharing an artifact with reviewers.
