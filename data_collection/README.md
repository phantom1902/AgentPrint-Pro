# ANT collection sources

[English](README.md) | [简体中文](README.zh-CN.md) | [Benchmark entry point](../README.md)

These sources cover orchestration and utilities for collecting agent-episode traffic before constructing the released ANT splits. They accompany the classification suite; the classification suite does not invoke them.

## What is available

The import preserves 25 nonempty source/support files from `data_collection_code.zip`, with the small changes recorded in [IMPORT_REPORT.json](IMPORT_REPORT.json):

| Location | Delivered content |
| --- | --- |
| `scripts/batch_run_*.py` | CLI/web batch orchestration, parallel launchers, and provider-specific variants |
| `scripts/copy_*_sessions.py`, `scripts/codex_session_utils.py` | Session lookup and copying utilities |
| `tools/import_tasks.py` | Import task inputs from a separate task tree or ZIP without overwriting existing files |
| `tools/check_runtime.py` | Check manifest hashes, source references, modules, binaries, and mock availability |
| `tools/smoke_collection.py`, `tools/fixture_agent.py`, `examples/mock_smoke.yaml` | Infrastructure smoke-test driver and fixture; required runtime absent |
| `tasks/index.yaml` | Task catalog; not the task implementations or benchmark label mapping |
| `requirements*.txt` | Direct collection requirements and a broader task-environment snapshot |

**This is an incomplete source import, not a runnable collection release.** The supplied manifest declares 521 files: 480 declared paths are absent, 13 files are empty, and three nonempty files differ from their recorded hashes. A private note also has a filename-encoding mismatch. The missing-path count is measured against the declared paths, not a count of 480 missing executable programs.

Missing or empty components include `scripts/runner/`, `scripts/env/`, all eight `scripts/mock/mock_*_server.py` services, `setup_server.sh`, `label_pcap.py`, `visualize_episode.py`, production task wrappers, and task-specific helpers. The invalid/truncated `tasks/.zip`, empty files, private development notes, and macOS metadata were excluded. The original READMEs' claims that all runtime components were included do not describe the received archive. They have been replaced by these instructions and the integrated root README.

[IMPORT_REPORT.json](IMPORT_REPORT.json) records original archive hashes and availability. [MANIFEST.json](MANIFEST.json) records the actual delivered files and their current hashes; it does not assert runtime completeness. Session lookup was changed to use session workspace metadata instead of a hardcoded development-machine path. The smoke driver now rejects missing runtime components before creating fixtures or probing ports.

## Check the import

From the repository root, use Python 3.11 or later:

```bash
python data_collection/tools/check_runtime.py --source-only
```

For this import, the command **is expected to exit nonzero** because runner references and mock services are unresolved. `--source-only` skips installed dependency checks; it does not waive missing runtime components. Source syntax and delivered-file integrity were checked separately during integration. Do not treat installing dependencies as a fix for absent source files.

To enable collection, restore the missing nonempty runner, environment, mock, postprocessing, and task-helper sources from the original collection project, then re-audit identities, credentials, licenses, and file hashes. Complete task inputs and agent setup are also required. Only after that should the full runtime checker and infrastructure smoke test be run. No working installation or collection command is claimed for the current import.

## Dependency and task boundaries

The collection environment targets Linux and Python 3.11+, separately from the benchmark's main and NetMamba environments. `requirements.txt` lists direct collection dependencies. `requirements-task-environment.txt` is a broader source-environment snapshot, not a clean-machine-tested lockfile. The intended runtime also uses network namespaces, iproute2, iptables, tcpdump, bubblewrap, agent CLIs, and task-dependent tools such as Docker; web collection needs Playwright and browser/account setup. These components are not installed by importing this folder.

Complete task instances and external task datasets must be provided separately. The retained importer can copy a trusted task tree or ZIP into `tasks/`, preserving existing files. Importing task inputs alone does not restore the missing collection runtime. Upstream task materials retain their own licenses.

Provider variants read credentials from environment variables, including `MIMO_API_KEY` or `XIAOAI_API_KEY` with `ANTHROPIC_AUTH_TOKEN` as a fallback. Endpoints remain as in the source; configure them for the intended provider. Keep real credentials, browser profiles, session histories, and generated outputs outside the source release. Some batch launchers perform startup cleanup; inspect their options before using them on a shared host. No agents, collection tasks, capture commands, or cleanup routines were executed during this integration.

## Relationship to the benchmark

Raw output paths include `episodes/Agent/<engine>/Axxxx/`, with captures, session records, events, and workspaces depending on the runner. The released benchmark instead expects `Benchmark/data/ANT/<train|val|test>/<engine>/<family>/Axxxx/network.pcap`. The import does not provide a verified export or fixed-split construction pipeline. Preserve the published episode selection and split assignments rather than inventing a new split. Use the root [data and label definitions](../README.md#data-layout-and-labels) for classification.

Original project contributions follow [GPL-3.0-only](../LICENSE). The separate dataset grant is [CC-BY-4.0](../LICENSE-DATA), within the boundaries in [LICENSING.md](../LICENSING.md). External task inputs, packages, services, and model weights are not relicensed by this import.
