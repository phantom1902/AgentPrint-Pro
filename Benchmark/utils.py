from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
COMMAND_HISTORY = REPO / "command_history.md"
SUPPORTED_ALGORITHMS = {"FlowLens", "Kitsune", "TFusion", "TrafficFormer", "NetMamba"}


class BenchmarkError(RuntimeError):
    pass


class UnsupportedAlgorithmMode(BenchmarkError):
    pass


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO / candidate


def rel(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(candidate)


def read_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def split_multi_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                paths.append(resolve(item))
    if not paths:
        raise BenchmarkError("At least one dataset path is required")
    return paths


def default_config_path(algorithm: str) -> Path:
    return BENCHMARK_DIR / "default_configs" / f"{algorithm}.yaml"


def load_benchmark_config(
    algorithm: str,
    config_path: str | None,
    hyperparameters_path: str | Path | None = None,
) -> dict[str, Any]:
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise BenchmarkError(f"Unsupported algorithm: {algorithm}")

    default_path = default_config_path(algorithm)
    if not default_path.exists():
        raise BenchmarkError(f"Default config is missing for {algorithm}: {default_path}")
    config = read_config(default_path)

    if config_path is not None:
        supplied_path = resolve(config_path)
        supplied = read_config(supplied_path)
        declared = supplied.get("algorithm")
        if declared is not None and str(declared) != algorithm:
            raise BenchmarkError(f"Config algorithm={declared!r} does not match --algorithm {algorithm!r}")
        if "algorithms" in supplied and algorithm not in supplied["algorithms"]:
            raise BenchmarkError(f"Config does not contain settings for --algorithm {algorithm!r}")

        if "algorithms" in supplied:
            patch = {
                "runtime": supplied.get("runtime", {}),
                "data": supplied.get("data", supplied.get("dataset", {})),
                "algorithm_config": supplied["algorithms"].get(algorithm, {}),
            }
        else:
            patch = supplied
        config = deep_update(config, patch)
        config["config_path"] = rel(supplied_path)

    from .hyperparameters import apply_algorithm_hyperparameters

    return apply_algorithm_hyperparameters(config, algorithm, hyperparameters_path)


def runtime_python(config: dict[str, Any]) -> str:
    configured = config.get("runtime", {}).get("python")
    if configured and Path(configured).exists():
        return str(configured)
    return sys.executable


def runtime_env(config: dict[str, Any]) -> dict[str, str]:
    cuda = config.get("runtime", {}).get("cuda_visible_devices")
    return {"CUDA_VISIBLE_DEVICES": str(cuda)} if cuda is not None else {}


def apply_runtime_overrides(
    config: dict[str, Any],
    cuda_visible_devices: str | None = None,
    require_cuda: bool = False,
    runtime_python_override: str | None = None,
) -> None:
    runtime = config.setdefault("runtime", {})
    if runtime_python_override is not None:
        if runtime_python_override in {"current", "same", "sys.executable"}:
            runtime["python"] = sys.executable
        elif runtime_python_override != "config":
            runtime["python"] = runtime_python_override
    if cuda_visible_devices is not None:
        runtime["cuda_visible_devices"] = str(cuda_visible_devices)

    for key, value in runtime_env(config).items():
        os.environ[key] = value

    if require_cuda:
        try:
            import torch
        except ImportError as exc:
            raise BenchmarkError("--require-cuda was set, but torch is not importable in the controller process") from exc
        if not torch.cuda.is_available():
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            raise BenchmarkError(f"--require-cuda was set, but torch.cuda.is_available() is false; CUDA_VISIBLE_DEVICES={visible!r}")


def append_history(command: list[str]) -> None:
    COMMAND_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with COMMAND_HISTORY.open("a", encoding="utf-8") as handle:
        handle.write("- `" + " ".join(command) + "`\n")


def run_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    config: dict[str, Any],
    timeout: int | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Each run already records the command in log_path; keep the code tree clean.
    env = {**os.environ, **runtime_env(config)}
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now()}] $ {' '.join(command)}\n")
        if "CUDA_VISIBLE_DEVICES" in env:
            log.write(f"[{now()}] CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
        log.flush()
        proc = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout)
        log.write(f"[{now()}] exit={proc.returncode}\n")
        return int(proc.returncode)
