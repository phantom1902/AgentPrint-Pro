from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .utils import BENCHMARK_DIR, BenchmarkError, deep_update, rel, resolve


DEFAULT_HYPERPARAMETERS_PATH = BENCHMARK_DIR / "all_default_configs"

ALGORITHM_BASELINES = {
    "FlowLens": "2021-NDSS-FlowLens",
    "Kitsune": "2018-NDSS-Kitsune",
    "NetMamba": "2024-ICNP-NetMamba",
    "TFusion": "2025-CCS-TFusion",
    "TrafficFormer": "2025-SP-TrafficFormer",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_file(directory: Path, baseline: str) -> Path:
    return directory / f"{baseline}.json"


def _baseline_payload(payload: dict[str, Any], baseline: str, source: Path) -> dict[str, Any]:
    if "baselines" in payload:
        try:
            return deepcopy(payload["baselines"][baseline])
        except KeyError as exc:
            raise BenchmarkError(f"No performance hyperparameters for baseline {baseline!r} in {source}") from exc
    if payload.get("baseline") != baseline:
        raise BenchmarkError(
            f"Hyperparameter file {source} is for baseline {payload.get('baseline')!r}, not {baseline!r}"
        )
    try:
        return deepcopy(payload["performance_hyperparameters"])
    except KeyError as exc:
        raise BenchmarkError(f"Hyperparameter file {source} has no performance_hyperparameters object") from exc


def load_performance_hyperparameters(path: str | Path | None = None) -> dict[str, Any]:
    hparams_path = resolve(path) if path else DEFAULT_HYPERPARAMETERS_PATH
    if not hparams_path.exists():
        raise BenchmarkError(f"Baseline performance hyperparameter path is missing: {hparams_path}")
    if hparams_path.is_file():
        payload = _read_json(hparams_path)
        if "baselines" in payload:
            return payload
        baseline = payload.get("baseline")
        if baseline is None:
            raise BenchmarkError(f"Hyperparameter file {hparams_path} has neither baselines nor baseline")
        return {
            "schema_version": payload.get("schema_version"),
            "scope": payload.get("scope"),
            "common_ant_defaults": payload.get("common_ant_defaults", {}),
            "baselines": {baseline: payload["performance_hyperparameters"]},
        }
    baselines: dict[str, Any] = {}
    schema_version = None
    scope = None
    common_ant_defaults: dict[str, Any] = {}
    for item in sorted(hparams_path.glob("*.json")):
        payload = _read_json(item)
        baseline = payload.get("baseline")
        if baseline is None:
            continue
        baselines[str(baseline)] = payload.get("performance_hyperparameters", {})
        schema_version = schema_version or payload.get("schema_version")
        scope = scope or payload.get("scope")
        if not common_ant_defaults:
            common_ant_defaults = payload.get("common_ant_defaults", {})
    if not baselines:
        raise BenchmarkError(f"No per-baseline hyperparameter JSON files found in {hparams_path}")
    return {
        "schema_version": schema_version,
        "scope": scope,
        "common_ant_defaults": common_ant_defaults,
        "baselines": baselines,
    }


def baseline_hyperparameters(baseline: str, path: str | Path | None = None) -> dict[str, Any]:
    hparams_path = resolve(path) if path else DEFAULT_HYPERPARAMETERS_PATH
    if hparams_path.is_dir():
        source = _baseline_file(hparams_path, baseline)
        if not source.exists():
            raise BenchmarkError(f"Baseline hyperparameter file is missing: {source}")
        return _baseline_payload(_read_json(source), baseline, source)
    data = load_performance_hyperparameters(hparams_path)
    try:
        return deepcopy(data["baselines"][baseline])
    except KeyError as exc:
        raise BenchmarkError(f"No performance hyperparameters for baseline {baseline!r}") from exc


def hyperparameter_source_path(baseline: str, path: str | Path | None = None) -> Path:
    hparams_path = resolve(path) if path else DEFAULT_HYPERPARAMETERS_PATH
    return _baseline_file(hparams_path, baseline) if hparams_path.is_dir() else hparams_path


def algorithm_hyperparameters(algorithm: str, path: str | Path | None = None) -> dict[str, Any]:
    try:
        baseline = ALGORITHM_BASELINES[algorithm]
    except KeyError as exc:
        raise BenchmarkError(f"No baseline mapping for algorithm {algorithm!r}") from exc
    return baseline_hyperparameters(baseline, path)


def _flowlens_patch(hp: dict[str, Any]) -> dict[str, Any]:
    adapter = hp.get("adapter_hyperparameters", {})
    training = hp.get("model_training_hyperparameters", {})
    evaluation = hp.get("evaluation_hyperparameters", {})
    split = hp.get("split_hyperparameters", {})
    flowlens_keys = [
        "packet_size_bin_width",
        "ipt_bin_width_seconds",
        "timegap_seconds",
        "flowgap_seconds",
        "packet_length_source",
        "skip_non_tcp_udp",
        "skip_zero_payload_packets",
        "five_tuple_timeout_seconds",
    ]
    flowlens = {key: adapter[key] for key in flowlens_keys if key in adapter}
    if "five_tuple_mode" in adapter:
        flowlens["five_tuple_mode"] = str(adapter["five_tuple_mode"]).split()[0]
    classifier = {
        key: training[key]
        for key in ["random_state", "n_estimators", "max_depth", "class_weight"]
        if key in training
    }
    labels = {}
    if "superflow_label_policy" in evaluation:
        labels["superflow_label_policy"] = evaluation["superflow_label_policy"]
    return {
        "algorithm_config": {
            "classifier": {
                "flowlens": flowlens,
                "classifier": classifier,
                "labels": labels,
            }
        },
        "data": {
            "split_random_state": split.get("random_state"),
        },
    }


def _kitsune_patch(hp: dict[str, Any]) -> dict[str, Any]:
    adapter = hp.get("adapter_hyperparameters", {})
    training = hp.get("model_training_hyperparameters", {})
    evaluation = hp.get("evaluation_hyperparameters", {})
    return {
        "algorithm_config": {
            "input": {
                "max_packets": adapter.get("max_packets"),
                "max_packets_per_source": adapter.get("max_packets_per_source"),
            },
            "model": {
                "max_features_per_cluster": training.get("max_features_per_cluster"),
                "compression_rate": training.get("compression_rate"),
                "dropout_rate": training.get("dropout_rate"),
            },
            "training": {
                "max_epoch": training.get("epochs"),
                "batch_size": training.get("batch_size"),
                "learning_rate": training.get("learning_rate"),
                "weight_decay": training.get("weight_decay"),
                "shuffle_batches": training.get("shuffle_batches"),
            },
            "threshold_search": {
                "selection_source": evaluation.get("threshold_selection_source"),
                "selection_metric": evaluation.get("threshold_selection_metric"),
                "max_candidates": evaluation.get("max_threshold_candidates"),
            },
        }
    }


def _netmamba_patch(hp: dict[str, Any]) -> dict[str, Any]:
    adapter = hp.get("adapter_hyperparameters", {})
    training = hp.get("model_training_hyperparameters", {})
    evaluation = hp.get("evaluation_hyperparameters", {})
    return {
        "algorithm_config": {
            "conversion": {
                "input_representation": adapter.get("input_representation"),
                "five_tuple_mode": adapter.get("five_tuple_mode"),
                "five_tuple_timeout_seconds": adapter.get("five_tuple_timeout_seconds"),
                "remove_ip": adapter.get("remove_ip"),
                "keep_payload": adapter.get("keep_payload"),
                "if_augment": adapter.get("if_augment"),
                "max_packets_per_source": adapter.get("max_packets_per_source"),
                "max_flows_per_source": adapter.get("max_flows_per_source"),
            },
            "finetune": {
                "blr": training.get("base_learning_rate"),
                "weight_decay": training.get("weight_decay"),
                "drop_path": training.get("drop_path"),
                "max_epoch": training.get("max_epoch"),
                "batch_size": training.get("batch_size"),
                "model": training.get("model"),
                "num_workers": training.get("num_workers"),
                "no_amp": training.get("no_amp"),
            },
            "early_stop": {
                "validation_threshold": evaluation.get("early_stop_validation_threshold"),
                "test_threshold": evaluation.get("early_stop_test_threshold"),
                "metrics": evaluation.get("early_stop_metrics"),
            },
        }
    }


def _tfusion_patch(hp: dict[str, Any]) -> dict[str, Any]:
    adapter = hp.get("adapter_hyperparameters", {})
    training = hp.get("model_training_hyperparameters", {}).get("stage1_contrastive", {})
    evaluation = hp.get("evaluation_hyperparameters", {})
    detector_params: dict[str, Any] = {}
    detector_name = evaluation.get("detection_model")
    if detector_name == "kmeans" and "kmeans_detector" in evaluation:
        detector_params.update(evaluation["kmeans_detector"])
    elif detector_name in {"rforest", "rf", "random_forest"} and "alternative_rf_detector" in evaluation:
        detector_params.update(evaluation["alternative_rf_detector"])
    elif detector_name == "knn" and "knn_detector" in evaluation:
        detector_params.update(evaluation["knn_detector"])
    return {
        "algorithm_config": {
            "conversion": {
                "five_tuple_mode": adapter.get("five_tuple_mode"),
                "five_tuple_timeout_seconds": adapter.get("five_tuple_timeout_seconds"),
                "skip_zero_payload_packets": adapter.get("skip_zero_payload_packets"),
            },
            "infer_batch_size": evaluation.get("infer_batch_size"),
            "detection_model": detector_name,
            "detector_params": detector_params,
            "data_construct_param": {
                "segment_len": adapter.get("segment_len", training.get("segment_len")),
            },
            "training": {
                "max_epoch": training.get("max_epoch"),
                "learning_rate": training.get("learning_rate"),
                "weight_decay": training.get("weight_decay"),
                "contrast_batch_size": training.get("contrast_batch_size"),
                "seed": training.get("seed"),
            },
        }
    }


def _trafficformer_patch(hp: dict[str, Any]) -> dict[str, Any]:
    adapter = hp.get("adapter_hyperparameters", {})
    training = hp.get("model_training_hyperparameters", {})
    evaluation = hp.get("evaluation_hyperparameters", {})
    return {
        "algorithm_config": {
            "conversion": {
                "input_representation": adapter.get("input_representation"),
                "payload_length": adapter.get("payload_length"),
                "payload_packet": adapter.get("payload_packet"),
                "start_index": adapter.get("start_index"),
                "five_tuple_mode": adapter.get("five_tuple_mode"),
                "five_tuple_timeout_seconds": adapter.get("five_tuple_timeout_seconds"),
                "random_seed": adapter.get("random_seed"),
                "max_rows": adapter.get("max_rows"),
            },
            "finetune": {
                "max_epoch": training.get("max_epoch"),
                "batch_size": training.get("batch_size"),
                "embedding": training.get("embedding"),
                "encoder": training.get("encoder"),
                "mask": training.get("mask"),
                "seq_length": training.get("seq_length"),
                "learning_rate": training.get("learning_rate"),
                "dropout": training.get("dropout"),
                "warmup": training.get("warmup"),
            },
            "early_stop": {
                "validation_threshold": evaluation.get("early_stop_validation_threshold"),
                "test_threshold": evaluation.get("early_stop_test_threshold"),
                "metrics": evaluation.get("early_stop_metrics"),
            },
        }
    }


PATCH_BUILDERS = {
    "FlowLens": _flowlens_patch,
    "Kitsune": _kitsune_patch,
    "NetMamba": _netmamba_patch,
    "TFusion": _tfusion_patch,
    "TrafficFormer": _trafficformer_patch,
}


def _drop_none(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _drop_none(value) for key, value in obj.items() if value is not None}
    if isinstance(obj, list):
        return [_drop_none(value) for value in obj]
    return obj


def apply_algorithm_hyperparameters(
    config: dict[str, Any],
    algorithm: str,
    path: str | Path | None = None,
) -> dict[str, Any]:
    try:
        builder = PATCH_BUILDERS[algorithm]
    except KeyError as exc:
        raise BenchmarkError(f"No performance hyperparameter mapping for algorithm {algorithm!r}") from exc
    hp = algorithm_hyperparameters(algorithm, path)
    patch = _drop_none(builder(hp))
    deep_update(config, patch)
    config.setdefault("performance_hyperparameters", {})
    config["performance_hyperparameters"].update(
        {
            "source": rel(hyperparameter_source_path(ALGORITHM_BASELINES[algorithm], path)),
            "baseline": ALGORITHM_BASELINES[algorithm],
        }
    )
    return config


def wflib_hyperparameters(path: str | Path | None = None) -> dict[str, Any]:
    return baseline_hyperparameters("2024_CCS_WFlib_Early_Stage_WF", path)
