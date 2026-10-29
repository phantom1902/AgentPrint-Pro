from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import struct
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .data import PacketDataset, PreparedData
from .utils import BenchmarkError, UnsupportedAlgorithmMode, rel, resolve, run_command, runtime_python, write_json


@dataclass
class PredictionFiles:
    hard_predictions: Path
    test_labels: Path
    soft_predictions: Path | None
    label_mapping: Path | None
    output_dir: Path
    metadata: dict[str, Any]


def append_extra_args(command: list[str], extra_args: object) -> list[str]:
    if not extra_args:
        return command
    if isinstance(extra_args, list):
        return command + [str(item) for item in extra_args]
    if isinstance(extra_args, dict):
        extended = list(command)
        for key, value in extra_args.items():
            flag = str(key) if str(key).startswith("--") else f"--{key}"
            if isinstance(value, bool):
                if value:
                    extended.append(flag)
            elif value is None:
                continue
            elif isinstance(value, list):
                extended.append(flag)
                extended.extend(str(item) for item in value)
            else:
                extended.extend([flag, str(value)])
        return extended
    raise TypeError(f"extra_args must be a list or dict, got {type(extra_args)!r}")


def run_trafficformer(prepared: PreparedData, config: dict[str, Any], run_dir: Path) -> PredictionFiles:
    alg = config["algorithm_config"]
    ft = alg["finetune"]
    epochs_num = int(ft.get("max_epoch", ft.get("epochs_num", 4)))
    earlystop = int(ft.get("earlystop", epochs_num))
    eval_checkpoint = alg.get("eval_checkpoint_path")
    weight = resolve(eval_checkpoint) if eval_checkpoint else resolve(alg["pretrained_model_path"])
    if not weight.exists():
        raise FileNotFoundError(f"TrafficFormer model not found: {weight}")
    out_dir = run_dir / "algorithm_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    hard = out_dir / "hard_predictions.txt"
    soft = out_dir / "soft_predictions.npy"
    labels = out_dir / "test_labels.txt"
    dev_hard = out_dir / "validation_hard_predictions.txt"
    dev_soft = out_dir / "validation_soft_predictions.npy"
    dev_labels = out_dir / "validation_labels.txt"
    cmd = [
        runtime_python(config),
        "fine-tuning/run_classifier.py",
        "--vocab_path",
        ft["vocab_path"],
        "--test_path",
        str(prepared.test_path.resolve()),
        "--pretrained_model_path",
        str(weight.resolve()),
        "--output_model_path",
        str((out_dir / "trafficformer_model.bin").resolve()),
        "--epochs_num",
        str(epochs_num),
        "--earlystop",
        str(min(earlystop, epochs_num)),
        "--batch_size",
        str(ft["batch_size"]),
        "--embedding",
        ft["embedding"],
        "--encoder",
        ft["encoder"],
        "--mask",
        ft["mask"],
        "--seq_length",
        str(ft["seq_length"]),
        "--learning_rate",
        str(ft["learning_rate"]),
        "--dropout",
        str(ft.get("dropout", 0.5)),
        "--warmup",
        str(ft.get("warmup", 0.1)),
        "--prediction_path",
        str(hard.resolve()),
        "--soft_prediction_path",
        str(soft.resolve()),
        "--label_path",
        str(labels.resolve()),
    ]
    has_validation = prepared.dev_path is not None and prepared.dev_path.exists()
    if not eval_checkpoint:
        cmd.extend(
            [
                "--train_path",
                str(prepared.train_path.resolve()),
                "--dev_path",
                str(prepared.dev_path.resolve()),
            ]
        )
    elif has_validation:
        cmd.extend(["--dev_path", str(prepared.dev_path.resolve())])
    if has_validation:
        cmd.extend(
            [
                "--dev_prediction_path",
                str(dev_hard.resolve()),
                "--dev_soft_prediction_path",
                str(dev_soft.resolve()),
                "--dev_label_path",
                str(dev_labels.resolve()),
            ]
        )
    epochwise_output_dir = alg.get("epochwise_output_dir")
    if epochwise_output_dir:
        cmd.extend(["--epochwise_output_dir", str(resolve(epochwise_output_dir).resolve())])
    early_stop_config = alg.get("early_stop", {})
    if early_stop_config:
        if early_stop_config.get("validation_threshold") is not None:
            cmd.extend(["--early_stop_metric_threshold", str(early_stop_config["validation_threshold"])])
        if early_stop_config.get("test_threshold") is not None:
            cmd.extend(["--early_stop_test_threshold", str(early_stop_config["test_threshold"])])
        if early_stop_config.get("metrics"):
            metrics = early_stop_config["metrics"]
            if isinstance(metrics, (list, tuple)):
                metrics = ",".join(str(item) for item in metrics)
            cmd.extend(["--early_stop_metrics", str(metrics)])
    cmd = append_extra_args(cmd, alg.get("extra_args"))
    rc = run_command(
        cmd,
        resolve("Benchmark/2025-SP-TrafficFormer"),
        run_dir / "run.log",
        config,
        config.get("runtime", {}).get("timeout_seconds"),
    )
    if rc != 0:
        raise RuntimeError(f"TrafficFormer failed with exit code {rc}")
    metadata = {"algorithm": "TrafficFormer", "eval_checkpoint_path": rel(weight) if eval_checkpoint else None}
    if dev_soft.exists():
        metadata.update(
            {
                "validation_hard_predictions": rel(dev_hard),
                "validation_soft_predictions": rel(dev_soft),
                "validation_labels": rel(dev_labels),
            }
        )
    return PredictionFiles(
        hard,
        labels,
        soft,
        prepared.label_mapping_path,
        out_dir,
        metadata,
    )


def run_netmamba(prepared: PreparedData, config: dict[str, Any], run_dir: Path) -> PredictionFiles:
    alg = config["algorithm_config"]
    if alg.get("cpu_smoke_fallback"):
        return run_netmamba_cpu_smoke_fallback(prepared, config, run_dir, "cpu_smoke_fallback enabled")
    eval_checkpoint = alg.get("eval_checkpoint_path")
    checkpoint = resolve(eval_checkpoint) if eval_checkpoint else resolve(alg["checkpoint"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"NetMamba checkpoint not found: {checkpoint}")
    out_dir = run_dir / "algorithm_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    ft = alg["finetune"]
    epochs = int(ft.get("max_epoch", ft.get("epochs", 120)))
    num_classes = len((prepared.metadata or {}).get("label_mapping", {}))
    if eval_checkpoint:
        cmd = [
            runtime_python(config),
            "src/eval.py",
            "--batch_size",
            str(ft["batch_size"]),
            "--drop_path",
            str(ft.get("drop_path", 0.1)),
            "--nb_classes",
            str(num_classes or ft["nb_classes"]),
            "--resume",
            str(checkpoint.resolve()),
            "--data_path",
            str(prepared.work_dir.resolve()),
            "--output_dir",
            str(out_dir.resolve()),
            "--log_dir",
            str(out_dir.resolve()),
            "--model",
            ft["model"],
            "--device",
            ft["device"],
            "--num_workers",
            str(ft["num_workers"]),
            "--write_predictions",
        ]
        cmd = append_extra_args(cmd, alg.get("extra_args"))
        rc = run_command(
            cmd,
            resolve("Benchmark/2024-ICNP-NetMamba"),
            run_dir / "run.log",
            config,
            config.get("runtime", {}).get("timeout_seconds"),
        )
        if rc != 0:
            raise RuntimeError(f"NetMamba eval failed with exit code {rc}")
        return PredictionFiles(
            out_dir / "hard_predictions.txt",
            out_dir / "test_labels.txt",
            out_dir / "soft_predictions.npy",
            prepared.label_mapping_path,
            out_dir,
            {"algorithm": "NetMamba", "eval_checkpoint_path": rel(checkpoint)},
        )
    cmd = [
        runtime_python(config),
        "src/fine-tune.py",
        "--blr",
        str(ft["blr"]),
        "--epochs",
        str(epochs),
        "--batch_size",
        str(ft["batch_size"]),
        "--weight_decay",
        str(ft.get("weight_decay", 0.05)),
        "--drop_path",
        str(ft.get("drop_path", 0.1)),
        "--nb_classes",
        str(num_classes or ft["nb_classes"]),
        "--finetune",
        str(checkpoint.resolve()),
        "--data_path",
        str(prepared.work_dir.resolve()),
        "--output_dir",
        str(out_dir.resolve()),
        "--log_dir",
        str(out_dir.resolve()),
        "--model",
        ft["model"],
        "--device",
        ft["device"],
        "--num_workers",
        str(ft["num_workers"]),
    ]
    if ft.get("no_amp", False):
        cmd.append("--no_amp")
    epochwise_output_dir = alg.get("epochwise_output_dir")
    if epochwise_output_dir:
        cmd.extend(["--epochwise_output_dir", str(resolve(epochwise_output_dir).resolve())])
    early_stop_config = alg.get("early_stop", {})
    if early_stop_config:
        if early_stop_config.get("validation_threshold") is not None:
            cmd.extend(["--early_stop_metric_threshold", str(early_stop_config["validation_threshold"])])
        if early_stop_config.get("test_threshold") is not None:
            cmd.extend(["--early_stop_test_threshold", str(early_stop_config["test_threshold"])])
        if early_stop_config.get("metrics"):
            metrics = early_stop_config["metrics"]
            if isinstance(metrics, (list, tuple)):
                metrics = ",".join(str(item) for item in metrics)
            cmd.extend(["--early_stop_metrics", str(metrics)])
    cmd = append_extra_args(cmd, alg.get("extra_args"))
    rc = run_command(cmd, resolve("Benchmark/2024-ICNP-NetMamba"), run_dir / "run.log", config, config.get("runtime", {}).get("timeout_seconds"))
    if rc != 0:
        if alg.get("cpu_smoke_fallback"):
            return run_netmamba_cpu_smoke_fallback(prepared, config, run_dir, f"native NetMamba failed with exit code {rc}")
        raise RuntimeError(f"NetMamba failed with exit code {rc}")
    return PredictionFiles(
        out_dir / "hard_predictions.txt",
        out_dir / "test_labels.txt",
        out_dir / "soft_predictions.npy",
        prepared.label_mapping_path,
        out_dir,
        {"algorithm": "NetMamba"},
    )


def _netmamba_image_rows(root: Path, mapping_path: Path) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    from PIL import Image

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    class_dirs = mapping.get("class_directories", {})
    dir_to_label = {directory: label for label, directory in class_dirs.items()}
    rows = []
    labels = []
    counts: Counter = Counter()
    for image_path in sorted(root.glob("*/*.png")):
        directory = image_path.parent.name
        label = dir_to_label.get(directory, directory)
        image = Image.open(image_path).convert("L")
        rows.append(np.asarray(image, dtype=np.float32).reshape(-1) / 255.0)
        labels.append(label)
        counts[label] += 1
    if not rows:
        raise BenchmarkError(f"NetMamba fallback found no images under {root}")
    return np.stack(rows, axis=0), labels, {"rows": len(rows), "label_counts": dict(sorted(counts.items()))}


def run_netmamba_cpu_smoke_fallback(prepared: PreparedData, config: dict[str, Any], run_dir: Path, reason: str) -> PredictionFiles:
    from sklearn.ensemble import RandomForestClassifier

    out_dir = run_dir / "algorithm_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    if prepared.train_path is None or prepared.test_path is None or prepared.label_mapping_path is None:
        raise BenchmarkError("NetMamba fallback requires prepared ImageFolder train/test data and label mapping")
    train_x, train_labels, train_meta = _netmamba_image_rows(prepared.train_path, prepared.label_mapping_path)
    test_x, test_labels, test_meta = _netmamba_image_rows(prepared.test_path, prepared.label_mapping_path)
    mapping = json.loads(prepared.label_mapping_path.read_text(encoding="utf-8"))
    label_to_id = {str(label): int(index) for label, index in mapping.get("label_to_id", {}).items()}
    label_order = [label for label, _index in sorted(label_to_id.items(), key=lambda item: item[1])]
    model = RandomForestClassifier(n_estimators=25, random_state=42)
    model.fit(train_x, train_labels)
    predictions = model.predict(test_x).tolist()
    proba_native = model.predict_proba(test_x)
    soft = np.zeros((len(test_labels), len(label_order)), dtype=np.float32)
    for native_index, label in enumerate(model.classes_):
        if label in label_to_id:
            soft[:, label_to_id[label]] = proba_native[:, native_index]
    hard_path = out_dir / "hard_predictions.txt"
    labels_path = out_dir / "test_labels.txt"
    soft_path = out_dir / "soft_predictions.npy"
    hard_path.write_text("\n".join(str(label_to_id.get(label, label)) for label in predictions) + "\n", encoding="utf-8")
    labels_path.write_text("\n".join(str(label_to_id.get(label, label)) for label in test_labels) + "\n", encoding="utf-8")
    np.save(soft_path, soft)
    metadata = {
        "algorithm": "NetMamba",
        "cpu_smoke_fallback": True,
        "fallback_reason": reason,
        "fallback_classifier": "RandomForestClassifier",
        "native_netmamba_executed": False,
        "train": train_meta,
        "test": test_meta,
    }
    write_json(out_dir / "metrics.json", metadata)
    return PredictionFiles(
        hard_path,
        labels_path,
        soft_path,
        prepared.label_mapping_path,
        out_dir,
        metadata,
    )


def _pcap_endian(header: bytes, path: Path) -> str:
    magic = header[:4]
    if magic in {b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"}:
        return "<"
    if magic in {b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"}:
        return ">"
    raise BenchmarkError(f"Unsupported pcap format for Kitsune input: {path}")


def _packet_labels(path: Path, max_labels: int | None = None) -> list[str]:
    labels: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            value = row[-1].strip()
            if value.lower() in {"label", "labels", "x"}:
                continue
            labels.append("benign" if value.lower() in {"0", "0.0", "benign", "normal", "false"} else "malicious")
            if max_labels is not None and len(labels) >= max_labels:
                break
    return labels


def _copy_pcap_packets(
    source: PacketDataset,
    out_handle,
    labels_handle,
    expected_header: bytes | None,
    benign_only: bool,
    max_packets: int | None = None,
) -> tuple[bytes, int, int]:
    labels = _packet_labels(source.packet_labels, max_packets)
    with source.pcap.open("rb") as handle:
        header = handle.read(24)
        if len(header) != 24:
            raise BenchmarkError(f"Invalid pcap header: {source.pcap}")
        _pcap_endian(header, source.pcap)
        if expected_header is not None and header[:4] != expected_header[:4]:
            raise BenchmarkError(f"Cannot combine pcaps with different pcap magic: {source.pcap}")
        if expected_header is None:
            out_handle.write(header)
        unpack = struct.Struct(_pcap_endian(header, source.pcap) + "IIII").unpack
        written = 0
        seen = 0
        while seen < len(labels):
            record_header = handle.read(16)
            if not record_header:
                break
            if len(record_header) != 16:
                warnings.warn(f"Skipping truncated pcap record header at end of file: {source.pcap}", RuntimeWarning)
                break
            _ts_sec, _ts_sub, incl_len, _orig_len = unpack(record_header)
            payload = handle.read(incl_len)
            if len(payload) != incl_len:
                warnings.warn(f"Skipping truncated pcap packet payload at end of file: {source.pcap}", RuntimeWarning)
                break
            label = labels[seen]
            seen += 1
            if benign_only and label != "benign":
                continue
            out_handle.write(record_header)
            out_handle.write(payload)
            labels_handle.write(label + "\n")
            written += 1
    return header, written, seen


def prepare_kitsune_external_split(prepared: PreparedData, config: dict[str, Any], work_dir: Path) -> tuple[Path, Path, int, dict[str, Any]]:
    if not prepared.training_packets or not prepared.test_packets:
        raise BenchmarkError("Kitsune requires packet-level train and test inputs")
    max_packets_per_source = config.get("input", {}).get("max_packets_per_source")
    max_packets_per_source = int(max_packets_per_source) if max_packets_per_source is not None else None
    combined_pcap = work_dir / "kitsune_combined.pcap"
    combined_labels = work_dir / "kitsune_combined_labels.csv"
    metadata: dict[str, Any] = {
        "training_policy": "benign packets from --training_set only, followed by optional validation packets, followed by all --test_set packets",
        "training_sources": [],
        "validation_sources": [],
        "test_sources": [],
    }
    train_packets = 0
    validation_packets = 0
    expected_header: bytes | None = None
    with combined_pcap.open("wb") as pcap_out, combined_labels.open("w", encoding="utf-8") as labels_out:
        for source in prepared.training_packets:
            labels = _packet_labels(source.packet_labels, max_packets_per_source)
            header, written, seen = _copy_pcap_packets(
                source,
                pcap_out,
                labels_out,
                expected_header,
                benign_only=True,
                max_packets=max_packets_per_source,
            )
            if expected_header is None:
                expected_header = header
            train_packets += written
            metadata["training_sources"].append(
                {
                    "pcap": rel(source.pcap),
                    "packet_labels": rel(source.packet_labels),
                    "packets_with_labels": seen,
                    "benign_training_packets": written,
                    "malicious_or_unknown_skipped": len(labels) - written,
                }
            )
        if expected_header is None:
            raise BenchmarkError("No Kitsune training pcap was loaded")
        if train_packets <= 0:
            raise BenchmarkError("Kitsune training set contains no benign packets")
        for source in prepared.validation_packets or []:
            _labels = _packet_labels(source.packet_labels, max_packets_per_source)
            header, written, seen = _copy_pcap_packets(
                source,
                pcap_out,
                labels_out,
                expected_header,
                benign_only=False,
                max_packets=max_packets_per_source,
            )
            validation_packets += written
            metadata["validation_sources"].append(
                {
                    "pcap": rel(source.pcap),
                    "packet_labels": rel(source.packet_labels),
                    "packets_with_labels": seen,
                    "validation_packets": written,
                }
            )
        for source in prepared.test_packets:
            _labels = _packet_labels(source.packet_labels, max_packets_per_source)
            header, written, seen = _copy_pcap_packets(
                source,
                pcap_out,
                labels_out,
                expected_header,
                benign_only=False,
                max_packets=max_packets_per_source,
            )
            metadata["test_sources"].append(
                {
                    "pcap": rel(source.pcap),
                    "packet_labels": rel(source.packet_labels),
                    "packets_with_labels": seen,
                    "test_packets": written,
                }
            )
    metadata["train_packets"] = train_packets
    metadata["validation_packets"] = validation_packets
    metadata["combined_pcap"] = rel(combined_pcap)
    metadata["combined_packet_labels"] = rel(combined_labels)
    return combined_pcap, combined_labels, train_packets, metadata


def _kitsune_source_fingerprint(source: PacketDataset) -> dict[str, Any]:
    pcap = source.pcap.resolve()
    labels = source.packet_labels.resolve()
    pcap_stat = pcap.stat()
    labels_stat = labels.stat()
    return {
        "source": source.source,
        "pcap": str(pcap),
        "pcap_size": int(pcap_stat.st_size),
        "pcap_mtime_ns": int(pcap_stat.st_mtime_ns),
        "packet_labels": str(labels),
        "packet_labels_size": int(labels_stat.st_size),
        "packet_labels_mtime_ns": int(labels_stat.st_mtime_ns),
    }


def _kitsune_afterimage_cache_dir(prepared: PreparedData, alg: dict[str, Any], metadata: dict[str, Any]) -> Path:
    payload = {
        "algorithm": "Kitsune",
        "cache_version": 1,
        "training_policy": metadata.get("training_policy"),
        "max_packets_per_source": alg.get("input", {}).get("max_packets_per_source"),
        "max_packets": alg.get("input", {}).get("max_packets"),
        "training_sources": [_kitsune_source_fingerprint(source) for source in (prepared.training_packets or [])],
        "validation_sources": [_kitsune_source_fingerprint(source) for source in (prepared.validation_packets or [])],
        "test_sources": [_kitsune_source_fingerprint(source) for source in (prepared.test_packets or [])],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    cache_dir = resolve("Benchmark/cache/kitsune_afterimage") / digest
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(cache_dir / "cache_manifest.json", payload)
    return cache_dir


def run_kitsune(prepared: PreparedData, config: dict[str, Any], run_dir: Path) -> PredictionFiles:
    alg = json.loads(json.dumps(config["algorithm_config"]))
    if alg.get("training", {}).get("max_epoch") is not None:
        alg.setdefault("training", {})["epochs"] = int(alg["training"]["max_epoch"])
    out_dir = run_dir / "algorithm_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_pcap, combined_labels, train_packets, metadata = prepare_kitsune_external_split(prepared, alg, prepared.work_dir)
    cache_dir = _kitsune_afterimage_cache_dir(prepared, alg, metadata)
    alg.setdefault("input", {})
    alg["input"].update(
        {
            "pcap": str(combined_pcap.resolve()),
            "packet_labels": str(combined_labels.resolve()),
            "feature_cache_csv": str((cache_dir / "afterimage_features.csv").resolve()),
            "tsv_cache": str((cache_dir / "afterimage_packets.tsv").resolve()),
            "reuse_feature_cache": True,
            "reuse_tsv_cache": True,
        }
    )
    alg.setdefault("split", {})
    alg["split"].update(
        {
            "method": "fixed_prefix",
            "train_packets": train_packets,
            "validation_packets": int(metadata.get("validation_packets", 0)),
            "require_benign_train": True,
            "min_train_packets": 1,
        }
    )
    alg.setdefault("output", {})
    alg["output"].update({"dir": str(out_dir.resolve())})
    runner_config = out_dir / "kitsune_runner_config.json"
    write_json(runner_config, alg)
    rc = run_command(
        [runtime_python(config), "run_packet_unified.py", "--config", str(runner_config.resolve())],
        resolve("Benchmark/2018-NDSS-Kitsune"),
        run_dir / "run.log",
        config,
        config.get("runtime", {}).get("timeout_seconds"),
    )
    if rc != 0:
        raise RuntimeError(f"Kitsune failed with exit code {rc}")
    metadata.update(
        {
            "algorithm": "Kitsune",
            "runner_config": rel(runner_config),
            "afterimage_cache_dir": rel(cache_dir),
            "afterimage_tsv_cache": rel(cache_dir / "afterimage_packets.tsv"),
            "afterimage_feature_cache": rel(cache_dir / "afterimage_features.csv"),
        }
    )
    return PredictionFiles(
        out_dir / alg["output"].get("hard_labels_txt", "hard_predictions.txt"),
        out_dir / alg["output"].get("test_labels_txt", "test_labels.txt"),
        out_dir / alg["output"].get("soft_labels_npy", "soft_predictions.npy"),
        None,
        out_dir,
        metadata,
    )


def _load_flowlens_runner():
    module_path = resolve("Benchmark/2021-NDSS-FlowLens/run_unified_classifier.py")
    spec = importlib.util.spec_from_file_location("benchmark_flowlens_runner", module_path)
    if spec is None or spec.loader is None:
        raise BenchmarkError(f"Cannot load FlowLens runner module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_flowlens_runner"] = module
    spec.loader.exec_module(module)
    return module


def _flownlens_parser_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": {
            "packet_label_delimiter": config.get("input", {}).get("packet_label_delimiter", ","),
        },
        "labels": {
            "benign_values": config["labels"].get("benign_values", []),
            "malicious_values": config["labels"].get("malicious_values", []),
        },
        "accumulator": {
            "packet_length_source": config["flowlens"]["packet_length_source"],
            "skip_non_tcp_udp": config["flowlens"]["skip_non_tcp_udp"],
            "skip_zero_payload_packets": config["flowlens"]["skip_zero_payload_packets"],
        },
    }


def _flowlens_examples(
    runner,
    sources: list[PacketDataset],
    config: dict[str, Any],
    role: str,
) -> tuple[list[Any], list[str], dict[str, Any], dict[str, Any]]:
    parser_config = _flownlens_parser_config(config)
    superflow_policy = config["labels"]["superflow_label_policy"]
    examples: list[Any] = []
    labels: list[str] = []
    source_meta = []
    five_tuple_by_id: dict[str, Any] = {}
    for source_index, source in enumerate(sources):
        packet_labels = runner.acc.load_packet_labels(str(source.packet_labels), parser_config)
        packet_iter = runner.acc.iter_pcap(source.pcap, parser_config, packet_labels)
        max_packets = config.get("input", {}).get("max_packets")
        if max_packets is not None:
            packet_iter = runner.islice(packet_iter, int(max_packets))
        packets = list(packet_iter)
        five_tuple_flows, packet_to_flow_id = runner.build_five_tuple_flows(packets, config)
        for flow_id, flow in five_tuple_flows.items():
            five_tuple_by_id[f"{role}{source_index}:{flow_id}"] = flow
        flows = runner.build_flows(packets, packet_to_flow_id, config)
        superflows = runner.build_superflows(flows, config)
        source_valid = 0
        for flow in superflows:
            label = runner.label_from_votes(flow.label_votes, superflow_policy)
            if not label or label == "mixed" or flow.packets <= 0 or flow.bytes <= 0 or flow.duration <= 0:
                continue
            flow.five_tuple_ids = {f"{role}{source_index}:{flow_id}" for flow_id in flow.five_tuple_ids}
            examples.append(flow)
            labels.append(label)
            source_valid += 1
        source_meta.append(
            {
                "pcap": rel(source.pcap),
                "packet_labels": rel(source.packet_labels),
                "packets": len(packets),
                "five_tuple_flows": len(five_tuple_flows),
                "superflows": len(superflows),
                "valid_superflows": source_valid,
                "label_counts": dict(sorted(Counter(labels[-source_valid:]).items())) if source_valid else {},
            }
        )
    return examples, labels, {"sources": source_meta}, five_tuple_by_id


def _flowlens_feature_matrix(superflows: list[Any]) -> np.ndarray:
    return np.asarray([[flow.packets, flow.bytes, flow.median_ipt, flow.duration] for flow in superflows], dtype=float)


def _flowlens_probability_matrix(model: RandomForestClassifier, features: np.ndarray, labels: list[str]) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    output = np.zeros((features.shape[0], len(labels)), dtype=float)
    for source_index, label in enumerate(classes):
        if label in labels:
            output[:, labels.index(label)] = probabilities[:, source_index]
    return output


def _binary_label(value: object) -> str:
    return "benign" if str(value).strip().lower() in {"0", "0.0", "benign", "normal", "false"} else "malicious"


def _rewrite_source_label_counts(meta: dict[str, Any], labels: list[str]) -> None:
    offset = 0
    for source in meta.get("sources", []):
        count = int(source.get("valid_superflows", 0))
        source["label_counts"] = dict(sorted(Counter(labels[offset:offset + count]).items())) if count else {}
        offset += count


def _flowlens_expand_to_five_tuple(
    runner,
    superflows: list[Any],
    superflow_predictions: list[str],
    superflow_probabilities: np.ndarray,
    five_tuple_by_id: dict[str, Any],
    label_order: list[str],
    label_policy: str,
    force_binary: bool,
    output_csv: Path,
) -> tuple[list[str], list[str], np.ndarray, dict[str, Any]]:
    flow_true: list[str] = []
    flow_pred: list[str] = []
    flow_soft: list[np.ndarray] = []
    rows: list[list[Any]] = []
    for superflow_index, (superflow, prediction) in enumerate(zip(superflows, superflow_predictions)):
        probability = superflow_probabilities[superflow_index]
        for flow_id in sorted(superflow.five_tuple_ids):
            flow = five_tuple_by_id.get(flow_id)
            if flow is None:
                continue
            true_label = runner.label_from_votes(flow.label_votes, label_policy)
            if not true_label or true_label == "mixed":
                continue
            if force_binary:
                true_label = _binary_label(true_label)
            flow_true.append(true_label)
            flow_pred.append(prediction)
            flow_soft.append(probability)
            rows.append(
                [
                    flow_id,
                    f"superflow_{superflow_index}",
                    getattr(flow, "src_ip", ""),
                    getattr(flow, "dst_ip", ""),
                    getattr(flow, "src_port", ""),
                    getattr(flow, "dst_port", ""),
                    getattr(flow, "protocol", ""),
                    f"{float(getattr(flow, 'start', 0.0)):.9f}",
                    f"{float(getattr(flow, 'end', 0.0)):.9f}",
                    int(getattr(flow, "packets", 0)),
                    int(getattr(flow, "bytes", 0)),
                    true_label,
                    prediction,
                ]
            )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "FLOW_ID",
                "SUPERFLOW_ID",
                "SRC_IP",
                "DST_IP",
                "SRC_PORT",
                "DST_PORT",
                "PROTOCOL",
                "START_TIME",
                "END_TIME",
                "PACKETS",
                "BYTES",
                "TRUE_LABEL",
                "PRED_LABEL",
            ]
        )
        writer.writerows(rows)
    if flow_soft:
        soft = np.asarray(flow_soft, dtype=float)
    else:
        soft = np.empty((0, len(label_order)), dtype=float)
    metadata = {
        "evaluated_five_tuple_flows": len(flow_true),
        "five_tuple_label_counts": dict(sorted(Counter(flow_true).items())),
        "five_tuple_prediction_counts": dict(sorted(Counter(flow_pred).items())),
        "five_tuple_predictions": rel(output_csv),
    }
    return flow_pred, flow_true, soft, metadata


def _binary_threshold_metrics(scores: np.ndarray, labels: list[str] | np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray([_binary_label(value) for value in labels])
    y_pred = np.where(scores >= threshold, "malicious", "benign")
    tp = int(np.sum((y_true == "malicious") & (y_pred == "malicious")))
    fp = int(np.sum((y_true == "benign") & (y_pred == "malicious")))
    fn = int(np.sum((y_true == "malicious") & (y_pred == "benign")))
    tn = int(np.sum((y_true == "benign") & (y_pred == "benign")))
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    accuracy = float((tp + tn) / len(y_true)) if len(y_true) else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "malicious_f1": f1,
        "f1": f1,
        "fpr": fpr,
        "accuracy": accuracy,
    }


def _select_binary_threshold(
    scores: np.ndarray,
    labels: list[str] | np.ndarray,
    max_candidates: int = 2000,
) -> dict[str, Any]:
    finite_scores = np.asarray(scores[np.isfinite(scores)], dtype=float)
    if finite_scores.size == 0:
        raise BenchmarkError("Cannot select threshold: no finite scores")
    thresholds = np.unique(finite_scores)
    if max_candidates > 0 and thresholds.size > max_candidates:
        thresholds = np.quantile(thresholds, np.linspace(0.0, 1.0, max_candidates))
        thresholds = np.unique(thresholds)
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        metrics = _binary_threshold_metrics(scores, labels, float(threshold))
        if best is None:
            best = metrics
            continue
        current_key = (metrics["malicious_f1"], metrics["recall"], -metrics["fpr"], -metrics["threshold"])
        best_key = (best["malicious_f1"], best["recall"], -best["fpr"], -best["threshold"])
        if current_key > best_key:
            best = metrics
    assert best is not None
    best["candidate_count"] = int(len(thresholds))
    return best


def run_flowlens(prepared: PreparedData, config: dict[str, Any], run_dir: Path) -> PredictionFiles:
    if not prepared.training_packets or not prepared.test_packets:
        raise BenchmarkError("FlowLens requires packet-level train and test inputs")
    runner = _load_flowlens_runner()
    alg = json.loads(json.dumps(config["algorithm_config"]))
    flowlens_config = runner.deep_merge(runner.DEFAULT_CONFIG, alg.get("classifier", {}))
    flowlens_config.setdefault("input", {})
    flowgap_seconds = float(flowlens_config["flowlens"].get("flowgap_seconds", 0.0))
    if flowgap_seconds > 1200.0:
        raise BenchmarkError(f"FlowLens flowgap_seconds must be <= 1200.0 for Benchmark A, got {flowgap_seconds}")
    out_dir = run_dir / "algorithm_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    force_binary = str(config.get("data", {}).get("label_mode", "")).lower() == "binary"
    train_flows, train_labels, train_meta, _train_five_tuple_by_id = _flowlens_examples(
        runner, prepared.training_packets, flowlens_config, "train"
    )
    test_flows, test_labels, test_meta, test_five_tuple_by_id = _flowlens_examples(
        runner, prepared.test_packets, flowlens_config, "test"
    )
    val_flows: list[Any] = []
    val_labels: list[str] = []
    val_meta: dict[str, Any] = {"sources": []}
    if prepared.validation_packets:
        val_flows, val_labels, val_meta, _val_five_tuple_by_id = _flowlens_examples(
            runner, prepared.validation_packets, flowlens_config, "val"
        )
    if force_binary:
        train_labels = [_binary_label(label) for label in train_labels]
        test_labels = [_binary_label(label) for label in test_labels]
        val_labels = [_binary_label(label) for label in val_labels]
        _rewrite_source_label_counts(train_meta, train_labels)
        _rewrite_source_label_counts(test_meta, test_labels)
        _rewrite_source_label_counts(val_meta, val_labels)
    if not train_flows:
        raise BenchmarkError("FlowLens training inputs produced no labelled superflows")
    if not test_flows:
        raise BenchmarkError("FlowLens test inputs produced no labelled superflows")

    train_features = _flowlens_feature_matrix(train_flows)
    test_features = _flowlens_feature_matrix(test_flows)
    classifier_cfg = flowlens_config["classifier"]
    eval_model_path = alg.get("eval_model_path")
    if eval_model_path:
        import joblib

        bundle = joblib.load(resolve(eval_model_path))
        model = bundle["model"]
    else:
        model = RandomForestClassifier(
            random_state=int(classifier_cfg["random_state"]),
            n_estimators=int(classifier_cfg.get("n_estimators", 100)),
            max_depth=classifier_cfg.get("max_depth"),
            class_weight=classifier_cfg.get("class_weight"),
        )
        model.fit(train_features, np.asarray(train_labels))
    predictions = model.predict(test_features).tolist()
    label_order = ["benign", "malicious"] if set(train_labels + test_labels) <= {"benign", "malicious"} else sorted(set(train_labels + test_labels))
    if eval_model_path and bundle.get("label_order"):
        label_order = [str(label) for label in bundle["label_order"]]
    probabilities = _flowlens_probability_matrix(model, test_features, label_order)
    threshold_metadata: dict[str, Any] = {"threshold_source": "native_hard_labels"}
    val_probabilities = None
    if eval_model_path and isinstance(bundle.get("threshold_metadata"), dict):
        threshold_metadata = dict(bundle["threshold_metadata"])
        if "threshold" in threshold_metadata and label_order == ["benign", "malicious"]:
            threshold = float(threshold_metadata["threshold"])
            predictions = np.where(probabilities[:, 1] >= threshold, "malicious", "benign").tolist()
            threshold_metadata["threshold_source"] = f"loaded_{threshold_metadata.get('threshold_source', 'threshold')}"
    elif label_order == ["benign", "malicious"] and val_flows and val_labels:
        val_features = _flowlens_feature_matrix(val_flows)
        val_probabilities = _flowlens_probability_matrix(model, val_features, label_order)
        selected = _select_binary_threshold(val_probabilities[:, 1], val_labels)
        threshold = float(selected["threshold"])
        predictions = np.where(probabilities[:, 1] >= threshold, "malicious", "benign").tolist()
        threshold_metadata = {
            "threshold_source": "validation_threshold_scan",
            "selection_metric": "malicious_f1",
            "threshold": threshold,
            "validation_metrics_at_selected_threshold": selected,
            "test_superflow_metrics_at_selected_threshold": _binary_threshold_metrics(probabilities[:, 1], test_labels, threshold),
        }

    superflow_hard_path = out_dir / "superflow_hard_predictions.txt"
    superflow_labels_path = out_dir / "superflow_test_labels.txt"
    superflow_soft_path = out_dir / "superflow_soft_predictions.npy"
    hard_path = out_dir / "hard_predictions.txt"
    labels_path = out_dir / "test_labels.txt"
    soft_path = out_dir / "soft_predictions.npy"
    validation_soft_path = out_dir / "validation_soft_predictions.npy"
    validation_labels_path = out_dir / "validation_labels.txt"
    dataset_path = out_dir / "superflow_dataset.csv"
    five_tuple_predictions_path = out_dir / "five_tuple_predictions.csv"
    superflow_hard_path.write_text("\n".join(predictions) + "\n", encoding="utf-8")
    superflow_labels_path.write_text("\n".join(test_labels) + "\n", encoding="utf-8")
    np.save(superflow_soft_path, probabilities)
    runner.write_superflow_dataset(test_flows, test_labels, dataset_path)
    five_tuple_label_policy = flowlens_config["labels"].get("five_tuple_label_policy", "majority_vote")
    flow_predictions, flow_labels, flow_probabilities, flow_metadata = _flowlens_expand_to_five_tuple(
        runner,
        test_flows,
        predictions,
        probabilities,
        test_five_tuple_by_id,
        label_order,
        str(five_tuple_label_policy),
        force_binary,
        five_tuple_predictions_path,
    )
    if not flow_labels:
        raise BenchmarkError("FlowLens test superflows mapped to no labelled five-tuple flows")
    if label_order == ["benign", "malicious"] and "threshold" in threshold_metadata:
        threshold_metadata["test_metrics_at_selected_threshold"] = _binary_threshold_metrics(
            flow_probabilities[:, 1], flow_labels, float(threshold_metadata["threshold"])
        )
    hard_path.write_text("\n".join(flow_predictions) + "\n", encoding="utf-8")
    labels_path.write_text("\n".join(flow_labels) + "\n", encoding="utf-8")
    np.save(soft_path, flow_probabilities)
    if val_probabilities is not None:
        validation_labels_path.write_text("\n".join(val_labels) + "\n", encoding="utf-8")
        np.save(validation_soft_path, val_probabilities.astype(np.float32))

    mapping_path = out_dir / "label_mapping.json"
    write_json(
        mapping_path,
        {
            "label_to_id": {label: index for index, label in enumerate(label_order)},
            "id_to_label": {str(index): label for index, label in enumerate(label_order)},
        },
    )
    metadata = {
        "algorithm": "FlowLens",
        "classifier": "RandomForestClassifier",
        "classifier_params": {
            "random_state": int(classifier_cfg["random_state"]),
            "n_estimators": int(classifier_cfg.get("n_estimators", 100)),
            "max_depth": classifier_cfg.get("max_depth"),
            "class_weight": classifier_cfg.get("class_weight"),
        },
        "feature_columns": ["NumberOfPackets", "TotalBytesTransmitted", "MedianIPT", "ConversationDuration"],
        "grain": "five_tuple_flow",
        "prediction_grain": "superflow_mapped_to_five_tuple_flow",
        "timegap_seconds": float(flowlens_config["flowlens"].get("timegap_seconds", 120.0)),
        "flowgap_seconds": flowgap_seconds,
        "five_tuple_mode": flowlens_config["flowlens"].get("five_tuple_mode"),
        "five_tuple_timeout_seconds": float(flowlens_config["flowlens"].get("five_tuple_timeout_seconds", 120.0)),
        "train_superflows": len(train_flows),
        "validation_superflows": len(val_flows),
        "test_superflows": len(test_flows),
        "test_flows": len(flow_labels),
        "train_label_counts": dict(sorted(Counter(train_labels).items())),
        "validation_label_counts": dict(sorted(Counter(val_labels).items())),
        "test_superflow_label_counts": dict(sorted(Counter(test_labels).items())),
        "test_label_counts": dict(sorted(Counter(flow_labels).items())),
        "training_sources": train_meta["sources"],
        "validation_sources": val_meta["sources"],
        "test_sources": test_meta["sources"],
        "superflow_dataset": rel(dataset_path),
        "superflow_hard_predictions": rel(superflow_hard_path),
        "superflow_test_labels": rel(superflow_labels_path),
        "superflow_soft_predictions": rel(superflow_soft_path),
        "classes": list(model.classes_),
        **flow_metadata,
        **threshold_metadata,
    }
    if val_probabilities is not None:
        metadata["validation_soft_predictions"] = rel(validation_soft_path)
        metadata["validation_labels"] = rel(validation_labels_path)
    if not eval_model_path:
        import joblib

        model_path = out_dir / "flowlens_model.joblib"
        joblib.dump(
            {
                "model": model,
                "label_order": label_order,
                "threshold_metadata": threshold_metadata,
                "flowlens_config": flowlens_config,
            },
            model_path,
        )
        metadata["model_path"] = rel(model_path)
    else:
        metadata["eval_model_path"] = rel(resolve(eval_model_path))
    write_json(out_dir / "metrics.json", metadata)
    if threshold_metadata.get("threshold_source") == "validation_threshold_scan":
        write_json(out_dir / "threshold_metrics.json", threshold_metadata)
    return PredictionFiles(hard_path, labels_path, soft_path, mapping_path, out_dir, metadata)


def _tfusion_label_to_int(value: object) -> int:
    return 0 if str(value).strip().lower() in {"0", "0.0", "benign", "normal", "false"} else 1


def _load_flowlens_parser_for_tfusion():
    module_path = resolve("Benchmark/2021-NDSS-FlowLens/unified_flowlens.py")
    spec = importlib.util.spec_from_file_location("benchmark_tfusion_flowlens_parser", module_path)
    if spec is None or spec.loader is None:
        raise BenchmarkError(f"Cannot load pcap parser module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_tfusion_flowlens_parser"] = module
    spec.loader.exec_module(module)
    return module


def _tfusion_canonical_five_tuple(packet, mode: str) -> tuple[str, str, str, str, str]:
    src = (str(packet.src_ip), str(packet.src_port))
    dst = (str(packet.dst_ip), str(packet.dst_port))
    protocol = str(packet.protocol)
    if mode == "unidirectional":
        return (src[0], dst[0], src[1], dst[1], protocol)
    if src <= dst:
        return (src[0], dst[0], src[1], dst[1], protocol)
    return (dst[0], src[0], dst[1], src[1], protocol)


def _tfusion_cache_enabled(alg: dict[str, Any]) -> bool:
    cache_cfg = alg.get("cache", {})
    if isinstance(cache_cfg, dict) and cache_cfg.get("enabled") is False:
        return False
    return True


def _tfusion_cache_root(alg: dict[str, Any]) -> Path:
    cache_cfg = alg.get("cache", {})
    if isinstance(cache_cfg, dict) and cache_cfg.get("dir"):
        return resolve(cache_cfg["dir"])
    return resolve("Benchmark/cache/tfusion")


def _tfusion_file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": rel(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _tfusion_cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _tfusion_pickle_dump_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def _tfusion_pickle_load(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _tfusion_raw_cache_payload(
    sources: list[PacketDataset],
    alg: dict[str, Any],
    role: str,
    parser_config: dict[str, Any],
    max_packets: int | None,
    timeout: float,
    mode: str,
) -> dict[str, Any]:
    return {
        "version": 2,
        "role": role,
        "sources": [
            {
                "source": source.source,
                "pcap": _tfusion_file_signature(source.pcap),
                "packet_labels": _tfusion_file_signature(source.packet_labels),
            }
            for source in sources
        ],
        "parser_config": parser_config,
        "max_packets": max_packets,
        "timeout": timeout,
        "mode": mode,
    }


def _tfusion_packet_sources_to_raw_flows(
    sources: list[PacketDataset],
    alg: dict[str, Any],
    role: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parser = _load_flowlens_parser_for_tfusion()
    conversion = alg.get("conversion", {}) or {}
    max_packets = conversion.get("max_packets_per_source")
    max_packets = int(max_packets) if max_packets is not None else None
    timeout = float(conversion.get("five_tuple_timeout_seconds", 120.0))
    mode = str(conversion.get("five_tuple_mode", "bidirectional"))
    parser_config = {
        "flow": {"skip_zero_payload_packets": bool(conversion.get("skip_zero_payload_packets", False))},
        "labels": {
            "benign_values": ["0", "0.0", "benign", "normal", "false"],
            "malicious_values": ["1", "1.0", "malicious", "attack", "true"],
        },
    }
    cache_payload = _tfusion_raw_cache_payload(sources, alg, role, parser_config, max_packets, timeout, mode)
    cache_key = _tfusion_cache_key(cache_payload)
    cache_path = _tfusion_cache_root(alg) / "raw" / f"{cache_key}.pkl"
    if _tfusion_cache_enabled(alg) and cache_path.exists():
        cached = _tfusion_pickle_load(cache_path)
        meta = dict(cached["meta"])
        meta["cache"] = {"hit": True, "key": cache_key, "path": rel(cache_path)}
        source_to_pcap = {str(item.get("source", "")): str(item.get("pcap", "")) for item in meta.get("sources", [])}
        flows = cached["flows"]
        for flow in flows:
            if not flow.get("source_file"):
                flow["source_file"] = source_to_pcap.get(str(flow.get("source", "")), str(flow.get("source", "")))
        return flows, meta

    all_flows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for source in sources:
        labels = _packet_labels(source.packet_labels, max_packets)
        packets = []
        for packet in parser.iter_pcap(source.pcap, parser_config, labels):
            if not getattr(packet, "label", ""):
                packet.label = "benign"
            packets.append(packet)
            if max_packets is not None and len(packets) >= max_packets:
                break
        packets.sort(key=lambda item: (_tfusion_canonical_five_tuple(item, mode), float(item.timestamp)))
        active: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
        source_flows: list[dict[str, Any]] = []
        for packet in packets:
            key = _tfusion_canonical_five_tuple(packet, mode)
            current = active.get(key)
            timestamp = float(packet.timestamp)
            if current is None or (timestamp - float(current["end_time"])) > timeout:
                current = {
                    "source": source.source,
                    "source_file": rel(source.pcap),
                    "key": key,
                    "id": [key[0], key[1]],
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "packets": [],
                    "labels": Counter(),
                    "bytes": 0,
                    "forward_bytes": 0,
                    "reverse_bytes": 0,
                    "ipts": [],
                }
                active[key] = current
                source_flows.append(current)
            if current["packets"]:
                current["ipts"].append(max(0.0, timestamp - float(current["end_time"])))
            current["end_time"] = max(float(current["end_time"]), timestamp)
            current["packets"].append(packet)
            current["labels"][str(packet.label)] += 1
            payload_length = max(0, int(packet.payload_length))
            current["bytes"] += payload_length
            if str(packet.src_ip) == key[0] and str(packet.dst_ip) == key[1]:
                current["forward_bytes"] += payload_length
            else:
                current["reverse_bytes"] += payload_length
        all_flows.extend(source_flows)
        source_rows.append(
            {
                "source": source.source,
                "pcap": rel(source.pcap),
                "packet_labels": rel(source.packet_labels),
                "packets_loaded": len(packets),
                "flows": len(source_flows),
                "role": role,
            }
        )
    meta = {"sources": source_rows, "flows": len(all_flows), "cache": {"hit": False, "key": cache_key, "path": rel(cache_path)}}
    if _tfusion_cache_enabled(alg):
        _tfusion_pickle_dump_atomic(cache_path, {"flows": all_flows, "meta": meta, "payload": cache_payload})
    return all_flows, meta


def _tfusion_flow_label(raw_flow: dict[str, Any]) -> str:
    labels: Counter = raw_flow["labels"]
    return "malicious" if any(str(label).lower() not in {"0", "0.0", "benign", "normal", "false"} for label in labels) else "benign"


def _tfusion_median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def _tfusion_feature_stats(flows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    values_by_field: dict[str, list[float]] = {
        "packets": [],
        "bytes": [],
        "duration": [],
        "median_ipt": [],
    }
    for flow in flows:
        values_by_field["packets"].append(float(len(flow["packets"])))
        values_by_field["bytes"].append(float(flow["bytes"]))
        values_by_field["duration"].append(max(0.0, float(flow["end_time"]) - float(flow["start_time"])))
        values_by_field["median_ipt"].append(_tfusion_median(flow["ipts"]))
    stats: dict[str, tuple[float, float]] = {}
    for field, values in values_by_field.items():
        if not values:
            values = [0.0]
        stats[field] = (min(values), max(values))
    return stats


def _tfusion_scale(value: float, lo: float, hi: float) -> float:
    if np.isclose(lo, hi):
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _tfusion_finalize_flows(raw_flows: list[dict[str, Any]], stats: dict[str, tuple[float, float]], segment_len: int) -> list[dict[str, Any]]:
    flows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_flows):
        packets = raw["packets"]
        packet_count = max(1, len(packets))
        byte_count = max(0, int(raw["bytes"]))
        duration = max(0.0, float(raw["end_time"]) - float(raw["start_time"]))
        median_ipt = _tfusion_median(raw["ipts"])
        proto = int(float(raw["key"][4] or 0))
        src_port = int(float(raw["key"][2] or 0))
        dst_port = int(float(raw["key"][3] or 0))
        seq = []
        for packet in packets[:segment_len]:
            payload_length = max(0, int(packet.payload_length))
            length_token = min(4499, max(1, payload_length * 3 + min(max(proto, 0), 4499)))
            if duration > 0:
                time_token = min(199, max(0, int(199 * (float(packet.timestamp) - float(raw["start_time"])) / duration)))
            else:
                time_token = 0
            seq.append([length_token, time_token])
        while len(seq) < segment_len:
            seq.append([0, 0])
        scaled_packets = _tfusion_scale(float(packet_count), *stats["packets"])
        scaled_bytes = _tfusion_scale(float(byte_count), *stats["bytes"])
        scaled_duration = _tfusion_scale(duration, *stats["duration"])
        scaled_median_ipt = _tfusion_scale(median_ipt, *stats["median_ipt"])
        tcp = 1.0 if proto == 6 else 0.0
        udp = 1.0 if proto == 17 else 0.0
        icmp = 1.0 if proto == 1 else 0.0
        forward_ratio = float(raw["forward_bytes"] / byte_count) if byte_count else 0.0
        reverse_ratio = float(raw["reverse_bytes"] / byte_count) if byte_count else 0.0
        label = _tfusion_flow_label(raw)
        flows.append(
            {
                "id": raw["id"],
                "source": raw["source"],
                "source_file": raw.get("source_file", raw["source"]),
                "key": list(raw["key"]),
                "flow_id": f"{raw['source']}_flow_{index}",
                "seq_feature": seq,
                "nonseq_feature": [
                    scaled_packets,
                    scaled_bytes,
                    scaled_duration,
                    scaled_median_ipt,
                    min(max(proto, 0) / 255.0, 1.0),
                    min(max(src_port, dst_port) / 65535.0, 1.0),
                ],
                "interact_feature": [
                    min(src_port / 65535.0, 1.0),
                    min(dst_port / 65535.0, 1.0),
                    tcp,
                    udp,
                    icmp,
                    1.0 if dst_port in {80, 443, 53, 22, 23, 8080} or src_port in {80, 443, 53, 22, 23, 8080} else 0.0,
                    scaled_packets,
                    scaled_bytes,
                    scaled_duration,
                    scaled_median_ipt,
                    forward_ratio,
                    reverse_ratio,
                ],
                "len": packet_count,
                "label": label,
            }
        )
    return flows


def _tfusion_ant_metadata_from_source(source_file: object) -> dict[str, str]:
    parts = Path(str(source_file)).parts
    for split in ("train", "val", "test"):
        if split not in parts:
            continue
        index = parts.index(split)
        if index >= 1 and parts[index - 1] == "ANT" and len(parts) > index + 3:
            return {
                "ANT_SPLIT": split,
                "ANT_ENGINE": parts[index + 1],
                "ANT_TASK_FAMILY": parts[index + 2],
                "ANT_EPISODE": parts[index + 3],
            }
    return {"ANT_SPLIT": "", "ANT_ENGINE": "", "ANT_TASK_FAMILY": "", "ANT_EPISODE": ""}


def _tfusion_write_row_metadata(flows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ROW_INDEX",
                "TRUE_LABEL",
                "SOURCE_DATASET",
                "SOURCE_FILE",
                "SRC_IP",
                "SRC_PORT",
                "DST_IP",
                "DST_PORT",
                "PROTOCOL",
                "ANT_SPLIT",
                "ANT_ENGINE",
                "ANT_TASK_FAMILY",
                "ANT_EPISODE",
            ]
        )
        for index, flow in enumerate(flows):
            source = str(flow.get("source", ""))
            source_file = str(flow.get("source_file", source))
            key = list(flow.get("key") or ["", "", "", "", ""])
            if len(key) < 5:
                key.extend([""] * (5 - len(key)))
            ant = _tfusion_ant_metadata_from_source(source_file)
            writer.writerow(
                [
                    index,
                    str(flow.get("label", "")).strip(),
                    source,
                    source_file,
                    key[0],
                    key[2],
                    key[1],
                    key[3],
                    key[4],
                    ant["ANT_SPLIT"],
                    ant["ANT_ENGINE"],
                    ant["ANT_TASK_FAMILY"],
                    ant["ANT_EPISODE"],
                ]
            )


def _tfusion_write_native_json(path: Path, flows: list[dict[str, Any]]) -> dict[str, int]:
    payload = {
        "benign": [{k: v for k, v in flow.items() if k != "label"} for flow in flows if flow["label"] == "benign"],
        "attack": [{k: v for k, v in flow.items() if k != "label"} for flow in flows if flow["label"] != "benign"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"benign": len(payload["benign"]), "malicious": len(payload["attack"])}


def _tfusion_arrays_from_flows(flows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "seq": np.asarray([flow["seq_feature"] for flow in flows], dtype=np.float32),
        "nonseq": np.asarray([flow["nonseq_feature"] for flow in flows], dtype=np.float32),
        "interact": np.asarray([flow["interact_feature"] for flow in flows], dtype=np.float32),
        "sizes": np.asarray([int(flow["len"]) for flow in flows], dtype=np.int32),
        "labels": np.asarray([_tfusion_label_to_int(flow["label"]) for flow in flows], dtype=np.int32),
    }


def _tfusion_tensors_from_arrays(arrays: dict[str, np.ndarray]):
    import torch

    return (
        torch.as_tensor(arrays["seq"], dtype=torch.float32),
        torch.as_tensor(arrays["nonseq"], dtype=torch.float32),
        torch.as_tensor(arrays["interact"], dtype=torch.float32),
        arrays["sizes"].astype(np.int32, copy=False),
        arrays["labels"].astype(np.int32, copy=False),
    )


def _tfusion_tensors_from_flows(flows: list[dict[str, Any]]):
    return _tfusion_tensors_from_arrays(_tfusion_arrays_from_flows(flows))


def _tfusion_stats_payload(stats: dict[str, tuple[float, float]]) -> dict[str, list[float]]:
    return {key: [float(value[0]), float(value[1])] for key, value in sorted(stats.items())}


def _tfusion_final_cache_payload(
    raw_meta: dict[str, Any],
    stats: dict[str, tuple[float, float]],
    segment_len: int,
    role: str,
) -> dict[str, Any]:
    return {
        "version": 2,
        "role": role,
        "raw_key": (raw_meta.get("cache") or {}).get("key"),
        "flows": int(raw_meta.get("flows", 0)),
        "stats": _tfusion_stats_payload(stats),
        "segment_len": int(segment_len),
    }


def _tfusion_load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _tfusion_prepare_finalized_cached(
    raw_flows: list[dict[str, Any]],
    raw_meta: dict[str, Any],
    stats: dict[str, tuple[float, float]],
    segment_len: int,
    alg: dict[str, Any],
    role: str,
    native_json_path: Path,
) -> tuple[list[dict[str, Any]], tuple[Any, Any, Any, np.ndarray, np.ndarray], dict[str, int], dict[str, Any]]:
    payload = _tfusion_final_cache_payload(raw_meta, stats, segment_len, role)
    cache_key = _tfusion_cache_key(payload)
    cache_dir = _tfusion_cache_root(alg) / "final" / cache_key
    flows_path = cache_dir / "flows.pkl"
    tensors_path = cache_dir / "tensors.npz"
    native_cache_path = cache_dir / "native.json"
    counts_path = cache_dir / "counts.json"
    if (
        _tfusion_cache_enabled(alg)
        and flows_path.exists()
        and tensors_path.exists()
        and native_cache_path.exists()
        and counts_path.exists()
    ):
        flows = _tfusion_pickle_load(flows_path)
        tensors = _tfusion_tensors_from_arrays(_tfusion_load_npz(tensors_path))
        native_json_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(native_cache_path, native_json_path)
        counts = json.loads(counts_path.read_text(encoding="utf-8"))
        meta = {"hit": True, "key": cache_key, "dir": rel(cache_dir), "native_json": rel(native_cache_path)}
        return flows, tensors, counts, meta

    flows = _tfusion_finalize_flows(raw_flows, stats, segment_len)
    counts = _tfusion_write_native_json(native_json_path, flows)
    arrays = _tfusion_arrays_from_flows(flows)
    tensors = _tfusion_tensors_from_arrays(arrays)
    if _tfusion_cache_enabled(alg):
        cache_dir.mkdir(parents=True, exist_ok=True)
        _tfusion_pickle_dump_atomic(flows_path, flows)
        tmp_npz = tensors_path.with_name(f"{tensors_path.name}.tmp.{os.getpid()}")
        np.savez(tmp_npz, **arrays)
        tmp_npz_path = tmp_npz if tmp_npz.exists() else tmp_npz.with_suffix(tmp_npz.suffix + ".npz")
        tmp_npz_path.replace(tensors_path)
        shutil.copyfile(native_json_path, native_cache_path)
        write_json(counts_path, counts)
        write_json(cache_dir / "payload.json", payload)
    meta = {"hit": False, "key": cache_key, "dir": rel(cache_dir), "native_json": rel(native_cache_path)}
    return flows, tensors, counts, meta


def _tfusion_embeddings(model, tensors: tuple[Any, Any, Any, np.ndarray, np.ndarray], device: str, batch_size: int) -> np.ndarray:
    import torch

    seq, nonseq, interact, _sizes, _labels = tensors
    embeddings = []
    with torch.no_grad():
        for start in range(0, seq.size(0), batch_size):
            batch = (
                seq[start : start + batch_size].to(device),
                nonseq[start : start + batch_size].to(device),
                interact[start : start + batch_size].to(device),
            )
            embeddings.append(model(batch).detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 100), dtype=np.float32)


def _tfusion_soft_from_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if np.isclose(lo, hi):
        attack = np.zeros_like(scores, dtype=np.float32)
    else:
        attack = np.clip((scores - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return np.stack([1.0 - attack, attack], axis=1)


def _tfusion_build_contrast_batches(flows: list[dict[str, Any]], batch_size: int, seed: int):
    import random
    import torch

    if not flows:
        return None
    rng = random.Random(seed)
    batch_size = max(2, int(batch_size))
    if batch_size % 2:
        batch_size += 1
    half = batch_size // 2
    by_src: dict[str, list[int]] = {}
    by_dst: dict[str, list[int]] = {}
    for index, flow in enumerate(flows):
        src, dst = str(flow["id"][0]), str(flow["id"][1])
        by_src.setdefault(src, []).append(index)
        by_dst.setdefault(dst, []).append(index)
    eligible_groups = [
        (group, set(group))
        for group in list(by_src.values()) + list(by_dst.values())
        if len(group) >= half
    ]
    batch_count = max(1, len(flows) // batch_size)
    batches: list[list[dict[str, Any]]] = []
    all_indices = list(range(len(flows)))
    for batch_index in range(batch_count):
        if eligible_groups:
            same_pool, same_set = eligible_groups[batch_index % len(eligible_groups)]
            same = rng.sample(same_pool, half)
            if len(all_indices) - len(same_set) >= half:
                different = []
                while len(different) < half:
                    candidate = rng.choice(all_indices)
                    if candidate not in same_set:
                        different.append(candidate)
            else:
                different = [rng.choice(all_indices) for _ in range(half)]
            indices = same + different
        else:
            indices = [rng.choice(all_indices) for _ in range(batch_size)]
        batches.append([flows[idx] for idx in indices])
    return (
        torch.as_tensor([[flow["seq_feature"] for flow in batch] for batch in batches], dtype=torch.float32),
        torch.as_tensor([[flow["nonseq_feature"] for flow in batch] for batch in batches], dtype=torch.float32),
        torch.as_tensor([[flow["interact_feature"] for flow in batch] for batch in batches], dtype=torch.float32),
    )


def _tfusion_stage1_model(device: str):
    import torch.nn as nn
    from model import TFusion

    n_hidden = 100
    n_project = 128
    model = TFusion().to(device)
    model_project = nn.Sequential(
        nn.Linear(n_hidden, n_project),
        nn.ReLU(),
        nn.Linear(n_project, n_project),
    ).to(device)
    return model, nn.Sequential(model, model_project)


def _tfusion_stats_from_preprocess(preprocess: dict[str, Any]) -> dict[str, tuple[float, float]]:
    stats = preprocess.get("stats", {})
    return {
        str(key): (float(value[0]), float(value[1]))
        for key, value in stats.items()
        if isinstance(value, (list, tuple)) and len(value) == 2
    }


def _tfusion_supervised_soft(detector, features: np.ndarray, hard: np.ndarray) -> np.ndarray:
    if hasattr(detector.model, "predict_proba"):
        soft = np.asarray(detector.model.predict_proba(features), dtype=np.float32)
        if soft.shape[1] == 1:
            if list(detector.model.classes_)[0] == 0:
                soft = np.column_stack([soft[:, 0], np.zeros(soft.shape[0], dtype=np.float32)])
            else:
                soft = np.column_stack([np.zeros(soft.shape[0], dtype=np.float32), soft[:, 0]])
        return soft.astype(np.float32)
    return np.stack([1.0 - hard, hard], axis=1).astype(np.float32)


def _tfusion_detector_outputs(
    detector,
    detector_name: str,
    features: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_scores = np.asarray(detector.test(features), dtype=np.float32)
    if detector_name in {"rforest", "knn", "bayes", "logistic"}:
        hard = (raw_scores > 0.5).astype(np.int32)
        soft = _tfusion_supervised_soft(detector, features, hard)
    else:
        hard = (raw_scores >= threshold).astype(np.int32)
        soft = _tfusion_soft_from_scores(raw_scores)
    return raw_scores, hard, soft


def _run_tfusion_eval_only(prepared: PreparedData, config: dict[str, Any], run_dir: Path) -> PredictionFiles:
    import torch

    if not prepared.test_packets:
        raise BenchmarkError("TFusion eval-only requires native packet-level pcap plus packet-label test inputs")

    tfusion_root = resolve("Benchmark/2025-CCS-TFusion")
    if str(tfusion_root) not in sys.path:
        sys.path.insert(0, str(tfusion_root))
    from model.detection import DetectorFactory

    alg = config["algorithm_config"]
    bundle_path = resolve(alg.get("tfusion_model_bundle") or alg.get("eval_bundle_path"))
    if not bundle_path.exists():
        raise FileNotFoundError(f"TFusion eval bundle not found: {bundle_path}")
    preprocess = json.loads(bundle_path.read_text(encoding="utf-8"))
    out_dir = run_dir / "algorithm_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    segment_len = int(preprocess.get("segment_len", alg.get("data_construct_param", {}).get("segment_len", 100)))
    infer_batch_size = int(alg.get("infer_batch_size", 256))
    stats = _tfusion_stats_from_preprocess(preprocess)
    required_stats = {"packets", "bytes", "duration", "median_ipt"}
    if not required_stats <= set(stats):
        raise BenchmarkError(f"TFusion eval bundle has incomplete stats: {bundle_path}")
    device_config = str(alg.get("device", "auto"))
    device = "cuda:0" if device_config == "auto" and torch.cuda.is_available() else device_config
    if device == "auto":
        device = "cpu"

    test_raw, test_source_meta = _tfusion_packet_sources_to_raw_flows(prepared.test_packets, alg, "test")
    if not test_raw:
        raise BenchmarkError("TFusion eval-only requires non-empty test native flow set")
    native_dir = out_dir / "native_input"
    test_flows, test_tensors, test_counts, test_final_cache = _tfusion_prepare_finalized_cached(
        test_raw, test_source_meta, stats, segment_len, alg, "test", native_dir / "test.json"
    )

    model, stage1 = _tfusion_stage1_model(device)
    stage1_checkpoint = resolve(preprocess["stage1_checkpoint"])
    if not stage1_checkpoint.exists():
        raise FileNotFoundError(f"TFusion stage1 checkpoint not found: {stage1_checkpoint}")
    stage1.load_state_dict(torch.load(stage1_checkpoint, map_location=device))
    stage1.eval()
    model.eval()
    test_features = _tfusion_embeddings(model, test_tensors, device, infer_batch_size)
    test_labels = test_tensors[-1]

    detector_name = str(preprocess.get("detector", alg.get("detection_model", "iforest")))
    detector = DetectorFactory.create(detector_name)
    detector_prefix = resolve(preprocess["detector_path_prefix"])
    detector.load(str(detector_prefix))
    threshold = float(preprocess.get("threshold", 0.5))
    raw_scores, hard, soft = _tfusion_detector_outputs(detector, detector_name, test_features, threshold)

    hard_path = out_dir / "hard_predictions.txt"
    labels_path = out_dir / "test_labels.txt"
    soft_path = out_dir / "soft_predictions.npy"
    hard_path.write_text("\n".join(str(int(value)) for value in hard) + "\n", encoding="utf-8")
    labels_path.write_text("\n".join(str(int(value)) for value in test_labels) + "\n", encoding="utf-8")
    np.save(soft_path, soft.astype(np.float32))
    _tfusion_write_row_metadata(test_flows, prepared.work_dir / "test_row_metadata.csv")
    shutil.copyfile(bundle_path, out_dir / "tfusion_preprocess.json")

    metadata = {
        "algorithm": "TFusion",
        "native_input_runner": True,
        "eval_only": True,
        "input_grain": "native_flow_with_packet_sequence",
        "detector": detector_name,
        "detector_params": preprocess.get("detector_params", {}),
        "threshold": threshold,
        "threshold_metadata": preprocess.get("threshold_metadata", {}),
        "segment_len": segment_len,
        "stage1_checkpoint": rel(stage1_checkpoint),
        "detector_path_prefix": rel(detector_prefix),
        "preprocess_path": rel(bundle_path),
        "native_json": {"train": None, "validation": None, "test": rel(native_dir / "test.json")},
        "test_sources": test_source_meta,
        "cache": {"raw": {"test": test_source_meta.get("cache")}, "finalized": {"test": test_final_cache}},
        "test_flows": int(len(test_flows)),
        "native_label_counts": {"test": test_counts},
        "test_label_counts": dict(sorted(Counter(str(int(x)) for x in test_labels).items())),
        "raw_score_min": float(np.min(raw_scores)) if raw_scores.size else None,
        "raw_score_max": float(np.max(raw_scores)) if raw_scores.size else None,
    }
    write_json(out_dir / "metrics.json", metadata)
    return PredictionFiles(hard_path, labels_path, soft_path, prepared.label_mapping_path, out_dir, metadata)


def _tfusion_pretrain_stage1(
    stage1,
    batches: tuple[Any, Any, Any] | None,
    device: str,
    checkpoint_path: Path,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
) -> dict[str, Any]:
    import torch

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if batches is None:
        torch.save(stage1.state_dict(), checkpoint_path)
        return {"pretrained": False, "reason": "no benign pretrain flows", "epochs": 0}
    batch_seq, batch_nonseq, batch_interact = batches
    optimizer = torch.optim.Adam(stage1.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def contrastive_loss(x_predict):
        same_line = x_predict.size(0) // 2
        loss_val = torch.zeros(1, device=device)
        eps = 1e-12
        for index in range(max(0, same_line - 1)):
            anchor = x_predict[index]
            positive = x_predict[index + 1]
            negative = x_predict[index + same_line]
            pos_sim = torch.exp(torch.dot(anchor, positive) / (anchor.norm() * positive.norm() + eps))
            neg_sim = torch.exp(torch.dot(anchor, negative) / (anchor.norm() * negative.norm() + eps))
            loss_val += -torch.log(pos_sim / (pos_sim + neg_sim + eps))
        return loss_val / max(1, same_line)

    losses: list[float] = []
    for _epoch in range(max(1, int(epochs))):
        stage1.train()
        epoch_loss = 0.0
        for index in range(batch_seq.size(0)):
            batch = (
                batch_seq[index].to(device),
                batch_nonseq[index].to(device),
                batch_interact[index].to(device),
            )
            optimizer.zero_grad()
            mapped = stage1(batch)
            loss = contrastive_loss(mapped)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
        losses.append(epoch_loss / max(1, int(batch_seq.size(0))))
    stage1.eval()
    torch.save(stage1.state_dict(), checkpoint_path)
    return {"pretrained": True, "epochs": int(max(1, epochs)), "losses": losses}


def run_tfusion(prepared: PreparedData, config: dict[str, Any], run_dir: Path) -> PredictionFiles:
    import torch

    alg = config["algorithm_config"]
    if alg.get("tfusion_model_bundle") or alg.get("eval_bundle_path"):
        return _run_tfusion_eval_only(prepared, config, run_dir)

    if not prepared.training_packets or not prepared.test_packets:
        raise BenchmarkError("TFusion now requires native packet-level pcap plus packet-label inputs")

    tfusion_root = resolve("Benchmark/2025-CCS-TFusion")
    if str(tfusion_root) not in sys.path:
        sys.path.insert(0, str(tfusion_root))
    from model.detection import DetectorFactory

    out_dir = run_dir / "algorithm_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    segment_len = int(alg.get("data_construct_param", {}).get("segment_len", 100))
    infer_batch_size = int(alg.get("infer_batch_size", 256))
    device_config = str(alg.get("device", "auto"))
    device = "cuda:0" if device_config == "auto" and torch.cuda.is_available() else device_config
    if device == "auto":
        device = "cpu"

    train_raw, train_source_meta = _tfusion_packet_sources_to_raw_flows(prepared.training_packets, alg, "train")
    dev_raw: list[dict[str, Any]] = []
    dev_source_meta: dict[str, Any] = {"sources": [], "flows": 0}
    if prepared.validation_packets:
        dev_raw, dev_source_meta = _tfusion_packet_sources_to_raw_flows(prepared.validation_packets, alg, "validation")
    test_raw, test_source_meta = _tfusion_packet_sources_to_raw_flows(prepared.test_packets, alg, "test")
    if not train_raw or not test_raw:
        raise BenchmarkError("TFusion requires non-empty train and test native flow sets")
    stats = _tfusion_feature_stats(train_raw)
    native_dir = out_dir / "native_input"
    train_flows, train_tensors, train_counts, train_final_cache = _tfusion_prepare_finalized_cached(
        train_raw, train_source_meta, stats, segment_len, alg, "train", native_dir / "train.json"
    )
    if dev_raw:
        dev_flows, dev_tensors, dev_counts, dev_final_cache = _tfusion_prepare_finalized_cached(
            dev_raw, dev_source_meta, stats, segment_len, alg, "validation", native_dir / "validation.json"
        )
    else:
        dev_flows = []
        dev_tensors = None
        dev_counts = {}
        dev_final_cache = {}
    test_flows, test_tensors, test_counts, test_final_cache = _tfusion_prepare_finalized_cached(
        test_raw, test_source_meta, stats, segment_len, alg, "test", native_dir / "test.json"
    )

    model, stage1 = _tfusion_stage1_model(device)
    training_cfg = alg.get("training", alg.get("train_param", {})) or {}
    checkpoint_path = (
        resolve(alg["stage1_model_path"])
        if alg.get("stage1_model_path")
        else resolve(training_cfg["checkpoint_dir"]) / "stage1_model.pt"
        if training_cfg.get("checkpoint_dir")
        else out_dir / "stage1_model.pt"
    )
    if alg.get("stage1_model_path"):
        stage1.load_state_dict(torch.load(checkpoint_path, map_location=device))
        stage1_metadata = {"pretrained": True, "loaded_existing_checkpoint": rel(checkpoint_path), "epochs": 0}
    else:
        benign_pretrain_flows = [flow for flow in train_flows if flow["label"] == "benign"]
        batches = _tfusion_build_contrast_batches(
            benign_pretrain_flows,
            int(training_cfg.get("contrast_batch_size", 50)),
            int(training_cfg.get("seed", 777)),
        )
        stage1_metadata = _tfusion_pretrain_stage1(
            stage1,
            batches,
            device,
            checkpoint_path,
            float(training_cfg.get("learning_rate", training_cfg.get("lr", 0.001))),
            float(training_cfg.get("weight_decay", training_cfg.get("wd", 1e-4))),
            int(training_cfg.get("max_epoch", training_cfg.get("num_epoch", 15))),
        )
        stage1.load_state_dict(torch.load(checkpoint_path, map_location=device))
    stage1.eval()
    model.eval()
    train_features = _tfusion_embeddings(model, train_tensors, device, infer_batch_size)
    dev_features = _tfusion_embeddings(model, dev_tensors, device, infer_batch_size) if dev_tensors is not None else None
    test_features = _tfusion_embeddings(model, test_tensors, device, infer_batch_size)
    train_labels = train_tensors[-1]
    dev_labels = dev_tensors[-1] if dev_tensors is not None else None
    test_labels = test_tensors[-1]

    detector_name = str(alg.get("detection_model", "iforest"))
    detector = DetectorFactory.create(detector_name)
    detector_params = alg.get("detector_params", {}) or {}
    if detector_name == "kmeans":
        benign_count = int(np.sum(train_labels == 0))
        requested = int(detector_params.get("n_clusters", getattr(detector.model, "n_clusters", 10)))
        detector_params["n_clusters"] = max(1, min(requested, max(1, benign_count)))
    if detector_params and hasattr(detector.model, "set_params"):
        detector.model.set_params(**detector_params)
    detector.train(train_features, train_labels)
    raw_scores = np.asarray(detector.test(test_features), dtype=np.float32)
    validation_soft = None

    if detector_name in {"rforest", "knn", "bayes", "logistic"}:
        hard = (raw_scores > 0.5).astype(np.int32)
        if hasattr(detector.model, "predict_proba"):
            soft = np.asarray(detector.model.predict_proba(test_features), dtype=np.float32)
            if soft.shape[1] == 1:
                if list(detector.model.classes_)[0] == 0:
                    soft = np.column_stack([soft[:, 0], np.zeros(soft.shape[0], dtype=np.float32)])
                else:
                    soft = np.column_stack([np.zeros(soft.shape[0], dtype=np.float32), soft[:, 0]])
        else:
            soft = np.stack([1.0 - hard, hard], axis=1).astype(np.float32)
        if dev_features is not None and dev_labels is not None and len(dev_labels):
            if hasattr(detector.model, "predict_proba"):
                validation_soft = np.asarray(detector.model.predict_proba(dev_features), dtype=np.float32)
                if validation_soft.shape[1] == 1:
                    if list(detector.model.classes_)[0] == 0:
                        validation_soft = np.column_stack(
                            [validation_soft[:, 0], np.zeros(validation_soft.shape[0], dtype=np.float32)]
                        )
                    else:
                        validation_soft = np.column_stack(
                            [np.zeros(validation_soft.shape[0], dtype=np.float32), validation_soft[:, 0]]
                        )
            else:
                dev_hard = (np.asarray(detector.test(dev_features), dtype=np.float32) > 0.5).astype(np.int32)
                validation_soft = np.stack([1.0 - dev_hard, dev_hard], axis=1).astype(np.float32)
        threshold = 0.5
        threshold_metadata = {"threshold_source": "native_hard_labels", "threshold": threshold}
    else:
        if dev_features is not None and dev_labels is not None and len(dev_labels):
            dev_scores = np.asarray(detector.test(dev_features), dtype=np.float32)
            validation_soft = _tfusion_soft_from_scores(dev_scores)
            selected = _select_binary_threshold(dev_scores, np.asarray(dev_labels).astype(str).tolist())
            threshold = float(selected["threshold"])
            threshold_metadata = {
                "threshold_source": "validation_threshold_scan",
                "selection_metric": "malicious_f1",
                "threshold": threshold,
                "validation_metrics_at_selected_threshold": selected,
                "test_metrics_at_selected_threshold": _binary_threshold_metrics(raw_scores, np.asarray(test_labels).astype(str).tolist(), threshold),
            }
        else:
            train_scores = np.asarray(detector.test(train_features), dtype=np.float32)
            benign_train_scores = train_scores[train_labels == 0]
            if benign_train_scores.size == 0:
                benign_train_scores = train_scores
            threshold_config = alg.get("threshold", {})
            method = str(threshold_config.get("method", "train_quantile"))
            if method == "train_max":
                threshold = float(np.max(benign_train_scores))
            else:
                threshold = float(np.quantile(benign_train_scores, float(threshold_config.get("quantile", 0.95))))
            threshold *= float(threshold_config.get("beta", 1.0))
            threshold_metadata = {"threshold_source": method, "threshold": threshold}
        hard = (raw_scores >= threshold).astype(np.int32)
        soft = _tfusion_soft_from_scores(raw_scores)

    hard_path = out_dir / "hard_predictions.txt"
    labels_path = out_dir / "test_labels.txt"
    soft_path = out_dir / "soft_predictions.npy"
    validation_soft_path = out_dir / "validation_soft_predictions.npy"
    validation_labels_path = out_dir / "validation_labels.txt"
    hard_path.write_text("\n".join(str(int(value)) for value in hard) + "\n", encoding="utf-8")
    labels_path.write_text("\n".join(str(int(value)) for value in test_labels) + "\n", encoding="utf-8")
    np.save(soft_path, soft.astype(np.float32))
    if validation_soft is not None and dev_labels is not None:
        validation_labels_path.write_text("\n".join(str(int(value)) for value in dev_labels) + "\n", encoding="utf-8")
        np.save(validation_soft_path, validation_soft.astype(np.float32))
    _tfusion_write_row_metadata(test_flows, prepared.work_dir / "test_row_metadata.csv")
    metadata = {
        "algorithm": "TFusion",
        "native_input_runner": True,
        "input_grain": "native_flow_with_packet_sequence",
        "detector": detector_name,
        "detector_params": detector_params,
        "threshold": threshold,
        **threshold_metadata,
        "segment_len": segment_len,
        "stage1_checkpoint": rel(checkpoint_path),
        "stage1": stage1_metadata,
        "native_json": {
            "train": rel(native_dir / "train.json"),
            "validation": rel(native_dir / "validation.json") if dev_flows else None,
            "test": rel(native_dir / "test.json"),
        },
        "train_sources": train_source_meta,
        "validation_sources": dev_source_meta,
        "test_sources": test_source_meta,
        "cache": {
            "raw": {
                "train": train_source_meta.get("cache"),
                "validation": dev_source_meta.get("cache"),
                "test": test_source_meta.get("cache"),
            },
            "finalized": {
                "train": train_final_cache,
                "validation": dev_final_cache,
                "test": test_final_cache,
            },
        },
        "train_flows": int(len(train_flows)),
        "validation_flows": int(len(dev_flows)),
        "test_flows": int(len(test_flows)),
        "native_label_counts": {"train": train_counts, "validation": dev_counts, "test": test_counts},
        "train_label_counts": dict(sorted(Counter(str(int(x)) for x in train_labels).items())),
        "validation_label_counts": dict(sorted(Counter(str(int(x)) for x in dev_labels).items())) if dev_labels is not None else {},
        "test_label_counts": dict(sorted(Counter(str(int(x)) for x in test_labels).items())),
    }
    if validation_soft is not None:
        metadata["validation_soft_predictions"] = rel(validation_soft_path)
        metadata["validation_labels"] = rel(validation_labels_path)
    detector_path = out_dir / "tfusion_detector"
    detector.save(str(detector_path))
    preprocess_path = out_dir / "tfusion_preprocess.json"
    write_json(
        preprocess_path,
        {
            "stats": {key: [float(value[0]), float(value[1])] for key, value in stats.items()},
            "segment_len": segment_len,
            "detector": detector_name,
            "detector_params": detector_params,
            "threshold": float(threshold),
            "threshold_metadata": threshold_metadata,
            "stage1_checkpoint": rel(checkpoint_path),
            "detector_path_prefix": rel(detector_path),
        },
    )
    metadata["detector_path_prefix"] = rel(detector_path)
    metadata["preprocess_path"] = rel(preprocess_path)
    write_json(out_dir / "metrics.json", metadata)
    if threshold_metadata.get("threshold_source") == "validation_threshold_scan":
        write_json(out_dir / "threshold_metrics.json", threshold_metadata)
    return PredictionFiles(hard_path, labels_path, soft_path, prepared.label_mapping_path, out_dir, metadata)


def run_algorithm(prepared: PreparedData, config: dict[str, Any], run_dir: Path) -> PredictionFiles:
    if prepared.algorithm == "TrafficFormer":
        return run_trafficformer(prepared, config, run_dir)
    if prepared.algorithm == "NetMamba":
        return run_netmamba(prepared, config, run_dir)
    if prepared.algorithm == "TFusion":
        return run_tfusion(prepared, config, run_dir)
    if prepared.algorithm == "Kitsune":
        return run_kitsune(prepared, config, run_dir)
    if prepared.algorithm == "FlowLens":
        return run_flowlens(prepared, config, run_dir)
    raise BenchmarkError(f"Unsupported algorithm: {prepared.algorithm}")


def write_prediction_manifest(predictions: PredictionFiles, path: Path) -> None:
    native_metrics = []
    for filename in ["native_metrics.json", "metrics.json", "threshold_metrics.json"]:
        candidate = predictions.output_dir / filename
        if candidate.exists():
            native_metrics.append(rel(candidate))
    write_json(
        path,
        {
            "evaluation_contract": {
                "primary_prediction_input": "soft_predictions.npy",
                "ground_truth_input": "test_labels.txt",
                "task_type": "single_label_classification",
                "hard_predictions": "derived by Benchmark/evaluation.py via argmax over soft labels when soft_predictions.npy exists",
                "native_metrics": "preserved separately and not used as the primary comparable metric source",
            },
            "hard_predictions": rel(predictions.hard_predictions),
            "test_labels": rel(predictions.test_labels),
            "soft_predictions": rel(predictions.soft_predictions) if predictions.soft_predictions else None,
            "label_mapping": rel(predictions.label_mapping) if predictions.label_mapping else None,
            "native_metrics": native_metrics,
            "output_dir": rel(predictions.output_dir),
            "metadata": predictions.metadata,
        },
    )
