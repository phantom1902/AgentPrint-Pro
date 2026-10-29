#!/usr/bin/env python3
"""Run a small ANT semantic-label smoke test for WFlib WF baselines."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "Benchmark"
WFLIB_ROOT = BENCHMARK / "2024_CCS_WFlib_Early_Stage_WF"

sys.path.insert(0, str(BENCHMARK))
sys.path.insert(0, str(WFLIB_ROOT))
sys.path.insert(0, str(ROOT))

from ant_pcap_native import discover_episodes, read_pcap_packets  # noqa: E402
from Benchmark.evaluation import evaluate_predictions  # noqa: E402
from Benchmark.hyperparameters import wflib_hyperparameters  # noqa: E402
from WFlib.tools import data_processor  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    name: str
    class_name: str
    source_work: str
    feature: str
    seq_len: int
    raw_seq_len: int
    precompute_seq_len: int | None = None
    num_tab: int | None = None
    train_epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    optimizer: str | None = None
    loss: str | None = None
    unsupported_reason: str | None = None


CLASS_NAMES = {
    "DF": "DF",
    "AWF": "AWF",
    "TF": "TF",
    "TikTok": "TikTok",
    "VarCNN": "VarCNN",
    "RF": "RF",
    "ARES": "ARES",
    "NetCLR": "NetCLR",
    "TMWF": "TMWF",
    "BAPM": "BAPM",
    "Holmes": "Holmes",
}

SOURCE_WORKS = {
    "DF": "2018_CCS_DF",
    "AWF": "2018_NDSS_AWF",
    "TF": "2019_CCS_TF",
    "TikTok": "2019_PETS_Tik-Tok",
    "VarCNN": "2019_PETS_Var-CNN",
    "RF": "2023_USENIX_RF",
    "ARES": "2023_SP_ARES",
    "NetCLR": "2023_CCS_NetCLR",
    "TMWF": "2023_CCS_TMWF",
    "BAPM": "WFlib_BAPM",
    "Holmes": "WFlib_Holmes",
}


def build_model_specs(hyperparameters_path: Path | None = None) -> dict[str, ModelSpec]:
    hp = wflib_hyperparameters(hyperparameters_path)
    specs: dict[str, ModelSpec] = {}
    for name, item in hp["submodels"].items():
        status = str(item.get("current_benchmark_status", ""))
        unsupported = status if "unsupported" in status.lower() else None
        specs[name] = ModelSpec(
            name=name,
            class_name=CLASS_NAMES.get(name, name),
            source_work=SOURCE_WORKS.get(name, item.get("source_work", f"WFlib_{name}")),
            feature=item["feature"],
            seq_len=int(item["seq_len"]),
            raw_seq_len=int(item["raw_seq_len"]),
            precompute_seq_len=int(item["precompute_seq_len"]) if item.get("precompute_seq_len") is not None else None,
            num_tab=int(item["num_tab"]) if item.get("num_tab") is not None else None,
            train_epochs=int(item["train_epochs"]) if item.get("train_epochs") is not None else None,
            batch_size=int(item["batch_size"]) if item.get("batch_size") is not None else None,
            learning_rate=float(item["learning_rate"]) if item.get("learning_rate") is not None else None,
            optimizer=item.get("optimizer"),
            loss=item.get("loss"),
            unsupported_reason=unsupported,
        )
    return specs


MODEL_SPECS: dict[str, ModelSpec] = build_model_specs()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_sequence(seq: list[float], seq_len: int) -> np.ndarray:
    out = np.zeros(seq_len, dtype=np.float32)
    if seq:
        out[: min(len(seq), seq_len)] = np.asarray(seq[:seq_len], dtype=np.float32)
    return out


def episode_sequence(pcap_path: Path, seq_len: int, protocols: set[int] | None) -> np.ndarray | None:
    packets = read_pcap_packets(pcap_path, protocols=protocols, max_records=seq_len)
    if not packets:
        return None
    start = packets[0].timestamp
    first_src = packets[0].src_ip
    seq = []
    for packet in packets:
        direction = 1.0 if packet.src_ip == first_src else -1.0
        seq.append(direction * max(packet.timestamp - start, 0.0))
    return fit_sequence(seq, seq_len)


def proto_set(value: str) -> set[int] | None:
    if value == "all":
        return None
    mapping = {"tcp": 6, "udp": 17, "icmp": 1, "icmpv6": 58}
    return {mapping[item.strip()] for item in value.split(",") if item.strip()}


def select_episodes(data_root: Path, per_engine_label: int, splits: list[str], label_field: str) -> dict[str, list[Any]]:
    all_eps = discover_episodes(data_root, splits=splits)
    by_split = {split: [ep for ep in all_eps if ep.split == split] for split in splits}
    common_labels = set.intersection(*[{ep.labels.value(label_field) for ep in eps} for eps in by_split.values()])
    selected: dict[str, list[Any]] = {}
    for split, episodes in by_split.items():
        grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
        for ep in episodes:
            label = ep.labels.value(label_field)
            if label in common_labels:
                grouped[(label, ep.engine)].append(ep)
        rows = []
        for key in sorted(grouped):
            rows.extend(grouped[key][:per_engine_label])
        selected[split] = rows
    return selected


def write_label_mapping(path: Path, label2id: dict[str, int]) -> dict[str, str]:
    id_to_label = {str(index): label for label, index in sorted(label2id.items(), key=lambda item: item[1])}
    payload = {"label_to_id": label2id, "id_to_label": id_to_label}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return id_to_label


def write_raw_npz(
    selected: dict[str, list[Any]],
    out_dir: Path,
    raw_seq_len: int,
    protocols: set[int] | None,
    label_field: str,
) -> dict[str, Any]:
    labels = sorted({ep.labels.value(label_field) for episodes in selected.values() for ep in episodes})
    label2id = {label: index for index, label in enumerate(labels)}
    manifest: dict[str, Any] = {
        "label_field": label_field,
        "engine_policy": f"same {label_field} across all engines is one class",
        "trace_value": "direction * relative_time_seconds",
        "direction_rule": "first packet source IP in the episode is positive; the opposite direction is negative",
        "protocols": "tcp" if protocols == {6} else ("all_csv_protocols" if protocols is None else sorted(protocols)),
        "raw_seq_len": raw_seq_len,
        "splits": {},
    }
    for split, episodes in selected.items():
        X, y = [], []
        label_counts = Counter()
        scenario_counts = Counter()
        engines = Counter()
        sources = []
        for ep in episodes:
            seq = episode_sequence(ep.pcap_path, raw_seq_len, protocols)
            if seq is None:
                continue
            label = ep.labels.value(label_field)
            X.append(seq)
            y.append(label2id[label])
            label_counts[label] += 1
            scenario_counts[ep.scenario] += 1
            engines[ep.engine] += 1
            sources.append(
                {
                    "pcap": str(ep.pcap_path.resolve()),
                    "engine": ep.engine,
                    "scenario": ep.scenario,
                    "episode": ep.episode_id,
                    "label": label,
                }
            )
        npz_path = out_dir / f"{split}.npz"
        np.savez_compressed(npz_path, X=np.asarray(X, dtype=np.float32), y=np.asarray(y, dtype=np.int64))
        (out_dir / f"{split}_labels.txt").write_text(
            "\n".join(str(item) for item in y) + ("\n" if y else ""),
            encoding="utf-8",
        )
        manifest["splits"][split] = {
            "path": rel(npz_path),
            "samples": len(X),
            "counts_by_label": dict(sorted(label_counts.items())),
            "counts_by_scenario": dict(sorted(scenario_counts.items())),
            "counts_by_engine": dict(sorted(engines.items())),
            "sources": sources,
        }
    write_label_mapping(out_dir / "label_mapping.json", label2id)
    (out_dir / "raw_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_model_class(spec: ModelSpec):
    path = WFLIB_ROOT / "WFlib" / "models" / f"{spec.class_name}.py"
    module_name = f"wflib_smoke_{spec.class_name}_{time.time_ns()}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load model file: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, spec.class_name)


def instantiate_model(spec: ModelSpec, num_classes: int) -> nn.Module:
    cls = load_model_class(spec)
    if spec.num_tab is not None:
        return cls(num_classes=num_classes, num_tab=spec.num_tab)
    if spec.name == "NetCLR":
        return cls(out_dim=num_classes)
    return cls(num_classes=num_classes)


def precompute_feature(raw_npz: Path, spec: ModelSpec, out_npz: Path, workers: int) -> Path:
    if out_npz.exists():
        return out_npz
    data = np.load(raw_npz)
    X = data["X"].astype(np.float32, copy=True)
    y = data["y"]
    if spec.feature == "TAM":
        X = data_processor.length_align(X, spec.precompute_seq_len or spec.raw_seq_len)
        X = data_processor.extract_TAM(X, num_workers=workers)
    elif spec.feature == "TAF":
        X = data_processor.length_align(X, spec.precompute_seq_len or spec.raw_seq_len)
        X = data_processor.extract_TAF(X, num_workers=workers)
    elif spec.feature == "MTAF":
        X = data_processor.length_align(X, spec.precompute_seq_len or spec.raw_seq_len)
        X = data_processor.extract_MTAF(X, num_workers=workers)
    else:
        raise ValueError(f"{spec.name} does not require precompute for {spec.feature}")
    np.savez_compressed(out_npz, X=X.astype(np.float32), y=y.astype(np.int64))
    return out_npz


def tensors_for_split(raw_dir: Path, split: str, spec: ModelSpec, workers: int) -> tuple[torch.Tensor, torch.Tensor, Path]:
    raw_npz = raw_dir / f"{split}.npz"
    input_npz = raw_npz
    if spec.feature in {"TAM", "TAF", "MTAF"}:
        input_npz = raw_dir / f"{spec.feature.lower()}_{split}.npz"
        input_npz = precompute_feature(raw_npz, spec, input_npz, workers)
    X, y = data_processor.load_data(str(input_npz), spec.feature, spec.seq_len, num_tab=spec.num_tab or 1)
    return X, y, input_npz


def make_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, train: bool) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(X, y)
    effective = min(batch_size, max(1, len(dataset)))
    drop_last = train and len(dataset) > effective
    return torch.utils.data.DataLoader(dataset, batch_size=effective, shuffle=train, drop_last=drop_last, num_workers=0)


def tensor_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int, train: bool):
    """Batch whole tensors directly; useful when the cached split fits on the device."""
    effective = min(batch_size, max(1, len(X)))
    stop = (len(X) // effective) * effective if train and len(X) > effective else len(X)
    order = torch.randperm(len(X), device=X.device) if train else torch.arange(len(X), device=X.device)
    for start in range(0, stop, effective):
        index = order[start : min(start + effective, stop)]
        yield X.index_select(0, index), y.index_select(0, index)


def make_optimizer(name: str, params, learning_rate: float):
    normalized = name.lower()
    if normalized == "adam":
        return torch.optim.Adam(params, lr=learning_rate)
    if normalized == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate)
    if normalized == "adamax":
        return torch.optim.Adamax(params, lr=learning_rate)
    if normalized == "rmsprop":
        return torch.optim.RMSprop(params, lr=learning_rate)
    if normalized == "sgd":
        return torch.optim.SGD(params, lr=learning_rate)
    raise ValueError(f"Unsupported optimizer in Benchmark/all_default_configs: {name}")


def classification_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> dict[str, float]:
    """Compute validation metrics without introducing an sklearn dependency."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        confusion[int(truth), int(pred)] += 1
    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / total) if total else 0.0
    recalls = []
    f1s = []
    for class_id in range(num_classes):
        tp = int(confusion[class_id, class_id])
        fn = int(confusion[class_id, :].sum() - tp)
        fp = int(confusion[:, class_id].sum() - tp)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


def load_row_labels(path: Path, expected_count: int, num_classes: int) -> torch.Tensor:
    values = np.load(path, allow_pickle=False)
    if values.ndim != 1 or len(values) != expected_count or values.dtype.kind not in "iu":
        raise ValueError(f"Invalid row labels in {path}: {values.shape}, {values.dtype}")
    if len(values) == 0 or values.min() < 0 or values.max() >= num_classes:
        raise ValueError(f"Out-of-range or empty row labels in {path}")
    return torch.from_numpy(values.astype(np.int64, copy=True))


def select_input_rows(X: torch.Tensor, y: torch.Tensor, path: Path):
    indices = np.load(path, allow_pickle=False)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) == 0:
        raise ValueError(f"Invalid row selection in {path}")
    if len(X) != len(y) or indices[0] < 0 or indices[-1] >= len(X) or np.any(indices[1:] <= indices[:-1]):
        raise ValueError(f"Row indices must be in range and strictly increasing: {path}")
    index = torch.from_numpy(indices.astype(np.int64, copy=True))
    return X.index_select(0, index), y.index_select(0, index)


def load_resample_indices(path: Path, expected_count: int) -> torch.Tensor:
    indices = np.load(path, allow_pickle=False)
    if indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) == 0:
        raise ValueError(f"Invalid training resample indices in {path}")
    if indices.min() < 0 or indices.max() >= expected_count:
        raise ValueError(f"Training resample indices out of range in {path}")
    return torch.from_numpy(indices.astype(np.int64, copy=True))


def train_one_model(
    spec: ModelSpec,
    raw_dir: Path,
    run_dir: Path,
    num_classes: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if spec.unsupported_reason:
        return {
            "algorithm": f"WFlib-{spec.name}",
            "status": "unsupported",
            "source_work": spec.source_work,
            "feature": spec.feature,
            "seq_len": spec.seq_len,
            "reason": spec.unsupported_reason,
        }

    started = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        if getattr(args, "require_cuda", False):
            raise RuntimeError("CUDA is required for this run; refusing CPU fallback")
        device_name = "cpu"
    device = torch.device(device_name)

    X_train, y_train, train_input = tensors_for_split(raw_dir, "train", spec, args.feature_workers)
    X_val, y_val, val_input = tensors_for_split(raw_dir, "val", spec, args.feature_workers)
    X_test, y_test, test_input = tensors_for_split(raw_dir, "test", spec, args.feature_workers)
    source_sample_counts = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
    row_indices_dir = getattr(args, "row_indices_dir", None)
    if row_indices_dir is not None:
        X_train, y_train = select_input_rows(X_train, y_train, Path(row_indices_dir) / "train_indices.npy")
        X_val, y_val = select_input_rows(X_val, y_val, Path(row_indices_dir) / "val_indices.npy")
        X_test, y_test = select_input_rows(X_test, y_test, Path(row_indices_dir) / "test_indices.npy")
    label_id_remap = getattr(args, "label_id_remap", None)
    row_labels_dir = getattr(args, "row_labels_dir", None)
    output_label_mapping = Path(getattr(args, "output_label_mapping", raw_dir / "label_mapping.json"))
    if row_labels_dir is not None:
        if label_id_remap is not None:
            raise ValueError("Cannot combine per-row labels with a class-ID remap")
        y_train = load_row_labels(Path(row_labels_dir) / "train_labels.npy", len(X_train), num_classes)
        y_val = load_row_labels(Path(row_labels_dir) / "val_labels.npy", len(X_val), num_classes)
        y_test = load_row_labels(Path(row_labels_dir) / "test_labels.npy", len(X_test), num_classes)
    if label_id_remap is not None:
        remap = torch.tensor(label_id_remap, dtype=torch.int64)
        if int(torch.max(y_train).item()) >= len(remap) or int(torch.max(y_val).item()) >= len(remap) or int(torch.max(y_test).item()) >= len(remap):
            raise ValueError(f"{spec.name} label_id_remap has {len(remap)} entries but observed a larger label id")
        y_train = remap[y_train]
        y_val = remap[y_val]
        y_test = remap[y_test]

    resample_path = getattr(args, "train_resample_indices", None)
    if resample_path is not None:
        train_index = load_resample_indices(Path(resample_path), len(y_train))
        X_train = X_train.index_select(0, train_index)
        y_train = y_train.index_select(0, train_index)

    model = instantiate_model(spec, num_classes).to(device)
    learning_rate = float(args.learning_rate if args.learning_rate is not None else spec.learning_rate)
    batch_size = int(args.batch_size if args.batch_size is not None else spec.batch_size)
    epochs = int(args.epochs if args.epochs is not None else spec.train_epochs)
    optimizer_name = str(args.optimizer if args.optimizer is not None else spec.optimizer or "Adam")
    optimizer = make_optimizer(optimizer_name, model.parameters(), learning_rate)
    criterion = nn.CrossEntropyLoss()

    preload_device = bool(getattr(args, "preload_device", False))
    if preload_device:
        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)
        train_loader = None
        val_loader = None
    else:
        train_loader = make_loader(X_train, y_train, batch_size, train=True)
        val_loader = make_loader(X_val, y_val, batch_size, train=False)
    checkpoint_path = run_dir / "checkpoint-best.pt"
    checkpoint_metric = str(getattr(args, "checkpoint_metric", "accuracy"))
    native_metrics = {
        "model": spec.name,
        "device": str(device),
        "source_work": spec.source_work,
        "feature": spec.feature,
        "seq_len": spec.seq_len,
        "raw_seq_len": spec.raw_seq_len,
        "train_input": rel(train_input),
        "val_input": rel(val_input),
        "test_input": rel(test_input),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,
        "preload_device": preload_device,
        "json_train_epochs": spec.train_epochs,
        "json_batch_size": spec.batch_size,
        "json_learning_rate": spec.learning_rate,
        "json_optimizer": spec.optimizer,
        "label_id_remap": label_id_remap,
        "row_labels_dir": rel(Path(row_labels_dir)) if row_labels_dir is not None else None,
        "row_indices_dir": rel(Path(row_indices_dir)) if row_indices_dir is not None else None,
        "source_sample_counts": source_sample_counts,
        "selected_sample_counts": {"train": len(y_train), "val": len(y_val), "test": len(y_test)},
        "train_resample_indices": rel(Path(resample_path)) if resample_path is not None else None,
        "output_label_mapping": rel(output_label_mapping),
        "best_checkpoint": rel(checkpoint_path),
        "best_val_accuracy": None,
        "best_val_balanced_accuracy": None,
        "best_val_macro_f1": None,
        "checkpoint_metric": checkpoint_metric,
        "best_checkpoint_score": None,
        "best_epoch": None,
        "loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
        "val_macro_f1": [],
    }
    model.train()
    best_checkpoint_score = -1.0
    for epoch in range(epochs):
        losses = []
        model.train()
        train_batches = tensor_batches(X_train, y_train, batch_size, train=True) if preload_device else train_loader
        for batch_x, batch_y in train_batches:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            if logits.ndim != 2 or logits.shape[1] != num_classes:
                raise ValueError(
                    f"{spec.name} output shape {tuple(logits.shape)} is not [batch, {num_classes}] logits"
                )
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        native_metrics["loss"].append(float(np.mean(losses)) if losses else None)
        val_true = []
        val_pred = []
        model.eval()
        with torch.no_grad():
            val_batches = tensor_batches(X_val, y_val, batch_size, train=False) if preload_device else val_loader
            for batch_x, batch_y in val_batches:
                logits = model(batch_x.to(device))
                pred = torch.argmax(logits, dim=1).detach().cpu()
                val_true.extend(batch_y.detach().cpu().tolist())
                val_pred.extend(pred.tolist())
        val_metrics = classification_metrics(val_true, val_pred, num_classes)
        for metric_name, metric_value in val_metrics.items():
            native_metrics[f"val_{metric_name}"].append(metric_value)
        checkpoint_score = val_metrics[checkpoint_metric]
        if checkpoint_score >= best_checkpoint_score:
            best_checkpoint_score = checkpoint_score
            native_metrics["best_val_accuracy"] = val_metrics["accuracy"]
            native_metrics["best_val_balanced_accuracy"] = val_metrics["balanced_accuracy"]
            native_metrics["best_val_macro_f1"] = val_metrics["macro_f1"]
            native_metrics["best_checkpoint_score"] = checkpoint_score
            native_metrics["best_epoch"] = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_metrics": val_metrics,
                    "checkpoint_metric": checkpoint_metric,
                    "model": spec.name,
                    "num_classes": num_classes,
                },
                checkpoint_path,
            )
        epoch_metrics_path = getattr(args, "epoch_metrics_path", None)
        if epoch_metrics_path is not None:
            Path(epoch_metrics_path).write_text(json.dumps(native_metrics, indent=2), encoding="utf-8")
            print(
                f"{spec.name} epoch={epoch + 1}/{epochs} loss={native_metrics['loss'][-1]} "
                f"val_accuracy={val_metrics['accuracy']:.6f} val_balanced_accuracy={val_metrics['balanced_accuracy']:.6f} "
                f"val_macro_f1={val_metrics['macro_f1']:.6f} best_{checkpoint_metric}={best_checkpoint_score:.6f}",
                flush=True,
            )

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    soft_batches = []
    hard = []
    embedding_batches = []
    embedding_hook = None
    embedding_layer = getattr(args, "embedding_layer", None)
    if embedding_layer:
        def capture_embedding(_module, inputs):
            values = inputs[0].detach().cpu().float()
            if values.ndim == 3 and values.shape[1] == 1:
                values = values.squeeze(1)
            if values.ndim != 2:
                raise ValueError(f"Expected segment vectors, got {tuple(values.shape)}")
            embedding_batches.append(values.numpy())
        embedding_hook = model.get_submodule(embedding_layer).register_forward_pre_hook(capture_embedding)
    model.eval()
    test_loader = tensor_batches(X_test, y_test, batch_size, train=False) if preload_device else make_loader(X_test, y_test, batch_size, train=False)
    with torch.no_grad():
        for batch_x, _batch_y in test_loader:
            logits = model(batch_x.to(device))
            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
            soft_batches.append(prob)
            hard.extend(np.argmax(prob, axis=1).astype(str).tolist())
    soft = np.concatenate(soft_batches, axis=0) if soft_batches else np.zeros((0, num_classes), dtype=np.float32)
    if embedding_hook is not None:
        embedding_hook.remove()
        embeddings = np.concatenate(embedding_batches)
        if len(embeddings) != len(y_test) or not np.isfinite(embeddings).all():
            raise ValueError("Invalid test embeddings")
        np.save(run_dir / "test_embeddings.npy", embeddings)
        (run_dir / "test_embeddings_meta.json").write_text(json.dumps({
            "layer": embedding_layer, "point": "input_to_classifier",
            "shape": list(embeddings.shape), "checkpoint": str(checkpoint_path.resolve()),
            "best_epoch": native_metrics["best_epoch"], "order": "unaltered test input order",
        }, indent=2))
    np.save(run_dir / "soft_predictions.npy", soft.astype(np.float32))
    (run_dir / "hard_predictions.txt").write_text("\n".join(hard) + ("\n" if hard else ""), encoding="utf-8")
    (run_dir / "test_labels.txt").write_text(
        "\n".join(str(int(item)) for item in y_test.detach().cpu().numpy()) + ("\n" if len(y_test) else ""),
        encoding="utf-8",
    )
    (run_dir / "label_mapping.json").write_text(output_label_mapping.read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "native_metrics.json").write_text(json.dumps(native_metrics, indent=2, sort_keys=True), encoding="utf-8")
    metrics = evaluate_predictions(
        run_dir / "hard_predictions.txt",
        run_dir / "test_labels.txt",
        run_dir / "evaluation_metrics.json",
        soft_path=run_dir / "soft_predictions.npy",
        label_mapping_path=run_dir / "label_mapping.json",
    )
    return {
        "algorithm": f"WFlib-{spec.name}",
        "status": "completed",
        "source_work": spec.source_work,
        "feature": spec.feature,
        "seq_len": spec.seq_len,
        "run_dir": rel(run_dir),
        "elapsed_seconds": round(time.time() - started, 3),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "weighted_f1": metrics.get("weighted_f1"),
        "prediction_source": metrics.get("prediction_source"),
    }


def write_summary(results: list[dict[str, Any]], manifests: dict[str, Any], out_dir: Path) -> None:
    payload = {"manifests": manifests, "results": results}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# ANT Semantic WF Smoke Test",
        "",
        f"标签定义：`{manifests['label_field']}`；同一 label 在所有 agent engine 中合并为同一类。",
        "",
        "## 数据规模",
        "",
        "| split | counts_by_label | counts_by_engine |",
        "| --- | --- | --- |",
    ]
    for split in ["train", "val", "test"]:
        item = manifests.get("splits", {}).get(split, {})
        item.setdefault("counts_by_label", {})
        item.setdefault("counts_by_engine", {})
        lines.append(
            f"| {split} | `{json.dumps(item['counts_by_label'], sort_keys=True, ensure_ascii=False)}` | "
            f"`{json.dumps(item['counts_by_engine'], sort_keys=True, ensure_ascii=False)}` |"
        )
    lines.extend(
        [
            "",
            "## WF baseline 结果",
            "",
            "| baseline | source_work | status | feature | seq_len | accuracy | balanced_accuracy | macro_f1 | weighted_f1 | prediction_source | run_dir |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in results:
        lines.append(
            "| {algorithm} | {source_work} | {status} | {feature} | {seq_len} | {accuracy} | {balanced_accuracy} | {macro_f1} | {weighted_f1} | {prediction_source} | `{run_dir}` |".format(
                algorithm=row.get("algorithm", ""),
                source_work=row.get("source_work", ""),
                status=row.get("status", ""),
                feature=row.get("feature", ""),
                seq_len=row.get("seq_len", ""),
                accuracy="" if row.get("accuracy") is None else f"{row['accuracy']:.4f}",
                balanced_accuracy="" if row.get("balanced_accuracy") is None else f"{row['balanced_accuracy']:.4f}",
                macro_f1="" if row.get("macro_f1") is None else f"{row['macro_f1']:.4f}",
                weighted_f1="" if row.get("weighted_f1") is None else f"{row['weighted_f1']:.4f}",
                prediction_source=row.get("prediction_source", ""),
                run_dir=row.get("run_dir", ""),
            )
        )
    skipped = [row for row in results if row["status"] in {"unsupported", "failed"}]
    if skipped:
        lines.extend(["", "## 未完成项", ""])
        for row in skipped:
            reason = row.get("reason") or row.get("error") or ""
            lines.append(f"- `{row['algorithm']}` ({row['status']}): {reason}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hyperparameters",
        type=Path,
        default=None,
        help="Baseline performance hyperparameter JSON directory or file; defaults to Benchmark/all_default_configs",
    )
    parser.add_argument("--data-root", type=Path, default=BENCHMARK / "data" / "ANT")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--raw-dir", type=Path, default=None, help="Use an already prepared native WFlib raw directory")
    parser.add_argument("--per-engine-scenario", type=int, default=1)
    parser.add_argument("--label-field", default="Scenario", choices=["Engine", "Harness", "Model", "Scenario", "Task", "Risk"])
    parser.add_argument("--protocols", default="tcp")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--optimizer", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--feature-workers", type=int, default=None)
    parser.add_argument("--train-resample-indices", type=Path, default=None)
    parser.add_argument("--row-labels-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint-metric",
        default="accuracy",
        choices=["accuracy", "balanced_accuracy", "macro_f1"],
        help="Validation metric used to select checkpoint-best.pt.",
    )
    parser.add_argument(
        "--epoch-metrics-path",
        type=Path,
        default=None,
        help="Write in-progress training metrics after every epoch.",
    )
    parser.add_argument(
        "--preload-device",
        action="store_true",
        help="Preload cached tensors onto the selected device and form batches there.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare and validate raw/model feature caches without training models.",
    )
    parser.add_argument(
        "--use-default-config-training",
        action="store_true",
        help="Use each WFlib submodel's all_default_configs training hyperparameters instead of smoke_runner_hyperparameters.",
    )
    args = parser.parse_args()

    model_specs = build_model_specs(args.hyperparameters)
    hp = wflib_hyperparameters(args.hyperparameters)
    smoke = hp.get("smoke_runner_hyperparameters", {})
    if args.epochs is None and not args.use_default_config_training:
        args.epochs = int(smoke.get("epochs", 1))
    if args.batch_size is None and not args.use_default_config_training:
        args.batch_size = int(smoke.get("batch_size", 8))
    if args.learning_rate is None and not args.use_default_config_training:
        args.learning_rate = float(smoke.get("learning_rate", 0.001))
    if args.optimizer is None and not args.use_default_config_training:
        args.optimizer = str(smoke.get("optimizer", "Adam"))
    if args.device is None:
        args.device = str(smoke.get("device", "cuda")).split()[0]
    if args.seed is None:
        args.seed = int(smoke.get("seed", 42))
    if args.feature_workers is None:
        args.feature_workers = int(smoke.get("feature_workers", 2))

    set_seed(args.seed)
    out_dir = args.out_dir or BENCHMARK / "ant_scenario_smoke" / f"wf_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_names = args.models or list(model_specs)
    requested = [model_specs[name] for name in model_names]
    raw_seq_len = max(spec.raw_seq_len for spec in requested)
    selected = select_episodes(args.data_root, args.per_engine_scenario, ["train", "val", "test"], args.label_field)
    raw_dir = args.raw_dir
    if raw_dir is None:
        raw_dir = out_dir / "wflib_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        manifest = write_raw_npz(selected, raw_dir, raw_seq_len, proto_set(args.protocols), args.label_field)
    else:
        manifest = json.loads((raw_dir / "raw_manifest.json").read_text(encoding="utf-8")) if (raw_dir / "raw_manifest.json").exists() else {"label_field": args.label_field}
    label_payload = json.loads((raw_dir / "label_mapping.json").read_text(encoding="utf-8"))
    num_classes = len(label_payload["id_to_label"])

    if args.prepare_only:
        prepared = set()
        for spec in requested:
            if spec.feature not in {"TAM", "TAF", "MTAF"}:
                continue
            for split in ("train", "val", "test"):
                key = (spec.feature, split)
                if key in prepared:
                    continue
                precompute_feature(
                    raw_dir / f"{split}.npz",
                    spec,
                    raw_dir / f"{spec.feature.lower()}_{split}.npz",
                    args.feature_workers,
                )
                prepared.add(key)
        write_summary([], manifest, out_dir)
        print(json.dumps({"status": "prepared", "raw_dir": rel(raw_dir), "models": model_names}, indent=2))
        return 0

    results = []
    for spec in requested:
        run_dir = out_dir / f"WFlib-{spec.name}"
        try:
            result = train_one_model(spec, raw_dir, run_dir, num_classes, args)
        except Exception as exc:
            result = {
                "algorithm": f"WFlib-{spec.name}",
                "status": "failed",
                "source_work": spec.source_work,
                "feature": spec.feature,
                "seq_len": spec.seq_len,
                "run_dir": rel(run_dir),
                "error": repr(exc),
            }
        results.append(result)
        write_summary(results, manifest, out_dir)

    print(json.dumps({"summary": rel(out_dir / "summary.md"), "results": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
