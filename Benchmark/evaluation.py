from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .utils import write_json


BINARY_BENIGN = {"0", "0.0", "benign", "normal", "false"}
BINARY_MALICIOUS = {"1", "1.0", "malicious", "attack", "true"}


def read_label_file(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_binary_label(value: object) -> str:
    text = str(value).strip().lower()
    if text in BINARY_BENIGN:
        return "benign"
    if text in BINARY_MALICIOUS:
        return "malicious"
    return str(value).strip()


def load_mapping(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in payload.get("id_to_label", {}).items()}


def _numeric_label_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _ordered_mapping_labels(id_to_label: dict[str, str]) -> list[str]:
    return sorted(id_to_label, key=_numeric_label_sort_key)


def _labels_from_observed(observed: set[str]) -> list[str]:
    if observed <= {"benign", "malicious"}:
        return ["benign", "malicious"]
    return sorted(observed, key=_numeric_label_sort_key)


def _normalize_with_mapping(raw: list[str], id_to_label: dict[str, str]) -> list[str] | None:
    if not id_to_label:
        return None
    keys = set(id_to_label)
    if set(raw) <= keys:
        return raw
    label_to_id = {label: label_id for label_id, label in id_to_label.items()}
    if set(raw) <= set(label_to_id):
        return [label_to_id[value] for value in raw]
    return None


def _normalize_truth(raw_true: list[str], id_to_label: dict[str, str]) -> tuple[list[str], list[str]]:
    mapped = _normalize_with_mapping(raw_true, id_to_label)
    if mapped is not None:
        return mapped, _ordered_mapping_labels(id_to_label)
    y_true = [normalize_binary_label(value) for value in raw_true]
    return y_true, _labels_from_observed(set(y_true))


def _labels_for_soft(raw_true: list[str], id_to_label: dict[str, str], class_count: int) -> tuple[list[str], list[str]]:
    mapped = _normalize_with_mapping(raw_true, id_to_label)
    if mapped is not None:
        labels = _ordered_mapping_labels(id_to_label)
        if len(labels) != class_count:
            raise ValueError(
                f"Soft prediction class count mismatch: {class_count}; label mapping contains {len(labels)} classes"
            )
        return mapped, labels

    normalized = [normalize_binary_label(value) for value in raw_true]
    observed = set(normalized)
    if observed <= {"benign", "malicious"} and class_count == 2:
        return normalized, ["benign", "malicious"]

    raw_numeric = []
    for value in raw_true:
        try:
            parsed = int(float(value))
        except ValueError:
            raw_numeric = []
            break
        raw_numeric.append(parsed)
    if raw_numeric and min(raw_numeric) >= 0 and max(raw_numeric) < class_count:
        return [str(value) for value in raw_numeric], [str(index) for index in range(class_count)]

    labels = _labels_from_observed(observed)
    if len(labels) != class_count:
        raise ValueError(
            "Cannot infer soft prediction label order: provide label_mapping.json "
            f"or make soft columns match observed labels; got {class_count} columns and {len(labels)} observed labels"
        )
    return normalized, labels


def _hard_from_soft(soft: np.ndarray, labels: list[str]) -> list[str]:
    if soft.ndim != 2:
        raise ValueError(f"Soft prediction array must be 2-D, got shape {soft.shape}")
    if soft.shape[1] != len(labels):
        raise ValueError(f"Soft prediction class count mismatch: {soft.shape[1]} vs {len(labels)} labels")
    indices = np.argmax(soft, axis=1)
    return [labels[int(index)] for index in indices]


def evaluate_predictions(
    hard_path: Path | None,
    label_path: Path,
    output_path: Path,
    soft_path: Path | None = None,
    label_mapping_path: Path | None = None,
    skip_invalid_soft: bool = False,
) -> dict[str, Any]:
    raw_true = read_label_file(label_path)
    id_to_label = load_mapping(label_mapping_path)

    soft = None
    skipped_invalid_soft_indices: list[int] = []
    original_sample_count = len(raw_true)
    provided_hard = None
    if hard_path is not None and hard_path.exists():
        provided_hard = read_label_file(hard_path)

    if soft_path is not None and soft_path.exists():
        soft = np.load(soft_path)
        if soft.ndim != 2 or soft.shape[0] != len(raw_true):
            raise ValueError(f"Soft prediction shape mismatch: {soft.shape}; expected first dimension {len(raw_true)}")
        y_true, labels = _labels_for_soft(raw_true, id_to_label, int(soft.shape[1]))
        finite_rows = np.all(np.isfinite(soft), axis=1)
        if not bool(np.all(finite_rows)):
            skipped_invalid_soft_indices = [int(index) for index in np.flatnonzero(~finite_rows)]
            if not skip_invalid_soft:
                preview = skipped_invalid_soft_indices[:20]
                suffix = "" if len(skipped_invalid_soft_indices) <= 20 else "..."
                raise ValueError(
                    "Soft predictions contain non-finite values: "
                    f"{len(skipped_invalid_soft_indices)} rows, indices={preview}{suffix}"
                )
            if not bool(np.any(finite_rows)):
                raise ValueError("All soft prediction rows contain non-finite values; cannot evaluate")
            valid_indices = [int(index) for index in np.flatnonzero(finite_rows)]
            soft = soft[finite_rows]
            y_true = [y_true[index] for index in valid_indices]
        y_pred = _hard_from_soft(soft, labels)
        prediction_source = "soft_predictions_argmax"
    else:
        if provided_hard is None:
            raise ValueError("Either soft_predictions.npy or hard_predictions.txt is required")
        if len(provided_hard) != len(raw_true):
            raise ValueError(f"Prediction/label length mismatch: {len(provided_hard)} vs {len(raw_true)}")
        y_true, true_labels = _normalize_truth(raw_true, id_to_label)
        mapped_pred = _normalize_with_mapping(provided_hard, id_to_label)
        y_pred = mapped_pred if mapped_pred is not None else [normalize_binary_label(value) for value in provided_hard]
        labels = _labels_from_observed(set(y_pred) | set(y_true))
        if id_to_label and set(labels) <= set(id_to_label):
            labels = _ordered_mapping_labels(id_to_label)
        elif set(true_labels) - set(labels):
            labels = _labels_from_observed(set(labels) | set(true_labels))
        prediction_source = "hard_predictions"

    if len(y_pred) != len(y_true):
        raise ValueError(f"Prediction/label length mismatch: {len(y_pred)} vs {len(y_true)}")
    derived_hard_path = output_path.with_name("derived_hard_predictions.txt")
    derived_hard_path.write_text("\n".join(y_pred) + "\n", encoding="utf-8")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fpr_values = []
    total = int(cm.sum())
    for index in range(len(labels)):
        fp = int(cm[:, index].sum() - cm[index, index])
        tn = int(total - cm[index, :].sum() - cm[:, index].sum() + cm[index, index])
        denominator = fp + tn
        fpr_values.append(float(fp / denominator) if denominator else 0.0)
    metrics: dict[str, Any] = {
        "prediction_source": prediction_source,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "labels": labels,
        "label_names": [id_to_label.get(label, label) for label in labels],
        "original_sample_count": int(original_sample_count),
        "evaluated_sample_count": int(len(y_true)),
        "skipped_invalid_soft_rows": int(len(skipped_invalid_soft_indices)),
        "derived_hard_predictions": str(derived_hard_path),
        "confusion_matrix": cm.tolist(),
        "per_label": {
            id_to_label.get(label, label): {
                "label_id": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "false_positive_rate": float(fpr_values[index]),
                "fpr": float(fpr_values[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }
    for average in ["macro", "micro", "weighted"]:
        avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        )
        metrics[f"{average}_precision"] = float(avg_precision)
        metrics[f"{average}_recall"] = float(avg_recall)
        metrics[f"{average}_f1"] = float(avg_f1)
    metrics["macro_fpr"] = float(np.mean(fpr_values)) if fpr_values else 0.0
    support_total = int(np.sum(support))
    metrics["weighted_fpr"] = float(np.average(fpr_values, weights=support)) if support_total else 0.0
    malicious = metrics["per_label"].get("malicious")
    if isinstance(malicious, dict):
        metrics["malicious_precision"] = float(malicious["precision"])
        metrics["malicious_recall"] = float(malicious["recall"])
        metrics["malicious_f1"] = float(malicious["f1"])
        metrics["malicious_fpr"] = float(malicious["false_positive_rate"])
        metrics["false_positive_rate"] = float(malicious["false_positive_rate"])
        metrics["fpr"] = float(malicious["false_positive_rate"])

    if provided_hard is not None:
        mapped_hard = _normalize_with_mapping(provided_hard, id_to_label)
        normalized_hard = mapped_hard if mapped_hard is not None else [normalize_binary_label(value) for value in provided_hard]
        if skipped_invalid_soft_indices and len(normalized_hard) == original_sample_count:
            skipped = set(skipped_invalid_soft_indices)
            normalized_hard = [value for index, value in enumerate(normalized_hard) if index not in skipped]
        if len(normalized_hard) == len(y_pred):
            mismatch_count = sum(1 for left, right in zip(normalized_hard, y_pred) if left != right)
            metrics["provided_hard_predictions"] = str(hard_path)
            metrics["provided_hard_mismatch_count"] = int(mismatch_count)

    if soft is not None:
        metrics["soft_predictions"] = str(soft_path)
        if skipped_invalid_soft_indices:
            skipped_path = output_path.with_name("skipped_invalid_soft_rows.json")
            write_json(
                skipped_path,
                {
                    "reason": "non-finite values in soft_predictions.npy",
                    "skipped_count": len(skipped_invalid_soft_indices),
                    "skipped_indices": skipped_invalid_soft_indices,
                },
            )
            metrics["skipped_invalid_soft_indices"] = skipped_invalid_soft_indices
            metrics["skipped_invalid_soft_rows_file"] = str(skipped_path)
        y_binary = label_binarize(y_true, classes=labels)
        if len(labels) == 2:
            y_binary = np.column_stack([1 - y_binary[:, 0], y_binary[:, 0]])
        curves: dict[str, Any] = {"classes": {}}
        auc_by_label: dict[str, float] = {}
        for index, label in enumerate(labels):
            if index >= soft.shape[1]:
                continue
            binary_column = y_binary[:, index]
            if len(np.unique(binary_column)) < 2:
                continue
            fpr, tpr, thresholds = roc_curve(binary_column, soft[:, index])
            youden = tpr - fpr
            best_index = int(np.argmax(youden))
            label_name = id_to_label.get(label, label)
            auc = float(roc_auc_score(binary_column, soft[:, index]))
            auc_by_label[label_name] = auc
            curves["classes"][label_name] = {
                "label_id": label,
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": [float(value) if np.isfinite(value) else None for value in thresholds],
                "auc": auc,
                "best_threshold": float(thresholds[best_index]) if np.isfinite(thresholds[best_index]) else None,
                "best_youden_j": float(youden[best_index]),
            }
        metrics["roc_auc_ovr_per_label"] = auc_by_label
        try:
            metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_binary, soft[:, : len(labels)], average="macro"))
            metrics["roc_auc_ovr_weighted"] = float(roc_auc_score(y_binary, soft[:, : len(labels)], average="weighted"))
            metrics["roc_auc_ovr_micro"] = float(roc_auc_score(y_binary.ravel(), soft[:, : len(labels)].ravel()))
        except ValueError:
            pass
        curves_path = output_path.with_name("roc_curves.json")
        write_json(curves_path, curves)
        metrics["roc_curves"] = str(curves_path)

    write_json(output_path, metrics)
    return metrics
