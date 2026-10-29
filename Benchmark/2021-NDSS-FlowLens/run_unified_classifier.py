#!/usr/bin/env python3
"""Unified FlowLens botnet classifier runner.

This runner keeps the original FlowLens BotnetAnalysis classifier choice:
RandomForestClassifier(random_state=42).  It accepts benchmark pcap input,
constructs 2021-NDSS-FlowLens/PeerShark-style IP-pair flows and superflows, trains/tests
with the original stratified random split, and maps superflow predictions back
to five-tuple flows for flow-level benchmark metrics.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


REPO = Path(__file__).resolve().parents[1]
ACCUMULATOR_PATH = REPO / "2021-NDSS-FlowLens" / "offline_flow_marker_accumulator.py"
SPEC = importlib.util.spec_from_file_location("offline_flow_marker_accumulator", ACCUMULATOR_PATH)
acc = importlib.util.module_from_spec(SPEC)
sys.modules["offline_flow_marker_accumulator"] = acc
assert SPEC.loader is not None
SPEC.loader.exec_module(acc)


DEFAULT_CONFIG = {
    "input": {
        "pcap": "",
        "packet_labels": "",
        "packet_label_delimiter": ",",
        "max_packets": None,
    },
    "labels": {
        "benign_values": ["0", "benign", "normal", "false"],
        "malicious_values": ["1", "malicious", "attack", "true"],
        "superflow_label_policy": "majority_vote",
        "five_tuple_label_policy": "majority_vote",
    },
    "flowlens": {
        "packet_size_bin_width": 16,
        "ipt_bin_width_seconds": 0,
        "timegap_seconds": 2000.0,
        "flowgap_seconds": 3600.0,
        "packet_length_source": "orig_len",
        "skip_non_tcp_udp": True,
        "skip_zero_payload_packets": False,
        "five_tuple_mode": "unidirectional",
        "five_tuple_timeout_seconds": 120.0,
    },
    "split": {
        "test_size": 0.33,
        "random_state": 42,
        "stratify": True,
    },
    "classifier": {
        "random_state": 42,
    },
    "output": {
        "dir": "2021-NDSS-FlowLens/outputs/unified_classifier",
        "hard_predictions": "hard_predictions.txt",
        "soft_predictions": "soft_predictions.npy",
        "test_labels": "test_labels.txt",
        "five_tuple_predictions": "five_tuple_predictions.csv",
        "superflow_dataset": "superflow_dataset.csv",
        "metrics": "metrics.json",
    },
}


@dataclass
class DirectionStats:
    packets: int = 0
    bytes: int = 0
    start: float = 0.0
    end: float = 0.0
    ipts: List[float] = field(default_factory=list)

    def add(self, timestamp: float, size: int) -> None:
        if self.packets == 0:
            self.start = timestamp
            self.end = timestamp
        elif self.end < timestamp:
            self.ipts.append(timestamp - self.end)
            self.end = timestamp
        elif self.start > timestamp:
            self.ipts.append(self.start - timestamp)
            self.start = timestamp
        self.packets += 1
        self.bytes += size


@dataclass
class FlowLensFlow:
    ip1: str
    ip2: str
    key: Tuple[str, str]
    direction1: DirectionStats = field(default_factory=DirectionStats)
    direction2: DirectionStats = field(default_factory=DirectionStats)
    label_votes: Counter = field(default_factory=Counter)
    five_tuple_ids: set = field(default_factory=set)

    def add_packet(self, packet, size: int, five_tuple_id: str) -> None:
        if packet.src_ip == self.ip1 and packet.dst_ip == self.ip2:
            self.direction1.add(packet.timestamp, size)
        else:
            self.direction2.add(packet.timestamp, size)
        if packet.label:
            self.label_votes[packet.label] += 1
        self.five_tuple_ids.add(five_tuple_id)

    @property
    def packets(self) -> int:
        return self.direction1.packets + self.direction2.packets

    @property
    def bytes(self) -> int:
        return self.direction1.bytes + self.direction2.bytes

    @property
    def start(self) -> float:
        return min_nonzero(self.direction1.start, self.direction2.start)

    @property
    def end(self) -> float:
        return max(self.direction1.end, self.direction2.end)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def median_ipt(self) -> float:
        return median(self.direction1.ipts + self.direction2.ipts)

    def merge(self, other: "FlowLensFlow") -> None:
        self.direction1 = merge_direction(self.direction1, other.direction1)
        self.direction2 = merge_direction(self.direction2, other.direction2)
        self.label_votes.update(other.label_votes)
        self.five_tuple_ids.update(other.five_tuple_ids)


@dataclass
class FiveTupleFlow:
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: str
    dst_port: str
    protocol: str
    start: float
    end: float
    packets: int = 0
    bytes: int = 0
    label_votes: Counter = field(default_factory=Counter)

    def add_packet(self, packet, size: int) -> None:
        self.start = min(self.start, packet.timestamp)
        self.end = max(self.end, packet.timestamp)
        self.packets += 1
        self.bytes += size
        if packet.label:
            self.label_votes[packet.label] += 1


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return deep_merge(DEFAULT_CONFIG, json.load(handle))


def min_nonzero(a: float, b: float) -> float:
    if a == 0:
        return b
    if b == 0:
        return a
    return min(a, b)


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def merge_direction(a: DirectionStats, b: DirectionStats) -> DirectionStats:
    merged = DirectionStats()
    merged.packets = a.packets + b.packets
    merged.bytes = a.bytes + b.bytes
    merged.start = min_nonzero(a.start, b.start)
    merged.end = max(a.end, b.end)
    merged.ipts = a.ipts + b.ipts
    if a.packets and b.packets:
        if a.end <= b.start:
            merged.ipts.append(b.start - a.end)
        elif b.end <= a.start:
            merged.ipts.append(a.start - b.end)
    return merged


def round_to_nearest(value: int, width: int) -> int:
    if width <= 1:
        return value
    remainder = value % width
    if remainder + remainder >= width:
        return value + width - remainder
    return value - remainder


def canonical_ip_pair(packet) -> Tuple[str, str]:
    return (packet.src_ip, packet.dst_ip) if packet.src_ip < packet.dst_ip else (packet.dst_ip, packet.src_ip)


def five_tuple_key(packet, mode: str) -> Tuple[str, str, str, str, str]:
    if mode == "unidirectional":
        return (packet.src_ip, packet.dst_ip, packet.src_port, packet.dst_port, packet.protocol)
    if mode == "bidirectional":
        left = (packet.src_ip, packet.src_port)
        right = (packet.dst_ip, packet.dst_port)
        if left <= right:
            return (packet.src_ip, packet.dst_ip, packet.src_port, packet.dst_port, packet.protocol)
        return (packet.dst_ip, packet.src_ip, packet.dst_port, packet.src_port, packet.protocol)
    raise ValueError(f"Unsupported flowlens.five_tuple_mode: {mode}")


def packet_size(packet, config: dict) -> int:
    source = config["flowlens"]["packet_length_source"]
    if source == "payload_length":
        raw = packet.payload_length
    else:
        raw = packet.packet_length
    return round_to_nearest(raw, int(config["flowlens"]["packet_size_bin_width"]))


def packet_timestamp(packet, config: dict) -> float:
    width = int(config["flowlens"]["ipt_bin_width_seconds"])
    if width <= 0:
        return packet.timestamp
    return float(round_to_nearest(int(packet.timestamp), width))


def label_from_votes(votes: Counter, policy: str) -> str:
    if not votes:
        return ""
    if policy == "majority_vote":
        return votes.most_common(1)[0][0]
    if policy == "any_malicious":
        return "malicious" if votes.get("malicious", 0) > 0 else votes.most_common(1)[0][0]
    if policy == "drop_mixed":
        return "mixed" if len(votes) > 1 else votes.most_common(1)[0][0]
    raise ValueError(f"Unsupported label policy: {policy}")


def build_five_tuple_flows(packets: Sequence, config: dict) -> Tuple[Dict[str, FiveTupleFlow], Dict[int, str]]:
    mode = config["flowlens"]["five_tuple_mode"]
    timeout = float(config["flowlens"]["five_tuple_timeout_seconds"])
    active: Dict[Tuple[str, str, str, str, str], FiveTupleFlow] = {}
    sequence: defaultdict[Tuple[str, str, str, str, str], int] = defaultdict(int)
    flows: Dict[str, FiveTupleFlow] = {}
    packet_to_flow_id: Dict[int, str] = {}

    for packet in sorted(packets, key=lambda item: item.timestamp):
        key = five_tuple_key(packet, mode)
        current = active.get(key)
        if current is None or packet.timestamp - current.end > timeout:
            sequence[key] += 1
            flow_id = "flow:{}>{}:{}>{}:{}:{}".format(key[0], key[1], key[2], key[3], key[4], sequence[key])
            current = FiveTupleFlow(flow_id, key[0], key[1], key[2], key[3], key[4], packet.timestamp, packet.timestamp)
            active[key] = current
            flows[flow_id] = current
        current.add_packet(packet, packet_size(packet, config))
        packet_to_flow_id[packet.packet_index] = current.flow_id
    return flows, packet_to_flow_id


def build_flows(packets: Sequence, packet_to_flow_id: Dict[int, str], config: dict) -> List[FlowLensFlow]:
    timegap = float(config["flowlens"]["timegap_seconds"])
    prepared = []
    for packet in packets:
        packet.timestamp = packet_timestamp(packet, config)
        prepared.append(packet)
    prepared.sort(key=lambda item: (canonical_ip_pair(item), item.timestamp))

    flows: List[FlowLensFlow] = []
    current: Optional[FlowLensFlow] = None
    for packet in prepared:
        key = canonical_ip_pair(packet)
        if current is None or current.key != key or packet.timestamp - current.end > timegap:
            current = FlowLensFlow(key[0], key[1], key)
            flows.append(current)
        current.add_packet(packet, packet_size(packet, config), packet_to_flow_id[packet.packet_index])
    return flows


def build_superflows(flows: Sequence[FlowLensFlow], config: dict) -> List[FlowLensFlow]:
    flowgap = float(config["flowlens"]["flowgap_seconds"])
    ordered = sorted(flows, key=lambda item: (item.key, item.start))
    superflows: List[FlowLensFlow] = []
    current: Optional[FlowLensFlow] = None
    for flow in ordered:
        if current is None or current.key != flow.key or flow.start - current.end > flowgap:
            current = FlowLensFlow(flow.ip1, flow.ip2, flow.key)
            current.merge(flow)
            superflows.append(current)
        else:
            current.merge(flow)
    return superflows


def write_superflow_dataset(superflows: Sequence[FlowLensFlow], labels: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "SUPERFLOW_ID",
                "IP1",
                "IP2",
                "PACKETS",
                "BYTES",
                "MEDIAN_IPT",
                "DURATION",
                "LABEL",
                "FIVE_TUPLE_FLOW_IDS",
            ]
        )
        for index, (flow, label) in enumerate(zip(superflows, labels)):
            writer.writerow(
                [
                    f"superflow_{index}",
                    flow.ip1,
                    flow.ip2,
                    flow.packets,
                    flow.bytes,
                    f"{flow.median_ipt:.9f}",
                    f"{flow.duration:.9f}",
                    label,
                    ";".join(sorted(flow.five_tuple_ids)),
                ]
            )


def metrics_dict(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy),
        "labels": list(labels),
        "confusion_matrix": cm.tolist(),
        "per_label": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run unified FlowLens classifier and flow-level evaluation.")
    parser.add_argument("--config", required=True, help="FlowLens classifier config JSON.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    parser_config = {
        "input": {
            "packet_label_delimiter": config["input"].get("packet_label_delimiter", ","),
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
    packet_labels = acc.load_packet_labels(config["input"].get("packet_labels", ""), parser_config)
    packet_iter = acc.iter_pcap(Path(config["input"]["pcap"]), parser_config, packet_labels)
    if config["input"].get("max_packets") is not None:
        packet_iter = islice(packet_iter, int(config["input"]["max_packets"]))
    packets = list(packet_iter)
    five_tuple_flows, packet_to_flow_id = build_five_tuple_flows(packets, config)
    flows = build_flows(packets, packet_to_flow_id, config)
    superflows = build_superflows(flows, config)

    superflow_policy = config["labels"]["superflow_label_policy"]
    five_tuple_policy = config["labels"]["five_tuple_label_policy"]
    labels = [label_from_votes(flow.label_votes, superflow_policy) for flow in superflows]
    valid = [(flow, label) for flow, label in zip(superflows, labels) if label and label != "mixed" and flow.packets > 0 and flow.bytes > 0 and flow.duration > 0]
    if len(valid) < 2:
        raise ValueError("Not enough labelled FlowLens superflows for classification")
    superflows = [item[0] for item in valid]
    labels = [item[1] for item in valid]

    features = np.asarray([[flow.packets, flow.bytes, flow.median_ipt, flow.duration] for flow in superflows], dtype=float)
    labels_np = np.asarray(labels)
    stratify = labels_np if config["split"].get("stratify", True) and min(Counter(labels).values()) >= 2 else None
    indices = np.arange(len(superflows))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=float(config["split"]["test_size"]),
        random_state=int(config["split"]["random_state"]),
        stratify=stratify,
    )

    model = RandomForestClassifier(random_state=int(config["classifier"]["random_state"]))
    model.fit(features[train_idx], labels_np[train_idx])
    predictions = model.predict(features[test_idx])
    probabilities = model.predict_proba(features[test_idx])
    classes = list(model.classes_)

    hard_path = output_dir / config["output"]["hard_predictions"]
    soft_path = output_dir / config["output"]["soft_predictions"]
    labels_path = output_dir / config["output"]["test_labels"]
    dataset_path = output_dir / config["output"]["superflow_dataset"]
    five_tuple_pred_path = output_dir / config["output"]["five_tuple_predictions"]
    metrics_path = output_dir / config["output"]["metrics"]

    hard_path.write_text("\n".join(predictions.tolist()) + "\n", encoding="utf-8")
    labels_path.write_text("\n".join(labels_np[test_idx].tolist()) + "\n", encoding="utf-8")
    np.save(soft_path, probabilities)
    write_superflow_dataset(superflows, labels, dataset_path)

    flow_true: List[str] = []
    flow_pred: List[str] = []
    with five_tuple_pred_path.open("w", newline="", encoding="utf-8") as handle:
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
                "TRUE_LABEL",
                "PRED_LABEL",
            ]
        )
        for superflow_index, prediction in zip(test_idx.tolist(), predictions.tolist()):
            superflow = superflows[superflow_index]
            for flow_id in sorted(superflow.five_tuple_ids):
                flow = five_tuple_flows.get(flow_id)
                if flow is None:
                    continue
                true_label = label_from_votes(flow.label_votes, five_tuple_policy)
                if not true_label or true_label == "mixed":
                    continue
                flow_true.append(true_label)
                flow_pred.append(prediction)
                writer.writerow(
                    [
                        flow.flow_id,
                        f"superflow_{superflow_index}",
                        flow.src_ip,
                        flow.dst_ip,
                        flow.src_port,
                        flow.dst_port,
                        flow.protocol,
                        f"{flow.start:.9f}",
                        f"{flow.end:.9f}",
                        true_label,
                        prediction,
                    ]
                )

    metric_labels = sorted(set(labels) | set(predictions.tolist()) | set(flow_true) | set(flow_pred))
    metrics = {
        "status": "completed",
        "algorithm": "FlowLens",
        "classifier": "RandomForestClassifier",
        "classifier_params": {"random_state": int(config["classifier"]["random_state"])},
        "feature_columns": ["NumberOfPackets", "TotalBytesTransmitted", "MedianIPT", "ConversationDuration"],
        "packet_size_bin_width": config["flowlens"]["packet_size_bin_width"],
        "ipt_bin_width_seconds": config["flowlens"]["ipt_bin_width_seconds"],
        "timegap_seconds": config["flowlens"]["timegap_seconds"],
        "flowgap_seconds": config["flowlens"]["flowgap_seconds"],
        "split": config["split"],
        "classes": classes,
        "superflows": len(superflows),
        "train_superflows": int(len(train_idx)),
        "test_superflows": int(len(test_idx)),
        "five_tuple_flows": len(five_tuple_flows),
        "evaluated_five_tuple_flows": len(flow_true),
        "superflow_metrics": metrics_dict(labels_np[test_idx].tolist(), predictions.tolist(), metric_labels),
        "flow_metrics": metrics_dict(flow_true, flow_pred, metric_labels) if flow_true else {},
        "hard_predictions": str(hard_path),
        "soft_predictions": str(soft_path),
        "test_labels": str(labels_path),
        "five_tuple_predictions": str(five_tuple_pred_path),
        "superflow_dataset": str(dataset_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "status=completed superflow_accuracy={:.6f} flow_accuracy={:.6f} output_dir={}".format(
            metrics["superflow_metrics"]["accuracy"],
            metrics["flow_metrics"].get("accuracy", 0.0),
            output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
