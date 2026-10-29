from __future__ import annotations

import csv
import importlib.util
import json
import math
import struct
import hashlib
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

from .utils import BenchmarkError, UnsupportedAlgorithmMode, rel, resolve, run_command, runtime_python, write_json


BINARY_BENIGN = {"0", "0.0", "benign", "normal", "false"}
BINARY_MALICIOUS = {"1", "1.0", "malicious", "attack", "true"}
FLOW_ALGORITHMS = {"TrafficFormer", "NetMamba"}
PACKET_ALGORITHMS = {"Kitsune", "FlowLens", "TFusion"}
USTC_BENIGN_CLASSES = {
    "bittorrent",
    "facetime",
    "ftp",
    "gmail",
    "mysql",
    "outlook",
    "skype",
    "smb",
    "weibo",
    "worldofwarcraft",
}
USTC_MALWARE_CLASSES = {
    "cridex",
    "geodo",
    "htbot",
    "miuref",
    "neris",
    "nsis-ay",
    "shifu",
    "tinba",
    "virut",
    "zeus",
}
ANT_MALICIOUS_TASK_FAMILIES = {"attack_code_generation", "attack_execution"}
PCAP_SUFFIXES = {".pcap", ".pcapng"}
LABEL_SUFFIXES = {".txt", ".csv"}


@dataclass
class PacketDataset:
    pcap: Path
    packet_labels: Path
    source: str


@dataclass
class PreparedData:
    algorithm: str
    grain: str
    work_dir: Path
    train_path: Path | None = None
    dev_path: Path | None = None
    test_path: Path | None = None
    label_mapping_path: Path | None = None
    training_packets: list[PacketDataset] | None = None
    test_packets: list[PacketDataset] | None = None
    metadata: dict[str, Any] | None = None
    validation_packets: list[PacketDataset] | None = None


@dataclass
class GeneratedFlow:
    key: tuple[str, str, str, str, str]
    src_ip: str
    dst_ip: str
    src_port: str
    dst_port: str
    protocol: str
    start_time: float
    end_time: float
    packets: int = 0
    bytes: int = 0
    ipts: list[float] = field(default_factory=list)
    label_votes: Counter = field(default_factory=Counter)

    def add(self, packet) -> None:
        if self.packets > 0:
            self.ipts.append(max(0.0, float(packet.timestamp) - self.end_time))
        self.start_time = min(self.start_time, float(packet.timestamp))
        self.end_time = max(self.end_time, float(packet.timestamp))
        self.packets += 1
        self.bytes += int(packet.payload_length)
        if getattr(packet, "label", ""):
            self.label_votes[str(packet.label)] += 1

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def median_ipt(self) -> float:
        return float(median(self.ipts)) if self.ipts else 0.0


def _read_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _source_specs(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        payload = _read_manifest(path)
        if "sources" in payload:
            specs = payload["sources"]
        else:
            specs = [payload]
        result = []
        for spec in specs:
            item = dict(spec)
            if "path" in item:
                item["path"] = str(resolve(item["path"]))
            result.append(item)
        return result
    if path.is_dir():
        csvs = [item for item in sorted(path.rglob("*.csv")) if item.is_file() and "label" not in item.name.lower()]
        pcap_flow_csvs = [item for item in csvs if item.name == "pcap_flows.csv"]
        if pcap_flow_csvs:
            csvs = pcap_flow_csvs
        return [{"path": str(item), "source": _source_name_from_path(item)} for item in csvs]
    return [{"path": str(path), "source": path.stem}]


def _clean_class_name(value: str) -> str:
    text = value.strip()
    for suffix in ("_packet_labels", "_labels"):
        if text.lower().endswith(suffix):
            return text[: -len(suffix)]
    return text


def _source_name_from_path(path: Path) -> str:
    if path.name in {"pcap_flows.csv", "cat_flows.csv", "network.pcap"} and len(path.parts) >= 3:
        return path.parent.name
    return _clean_class_name(path.stem)


def _ant_category(path: Path) -> str | None:
    parts = path.parts
    for split in ("train", "val", "test"):
        if split not in parts:
            continue
        index = parts.index(split)
        if index >= 1 and parts[index - 1] == "ANT" and len(parts) > index + 2:
            return parts[index + 2]
    return None


def infer_label_from_path(path: Path) -> str:
    parts_lower = [part.lower() for part in path.parts]
    stem = _clean_class_name(path.stem)
    stem_lower = stem.lower()

    if "ant" in parts_lower:
        category = _ant_category(path)
        if category:
            return "malicious" if category in ANT_MALICIOUS_TASK_FAMILIES else "benign"

    if "ustc-tfc2016" in parts_lower:
        if stem_lower in USTC_BENIGN_CLASSES:
            return "benign"
        if stem_lower in USTC_MALWARE_CLASSES:
            return "malicious"

    if "iscx-nonvpn" in parts_lower:
        return stem

    if "benign" in parts_lower or stem_lower in BINARY_BENIGN:
        return "benign"
    if "malware" in parts_lower or "malicious" in parts_lower or stem_lower in BINARY_MALICIOUS:
        return "malicious"
    return stem


def infer_packet_label_from_path(path: Path) -> str:
    parts_lower = [part.lower() for part in path.parts]
    if "iscx-nonvpn" in parts_lower:
        return "benign"
    return infer_label_from_path(path)


def protocol_number(value: object) -> int:
    text = str(value).strip().lower()
    if text in {"tcp", "6"}:
        return 6
    if text in {"udp", "17"}:
        return 17
    if text in {"icmp", "1"}:
        return 1
    try:
        return int(float(text))
    except ValueError:
        return 0


def canonicalize_flow_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "src_port": "SRC_PORT",
        "dst_port": "DST_PORT",
        "protocol": "PROTOCOL",
        "start_ts": "START_TIME",
        "end_ts": "END_TIME",
        "pkt_count": "PACKETS",
        "byte_count": "BYTES",
        "duration_s": "DURATION",
    }
    updates = {old: new for old, new in rename.items() if old in frame.columns and new not in frame.columns}
    if updates:
        frame = frame.rename(columns=updates)
    if "PROTOCOL" in frame.columns:
        frame["PROTOCOL"] = frame["PROTOCOL"].map(protocol_number)
    return frame


def _load_flow_csv(spec: dict[str, Any]) -> pd.DataFrame:
    path = resolve(spec["path"])
    if path.suffix.lower() != ".csv":
        raise BenchmarkError(f"Flow datasets must be CSV files or manifests containing CSV paths: {path}")
    frame = pd.read_csv(path)
    label_column = spec.get("label_column", "LABEL")
    if label_column in frame.columns and label_column != "LABEL":
        frame = frame.rename(columns={label_column: "LABEL"})
    if "LABEL" not in frame.columns:
        frame["LABEL"] = infer_label_from_path(path)
    frame["LABEL"] = frame["LABEL"].map(lambda value: str(value).strip())
    frame["SOURCE_DATASET"] = str(spec.get("source", path.stem))
    frame["SOURCE_FILE"] = rel(path)
    return canonicalize_flow_columns(frame)


def load_flow_sources(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        for spec in _source_specs(path):
            frames.append(_load_flow_csv(spec))
    if not frames:
        raise BenchmarkError("No flow CSV inputs were loaded")
    return pd.concat(frames, ignore_index=True)


def _load_flowlens_parser():
    module_path = resolve("Benchmark/2021-NDSS-FlowLens/unified_flowlens.py")
    spec = importlib.util.spec_from_file_location("benchmark_data_flowlens_parser", module_path)
    if spec is None or spec.loader is None:
        raise BenchmarkError(f"Cannot load pcap parser module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_data_flowlens_parser"] = module
    spec.loader.exec_module(module)
    return module


def _read_packet_labels(path: Path, max_labels: int | None = None) -> list[str]:
    labels: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            value = row[-1].strip()
            if value.lower() in {"label", "labels", "x"}:
                continue
            labels.append("benign" if value.lower() in BINARY_BENIGN else "malicious")
            if max_labels is not None and len(labels) >= max_labels:
                break
    return labels


def _pcap_label_specs(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        payload = _read_manifest(path)
        return [dict(spec) for spec in payload.get("sources", [payload])]
    if path.is_dir():
        pcaps = sorted([p for p in path.rglob("*") if p.suffix.lower() in PCAP_SUFFIXES])
        labels = sorted([p for p in path.rglob("*") if "label" in p.name.lower() and p.suffix.lower() in LABEL_SUFFIXES])
        specs = []
        for pcap in pcaps:
            label_path = _paired_label_path(pcap, labels)
            spec: dict[str, Any] = {"pcap": str(pcap), "source": _source_name_from_path(pcap)}
            if label_path is not None:
                spec["packet_labels"] = str(label_path)
            else:
                spec["default_label"] = infer_packet_label_from_path(pcap)
            specs.append(spec)
        return specs
    if path.suffix.lower() in PCAP_SUFFIXES:
        return [{"pcap": str(path), "source": _source_name_from_path(path), "default_label": infer_packet_label_from_path(path)}]
    return []


def _canonical_five_tuple(packet, mode: str) -> tuple[str, str, str, str, str]:
    src = (str(packet.src_ip), str(packet.src_port))
    dst = (str(packet.dst_ip), str(packet.dst_port))
    protocol = str(packet.protocol)
    if mode == "unidirectional":
        return (src[0], dst[0], src[1], dst[1], protocol)
    if src <= dst:
        return (src[0], dst[0], src[1], dst[1], protocol)
    return (dst[0], src[0], dst[1], src[1], protocol)


def _flow_label(flow: GeneratedFlow, default_label: str = "unknown") -> str:
    if not flow.label_votes:
        return default_label
    return flow.label_votes.most_common(1)[0][0]


def _packet_spec_to_flow_csv(spec: dict[str, Any], output_csv: Path, config: dict[str, Any]) -> dict[str, Any]:
    parser = _load_flowlens_parser()
    conversion = config.get("algorithm_config", {}).get("conversion", {})
    max_packets = conversion.get("max_packets_per_source")
    max_packets = int(max_packets) if max_packets is not None else None
    timeout = float(conversion.get("five_tuple_timeout_seconds", 120.0))
    mode = str(conversion.get("five_tuple_mode", "bidirectional"))
    pcap = resolve(spec.get("pcap", spec.get("path", "")))
    label_path = spec.get("packet_labels", spec.get("labels"))
    labels = _read_packet_labels(resolve(label_path), max_packets) if label_path else []
    default_label = str(spec.get("default_label", infer_packet_label_from_path(pcap)))
    parser_config = {
        "flow": {
            "skip_zero_payload_packets": bool(conversion.get("skip_zero_payload_packets", False)),
        }
    }
    packets = []
    for packet in parser.iter_pcap(pcap, parser_config, labels):
        if not getattr(packet, "label", ""):
            packet.label = default_label
        packets.append(packet)
        if max_packets is not None and len(packets) >= max_packets:
            break
    packets.sort(key=lambda item: (_canonical_five_tuple(item, mode), float(item.timestamp)))
    flows: list[GeneratedFlow] = []
    current: GeneratedFlow | None = None
    for packet in packets:
        key = _canonical_five_tuple(packet, mode)
        if current is None or current.key != key or (float(packet.timestamp) - current.end_time) > timeout:
            current = GeneratedFlow(
                key=key,
                src_ip=key[0],
                dst_ip=key[1],
                src_port=key[2],
                dst_port=key[3],
                protocol=key[4],
                start_time=float(packet.timestamp),
                end_time=float(packet.timestamp),
            )
            flows.append(current)
        current.add(packet)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "FLOW_ID",
                "SRC_IP",
                "DST_IP",
                "SRC_PORT",
                "DST_PORT",
                "PROTOCOL",
                "START_TIME",
                "END_TIME",
                "PACKETS",
                "BYTES",
                "MEDIAN_IPT",
                "DURATION",
                "LABEL",
            ]
        )
        for index, flow in enumerate(flows):
            writer.writerow(
                [
                    f"{pcap.stem}_flow_{index}",
                    flow.src_ip,
                    flow.dst_ip,
                    flow.src_port,
                    flow.dst_port,
                    flow.protocol,
                    f"{flow.start_time:.9f}",
                    f"{flow.end_time:.9f}",
                    flow.packets,
                    flow.bytes,
                    f"{flow.median_ipt:.9f}",
                    f"{flow.duration:.9f}",
                    _flow_label(flow, default_label),
                ]
            )
    return {
        "pcap": rel(pcap),
        "packet_labels": rel(label_path) if label_path else None,
        "default_label": default_label if not label_path else None,
        "packets": len(packets),
        "flows": len(flows),
        "output_csv": rel(output_csv),
    }


def load_flow_inputs(paths: list[Path], config: dict[str, Any], generated_root: Path, role: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for path_index, path in enumerate(paths):
        try:
            specs = _source_specs(path)
            csv_specs = [spec for spec in specs if resolve(spec["path"]).suffix.lower() == ".csv"]
            if csv_specs:
                for spec in csv_specs:
                    frames.append(_load_flow_csv(spec))
                    metadata.append({"source": rel(spec["path"]), "kind": "csv"})
                continue
        except BenchmarkError:
            pass
        packet_specs = _pcap_label_specs(path)
        if not packet_specs:
            raise BenchmarkError(f"Flow input path has neither flow CSV nor pcap sources: {path}")
        for spec_index, spec in enumerate(packet_specs):
            output_csv = generated_root / role / f"source_{path_index:03d}_{spec_index:04d}.csv"
            summary = _packet_spec_to_flow_csv(spec, output_csv, config)
            frames.append(_load_flow_csv({"path": str(output_csv), "source": spec.get("source", output_csv.stem)}))
            metadata.append({"source": rel(spec.get("pcap", spec.get("path", ""))), "kind": "generated_from_pcap", **summary})
    if not frames:
        raise BenchmarkError("No flow inputs were loaded")
    return pd.concat(frames, ignore_index=True), metadata


def _packet_from_spec(spec: dict[str, Any], source: str, generated_label_dir: Path | None = None) -> PacketDataset:
    pcap = resolve(spec.get("pcap", spec.get("path", "")))
    label_value = spec.get("packet_labels", spec.get("labels", ""))
    labels = resolve(label_value) if label_value else None
    if labels is None and "default_label" in spec and generated_label_dir is not None:
        labels = _write_inferred_packet_labels(pcap, str(spec["default_label"]), generated_label_dir)
    if labels is None or not pcap.exists() or not labels.exists():
        raise BenchmarkError(f"Packet source must contain existing pcap and packet_labels paths: {source}")
    return PacketDataset(pcap=pcap, packet_labels=labels, source=str(spec.get("source", source)))


def _paired_label_path(pcap: Path, all_labels: list[Path]) -> Path | None:
    label_names = {
        f"{pcap.stem}_packet_labels",
        f"{pcap.stem}_labels",
        pcap.stem,
    }
    same_dir = [label for label in all_labels if label.parent == pcap.parent and _clean_class_name(label.stem) in label_names]
    if same_dir:
        return sorted(same_dir)[0]
    return None


def _pcap_packet_count(path: Path) -> int:
    with path.open("rb") as handle:
        header = handle.read(24)
        if len(header) != 24:
            raise BenchmarkError(f"Invalid pcap header: {path}")
        magic = header[:4]
        if magic in {b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1"}:
            endian = "<"
        elif magic in {b"\xa1\xb2\xc3\xd4", b"\xa1\xb2\x3c\x4d"}:
            endian = ">"
        else:
            raise BenchmarkError(f"Cannot infer packet labels for non-pcap input without a label file: {path}")
        unpack = struct.Struct(endian + "IIII").unpack
        packets = 0
        while True:
            record_header = handle.read(16)
            if not record_header:
                return packets
            if len(record_header) != 16:
                warnings.warn(f"Skipping truncated pcap record header at end of file: {path}", RuntimeWarning)
                return packets
            _ts_sec, _ts_sub, incl_len, _orig_len = unpack(record_header)
            if len(handle.read(incl_len)) != incl_len:
                warnings.warn(f"Skipping truncated pcap packet payload at end of file: {path}", RuntimeWarning)
                return packets
            packets += 1


def _write_inferred_packet_labels(pcap: Path, label: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(pcap.resolve()).encode("utf-8")).hexdigest()[:12]
    label_path = output_dir / f"{pcap.stem}_{digest}_packet_labels.txt"
    count = _pcap_packet_count(pcap)
    with label_path.open("w", encoding="utf-8") as handle:
        for _ in range(count):
            handle.write(label + "\n")
    return label_path


def _discover_packet_datasets(path: Path, generated_label_dir: Path | None = None) -> list[PacketDataset]:
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        payload = _read_manifest(path)
        specs = payload.get("sources", [payload])
        return [_packet_from_spec(dict(spec), path.stem, generated_label_dir) for spec in specs]
    if path.is_dir():
        pcaps = sorted([p for p in path.rglob("*") if p.suffix.lower() in PCAP_SUFFIXES])
        labels = sorted([p for p in path.rglob("*") if "label" in p.name.lower() and p.suffix.lower() in LABEL_SUFFIXES])
        datasets: list[PacketDataset] = []
        for pcap in pcaps:
            label_path = _paired_label_path(pcap, labels)
            if label_path is None:
                if generated_label_dir is None:
                    raise BenchmarkError(f"Packet pcap has no paired label file and no generated label directory: {pcap}")
                label_path = _write_inferred_packet_labels(pcap, infer_packet_label_from_path(pcap), generated_label_dir)
            datasets.append(PacketDataset(pcap=pcap, packet_labels=label_path, source=_source_name_from_path(pcap)))
        if not datasets:
            raise BenchmarkError(f"Packet directory does not contain any pcap/pcapng files: {path}")
        return datasets
    if path.suffix.lower() in PCAP_SUFFIXES:
        if generated_label_dir is None:
            raise BenchmarkError(f"Packet pcap has no paired label file and no generated label directory: {path}")
        return [PacketDataset(pcap=path, packet_labels=_write_inferred_packet_labels(path, infer_packet_label_from_path(path), generated_label_dir), source=_source_name_from_path(path))]
    raise BenchmarkError("Packet datasets require a manifest JSON/YAML or a directory with one pcap and one label file")


def load_packet_sources(paths: list[Path], role: str, generated_label_root: Path | None = None) -> list[PacketDataset]:
    datasets: list[PacketDataset] = []
    for path in paths:
        generated_label_dir = generated_label_root / role if generated_label_root else None
        datasets.extend(_discover_packet_datasets(path, generated_label_dir))
    if not datasets:
        raise BenchmarkError("No packet inputs were loaded")
    return datasets


def normalize_label(value: object, mapping: dict[str, int]) -> int:
    text = str(value).strip()
    if mapping == {"benign": 0, "malicious": 1}:
        return 0 if text.lower() in BINARY_BENIGN else 1
    return mapping[text]


def build_label_mapping(train: pd.Series, test: pd.Series, mode: str) -> dict[str, int]:
    values = [str(value).strip() for value in pd.concat([train, test], ignore_index=True)]
    lowered = {value.lower() for value in values}
    if mode == "binary" or (mode == "auto" and lowered <= (BINARY_BENIGN | BINARY_MALICIOUS)):
        return {"benign": 0, "malicious": 1}
    labels = sorted(set(values), key=lambda value: value.casefold())
    return {label: index for index, label in enumerate(labels)}


def stratify_or_none(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    return labels if len(counts) > 1 and int(counts.min()) >= 2 else None


def split_train_dev(frame: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation_size = float(config.get("data", {}).get("validation_size", 0.1))
    random_state = int(config.get("data", {}).get("split_random_state", 42))
    if validation_size <= 0.0 or len(frame) < 2:
        return frame.reset_index(drop=True), frame.iloc[:0].copy()
    train, dev = train_test_split(
        frame,
        test_size=validation_size,
        random_state=random_state,
        stratify=stratify_or_none(frame["LABEL"]),
    )
    return train.sort_index().reset_index(drop=True), dev.sort_index().reset_index(drop=True)


def limit_rows(frame: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    labels = frame["LABEL"].astype(str)
    selected_parts = []
    selected_indices: set[int] = set()
    if max_rows >= labels.nunique():
        for _label, group in frame.groupby(labels, sort=True):
            sampled = group.sample(n=1, random_state=random_state)
            selected_parts.append(sampled)
            selected_indices.update(int(index) for index in sampled.index)
    remaining_rows = max_rows - len(selected_indices)
    if remaining_rows <= 0:
        return pd.concat(selected_parts).sort_index().reset_index(drop=True)
    if selected_indices:
        remaining_frame = frame.drop(index=list(selected_indices))
        remaining_stratify = stratify_or_none(remaining_frame["LABEL"])
        if len(remaining_frame) <= remaining_rows:
            sampled_remaining = remaining_frame
        elif remaining_stratify is not None:
            sampled_remaining, _ = train_test_split(
                remaining_frame,
                train_size=remaining_rows,
                random_state=random_state,
                stratify=remaining_stratify,
            )
        else:
            sampled_remaining = remaining_frame.sample(n=remaining_rows, random_state=random_state)
        return pd.concat(selected_parts + [sampled_remaining]).sort_index().reset_index(drop=True)
    stratify = stratify_or_none(frame["LABEL"])
    if stratify is not None:
        sampled, _ = train_test_split(
            frame,
            train_size=max_rows,
            random_state=random_state,
            stratify=stratify,
        )
        return sampled.sort_index().reset_index(drop=True)
    return frame.sample(n=max_rows, random_state=random_state).sort_index().reset_index(drop=True)


def apply_flow_row_limits(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
    val: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    conversion = config.get("algorithm_config", {}).get("conversion", {})
    random_state = int(conversion.get("random_state", config.get("data", {}).get("split_random_state", 42)))
    max_rows = conversion.get("max_rows")
    max_train_rows = conversion.get("max_train_rows", max_rows)
    max_test_rows = conversion.get("max_test_rows", max_rows)
    max_val_rows = conversion.get("max_val_rows", max_rows)
    max_train_rows = int(max_train_rows) if max_train_rows is not None else None
    max_test_rows = int(max_test_rows) if max_test_rows is not None else None
    if val is None:
        return limit_rows(train, max_train_rows, random_state), limit_rows(test, max_test_rows, random_state + 1)
    max_val_rows = int(max_val_rows) if max_val_rows is not None else None
    return (
        limit_rows(train, max_train_rows, random_state),
        limit_rows(test, max_test_rows, random_state + 1),
        limit_rows(val, max_val_rows, random_state + 2),
    )


def int_to_hex_tokens(value: int, width: int = 4) -> list[str]:
    text = f"{max(0, value):0{width}x}"[-width:]
    return [text[i : i + 2] for i in range(0, len(text), 2)]


def row_to_trafficformer_text(row: pd.Series, max_tokens: int) -> str:
    fields = [
        int(float(row.get("PROTOCOL", 0) or 0)),
        int(float(row.get("SRC_PORT", 0) or 0)),
        int(float(row.get("DST_PORT", 0) or 0)),
        int(float(row.get("PACKETS", row.get("pkt_count", 0)) or 0)),
        int(float(row.get("BYTES", row.get("byte_count", 0)) or 0)),
        int(float(row.get("DURATION", row.get("duration_s", 0)) or 0) * 1000),
        int(float(row.get("MEDIAN_IPT", 0) or 0) * 1000000),
    ]
    tokens: list[str] = []
    for value in fields:
        tokens.extend(int_to_hex_tokens(value))
    while len(tokens) < max_tokens:
        tokens.extend(tokens[: max(1, min(len(tokens), max_tokens - len(tokens)))])
    return " ".join(tokens[:max_tokens])


def write_trafficformer_tsv(frame: pd.DataFrame, path: Path, mapping: dict[str, int], max_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("label\ttext_a\n")
        for _, row in frame.iterrows():
            handle.write(f"{normalize_label(row['LABEL'], mapping)}\t{row_to_trafficformer_text(row, max_tokens)}\n")


def ant_metadata_from_source_file(source_file: object) -> dict[str, str]:
    parts = Path(str(source_file)).parts
    for split in ("train", "val", "test"):
        if split not in parts:
            continue
        index = parts.index(split)
        if index >= 1 and parts[index - 1] == "ANT" and len(parts) > index + 3:
            return {
                "ant_split": split,
                "ant_engine": parts[index + 1],
                "ant_task_family": parts[index + 2],
                "ant_episode": parts[index + 3],
            }
    return {"ant_split": "", "ant_engine": "", "ant_task_family": "", "ant_episode": ""}


def write_flow_row_metadata(frame: pd.DataFrame, path: Path, order: list[int] | None = None) -> None:
    rows = frame.reset_index(drop=True)
    if order is not None:
        rows = rows.iloc[order].reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "ROW_INDEX",
                "TRUE_LABEL",
                "SOURCE_DATASET",
                "SOURCE_FILE",
                "ANT_SPLIT",
                "ANT_ENGINE",
                "ANT_TASK_FAMILY",
                "ANT_EPISODE",
            ]
        )
        for output_index, (_row_index, row) in enumerate(rows.iterrows()):
            source_file = row.get("SOURCE_FILE", "")
            ant = ant_metadata_from_source_file(source_file)
            writer.writerow(
                [
                    output_index,
                    str(row.get("LABEL", "")).strip(),
                    str(row.get("SOURCE_DATASET", "")).strip(),
                    str(source_file),
                    ant["ant_split"],
                    ant["ant_engine"],
                    ant["ant_task_family"],
                    ant["ant_episode"],
                ]
            )


def class_dir_name(label: str, index: int, mapping: dict[str, int]) -> str:
    if mapping == {"benign": 0, "malicious": 1}:
        return label
    slug = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)
    return f"{index:03d}_{slug}"


def numeric_matrix(
    frame: pd.DataFrame,
    lo: np.ndarray | None = None,
    hi: np.ndarray | None = None,
    columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    preferred = ["SRC_PORT", "DST_PORT", "PROTOCOL", "START_TIME", "END_TIME", "PACKETS", "BYTES", "MEDIAN_IPT", "DURATION"]
    cols = list(columns) if columns is not None else [column for column in preferred if column in frame.columns]
    if not cols and columns is None:
        cols = [column for column in frame.columns if column != "LABEL" and pd.api.types.is_numeric_dtype(frame[column])]
    if not cols:
        raise BenchmarkError("No numeric columns are available for NetMamba image conversion")
    values_frame = pd.DataFrame(index=frame.index)
    for column in cols:
        if column in frame.columns:
            values_frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        else:
            values_frame[column] = 0.0
    values = values_frame[cols].to_numpy(dtype=np.float32)
    lo = np.nanmin(values, axis=0) if lo is None else lo
    hi = np.nanmax(values, axis=0) if hi is None else hi
    if len(lo) != values.shape[1] or len(hi) != values.shape[1]:
        raise BenchmarkError(
            f"NetMamba numeric normalization shape mismatch: values={values.shape[1]} lo={len(lo)} hi={len(hi)} columns={cols}"
        )
    span = np.where(np.isclose(hi, lo), 1.0, hi - lo)
    return np.clip((values - lo) / span, 0.0, 1.0), lo, hi, cols


def row_to_image(values: np.ndarray, size: int) -> Image.Image:
    needed = size * size
    repeats = int(math.ceil(needed / len(values)))
    flat = np.tile(values, repeats)[:needed]
    return Image.fromarray((flat.reshape(size, size) * 255.0).astype(np.uint8), mode="L")


def write_netmamba_split(frame: pd.DataFrame, values: np.ndarray, split: str, output_dir: Path, size: int, mapping: dict[str, int]) -> dict[str, int]:
    counts = {label: 0 for label in mapping}
    for row_index, row in frame.reset_index(drop=True).iterrows():
        label = str(row["LABEL"]).strip()
        if mapping == {"benign": 0, "malicious": 1}:
            label = "benign" if label.lower() in BINARY_BENIGN else "malicious"
        cls_dir = output_dir / split / class_dir_name(label, mapping[label], mapping)
        cls_dir.mkdir(parents=True, exist_ok=True)
        row_to_image(values[row_index], size).save(cls_dir / f"flow_{row_index:08d}.png")
        counts[label] += 1
    return counts


def prepare_trafficformer(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame, mapping: dict[str, int], config: dict[str, Any], out_dir: Path) -> PreparedData:
    alg = config["algorithm_config"]
    max_tokens = int(alg.get("conversion", {}).get("max_tokens", alg.get("finetune", {}).get("seq_length", 320)))
    if dev.empty:
        dev = train.copy()
    eval_only = bool(alg.get("eval_checkpoint_path"))
    if not eval_only:
        write_trafficformer_tsv(train, out_dir / "train.tsv", mapping, max_tokens)
        write_trafficformer_tsv(dev, out_dir / "dev.tsv", mapping, max_tokens)
    write_trafficformer_tsv(test, out_dir / "test.tsv", mapping, max_tokens)
    write_flow_row_metadata(test, out_dir / "test_row_metadata.csv")
    mapping_payload = {"label_to_id": mapping, "id_to_label": {str(index): label for label, index in mapping.items()}}
    mapping_path = out_dir / "label_mapping.json"
    write_json(mapping_path, mapping_payload)
    return PreparedData(
        "TrafficFormer",
        "flow",
        out_dir,
        None if eval_only else out_dir / "train.tsv",
        None if eval_only else out_dir / "dev.tsv",
        out_dir / "test.tsv",
        mapping_path,
    )


def _collect_packet_specs(paths: list[Path]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for path in paths:
        path_specs = _pcap_label_specs(path)
        if not path_specs:
            raise BenchmarkError(f"TrafficFormer native datagram input requires pcap/pcapng sources, got: {path}")
        for spec in path_specs:
            item = dict(spec)
            if "pcap" in item:
                item["pcap"] = str(resolve(item["pcap"]))
            if "path" in item and "pcap" not in item:
                item["pcap"] = str(resolve(item["path"]))
            if "packet_labels" in item:
                item["packet_labels"] = str(resolve(item["packet_labels"]))
            if "labels" in item:
                item["labels"] = str(resolve(item["labels"]))
            item.setdefault("default_label", infer_packet_label_from_path(resolve(item["pcap"])))
            specs.append(item)
    return specs


def prepare_trafficformer_native(
    training_paths: list[Path],
    validation_paths: list[Path] | None,
    test_paths: list[Path],
    config: dict[str, Any],
    out_dir: Path,
) -> PreparedData:
    alg = config["algorithm_config"]
    conversion = dict(alg.get("conversion", {}))
    label_mode = str(config.get("data", {}).get("label_mode", conversion.get("label_mode", "auto")))
    manifest = {
        "label_mode": label_mode,
        "conversion": conversion,
        "train_sources": _collect_packet_specs(training_paths),
        "dev_sources": _collect_packet_specs(validation_paths) if validation_paths else [],
        "test_sources": _collect_packet_specs(test_paths),
    }
    manifest_path = out_dir / "trafficformer_native_manifest.json"
    write_json(manifest_path, manifest)
    script = resolve("Benchmark/scripts/prepare_trafficformer_native.py")
    cmd = [
        runtime_python(config),
        str(script),
        "--manifest",
        str(manifest_path.resolve()),
        "--output-dir",
        str(out_dir.resolve()),
    ]
    rc = run_command(cmd, resolve("."), out_dir / "trafficformer_native_prepare.log", config, None)
    if rc != 0:
        raise RuntimeError(f"TrafficFormer native datagram preparation failed with exit code {rc}")
    mapping_path = out_dir / "label_mapping.json"
    summary_path = out_dir / "native_conversion_summary.json"
    metadata = {
        "native_input_format": "TrafficFormer native datagram TSV",
        "existing_conversion_scripts_used": True,
        "native_conversion_script": rel(script),
        "native_function": "Benchmark/2025-SP-TrafficFormer/data_generation/finetuning_data_gen.py:get_feature_flow",
        "legacy_flow_feature_tsv_used": False,
        "benchmark_split_preserved": True,
        "training_sources": [rel(path) for path in training_paths],
        "validation_sources": [rel(path) for path in validation_paths] if validation_paths else [],
        "test_sources": [rel(path) for path in test_paths],
    }
    if summary_path.exists():
        metadata["native_conversion_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    write_json(out_dir / "metadata.json", metadata)
    return PreparedData(
        "TrafficFormer",
        "flow",
        out_dir,
        train_path=out_dir / "train.tsv",
        dev_path=out_dir / "dev.tsv",
        test_path=out_dir / "test.tsv",
        label_mapping_path=mapping_path,
        metadata=metadata,
    )


def prepare_netmamba_native(
    training_paths: list[Path],
    validation_paths: list[Path] | None,
    test_paths: list[Path],
    config: dict[str, Any],
    out_dir: Path,
) -> PreparedData:
    alg = config["algorithm_config"]
    conversion = dict(alg.get("conversion", {}))
    label_mode = str(config.get("data", {}).get("label_mode", conversion.get("label_mode", "auto")))
    manifest = {
        "label_mode": label_mode,
        "conversion": conversion,
        "train_sources": _collect_packet_specs(training_paths),
        "valid_sources": _collect_packet_specs(validation_paths) if validation_paths else [],
        "test_sources": _collect_packet_specs(test_paths),
    }
    manifest_path = out_dir / "netmamba_native_manifest.json"
    write_json(manifest_path, manifest)
    script = resolve("Benchmark/scripts/prepare_netmamba_native.py")
    cmd = [
        runtime_python(config),
        str(script),
        "--manifest",
        str(manifest_path.resolve()),
        "--output-dir",
        str(out_dir.resolve()),
    ]
    rc = run_command(cmd, resolve("."), out_dir / "netmamba_native_prepare.log", config, None)
    if rc != 0:
        raise RuntimeError(f"NetMamba native 5-packet image preparation failed with exit code {rc}")
    mapping_path = out_dir / "label_mapping.json"
    summary_path = out_dir / "native_conversion_summary.json"
    metadata = {
        "native_input_format": "NetMamba native 5-packet byte ImageFolder",
        "existing_conversion_scripts_used": True,
        "native_conversion_script": rel(script),
        "native_function": "Benchmark/2024-ICNP-NetMamba/dataset/dataset_common.py:read_5hp_list",
        "legacy_flow_feature_image_used": False,
        "benchmark_split_preserved": True,
        "training_sources": [rel(path) for path in training_paths],
        "validation_sources": [rel(path) for path in validation_paths] if validation_paths else [],
        "test_sources": [rel(path) for path in test_paths],
    }
    if summary_path.exists():
        metadata["native_conversion_summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata["label_mapping"] = metadata["native_conversion_summary"].get("mapping", {})
    write_json(out_dir / "metadata.json", metadata)
    return PreparedData(
        "NetMamba",
        "flow",
        out_dir,
        train_path=out_dir / "train",
        dev_path=out_dir / "valid",
        test_path=out_dir / "test",
        label_mapping_path=mapping_path,
        metadata=metadata,
    )


def prepare_netmamba(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame, mapping: dict[str, int], config: dict[str, Any], out_dir: Path) -> PreparedData:
    alg = config["algorithm_config"]
    image_size = int(alg.get("conversion", {}).get("image_size", 40))
    if dev.empty:
        dev = train.copy()
    eval_only = bool(alg.get("eval_checkpoint_path"))
    train_values, lo, hi, numeric_columns = numeric_matrix(train)
    test_values, _, _, _ = numeric_matrix(test, lo, hi, numeric_columns)
    splits = {}
    if not eval_only:
        dev_values, _, _, _ = numeric_matrix(dev, lo, hi, numeric_columns)
        splits["train"] = write_netmamba_split(train, train_values, "train", out_dir, image_size, mapping)
        splits["valid"] = write_netmamba_split(dev, dev_values, "valid", out_dir, image_size, mapping)
    splits["test"] = write_netmamba_split(test, test_values, "test", out_dir, image_size, mapping)
    test_reset = test.reset_index(drop=True)
    imagefolder_order: list[int] = []
    for label, index in sorted(mapping.items(), key=lambda item: class_dir_name(item[0], item[1], mapping)):
        native_label = label
        if mapping == {"benign": 0, "malicious": 1}:
            imagefolder_order.extend(
                int(row_index)
                for row_index, value in enumerate(test_reset["LABEL"])
                if ("benign" if str(value).strip().lower() in BINARY_BENIGN else "malicious") == native_label
            )
        else:
            imagefolder_order.extend(
                int(row_index)
                for row_index, value in enumerate(test_reset["LABEL"])
                if str(value).strip() == native_label
            )
    write_flow_row_metadata(test, out_dir / "test_row_metadata.csv", imagefolder_order)
    mapping_payload = {
        "label_to_id": mapping,
        "id_to_label": {str(index): label for label, index in mapping.items()},
        "class_directories": {label: class_dir_name(label, index, mapping) for label, index in mapping.items()},
    }
    mapping_path = out_dir / "label_mapping.json"
    write_json(mapping_path, mapping_payload)
    write_json(out_dir / "conversion_summary.json", {"splits": splits, "num_classes": len(mapping), "image_size": image_size})
    return PreparedData(
        "NetMamba",
        "flow",
        out_dir,
        train_path=None if eval_only else out_dir / "train",
        dev_path=None if eval_only else out_dir / "valid",
        test_path=out_dir / "test",
        label_mapping_path=mapping_path,
    )


def normalize_frame_labels(frame: pd.DataFrame, mapping: dict[str, int]) -> pd.DataFrame:
    if mapping != {"benign": 0, "malicious": 1}:
        return frame
    frame = frame.copy()
    frame["LABEL"] = frame["LABEL"].map(lambda value: "benign" if str(value).strip().lower() in BINARY_BENIGN else "malicious")
    return frame


def mapped_label_set(labels: pd.Series, mapping: dict[str, int]) -> set[str]:
    if mapping == {"benign": 0, "malicious": 1}:
        return {"benign" if str(value).strip().lower() in BINARY_BENIGN else "malicious" for value in labels}
    return {str(value).strip() for value in labels}


def prepare_tfusion(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame, mapping: dict[str, int], _config: dict[str, Any], out_dir: Path) -> PreparedData:
    train = normalize_frame_labels(train, mapping)
    dev = normalize_frame_labels(dev, mapping)
    test = normalize_frame_labels(test, mapping)
    if dev.empty:
        dev = train.copy()
    train_path = out_dir / "train.csv"
    dev_path = out_dir / "dev.csv"
    test_path = out_dir / "test.csv"
    train.to_csv(train_path, index=False)
    dev.to_csv(dev_path, index=False)
    test.to_csv(test_path, index=False)
    mapping_payload = {"label_to_id": mapping, "id_to_label": {str(index): label for label, index in mapping.items()}}
    mapping_path = out_dir / "label_mapping.json"
    write_json(mapping_path, mapping_payload)
    return PreparedData("TFusion", "flow", out_dir, train_path=train_path, dev_path=dev_path, test_path=test_path, label_mapping_path=mapping_path)


def prepare_data(
    algorithm: str,
    training_paths: list[Path],
    test_paths: list[Path],
    config: dict[str, Any],
    run_dir: Path,
    validation_paths: list[Path] | None = None,
) -> PreparedData:
    prepared_dir = run_dir / "prepared_data"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    if algorithm in PACKET_ALGORITHMS:
        generated_label_root = prepared_dir / "generated_packet_labels"
        training_packets = load_packet_sources(training_paths, "training_set", generated_label_root)
        test_packets = load_packet_sources(test_paths, "test_set", generated_label_root)
        validation_packets = load_packet_sources(validation_paths, "validation_set", generated_label_root) if validation_paths else None
        metadata = {
            "native_input_format": "pcap/pcapng plus packet-label file list",
            "existing_conversion_scripts_used": False,
            "training_sources": [{"pcap": rel(item.pcap), "packet_labels": rel(item.packet_labels), "source": item.source} for item in training_packets],
            "validation_sources": [
                {"pcap": rel(item.pcap), "packet_labels": rel(item.packet_labels), "source": item.source}
                for item in (validation_packets or [])
            ],
            "test_sources": [{"pcap": rel(item.pcap), "packet_labels": rel(item.packet_labels), "source": item.source} for item in test_packets],
        }
        write_json(prepared_dir / "metadata.json", metadata)
        return PreparedData(
            algorithm,
            "packet",
            prepared_dir,
            training_packets=training_packets,
            test_packets=test_packets,
            metadata=metadata,
            validation_packets=validation_packets,
        )

    if algorithm not in FLOW_ALGORITHMS:
        raise BenchmarkError(f"Unknown algorithm: {algorithm}")
    if algorithm == "TrafficFormer":
        conversion = config.get("algorithm_config", {}).get("conversion", {})
        representation = str(conversion.get("input_representation", "native_datagram"))
        use_native = bool(conversion.get("use_native_datagram", representation != "legacy_flow_features"))
        if use_native:
            return prepare_trafficformer_native(training_paths, validation_paths, test_paths, config, prepared_dir)
    if algorithm == "NetMamba":
        conversion = config.get("algorithm_config", {}).get("conversion", {})
        representation = str(conversion.get("input_representation", "native_5hp_image"))
        use_native = bool(conversion.get("use_native_5hp_image", representation != "legacy_flow_features"))
        if use_native:
            return prepare_netmamba_native(training_paths, validation_paths, test_paths, config, prepared_dir)
    raw_train_frame, train_input_metadata = load_flow_inputs(training_paths, config, prepared_dir / "generated_flow_csv", "training_set")
    raw_test_frame, test_input_metadata = load_flow_inputs(test_paths, config, prepared_dir / "generated_flow_csv", "test_set")
    raw_val_frame = None
    val_input_metadata: list[dict[str, Any]] = []
    if validation_paths:
        raw_val_frame, val_input_metadata = load_flow_inputs(validation_paths, config, prepared_dir / "generated_flow_csv", "validation_set")
        train_frame, test_frame, val_frame = apply_flow_row_limits(raw_train_frame, raw_test_frame, config, raw_val_frame)
    else:
        train_frame, test_frame = apply_flow_row_limits(raw_train_frame, raw_test_frame, config)
        val_frame = None
    label_mode = str(config.get("data", {}).get("label_mode", config.get("algorithm_config", {}).get("conversion", {}).get("label_mode", "auto")))
    mapping_test = pd.concat([test_frame["LABEL"], val_frame["LABEL"]], ignore_index=True) if val_frame is not None else test_frame["LABEL"]
    mapping = build_label_mapping(train_frame["LABEL"], mapping_test, label_mode)
    missing = mapped_label_set(test_frame["LABEL"], mapping) - mapped_label_set(train_frame["LABEL"], mapping)
    if missing and bool(config.get("data", {}).get("require_train_covers_test_labels", True)):
        raise BenchmarkError(f"Test set contains labels absent from training set: {sorted(missing)}")
    if val_frame is not None:
        dev_frame = val_frame.reset_index(drop=True)
    else:
        train_frame, dev_frame = split_train_dev(train_frame, config)
    if algorithm == "TrafficFormer":
        prepared = prepare_trafficformer(train_frame, dev_frame, test_frame, mapping, config, prepared_dir)
    else:
        prepared = prepare_netmamba(train_frame, dev_frame, test_frame, mapping, config, prepared_dir)
    prepared.metadata = {
        "native_input_format": "TrafficFormer train/dev/test TSV" if algorithm == "TrafficFormer" else "NetMamba ImageFolder train/valid/test",
        "existing_conversion_scripts_used": False,
        "raw_train_rows": int(len(raw_train_frame)),
        "raw_val_rows": int(len(raw_val_frame)) if raw_val_frame is not None else None,
        "raw_test_rows": int(len(raw_test_frame)),
        "train_rows": int(len(train_frame)),
        "dev_rows": int(len(dev_frame)),
        "test_rows": int(len(test_frame)),
        "label_mapping": mapping,
        "training_sources": [rel(path) for path in training_paths],
        "validation_sources": [rel(path) for path in validation_paths] if validation_paths else [],
        "test_sources": [rel(path) for path in test_paths],
        "training_input_metadata": train_input_metadata,
        "validation_input_metadata": val_input_metadata,
        "test_input_metadata": test_input_metadata,
    }
    write_json(prepared_dir / "metadata.json", prepared.metadata)
    return prepared
