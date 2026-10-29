#!/usr/bin/env python3
"""Prepare TrafficFormer fine-tuning TSVs with the native datagram logic.

This script intentionally reuses TrafficFormer's native get_feature_flow()
routine to build text_a.  It does not call dataset_extract(), because that
function performs its own random 80/10/10 split.  Benchmark splits are already
reserved before run_benchmark.py is invoked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TRAFFICFORMER_DATA_GEN = ROOT / "Benchmark" / "2025-SP-TrafficFormer" / "data_generation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRAFFICFORMER_DATA_GEN))

import scapy.all as scapy  # noqa: E402
from finetuning_data_gen import get_feature_flow  # noqa: E402


BINARY_BENIGN = {"0", "0.0", "benign", "normal", "false"}
BINARY_MALICIOUS = {"1", "1.0", "malicious", "attack", "true"}


@dataclass
class FlowState:
    key: tuple[str, str, str, str, str]
    path: Path
    writer: Any
    start_time: float
    end_time: float
    labels: Counter[str] = field(default_factory=Counter)
    packets: int = 0


def rel(path: str | Path) -> str:
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(candidate)


def read_packet_labels(path: Path | None) -> list[str]:
    if path is None:
        return []
    labels: list[str] = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            value = row[-1].strip()
            if value.lower() in {"label", "labels", "x"}:
                continue
            labels.append("benign" if value.lower() in BINARY_BENIGN else "malicious")
    return labels


def normalize_label(value: object, label_mode: str) -> str:
    text = str(value).strip()
    if label_mode == "binary":
        return "benign" if text.lower() in BINARY_BENIGN else "malicious"
    return text


def label_for_flow(counter: Counter[str], default_label: str, label_mode: str) -> str:
    if label_mode == "binary":
        for label, count in counter.items():
            if count > 0 and label.lower() not in BINARY_BENIGN:
                return "malicious"
        return "benign" if default_label.lower() in BINARY_BENIGN else "malicious"
    if counter:
        return counter.most_common(1)[0][0]
    return default_label


def packet_key(packet: Any, mode: str) -> tuple[str, str, str, str, str] | None:
    if scapy.IP not in packet:
        return None
    ip = packet[scapy.IP]
    proto = str(int(ip.proto))
    if scapy.TCP in packet:
        sport = str(int(packet[scapy.TCP].sport))
        dport = str(int(packet[scapy.TCP].dport))
    elif scapy.UDP in packet:
        sport = str(int(packet[scapy.UDP].sport))
        dport = str(int(packet[scapy.UDP].dport))
    else:
        return None
    src = (str(ip.src), sport)
    dst = (str(ip.dst), dport)
    if mode == "unidirectional":
        return (src[0], dst[0], src[1], dst[1], proto)
    if src <= dst:
        return (src[0], dst[0], src[1], dst[1], proto)
    return (dst[0], src[0], dst[1], src[1], proto)


def label_path_for_spec(spec: dict[str, Any]) -> str | None:
    label_path = spec.get("packet_labels") or spec.get("labels")
    return str(label_path) if label_path else None


def can_use_label_independent_cache(spec: dict[str, Any], conversion: dict[str, Any]) -> bool:
    if label_path_for_spec(spec):
        return False
    return bool(conversion.get("label_independent_cache", True))


def legacy_source_cache_key(spec: dict[str, Any], conversion: dict[str, Any]) -> str:
    pcap = Path(spec["pcap"]).resolve()
    label_path = label_path_for_spec(spec)
    payload = {
        "pcap": str(pcap),
        "pcap_size": pcap.stat().st_size,
        "pcap_mtime_ns": pcap.stat().st_mtime_ns,
        "label_path": str(Path(label_path).resolve()) if label_path else None,
        "label_size": Path(label_path).stat().st_size if label_path and Path(label_path).exists() else None,
        "label_mtime_ns": Path(label_path).stat().st_mtime_ns if label_path and Path(label_path).exists() else None,
        "default_label": spec.get("default_label"),
        "five_tuple_mode": conversion.get("five_tuple_mode", "bidirectional"),
        "five_tuple_timeout_seconds": float(conversion.get("five_tuple_timeout_seconds", 120.0)),
        "max_packets_per_source": conversion.get("max_packets_per_source"),
        "max_flows_per_source": conversion.get("max_flows_per_source"),
        "label_mode": conversion.get("_label_mode", "binary"),
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def source_cache_key(spec: dict[str, Any], conversion: dict[str, Any]) -> str:
    if not can_use_label_independent_cache(spec, conversion):
        return legacy_source_cache_key(spec, conversion)
    pcap = Path(spec["pcap"]).resolve()
    payload = {
        "cache_version": 2,
        "label_independent": True,
        "pcap": str(pcap),
        "pcap_size": pcap.stat().st_size,
        "pcap_mtime_ns": pcap.stat().st_mtime_ns,
        "five_tuple_mode": conversion.get("five_tuple_mode", "bidirectional"),
        "five_tuple_timeout_seconds": float(conversion.get("five_tuple_timeout_seconds", 120.0)),
        "max_packets_per_source": conversion.get("max_packets_per_source"),
        "max_flows_per_source": conversion.get("max_flows_per_source"),
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def feature_cache_key(conversion: dict[str, Any]) -> str:
    payload = {
        "payload_length": int(conversion.get("payload_length", 64)),
        "payload_packet": int(conversion.get("payload_packet", 5)),
        "start_index": int(conversion.get("start_index", 76)),
        "random_seed": int(conversion.get("random_seed", 10)),
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def relabel_rows_for_spec(rows: list[dict[str, Any]], spec: dict[str, Any], label_mode: str) -> list[dict[str, Any]]:
    if label_path_for_spec(spec):
        return rows
    default_label = normalize_label(spec.get("default_label", Path(spec["pcap"]).stem), label_mode)
    relabeled: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["label"] = default_label
        relabeled.append(item)
    return relabeled


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp_path.replace(path)


def copy_file_to_cache(src: Path, dst: Path) -> bool:
    if not src.exists() or dst.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + ".tmp")
    try:
        os.link(src, tmp_path)
    except OSError:
        shutil.copy2(src, tmp_path)
    tmp_path.replace(dst)
    return True


def path_from_cache_index(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def source_pcap_from_summary(summary_path: Path) -> str | None:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    source = summary.get("source_pcap")
    if not source:
        return None
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    return str(source_path.resolve())


def build_feature_cache_index(cache_root: Path, conversion: dict[str, Any]) -> Path:
    feature_name = f"features_{feature_cache_key(conversion)}.jsonl"
    index_path = cache_root / f".source_pcap_index_{feature_cache_key(conversion)}.json"
    entries: dict[str, str] = {}
    for summary_path in cache_root.glob("*/split_summary.json"):
        feature_path = summary_path.parent / feature_name
        if not feature_path.exists():
            continue
        source_pcap = source_pcap_from_summary(summary_path)
        if source_pcap and source_pcap not in entries:
            entries[source_pcap] = rel(feature_path)
    tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(index_path)
    return index_path


def materialize_legacy_feature_cache(cache_root: Path, spec: dict[str, Any], conversion: dict[str, Any], feature_path: Path) -> bool:
    if not can_use_label_independent_cache(spec, conversion):
        return False
    legacy_dir = cache_root / legacy_source_cache_key(spec, conversion)
    legacy_feature_path = legacy_dir / f"features_{feature_cache_key(conversion)}.jsonl"
    source_dir = legacy_dir
    if not legacy_feature_path.exists():
        index_path = conversion.get("_feature_cache_index")
        if not index_path:
            return False
        try:
            index = json.loads(path_from_cache_index(str(index_path)).read_text(encoding="utf-8"))
        except Exception:
            return False
        indexed = index.get(str(Path(spec["pcap"]).resolve()))
        if not indexed:
            return False
        legacy_feature_path = path_from_cache_index(str(indexed))
        source_dir = legacy_feature_path.parent
    if not copy_file_to_cache(legacy_feature_path, feature_path):
        return False
    for name in ("split_summary.json", "feature_summary.json", "flow_manifest.jsonl"):
        copy_file_to_cache(source_dir / name, feature_path.parent / name)
    return True


def split_source_to_flow_pcaps(
    spec: dict[str, Any],
    cache_dir: Path,
    conversion: dict[str, Any],
    label_mode: str,
) -> list[dict[str, Any]]:
    pcap = Path(spec["pcap"]).resolve()
    label_path = spec.get("packet_labels") or spec.get("labels")
    labels = read_packet_labels(Path(label_path).resolve() if label_path else None)
    default_label = str(spec.get("default_label", Path(pcap).stem))
    mode = str(conversion.get("five_tuple_mode", "bidirectional"))
    timeout = float(conversion.get("five_tuple_timeout_seconds", 120.0))
    max_packets = conversion.get("max_packets_per_source")
    max_packets = int(max_packets) if max_packets is not None else None
    max_flows = conversion.get("max_flows_per_source")
    max_flows = int(max_flows) if max_flows is not None else None
    flow_dir = cache_dir / "flow_pcaps"
    flow_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "flow_manifest.jsonl"
    if manifest_path.exists():
        rows = read_jsonl(manifest_path)
        if rows and all(Path(row["flow_pcap"]).exists() for row in rows):
            return rows

    active: dict[tuple[str, str, str, str, str], FlowState] = {}
    completed: list[FlowState] = []
    next_flow_id = 0

    def close_flow(key: tuple[str, str, str, str, str]) -> None:
        state = active.pop(key)
        state.writer.close()
        completed.append(state)

    def new_flow(key: tuple[str, str, str, str, str], timestamp: float) -> FlowState:
        nonlocal next_flow_id
        path = flow_dir / f"flow_{next_flow_id:08d}.pcap"
        next_flow_id += 1
        writer = scapy.PcapWriter(str(path), append=False, sync=False)
        state = FlowState(key=key, path=path, writer=writer, start_time=timestamp, end_time=timestamp)
        active[key] = state
        return state

    packet_count = 0
    inspected_packets = 0
    skipped_packets = 0
    with scapy.PcapReader(str(pcap)) as reader:
        for packet_index, packet in enumerate(reader):
            if max_packets is not None and inspected_packets >= max_packets:
                break
            inspected_packets += 1
            key = packet_key(packet, mode)
            if key is None:
                skipped_packets += 1
                continue
            timestamp = float(getattr(packet, "time", 0.0))
            state = active.get(key)
            if state is not None and timestamp - state.end_time > timeout:
                close_flow(key)
                state = None
            if state is None:
                if max_flows is not None and next_flow_id >= max_flows:
                    continue
                state = new_flow(key, timestamp)
            state.writer.write(packet)
            state.end_time = max(state.end_time, timestamp)
            state.packets += 1
            raw_label = labels[packet_index] if packet_index < len(labels) else default_label
            state.labels[normalize_label(raw_label, label_mode)] += 1
            packet_count += 1
            if packet_count % 100000 == 0:
                print(f"[trafficformer-native] split {pcap.name}: packets={packet_count} active_flows={len(active)}", flush=True)

    for key in list(active):
        close_flow(key)

    rows = []
    for index, state in enumerate(completed):
        rows.append(
            {
                "flow_index": index,
                "flow_pcap": str(state.path),
                "source_pcap": str(pcap),
                "source": str(spec.get("source", pcap.stem)),
                "label": label_for_flow(state.labels, normalize_label(default_label, label_mode), label_mode),
                "packets": state.packets,
                "start_time": state.start_time,
                "end_time": state.end_time,
            }
        )
    write_jsonl_atomic(manifest_path, rows)
    (cache_dir / "split_summary.json").write_text(
        json.dumps(
            {
                "source_pcap": rel(pcap),
                "packet_labels": rel(label_path) if label_path else None,
                "flows": len(rows),
                "packets": packet_count,
                "inspected_packets": inspected_packets,
                "skipped_packets": skipped_packets,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return rows


def extract_native_rows(
    spec: dict[str, Any],
    cache_root: Path,
    conversion: dict[str, Any],
    label_mode: str,
) -> list[dict[str, Any]]:
    conversion = dict(conversion)
    conversion["_label_mode"] = label_mode
    cache_dir = cache_root / source_cache_key(spec, conversion)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / f"features_{feature_cache_key(conversion)}.jsonl"
    if feature_path.exists():
        return relabel_rows_for_spec(read_jsonl(feature_path), spec, label_mode)
    if materialize_legacy_feature_cache(cache_root, spec, conversion, feature_path):
        return relabel_rows_for_spec(read_jsonl(feature_path), spec, label_mode)
    if bool(conversion.get("require_feature_cache", False)):
        raise RuntimeError(
            "TrafficFormer native feature cache miss while require_feature_cache=true: "
            f"{feature_path}. Disable require_feature_cache for an intentional precompute run."
        )

    flows = split_source_to_flow_pcaps(spec, cache_dir, conversion, label_mode)
    payload_length = int(conversion.get("payload_length", 64))
    payload_packet = int(conversion.get("payload_packet", 5))
    start_index = int(conversion.get("start_index", 76))
    random_seed = int(conversion.get("random_seed", 10))
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    for flow in flows:
        flow_pcap = flow["flow_pcap"]
        try:
            random.seed(random_seed + int(flow["flow_index"]))
            feature_data = get_feature_flow(
                flow_pcap,
                select_packet_len=payload_length,
                packets_num=payload_packet,
                start_index=start_index,
            )
        except Exception as exc:
            skipped[type(exc).__name__] += 1
            continue
        if feature_data == -1:
            skipped["native_rejected"] += 1
            continue
        rows.append({**flow, "text_a": feature_data[0]})
        if len(rows) % 10000 == 0:
            print(f"[trafficformer-native] features {Path(spec['pcap']).name}: rows={len(rows)}", flush=True)

    write_jsonl_atomic(feature_path, rows)
    (cache_dir / "feature_summary.json").write_text(
        json.dumps({"rows": len(rows), "skipped": dict(skipped)}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return relabel_rows_for_spec(rows, spec, label_mode)


def build_mapping(rows_by_split: dict[str, list[dict[str, Any]]], label_mode: str) -> dict[str, int]:
    if label_mode == "binary":
        return {"benign": 0, "malicious": 1}
    labels = sorted({str(row["label"]) for rows in rows_by_split.values() for row in rows}, key=str.casefold)
    return {label: index for index, label in enumerate(labels)}


def label_to_id(label: str, mapping: dict[str, int]) -> int:
    if mapping == {"benign": 0, "malicious": 1}:
        return 0 if label.lower() in BINARY_BENIGN else 1
    return mapping[label]


def limit_rows(rows: list[dict[str, Any]], max_rows: int | None, random_state: int) -> list[dict[str, Any]]:
    if max_rows is None or len(rows) <= max_rows:
        return rows
    rng = random.Random(random_state)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    selected = sorted(indices[:max_rows])
    return [rows[index] for index in selected]


def write_split(rows: list[dict[str, Any]], path: Path, mapping: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["label", "text_a"])
        for row in rows:
            writer.writerow([label_to_id(str(row["label"]), mapping), row["text_a"]])


def ant_metadata_from_source_file(source_file: object) -> dict[str, str]:
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


def write_row_metadata(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "ROW_INDEX",
            "TRUE_LABEL",
            "SOURCE_DATASET",
            "SOURCE_FILE",
            "NATIVE_FLOW_PCAP",
            "NATIVE_FLOW_PACKETS",
            "ANT_SPLIT",
            "ANT_ENGINE",
            "ANT_TASK_FAMILY",
            "ANT_EPISODE",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(rows):
            source_file = row.get("source_pcap", "")
            ant = ant_metadata_from_source_file(source_file)
            writer.writerow(
                {
                    "ROW_INDEX": index,
                    "TRUE_LABEL": row.get("label", ""),
                    "SOURCE_DATASET": row.get("source", ""),
                    "SOURCE_FILE": rel(source_file),
                    "NATIVE_FLOW_PCAP": rel(row.get("flow_pcap", "")),
                    "NATIVE_FLOW_PACKETS": row.get("packets", ""),
                    **ant,
                }
            )


def prepare_source(
    source_index: int,
    spec: dict[str, Any],
    cache_root: str,
    conversion: dict[str, Any],
    label_mode: str,
) -> tuple[int, list[dict[str, Any]]]:
    rows = extract_native_rows(spec, Path(cache_root), conversion, label_mode)
    return source_index, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("ANT_NATIVE_WORKERS", "1")))
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conversion = dict(manifest.get("conversion", {}))
    label_mode = str(manifest.get("label_mode", "binary"))
    cache_root = Path(conversion.get("cache_dir", "Benchmark/cache/trafficformer_native"))
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    conversion["_feature_cache_index"] = str(build_feature_cache_index(cache_root, conversion))
    random_state = int(conversion.get("random_state", 42))

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    summaries: dict[str, Any] = {}
    workers = max(1, int(args.workers))
    for split_name in ("train", "dev", "test"):
        split_specs = manifest.get(f"{split_name}_sources", [])
        results: dict[int, list[dict[str, Any]]] = {}
        if workers > 1 and len(split_specs) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(split_specs))) as executor:
                futures = [
                    executor.submit(prepare_source, source_index, spec, str(cache_root), conversion, label_mode)
                    for source_index, spec in enumerate(split_specs)
                ]
                completed_rows = 0
                for future in as_completed(futures):
                    source_index, rows = future.result()
                    results[source_index] = rows
                    completed_rows += len(rows)
                    print(
                        f"[trafficformer-native] split={split_name} source={source_index} "
                        f"rows={len(rows)} completed_total={completed_rows}",
                        flush=True,
                    )
        else:
            for source_index, spec in enumerate(split_specs):
                source_index, rows = prepare_source(source_index, spec, str(cache_root), conversion, label_mode)
                results[source_index] = rows
                print(f"[trafficformer-native] split={split_name} source={source_index} rows={len(rows)}", flush=True)
        split_rows: list[dict[str, Any]] = []
        for source_index in sorted(results):
            split_rows.extend(results[source_index])
        print(f"[trafficformer-native] split={split_name} total_rows={len(split_rows)}", flush=True)
        max_key = {"train": "max_train_rows", "dev": "max_val_rows", "test": "max_test_rows"}[split_name]
        max_rows = conversion.get(max_key, conversion.get("max_rows"))
        max_rows = int(max_rows) if max_rows is not None else None
        rows_by_split[split_name] = limit_rows(split_rows, max_rows, random_state + {"train": 0, "dev": 1, "test": 2}[split_name])
        summaries[split_name] = {
            "raw_rows": len(split_rows),
            "rows": len(rows_by_split[split_name]),
            "labels": dict(Counter(str(row["label"]) for row in rows_by_split[split_name])),
        }

    if not rows_by_split["dev"]:
        raise ValueError("Validation preprocessing produced no rows")
        summaries["dev"]["copied_from_train"] = True

    mapping = build_mapping(rows_by_split, label_mode)
    write_split(rows_by_split["train"], output_dir / "train.tsv", mapping)
    write_split(rows_by_split["dev"], output_dir / "dev.tsv", mapping)
    write_split(rows_by_split["test"], output_dir / "test.tsv", mapping)
    for split in ("train", "dev", "test"):
        write_row_metadata(rows_by_split[split], output_dir / f"{split}_row_metadata.csv")
    (output_dir / "label_mapping.json").write_text(
        json.dumps({"label_to_id": mapping, "id_to_label": {str(v): k for k, v in mapping.items()}}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "native_conversion_summary.json").write_text(
        json.dumps(
            {
                "input_format": "TrafficFormer native datagram text_a",
                "native_function": "Benchmark/2025-SP-TrafficFormer/data_generation/finetuning_data_gen.py:get_feature_flow",
                "benchmark_split_preserved": True,
                "conversion": conversion,
                "cache_key": {
                    "version": 2,
                    "label_independent_without_packet_labels": True,
                    "legacy_v1_label_dependent_cache_read": True,
                    "labels_reapplied_from_current_manifest": True,
                },
                "label_mode": label_mode,
                "splits": summaries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "completed", "splits": summaries}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
