#!/usr/bin/env python3
"""Prepare NetMamba ImageFolder data with the native 5-packet byte-image logic.

NetMamba's native preprocessing first splits traffic into flow pcaps, then
turns each flow into a 40x40 grayscale image with read_5hp_list().  This script
keeps Benchmark's pre-reserved train/valid/test split and reuses NetMamba's
native packet-to-image routine for each generated flow pcap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NETMAMBA_DATASET = ROOT / "Benchmark" / "2024-ICNP-NetMamba" / "dataset"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NETMAMBA_DATASET))

import scapy.all as scapy  # noqa: E402
from dataset_common import read_5hp_list  # noqa: E402
from PIL import Image  # noqa: E402


BINARY_BENIGN = {"0", "0.0", "benign", "normal", "false"}


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
        if any(label.lower() not in BINARY_BENIGN for label in counter):
            return "malicious"
        return "benign" if default_label.lower() in BINARY_BENIGN else "malicious"
    if counter:
        return counter.most_common(1)[0][0]
    return default_label


def packet_key(packet: Any, mode: str) -> tuple[str, str, str, str, str]:
    if scapy.IP in packet:
        ip = packet[scapy.IP]
        proto = str(int(ip.proto))
        if scapy.TCP in packet:
            sport = str(int(packet[scapy.TCP].sport))
            dport = str(int(packet[scapy.TCP].dport))
        elif scapy.UDP in packet:
            sport = str(int(packet[scapy.UDP].sport))
            dport = str(int(packet[scapy.UDP].dport))
        else:
            sport = "0"
            dport = "0"
        src = (str(ip.src), sport)
        dst = (str(ip.dst), dport)
    else:
        src_addr = str(getattr(packet, "src", "unknown-src"))
        dst_addr = str(getattr(packet, "dst", "unknown-dst"))
        proto = str(int(getattr(packet, "type", 0) or 0))
        src = (src_addr, "0")
        dst = (dst_addr, "0")
    if mode == "unidirectional":
        return (src[0], dst[0], src[1], dst[1], proto)
    if src <= dst:
        return (src[0], dst[0], src[1], dst[1], proto)
    return (dst[0], src[0], dst[1], src[1], proto)


def source_cache_key(spec: dict[str, Any], conversion: dict[str, Any], label_mode: str) -> str:
    pcap = Path(spec["pcap"]).resolve()
    label_path = spec.get("packet_labels") or spec.get("labels")
    payload = {
        "version": 2,
        "pcap": str(pcap),
        "pcap_size": pcap.stat().st_size,
        "pcap_mtime_ns": pcap.stat().st_mtime_ns,
        "five_tuple_mode": conversion.get("five_tuple_mode", "bidirectional"),
        "five_tuple_timeout_seconds": float(conversion.get("five_tuple_timeout_seconds", 120.0)),
        "max_packets_per_source": conversion.get("max_packets_per_source"),
        "max_flows_per_source": conversion.get("max_flows_per_source"),
    }
    if label_path:
        label_file = Path(label_path)
        payload.update(
            {
                "label_path": str(label_file.resolve()),
                "label_size": label_file.stat().st_size if label_file.exists() else None,
                "label_mtime_ns": label_file.stat().st_mtime_ns if label_file.exists() else None,
                "label_mode": label_mode,
            }
        )
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def class_dir_name(label: str, index: int, mapping: dict[str, int]) -> str:
    if mapping == {"benign": 0, "malicious": 1}:
        return label
    slug = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)
    return f"{index:03d}_{slug}"


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
        try:
            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except json.JSONDecodeError:
            rows = []
        if rows:
            if all(Path(row["flow_pcap"]).exists() for row in rows):
                if not label_path:
                    label = normalize_label(default_label, label_mode)
                    for row in rows:
                        row["label"] = label
                return rows
        shutil.rmtree(flow_dir, ignore_errors=True)
        flow_dir.mkdir(parents=True, exist_ok=True)

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
    with scapy.PcapReader(str(pcap)) as reader:
        for packet_index, packet in enumerate(reader):
            if max_packets is not None and inspected_packets >= max_packets:
                break
            inspected_packets += 1
            key = packet_key(packet, mode)
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
                print(f"[netmamba-native] split {pcap.name}: packets={packet_count} active_flows={len(active)}", flush=True)

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
    tmp_manifest_path = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    with tmp_manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    tmp_manifest_path.replace(manifest_path)
    summary_path = cache_dir / "split_summary.json"
    tmp_summary_path = summary_path.with_name(f".{summary_path.name}.{os.getpid()}.tmp")
    tmp_summary_path.write_text(
        json.dumps(
            {
                "source_pcap": rel(pcap),
                "packet_labels": rel(label_path) if label_path else None,
                "flows": len(rows),
                "packets": packet_count,
                "inspected_packets": inspected_packets,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    tmp_summary_path.replace(summary_path)
    return rows


def native_flow_image(flow_pcap: str, conversion: dict[str, Any]) -> np.ndarray:
    rows = read_5hp_list(
        flow_pcap,
        if_augment=bool(conversion.get("if_augment", False)),
        remove_ip=bool(conversion.get("remove_ip", True)),
        keep_payload=bool(conversion.get("keep_payload", True)),
    )
    if not rows:
        raise ValueError("read_5hp_list returned no rows")
    data = rows[0]["data"]
    return np.asarray(data, dtype=np.uint8).reshape(40, 40)


def image_cache_key(row: dict[str, Any], conversion: dict[str, Any]) -> str:
    flow_pcap = Path(row["flow_pcap"]).resolve()
    source_pcap = Path(row["source_pcap"]).resolve()
    payload = {
        "version": 2,
        "native_function": "Benchmark/2024-ICNP-NetMamba/dataset/dataset_common.py:read_5hp_list",
        "flow_pcap": str(flow_pcap),
        "flow_size": flow_pcap.stat().st_size,
        "flow_mtime_ns": flow_pcap.stat().st_mtime_ns,
        "source_pcap": str(source_pcap),
        "source_size": source_pcap.stat().st_size,
        "source_mtime_ns": source_pcap.stat().st_mtime_ns,
        "flow_index": row.get("flow_index"),
        "packets": row.get("packets"),
        "start_time": row.get("start_time"),
        "end_time": row.get("end_time"),
        "remove_ip": bool(conversion.get("remove_ip", True)),
        "keep_payload": bool(conversion.get("keep_payload", True)),
        "if_augment": bool(conversion.get("if_augment", False)),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def legacy_image_cache_key(row: dict[str, Any], conversion: dict[str, Any], label_mode: str) -> str:
    flow_pcap = Path(row["flow_pcap"]).resolve()
    source_pcap = Path(row["source_pcap"]).resolve()
    payload = {
        "version": 1,
        "native_function": "Benchmark/2024-ICNP-NetMamba/dataset/dataset_common.py:read_5hp_list",
        "flow_pcap": str(flow_pcap),
        "flow_size": flow_pcap.stat().st_size,
        "flow_mtime_ns": flow_pcap.stat().st_mtime_ns,
        "source_pcap": str(source_pcap),
        "source_size": source_pcap.stat().st_size,
        "source_mtime_ns": source_pcap.stat().st_mtime_ns,
        "flow_index": row.get("flow_index"),
        "label": row.get("label"),
        "packets": row.get("packets"),
        "start_time": row.get("start_time"),
        "end_time": row.get("end_time"),
        "remove_ip": bool(conversion.get("remove_ip", True)),
        "keep_payload": bool(conversion.get("keep_payload", True)),
        "if_augment": bool(conversion.get("if_augment", False)),
        "label_mode": label_mode,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def materialize_cache_image(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        os.link(source, tmp_path)
    except OSError:
        shutil.copy2(source, tmp_path)
    try:
        tmp_path.replace(target)
    except FileExistsError:
        tmp_path.unlink(missing_ok=True)


def write_cached_image(row: dict[str, Any], conversion: dict[str, Any], image_cache_root: Path, label_mode: str) -> tuple[Path, bool]:
    key = image_cache_key(row, conversion)
    cache_path = image_cache_root / key[:2] / f"{key}.png"
    if cache_path.exists():
        return cache_path, True
    legacy_key = legacy_image_cache_key(row, conversion, label_mode)
    legacy_path = image_cache_root / legacy_key[:2] / f"{legacy_key}.png"
    if legacy_path.exists():
        materialize_cache_image(legacy_path, cache_path)
        return cache_path, True
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    try:
        image = Image.fromarray(native_flow_image(row["flow_pcap"], conversion), mode="L")
        image.save(tmp_path, format="PNG")
        try:
            tmp_path.replace(cache_path)
        except FileExistsError:
            tmp_path.unlink(missing_ok=True)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return cache_path, False


def link_or_copy_image(source: Path, target: Path) -> str:
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
        return "symlinked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def prepare_image_row(
    row_index: int,
    row: dict[str, Any],
    split: str,
    output_dir: str,
    mapping: dict[str, int],
    conversion: dict[str, Any],
    image_cache_root: str,
    label_mode: str,
) -> tuple[int, dict[str, Any] | None, str | None, bool, str | None]:
    label = str(row["label"])
    if mapping == {"benign": 0, "malicious": 1}:
        label = "benign" if label.lower() in BINARY_BENIGN else "malicious"
    try:
        class_dir = Path(output_dir) / split / class_dir_name(label, mapping[label], mapping)
        class_dir.mkdir(parents=True, exist_ok=True)
        image_path = class_dir / f"flow_{row_index:08d}.png"
        cache_path, hit = write_cached_image({**row, "label": label}, conversion, Path(image_cache_root), label_mode)
        link_status = link_or_copy_image(cache_path, image_path)
        return row_index, {**row, "label": label, "image_path": str(image_path)}, link_status, hit, None
    except Exception as exc:
        return row_index, None, None, False, type(exc).__name__


def build_mapping(rows_by_split: dict[str, list[dict[str, Any]]], label_mode: str) -> dict[str, int]:
    if label_mode == "binary":
        return {"benign": 0, "malicious": 1}
    labels = sorted({str(row["label"]) for rows in rows_by_split.values() for row in rows}, key=str.casefold)
    return {label: index for index, label in enumerate(labels)}


def write_split(
    rows: list[dict[str, Any]],
    split: str,
    output_dir: Path,
    mapping: dict[str, int],
    conversion: dict[str, Any],
    image_cache_root: Path,
    label_mode: str,
    workers: int = 1,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    written_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    cache_stats: Counter[str] = Counter()
    results: dict[int, tuple[dict[str, Any] | None, str | None, bool, str | None]] = {}
    if workers > 1 and len(rows) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(rows))) as executor:
            futures = [
                executor.submit(
                    prepare_image_row,
                    row_index,
                    row,
                    split,
                    str(output_dir),
                    mapping,
                    conversion,
                    str(image_cache_root),
                    label_mode,
                )
                for row_index, row in enumerate(rows)
            ]
            for future in as_completed(futures):
                row_index, written, link_status, hit, error = future.result()
                results[row_index] = (written, link_status, hit, error)
                if len(results) % 1000 == 0:
                    print(f"[netmamba-native] split={split} image_tasks_done={len(results)}", flush=True)
    else:
        for row_index, row in enumerate(rows):
            row_index, written, link_status, hit, error = prepare_image_row(
                row_index,
                row,
                split,
                str(output_dir),
                mapping,
                conversion,
                str(image_cache_root),
                label_mode,
            )
            results[row_index] = (written, link_status, hit, error)
            if len(results) % 1000 == 0:
                print(f"[netmamba-native] split={split} image_tasks_done={len(results)}", flush=True)
    for row_index in sorted(results):
        written, link_status, hit, error = results[row_index]
        if error is not None or written is None:
            skipped[error or "unknown"] += 1
            continue
        cache_stats["hits" if hit else "misses"] += 1
        if link_status:
            cache_stats[link_status] += 1
        counts[str(written["label"])] += 1
        written_rows.append(written)
    return {
        "rows": written_rows,
        "counts": dict(sorted(counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "image_cache": dict(sorted(cache_stats.items())),
    }


def write_row_metadata(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: str(row["image_path"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ROW_INDEX",
                "TRUE_LABEL",
                "SOURCE_DATASET",
                "SOURCE_FILE",
                "NATIVE_FLOW_PCAP",
                "NATIVE_IMAGE",
                "NATIVE_FLOW_PACKETS",
                "START_TIME",
                "END_TIME",
            ],
        )
        writer.writeheader()
        for index, row in enumerate(ordered):
            writer.writerow(
                {
                    "ROW_INDEX": index,
                    "TRUE_LABEL": row.get("label", ""),
                    "SOURCE_DATASET": row.get("source", ""),
                    "SOURCE_FILE": rel(row.get("source_pcap", "")),
                    "NATIVE_FLOW_PCAP": rel(row.get("flow_pcap", "")),
                    "NATIVE_IMAGE": rel(row.get("image_path", "")),
                    "NATIVE_FLOW_PACKETS": row.get("packets", ""),
                    "START_TIME": f"{float(row.get('start_time', 0.0)):.9f}",
                    "END_TIME": f"{float(row.get('end_time', 0.0)):.9f}",
                }
            )


def prepare_source(
    source_index: int,
    spec: dict[str, Any],
    conversion: dict[str, Any],
    label_mode: str,
    cache_root: str,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    cache_root_path = Path(cache_root)
    cache_dir = cache_root_path / source_cache_key(spec, conversion, label_mode)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = split_source_to_flow_pcaps(spec, cache_dir, conversion, label_mode)
    return (
        source_index,
        rows,
        {
            "source_index": source_index,
            "source_pcap": rel(spec["pcap"]),
            "cache_dir": rel(cache_dir),
            "flows": len(rows),
        },
    )


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
    cache_root = Path(conversion.get("cache_dir", "Benchmark/cache/netmamba_native"))
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    cache_root.mkdir(parents=True, exist_ok=True)
    image_cache_root = Path(conversion.get("image_cache_dir") or (cache_root / "image_cache"))
    if not image_cache_root.is_absolute():
        image_cache_root = ROOT / image_cache_root
    image_cache_root.mkdir(parents=True, exist_ok=True)

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    split_cache_summary: dict[str, Any] = {}
    workers = max(1, int(args.workers))
    for split_name in ("train", "valid", "test"):
        split_specs = manifest.get(f"{split_name}_sources", [])
        split_cache_summary[split_name] = []
        results: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
        if workers > 1 and len(split_specs) > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(split_specs))) as executor:
                futures = [
                    executor.submit(prepare_source, source_index, spec, conversion, label_mode, str(cache_root))
                    for source_index, spec in enumerate(split_specs)
                ]
                completed_rows = 0
                for future in as_completed(futures):
                    source_index, rows, summary_row = future.result()
                    results[source_index] = (rows, summary_row)
                    completed_rows += len(rows)
                    print(
                        f"[netmamba-native] split={split_name} source={source_index} "
                        f"flows={len(rows)} completed_total={completed_rows}",
                        flush=True,
                    )
        else:
            for source_index, spec in enumerate(split_specs):
                source_index, rows, summary_row = prepare_source(source_index, spec, conversion, label_mode, str(cache_root))
                results[source_index] = (rows, summary_row)
                print(f"[netmamba-native] split={split_name} source={source_index} flows={len(rows)}", flush=True)
        split_rows: list[dict[str, Any]] = []
        for source_index in sorted(results):
            rows, summary_row = results[source_index]
            split_rows.extend(rows)
            split_cache_summary[split_name].append(summary_row)
        rows_by_split[split_name] = split_rows
        print(f"[netmamba-native] split={split_name} total_flows={len(split_rows)}", flush=True)

    if not rows_by_split["valid"]:
        raise ValueError("Validation preprocessing produced no rows")
    mapping = build_mapping(rows_by_split, label_mode)
    written: dict[str, Any] = {}
    for split_name in ("train", "valid", "test"):
        written[split_name] = write_split(
            rows_by_split[split_name],
            split_name,
            output_dir,
            mapping,
            conversion,
            image_cache_root,
            label_mode,
            workers,
        )

    for split in ("train", "valid", "test"):
        write_row_metadata(written[split]["rows"], output_dir / f"{split}_row_metadata.csv")
    (output_dir / "label_mapping.json").write_text(
        json.dumps(
            {
                "label_to_id": mapping,
                "id_to_label": {str(index): label for label, index in mapping.items()},
                "class_directories": {label: class_dir_name(label, index, mapping) for label, index in mapping.items()},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary = {
        "input_format": "NetMamba native 5-packet byte ImageFolder",
        "native_function": "Benchmark/2024-ICNP-NetMamba/dataset/dataset_common.py:read_5hp_list",
        "benchmark_split_preserved": True,
        "conversion": conversion,
        "image_cache_dir": rel(image_cache_root),
        "image_cache_key": {
            "version": 2,
            "label_independent": True,
            "legacy_v1_label_dependent_cache_read": True,
        },
        "label_mode": label_mode,
        "mapping": mapping,
        "split_cache": split_cache_summary,
        "splits": {
            split: {
                "raw_flows": len(rows_by_split[split]),
                "images": len(written[split]["rows"]),
                "labels": written[split]["counts"],
                "skipped": written[split]["skipped"],
                "image_cache": written[split]["image_cache"],
            }
            for split in ("train", "valid", "test")
        },
    }
    (output_dir / "native_conversion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"status": "completed", "splits": summary["splits"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
