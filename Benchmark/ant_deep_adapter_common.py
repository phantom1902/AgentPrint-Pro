#!/usr/bin/env python3
"""Shared helpers for ANT adapters of byte/packet/flow representation models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ant_pcap_native import (
    comma_list,
    discover_episodes,
    dns_ip_to_hostname,
    group_flows,
    parse_risk_map,
    read_pcap_packets,
    safe_name,
)


@dataclass(frozen=True)
class AntSample:
    split: str
    engine: str
    scenario: str
    sample_id: str
    pcap_path: Path
    label: str
    source_kind: str


def protocol_set(value: str | None):
    if not value or value == "all":
        return None
    mapping = {"tcp": 6, "udp": 17, "icmp": 1, "icmpv6": 58}
    return {mapping[item] for item in comma_list(value, ["tcp", "udp"])}


def discover_ant_samples(
    ant_root: Path,
    splits: Iterable[str],
    label_field: str,
    source_mode: str = "episode",
    engines: Iterable[str] | None = None,
    max_samples: int | None = None,
) -> list[AntSample]:
    split_set = set(splits)
    engine_set = set(engines or [])
    out: list[AntSample] = []
    if source_mode == "segment":
        segment_root = ant_root / "segment_pcaps" if (ant_root / "segment_pcaps").exists() else ant_root
        for pcap_path in sorted(segment_root.glob("*/*/*/Aseg_*/network.pcap")):
            rel = pcap_path.relative_to(segment_root)
            if len(rel.parts) != 5:
                continue
            split, engine, macro_group, segment_id, _ = rel.parts
            if split not in split_set:
                continue
            if engine_set and engine not in engine_set:
                continue
            if label_field not in {"MacroGroup", "macro_group", "macro_group_label"}:
                raise ValueError("segment source_mode only supports MacroGroup label_field")
            out.append(
                AntSample(
                    split=split,
                    engine=engine,
                    scenario=macro_group,
                    sample_id=segment_id,
                    pcap_path=pcap_path,
                    label=macro_group,
                    source_kind="segment",
                )
            )
            if max_samples is not None and len(out) >= max_samples:
                break
        return out

    risk_map = parse_risk_map(None)
    episodes = discover_episodes(ant_root, split_set, engine_set, max_samples, risk_map)
    for ep in episodes:
        out.append(
            AntSample(
                split=ep.split,
                engine=ep.engine,
                scenario=ep.scenario,
                sample_id=ep.episode_id,
                pcap_path=ep.pcap_path,
                label=ep.labels.value(label_field),
                source_kind="episode",
            )
        )
    return out


def keep_common_labels(samples: list[AntSample], splits: Iterable[str]) -> list[AntSample]:
    split_list = list(splits)
    by_split = {
        split: {sample.label for sample in samples if sample.split == split}
        for split in split_list
    }
    non_empty = [values for values in by_split.values() if values]
    if not non_empty:
        return []
    common = set.intersection(*non_empty)
    return [sample for sample in samples if sample.label in common]


def limit_per_split_label(samples: list[AntSample], limit: int | None) -> list[AntSample]:
    if limit is None or limit <= 0:
        return samples
    counts: dict[tuple[str, str], int] = {}
    out: list[AntSample] = []
    for sample in samples:
        key = (sample.split, sample.label)
        current = counts.get(key, 0)
        if current >= limit:
            continue
        counts[key] = current + 1
        out.append(sample)
    return out


def label_mapping(samples: Iterable[AntSample]) -> dict[str, int]:
    return {label: index for index, label in enumerate(sorted({sample.label for sample in samples}))}


def write_label_mapping(path: Path, label2id: dict[str, int]) -> None:
    payload = {"id_to_label": {str(index): label for label, index in sorted(label2id.items(), key=lambda item: item[1])}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def iter_sample_flows(sample: AntSample, protocols, flow_timeout: float):
    packets = read_pcap_packets(sample.pcap_path, protocols=protocols)
    dns = dns_ip_to_hostname(packets)
    return group_flows(packets, timeout=flow_timeout, dns_mapping=dns)


def sample_stem(sample: AntSample) -> str:
    return "_".join(safe_name(part) for part in [sample.split, sample.engine, sample.scenario, sample.sample_id])
