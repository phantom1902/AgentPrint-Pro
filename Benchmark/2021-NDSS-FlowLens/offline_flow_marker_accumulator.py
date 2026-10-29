#!/usr/bin/env python3
"""Offline reproduction of the FlowLens P4 Flow Marker Accumulator.

The P4 program counts packet-length bins for each installed IP-pair flow in
register arrays.  This script performs the same accumulation from pcap input
and also emits a mapping from each accumulated IP-pair marker back to ordinary
five-tuple flows, so predictions made at the IP-pair level can be projected
back to five-tuple flow labels during evaluation.
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import shutil
import struct
import subprocess
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np


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
        "label_policy": "majority_vote",
    },
    "accumulator": {
        "bin_width_shift": 4,
        "flow_bins": 94,
        "ip_pair_mode": "directional",
        "packet_length_source": "orig_len",
        "skip_non_tcp_udp": True,
        "skip_zero_payload_packets": False,
        "registered_ip_pairs": [],
        "registered_ip_pairs_csv": "",
        "truncation": {
            "mode": "count_all",
            "rules": [],
        },
    },
    "mapping": {
        "five_tuple_mode": "unidirectional",
        "five_tuple_timeout_seconds": 120.0,
    },
    "output": {
        "dir": "2021-NDSS-FlowLens/outputs/offline_flow_marker_accumulator",
        "marker_csv": "flow_markers.csv",
        "marker_npy": "flow_markers.npy",
        "ip_pair_mapping_csv": "ip_pair_mapping.csv",
        "five_tuple_mapping_csv": "five_tuple_mapping.csv",
        "metadata_json": "metadata.json",
    },
}


@dataclass
class Packet:
    packet_index: int
    timestamp: float
    packet_length: int
    payload_length: int
    src_ip: str
    dst_ip: str
    src_port: str
    dst_port: str
    protocol: str
    label: str = ""


@dataclass
class MarkerState:
    marker_id: str
    src_ip: str
    dst_ip: str
    counts: List[int]
    packets: int = 0
    bytes: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    label_votes: Counter = field(default_factory=Counter)
    five_tuple_ids: set = field(default_factory=set)

    def add_packet(self, packet: Packet, bin_index: int) -> None:
        self.counts[bin_index] += 1
        self.packets += 1
        self.bytes += packet.packet_length
        self.start_time = packet.timestamp if self.start_time is None else min(self.start_time, packet.timestamp)
        self.end_time = packet.timestamp if self.end_time is None else max(self.end_time, packet.timestamp)
        if packet.label:
            self.label_votes[packet.label] += 1


@dataclass
class FiveTupleFlow:
    flow_id: str
    marker_id: str
    src_ip: str
    dst_ip: str
    src_port: str
    dst_port: str
    protocol: str
    start_time: float
    end_time: float
    packets: int = 0
    bytes: int = 0
    label_votes: Counter = field(default_factory=Counter)

    def add_packet(self, packet: Packet) -> None:
        self.start_time = min(self.start_time, packet.timestamp)
        self.end_time = max(self.end_time, packet.timestamp)
        self.packets += 1
        self.bytes += packet.packet_length
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


def load_config(path: Optional[Path]) -> dict:
    if path is None:
        return DEFAULT_CONFIG
    with path.open("r", encoding="utf-8") as handle:
        return deep_merge(DEFAULT_CONFIG, json.load(handle))


def normalize_label(raw: str, config: dict) -> str:
    label = str(raw).strip()
    low = label.lower()
    benign = {str(value).lower() for value in config["labels"].get("benign_values", [])}
    malicious = {str(value).lower() for value in config["labels"].get("malicious_values", [])}
    if low in benign:
        return "benign"
    if low in malicious:
        return "malicious"
    return label


def load_packet_labels(path: str, config: dict) -> List[str]:
    if not path:
        return []
    delimiter = config["input"].get("packet_label_delimiter", ",")
    labels: List[str] = []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row_index, row in enumerate(reader):
            if row:
                if row_index == 0 and row[-1].strip().lower() in {"label", "x"}:
                    continue
                labels.append(normalize_label(row[-1], config))
    return labels


def label_for_packet(labels: Sequence[str], packet_index: int) -> str:
    label_index = packet_index - 1
    if 0 <= label_index < len(labels):
        return labels[label_index]
    return ""


def iter_pcap(path: Path, config: dict, packet_labels: Sequence[str]) -> Iterator[Packet]:
    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic == b"\x0a\x0d\x0d\x0a":
            yield from iter_pcapng_tshark(path, config, packet_labels)
            return
        if magic == b"\xd4\xc3\xb2\xa1":
            endian, resolution = "<", 1_000_000
        elif magic == b"\xa1\xb2\xc3\xd4":
            endian, resolution = ">", 1_000_000
        elif magic == b"\x4d\x3c\xb2\xa1":
            endian, resolution = "<", 1_000_000_000
        elif magic == b"\xa1\xb2\x3c\x4d":
            endian, resolution = ">", 1_000_000_000
        else:
            raise ValueError(f"{path} is not a classic pcap or pcapng file")

        header = handle.read(20)
        if len(header) != 20:
            raise ValueError(f"{path} has a truncated pcap global header")
        network = struct.unpack(endian + "I", header[16:20])[0]
        if network != 1:
            raise ValueError(f"{path} uses unsupported linktype {network}; only Ethernet pcap is supported")

        packet_index = 0
        while True:
            packet_header = handle.read(16)
            if not packet_header:
                break
            if len(packet_header) != 16:
                warnings.warn(f"Skipping truncated pcap packet header at end of file: {path}", RuntimeWarning)
                break
            ts_sec, ts_frac, incl_len, orig_len = struct.unpack(endian + "IIII", packet_header)
            frame = handle.read(incl_len)
            if len(frame) != incl_len:
                warnings.warn(f"Skipping truncated pcap packet body at end of file: {path}", RuntimeWarning)
                break
            packet_index += 1
            packet = parse_ethernet_ipv4(
                frame=frame,
                packet_index=packet_index,
                timestamp=ts_sec + ts_frac / float(resolution),
                captured_len=incl_len,
                orig_len=orig_len,
                config=config,
            )
            if packet is None:
                continue
            label = label_for_packet(packet_labels, packet_index)
            if label:
                packet.label = label
            yield packet


def iter_pcapng_tshark(path: Path, config: dict, packet_labels: Sequence[str]) -> Iterator[Packet]:
    tshark = shutil.which("tshark")
    if not tshark:
        raise RuntimeError("pcapng input requires tshark, but tshark was not found in PATH")

    command = [
        tshark,
        "-r",
        str(path),
        "-Y",
        "(ip.proto == 6) || (ip.proto == 17)",
        "-T",
        "fields",
        "-E",
        "separator=\t",
        "-E",
        "occurrence=f",
        "-e",
        "frame.number",
        "-e",
        "frame.time_epoch",
        "-e",
        "frame.len",
        "-e",
        "ip.len",
        "-e",
        "ip.src",
        "-e",
        "ip.dst",
        "-e",
        "ip.proto",
        "-e",
        "tcp.srcport",
        "-e",
        "tcp.dstport",
        "-e",
        "udp.srcport",
        "-e",
        "udp.dstport",
        "-e",
        "tcp.len",
        "-e",
        "udp.length",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout is not None
    for line in process.stdout:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 13:
            continue
        (
            frame_number,
            timestamp,
            frame_len,
            ip_len,
            src_ip,
            dst_ip,
            protocol,
            tcp_src,
            tcp_dst,
            udp_src,
            udp_dst,
            tcp_len,
            udp_length,
        ) = fields[:13]
        if protocol == "6":
            src_port, dst_port = tcp_src, tcp_dst
            payload_length = parse_int(tcp_len)
        elif protocol == "17":
            src_port, dst_port = udp_src, udp_dst
            payload_length = max(parse_int(udp_length) - 8, 0)
        else:
            continue
        packet_index = parse_int(frame_number)
        packet_length = select_packet_length(parse_int(frame_len), parse_int(frame_len), parse_int(ip_len), config)
        packet = Packet(
            packet_index=packet_index,
            timestamp=parse_float(timestamp),
            packet_length=packet_length,
            payload_length=payload_length,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            label=label_for_packet(packet_labels, packet_index),
        )
        if should_skip_packet(packet, config):
            continue
        yield packet

    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"tshark failed for {path}: {stderr.strip()}")


def parse_ethernet_ipv4(
    frame: bytes,
    packet_index: int,
    timestamp: float,
    captured_len: int,
    orig_len: int,
    config: dict,
) -> Optional[Packet]:
    if len(frame) < 34:
        return None
    offset = 14
    ethertype = struct.unpack("!H", frame[12:14])[0]
    while ethertype in {0x8100, 0x88A8}:
        if len(frame) < offset + 4:
            return None
        ethertype = struct.unpack("!H", frame[offset + 2 : offset + 4])[0]
        offset += 4
    if ethertype != 0x0800 or len(frame) < offset + 20:
        return None

    version_ihl = frame[offset]
    if version_ihl >> 4 != 4:
        return None
    ihl = (version_ihl & 0x0F) * 4
    if len(frame) < offset + ihl:
        return None
    total_length = struct.unpack("!H", frame[offset + 2 : offset + 4])[0]
    protocol = frame[offset + 9]
    src_ip = ".".join(str(part) for part in frame[offset + 12 : offset + 16])
    dst_ip = ".".join(str(part) for part in frame[offset + 16 : offset + 20])
    transport_offset = offset + ihl
    ip_payload_length = max(total_length - ihl, 0)

    if protocol == 6:
        if len(frame) < transport_offset + 20:
            return None
        src_port, dst_port = struct.unpack("!HH", frame[transport_offset : transport_offset + 4])
        tcp_header_length = (frame[transport_offset + 12] >> 4) * 4
        payload_length = max(ip_payload_length - tcp_header_length, 0)
    elif protocol == 17:
        if len(frame) < transport_offset + 8:
            return None
        src_port, dst_port, udp_length = struct.unpack("!HHH", frame[transport_offset : transport_offset + 6])
        payload_length = max(udp_length - 8, 0)
    elif config["accumulator"].get("skip_non_tcp_udp", True):
        return None
    else:
        src_port, dst_port = "", ""
        payload_length = ip_payload_length

    packet = Packet(
        packet_index=packet_index,
        timestamp=timestamp,
        packet_length=select_packet_length(captured_len, orig_len, total_length, config),
        payload_length=payload_length,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=str(src_port),
        dst_port=str(dst_port),
        protocol=str(protocol),
    )
    if should_skip_packet(packet, config):
        return None
    return packet


def select_packet_length(captured_len: int, orig_len: int, ip_len: int, config: dict) -> int:
    source = config["accumulator"].get("packet_length_source", "orig_len")
    if source == "captured_len":
        return captured_len
    if source == "ip_len":
        return ip_len
    if source == "orig_len":
        return orig_len
    raise ValueError(f"Unsupported accumulator.packet_length_source: {source}")


def should_skip_packet(packet: Packet, config: dict) -> bool:
    if not packet.src_ip or not packet.dst_ip:
        return True
    if config["accumulator"].get("skip_zero_payload_packets", False) and packet.payload_length <= 0:
        return True
    return False


def parse_int(value: Union[str, int], default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value: Union[str, float], default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def canonical_ip_pair(packet: Packet, mode: str) -> Tuple[str, str]:
    if mode == "directional":
        return packet.src_ip, packet.dst_ip
    if mode == "bidirectional":
        src = ipaddress.ip_address(packet.src_ip).packed
        dst = ipaddress.ip_address(packet.dst_ip).packed
        return (packet.src_ip, packet.dst_ip) if src <= dst else (packet.dst_ip, packet.src_ip)
    raise ValueError(f"Unsupported accumulator.ip_pair_mode: {mode}")


def marker_id(src_ip: str, dst_ip: str) -> str:
    return f"ippair:{src_ip}>{dst_ip}"


def five_tuple_key(packet: Packet, mode: str) -> Tuple[str, str, str, str, str]:
    if mode == "unidirectional":
        return (packet.src_ip, packet.dst_ip, packet.src_port, packet.dst_port, packet.protocol)
    if mode == "bidirectional":
        left = (packet.src_ip, packet.src_port)
        right = (packet.dst_ip, packet.dst_port)
        if left <= right:
            return (packet.src_ip, packet.dst_ip, packet.src_port, packet.dst_port, packet.protocol)
        return (packet.dst_ip, packet.src_ip, packet.dst_port, packet.src_port, packet.protocol)
    raise ValueError(f"Unsupported mapping.five_tuple_mode: {mode}")


def truncate_bin(pre_index: int, config: dict) -> Optional[int]:
    truncation = config["accumulator"].get("truncation", {})
    mode = truncation.get("mode", "count_all")
    flow_bins = int(config["accumulator"]["flow_bins"])

    if mode == "count_all":
        return pre_index if 0 <= pre_index < flow_bins else None

    if mode == "drop_unlisted":
        for rule in truncation.get("rules", []):
            if rule_matches_bin(rule, pre_index):
                new_index = int(rule.get("new_index", rule.get("bin", pre_index)))
                flag = int(rule.get("flag", 1))
                if flag == 0:
                    return None
                return new_index if 0 <= new_index < flow_bins else None
        return None

    if mode == "rules_with_default_drop":
        for rule in truncation.get("rules", []):
            if rule_matches_bin(rule, pre_index):
                new_index = int(rule.get("new_index", rule.get("bin", pre_index)))
                flag = int(rule.get("flag", 1))
                if flag == 0:
                    return None
                return new_index if 0 <= new_index < flow_bins else None
        return None

    raise ValueError(f"Unsupported accumulator.truncation.mode: {mode}")


def rule_matches_bin(rule: dict, pre_index: int) -> bool:
    if "pre_index" in rule:
        return pre_index == int(rule["pre_index"])
    if "start" in rule or "end" in rule:
        start = int(rule.get("start", pre_index))
        end = int(rule.get("end", pre_index))
        return start <= pre_index <= end
    return False


def assign_label(votes: Counter, config: dict) -> str:
    if not votes:
        return ""
    policy = config["labels"].get("label_policy", "majority_vote")
    if policy == "majority_vote":
        return votes.most_common(1)[0][0]
    if policy == "any_malicious":
        return "malicious" if votes.get("malicious", 0) > 0 else votes.most_common(1)[0][0]
    if policy == "mixed":
        return "mixed" if len(votes) > 1 else votes.most_common(1)[0][0]
    raise ValueError(f"Unsupported labels.label_policy: {policy}")


def load_registered_ip_pairs(config: dict) -> Optional[set]:
    pairs = set()
    for item in config["accumulator"].get("registered_ip_pairs", []):
        if isinstance(item, dict):
            src_ip = item.get("src_ip", "")
            dst_ip = item.get("dst_ip", "")
        else:
            src_ip, dst_ip = item[:2]
        if src_ip and dst_ip:
            pairs.add((str(src_ip), str(dst_ip)))

    csv_path = config["accumulator"].get("registered_ip_pairs_csv", "")
    if csv_path:
        with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                src_ip = row.get("SRC_IP") or row.get("src_ip") or row.get("src") or ""
                dst_ip = row.get("DST_IP") or row.get("dst_ip") or row.get("dst") or ""
                if src_ip and dst_ip:
                    pairs.add((src_ip.strip(), dst_ip.strip()))

    return pairs if pairs else None


def accumulate(packets: Iterable[Packet], config: dict) -> Tuple[List[MarkerState], List[FiveTupleFlow]]:
    bin_shift = int(config["accumulator"]["bin_width_shift"])
    flow_bins = int(config["accumulator"]["flow_bins"])
    pair_mode = config["accumulator"].get("ip_pair_mode", "directional")
    tuple_mode = config["mapping"].get("five_tuple_mode", "unidirectional")
    timeout = float(config["mapping"].get("five_tuple_timeout_seconds", 120.0))
    registered_ip_pairs = load_registered_ip_pairs(config)

    markers: Dict[str, MarkerState] = {}
    five_tuple_flows: List[FiveTupleFlow] = []
    active_flows: Dict[Tuple[str, str, str, str, str], FiveTupleFlow] = {}
    flow_sequence: defaultdict[Tuple[str, str, str, str, str], int] = defaultdict(int)

    for packet in packets:
        src_ip, dst_ip = canonical_ip_pair(packet, pair_mode)
        if registered_ip_pairs is not None and (src_ip, dst_ip) not in registered_ip_pairs:
            continue
        mid = marker_id(src_ip, dst_ip)
        marker = markers.get(mid)
        if marker is None:
            marker = MarkerState(mid, src_ip, dst_ip, [0] * flow_bins)
            markers[mid] = marker

        pre_index = packet.packet_length >> bin_shift
        bin_index = truncate_bin(pre_index, config)
        if bin_index is not None:
            marker.add_packet(packet, bin_index)

        fkey = five_tuple_key(packet, tuple_mode)
        current = active_flows.get(fkey)
        if current is None or packet.timestamp - current.end_time > timeout:
            flow_sequence[fkey] += 1
            flow_id = "flow:{}>{}:{}>{}:{}:{}".format(
                fkey[0],
                fkey[1],
                fkey[2],
                fkey[3],
                fkey[4],
                flow_sequence[fkey],
            )
            current = FiveTupleFlow(
                flow_id=flow_id,
                marker_id=mid,
                src_ip=fkey[0],
                dst_ip=fkey[1],
                src_port=fkey[2],
                dst_port=fkey[3],
                protocol=fkey[4],
                start_time=packet.timestamp,
                end_time=packet.timestamp,
            )
            active_flows[fkey] = current
            five_tuple_flows.append(current)
        current.add_packet(packet)
        marker.five_tuple_ids.add(current.flow_id)

    return sorted(markers.values(), key=lambda item: item.marker_id), five_tuple_flows


def write_outputs(markers: Sequence[MarkerState], flows: Sequence[FiveTupleFlow], config: dict, pcap_path: Path) -> dict:
    output_cfg = config["output"]
    output_dir = Path(output_cfg["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_csv = output_dir / output_cfg["marker_csv"]
    marker_npy = output_dir / output_cfg["marker_npy"]
    ip_pair_mapping_csv = output_dir / output_cfg["ip_pair_mapping_csv"]
    five_tuple_mapping_csv = output_dir / output_cfg["five_tuple_mapping_csv"]
    metadata_json = output_dir / output_cfg["metadata_json"]

    matrix = np.array([marker.counts for marker in markers], dtype=np.uint32)
    np.save(marker_npy, matrix)

    bin_headers = [f"BIN_{index}" for index in range(int(config["accumulator"]["flow_bins"]))]
    with marker_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["MARKER_ID", "SRC_IP", "DST_IP", "PACKETS", "BYTES", "START_TIME", "END_TIME", "LABEL"] + bin_headers)
        for marker in markers:
            writer.writerow(
                [
                    marker.marker_id,
                    marker.src_ip,
                    marker.dst_ip,
                    marker.packets,
                    marker.bytes,
                    format_time(marker.start_time),
                    format_time(marker.end_time),
                    assign_label(marker.label_votes, config),
                    *marker.counts,
                ]
            )

    with ip_pair_mapping_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["MARKER_ID", "SRC_IP", "DST_IP", "FIVE_TUPLE_FLOW_IDS", "LABEL", "LABEL_COUNTS"])
        for marker in markers:
            writer.writerow(
                [
                    marker.marker_id,
                    marker.src_ip,
                    marker.dst_ip,
                    ";".join(sorted(marker.five_tuple_ids)),
                    assign_label(marker.label_votes, config),
                    json.dumps(dict(sorted(marker.label_votes.items())), sort_keys=True),
                ]
            )

    with five_tuple_mapping_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "FLOW_ID",
                "MARKER_ID",
                "SRC_IP",
                "DST_IP",
                "SRC_PORT",
                "DST_PORT",
                "PROTOCOL",
                "START_TIME",
                "END_TIME",
                "PACKETS",
                "BYTES",
                "LABEL",
                "LABEL_COUNTS",
            ]
        )
        for flow in flows:
            writer.writerow(
                [
                    flow.flow_id,
                    flow.marker_id,
                    flow.src_ip,
                    flow.dst_ip,
                    flow.src_port,
                    flow.dst_port,
                    flow.protocol,
                    f"{flow.start_time:.9f}",
                    f"{flow.end_time:.9f}",
                    flow.packets,
                    flow.bytes,
                    assign_label(flow.label_votes, config),
                    json.dumps(dict(sorted(flow.label_votes.items())), sort_keys=True),
                ]
            )

    marker_label_counts = Counter(assign_label(marker.label_votes, config) for marker in markers)
    flow_label_counts = Counter(assign_label(flow.label_votes, config) for flow in flows)
    metadata = {
        "pcap": str(pcap_path),
        "output_dir": str(output_dir),
        "marker_csv": str(marker_csv),
        "marker_npy": str(marker_npy),
        "ip_pair_mapping_csv": str(ip_pair_mapping_csv),
        "five_tuple_mapping_csv": str(five_tuple_mapping_csv),
        "ip_pair_mode": config["accumulator"]["ip_pair_mode"],
        "five_tuple_mode": config["mapping"]["five_tuple_mode"],
        "five_tuple_timeout_seconds": config["mapping"]["five_tuple_timeout_seconds"],
        "packet_length_source": config["accumulator"]["packet_length_source"],
        "bin_width_shift": config["accumulator"]["bin_width_shift"],
        "flow_bins": config["accumulator"]["flow_bins"],
        "markers": len(markers),
        "five_tuple_flows": len(flows),
        "marker_matrix_shape": list(matrix.shape),
        "marker_label_counts": dict(sorted(marker_label_counts.items())),
        "five_tuple_label_counts": dict(sorted(flow_label_counts.items())),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def format_time(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.9f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline FlowLens P4 flow-marker accumulator for pcap input.")
    parser.add_argument("--config", help="Optional JSON config. CLI values override it.")
    parser.add_argument("--pcap", help="Input pcap/pcapng path.")
    parser.add_argument("--packet-labels", help="Optional packet-level label txt/csv, one row per pcap packet.")
    parser.add_argument("--output-dir", help="Output directory.")
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    if args.pcap:
        config["input"]["pcap"] = args.pcap
    if args.packet_labels:
        config["input"]["packet_labels"] = args.packet_labels
    if args.output_dir:
        config["output"]["dir"] = args.output_dir

    pcap_raw = config["input"].get("pcap", "")
    if not pcap_raw:
        raise ValueError("Input pcap is required via --pcap or input.pcap in config")
    pcap = Path(pcap_raw)
    packet_labels = load_packet_labels(config["input"].get("packet_labels", ""), config)
    packets = iter_pcap(pcap, config, packet_labels)
    if config["input"].get("max_packets") is not None:
        packets = islice(packets, int(config["input"]["max_packets"]))
    markers, flows = accumulate(packets, config)
    metadata = write_outputs(markers, flows, config, pcap)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
