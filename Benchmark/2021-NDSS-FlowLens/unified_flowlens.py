#!/usr/bin/env python3
"""Unified data adapter for the FlowLens Botnet/PeerShark pipeline.

The original FlowLens botnet evaluation consumes PeerShark packet CSV files and
derives IP-pair flows/superflows before training.  This adapter keeps that
feature definition but accepts benchmark-level pcap, packet CSV, or flow CSV
inputs plus a separate flow-label CSV.
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import os
import shutil
import struct
import subprocess
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


DEFAULT_CONFIG = {
    "input": {
        "format": "auto",
        "paths": [],
        "delimiter": ",",
        "has_header": True,
    },
    "labels": {
        "path": "",
        "packet_path": "",
        "delimiter": ",",
        "packet_delimiter": ",",
        "default_label": "unknown",
        "benign_values": ["0", "benign", "normal", "false"],
        "malicious_values": ["1", "malicious", "attack", "true"],
    },
    "flow": {
        "definition": "bidirectional_ip_pair",
        "time_gap_seconds": 2000,
        "flow_gap_seconds": 3600,
        "skip_zero_payload_packets": True,
    },
    "quantization": {
        "packet_size_bin_width": 1,
        "ipt_bin_width_seconds": 0,
    },
    "output": {
        "dataset_csv": "2021-NDSS-FlowLens/outputs/unified_flowlens_dataset.csv",
        "intermediate_dir": "2021-NDSS-FlowLens/outputs/unified_intermediate",
        "write_intermediate": True,
    },
}

PACKET_COLUMNS = {
    "src_ip": ["src_ip", "source_ip", "src", "ip.src", "source"],
    "dst_ip": ["dst_ip", "destination_ip", "dst", "ip.dst", "dest"],
    "src_port": ["src_port", "source_port", "sport", "id.orig_p"],
    "dst_port": ["dst_port", "destination_port", "dport", "id.resp_p"],
    "protocol": ["protocol", "proto", "ip_proto", "ip.proto"],
    "timestamp": ["timestamp", "time", "ts", "frame.time_epoch"],
    "payload_length": [
        "payload_length",
        "packet_length",
        "length",
        "size",
        "tcp.len",
        "udp.length",
        "bytes",
    ],
    "label": ["label", "class", "y"],
}

FLOW_COLUMNS = {
    "src_ip": PACKET_COLUMNS["src_ip"],
    "dst_ip": PACKET_COLUMNS["dst_ip"],
    "src_port": PACKET_COLUMNS["src_port"],
    "dst_port": PACKET_COLUMNS["dst_port"],
    "protocol": PACKET_COLUMNS["protocol"],
    "start_time": ["start_time", "flow_start", "first_ts", "t_start", "ts"],
    "end_time": ["end_time", "flow_end", "last_ts", "t_end"],
    "duration": ["duration", "flow_duration", "t_duration"],
    "packets": ["packets", "packet_count", "packet_ct", "numberofpackets", "#packets"],
    "bytes": ["bytes", "byte_count", "sum_bytes", "totalbytestransmitted", "#bytes"],
    "median_ipt": ["median_ipt", "medianipt", "interarrivaltime", "avg_interval"],
    "label": ["label", "class", "y"],
}


@dataclass
class PacketRecord:
    src_ip: str
    dst_ip: str
    src_port: str
    dst_port: str
    protocol: str
    timestamp: float
    payload_length: int
    label: str = ""


@dataclass
class LabelRule:
    src_ip: str
    dst_ip: str
    src_port: str
    dst_port: str
    protocol: str
    start_time: Optional[float]
    end_time: Optional[float]
    label: str


@dataclass
class FlowState:
    ip1: str
    ip2: str
    key: Tuple[bytes, bytes]
    n_packet1: int = 0
    n_byte1: int = 0
    t_start1: float = 0.0
    t_end1: float = 0.0
    ipt1: List[float] = field(default_factory=list)
    n_packet2: int = 0
    n_byte2: int = 0
    t_start2: float = 0.0
    t_end2: float = 0.0
    ipt2: List[float] = field(default_factory=list)
    label_votes: Counter = field(default_factory=Counter)
    five_tuples: set = field(default_factory=set)

    @classmethod
    def from_packet(cls, packet: PacketRecord) -> "FlowState":
        ip1, ip2, key = canonical_ip_pair(packet.src_ip, packet.dst_ip)
        flow = cls(ip1=ip1, ip2=ip2, key=key)
        flow.add_packet(packet)
        return flow

    def add_packet(self, packet: PacketRecord) -> None:
        self.five_tuples.add(
            (
                packet.src_ip,
                packet.dst_ip,
                str(packet.src_port),
                str(packet.dst_port),
                str(packet.protocol),
            )
        )
        if packet.label:
            self.label_votes[packet.label] += 1

        if packet.src_ip == self.ip1 and packet.dst_ip == self.ip2:
            if self.n_packet1 == 0:
                self.t_start1 = packet.timestamp
                self.t_end1 = packet.timestamp
            else:
                if self.t_end1 < packet.timestamp:
                    self.ipt1.append(packet.timestamp - self.t_end1)
                    self.t_end1 = packet.timestamp
                elif self.t_start1 > packet.timestamp:
                    self.ipt1.append(self.t_start1 - packet.timestamp)
                    self.t_start1 = packet.timestamp
            self.n_packet1 += 1
            self.n_byte1 += packet.payload_length
            return

        if packet.src_ip == self.ip2 and packet.dst_ip == self.ip1:
            if self.n_packet2 == 0:
                self.t_start2 = packet.timestamp
                self.t_end2 = packet.timestamp
            else:
                if self.t_end2 < packet.timestamp:
                    self.ipt2.append(packet.timestamp - self.t_end2)
                    self.t_end2 = packet.timestamp
                elif self.t_start2 > packet.timestamp:
                    self.ipt2.append(self.t_start2 - packet.timestamp)
                    self.t_start2 = packet.timestamp
            self.n_packet2 += 1
            self.n_byte2 += packet.payload_length
            return

        raise ValueError("packet endpoints do not belong to this FlowLens IP-pair flow")

    def add_flow(self, other: "FlowState") -> None:
        self.n_packet1 += other.n_packet1
        self.n_byte1 += other.n_byte1
        self.n_packet2 += other.n_packet2
        self.n_byte2 += other.n_byte2
        self.ipt1.extend(other.ipt1)
        self.ipt2.extend(other.ipt2)
        self.label_votes.update(other.label_votes)
        self.five_tuples.update(other.five_tuples)
        self.t_start1 = merge_start(self.t_start1, other.t_start1)
        self.t_start2 = merge_start(self.t_start2, other.t_start2)
        self.t_end1 = max(self.t_end1, other.t_end1)
        self.t_end2 = max(self.t_end2, other.t_end2)

    @property
    def packets(self) -> int:
        return self.n_packet1 + self.n_packet2

    @property
    def bytes(self) -> int:
        return self.n_byte1 + self.n_byte2

    @property
    def start(self) -> float:
        return merge_start(self.t_start1, self.t_start2)

    @property
    def end(self) -> float:
        return max(self.t_end1, self.t_end2)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def median_ipt(self) -> float:
        combined = self.ipt1 + self.ipt2
        return median(combined)


@dataclass
class TrainingRow:
    packets: int
    bytes: int
    median_ipt: float
    duration: float
    label: str


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


def canonical_ip_pair(src_ip: str, dst_ip: str) -> Tuple[str, str, Tuple[bytes, bytes]]:
    src = ipaddress.ip_address(src_ip).packed
    dst = ipaddress.ip_address(dst_ip).packed
    if src < dst:
        return src_ip, dst_ip, (src, dst)
    return dst_ip, src_ip, (dst, src)


def merge_start(a: float, b: float) -> float:
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


def round_to_nearest(value: int, width: int) -> int:
    if width <= 1:
        return value
    remainder = value % width
    if remainder + remainder >= width:
        return value + width - remainder
    return value - remainder


def normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def build_column_lookup(fieldnames: Sequence[str]) -> Dict[str, str]:
    return {normalize_header(name): name for name in fieldnames if name is not None}


def pick_column(row: dict, lookup: Dict[str, str], choices: Sequence[str], default: str = "") -> str:
    for choice in choices:
        source = lookup.get(normalize_header(choice))
        if source is not None:
            value = row.get(source, "")
            return "" if value is None else str(value).strip()
    return default


def normalize_label(raw: str, config: dict) -> str:
    label = str(raw).strip()
    low = label.lower()
    if low in {str(v).lower() for v in config["labels"]["benign_values"]}:
        return "benign"
    if low in {str(v).lower() for v in config["labels"]["malicious_values"]}:
        return "malicious"
    return label


def expand_input_paths(paths: Sequence[str], input_format: str) -> List[Path]:
    expanded: List[Path] = []
    suffixes = {".pcap", ".cap", ".csv"} if input_format == "auto" else None
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and (suffixes is None or child.suffix.lower() in suffixes):
                    expanded.append(child)
        elif path.is_file():
            expanded.append(path)
        else:
            raise FileNotFoundError(raw_path)
    return expanded


def detect_input_format(path: Path, configured: str) -> str:
    if configured != "auto":
        return configured
    if path.suffix.lower() in {".pcap", ".cap"}:
        return "pcap"
    return detect_csv_format(path)


def detect_csv_format(path: Path) -> str:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        sample = handle.readline()
    fields = [normalize_header(value) for value in sample.strip().split(",")]
    if any(name in fields for name in ["timestamp", "frame.time_epoch", "time", "ts"]):
        return "packet_csv"
    if any(name in fields for name in ["numberofpackets", "packet_count", "packets", "#packets"]):
        return "flow_csv"
    return "packet_csv"


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def label_from_packet_index(packet_labels: Sequence[str], packet_index: int) -> str:
    label_index = packet_index - 1
    if 0 <= label_index < len(packet_labels):
        return packet_labels[label_index]
    return ""


def iter_packet_csv(path: Path, config: dict, packet_labels: Sequence[str] = ()) -> Iterator[PacketRecord]:
    delimiter = config["input"].get("delimiter", ",")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        first = handle.readline()
        handle.seek(0)
        first_fields = [field.strip() for field in first.strip().split(delimiter)]
        has_header = config["input"].get("has_header", True)

        src_header_names = {normalize_header(name) for name in PACKET_COLUMNS["src_ip"]}
        if has_header and any(normalize_header(field) in src_header_names for field in first_fields):
            reader = csv.DictReader(handle, delimiter=delimiter)
            lookup = build_column_lookup(reader.fieldnames or [])
            for packet_index, row in enumerate(reader, start=1):
                label = pick_column(row, lookup, PACKET_COLUMNS["label"])
                if not label:
                    label = label_from_packet_index(packet_labels, packet_index)
                yield PacketRecord(
                    src_ip=pick_column(row, lookup, PACKET_COLUMNS["src_ip"]),
                    dst_ip=pick_column(row, lookup, PACKET_COLUMNS["dst_ip"]),
                    src_port=pick_column(row, lookup, PACKET_COLUMNS["src_port"]),
                    dst_port=pick_column(row, lookup, PACKET_COLUMNS["dst_port"]),
                    protocol=pick_column(row, lookup, PACKET_COLUMNS["protocol"]),
                    timestamp=parse_float(pick_column(row, lookup, PACKET_COLUMNS["timestamp"])),
                    payload_length=parse_int(pick_column(row, lookup, PACKET_COLUMNS["payload_length"])),
                    label=normalize_label(label, config) if label else "",
                )
            return

        reader = csv.reader(handle, delimiter=delimiter)
        for packet_index, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) >= 7:
                label = row[7].strip() if len(row) > 7 else ""
                if not label:
                    label = label_from_packet_index(packet_labels, packet_index)
                yield PacketRecord(
                    src_ip=row[0].strip(),
                    dst_ip=row[1].strip(),
                    src_port=row[2].strip(),
                    dst_port=row[3].strip(),
                    protocol=row[4].strip(),
                    timestamp=parse_float(row[5]),
                    payload_length=parse_int(row[6]),
                    label=normalize_label(label, config) if label else "",
                )
            elif len(row) >= 5:
                label = label_from_packet_index(packet_labels, packet_index)
                yield PacketRecord(
                    src_ip=row[0].strip(),
                    dst_ip=row[1].strip(),
                    src_port="",
                    dst_port="",
                    protocol=row[2].strip(),
                    timestamp=parse_float(row[3]),
                    payload_length=parse_int(row[4]),
                    label=label,
                )


def iter_flow_csv(path: Path, config: dict, labels: Sequence[LabelRule]) -> Iterator[TrainingRow]:
    delimiter = config["input"].get("delimiter", ",")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        first = handle.readline()
        handle.seek(0)
        fields = [normalize_header(value) for value in first.strip().split(delimiter)]
        has_header = any(name in fields for name in ["src_ip", "numberofpackets", "packets"])

        if has_header:
            reader = csv.DictReader(handle, delimiter=delimiter)
            lookup = build_column_lookup(reader.fieldnames or [])
            for row in reader:
                packets = parse_int(pick_column(row, lookup, FLOW_COLUMNS["packets"]))
                byte_count = parse_int(pick_column(row, lookup, FLOW_COLUMNS["bytes"]))
                start = parse_float(pick_column(row, lookup, FLOW_COLUMNS["start_time"]))
                end = parse_float(pick_column(row, lookup, FLOW_COLUMNS["end_time"]))
                duration = parse_float(pick_column(row, lookup, FLOW_COLUMNS["duration"]), end - start)
                median_ipt_raw = pick_column(row, lookup, FLOW_COLUMNS["median_ipt"])
                median_ipt = parse_float(median_ipt_raw, duration / max(packets - 1, 1))
                label = pick_column(row, lookup, FLOW_COLUMNS["label"])
                if not label:
                    flow = flow_state_from_summary(
                        src_ip=pick_column(row, lookup, FLOW_COLUMNS["src_ip"]),
                        dst_ip=pick_column(row, lookup, FLOW_COLUMNS["dst_ip"]),
                        src_port=pick_column(row, lookup, FLOW_COLUMNS["src_port"]),
                        dst_port=pick_column(row, lookup, FLOW_COLUMNS["dst_port"]),
                        protocol=pick_column(row, lookup, FLOW_COLUMNS["protocol"]),
                        start=start,
                        end=end,
                        packets=packets,
                        byte_count=byte_count,
                        median_ipt=median_ipt,
                        label="",
                    )
                    label = label_for_flow(flow, labels, config["labels"]["default_label"]) if flow else config["labels"]["default_label"]
                yield TrainingRow(
                    packets=packets,
                    bytes=byte_count,
                    median_ipt=median_ipt,
                    duration=duration,
                    label=normalize_label(label, config),
                )
            return

        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if len(row) >= 12:
                packets = parse_int(row[2]) + parse_int(row[7])
                byte_count = parse_int(row[3]) + parse_int(row[8])
                start = min_nonzero(parse_float(row[4]), parse_float(row[9]))
                end = max(parse_float(row[5]), parse_float(row[10]))
                median_ipt = median([parse_float(row[6]), parse_float(row[11])])
                label = row[12].strip() if len(row) > 12 else config["labels"]["default_label"]
                if len(row) <= 12:
                    flow = flow_state_from_summary(row[0], row[1], "", "", "", start, end, packets, byte_count, median_ipt, "")
                    label = label_for_flow(flow, labels, config["labels"]["default_label"]) if flow else label
                yield TrainingRow(packets, byte_count, median_ipt, end - start, normalize_label(label, config))
            elif len(row) >= 8:
                label = row[8].strip() if len(row) > 8 else config["labels"]["default_label"]
                if len(row) <= 8:
                    flow = flow_state_from_summary(
                        row[0],
                        row[1],
                        "",
                        "",
                        "",
                        parse_float(row[5]),
                        parse_float(row[6]),
                        parse_int(row[2]),
                        parse_int(row[3]),
                        parse_float(row[4]),
                        "",
                    )
                    label = label_for_flow(flow, labels, config["labels"]["default_label"]) if flow else label
                yield TrainingRow(
                    packets=parse_int(row[2]),
                    bytes=parse_int(row[3]),
                    median_ipt=parse_float(row[4]),
                    duration=parse_float(row[7]),
                    label=normalize_label(label, config),
                )


def min_nonzero(a: float, b: float) -> float:
    if a == 0:
        return b
    if b == 0:
        return a
    return min(a, b)


def flow_state_from_summary(
    src_ip: str,
    dst_ip: str,
    src_port: str,
    dst_port: str,
    protocol: str,
    start: float,
    end: float,
    packets: int,
    byte_count: int,
    median_ipt: float,
    label: str,
) -> Optional[FlowState]:
    if not src_ip or not dst_ip:
        return None
    ip1, ip2, key = canonical_ip_pair(src_ip, dst_ip)
    flow = FlowState(ip1=ip1, ip2=ip2, key=key)
    if src_ip == ip1 and dst_ip == ip2:
        flow.n_packet1 = packets
        flow.n_byte1 = byte_count
        flow.t_start1 = start
        flow.t_end1 = end
        flow.ipt1 = [median_ipt] if median_ipt else []
    else:
        flow.n_packet2 = packets
        flow.n_byte2 = byte_count
        flow.t_start2 = start
        flow.t_end2 = end
        flow.ipt2 = [median_ipt] if median_ipt else []
    if src_port or dst_port or protocol:
        flow.five_tuples.add((src_ip, dst_ip, str(src_port), str(dst_port), str(protocol)))
    if label:
        flow.label_votes[normalize_label(label, DEFAULT_CONFIG)] += 1
    return flow


def iter_pcap(path: Path, config: dict, packet_labels: Sequence[str] = ()) -> Iterator[PacketRecord]:
    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic == b"\x0a\x0d\x0d\x0a":
            yield from iter_pcap_tshark(path, config, packet_labels)
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
            raise ValueError(f"{path} is not a classic pcap file")

        header = handle.read(20)
        if len(header) != 20:
            raise ValueError(f"{path} has a truncated pcap global header")
        network = struct.unpack(endian + "I", header[16:20])[0]
        if network not in {1, 101}:
            raise ValueError(
                f"{path} uses unsupported pcap linktype {network}; "
                "only Ethernet (1) and Raw IP (101) are supported"
            )

        packet_index = 0
        while True:
            packet_header = handle.read(16)
            if not packet_header:
                break
            if len(packet_header) != 16:
                warnings.warn(f"Skipping truncated pcap packet header at end of file: {path}", RuntimeWarning)
                break
            ts_sec, ts_frac, incl_len, _orig_len = struct.unpack(endian + "IIII", packet_header)
            frame = handle.read(incl_len)
            if len(frame) != incl_len:
                warnings.warn(f"Skipping truncated pcap packet body at end of file: {path}", RuntimeWarning)
                break
            packet_index += 1
            timestamp = ts_sec + (ts_frac / float(resolution))
            parsed = (
                parse_ethernet_ipv4(frame, timestamp)
                if network == 1
                else parse_ipv4_packet(frame, timestamp)
            )
            if parsed is None:
                continue
            if config["flow"].get("skip_zero_payload_packets", True) and parsed.payload_length <= 0:
                continue
            label = label_from_packet_index(packet_labels, packet_index)
            if label:
                parsed.label = label
            yield parsed


def iter_pcap_tshark(path: Path, config: dict, packet_labels: Sequence[str]) -> Iterator[PacketRecord]:
    tshark = shutil.which("tshark")
    if not tshark:
        raise RuntimeError("pcapng input requires tshark, but tshark was not found in PATH")

    command = [
        tshark,
        "-r",
        str(path),
        "-Y",
        "(ip.proto == 6) || (ip.proto == 17) || arp",
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
        "-e",
        "frame.len",
        "-e",
        "eth.src",
        "-e",
        "eth.dst",
        "-e",
        "eth.type",
        "-e",
        "arp.src.proto_ipv4",
        "-e",
        "arp.dst.proto_ipv4",
        "-e",
        "arp.src.hw_mac",
        "-e",
        "arp.dst.hw_mac",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 19:
            continue
        (
            frame_number,
            timestamp,
            src_ip,
            dst_ip,
            protocol,
            tcp_src,
            tcp_dst,
            udp_src,
            udp_dst,
            tcp_len,
            udp_length,
            frame_len,
            eth_src,
            eth_dst,
            eth_type,
            arp_src_ip,
            arp_dst_ip,
            arp_src_mac,
            arp_dst_mac,
        ) = fields[:19]
        if protocol == "6":
            src_port = tcp_src
            dst_port = tcp_dst
            payload_length = parse_int(tcp_len)
        elif protocol == "17":
            src_port = udp_src
            dst_port = udp_dst
            payload_length = max(parse_int(udp_length) - 8, 0)
        elif eth_type.lower() in {"0x0806", "0806"} or arp_src_ip or arp_dst_ip:
            src_ip = arp_src_ip or arp_src_mac or eth_src
            dst_ip = arp_dst_ip or arp_dst_mac or eth_dst
            protocol = "2054"
            src_port = "0"
            dst_port = "0"
            payload_length = parse_int(frame_len)
        else:
            continue
        if not src_ip or not dst_ip or not protocol:
            continue
        if "," in src_ip or "," in dst_ip or "," in protocol:
            continue
        if config["flow"].get("skip_zero_payload_packets", True) and payload_length <= 0:
            continue
        label = label_from_packet_index(packet_labels, parse_int(frame_number))
        yield PacketRecord(
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            timestamp=parse_float(timestamp),
            payload_length=payload_length,
            label=label,
        )

    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"tshark failed for {path}: {stderr.strip()}")


def parse_ethernet_ipv4(frame: bytes, timestamp: float) -> Optional[PacketRecord]:
    if len(frame) < 34:
        return None
    offset = 14
    ethertype = struct.unpack("!H", frame[12:14])[0]
    while ethertype in {0x8100, 0x88A8}:
        if len(frame) < offset + 4:
            return None
        ethertype = struct.unpack("!H", frame[offset + 2 : offset + 4])[0]
        offset += 4
    if ethertype == 0x0806:
        if len(frame) < offset + 28:
            return None
        hardware_length = frame[offset + 4]
        protocol_length = frame[offset + 5]
        if hardware_length == 6 and protocol_length == 4 and len(frame) >= offset + 28:
            sender_ip = ".".join(str(part) for part in frame[offset + 14 : offset + 18])
            target_ip = ".".join(str(part) for part in frame[offset + 24 : offset + 28])
            sender_mac = ":".join(f"{part:02x}" for part in frame[offset + 8 : offset + 14])
            target_mac = ":".join(f"{part:02x}" for part in frame[offset + 18 : offset + 24])
            return PacketRecord(
                src_ip=sender_ip if sender_ip != "0.0.0.0" else sender_mac,
                dst_ip=target_ip if target_ip != "0.0.0.0" else target_mac,
                src_port="0",
                dst_port="0",
                protocol="2054",
                timestamp=timestamp,
                payload_length=max(len(frame) - offset, 0),
            )
        return None
    if ethertype != 0x0800:
        return None
    return parse_ipv4_packet(frame, timestamp, offset)


def parse_ipv4_packet(
    frame: bytes,
    timestamp: float,
    offset: int = 0,
) -> Optional[PacketRecord]:
    if len(frame) < offset + 20:
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
    else:
        return None

    return PacketRecord(
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=str(src_port),
        dst_port=str(dst_port),
        protocol=str(protocol),
        timestamp=timestamp,
        payload_length=payload_length,
    )


def load_label_rules(config: dict) -> List[LabelRule]:
    label_path = config["labels"].get("path", "")
    if not label_path:
        return []
    path = Path(label_path)
    delimiter = config["labels"].get("delimiter", ",")
    rules: List[LabelRule] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        lookup = build_column_lookup(reader.fieldnames or [])
        for row in reader:
            label = pick_column(row, lookup, FLOW_COLUMNS["label"])
            if not label:
                continue
            start_raw = pick_column(row, lookup, FLOW_COLUMNS["start_time"])
            end_raw = pick_column(row, lookup, FLOW_COLUMNS["end_time"])
            rules.append(
                LabelRule(
                    src_ip=pick_column(row, lookup, FLOW_COLUMNS["src_ip"]),
                    dst_ip=pick_column(row, lookup, FLOW_COLUMNS["dst_ip"]),
                    src_port=pick_column(row, lookup, FLOW_COLUMNS["src_port"]),
                    dst_port=pick_column(row, lookup, FLOW_COLUMNS["dst_port"]),
                    protocol=pick_column(row, lookup, FLOW_COLUMNS["protocol"]),
                    start_time=parse_float(start_raw) if start_raw else None,
                    end_time=parse_float(end_raw) if end_raw else None,
                    label=normalize_label(label, config),
                )
            )
    return rules


def load_packet_labels(config: dict) -> List[str]:
    label_path = config["labels"].get("packet_path", "")
    if not label_path:
        return []
    delimiter = config["labels"].get("packet_delimiter", config["labels"].get("delimiter", ","))
    labels: List[str] = []
    with Path(label_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row_index, row in enumerate(reader):
            if not row:
                continue
            if row_index == 0 and row[-1].strip().lower() in {"label", "x"}:
                continue
            labels.append(normalize_label(row[-1], config))
    return labels


def label_for_flow(flow: FlowState, labels: Sequence[LabelRule], default_label: str) -> str:
    if flow.label_votes:
        return flow.label_votes.most_common(1)[0][0]

    votes: Counter = Counter()
    for rule in labels:
        if rule_matches_flow(rule, flow):
            votes[rule.label] += 1
    if votes:
        return votes.most_common(1)[0][0]
    return default_label


def rule_matches_flow(rule: LabelRule, flow: FlowState) -> bool:
    if not rule.src_ip or not rule.dst_ip:
        return False
    rule_pair = canonical_ip_pair(rule.src_ip, rule.dst_ip)[2]
    if rule_pair != flow.key:
        return False
    if rule.protocol and not any(item[4] == str(rule.protocol) for item in flow.five_tuples):
        return False
    if rule.src_port and rule.dst_port:
        forward = (rule.src_ip, rule.dst_ip, str(rule.src_port), str(rule.dst_port), str(rule.protocol))
        reverse = (rule.dst_ip, rule.src_ip, str(rule.dst_port), str(rule.src_port), str(rule.protocol))
        if flow.five_tuples and forward not in flow.five_tuples and reverse not in flow.five_tuples:
            return False
    if rule.start_time is not None and flow.end < rule.start_time:
        return False
    if rule.end_time is not None and flow.start > rule.end_time:
        return False
    return True


def packet_records_to_flows(packets: Iterable[PacketRecord], config: dict) -> List[FlowState]:
    bin_width = int(config["quantization"]["packet_size_bin_width"])
    ipt_bin_width = int(config["quantization"]["ipt_bin_width_seconds"])
    time_gap = float(config["flow"]["time_gap_seconds"])

    prepared: List[PacketRecord] = []
    for packet in packets:
        if not packet.src_ip or not packet.dst_ip:
            continue
        size = round_to_nearest(packet.payload_length, bin_width)
        timestamp = packet.timestamp
        if ipt_bin_width > 0:
            timestamp = float(round_to_nearest(int(timestamp), ipt_bin_width))
        prepared.append(
            PacketRecord(
                packet.src_ip,
                packet.dst_ip,
                packet.src_port,
                packet.dst_port,
                packet.protocol,
                timestamp,
                size,
                packet.label,
            )
        )

    prepared.sort(key=lambda item: (canonical_ip_pair(item.src_ip, item.dst_ip)[2], item.timestamp))
    flows: List[FlowState] = []
    current: Optional[FlowState] = None
    for packet in prepared:
        key = canonical_ip_pair(packet.src_ip, packet.dst_ip)[2]
        if current is None or current.key != key or (packet.timestamp - current.end) > time_gap:
            current = FlowState.from_packet(packet)
            flows.append(current)
        else:
            current.add_packet(packet)
    return flows


def combine_superflows(flows: Sequence[FlowState], config: dict) -> List[FlowState]:
    flow_gap = float(config["flow"]["flow_gap_seconds"])
    ordered = sorted(flows, key=lambda item: (item.key, item.start))
    combined: List[FlowState] = []
    current: Optional[FlowState] = None
    for flow in ordered:
        if current is None or current.key != flow.key or (flow.start - current.end) > flow_gap:
            current = flow
            combined.append(current)
        else:
            current.add_flow(flow)
    return combined


def training_rows_from_packets(packets: Iterable[PacketRecord], labels: Sequence[LabelRule], config: dict) -> List[TrainingRow]:
    default_label = config["labels"]["default_label"]
    flows = packet_records_to_flows(packets, config)
    superflows = combine_superflows(flows, config)
    rows: List[TrainingRow] = []
    for flow in superflows:
        if flow.packets == 0 or flow.bytes == 0:
            continue
        rows.append(
            TrainingRow(
                packets=flow.packets,
                bytes=flow.bytes,
                median_ipt=flow.median_ipt,
                duration=flow.duration,
                label=label_for_flow(flow, labels, default_label),
            )
        )
    return rows


def write_training_rows(rows: Sequence[TrainingRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["NumberOfPackets", "TotalBytesTransmitted", "MedianIPT", "ConversationDuration", "class"])
        for row in rows:
            writer.writerow([row.packets, row.bytes, f"{row.median_ipt:.6f}", f"{row.duration:.6f}", row.label])


def write_intermediate(rows: Sequence[TrainingRow], config: dict) -> None:
    if not config["output"].get("write_intermediate", True):
        return
    path = Path(config["output"]["intermediate_dir"]) / "training_rows.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.__dict__, sort_keys=True) + "\n")


def build_dataset(config: dict) -> List[TrainingRow]:
    labels = load_label_rules(config)
    packet_labels = load_packet_labels(config)
    rows: List[TrainingRow] = []
    input_format = config["input"]["format"]
    input_paths = expand_input_paths(config["input"]["paths"], input_format)
    for path in input_paths:
        current_format = detect_input_format(path, input_format)
        if current_format == "pcap":
            rows.extend(training_rows_from_packets(iter_pcap(path, config, packet_labels), labels, config))
        elif current_format == "packet_csv":
            rows.extend(training_rows_from_packets(iter_packet_csv(path, config, packet_labels), labels, config))
        elif current_format == "flow_csv":
            rows.extend(iter_flow_csv(path, config, labels))
        else:
            raise ValueError(f"unsupported FlowLens input format: {current_format}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FlowLens training CSV from unified benchmark inputs.")
    parser.add_argument("--config", required=True, help="Path to a FlowLens unified JSON config.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    rows = build_dataset(config)
    output_path = Path(config["output"]["dataset_csv"])
    write_training_rows(rows, output_path)
    write_intermediate(rows, config)
    label_counts = Counter(row.label for row in rows)
    print(f"Wrote {len(rows)} FlowLens training rows to {output_path}")
    print("Label counts: " + json.dumps(dict(sorted(label_counts.items())), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
