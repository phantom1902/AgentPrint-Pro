#!/usr/bin/env python3
"""Native ANT pcap helpers shared by baseline adapters.

This module keeps ANT-specific label extraction and the required bidirectional
five-tuple + idle-timeout flow definition in one place. Baseline adapters should
use the original repository's feature extraction after this step whenever the
original code provides one.
"""

from __future__ import annotations

import csv
import json
import socket
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_FLOW_TIMEOUT = 120.0
DEFAULT_RISK_BY_SCENARIO = {
    "attack_code_generation": "malicious",
    "attack_execution": "malicious",
}


@dataclass(frozen=True)
class AntLabels:
    Engine: str
    Harness: str
    Model: str
    Scenario: str
    Task: str
    Risk: str
    Tool: str = ""
    Intent: str = ""

    def value(self, name: str) -> str:
        if not hasattr(self, name):
            raise ValueError(f"unknown ANT label field: {name}")
        return str(getattr(self, name))


@dataclass(frozen=True)
class AntEpisode:
    split: str
    engine: str
    scenario: str
    episode_id: str
    episode_dir: Path
    pcap_path: Path
    labels: AntLabels


@dataclass(frozen=True)
class PacketRecord:
    frame_no: int
    timestamp: float
    raw: bytes
    frame_len: int
    src_ip: str
    dst_ip: str
    proto: int
    src_port: int | None
    dst_port: int | None
    tcp_flags: int = 0
    payload_len: int = 0


@dataclass
class FlowRecord:
    flow_id: str
    key: tuple[str, int, int | None, str, int | None]
    packets: list[PacketRecord] = field(default_factory=list)
    hostname: str = ""

    @property
    def start_time(self) -> float:
        return self.packets[0].timestamp

    @property
    def end_time(self) -> float:
        return self.packets[-1].timestamp

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def proto(self) -> int:
        return self.key[1]

    @property
    def src_ip(self) -> str:
        return self.key[0]

    @property
    def src_port(self) -> int | None:
        return self.key[2]

    @property
    def dst_ip(self) -> str:
        return self.key[3]

    @property
    def dst_port(self) -> int | None:
        return self.key[4]


def comma_list(value: str | None, default: Iterable[str]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)) or "unknown"


def parse_risk_map(value: str | None) -> dict[str, str]:
    mapping = dict(DEFAULT_RISK_BY_SCENARIO)
    if not value:
        return mapping
    path = Path(value)
    if path.exists():
        mapping.update(json.loads(path.read_text(encoding="utf-8")))
    else:
        mapping.update(json.loads(value))
    return mapping


def infer_harness(engine: str) -> str:
    if engine in {"codex"}:
        return "Codex"
    if engine in {"qwen"}:
        return "QWEN"
    if engine.startswith("cc_") or engine.startswith("USA_cc_"):
        return "Claude code"
    return engine


def infer_model(engine: str) -> str:
    lower = engine.lower()
    if "dpsk" in lower:
        return "deepseek"
    if "mimo" in lower:
        return "mimo"
    if "xiaoai" in lower:
        return "sonnet"
    if lower == "codex":
        return "codex"
    if lower == "qwen":
        return "qwen"
    return engine


def labels_for_path(split: str, engine: str, scenario: str, episode_id: str, risk_map: dict[str, str]) -> AntLabels:
    return AntLabels(
        Engine=engine,
        Harness=infer_harness(engine),
        Model=infer_model(engine),
        Scenario=scenario,
        Task=episode_id,
        Risk=risk_map.get(scenario, "benign"),
        Tool="",
        Intent="",
    )


def discover_episodes(
    ant_root: Path,
    splits: Iterable[str],
    engines: Iterable[str] | None = None,
    max_episodes: int | None = None,
    risk_map: dict[str, str] | None = None,
) -> list[AntEpisode]:
    split_set = set(splits)
    engine_set = set(engines or [])
    risk_map = risk_map or dict(DEFAULT_RISK_BY_SCENARIO)
    episodes: list[AntEpisode] = []
    for pcap_path in sorted(ant_root.glob("*/*/*/A*/network.pcap")):
        rel = pcap_path.relative_to(ant_root)
        if len(rel.parts) != 5:
            continue
        split, engine, scenario, episode_id, _ = rel.parts
        if split not in split_set:
            continue
        if engine_set and engine not in engine_set:
            continue
        labels = labels_for_path(split, engine, scenario, episode_id, risk_map)
        episodes.append(AntEpisode(split, engine, scenario, episode_id, pcap_path.parent, pcap_path, labels))
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    return episodes


def label_mapping(episodes: Iterable[AntEpisode], label_field: str) -> dict[str, int]:
    values = sorted({ep.labels.value(label_field) for ep in episodes})
    return {value: index for index, value in enumerate(values)}


def _ip_text(raw: bytes) -> str:
    if len(raw) == 4:
        return socket.inet_ntop(socket.AF_INET, raw)
    return socket.inet_ntop(socket.AF_INET6, raw)


def read_pcap_packets(pcap_path: Path, protocols: set[int] | None = None, include_zero_payload: bool = True, max_records: int | None = None) -> list[PacketRecord]:
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    try:
        import dpkt
    except ModuleNotFoundError:
        return read_pcap_packets_stdlib(pcap_path, protocols=protocols, include_zero_payload=include_zero_payload, max_records=max_records)

    records: list[PacketRecord] = []
    with pcap_path.open("rb") as handle:
        try:
            reader = dpkt.pcap.Reader(handle)
        except (ValueError, dpkt.dpkt.NeedData):
            handle.seek(0)
            reader = dpkt.pcapng.Reader(handle)
        for frame_no, (ts, raw) in enumerate(reader, start=1):
            try:
                eth = dpkt.ethernet.Ethernet(raw)
            except (dpkt.dpkt.UnpackError, ValueError):
                continue
            network = eth.data
            if isinstance(network, dpkt.ip.IP):
                src_ip = _ip_text(network.src)
                dst_ip = _ip_text(network.dst)
                proto = int(network.p)
                network_payload_len = max(int(network.len) - int(network.hl) * 4, 0)
                transport = network.data
            elif isinstance(network, dpkt.ip6.IP6):
                src_ip = _ip_text(network.src)
                dst_ip = _ip_text(network.dst)
                proto = int(network.nxt)
                network_payload_len = len(bytes(network.data))
                transport = network.data
                while proto in {0, 43, 44, 51, 60} and hasattr(transport, "nxt"):
                    proto = int(transport.nxt)
                    transport = transport.data
            elif isinstance(network, dpkt.arp.ARP):
                src_ip = _ip_text(network.spa)
                dst_ip = _ip_text(network.tpa)
                proto = 0x0806
                network_payload_len = len(bytes(network))
                transport = None
            else:
                continue
            if protocols is not None and proto not in protocols:
                continue
            src_port = dst_port = None
            tcp_flags = 0
            payload_len = network_payload_len
            if proto == 6 and isinstance(transport, dpkt.tcp.TCP):
                tcp = transport
                src_port, dst_port = int(tcp.sport), int(tcp.dport)
                tcp_flags = int(tcp.flags)
                payload_len = len(tcp.data)
            elif proto == 17 and isinstance(transport, dpkt.udp.UDP):
                udp = transport
                src_port, dst_port = int(udp.sport), int(udp.dport)
                payload_len = len(udp.data)
            elif proto in {6, 17}:
                continue
            if not include_zero_payload and payload_len <= 0:
                continue
            records.append(
                PacketRecord(
                    frame_no=frame_no,
                    timestamp=float(ts),
                    raw=bytes(raw),
                    frame_len=len(raw),
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    proto=proto,
                    src_port=src_port,
                    dst_port=dst_port,
                    tcp_flags=tcp_flags,
                    payload_len=payload_len,
                )
            )
            if max_records is not None and len(records) >= max_records:
                break
    return records


def _pcap_endian_and_resolution(header: bytes) -> tuple[str, float]:
    magic = header[:4]
    if magic == b"\xd4\xc3\xb2\xa1":
        return "<", 1_000_000.0
    if magic == b"\xa1\xb2\xc3\xd4":
        return ">", 1_000_000.0
    if magic == b"\x4d\x3c\xb2\xa1":
        return "<", 1_000_000_000.0
    if magic == b"\xa1\xb2\x3c\x4d":
        return ">", 1_000_000_000.0
    raise ValueError("unsupported pcap magic; install dpkt/tshark for pcapng or non-classic pcap")


def _parse_ethernet_ip(raw: bytes):
    if len(raw) < 14:
        return None
    eth_type = struct.unpack("!H", raw[12:14])[0]
    offset = 14
    while eth_type in {0x8100, 0x88A8} and len(raw) >= offset + 4:
        eth_type = struct.unpack("!H", raw[offset + 2 : offset + 4])[0]
        offset += 4
    if eth_type == 0x0800:
        return _parse_ipv4(raw, offset)
    if eth_type == 0x86DD:
        return _parse_ipv6(raw, offset)
    if eth_type == 0x0806:
        return _parse_arp(raw, offset)
    return None


def _parse_arp(raw: bytes, offset: int):
    if len(raw) < offset + 28:
        return None
    htype, ptype, hlen, plen = struct.unpack("!HHBB", raw[offset : offset + 6])
    if htype != 1 or ptype != 0x0800 or hlen != 6 or plen != 4:
        return None
    src_offset = offset + 8 + hlen
    dst_offset = src_offset + plen + hlen
    src_ip = socket.inet_ntop(socket.AF_INET, raw[src_offset : src_offset + plen])
    dst_ip = socket.inet_ntop(socket.AF_INET, raw[dst_offset : dst_offset + plen])
    return src_ip, dst_ip, 0x0806, raw[offset:]


def _parse_ipv4(raw: bytes, offset: int):
    if len(raw) < offset + 20:
        return None
    first = raw[offset]
    ihl = (first & 0x0F) * 4
    if ihl < 20 or len(raw) < offset + ihl:
        return None
    total_len = struct.unpack("!H", raw[offset + 2 : offset + 4])[0]
    proto = raw[offset + 9]
    src_ip = socket.inet_ntop(socket.AF_INET, raw[offset + 12 : offset + 16])
    dst_ip = socket.inet_ntop(socket.AF_INET, raw[offset + 16 : offset + 20])
    ip_end = min(len(raw), offset + total_len) if total_len else len(raw)
    transport = raw[offset + ihl : ip_end]
    return src_ip, dst_ip, proto, transport


def _parse_ipv6(raw: bytes, offset: int):
    if len(raw) < offset + 40:
        return None
    payload_len = struct.unpack("!H", raw[offset + 4 : offset + 6])[0]
    proto = raw[offset + 6]
    src_ip = socket.inet_ntop(socket.AF_INET6, raw[offset + 8 : offset + 24])
    dst_ip = socket.inet_ntop(socket.AF_INET6, raw[offset + 24 : offset + 40])
    ip_end = min(len(raw), offset + 40 + payload_len)
    transport = raw[offset + 40 : ip_end]
    while proto in {0, 43, 44, 51, 60} and transport:
        next_proto = transport[0]
        if proto in {0, 43, 60}:
            if len(transport) < 2:
                break
            header_len = (transport[1] + 1) * 8
        elif proto == 44:
            header_len = 8
        else:
            if len(transport) < 2:
                break
            header_len = (transport[1] + 2) * 4
        if header_len > len(transport):
            break
        proto = next_proto
        transport = transport[header_len:]
    return src_ip, dst_ip, proto, transport


def _transport_fields(proto: int, transport: bytes) -> tuple[int | None, int | None, int, int]:
    src_port = dst_port = None
    tcp_flags = 0
    payload_len = len(transport)
    if proto == 6:
        if len(transport) < 20:
            return None, None, 0, 0
        src_port, dst_port = struct.unpack("!HH", transport[:4])
        data_offset = ((transport[12] >> 4) & 0x0F) * 4
        tcp_flags = transport[13]
        payload_len = max(0, len(transport) - data_offset)
    elif proto == 17:
        if len(transport) < 8:
            return None, None, 0, 0
        src_port, dst_port = struct.unpack("!HH", transport[:4])
        udp_len = struct.unpack("!H", transport[4:6])[0]
        payload_len = max(0, min(len(transport), udp_len) - 8)
    return src_port, dst_port, tcp_flags, payload_len


def read_pcap_packets_stdlib(pcap_path: Path, protocols: set[int] | None = None, include_zero_payload: bool = True, max_records: int | None = None) -> list[PacketRecord]:
    if max_records is not None and max_records < 1:
        raise ValueError("max_records must be positive")
    records: list[PacketRecord] = []
    with pcap_path.open("rb") as handle:
        header = handle.read(24)
        if len(header) != 24:
            return records
        endian, resolution = _pcap_endian_and_resolution(header)
        network = struct.unpack(endian + "I", header[20:24])[0]
        if network != 1:
            raise ValueError(f"{pcap_path} linktype {network} is unsupported without dpkt; expected Ethernet")
        frame_no = 0
        while True:
            pkt_header = handle.read(16)
            if not pkt_header:
                break
            if len(pkt_header) != 16:
                break
            ts_sec, ts_frac, incl_len, orig_len = struct.unpack(endian + "IIII", pkt_header)
            raw = handle.read(incl_len)
            if len(raw) != incl_len:
                break
            frame_no += 1
            parsed = _parse_ethernet_ip(raw)
            if parsed is None:
                continue
            src_ip, dst_ip, proto, transport = parsed
            if protocols is not None and proto not in protocols:
                continue
            src_port, dst_port, tcp_flags, payload_len = _transport_fields(proto, transport)
            if proto in {6, 17} and src_port is None:
                continue
            if not include_zero_payload and payload_len <= 0:
                continue
            records.append(
                PacketRecord(
                    frame_no=frame_no,
                    timestamp=float(ts_sec) + float(ts_frac) / resolution,
                    raw=raw,
                    frame_len=int(orig_len or incl_len),
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    proto=proto,
                    src_port=src_port,
                    dst_port=dst_port,
                    tcp_flags=tcp_flags,
                    payload_len=payload_len,
                )
            )
            if max_records is not None and len(records) >= max_records:
                break
    return records


def canonical_key(packet: PacketRecord) -> tuple[str, int, int | None, str, int | None]:
    left = (packet.src_ip, packet.src_port)
    right = (packet.dst_ip, packet.dst_port)
    if left <= right:
        return (packet.src_ip, packet.proto, packet.src_port, packet.dst_ip, packet.dst_port)
    return (packet.dst_ip, packet.proto, packet.dst_port, packet.src_ip, packet.src_port)


def packet_matches_flow_direction(packet: PacketRecord, flow: FlowRecord) -> int:
    return 1 if packet.src_ip == flow.src_ip and packet.src_port == flow.src_port else -1


def group_flows(
    packets: Iterable[PacketRecord],
    timeout: float = DEFAULT_FLOW_TIMEOUT,
    dns_mapping: dict[str, str] | None = None,
) -> list[FlowRecord]:
    dns_mapping = dns_mapping or {}
    flows: list[FlowRecord] = []
    active: dict[tuple[str, int, int | None, str, int | None], int] = {}
    counters: dict[tuple[str, int, int | None, str, int | None], int] = {}
    for packet in sorted(packets, key=lambda pkt: (pkt.timestamp, pkt.frame_no)):
        key = canonical_key(packet)
        idx = active.get(key)
        if idx is None or packet.timestamp - flows[idx].packets[-1].timestamp > timeout:
            counters[key] = counters.get(key, 0) + 1
            flow_id = f"flow_{len(flows):08d}_part{counters[key]:02d}"
            peer_a = key[0]
            peer_b = key[3]
            hostname = dns_mapping.get(peer_a) or dns_mapping.get(peer_b) or ""
            flows.append(FlowRecord(flow_id=flow_id, key=key, hostname=hostname))
            idx = len(flows) - 1
            active[key] = idx
        flows[idx].packets.append(packet)
    return flows


def _dns_name(payload: bytes, offset: int, depth: int = 0) -> tuple[str, int]:
    labels = []
    start = offset
    jumped = False
    while offset < len(payload):
        length = payload[offset]
        if length == 0:
            offset += 1
            break
        if length & 0xC0 == 0xC0:
            if offset + 1 >= len(payload) or depth > 8:
                return "", offset + 2
            ptr = ((length & 0x3F) << 8) | payload[offset + 1]
            pointed, _ = _dns_name(payload, ptr, depth + 1)
            labels.append(pointed)
            offset += 2
            jumped = True
            break
        offset += 1
        labels.append(payload[offset : offset + length].decode("utf-8", errors="ignore"))
        offset += length
    return ".".join(part for part in labels if part), (offset if not jumped else offset)


def _udp_payload(packet: PacketRecord) -> bytes:
    parsed = _parse_ethernet_ip(packet.raw)
    if parsed is None:
        return b""
    _src, _dst, proto, transport = parsed
    if proto != 17 or len(transport) < 8:
        return b""
    udp_len = struct.unpack("!H", transport[4:6])[0]
    return transport[8:min(len(transport), udp_len)]


def parse_dns_response(packet: PacketRecord) -> dict | None:
    payload = _udp_payload(packet)
    if len(payload) < 12:
        return None
    flags = struct.unpack("!H", payload[2:4])[0]
    if not flags & 0x8000:
        return None
    qd_count, an_count = struct.unpack("!HH", payload[4:8])
    offset = 12
    query = ""
    for _ in range(qd_count):
        query, offset = _dns_name(payload, offset)
        offset += 4
        if offset > len(payload):
            return None
    a_records: list[str] = []
    aaaa_records: list[str] = []
    cnames: list[str] = []
    resp_types: list[str] = []
    for _ in range(an_count):
        _name, offset = _dns_name(payload, offset)
        if offset + 10 > len(payload):
            break
        rtype, _rclass, _ttl, rdlen = struct.unpack("!HHIH", payload[offset : offset + 10])
        offset += 10
        rdata = payload[offset : offset + rdlen]
        offset += rdlen
        resp_types.append(str(rtype))
        if rtype == 1 and len(rdata) == 4:
            a_records.append(socket.inet_ntop(socket.AF_INET, rdata))
        elif rtype == 28 and len(rdata) == 16:
            aaaa_records.append(socket.inet_ntop(socket.AF_INET6, rdata))
        elif rtype == 5:
            cname, _ = _dns_name(payload, offset - rdlen)
            cnames.append(cname)
    if not (a_records or aaaa_records or cnames):
        return None
    return {
        "frame.time_epoch": f"{packet.timestamp:.9f}",
        "dns.qry.name": query,
        "dns.a": ",".join(a_records),
        "dns.aaaa": ",".join(aaaa_records),
        "dns.cname": ",".join(cnames),
        "dns.resp.type": ",".join(resp_types),
        "dns.count.answers": an_count,
    }


def dns_ip_to_hostname(packets: Iterable[PacketRecord]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for packet in packets:
        if packet.proto != 17 or packet.dst_port != 53 and packet.src_port != 53:
            continue
        row = parse_dns_response(packet)
        if not row:
            continue
        name = row["dns.qry.name"]
        for ip in str(row["dns.a"]).split(","):
            if ip:
                mapping[ip] = name
        for ip in str(row["dns.aaaa"]).split(","):
            if ip:
                mapping[ip] = name
    return mapping


def write_flow_pcaps(flows: Iterable[FlowRecord], output_dir: Path, class_name: str) -> list[Path]:
    written: list[Path] = []
    class_dir = output_dir / safe_name(class_name)
    class_dir.mkdir(parents=True, exist_ok=True)
    for flow in flows:
        if not flow.packets:
            continue
        path = class_dir / f"{safe_name(flow.flow_id)}.pcap"
        with path.open("wb") as handle:
            handle.write(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
            for packet in flow.packets:
                ts_sec = int(packet.timestamp)
                ts_usec = int(round((packet.timestamp - ts_sec) * 1_000_000))
                handle.write(struct.pack("<IIII", ts_sec, ts_usec, len(packet.raw), int(packet.frame_len or len(packet.raw))))
                handle.write(packet.raw)
        written.append(path)
    return written


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def common_arg_parser(description: str):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--ant-root", default="Benchmark/data/ANT")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--engines", default="")
    parser.add_argument("--label-field", default="Scenario", choices=["Engine", "Harness", "Model", "Scenario", "Task", "Risk", "Tool", "Intent"])
    parser.add_argument("--risk-map", default=None, help="JSON string or path mapping Scenario -> Risk")
    parser.add_argument("--flow-timeout", type=float, default=DEFAULT_FLOW_TIMEOUT)
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser
