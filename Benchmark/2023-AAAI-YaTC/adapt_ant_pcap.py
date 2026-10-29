#!/usr/bin/env python3
"""Convert ANT pcaps to YaTC ImageFolder MFR images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ant_deep_adapter_common import (  # noqa: E402
    discover_ant_samples,
    keep_common_labels,
    limit_per_split_label,
    label_mapping,
    protocol_set,
    sample_stem,
)
from ant_pcap_native import comma_list, group_flows, read_pcap_packets, safe_name, write_manifest  # noqa: E402


HEADER_HEX_LEN = 160
PAYLOAD_HEX_LEN = 480
PACKETS_PER_FLOW = 5
FLOW_BYTES = 1600


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ant-root", type=Path, default=ROOT / "data" / "ANT")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--label-field", default="Scenario")
    parser.add_argument("--source-mode", choices=["episode", "segment"], default="episode")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--engines", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-per-split-label", type=int, default=None)
    parser.add_argument("--max-flows-per-sample", type=int, default=None)
    parser.add_argument("--flow-timeout", type=float, default=120.0)
    parser.add_argument("--protocols", default="tcp,udp")
    parser.add_argument("--scope", choices=["flow", "episode"], default="flow")
    parser.add_argument("--zero-ip-addresses", action="store_true")
    return parser.parse_args()


def ethernet_ip_bytes(raw: bytes) -> bytes | None:
    if len(raw) < 14:
        return None
    eth_type = int.from_bytes(raw[12:14], "big")
    offset = 14
    while eth_type in {0x8100, 0x88A8} and len(raw) >= offset + 4:
        eth_type = int.from_bytes(raw[offset + 2 : offset + 4], "big")
        offset += 4
    if eth_type == 0x0800 and len(raw) >= offset + 20:
        total_len = int.from_bytes(raw[offset + 2 : offset + 4], "big")
        return raw[offset : min(len(raw), offset + total_len)] if total_len else raw[offset:]
    if eth_type == 0x86DD and len(raw) >= offset + 40:
        payload_len = int.from_bytes(raw[offset + 4 : offset + 6], "big")
        return raw[offset : min(len(raw), offset + 40 + payload_len)]
    return None


def split_header_payload(ip_bytes: bytes) -> tuple[bytes, bytes]:
    if not ip_bytes:
        return b"", b""
    ver = (ip_bytes[0] >> 4) & 0xF
    if ver == 4 and len(ip_bytes) >= 20:
        ihl = (ip_bytes[0] & 0x0F) * 4
        proto = ip_bytes[9]
        header_len = ihl
        if proto == 6 and len(ip_bytes) >= ihl + 20:
            header_len = ihl + max(20, ((ip_bytes[ihl + 12] >> 4) & 0x0F) * 4)
        elif proto == 17 and len(ip_bytes) >= ihl + 8:
            header_len = ihl + 8
        return ip_bytes[:header_len], ip_bytes[header_len:]
    if ver == 6 and len(ip_bytes) >= 40:
        nxt = ip_bytes[6]
        header_len = 40
        if nxt == 6 and len(ip_bytes) >= 60:
            header_len = 40 + max(20, ((ip_bytes[52] >> 4) & 0x0F) * 4)
        elif nxt == 17 and len(ip_bytes) >= 48:
            header_len = 48
        return ip_bytes[:header_len], ip_bytes[header_len:]
    return ip_bytes, b""


def maybe_zero_ip_addresses(ip_bytes: bytes) -> bytes:
    data = bytearray(ip_bytes)
    if not data:
        return ip_bytes
    ver = (data[0] >> 4) & 0xF
    if ver == 4 and len(data) >= 20:
        data[12:20] = b"\x00" * 8
    elif ver == 6 and len(data) >= 40:
        data[8:40] = b"\x00" * 32
    return bytes(data)


def packet_chunk(raw: bytes, zero_ip_addresses: bool) -> str:
    ip_bytes = ethernet_ip_bytes(raw)
    if ip_bytes is None:
        return "0" * (HEADER_HEX_LEN + PAYLOAD_HEX_LEN)
    if zero_ip_addresses:
        ip_bytes = maybe_zero_ip_addresses(ip_bytes)
    header, payload = split_header_payload(ip_bytes)
    h = header.hex()[:HEADER_HEX_LEN].ljust(HEADER_HEX_LEN, "0")
    p = payload.hex()[:PAYLOAD_HEX_LEN].ljust(PAYLOAD_HEX_LEN, "0")
    return h + p


def flow_image_array(raw_packets: list[bytes], zero_ip_addresses: bool) -> np.ndarray:
    chunks = [packet_chunk(raw, zero_ip_addresses) for raw in raw_packets[:PACKETS_PER_FLOW]]
    chunks.extend(["0" * (HEADER_HEX_LEN + PAYLOAD_HEX_LEN)] * max(0, PACKETS_PER_FLOW - len(chunks)))
    flow_hex = "".join(chunks)
    return np.fromiter((int(flow_hex[i : i + 2], 16) for i in range(0, FLOW_BYTES * 2, 2)), dtype=np.uint8)


def main() -> int:
    args = parse_args()
    splits = comma_list(args.splits, ["train", "val", "test"])
    engines = comma_list(args.engines, [])
    samples = limit_per_split_label(keep_common_labels(
        discover_ant_samples(args.ant_root, splits, args.label_field, args.source_mode, engines, args.max_samples),
        splits,
    ), args.max_per_split_label)
    label2id = label_mapping(samples)
    protocols = protocol_set(args.protocols)
    image_root = args.out_dir / "images"
    row_meta = []
    stats = {split: {"images": 0, "skipped": 0} for split in splits}

    for sample in samples:
        try:
            packets = read_pcap_packets(sample.pcap_path, protocols=protocols)
        except Exception:
            stats[sample.split]["skipped"] += 1
            continue
        flow_packet_groups = [packets] if args.scope == "episode" else [flow.packets for flow in group_flows(packets, timeout=args.flow_timeout)]
        if args.max_flows_per_sample is not None and args.max_flows_per_sample > 0:
            flow_packet_groups = flow_packet_groups[: args.max_flows_per_sample]
        for flow_index, flow_packets in enumerate(flow_packet_groups):
            if not flow_packets:
                stats[sample.split]["skipped"] += 1
                continue
            arr = flow_image_array([pkt.raw for pkt in flow_packets], args.zero_ip_addresses)
            out_class = image_root / ("valid" if sample.split == "val" else sample.split) / safe_name(sample.label)
            out_class.mkdir(parents=True, exist_ok=True)
            image_name = f"{sample_stem(sample)}_flow{flow_index:05d}.png"
            Image.fromarray(arr.reshape(40, 40), mode="L").save(out_class / image_name)
            row_meta.append(
                {
                    "split": sample.split,
                    "image": str((out_class / image_name).resolve()),
                    "label": sample.label,
                    "label_id": label2id[sample.label],
                    "source_pcap": str(sample.pcap_path.resolve()),
                    "source_kind": sample.source_kind,
                    "sample_id": sample.sample_id,
                    "engine": sample.engine,
                    "scenario": sample.scenario,
                    "flow_index": flow_index,
                }
            )
            stats[sample.split]["images"] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "row_metadata.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in row_meta) + ("\n" if row_meta else ""), encoding="utf-8")
    write_manifest(
        args.out_dir / "manifest.json",
        {
            "adapter": "ANT pcap -> YaTC MFR ImageFolder",
            "label_field": args.label_field,
            "source_mode": args.source_mode,
            "label2id": label2id,
            "scope": args.scope,
            "flow_definition": "bidirectional five-tuple with idle timeout when scope=flow",
            "flow_timeout": args.flow_timeout,
            "protocols": args.protocols,
            "image_shape": [40, 40],
            "packets_per_flow": PACKETS_PER_FLOW,
            "per_packet_bytes": "80 bytes of IP/transport header region + 240 bytes of payload, zero-padded/truncated",
            "zero_ip_addresses": bool(args.zero_ip_addresses),
            "outputs": {"image_root": str(image_root.resolve()), "row_metadata": str((args.out_dir / "row_metadata.jsonl").resolve())},
            "stats": stats,
        },
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
