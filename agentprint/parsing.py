"""Streaming classic-PCAP framing and dpkt IP decoding.

T01 deliberately excludes unsupported episodes rather than silently interpreting
IPv6, other transports, fragmented IP, or damaged/truncated traffic as clean TCP.
The record and decoded-packet seam lets T03 broaden packet quality handling.
"""
from collections import Counter
from dataclasses import dataclass
import ipaddress
import socket
import struct

import dpkt


ENDPOINT_RULE_VERSION = 'rfc1918-syn-participation-v1'
PRIVATE_NETS = tuple(ipaddress.ip_network(s) for s in
                     ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16'))


class EpisodeExcluded(ValueError):
    def __init__(self, reason, detail, **context):
        super().__init__(detail)
        self.reason, self.detail, self.context = reason, detail, context


@dataclass(frozen=True)
class Packet:
    time_ns: int
    src: str
    dst: str
    ip_length: int
    src_mac: str
    tcp_flags: int


@dataclass
class EpisodeScan:
    start_ns: int
    end_ns: int
    runtime_endpoints: list
    endpoint_source: dict
    quality: dict


def iter_records(path):
    """Yield exact integer-nanosecond timestamps and complete Ethernet frames."""
    with open(path, 'rb') as handle:
        header = handle.read(24)
        if len(header) != 24:
            raise EpisodeExcluded('damaged_capture', 'incomplete PCAP global header')
        formats = {b'\xd4\xc3\xb2\xa1': ('<', 1000), b'\xa1\xb2\xc3\xd4': ('>', 1000),
                   b'\x4d\x3c\xb2\xa1': ('<', 1), b'\xa1\xb2\x3c\x4d': ('>', 1)}
        if header[:4] not in formats:
            raise EpisodeExcluded('unsupported_capture_format', 'only classic PCAP is supported')
        endian, scale = formats[header[:4]]
        major, minor, _, _, snaplen, linktype = struct.unpack(endian + 'HHIIII', header[4:])
        if (major, minor) != (2, 4):
            raise EpisodeExcluded('unsupported_capture_format', 'PCAP version must be 2.4')
        if linktype != 1:
            raise EpisodeExcluded('unsupported_linktype', f'Ethernet required; linktype={linktype}')
        while True:
            record = handle.read(16)
            if not record:
                break
            if len(record) != 16:
                raise EpisodeExcluded('damaged_capture', 'incomplete PCAP record header')
            seconds, fraction, captured, wire = struct.unpack(endian + 'IIII', record)
            if fraction >= 1000000000 // scale or captured > wire or captured > snaplen:
                raise EpisodeExcluded('damaged_capture', 'invalid PCAP record lengths or timestamp')
            raw = handle.read(captured)
            if len(raw) != captured:
                raise EpisodeExcluded('damaged_capture', 'incomplete PCAP record payload')
            if captured < wire:
                raise EpisodeExcluded('unsupported_truncated_packet', 'T01 requires complete captured frames')
            yield seconds * 1000000000 + fraction * scale, raw


def decode_packet(time_ns, raw):
    try:
        ethernet = dpkt.ethernet.Ethernet(raw)
    except (dpkt.UnpackError, ValueError) as exc:
        raise EpisodeExcluded('unsupported_malformed_packet', str(exc)) from exc
    if ethernet.type == dpkt.ethernet.ETH_TYPE_IP6:
        raise EpisodeExcluded('unsupported_ipv6', 'IPv6 support belongs to T03')
    if ethernet.type != dpkt.ethernet.ETH_TYPE_IP:
        if ethernet.type in (0x8100, 0x88a8):
            raise EpisodeExcluded('unsupported_link_encapsulation', 'T01 requires untagged Ethernet')
        return None
    ip = ethernet.data
    if not isinstance(ip, dpkt.ip.IP) or ip.v != 4 or ip.hl < 5 or ip.len < ip.hl * 4:
        raise EpisodeExcluded('unsupported_malformed_packet', 'unreadable IPv4 header')
    if ip.len > len(raw) - 14:
        raise EpisodeExcluded('unsupported_truncated_packet', 'incomplete IP packet')
    if ip.offset or ip.mf:
        raise EpisodeExcluded('unsupported_fragment', 'T01 requires unfragmented IPv4')
    if ip.p != dpkt.ip.IP_PROTO_TCP:
        raise EpisodeExcluded('unsupported_transport', 'T01 clean path supports TCP only')
    if not isinstance(ip.data, dpkt.tcp.TCP) or ip.data.off < 5:
        raise EpisodeExcluded('unsupported_malformed_packet', 'unreadable TCP header')
    return Packet(time_ns, socket.inet_ntoa(ip.src), socket.inet_ntoa(ip.dst), ip.len,
                  ':'.join(f'{x:02x}' for x in ethernet.src), ip.data.flags)


def scan_episode(path):
    participation: Counter[str] = Counter()
    candidates: set[str] = set()
    macs: dict[str, set[str]] = {}
    start = end = None
    quality = {'records': 0, 'ip_packets': 0, 'non_ip_packets': 0,
               'direction_skipped': 0, 'truncated_packets': 0}
    for time_ns, raw in iter_records(path):
        start = time_ns if start is None else min(start, time_ns)
        end = time_ns if end is None else max(end, time_ns)
        quality['records'] += 1
        packet = decode_packet(time_ns, raw)
        if packet is None:
            quality['non_ip_packets'] += 1
            continue
        quality['ip_packets'] += 1
        participation.update({packet.src, packet.dst})
        macs.setdefault(packet.src, set()).add(packet.src_mac)
        if packet.tcp_flags & dpkt.tcp.TH_SYN and not packet.tcp_flags & dpkt.tcp.TH_ACK:
            address = ipaddress.ip_address(packet.src)
            if any(address in net for net in PRIVATE_NETS):
                candidates.add(packet.src)
    times = {'start_ns': start, 'end_ns': end,
             'duration_ns': None if start is None else end - start, 'quality': quality}
    if start is None:
        raise EpisodeExcluded('empty_capture', 'capture contains no records', **times)
    assert end is not None  # start and end are set together for each record.
    if start == end:
        raise EpisodeExcluded('zero_capture_duration', 'capture interval has zero duration', **times)
    if len(candidates) != 1:
        raise EpisodeExcluded('endpoint_syn_candidates', f'expected one RFC1918 SYN source; got {sorted(candidates)}', **times)
    endpoint = next(iter(candidates))
    maximum = max(participation.values())
    leaders = [address for address, count in participation.items() if count == maximum]
    evidence = {'rule_version': ENDPOINT_RULE_VERSION, 'kind': 'inferred',
                'syn_candidates': sorted(candidates), 'selected_ipv4': endpoint,
                'participation': dict(sorted(participation.items())),
                'ipv4_source_macs': sorted(macs[endpoint])}
    if leaders != [endpoint]:
        raise EpisodeExcluded('endpoint_participation_conflict',
                              'SYN source must be the unique packet-participation leader',
                              endpoint_source=evidence, **times)
    return EpisodeScan(start, end, [endpoint], evidence, quality)


def direction(packet, endpoints):
    outgoing, incoming = packet.src in endpoints, packet.dst in endpoints
    if outgoing == incoming:
        return None
    return 0 if outgoing else 1
