"""Streaming classic-PCAP framing, readable IP lengths and inferred endpoints.

Decode the dpkt fixed IP headers independently of transport payload parsing so
capture truncation cannot silently remove packets with known declared lengths.
"""
from collections import Counter
from dataclasses import dataclass
import ipaddress
import socket
import struct

import dpkt


ENDPOINT_RULE_VERSION = 'rfc1918-syn-participation-ipv6-mac-v2'
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
    payload_truncated: bool = False


@dataclass
class EpisodeScan:
    start_ns: int
    end_ns: int
    runtime_endpoints: list
    endpoint_source: dict
    quality: dict


def iter_records(path):
    """Yield integer nanoseconds and captured Ethernet bytes, including truncation.

Unsupported formats/linktypes and broken record streams exclude the episode.
An intact record containing a truncated frame is safe to continue reading.
"""
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
            yield seconds * 1000000000 + fraction * scale, raw


def decode_packet(time_ns, raw):
    """Return readable IPv4/IPv6 of any protocol, else None; never require payload."""
    return _decode_frame(time_ns, raw)[0]


def _decode_frame(time_ns, raw):
    """Return a packet and a mutually exclusive quality-counter category."""
    if len(raw) < 14:
        return None, 'malformed_link_packets'
    kind = struct.unpack_from('!H', raw, 12)[0]
    offset = 14
    # Ethernet 802.1Q / 802.1ad tags carry their actual inner EtherType.
    while kind in (0x8100, 0x88a8):
        if len(raw) < offset + 4:
            return None, 'malformed_link_packets'
        kind = struct.unpack_from('!H', raw, offset + 2)[0]
        offset += 4
    data = raw[offset:]
    src_mac = ':'.join(f'{value:02x}' for value in raw[6:12])
    tcp_flags = 0
    if kind == dpkt.ethernet.ETH_TYPE_IP:
        if len(data) < 20:
            return None, 'unreadable_ip_packets'
        ip = dpkt.ip.IP()
        # Calling the fixed-header implementation avoids parsing incomplete TCP,
        # UDP, IPv6 extension headers, or other transport payloads as prerequisites.
        dpkt.Packet.unpack(ip, data)
        header_length = ip.hl * 4
        if ip.v != 4 or header_length < 20 or len(data) < header_length or ip.len < header_length:
            return None, 'unreadable_ip_packets'
        src, dst = socket.inet_ntoa(ip.src), socket.inet_ntoa(ip.dst)
        ip_length = ip.len
        # Noninitial fragments contain arbitrary payload, never a TCP SYN header.
        # A readable IP packet is still retained if its transport header is absent.
        if ip.p == dpkt.ip.IP_PROTO_TCP and ip.offset == 0:
            tcp = data[header_length:min(ip_length, len(data))]
            if len(tcp) >= 14:
                tcp_length = (tcp[12] >> 4) * 4
                if 20 <= tcp_length <= ip_length - header_length:
                    tcp_flags = tcp[13]
    elif kind == dpkt.ethernet.ETH_TYPE_IP6:
        if len(data) < 40 or data[0] >> 4 != 6:
            return None, 'unreadable_ip_packets'
        ip6 = dpkt.ip6.IP6()
        dpkt.Packet.unpack(ip6, data)
        src = socket.inet_ntop(socket.AF_INET6, ip6.src)
        dst = socket.inet_ntop(socket.AF_INET6, ip6.dst)
        ip_length = 40 + ip6.plen
    else:
        return None, 'non_ip_packets'
    return Packet(time_ns, src, dst, ip_length, src_mac, tcp_flags,
                  len(data) < ip_length), 'ip_packets'


def _endpoint_evidence(candidates, participation, macs, complete):
    maximum = max(participation.values(), default=0)
    leaders = sorted(address for address, count in participation.items() if count == maximum)
    selected = next(iter(candidates)) if len(candidates) == 1 else None
    return {'rule_version': ENDPOINT_RULE_VERSION, 'kind': 'inferred',
            'capture_complete': complete, 'syn_candidates': sorted(candidates),
            'selected_ipv4': selected, 'participation': dict(sorted(participation.items())),
            'participation_leaders': leaders,
            'ipv4_source_macs': sorted(macs.get(selected, set()))}


def _associate_ipv6(endpoint, macs, evidence):
    """Associate only a unique unicast source address owned by the runtime MAC."""
    ipv6_macs = {address: sorted(owners) for address, owners in sorted(macs.items())
                 if isinstance(ipaddress.ip_address(address), ipaddress.IPv6Address)}
    evidence.update(ipv6_source_macs=ipv6_macs, ipv6_candidates=[], associated_ipv6=[],
                    ipv6_rejections={})
    runtime_macs = macs[endpoint]
    if len(runtime_macs) != 1:
        evidence['ipv6_association_status'] = 'runtime_source_mac_not_unique'
        return []
    runtime_mac = next(iter(runtime_macs))
    mac_bytes = bytes.fromhex(runtime_mac.replace(':', ''))
    if mac_bytes[0] & 1 or mac_bytes == bytes(6):
        evidence['ipv6_association_status'] = 'runtime_source_mac_not_unicast'
        return []
    associated = []
    candidates = []
    rejected = {}
    for address, owners in ipv6_macs.items():
        ip = ipaddress.ip_address(address)
        if runtime_mac not in owners:
            continue
        if ip.is_unspecified or ip.is_multicast:
            rejected[address] = 'unspecified_or_multicast_source'
            continue
        candidates.append(address)
        if owners == [runtime_mac]:
            associated.append(address)
        else:
            rejected[address] = 'source_mac_conflict'
    evidence.update(ipv6_candidates=candidates, associated_ipv6=associated,
                    ipv6_rejections=rejected, ipv6_association_status='source_mac_checked')
    return associated


def scan_episode(path):
    participation: Counter[str] = Counter()
    candidates: set[str] = set()
    macs: dict[str, set[str]] = {}
    start = end = None
    quality = {'records': 0, 'ip_packets': 0, 'non_ip_packets': 0,
               'unreadable_ip_packets': 0, 'malformed_link_packets': 0,
               'direction_skipped': 0, 'truncated_packets': 0}
    try:
        for time_ns, raw in iter_records(path):
            start = time_ns if start is None else min(start, time_ns)
            end = time_ns if end is None else max(end, time_ns)
            quality['records'] += 1
            packet, category = _decode_frame(time_ns, raw)
            quality[category] += 1
            if packet is None:
                continue
            quality['truncated_packets'] += int(packet.payload_truncated)
            participation.update({packet.src, packet.dst})
            macs.setdefault(packet.src, set()).add(packet.src_mac)
            if packet.tcp_flags & dpkt.tcp.TH_SYN and not packet.tcp_flags & dpkt.tcp.TH_ACK:
                address = ipaddress.ip_address(packet.src)
                if any(address in net for net in PRIVATE_NETS):
                    candidates.add(packet.src)
    except EpisodeExcluded as exc:
        exc.context.update(start_ns=start, end_ns=end,
                           duration_ns=None if start is None else end - start,
                           quality=quality,
                           endpoint_source=_endpoint_evidence(candidates, participation, macs, False))
        raise
    evidence = _endpoint_evidence(candidates, participation, macs, True)
    times = {'start_ns': start, 'end_ns': end,
             'duration_ns': None if start is None else end - start, 'quality': quality,
             'endpoint_source': evidence}
    if start is None:
        raise EpisodeExcluded('empty_capture', 'capture contains no records', **times)
    assert end is not None  # start and end are set together for each record.
    if start == end:
        raise EpisodeExcluded('zero_capture_duration', 'capture interval has zero duration', **times)
    if len(candidates) != 1:
        raise EpisodeExcluded('endpoint_syn_candidates', f'expected one RFC1918 SYN source; got {sorted(candidates)}', **times)
    endpoint = next(iter(candidates))
    if evidence['participation_leaders'] != [endpoint]:
        raise EpisodeExcluded('endpoint_participation_conflict',
                              'SYN source must be the unique packet-participation leader',
                              **times)
    associated = _associate_ipv6(endpoint, macs, evidence)
    return EpisodeScan(start, end, [endpoint, *associated], evidence, quality)


def direction(packet, endpoints):
    outgoing, incoming = packet.src in endpoints, packet.dst in endpoints
    if outgoing == incoming:
        return None
    return 0 if outgoing else 1
