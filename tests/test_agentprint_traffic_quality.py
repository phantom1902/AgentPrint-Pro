"""Hand-computed traffic quality fixtures exercised only via public commands."""
import json
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
EPOCH_NS = 1700000000000000000
RUNTIME_MAC = '020000000001'
PEER_MAC = '020000000002'


def ethernet(payload, kind=0x0800, mac=RUNTIME_MAC, destination=PEER_MAC, vlan=False):
    header = bytes.fromhex(destination + mac)
    if vlan:
        return header + struct.pack('!HHH', 0x8100, 7, kind) + payload
    return header + struct.pack('!H', kind) + payload


def ipv4(src='10.0.0.2', dst='8.8.8.8', protocol=6, flags=16,
         payload=None, declared=None, fragment=0, mac=RUNTIME_MAC, vlan=False):
    if payload is None:
        payload = struct.pack('!HHIIBBHHH', 12345, 443, 0, 0, 80, flags, 8192, 0, 0)
    length = 20 + len(payload) if declared is None else declared
    header = struct.pack('!BBHHHBBH4s4s', 69, 0, length, 0, fragment, 64, protocol,
                         0, socket.inet_aton(src), socket.inet_aton(dst))
    return ethernet(header + payload, mac=mac, vlan=vlan)


def ipv6(src='2001:db8::2', dst='2001:4860::1', mac=RUNTIME_MAC,
         destination=PEER_MAC, payload=bytes(8), declared=None):
    length = len(payload) if declared is None else declared
    header = struct.pack('!IHBB16s16s', 6 << 28, length, 17, 64,
                         socket.inet_pton(socket.AF_INET6, src),
                         socket.inet_pton(socket.AF_INET6, dst))
    return ethernet(header + payload, 0x86dd, mac, destination)


def write_capture(path, packets, *, endian='<', nano=True, linktype=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        magic = 0xa1b23c4d if nano else 0xa1b2c3d4
        handle.write(struct.pack(endian + 'IHHIIII', magic, 2, 4, 0, 0, 65535, linktype))
        for item in packets:
            offset, raw = item[:2]
            wire = item[2] if len(item) == 3 else len(raw)
            seconds, fraction = divmod(EPOCH_NS + offset, 1000000000)
            if not nano:
                fraction //= 1000
            handle.write(struct.pack(endian + 'IIII', seconds, fraction, len(raw), wire))
            handle.write(raw)


class TrafficQuality(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'quality.yaml'
        self.config.write_text(yaml.safe_dump({
            'extends': str(REPO / 'configs/agentprint/common.yaml'),
            'task': 'scenario', 'transform': 'raw', 'mode': 'smoke',
            'paths': {'data_root': str(self.root / 'data'),
                      'cache_root': str(self.root / 'cache'),
                      'output_root': str(self.root / 'runs')},
            'training': {'epochs': 1, 'batch_size': 2},
        }), encoding='utf-8')

    def command(self, *args):
        result = subprocess.run([sys.executable, '-m', 'agentprint', *args,
                                 '--config', str(self.config)], cwd=REPO,
                                text=True, encoding='utf-8', capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def episode(self, name, packets, split='train', **options):
        path = self.root / 'data' / split / 'engine/artifact_processing' / name / 'network.pcap'
        write_capture(path, packets, **options)
        return path

    def rows(self, kind='valid', split='train'):
        path = self.root / f'cache/scenario/{split}/{kind}.jsonl'
        return {row['episode_id']: row for row in map(json.loads, path.read_text().splitlines())}

    def values(self, row):
        return np.load(self.root / 'cache' / row['shard_relpath'])[row['row_index']]

    def clean_packets(self, src='10.0.0.2', mac=RUNTIME_MAC):
        return [(0, ipv4(src, '8.8.8.8', flags=2, mac=mac)),
                (1, ipv4(src, '1.1.1.1', mac=mac)),
                (2, ipv4('8.8.8.8', src, mac=PEER_MAC))]

    def test_mixed_ip_lengths_truncation_and_capture_bounds_feed_training(self):
        arp = ethernet(bytes(28), 0x0806)
        packets = [(0, arp),
                   (1000000000, ipv4(flags=2)),
                   (1000000000, ipv4(flags=2)),  # observed retransmission
                   (2000000000, ipv4(dst='1.1.1.1')),
                   (3000000000, ipv4(dst='9.9.9.9', protocol=17, payload=bytes(8))),
                   (4000000000, ipv4('8.8.8.8', '10.0.0.2', protocol=1, payload=bytes(8))),
                   (5000000000, ipv4(dst='4.4.4.4', protocol=99, payload=b'')),
                   (6000000000, ipv4('8.8.8.8', '10.0.0.2', payload=bytes(4), fragment=1)),
                   (7000000000, ipv4(dst='9.9.9.9', payload=b'', declared=100), 114),
                   (8000000000, ipv6()),
                   (9000000000, ipv6('2001:4860::1', '2001:db8::2', mac=PEER_MAC)),
                   (10000000000, ethernet(bytes.fromhex('45000028000000004006'))),
                   (11000000000, b'broken!'),
                   (1800000000000, arp)]
        for split in ('train', 'val'):
            self.episode('mixed', packets, split)
        self.command('prepare')
        self.assertIn('mixed', self.rows())
        row = self.rows()['mixed']
        self.assertEqual(row['runtime_endpoints'], ['10.0.0.2', '2001:db8::2'])
        self.assertEqual(row['packet_count'], [7, 3])
        self.assertEqual(row['ip_bytes'], [316, 100])
        self.assertEqual(row['start_ns'], EPOCH_NS)
        self.assertEqual(row['end_ns'], EPOCH_NS + 1800000000000)
        self.assertEqual(row['quality']['records'], 14)
        self.assertEqual(row['quality']['ip_packets'], 10)
        self.assertEqual(row['quality']['non_ip_packets'], 2)
        self.assertEqual(row['quality']['unreadable_ip_packets'], 1)
        self.assertEqual(row['quality']['malformed_link_packets'], 1)
        self.assertEqual(row['quality']['truncated_packets'], 1)
        expected = np.zeros((2, 2, 1800), dtype=np.uint64)
        expected[0, 0, [1, 2, 3, 5, 7, 8]] = [2, 1, 1, 1, 1, 1]
        expected[1, 0, [1, 2, 3, 5, 7, 8]] = [80, 40, 28, 20, 100, 48]
        expected[0, 1, [4, 6, 9]] = 1
        expected[1, 1, [4, 6, 9]] = [28, 24, 48]
        np.testing.assert_array_equal(self.values(row), expected)
        self.command('train', '--device', 'cpu')
        run = next((self.root / 'runs').iterdir())
        self.command('evaluate', '--checkpoint', str(run / 'best.pt'), '--split', 'val')
        summary = json.loads((run / 'evaluation-val/summary.json').read_text())
        self.assertEqual(summary['support'], [1, 0, 0, 0, 0])
        self.assertTrue(np.isfinite(summary['loss']))

    def test_ipv6_association_is_conflict_free_and_never_uses_destination_mac(self):
        packets = self.clean_packets() + [
            (3, ipv6()),
            (4, ipv6('2001:db8::3')),
            (5, ipv6('2001:db8::3', mac=PEER_MAC)),  # conflicting source ownership
            (6, ipv6('::')),
            (7, ipv6('ff02::1')),
            (8, ipv6('2001:db8::4', 'ff02::1', mac=PEER_MAC, destination=RUNTIME_MAC)),
            (9, ipv6('2001:db8::2', '2001:db8::2')),
            (10, ipv6('2001:db8::5', '2001:db8::2', mac=PEER_MAC)),
        ]
        # Give the IPv4 runtime a clear overall participation lead.
        packets += [(11 + n, ipv4(dst=f'9.9.9.{n + 1}')) for n in range(8)]
        self.episode('association', packets)
        self.command('prepare')
        self.assertIn('association', self.rows())
        row = self.rows()['association']
        self.assertEqual(row['runtime_endpoints'], ['10.0.0.2', '2001:db8::2'])
        self.assertEqual(row['quality']['direction_skipped'], 6)
        self.assertEqual(row['packet_count'], [11, 2])
        evidence = row['endpoint_source']
        self.assertEqual(evidence['ipv4_source_macs'], ['02:00:00:00:00:01'])
        self.assertEqual(evidence['ipv6_candidates'], ['2001:db8::2', '2001:db8::3'])
        self.assertEqual(evidence['associated_ipv6'], ['2001:db8::2'])
        self.assertEqual(evidence['ipv6_rejections']['2001:db8::3'], 'source_mac_conflict')
        self.assertEqual(evidence['ipv6_source_macs']['2001:db8::3'],
                         ['02:00:00:00:00:01', '02:00:00:00:00:02'])
        self.assertEqual(evidence['kind'], 'inferred')

    def test_multiple_runtime_macs_disable_ipv6_without_excluding_ipv4(self):
        packets = self.clean_packets() + [(3, ipv4(dst='9.9.9.9', mac=PEER_MAC)),
                                          (4, ipv6())]
        self.episode('multi_mac', packets)
        self.command('prepare')
        self.assertIn('multi_mac', self.rows())
        row = self.rows()['multi_mac']
        self.assertEqual(row['runtime_endpoints'], ['10.0.0.2'])
        self.assertEqual(row['packet_count'], [3, 1])
        self.assertEqual(row['quality']['direction_skipped'], 1)
        self.assertEqual(row['endpoint_source']['ipv6_association_status'],
                         'runtime_source_mac_not_unique')

    def test_readable_tcp_flags_survive_header_truncation_and_bad_ip_headers_skip(self):
        tcp_prefix = struct.pack('!HHIIBB', 12345, 443, 0, 0, 80, 2)
        short_options = bytearray(ipv4(payload=b''))
        short_options[14] = 0x46  # IHL=24, captured IP header is only 20 bytes.
        bad_length = ipv4(payload=b'', declared=19)
        wrong_ipv6_version = bytearray(ipv6())
        wrong_ipv6_version[14] = 0x50
        packets = [(0, ipv4(payload=tcp_prefix, declared=40), 54),
                   (1, ipv4(dst='1.1.1.1')),
                   (2, ipv6(payload=b'', declared=16), 70),
                   (3, bytes(short_options)), (4, bad_length),
                   (5, bytes(wrong_ipv6_version)),
                   (6, bytes.fromhex(PEER_MAC + RUNTIME_MAC + '810000'))]
        self.episode('short_headers', packets)
        self.command('prepare')
        self.assertIn('short_headers', self.rows())
        row = self.rows()['short_headers']
        self.assertEqual(row['packet_count'], [3, 0])
        self.assertEqual(row['ip_bytes'], [136, 0])
        self.assertEqual(row['runtime_endpoints'], ['10.0.0.2', '2001:db8::2'])
        self.assertEqual(row['quality']['truncated_packets'], 2)
        self.assertEqual(row['quality']['unreadable_ip_packets'], 3)
        self.assertEqual(row['quality']['malformed_link_packets'], 1)
        self.assertEqual(row['duration_ns'], 6)

    def test_only_packets_between_runtime_addresses_exclude_without_zero_sample(self):
        self.episode('self', [(0, ipv4(dst='10.0.0.2', flags=2)),
                              (1, ipv4(dst='10.0.0.2'))])
        self.command('prepare')
        self.assertEqual(self.rows(), {})
        row = self.rows('excluded')['self']
        self.assertEqual(row['reason'], 'no_direction_packets')
        self.assertEqual(row['quality']['direction_skipped'], 2)
        self.assertNotIn('shard_relpath', row)

    def test_endpoint_candidates_follow_exact_syn_rfc1918_and_overall_leader_rule(self):
        self.episode('none', [(0, ipv4(protocol=17)), (1, ipv4(dst='1.1.1.1', protocol=17))])
        self.episode('syn_ack', [(0, ipv4(flags=18)), (1, ipv4(dst='1.1.1.1'))])
        self.episode('outside_rfc1918', self.clean_packets('172.15.255.255'))
        self.episode('rfc1918_lower', self.clean_packets('172.16.0.0'))
        self.episode('rfc1918_upper', self.clean_packets('192.168.255.255'))
        self.episode('multiple', self.clean_packets() + [(3, ipv4('10.0.0.3', flags=2))])
        self.episode('tie', [(0, ipv4(flags=2)), (1, ipv4())])
        self.episode('ipv6_leader', self.clean_packets() + [
            (3 + n, ipv6('2001:db8::a', f'2001:db8::{n + 10}', mac=PEER_MAC))
            for n in range(4)])
        self.command('prepare')
        self.assertEqual(set(self.rows()), {'rfc1918_lower', 'rfc1918_upper'})
        excluded = self.rows('excluded')
        self.assertEqual({name: row['reason'] for name, row in excluded.items()}, {
            'none': 'endpoint_syn_candidates', 'syn_ack': 'endpoint_syn_candidates',
            'outside_rfc1918': 'endpoint_syn_candidates', 'multiple': 'endpoint_syn_candidates',
            'tie': 'endpoint_participation_conflict', 'ipv6_leader': 'endpoint_participation_conflict'})
        self.assertEqual(excluded['multiple']['endpoint_source']['syn_candidates'],
                         ['10.0.0.2', '10.0.0.3'])
        self.assertEqual(excluded['none']['endpoint_source']['selected_ipv4'], None)
        self.assertEqual(excluded['ipv6_leader']['endpoint_source']['participation_leaders'],
                         ['2001:db8::a'])

    def test_record_damage_excludes_whole_episode_with_available_quality(self):
        self.episode('good', self.clean_packets())
        path = self.episode('damaged', self.clean_packets())
        with path.open('ab') as handle:
            handle.write(b'partial')
        path = self.episode('short_record', self.clean_packets())
        with path.open('ab') as handle:
            handle.write(struct.pack('<IIII', 1700000001, 0, 50, 50) + b'abc')
        self.episode('empty', [])
        self.episode('zero', [(0, ipv4(flags=2)), (0, ipv4(dst='1.1.1.1'))])
        self.command('prepare')
        self.assertEqual(set(self.rows()), {'good'})
        excluded = self.rows('excluded')
        self.assertEqual({name: row['reason'] for name, row in excluded.items()}, {
            'damaged': 'damaged_capture', 'short_record': 'damaged_capture',
            'empty': 'empty_capture', 'zero': 'zero_capture_duration'})
        self.assertIn('quality', excluded['damaged'])
        self.assertEqual(excluded['damaged']['quality']['records'], 3)
        self.assertEqual(excluded['damaged']['start_ns'], EPOCH_NS)
        self.assertEqual(excluded['damaged']['endpoint_source']['capture_complete'], False)
        self.assertTrue(all('shard_relpath' not in row for row in excluded.values()))

    def test_pcap_byte_order_resolution_vlan_and_unsupported_formats(self):
        for name, endian, nano, step in (('le_ns', '<', True, 1), ('be_ns', '>', True, 1),
                                         ('le_us', '<', False, 1000), ('be_us', '>', False, 1000)):
            self.episode(name, [(0, ipv4(flags=2, vlan=True)),
                                (step, ipv4(dst='1.1.1.1', vlan=True)),
                                (1800 * step, ipv4('8.8.8.8', '10.0.0.2', vlan=True))],
                         endian=endian, nano=nano)
        self.episode('unknown_link', self.clean_packets(), linktype=101)
        path = self.episode('pcapng', [])
        path.write_bytes(bytes.fromhex('0a0d0d0a') + bytes(24))
        self.command('prepare')
        self.assertEqual(set(self.rows()), {'le_ns', 'be_ns', 'le_us', 'be_us'})
        for name, row in self.rows().items():
            self.assertEqual(row['start_ns'], EPOCH_NS)
            self.assertEqual(row['duration_ns'], 1800000 if name.endswith('us') else 1800)
            expected = np.zeros((2, 2, 1800), dtype=np.uint64)
            expected[0, 0, [0, 1]] = 1
            expected[1, 0, [0, 1]] = 40
            expected[0, 1, 1799] = 1
            expected[1, 1, 1799] = 40
            np.testing.assert_array_equal(self.values(row), expected)
        self.assertEqual({name: row['reason'] for name, row in self.rows('excluded').items()},
                         {'unknown_link': 'unsupported_linktype', 'pcapng': 'unsupported_capture_format'})


if __name__ == '__main__':
    unittest.main()
