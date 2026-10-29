"""Public command tests; synthetic captures are never research results."""
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


def frame(src, dst, flags=16):
    # Ethernet + minimal IPv4/TCP. IP total length=40, Ethernet length=54.
    ethernet = bytes.fromhex('0200000000020200000000010800')
    ip = struct.pack('!BBHHHBBH4s4s', 69, 0, 40, 0, 0, 64, 6, 0,
                     socket.inet_aton(src), socket.inet_aton(dst))
    tcp = struct.pack('!HHIIBBHHH', 12345, 443, 0, 0, 80, flags, 8192, 0, 0)
    return ethernet + ip + tcp


def capture(path, packets=None, linktype=1):
    packets = packets if packets is not None else [
        (0, frame('10.0.0.2', '8.8.8.8', 2)),
        (1, frame('10.0.0.2', '1.1.1.1', 2)),
        (900, frame('8.8.8.8', '10.0.0.2', 18)),
        (1800, frame('10.0.0.2', '8.8.8.8')),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        f.write(struct.pack('<IHHIIII', 0xa1b2c3d4, 2, 4, 0, 0, 65535, linktype))
        for seconds, raw in packets:
            f.write(struct.pack('<IIII', 1700000000 + seconds, 0, len(raw), len(raw)))
            f.write(raw)


class PublicCommands(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'smoke.yaml'
        self.config.write_text(yaml.safe_dump({
            'extends': str(REPO / 'configs/agentprint/common.yaml'),
            'task': 'scenario', 'transform': 'raw', 'mode': 'smoke',
            'paths': {'data_root': str(self.root / 'data'),
                      'cache_root': str(self.root / 'cache'),
                      'output_root': str(self.root / 'runs')},
            'training': {'epochs': 1, 'batch_size': 2},
        }), encoding='utf-8')

    def command(self, *args, ok=True):
        result = subprocess.run([sys.executable, '-m', 'agentprint', *args,
                                 '--config', str(self.config)], cwd=REPO,
                                text=True, capture_output=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def fixture(self):
        for split in ('train', 'val'):
            capture(self.root / 'data' / split / 'fixture_engine' /
                    'artifact_processing' / 'A001' / 'network.pcap')

    def test_prepare_public_artifacts_and_hand_computed_boundary_bins(self):
        self.fixture()
        self.command('prepare')
        base = self.root / 'cache/scenario/train'
        rows = [json.loads(x) for x in (base / 'valid.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        values = np.load(self.root / 'cache' / row['shard_relpath'], mmap_mode='r')
        expected = np.zeros((1, 2, 2, 1800), dtype=np.uint64)
        expected[0, 0, 0, [0, 1, 1799]] = 1
        expected[0, 1, 0, [0, 1, 1799]] = 40
        expected[0, 0, 1, 900] = 1
        expected[0, 1, 1, 900] = 40
        np.testing.assert_array_equal(values, expected)
        self.assertEqual(values.dtype, np.uint64)
        self.assertEqual(row['packet_count'], [3, 1])
        self.assertEqual(row['ip_bytes'], [120, 40])
        self.assertEqual(row['runtime_endpoints'], ['10.0.0.2'])
        self.assertEqual(row['start_ns'], 1700000000000000000)
        self.assertEqual(row['duration_ns'], 1800000000000)
        self.assertIsNone(row['annotation_relpath'])
        self.assertIsNone(row['annotation_row'])
        labels = json.loads((self.root / 'cache/label_map.json').read_text())
        self.assertEqual(labels['scenario']['business_services'], 1)
        self.assertEqual(labels['primitive']['N047'], 46)
        metadata = json.loads((base / 'metadata.json').read_text())
        for key in ('format_version', 'preprocessing_version', 'config_fingerprint',
                    'label_map_id', 'manifest_id'):
            self.assertIn(key, metadata)
        index = json.loads((self.root / 'cache/index.json').read_text())
        self.assertEqual(index['sources'][0]['pcap_relpath'],
                         'train/fixture_engine/artifact_processing/A001/network.pcap')
        self.assertIn('mtime_ns', index['sources'][0])
        self.assertEqual((base / 'excluded.jsonl').read_text(), '')

    def test_endpoint_ties_and_unsupported_linktype_are_explicit_exclusions(self):
        data = self.root / 'data/train/engine/artifact_processing'
        capture(data / 'tie/network.pcap', [
            (0, frame('10.0.0.2', '8.8.8.8', 2)),
            (1, frame('8.8.8.8', '10.0.0.2', 18))])
        capture(data / 'unsupported/network.pcap', linktype=101)
        self.command('prepare')
        base = self.root / 'cache/scenario/train'
        excluded = [json.loads(x) for x in (base / 'excluded.jsonl').read_text().splitlines()]
        self.assertEqual({x['reason'] for x in excluded},
                         {'endpoint_participation_conflict', 'unsupported_linktype'})
        self.assertEqual((base / 'valid.jsonl').read_text(), '')
        self.assertTrue(all('shard_relpath' not in x for x in excluded))

    def test_unknown_fields_and_experiment_cli_overrides_are_rejected(self):
        self.fixture()
        original = self.config.read_text()
        self.config.write_text(original + '\ntrainng: {}\n')
        self.assertIn('trainng', self.command('prepare', ok=False).stderr)
        self.config.write_text(original)
        self.assertIn('unrecognized', self.command('train', '--epochs', '1', ok=False).stderr)
        self.config.write_text(original + '\nseed: -1\n')
        self.assertIn('seed', self.command('prepare', ok=False).stderr)

    def test_full_capture_bounds_retransmissions_and_direction_skips(self):
        # ARP at both ends stretches the observation to 3600 seconds. Runtime
        # SYN at 2s enters bin 1; its observed retransmission remains in bin 1.
        arp = bytes.fromhex('ffffffffffff0200000000010806') + bytes(28)
        packets = [(0, arp),
                   (2, frame('10.0.0.2', '8.8.8.8', 2)),
                   (2, frame('10.0.0.2', '8.8.8.8', 2)),
                   (4, frame('10.0.0.2', '1.1.1.1', 1)),
                   (1800, frame('8.8.8.8', '10.0.0.2', 16)),
                   (2000, frame('9.9.9.9', '8.8.4.4', 16)),
                   (3600, arp)]
        capture(self.root / 'data/train/engine/artifact_processing/A001/network.pcap', packets)
        self.command('prepare')
        record = json.loads((self.root / 'cache/scenario/train/valid.jsonl').read_text())
        actual = np.load(self.root / 'cache' / record['shard_relpath'])[0]
        expected = np.zeros((2, 2, 1800), dtype=np.uint64)
        expected[0, 0, 1] = 2
        expected[1, 0, 1] = 80
        expected[0, 0, 2] = 1
        expected[1, 0, 2] = 40
        expected[0, 1, 900] = 1
        expected[1, 1, 900] = 40
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(record['quality']['non_ip_packets'], 2)
        self.assertEqual(record['quality']['direction_skipped'], 1)
        self.assertEqual(record['duration_ns'], 3600000000000)

    def test_ambiguous_and_unsupported_inputs_never_create_prefix_samples(self):
        base = self.root / 'data/train/engine/artifact_processing'
        capture(base / 'two_candidates/network.pcap', [
            (0, frame('10.0.0.2', '8.8.8.8', 2)),
            (1, frame('10.0.0.3', '8.8.8.8', 2))])
        capture(base / 'no_syn/network.pcap', [
            (0, frame('10.0.0.2', '8.8.8.8')),
            (1, frame('10.0.0.2', '1.1.1.1'))])
        capture(base / 'damaged/network.pcap')
        with (base / 'damaged/network.pcap').open('ab') as handle:
            handle.write(b'partial')
        udp = bytearray(frame('10.0.0.2', '8.8.8.8'))
        udp[23] = 17
        capture(base / 'udp/network.pcap', [(0, bytes(udp)), (1, bytes(udp))])
        ipv6 = bytes.fromhex('02000000000202000000000186dd') + bytes(40)
        capture(base / 'ipv6/network.pcap', [(0, ipv6), (1, ipv6)])
        self.command('prepare')
        target = self.root / 'cache/scenario/train'
        self.assertEqual((target / 'valid.jsonl').read_text(), '')
        self.assertEqual(list((target / 'shards').iterdir()), [])
        rows = [json.loads(line) for line in (target / 'excluded.jsonl').read_text().splitlines()]
        self.assertEqual({row['episode_id']: row['reason'] for row in rows}, {
            'two_candidates': 'endpoint_syn_candidates', 'no_syn': 'endpoint_syn_candidates',
            'damaged': 'damaged_capture', 'udp': 'endpoint_syn_candidates',
            'ipv6': 'endpoint_syn_candidates'})

    def test_raw_scenario_prepare_train_evaluate_commands(self):
        self.fixture()
        self.command('prepare')
        self.command('train', '--device', 'cpu')
        runs = list((self.root / 'runs').iterdir())
        self.assertEqual(len(runs), 1)
        run = runs[0]
        self.assertTrue((run / 'best.pt').is_file())
        self.assertTrue((run / 'resolved_config.yaml').is_file())
        self.command('evaluate', '--device', 'cpu', '--checkpoint', str(run / 'best.pt'),
                     '--split', 'val')
        summary = json.loads((run / 'evaluation-val/summary.json').read_text())
        self.assertEqual(summary['support'], [1, 0, 0, 0, 0])
        self.assertEqual(summary['evaluated_classes'], [0])
        self.assertTrue(np.isfinite(summary['loss']))
        self.assertGreaterEqual(summary['accuracy'], 0)
        self.assertLessEqual(summary['macro_f1'], 1)
        self.assertEqual(len((run / 'metrics.jsonl').read_text().splitlines()), 1)


class PublicModel(unittest.TestCase):
    def test_both_fixed_heads_support_finite_ce_and_backpropagation(self):
        import torch
        from agentprint.model import getRF
        torch.set_num_threads(1)
        for classes in (5, 47):
            model = getRF(classes)
            scores = model(torch.randn(2, 2, 2, 1800))
            self.assertEqual(tuple(scores.shape), (2, classes))
            loss = torch.nn.CrossEntropyLoss()(scores, torch.tensor([0, classes - 1]))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
