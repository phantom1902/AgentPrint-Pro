"""Primitive acceptance through CLI artifacts; all traffic is synthetic."""
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
BASE_NS = 1700000000000000000


def frame(src, dst, flags=16):
    ethernet = bytes.fromhex('0200000000020200000000010800')
    ip = struct.pack('!BBHHHBBH4s4s', 69, 0, 40, 0, 0, 64, 6, 0,
                     socket.inet_aton(src), socket.inet_aton(dst))
    tcp = struct.pack('!HHIIBBHHH', 12345, 443, 0, 0, 80, flags, 8192, 0, 0)
    return ethernet + ip + tcp


def capture(path, offsets=(0, 1, 4, 5, 8, 9, 12)):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        handle.write(struct.pack('<IHHIIII', 0xa1b23c4d, 2, 4, 0, 0, 65535, 1))
        for index, offset in enumerate(offsets):
            raw = frame('10.0.0.2', '8.8.8.8' if index % 2 else '1.1.1.1',
                        2 if index == 0 else 16)
            handle.write(struct.pack('<IIII', 1700000000, offset, len(raw), len(raw)))
            handle.write(raw)


def annotation(offset, label):
    return {'start_time': f'2023-11-14T22:13:20.{offset:09d}Z',
            'primitive_label_group_id': label, 'macro_group_id': label,
            'original_macro_group_id': 'G01', 'message': 'must not enter manifests'}


class PrimitiveCommands(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'smoke.yaml'
        self.settings = {
            'extends': str(REPO / 'configs/agentprint/common.yaml'),
            'task': 'primitive', 'transform': 'raw', 'mode': 'smoke',
            'paths': {'data_root': str(self.root / 'data'),
                      'cache_root': str(self.root / 'cache'),
                      'output_root': str(self.root / 'runs')},
            'training': {'epochs': 1, 'batch_size': 2}}
        self.save_config()

    def save_config(self):
        self.config.write_text(yaml.safe_dump(self.settings), encoding='utf-8')

    def command(self, *args, ok=True):
        result = subprocess.run([sys.executable, '-m', 'agentprint', *args,
                                 '--config', str(self.config)], cwd=REPO,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result

    def fixture(self, rows, episode='A001', split='train', offsets=(0, 1, 4, 5, 8, 9, 12)):
        target = self.root / 'data' / split / 'engine' / 'artifact_processing' / episode
        capture(target / 'network.pcap', offsets)
        if rows is not None:
            (target / 'messages.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        return target

    def records(self, filename, task='primitive', split='train', cache='cache'):
        path = self.root / cache / task / split / filename
        return [json.loads(line) for line in path.read_text().splitlines()]

    def values(self, record):
        return np.load(self.root / 'cache' / record['shard_relpath'], mmap_mode='r')[record['row_index']]

    def test_adjacent_starts_preserve_ns_full_intervals_and_final_capture_packet(self):
        # Deliberately unsorted log rows: interval order differs from source rows.
        self.fixture([annotation(8, 'N047'), annotation(0, 'N001'), annotation(4, 'N021')])
        self.command('prepare')
        rows = self.records('valid.jsonl')
        self.assertEqual([row['annotation_row'] for row in rows], [2, 3, 1])
        self.assertEqual([row['label_id'] for row in rows], [0, 20, 46])
        self.assertEqual([row['start_ns'] for row in rows], [BASE_NS, BASE_NS + 4, BASE_NS + 8])
        self.assertEqual([row['end_ns'] for row in rows], [BASE_NS + 4, BASE_NS + 8, BASE_NS + 12])
        for index, row in enumerate(rows):
            expected = np.zeros((2, 2, 1800), dtype=np.uint64)
            bins = [0, 450, 1799] if index == 2 else [0, 450]
            expected[0, 0, bins] = 1
            expected[1, 0, bins] = 40
            np.testing.assert_array_equal(self.values(row), expected)
            self.assertEqual(row['duration_ns'], 4)
            self.assertEqual(row['runtime_endpoints'], ['10.0.0.2'])
            self.assertEqual(row['annotation_relpath'], 'train/engine/artifact_processing/A001/messages.jsonl')
        self.assertEqual(rows[0]['endpoint_source'], rows[-1]['endpoint_source'])
        self.assertEqual(self.records('excluded.jsonl'), [])
        self.assertNotIn('must not enter manifests', (self.root / 'cache/primitive/train/valid.jsonl').read_text())
        index = json.loads((self.root / 'cache/index.json').read_text())
        self.assertEqual(len(index['sources'][0]['annotation_sha256']), 64)

    def test_duplicate_start_groups_exclude_only_their_intervals(self):
        self.fixture([annotation(0, 'N001'), annotation(4, 'N002'), annotation(4, 'N003'),
                      annotation(8, 'N047')])
        self.command('prepare')
        rows = self.records('valid.jsonl')
        excluded = self.records('excluded.jsonl')
        self.assertEqual([row['label_id'] for row in rows], [0, 46])
        self.assertEqual([row['packet_count'] for row in rows], [[2, 0], [3, 0]])
        self.assertEqual([row['annotation_row'] for row in excluded], [2, 3])
        self.assertEqual({row['reason'] for row in excluded}, {'duplicate_annotation_start'})
        self.assertTrue(all(row['start_ns'] == BASE_NS + 4 and row['end_ns'] == BASE_NS + 8
                            and 'row_index' not in row for row in excluded))

    def test_invalid_last_segment_never_moves_endpoint_packet_to_previous_valid_one(self):
        self.fixture([annotation(0, 'N001'), annotation(8, 'N002'), annotation(8, 'N003')])
        self.fixture([annotation(0, 'N001'), annotation(12, 'N047')], episode='zero')
        self.command('prepare')
        rows = self.records('valid.jsonl')
        excluded = self.records('excluded.jsonl')
        self.assertEqual([row['packet_count'] for row in rows], [[4, 0], [6, 0]])
        self.assertEqual({row['reason'] for row in excluded},
                         {'duplicate_annotation_start', 'zero_segment_duration'})

    def test_unaligned_empty_unknown_and_missing_annotations_leave_scenario_eligible(self):
        self.fixture([annotation(0, 'N001'), annotation(2, 'N002'), annotation(3, 'N003'),
                      annotation(8, 'G01')])
        self.fixture([{'start_time': '1970-01-01T00:00:42.973Z',
                       'macro_group_id': 'N001'}], episode='unaligned')
        missing = self.fixture(None, episode='missing')
        (missing / 'messages.jsonl.incomplete').write_text(json.dumps(annotation(0, 'N001')))
        self.command('prepare')
        self.assertEqual({row['reason'] for row in self.records('excluded.jsonl')},
                         {'no_direction_packets', 'invalid_primitive_label',
                          'annotation_outside_capture', 'missing_annotations'})
        self.settings['task'] = 'scenario'
        self.settings['paths']['cache_root'] = str(self.root / 'scenario-cache')
        self.save_config()
        self.command('prepare')
        scenario = self.records('valid.jsonl', task='scenario', cache='scenario-cache')
        self.assertEqual(len(scenario), 3)

    def test_47_class_raw_prepare_train_evaluate_commands(self):
        for split in ('train', 'val'):
            self.fixture([annotation(0, 'N001'), annotation(4, 'N047')], split=split)
        self.command('prepare')
        self.command('train', '--device', 'cpu')
        run, = list((self.root / 'runs').iterdir())
        self.assertIn('primitive-raw', run.name)
        self.command('evaluate', '--device', 'cpu', '--checkpoint', str(run / 'best.pt'),
                     '--split', 'val')
        summary = json.loads((run / 'evaluation-val/summary.json').read_text())
        self.assertEqual(summary['support'], [1] + [0] * 45 + [1])
        self.assertEqual(summary['evaluated_classes'], [0, 46])
        self.assertEqual(summary['sample_count'], 2)
        self.assertTrue(np.isfinite(summary['loss']))
        self.assertGreaterEqual(summary['fixed47_macro_f1'], 0)
        self.assertLessEqual(summary['fixed47_macro_f1'], 1)
        self.assertEqual(len((run / 'per_class.csv').read_text().splitlines()), 48)

    def test_episode_failure_exclusions_preserve_annotation_start_provenance(self):
        episode = self.fixture([annotation(4, 'N001'), annotation(8, 'N047')])
        with (episode / 'network.pcap').open('ab') as handle:
            handle.write(b'partial')
        tie = self.fixture([annotation(4, 'N001'), annotation(8, 'N047')],
                           episode='tie', offsets=(0, 12)) / 'network.pcap'
        tie.write_bytes(tie.read_bytes().replace(socket.inet_aton('8.8.8.8'),
                                                socket.inet_aton('1.1.1.1')))
        self.command('prepare')
        self.assertEqual(self.records('valid.jsonl'), [])
        rows = self.records('excluded.jsonl')
        self.assertEqual([row['start_ns'] for row in rows],
                         [BASE_NS + 4, BASE_NS + 8, BASE_NS + 4, BASE_NS + 8])
        self.assertEqual({row['reason'] for row in rows},
                         {'damaged_capture', 'endpoint_participation_conflict'})
        self.assertTrue(all('shard_relpath' not in row for row in rows))


if __name__ == '__main__':
    unittest.main()
