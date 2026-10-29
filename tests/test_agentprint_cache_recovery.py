"""Recovery acceptance through public CLI and durable cache artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import yaml

from test_agentprint import capture, frame


REPO = Path(__file__).resolve().parents[1]


class CacheRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'config.yaml'
        self.settings = {'extends': str(REPO / 'configs/agentprint/common.yaml'),
                         'task': 'scenario', 'transform': 'raw', 'mode': 'smoke',
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
                                capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result

    def episode(self, split, name):
        path = self.root / 'data' / split / 'engine/artifact_processing' / name / 'network.pcap'
        capture(path)
        return path

    def test_repeat_prepare_reuses_shards_for_both_input_transforms(self):
        self.episode('train', 'a')
        self.command('prepare')
        shard = self.root / 'cache/scenario/train/shards/00000.npy'
        stat = shard.stat()
        original = (self.root / 'cache/scenario/train/valid.jsonl').read_bytes()
        self.settings['transform'] = 'log1p'
        self.save_config()
        self.command('prepare')
        self.assertEqual(stat.st_mtime_ns, shard.stat().st_mtime_ns)
        self.assertEqual(original, (self.root / 'cache/scenario/train/valid.jsonl').read_bytes())

    def test_changed_annotation_endpoint_metadata_and_pcap_inventory_are_rejected(self):
        for changed in ('annotation', 'endpoint', 'pcap', 'added_source', 'version', 'configuration'):
            with self.subTest(changed=changed):
                cache = self.root / ('cache-' + changed)
                self.settings['paths']['cache_root'] = str(cache)
                self.save_config()
                path = self.episode('train', 'a')
                annotation = path.with_name('messages.jsonl')
                annotation.write_text('{"label":"N001"}\n', encoding='utf-8')
                endpoint = path.with_name('endpoint.json')
                endpoint.write_text('{"endpoint":"10.0.0.2"}', encoding='utf-8')
                self.command('prepare')
                if changed in ('annotation', 'endpoint'):
                    target = annotation if changed == 'annotation' else endpoint
                    before = target.stat()
                    target.write_bytes(target.read_bytes().replace(b'1', b'2'))
                    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
                elif changed == 'pcap':
                    before = path.stat()
                    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
                elif changed == 'added_source':
                    self.episode('train', 'extra')
                elif changed == 'version':
                    index_path = cache / 'index.json'
                    index = json.loads(index_path.read_text())
                    index['preprocessing_version'] = 'obsolete'
                    index_path.write_text(json.dumps(index))
                else:
                    self.settings['preprocessing'] = {'splits': ['train']}
                    self.save_config()
                result = self.command('prepare', ok=False)
                self.assertIn('incompatible', result.stderr.lower())
                self.assertTrue(any(reason in result.stderr.lower() for reason in
                                    ('source', 'metadata', 'version', 'config')))
                self.settings.pop('preprocessing', None)

    def test_terminated_prepare_recovers_across_1024_boundary_then_trains(self):
        self.episode('train', 'a')
        for index in range(1025):
            self.episode('val', f'{index:05d}')
        # A bounded final capture leaves ample time to terminate after shard 0.
        tail = self.episode('val', '99999')
        packets = [(0, frame('10.0.0.2', '8.8.8.8', 2)),
                   (1, frame('10.0.0.2', '1.1.1.1', 2))]
        packets += [(2, frame('8.8.8.8', '10.0.0.2'))] * 30000
        capture(tail, packets)
        # Also retain expected quality exclusions through a journal boundary.
        capture(self.episode('val', '00000-bad'), [])
        process = subprocess.Popen([sys.executable, '-m', 'agentprint', 'prepare',
                                    '--config', str(self.config)], cwd=REPO,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        commit = self.root / 'cache/scenario/val/commits/00000.json'
        deadline = time.monotonic() + 30
        while not commit.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if not commit.exists():
            process.terminate()
            stdout, stderr = process.communicate(timeout=10)
            self.fail('no committed shard before interruption: ' + (stdout + stderr).decode())
        process.terminate()
        process.communicate(timeout=10)
        shard = self.root / 'cache/scenario/val/shards/00000.npy'
        before = shard.stat().st_mtime_ns
        self.assertEqual(np.load(shard, mmap_mode='r').shape, (1024, 2, 2, 1800))
        cursor = json.loads(commit.read_text())['cursor']
        self.assertEqual(cursor['source_index'], 1025)  # includes one exclusion
        # An unpublished shard is never considered complete, even if named .npy.
        orphan = shard.with_name('00001.npy')
        orphan.write_bytes(b'not a complete numpy file')
        self.command('prepare')
        self.assertEqual(before, shard.stat().st_mtime_ns)
        base = self.root / 'cache/scenario/val'
        recovered = (base / 'valid.jsonl').read_bytes()
        excluded = (base / 'excluded.jsonl').read_bytes()
        records = [json.loads(line) for line in recovered.splitlines()]
        self.assertEqual(len(records), 1026)
        self.assertEqual(len({row['sample_id'] for row in records}), 1026)
        self.assertEqual(len(excluded.splitlines()), 1)
        expected_arrays = [np.load(path) for path in sorted((base / 'shards').glob('*.npy'))]
        self.assertEqual([a.shape[0] for a in expected_arrays], [1024, 2])
        self.assertTrue(all(a.dtype == np.uint64 for a in expected_arrays))
        self.command('prepare', '--cache-root', str(self.root / 'clean'))
        self.assertEqual(recovered, (self.root / 'clean/scenario/val/valid.jsonl').read_bytes())
        self.assertEqual(excluded, (self.root / 'clean/scenario/val/excluded.jsonl').read_bytes())
        for expected, path in zip(expected_arrays,
                                  sorted((self.root / 'clean/scenario/val/shards').glob('*.npy'))):
            np.testing.assert_array_equal(expected, np.load(path, mmap_mode='r'))
        self.command('train', '--device', 'cpu')
        checkpoint = next((self.root / 'runs').glob('*/best.pt'))
        self.command('evaluate', '--device', 'cpu', '--checkpoint', str(checkpoint), '--split', 'val')
        summary = json.loads((checkpoint.parent / 'evaluation-val/summary.json').read_text())
        self.assertEqual(summary['support'][0], 1026)
