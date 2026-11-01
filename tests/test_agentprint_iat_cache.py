"""Supplemental IAT cache acceptance through public CLI and loader outputs."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

from test_agentprint import REPO, frame
from test_agentprint_iat_fit import nano_capture, ARP


class IATCommands(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'config.yaml'
        self.settings = dict(extends=str(REPO / 'configs/agentprint/common-v2.yaml'),
            mode='smoke', paths={name + '_root': str(self.root / name)
                                 for name in ('data', 'cache', 'output')},
            preprocessing={'splits': ['train', 'val', 'test']},
            training={'epochs': 1, 'batch_size': 2})

    def episode(self, name, times, split='train'):
        path = self.root / 'data' / split / 'engine/artifact_processing' / name / 'network.pcap'
        nano_capture(path, times)
        return path

    def command(self, command, *args, ok=True):
        self.config.write_text(yaml.safe_dump(self.settings), encoding='utf-8')
        result = subprocess.run([sys.executable, '-m', 'agentprint', command,
            '--config', str(self.config), *args], cwd=REPO, capture_output=True,
            text=True, timeout=120)
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result

    def enable(self, root='iat'):
        self.settings['feature'] = dict(iat_enabled=True,
            fit_artifact=str(self.root / 'fit.json'), iat_cache_root=str(self.root / root))

    def fixture(self):
        self.episode('A', [0, 10, 14, 11, 14, 19, 26, 35, 46, 59, 1000])
        self.episode('B', [0, 10, 110, 1000])
        self.episode('C', [0, 10, 10, 1000])
        for split in ('val', 'test'):
            self.episode('A', [0, 10, 11, 1000], split)
        self.command('prepare')
        self.enable()
        self.command('fit-bins')

    def test_prepare_counts_all_intervals_in_current_direction_and_original_slots(self):
        self.fixture()
        self.command('prepare')
        from agentprint.config import load_config
        from agentprint.v2 import loader
        samples, _ = loader(load_config(self.config), 'train')
        values, _ = samples[0]
        self.assertEqual(values.shape, (10, 2, 1800))
        self.assertEqual(values.dtype, np.float32)
        expected = np.zeros((8, 2, 1800), dtype=np.float32)
        for bucket, direction, slot in [(1, 1, 19), (2, 0, 25), (0, 0, 25),
                (2, 1, 34), (3, 0, 46), (3, 1, 63), (4, 0, 82), (4, 1, 106)]:
            expected[bucket, direction, slot] += 1
        np.testing.assert_array_equal(values[2:], expected)
        self.assertEqual(int(samples[2][0][2:].sum()), 1)  # zero-only Episode remains eligible
        iat = self.root / 'iat/scenario/train'
        rows = [json.loads(line) for line in (iat / 'valid.jsonl').read_text().splitlines()]
        raw = np.load(self.root / 'iat' / rows[0]['shard_relpath'])
        self.assertEqual(raw.dtype, np.uint64)
        self.assertEqual(raw.shape, (3, 8, 2, 1800))
        self.assertEqual(samples.metadata['valid_count'], 3)
        self.assertEqual(samples.iat_metadata['fit_fingerprint'], json.loads((self.root / 'fit.json').read_text())['fingerprint'])

    def test_fit_and_cache_identity_are_saved_and_checked_for_val_and_test(self):
        import torch
        self.fixture()
        self.command('prepare')
        raw_metadata = (self.root / 'iat/scenario/train/metadata.json').read_bytes()
        from agentprint.config import load_config
        from agentprint.v2 import loader
        raw, _ = loader(load_config(self.config), 'train')
        self.addCleanup(lambda: (setattr(raw, '_iat_shard', None), setattr(raw, '_shard', None)))
        raw_values = raw[0][0]
        for transform in ('raw', 'log1p'):
            self.settings['transform'] = transform
            self.command('prepare')
            self.assertEqual((self.root / 'iat/scenario/train/metadata.json').read_bytes(), raw_metadata)
            samples, _ = loader(load_config(self.config), 'train')
            self.addCleanup(lambda s=samples: (setattr(s, '_iat_shard', None), setattr(s, '_shard', None)))
            np.testing.assert_array_equal(samples[0][0], raw_values if transform == 'raw' else np.log1p(raw_values))
            result = self.command('train')
            run = Path(json.loads(result.stdout.splitlines()[-1])['run_dir'])
            saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
            fit = json.loads((self.root / 'fit.json').read_text())
            self.assertEqual(saved['identity']['fit_fingerprint'], fit['fingerprint'])
            self.assertEqual(saved['identity']['iat_cache']['train'], samples.iat_metadata)
            self.assertEqual(saved['identity']['base_cache']['train'], samples.metadata)
            self.assertEqual(saved['feature_definition']['in_channels'], 10)
            for split in ('val', 'test'):
                self.command('evaluate', '--checkpoint', str(run / 'best.pt'), '--split', split)
            saved['identity']['iat_cache']['train']['fit_fingerprint'] = 'foreign-fit'
            from agentprint.config import fingerprint
            saved['experiment_id'] = fingerprint(saved['identity'])
            torch.save(saved, run / 'foreign.pt')
            self.assertIn('checkpoint identity', self.command('evaluate', '--checkpoint', str(run / 'foreign.pt'), ok=False).stderr)

    def test_overlapping_iat_root_refused_before_base_or_sources_can_be_modified(self):
        self.fixture()
        before = (self.root / 'cache/scenario/train/metadata.json').read_bytes()
        for target in (self.root / 'cache', self.root, self.root / 'data/nested', self.root / 'fit.json.progress'):
            self.settings['feature']['iat_cache_root'] = str(target)
            self.assertIn('overlap', self.command('prepare', ok=False).stderr)
        self.assertEqual(before, (self.root / 'cache/scenario/train/metadata.json').read_bytes())

    def test_missing_base_wrong_fit_changed_source_and_missing_sample_are_refused(self):
        self.enable()
        self.assertIn('fit artifact is missing', self.command('prepare', ok=False).stderr)
        self.settings.pop('feature')
        self.fixture()
        self.command('prepare')
        fit_path = self.root / 'fit.json'
        original = fit_path.read_bytes()
        from agentprint.config import fingerprint
        fit = json.loads(original)
        for field, value, message in [('scope', 'full', 'fit scope'),
                ('train_manifest', [], 'fit train_manifest'),
                ('boundaries_ns', [4, 7, 11, 100, 100, 100], 'cache source/base/fit')]:
            changed = dict(fit, **{field: value})
            changed['fingerprint'] = fingerprint({k: v for k, v in changed.items() if k != 'fingerprint'})
            fit_path.write_text(json.dumps(changed))
            self.assertIn(message, self.command('prepare', ok=False).stderr)
        fit_path.write_bytes(original)
        endpoint = self.root / 'data/val/engine/artifact_processing/A/runtime.json'
        endpoint.write_text('{}')
        self.assertIn('endpoint metadata', self.command('prepare', ok=False).stderr)
        endpoint.unlink()
        capture = endpoint.with_name('network.pcap')
        stat = capture.stat()
        os.utime(capture, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        self.assertIn('PCAP path/size/mtime', self.command('prepare', ok=False).stderr)
        os.utime(capture, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        manifest = self.root / 'iat/scenario/train/valid.jsonl'
        manifest.write_text('\n'.join(manifest.read_text().splitlines()[1:]) + '\n')
        self.assertIn('sample_id', self.command('train', ok=False).stderr)
        metadata_path = self.root / 'cache/scenario/train/metadata.json'
        metadata = json.loads(metadata_path.read_text())
        metadata['valid_count'] = 2
        metadata_path.write_text(json.dumps(metadata))
        self.assertIn('valid count', self.command('prepare', ok=False).stderr)
        metadata_path.unlink()
        self.assertIn('existing compatible base cache', self.command('prepare', ok=False).stderr)

    def test_all_intervals_not_fit_subsample_and_end_packet_uses_last_slot_and_bucket(self):
        self.settings['preprocessing']['splits'] = ['train', 'val']
        self.episode('A', [0, *range(1, 5002), 1000])
        last = self.episode('A', [0, 10, 1000], 'val')
        # Replace the final ARP record with an effective incoming IP packet;
        # preserve a full 1000 ns capture and place its interval at the endpoint.
        with last.open('wb') as handle:
            handle.write(struct.pack('<IHHIIII', 0xa1b23c4d, 2, 4, 0, 0, 65535, 1))
            for time, raw in [(0, ARP), (10, frame('10.0.0.2', '8.8.8.8', 2)),
                              (10, frame('10.0.0.2', '1.1.1.1')),
                              (1000, frame('8.8.8.8', '10.0.0.2'))]:
                handle.write(struct.pack('<IIII', 1700000000, time, len(raw), len(raw)))
                handle.write(raw)
        self.command('prepare')
        self.enable()
        self.command('fit-bins')
        self.command('prepare')
        from agentprint.config import load_config
        from agentprint.v2 import loader
        train, _ = loader(load_config(self.config), 'train')
        self.addCleanup(lambda: (setattr(train, '_iat_shard', None), setattr(train, '_shard', None)))
        self.assertEqual(int(train[0][0][2:].sum()), 4999)
        self.assertEqual(json.loads((self.root / 'fit.json').read_text())['statistics']['sampled_positive_intervals'], 4096)
        val, _ = loader(load_config(self.config), 'val')
        self.addCleanup(lambda: (setattr(val, '_iat_shard', None), setattr(val, '_shard', None)))
        self.assertEqual(val[0][0][9, 1, 1799], 1)
        self.assertEqual(int(val[0][0][2:].sum()), 2)

    def test_complete_256_sample_shard_resumes_and_joins_reordered_base_by_sample_id(self):
        self.settings['preprocessing']['splits'] = ['train']
        for index in range(257):
            self.episode(f'A{index:03d}', [0, 10, 11, 1000])
        self.command('prepare')
        from agentprint.config import fingerprint, load_config
        base = self.root / 'cache/scenario/train'
        records = [json.loads(line) for line in (base / 'valid.jsonl').read_text().splitlines()]
        records.reverse()  # original shard row pointers remain intact
        (base / 'valid.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
        metadata = json.loads((base / 'metadata.json').read_text())
        metadata['manifest_id'] = fingerprint(records)
        (base / 'metadata.json').write_text(json.dumps(metadata))
        self.enable()
        self.command('fit-bins')
        damaged = self.root / 'data/train/engine/artifact_processing/A256/network.pcap'
        original, stat = damaged.read_bytes(), damaged.stat()
        changed = bytearray(original)
        changed[-len(ARP)-8:-len(ARP)-4] = struct.pack('<I', 65536)
        damaged.write_bytes(changed)
        os.utime(damaged, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertIn('PCAP record', self.command('prepare', ok=False).stderr)
        iat = self.root / 'iat/scenario/train'
        self.assertFalse((iat / 'metadata.json').exists())
        committed = (iat / 'commits/00000.json').read_bytes()
        self.assertEqual(np.load(iat / 'shards/00000.npy', mmap_mode='r').shape, (256, 8, 2, 1800))
        # Uncommitted arrays never count as a successful shard.
        (iat / 'shards/00001.npy').write_bytes(b'incomplete orphan')
        damaged.write_bytes(original)
        os.utime(damaged, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.command('prepare')
        self.assertEqual((iat / 'commits/00000.json').read_bytes(), committed)
        resumed = (iat / 'metadata.json').read_bytes()
        from agentprint.v2 import loader
        samples, _ = loader(load_config(self.config), 'train')
        self.addCleanup(lambda: (setattr(samples, '_iat_shard', None), setattr(samples, '_shard', None)))
        self.assertTrue(samples.records[0]['episode_id'].endswith('256'))
        self.assertEqual(samples.iat_records[samples.records[0]['sample_id']]['row_index'], 0)
        self.assertTrue(samples.iat_records[samples.records[0]['sample_id']]['shard_relpath'].endswith('00001.npy'))
        self.assertEqual(samples[0][0][9, 0, 19], 1)
        self.enable('clean-iat')
        self.command('prepare')
        clean = self.root / 'clean-iat/scenario/train'
        self.assertEqual((clean / 'metadata.json').read_bytes(), resumed)
        for name in ('00000.npy', '00001.npy'):
            np.testing.assert_array_equal(np.load(iat / 'shards' / name), np.load(clean / 'shards' / name))


if __name__ == '__main__':
    unittest.main()
