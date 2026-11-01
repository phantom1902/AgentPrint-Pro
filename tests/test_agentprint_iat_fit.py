"""Public fit-bins contract against hand-computed, synthetic nanosecond PCAPs."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import yaml

from test_agentprint import REPO, frame


ARP = bytes.fromhex('ffffffffffff0200000000010806') + bytes(28)


def nano_capture(path, times):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        handle.write(struct.pack('<IHHIIII', 0xa1b23c4d, 2, 4, 0, 0, 65535, 1))
        for index, time in enumerate(times):
            raw = (ARP if time in (0, 1000) else
                   frame('10.0.0.2', '8.8.8.8', 2) if index == 1 else
                   frame('1.1.1.1', '10.0.0.2') if index % 2 else
                   frame('10.0.0.2', '1.1.1.1'))
            handle.write(struct.pack('<IIII', 1700000000, time, len(raw), len(raw)))
            handle.write(raw)


class FitCommands(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'config.yaml'
        self.settings = {
            'extends': str(REPO / 'configs/agentprint/common-v2.yaml'),
            'mode': 'smoke',
            'paths': {name + '_root': str(self.root / name)
                      for name in ('data', 'cache', 'output')},
            'preprocessing': {'splits': ['train']},
        }
        self.artifact = self.root / 'fit.json'

    def episode(self, name, times):
        path = self.root / 'data/train/engine/artifact_processing' / name / 'network.pcap'
        nano_capture(path, times)
        return path

    def command(self, name, ok=True):
        self.config.write_text(yaml.safe_dump(self.settings), encoding='utf-8')
        result = subprocess.run([sys.executable, '-m', 'agentprint', name,
                                 '--config', str(self.config)], cwd=REPO,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result

    def enable(self):
        self.settings['feature'] = dict(iat_enabled=True,
                                       fit_artifact=str(self.artifact),
                                       iat_cache_root=str(self.root / 'iat'))

    def test_public_fit_uses_train_episode_weights_and_stable_integer_time(self):
        self.episode('A', [0, 10, 14, 11, 14, 19, 26, 35, 46, 59, 1000])
        self.episode('B', [0, 10, 110, 1000])
        self.episode('C', [0, 10, 10, 1000])
        self.command('prepare')
        # Neither discovery nor validation/test decoding belongs to fit-bins.
        (self.root / 'data/test').mkdir()
        (self.root / 'data/test/network.pcap').write_bytes(b'damaged and forbidden')
        self.enable()
        self.command('fit-bins')
        fit = json.loads(self.artifact.read_text())
        self.assertEqual(fit['boundaries_ns'], [3, 7, 11, 100, 100, 100])
        self.assertEqual(fit['scope'], 'smoke')
        self.assertEqual(fit['statistics']['participating_episodes'], 2)
        self.assertEqual(fit['statistics']['no_positive_episodes'], 1)
        self.assertEqual(fit['statistics']['zero_intervals'], 2)
        self.assertEqual(fit['statistics']['timestamp_inversions'], 1)
        self.assertEqual(fit['statistics']['positive_intervals'], 8)
        self.assertEqual(fit['statistics']['empty_positive_buckets'], [5, 6])
        self.assertTrue(all(row['split'] == 'train' for row in fit['train_manifest']))
        before = self.artifact.read_bytes()
        index_path = self.root / 'cache/index.json'
        index = json.loads(index_path.read_text())
        index['sources'].reverse()
        index['tasks']['scenario']['sources'].reverse()
        index_path.write_text(json.dumps(index))
        self.settings.update(seed=7, transform='log1p', model={'variant': 'prepend'})
        self.command('fit-bins')
        self.assertEqual(self.artifact.read_bytes(), before)

    def test_episode_commits_resume_after_damaged_capture_without_prefix_success(self):
        self.episode('A', [0, *range(1, 5001), 1000])
        damaged = self.episode('B', [0, 10, 110, 1000])
        self.command('prepare')
        original, stat = damaged.read_bytes(), damaged.stat()
        # Keep the documented PCAP identity unchanged while simulating unreadable
        # storage after the first Episode has committed. No parser mocks.
        changed = bytearray(original)
        changed[-len(ARP)-8:-len(ARP)-4] = struct.pack('<I', 65536)
        damaged.write_bytes(changed)
        os.utime(damaged, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.enable()
        failure = self.command('fit-bins', ok=False)
        self.assertIn('PCAP record', failure.stderr)
        self.assertFalse(self.artifact.exists())
        commits = list(self.root.glob('fit.json.progress/episodes/*.json'))
        self.assertEqual(len(commits), 1)
        committed = commits[0].read_bytes()
        self.assertEqual(len(json.loads(committed)['sampled_positive_ns']), 4096)
        damaged.write_bytes(original)
        os.utime(damaged, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.command('fit-bins')
        resumed = self.artifact.read_bytes()
        self.assertEqual(commits[0].read_bytes(), committed)
        self.artifact = self.root / 'clean.json'
        self.settings.update(seed=99, transform='log1p', model={'variant': 'replace'})
        self.enable()
        self.command('fit-bins')
        self.assertEqual(self.artifact.read_bytes(), resumed)

    def test_all_zero_fit_fails_without_changing_base_classification_eligibility(self):
        self.episode('A', [0, 10, 10, 1000])
        self.command('prepare')
        base = self.root / 'cache/scenario/train/valid.jsonl'
        before = base.read_bytes()
        self.enable()
        self.assertIn('no positive intervals', self.command('fit-bins', ok=False).stderr)
        self.assertFalse(self.artifact.exists())
        self.assertEqual(base.read_bytes(), before)

    def test_changed_source_or_scope_cannot_reuse_fit_progress(self):
        capture = self.episode('A', [0, 10, 11, 20, 1000])
        self.command('prepare')
        self.enable()
        self.command('fit-bins')
        self.settings['mode'] = 'formal'
        self.assertIn('progress source/rules identity', self.command('fit-bins', ok=False).stderr)
        self.settings['mode'] = 'smoke'
        capture.with_name('runtime.json').write_text('{}')
        self.assertIn('endpoint metadata', self.command('fit-bins', ok=False).stderr)

    def test_v1_and_disabled_iat_reject_fit_command(self):
        self.assertIn('enabled IAT', self.command('fit-bins', ok=False).stderr)
        self.settings['extends'] = str(REPO / 'configs/agentprint/common.yaml')
        self.assertIn('schema_version=2', self.command('fit-bins', ok=False).stderr)

    def test_duplicate_quantiles_report_all_empty_positive_sample_buckets(self):
        self.episode('A', [0, 10, 11, 1000])
        self.command('prepare')
        self.enable()
        self.command('fit-bins')
        fit = json.loads(self.artifact.read_text())
        self.assertEqual(fit['boundaries_ns'], [1, 1, 1, 1, 1, 1])
        self.assertEqual(fit['statistics']['empty_positive_buckets'], [1, 2, 3, 4, 5, 6])

    def test_corrupt_committed_samples_are_rejected(self):
        self.episode('A', [0, 10, 11, 20, 1000])
        self.command('prepare')
        self.enable()
        self.command('fit-bins')
        commit = next(self.root.glob('fit.json.progress/episodes/*.json'))
        row = json.loads(commit.read_text())
        row['sampled_positive_ns'][0] = 2
        commit.write_text(json.dumps(row))
        self.assertIn('Episode commit', self.command('fit-bins', ok=False).stderr)

    def test_selected_base_view_does_not_read_other_training_sources(self):
        from agentprint.config import fingerprint
        self.episode('A', [0, 10, 11, 20, 1000])
        unselected = self.episode('B', [0, 10, 110, 1000])
        self.command('prepare')
        base = self.root / 'cache/scenario/train'
        rows = [json.loads(line) for line in (base / 'valid.jsonl').read_text().splitlines()]
        selected = [row for row in rows if row['episode_id'] == 'A']
        (base / 'valid.jsonl').write_text(json.dumps(selected[0]) + '\n')
        metadata = json.loads((base / 'metadata.json').read_text())
        metadata.update(manifest_id=fingerprint(selected), valid_count=1, support=[1, 0, 0, 0, 0])
        (base / 'metadata.json').write_text(json.dumps(metadata))
        # Original source index and MTAM arrays remain intact. The missing
        # unselected capture must not be inspected by fit-bins.
        unselected.unlink()
        self.enable()
        self.command('fit-bins')
        fit = json.loads(self.artifact.read_text())
        self.assertEqual(len(fit['sources']), 1)
        self.assertEqual(fit['train_manifest'][0]['episode_id'], 'A')


if __name__ == '__main__':
    unittest.main()
