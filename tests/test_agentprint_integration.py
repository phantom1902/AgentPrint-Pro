"""Cross-task preparation/recovery acceptance through CLI and cache artifacts."""
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

from test_agentprint_primitive import BASE_NS, annotation, capture


REPO = Path(__file__).resolve().parents[1]


class PreparationIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'config.yaml'
        self.settings = {
            'extends': str(REPO / 'configs/agentprint/common.yaml'),
            'task': 'primitive', 'transform': 'raw', 'mode': 'smoke',
            'paths': {'data_root': str(self.root / 'data'),
                      'cache_root': str(self.root / 'cache'),
                      'output_root': str(self.root / 'runs')},
            'preprocessing': {'splits': ['train']}}
        self.save_config()

    def save_config(self):
        self.config.write_text(yaml.safe_dump(self.settings), encoding='utf-8')

    def cli(self, *args):
        return [sys.executable, '-m', 'agentprint', 'prepare',
                '--config', str(self.config), *args]

    def command(self, *args, env=None):
        result = subprocess.run(self.cli(*args), cwd=REPO, env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def episode(self, name, rows, offsets):
        path = self.root / 'data/train/engine/artifact_processing' / name / 'network.pcap'
        capture(path, offsets)
        if rows is not None:
            path.with_name('messages.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        return path

    def records(self, name, task='primitive', cache='cache'):
        return [json.loads(line) for line in
                (self.root / cache / task / 'train' / name).read_text().splitlines()]

    def audit_environment(self, name, code):
        observer = self.root / name
        observer.mkdir()
        (observer / 'sitecustomize.py').write_text(code, encoding='utf-8')
        env = os.environ.copy()
        env['PYTHONPATH'] = str(observer) + os.pathsep + env.get('PYTHONPATH', '')
        return env

    def interrupt_after_first_shard(self, next_source):
        commit = self.root / 'cache/primitive/train/commits/00000.json'
        marker = self.root / 'next-source-opened'
        # Pause the *test subprocess* at its next-source file-open audit event.
        # stdin stays open without data until the parent terminates it. The
        # marker therefore has no short polling window, even on slow machines.
        env = self.audit_environment('interruption-observer',
            "import os, sys\n"
            "def observe(event, args):\n"
            "    if event == 'open' and isinstance(args[0], (str, bytes)):\n"
            f"        if os.path.abspath(os.fsdecode(args[0])) == {str(next_source)!r}:\n"
            f"            if os.path.isfile({str(commit)!r}):\n"
            f"                with open({str(marker)!r}, 'w') as handle:\n"
            "                    handle.write('ready')\n"
            "                sys.stdin.buffer.read(1)\n"
            "sys.addaudithook(observe)\n")
        process = subprocess.Popen(self.cli(), cwd=REPO, env=env, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.monotonic() + 30
        try:
            while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.005)
        finally:
            if process.poll() is None:
                process.terminate()
            stdout, stderr = process.communicate(timeout=10)
        self.assertTrue(marker.exists(), (stdout + stderr).decode())
        self.assertTrue(commit.exists())
        self.assertNotEqual(process.returncode, 0, 'preparation completed before interruption')
        self.assertFalse((self.root / 'cache/primitive/train/metadata.json').exists())
        return json.loads(commit.read_text())

    def observed_pcap_reads(self):
        # Observe actual file opens by the unchanged public CLI. This audit hook
        # only emits paths; it does not patch preparation or control its timing.
        env = self.audit_environment('read-observer',
            "import os, sys\n"
            "def observe(event, args):\n"
            "    if event == 'open' and isinstance(args[0], (str, bytes)):\n"
            "        path = os.fsdecode(args[0])\n"
            "        if path.endswith('network.pcap'):\n"
            "            print('PCAP_OPEN=' + os.path.abspath(path), file=sys.stderr, flush=True)\n"
            "sys.addaudithook(observe)\n")
        result = self.command(env=env)
        return [line.removeprefix('PCAP_OPEN=') for line in result.stderr.splitlines()
                if line.startswith('PCAP_OPEN=')]

    def assert_matches_clean(self):
        self.command('--cache-root', str(self.root / 'clean'))
        recovered = self.root / 'cache/primitive/train'
        clean = self.root / 'clean/primitive/train'
        for name in ('valid.jsonl', 'excluded.jsonl', 'metadata.json'):
            self.assertEqual((recovered / name).read_bytes(), (clean / name).read_bytes(), name)
        expected_paths = sorted((recovered / 'shards').glob('*.npy'))
        actual_paths = sorted((clean / 'shards').glob('*.npy'))
        self.assertEqual(len(expected_paths), len(actual_paths))
        for expected, actual in zip(expected_paths, actual_paths):
            left = np.load(expected, mmap_mode='r', allow_pickle=False)
            right = np.load(actual, mmap_mode='r', allow_pickle=False)
            self.assertEqual(left.dtype, np.uint64)
            self.assertLessEqual(left.shape[0], 1024)
            np.testing.assert_array_equal(left, right)

    def recovery_fixture(self, count):
        first = self.episode('A', [annotation(4 * n, 'N001') for n in range(count)],
                             range(0, 4 * count + 1, 4))
        tail = self.episode('B', [annotation(0, 'N047'), annotation(4, 'G01')],
                            (0, 1, 4, 5, 8))
        self.episode('C', None, (0, 1, 4))
        return first, tail

    def assert_recovered_provenance(self, count):
        valid = self.records('valid.jsonl')
        excluded = self.records('excluded.jsonl')
        self.assertEqual(len(valid), count + 1)
        self.assertEqual(len({row['sample_id'] for row in valid}), count + 1)
        self.assertEqual([row['annotation_row'] for row in valid[:count]], list(range(1, count + 1)))
        self.assertEqual([row['start_ns'] for row in valid[:count]],
                         [BASE_NS + 4 * n for n in range(count)])
        self.assertEqual(valid[-1]['label_id'], 46)
        self.assertEqual([(row['episode_id'], row['annotation_row'], row['reason'])
                          for row in excluded],
                         [('B', 2, 'invalid_primitive_label'), ('C', None, 'missing_annotations')])
        self.assertTrue(all('row_index' not in row for row in excluded))

    def test_completed_1024_row_primitive_source_is_skipped_after_interruption(self):
        first, tail = self.recovery_fixture(1024)
        commit = self.interrupt_after_first_shard(tail)
        self.assertEqual(commit['cursor'], {'source_index': 1, 'sample_offset': 0})
        shard = self.root / 'cache/primitive/train/shards/00000.npy'
        before = shard.stat().st_mtime_ns
        reads = self.observed_pcap_reads()
        self.assertNotIn(str(first), reads)
        self.assertIn(str(tail), reads)
        self.assertEqual(before, shard.stat().st_mtime_ns)
        self.assert_recovered_provenance(1024)
        self.assert_matches_clean()

    def test_primitive_source_spanning_shards_replays_only_current_source(self):
        first, tail = self.recovery_fixture(1026)
        commit = self.interrupt_after_first_shard(tail)
        self.assertEqual(commit['cursor'], {'source_index': 0, 'sample_offset': 1024})
        shard = self.root / 'cache/primitive/train/shards/00000.npy'
        before = shard.stat().st_mtime_ns
        reads = self.observed_pcap_reads()
        self.assertIn(str(first), reads)
        self.assertIn(str(tail), reads)
        self.assertEqual(before, shard.stat().st_mtime_ns)
        self.assert_recovered_provenance(1026)
        self.assert_matches_clean()

    def test_scenario_and_primitive_share_root_and_raw_log1p_reuse_each_task(self):
        self.episode('A', [annotation(0, 'N001'), annotation(4, 'N047')], (0, 1, 4, 5, 8))
        original = {}
        for task in ('scenario', 'primitive'):
            self.settings.update(task=task, transform='raw')
            self.save_config()
            self.command()
            base = self.root / 'cache' / task / 'train'
            original[task] = ((base / 'valid.jsonl').read_bytes(),
                              (base / 'shards/00000.npy').stat().st_mtime_ns)
        for task in ('scenario', 'primitive'):
            for transform in ('raw', 'log1p'):
                self.settings.update(task=task, transform=transform)
                self.save_config()
                self.command()
                base = self.root / 'cache' / task / 'train'
                self.assertEqual(original[task], ((base / 'valid.jsonl').read_bytes(),
                                                 (base / 'shards/00000.npy').stat().st_mtime_ns))
        index = json.loads((self.root / 'cache/index.json').read_text())
        self.assertEqual(set(index['tasks']), {'scenario', 'primitive'})
        self.assertEqual(set(index['datasets']), {'scenario/train', 'primitive/train'})
        self.assertTrue(index['complete'])
        self.assertTrue(all(task['complete'] for task in index['tasks'].values()))
        self.assertEqual(index['datasets']['scenario/train']['valid_count'], 1)
        self.assertEqual(index['datasets']['primitive/train']['valid_count'], 2)


if __name__ == '__main__':
    unittest.main()
