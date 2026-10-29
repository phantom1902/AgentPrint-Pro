"""Full delivery verification through prepared artifacts and the public utility."""
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

from test_agentprint_primitive import annotation, capture


REPO = Path(__file__).resolve().parents[1]
LABELS = ('artifact_processing', 'business_services', 'general_knowledge',
          'scientific_evidence', 'security_testing')


class FullCacheReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.fixture.cleanup)
        cls.prepared = Path(cls.fixture.name)
        for split in ('train', 'val', 'test'):
            for label in LABELS if split == 'train' else LABELS[:1]:
                base = cls.prepared / 'data' / split / 'engine' / label / 'good'
                capture(base / 'network.pcap')
                if label == LABELS[0]:
                    (base / 'messages.jsonl').write_text(
                        json.dumps(annotation(0, 'N001')) + '\n' +
                        json.dumps(annotation(8, 'UNKNOWN')) + '\n', encoding='utf-8')
        for name, offsets in (('direction', (0, 1)), ('empty', ())):
            base = cls.prepared / 'data/train/engine/artifact_processing' / name
            capture(base / 'network.pcap', offsets)
            (base / 'messages.jsonl').write_text(
                json.dumps(annotation(0, 'N001')) + '\n', encoding='utf-8')
        settings = {'extends': str(REPO / 'configs/agentprint/common.yaml'),
                    'paths': {'data_root': str(cls.prepared / 'data'),
                              'cache_root': str(cls.prepared / 'cache'),
                              'output_root': str(cls.prepared / 'runs')}}
        for task in ('scenario', 'primitive'):
            settings['task'] = task
            config = cls.prepared / f'{task}.yaml'
            config.write_text(yaml.safe_dump(settings), encoding='utf-8')
            result = subprocess.run([sys.executable, '-m', 'agentprint', 'prepare',
                                     '--config', str(config)], cwd=REPO,
                                    text=True, capture_output=True)
            assert result.returncode == 0, result.stdout + result.stderr

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('cache', 'data'):
            shutil.copytree(self.prepared / name, self.root / name)
        self.output = self.root / 'coverage.json'

    def run_report(self, ok=True, *extra):
        result = subprocess.run([sys.executable, '-m', 'agentprint.fullcache',
                                 '--cache-root', str(self.root / 'cache'),
                                 '--data-root', str(self.root / 'data'),
                                 '--output', str(self.output), *extra], cwd=REPO,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        self.assertTrue(self.output.is_file(), result.stdout + result.stderr)
        report = json.loads(self.output.read_text(encoding='utf-8'))
        self.assertEqual(report['verified'], ok)
        return report

    def edit_json(self, relative, edit):
        path = self.root / 'cache' / relative
        value = json.loads(path.read_text(encoding='utf-8'))
        edit(value)
        path.write_text(json.dumps(value), encoding='utf-8')

    def test_two_tasks_all_splits_quality_and_independent_training_gates(self):
        report = self.run_report()
        self.assertEqual(report['source_inventory']['capture_count'], 9)
        self.assertEqual(report['source_inventory']['annotation_count'], 5)
        self.assertEqual(report['datasets']['scenario/train']['valid'], 6)
        self.assertEqual(report['datasets']['scenario/train']['excluded'], 1)
        primitive = report['datasets']['primitive/train']
        self.assertEqual(primitive['excluded_reasons']['missing_annotations'], 4)
        self.assertEqual(primitive['classes']['__unknown__']['excluded'], 5)
        self.assertEqual(primitive['classes']['__unknown__']['excluded_labels'],
                         {'UNKNOWN': 1, '__unavailable__': 4})
        self.assertEqual(primitive['classes']['N001']['support'], 2)
        self.assertEqual(primitive['classes']['N001']['coverage'], 2 / 3)
        self.assertEqual(report['datasets']['scenario/train']['excluded_reasons'],
                         {'empty_capture': 1})
        self.assertTrue(report['formal_training']['scenario']['eligible'])
        self.assertFalse(report['formal_training']['primitive']['eligible'])
        self.assertEqual(report['formal_training']['primitive']['missing_train_classes'],
                         [f'N{index:03d}' for index in range(2, 48)])
        self.assertEqual(set(report['datasets']),
                         {f'{task}/{split}' for task in ('scenario', 'primitive')
                          for split in ('train', 'val', 'test')})
        self.assertEqual(report['verification_scope'], 'metadata_annotations_and_cached_arrays_no_pcap_payload')
        self.assertGreater(report['storage']['array_payload_bytes'], 0)

    def test_refuses_incomplete_task_or_split_even_if_root_marked_complete(self):
        self.edit_json('index.json', lambda value: value['tasks']['primitive'].update(complete=False))
        report = self.run_report(False)
        self.assertIn('incomplete', ' '.join(report['errors']))

    def rewrite_train_valid(self, edit):
        """Coherently republish row metadata to exercise semantic checks beyond hashes."""
        base = self.root / 'cache/scenario/train'
        rows = [json.loads(line) for line in (base / 'valid.jsonl').read_text().splitlines()]
        edit(rows)
        (base / 'valid.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        self.edit_json('scenario/train/commits/00000.json', lambda c: c.update(valid=rows))
        manifest_id = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.edit_json('scenario/train/metadata.json', lambda m: m.update(manifest_id=manifest_id))
        metadata = json.loads((base / 'metadata.json').read_text())
        self.edit_json('index.json', lambda i: i['datasets'].update({'scenario/train': metadata}))

    def test_rejects_coherent_metadata_tampering_of_endpoints_time_and_references(self):
        changes = (
            ('endpoint', lambda rows: rows[0]['endpoint_source'].update(kind='guessed')),
            ('interval', lambda rows: rows[0].update(duration_ns=0)),
            ('row', lambda rows: rows[1].update(row_index=rows[0]['row_index'])),
            ('provenance', lambda rows: rows[0].update(annotation_row=99)),
        )
        for message, edit in changes:
            with self.subTest(message=message):
                shutil.rmtree(self.root / 'cache')
                shutil.copytree(self.prepared / 'cache', self.root / 'cache')
                self.rewrite_train_valid(edit)
                self.assertIn(message, ' '.join(self.run_report(False)['errors']))

    def test_rejects_array_dtype_shape_and_trailing_bytes(self):
        path = self.root / 'cache/scenario/train/shards/00000.npy'
        original = np.load(path)
        for values in (original.astype(np.float32), original[:, :, :, :-1]):
            np.save(path, values)
            self.assertIn('dtype/shape', ' '.join(self.run_report(False)['errors']))
        np.save(path, original)
        with path.open('ab') as handle:
            handle.write(b'junk')
        self.assertIn('byte count', ' '.join(self.run_report(False)['errors']))

    def test_detects_unreferenced_shards_and_array_payload_tamper(self):
        shard = self.root / 'cache/scenario/train/shards/00000.npy'
        values = np.load(shard)
        values[0, 0, 0, 0] += 1
        np.save(shard, values)
        self.assertIn('packet_count', ' '.join(self.run_report(False)['errors']))
        shutil.copy2(self.prepared / 'cache/scenario/train/shards/00000.npy', shard)
        shutil.copy2(shard, shard.with_name('orphan.npy'))
        self.assertIn('shard', ' '.join(self.run_report(False)['errors']))

    def test_detects_manifest_omission_and_duplicate_references(self):
        path = self.root / 'cache/scenario/train/valid.jsonl'
        lines = path.read_text().splitlines()
        path.write_text('\n'.join(lines[1:]) + '\n')
        self.assertIn('manifest', ' '.join(self.run_report(False)['errors']))
        path.write_text('\n'.join([lines[0], *lines]) + '\n')
        self.assertIn('manifest', ' '.join(self.run_report(False)['errors']))

    def test_detects_annotation_identity_change_and_new_source(self):
        path = self.root / 'data/train/engine/artifact_processing/good/messages.jsonl'
        path.write_text(path.read_text().replace('N001', 'N002'))
        self.assertIn('source inventory', ' '.join(self.run_report(False)['errors']))
        shutil.copy2(self.prepared / 'data/train/engine/artifact_processing/good/messages.jsonl', path)
        capture(self.root / 'data/test/engine/artifact_processing/added/network.pcap')
        self.assertIn('source inventory', ' '.join(self.run_report(False)['errors']))

    def test_runtime_record_is_separate_and_running_is_refused(self):
        runtime = self.root / 'execution.json'
        runtime.write_text(json.dumps({'status': 'running', 'git_commit': 'prepared-code'}))
        report = self.run_report(False, '--runtime-record', str(runtime))
        self.assertIn('runtime', ' '.join(report['errors']))
        value = {'status': 'prepared_pending_verification', 'git_commit': 'prepared-code',
                 'commands': [{'task': task, 'returncode': 0, 'elapsed_seconds': 1}
                              for task in ('scenario', 'primitive')]}
        runtime.write_text(json.dumps(value))
        report = self.run_report(True, '--runtime-record', str(runtime))
        self.assertEqual(report['preparation_execution']['record'], value)


if __name__ == '__main__':
    unittest.main()
