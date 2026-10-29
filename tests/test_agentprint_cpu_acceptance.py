"""Acceptance-helper boundaries; tiny captures are synthetic, never ANT evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from test_agentprint_primitive import annotation, capture
from tools.agentprint_cpu_acceptance import choose, inventory

REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / 'tools/agentprint_cpu_acceptance.py'
ENVIRONMENT_PROBE = REPO / 'tools/agentprint_cpu_environment.py'
SCENARIOS = ('artifact_processing', 'business_services', 'general_knowledge',
             'scientific_evidence', 'security_testing')


class CPUAcceptanceHelper(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'original'
        self.work = self.root / 'acceptance'

    def episode(self, split, scenario, name, count=2):
        path = self.data / split / 'engine' / scenario / name / 'network.pcap'
        capture(path, (0, 1, 4, 5, 8, 9, 12))
        rows = [annotation(index * 4, 'N001' if index % 2 == 0 else 'N047')
                for index in range(count)]
        path.with_name('messages.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        path.with_name('runtime.json').write_text('{"audit": "keep intact"}', encoding='utf-8')
        return path

    def command(self, action, ok=True):
        result = subprocess.run([sys.executable, str(HELPER), action,
                                 '--data-root', str(self.data), '--work-root', str(self.work)],
                                cwd=REPO, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result

    def require_symlinks(self):
        original = self.root / 'link-probe-source'
        original.write_text('probe')
        try:
            (self.root / 'link-probe').symlink_to(original)
        except OSError as exc:
            self.skipTest(f'individual-file symlinks unavailable on this host: {exc}')

    def test_budget_selection_is_deterministic_and_preserves_feasible_five_class_coverage(self):
        for split in ('train', 'val'):
            for scenario in SCENARIOS:
                self.episode(split, scenario, 'small-annotated', count=2)
                self.episode(split, scenario, 'large-annotation-cost', count=30)
        sources = inventory(self.data)
        first = choose(sources)
        self.assertEqual(first, choose(list(reversed(sources))))
        selected, summary, _ = first
        for split, cap in (('train', 64), ('val', 32)):
            self.assertEqual(set(summary[split]['selected_scenarios']), set(SCENARIOS))
            self.assertLessEqual(summary[split]['selected_annotation_rows'], cap)
        self.assertTrue(all(source['split'] in ('train', 'val') for source in selected))

    def test_complete_annotation_budget_limits_episode_count_without_changing_rows(self):
        for split in ('train', 'val'):
            for scenario in SCENARIOS:
                self.episode(split, scenario, 'A', count=20)
        selected, summary, limitations = choose(inventory(self.data))
        self.assertEqual(summary['train']['selected_annotation_rows'], 60)
        self.assertEqual(summary['val']['selected_annotation_rows'], 20)
        self.assertEqual(len(selected), 4)
        self.assertTrue(limitations)
        self.assertTrue(all(len((self.data / source['annotation_relpath']).read_text().splitlines()) == 20
                            for source in selected))

    def read(self, name):
        return json.loads((self.work / name).read_text(encoding='utf-8'))

    def test_environment_probe_public_command_records_import_and_tensor_evidence(self):
        result = subprocess.run([sys.executable, str(ENVIRONMENT_PROBE)], cwd=REPO,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        packages = json.loads(result.stdout)
        self.assertEqual(set(packages), {'torch', 'numpy', 'yaml', 'dpkt', 'sklearn',
                                         'tensorboard', 'scapy.all', 'unittest'})
        for package in packages.values():
            self.assertIn(package['import'], ('ok', 'failed'))
            self.assertIn('version' if package['import'] == 'ok' else 'error', package)
        if packages['torch']['import'] == 'ok':
            self.assertEqual(packages['torch']['tensor_cpu_probe'], 1.0)

    def test_prepare_public_cli_does_not_retain_probe_frameworks_in_parent(self):
        self.require_symlinks()
        for split in ('train', 'val'):
            self.episode(split, SCENARIOS[0], 'A')
        audit = self.root / 'audit'
        audit.mkdir()
        (audit / 'sitecustomize.py').write_text('''import json
import os
import sys

def record_subprocess(event, args):
    if event == 'subprocess.Popen':
        with open(os.environ['ACCEPTANCE_IMPORT_AUDIT'], 'a', encoding='utf-8') as handle:
            handle.write(json.dumps({'pid': os.getpid(), 'argv': args[1],
                                     'retained': sorted(set(sys.modules) &
                                                        {'torch', 'sklearn', 'scapy'})}) + '\\n')

sys.addaudithook(record_subprocess)
''', encoding='utf-8')
        audit_path = self.root / 'subprocess-audit.jsonl'
        env = dict(os.environ, ACCEPTANCE_IMPORT_AUDIT=str(audit_path),
                   PYTHONPATH=os.pathsep.join([str(audit), os.environ.get('PYTHONPATH', '')]))
        result = subprocess.run([sys.executable, str(HELPER), 'prepare',
                                 '--data-root', str(self.data), '--work-root', str(self.work)],
                                cwd=REPO, capture_output=True, text=True, env=env, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        events = [json.loads(line) for line in audit_path.read_text().splitlines()]
        preparations = [row for row in events if row['argv'][1:4] ==
                        ['-m', 'agentprint', 'prepare']]
        self.assertEqual(len(preparations), 2)
        self.assertEqual(len({row['pid'] for row in preparations}), 1)
        self.assertTrue(all(not row['retained'] for row in preparations), preparations)
        probes = [row for row in events if len(row['argv']) > 1 and
                  Path(row['argv'][1]).name == ENVIRONMENT_PROBE.name]
        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0]['pid'], preparations[0]['pid'])
        environment = self.read('environment.json')
        self.assertTrue(all(package['import'] == 'ok'
                            for package in environment['packages'].values()))
        self.assertEqual(environment['packages']['torch']['tensor_cpu_probe'], 1.0)
        self.assertEqual(self.read('summary.json')['status'], 'prepared_not_accepted')

    def test_selection_caps_whole_files_and_ignores_test(self):
        self.require_symlinks()
        for split in ('train', 'val', 'test'):
            for scenario in SCENARIOS:
                for number in range(5):
                    self.episode(split, scenario, f'A{number}')
        self.command('select')
        selection = self.read('selection.json')
        self.assertEqual(selection['seed'], 42)
        self.assertEqual(selection['split_summary']['train']['selected_episodes'], 20)
        self.assertEqual(selection['split_summary']['val']['selected_episodes'], 10)
        self.assertEqual(set(selection['split_summary']), {'train', 'val'})
        self.assertEqual(selection['split_summary']['train']['selected_annotation_rows'], 40)
        self.assertEqual(set(selection['split_summary']['val']['selected_scenarios']), set(SCENARIOS))
        self.assertFalse((self.work / 'selected-data/test').exists())
        for source in selection['sources']:
            self.assertIn(source['split'], ('train', 'val'))
            for file in source['files']:
                original = self.data / file['relpath']
                link = self.work / 'selected-data' / file['relpath']
                self.assertTrue(link.is_symlink())
                self.assertEqual(link.resolve(), original.resolve())
                self.assertEqual(link.read_bytes(), original.read_bytes())
        # Selecting never overwrites an existing audit workspace.
        self.command('select', ok=False)

    def test_annotation_budget_does_not_truncate_boundaries(self):
        self.require_symlinks()
        for split in ('train', 'val'):
            for scenario in SCENARIOS:
                self.episode(split, scenario, 'A', count=20)
        self.command('select')
        selection = self.read('selection.json')
        self.assertEqual(selection['split_summary']['train']['selected_annotation_rows'], 60)
        self.assertEqual(selection['split_summary']['val']['selected_annotation_rows'], 20)
        self.assertEqual(len(selection['sources']), 4)
        for source in selection['sources']:
            path = self.work / 'selected-data' / source['annotation_relpath']
            self.assertEqual(len(path.read_text().splitlines()), 20)
        self.assertTrue(selection['limitations'])

    def test_prepare_public_cli_verifies_full_uint64_provenance_and_detects_tamper(self):
        self.require_symlinks()
        for split in ('train', 'val'):
            for scenario in SCENARIOS:
                self.episode(split, scenario, 'A')
        self.command('prepare')
        report = self.read('cache-verification.json')
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['datasets']['scenario/train']['valid_count'], 5)
        self.assertEqual(report['datasets']['primitive/val']['valid_count'], 10)
        manifest = self.work / 'cache/primitive/val/valid.jsonl'
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        self.assertEqual(rows[0]['start_ns'], 1700000000000000000)
        self.assertEqual(rows[0]['end_ns'], 1700000000000000004)
        values = np.load(self.work / 'cache' / rows[0]['shard_relpath'], allow_pickle=False)
        self.assertEqual(values.dtype, np.uint64)
        self.assertGreater(int(values.sum()), 0)
        commands = [json.loads(line) for line in (self.work / 'commands.jsonl').read_text().splitlines()]
        self.assertEqual(len(commands), 2)
        self.assertTrue(all(row['argv'][1:4] == ['-m', 'agentprint', 'prepare'] for row in commands))
        self.assertTrue(all(row['returncode'] == 0 and row['elapsed_seconds'] > 0 for row in commands))
        self.assertEqual(self.read('summary.json')['status'], 'prepared_not_accepted')
        manifest.write_text(manifest.read_text().replace('"row_index": 0', '"row_index": 999'))
        self.command('verify-cache', ok=False)
        self.assertEqual(self.read('summary.json')['status'], 'incomplete')

    def test_zero_valid_primitive_reports_incomplete_without_training(self):
        self.require_symlinks()
        for split in ('train', 'val'):
            self.episode(split, SCENARIOS[0], 'A', count=0)
        self.command('prepare', ok=False)
        self.assertEqual(self.read('summary.json')['status'], 'incomplete')
        self.assertIn('no valid samples', self.read('summary.json')['reason'])
        self.assertFalse((self.work / 'runs').exists())


if __name__ == '__main__':
    unittest.main()
