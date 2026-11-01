from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
import yaml

from agentprint.__main__ import main as command
from test_agentprint import REPO
from test_agentprint_iat_fit import nano_capture


class IATSmokeSelection(unittest.TestCase):
    def test_public_selected_view_obeys_budget_order_and_never_needs_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, cache = root / 'data', root / 'cache'
            classes = ['artifact_processing', 'business_services', 'general_knowledge',
                       'scientific_evidence', 'security_testing']
            for split, count in [('train', 6), ('val', 3)]:
                for label in classes:
                    for episode in range(count):
                        # e00 is larger; ties for the others resolve by sample_id.
                        times = [0, 10, 20, 30, 1000] if episode == 0 else [0, 10, 20, 1000]
                        nano_capture(data / split / 'engine' / label / f'e{episode:02d}' / 'network.pcap', times)
            config = root / 'base.yaml'
            config.write_text(yaml.safe_dump(dict(extends=str(REPO / 'configs/agentprint/common-v2.yaml'),
                paths=dict(data_root=str(data), cache_root=str(cache), output_root=str(root / 'unused')))),
                encoding='utf-8')
            with redirect_stdout(io.StringIO()):
                self.assertEqual(command(['prepare', '--config', str(config)]), 0)
            # A test cache must never be required or opened by the acceptance entry.
            forbidden = cache / 'scenario/test'
            forbidden.mkdir()
            (forbidden / 'valid.jsonl').write_text('not JSON')
            from agentprint.iat_smoke import main
            work = root / 'acceptance'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--config', str(config), '--workspace-root', str(root),
                                       '--work-root', str(work), '--select-only']), 0)
            selection = json.loads((work / 'selection.json').read_text())
            self.assertEqual(selection['status'], 'selected_only')
            self.assertEqual(len(selection['git_revision']), 40)
            for split, expected_count, expected_ids in [('train', 20, ['e01', 'e02', 'e03', 'e04']),
                                                       ('val', 10, ['e01', 'e02'])]:
                rows = selection['selected'][split]
                self.assertEqual(len(rows), expected_count)
                for label in classes:
                    self.assertEqual([r['episode_id'] for r in rows if r['label_name'] == label], expected_ids)
                metadata = json.loads((work / 'base-view/scenario' / split / 'metadata.json').read_text())
                self.assertEqual(metadata['support'], [expected_count // 5] * 5)
                for row in rows:
                    link = work / 'base-view' / row['shard_relpath']
                    self.assertTrue(link.samefile(cache / row['shard_relpath']))
            self.assertFalse((work / 'base-view/scenario/test').exists())
            configs = list((work / 'configs').glob('*.yaml'))
            self.assertEqual(len(configs), 6)
            for path in configs:
                settings = yaml.safe_load(path.read_text())
                self.assertTrue(settings['feature']['iat_enabled'])
                self.assertEqual(settings['training']['epochs'], 2)
                self.assertEqual(settings['training']['batch_size'], 200)
                self.assertEqual(settings['seed'], 42)
            with redirect_stdout(io.StringIO()):
                # Reusing an existing work root is an explicit failure, no overwrite.
                self.assertEqual(main(['--config', str(config), '--workspace-root', str(root),
                                       '--work-root', str(work), '--select-only']), 1)

    def test_public_helper_runs_only_six_two_epoch_smoke_configs_and_reloads(self):
        from agentprint.iat_smoke import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ('train', 'val'):
                for label in ('artifact_processing', 'business_services', 'general_knowledge',
                              'scientific_evidence', 'security_testing'):
                    nano_capture(root / 'data' / split / 'engine' / label / 'A/network.pcap',
                                 [0, 10, 20, 40, 1000])
            config = root / 'base.yaml'
            config.write_text(yaml.safe_dump(dict(extends=str(REPO / 'configs/agentprint/common-v2.yaml'),
                paths=dict(data_root=str(root / 'data'), cache_root=str(root / 'cache'),
                           output_root=str(root / 'unused')))), encoding='utf-8')
            with redirect_stdout(io.StringIO()):
                self.assertEqual(command(['prepare', '--config', str(config)]), 0)
                self.assertEqual(main(['--config', str(config), '--workspace-root', str(root),
                                       '--work-root', str(root / 'smoke')]), 0)
            summary = json.loads((root / 'smoke/summary.json').read_text())
            self.assertEqual(summary['status'], 'passed')
            self.assertFalse(summary['uses_test'])
            self.assertFalse(summary['formal_training'])
            self.assertFalse(summary['full_preprocessing'])
            self.assertEqual({r['name'] for r in summary['configurations']},
                             {'single-raw', 'single-log1p', 'replace-raw', 'replace-log1p',
                              'prepend-raw', 'prepend-log1p'})
            for result in summary['configurations']:
                self.assertEqual(result['result']['epochs'], 2)
                self.assertFalse(result['result']['converged'])
                self.assertTrue(result['resume_unchanged'])
                self.assertEqual(result['validation']['split'], 'val')
                identity = json.loads((Path(result['run_dir']) / 'identity.json').read_text())['identity']
                self.assertEqual(identity['fit_fingerprint'], summary['fit_fingerprint'])
                self.assertEqual(identity['feature_definition']['in_channels'], 10)


if __name__ == '__main__':
    unittest.main()
