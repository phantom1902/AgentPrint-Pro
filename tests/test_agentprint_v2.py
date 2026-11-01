"""v2 tracer acceptance through public commands and saved artifacts."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

from test_agentprint import REPO, capture


class V2Commands(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'v2.yaml'
        self.settings = {
            'extends': str(REPO / 'configs/agentprint/common-v2.yaml'),
            'mode': 'smoke',
            'paths': {name + '_root': str(self.root / name)
                      for name in ('data', 'cache', 'output')},
            'preprocessing': {'splits': ['train', 'val']},
            'training': {'epochs': 1, 'batch_size': 2},
        }
        for split in ('train', 'val'):
            capture(self.root / 'data' / split / 'engine' /
                    'artifact_processing/A001/network.pcap')

    def command(self, command, *args, ok=True):
        self.config.write_text(yaml.safe_dump(self.settings), encoding='utf-8')
        result = subprocess.run([sys.executable, '-m', 'agentprint', command,
                                 '--config', str(self.config), *args], cwd=REPO,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result

    def test_disabled_iat_single_model_public_pipeline_saves_rebuildable_identity(self):
        import torch
        self.command('prepare')
        self.command('train')
        run = next((self.root / 'output').iterdir())
        saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
        self.assertEqual(saved['checkpoint_format'], 2)
        self.assertEqual(saved['model_definition']['width'], 2)
        self.assertEqual(saved['feature_definition']['in_channels'], 2)
        self.assertFalse(saved['feature_definition']['iat_enabled'])
        self.assertIn('base_cache', saved['identity'])
        self.assertEqual(saved['config']['training']['learning_rate'], 0.001)
        row = json.loads((run / 'metrics.jsonl').read_text())
        self.assertEqual(row['learning_rate'], 0.001)
        result = self.command('evaluate', '--checkpoint', str(run / 'best.pt'))
        summary = json.loads(result.stdout.splitlines()[-1])
        self.assertEqual(summary['support'], [1, 0, 0, 0, 0])
        self.assertEqual(summary['experiment_id'], saved['experiment_id'])
        self.assertTrue((run / 'evaluation-val/summary.json').is_file())

    def test_log1p_pipeline_reuses_base_cache_and_rejects_changed_model_identity(self):
        self.command('prepare')
        metadata = self.root / 'cache/scenario/train/metadata.json'
        before = metadata.read_bytes()
        self.settings['transform'] = 'log1p'
        self.command('prepare')
        self.assertEqual(before, metadata.read_bytes())
        self.command('train')
        run = next((self.root / 'output').iterdir())
        self.command('evaluate', '--checkpoint', str(run / 'best.pt'))
        self.settings['model'] = {'width': 1.0}
        result = self.command('evaluate', '--checkpoint', str(run / 'best.pt'), ok=False)
        self.assertIn('incompatible checkpoint model', result.stderr)

    def test_invalid_variant_pending_iat_and_v1_reject_v2_fields(self):
        self.settings['model'] = {'variant': 'replace-and-prepend'}
        self.assertIn('unknown model.variant', self.command('prepare', ok=False).stderr)
        self.settings.pop('model')
        self.settings['feature'] = {'iat_enabled': True,
                                    'fit_artifact': str(self.root / 'bins.json'),
                                    'iat_cache_root': str(self.root / 'iat')}
        self.assertIn('fit artifact is missing', self.command('prepare', ok=False).stderr)
        self.settings['schema_version'] = 1
        self.assertIn('unknown configuration field', self.command('prepare', ok=False).stderr)

    def test_evaluation_rejects_checkpoint_identity_that_no_longer_matches_saved_definition(self):
        import torch
        self.command('prepare')
        self.command('train')
        run = next((self.root / 'output').iterdir())
        saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
        saved['identity']['fit_fingerprint'] = 'foreign-fit'
        torch.save(saved, run / 'changed.pt')
        result = self.command('evaluate', '--checkpoint', str(run / 'changed.pt'), ok=False)
        self.assertIn('checkpoint identity', result.stderr)


class V2Model(unittest.TestCase):
    def test_public_single_factory_two_and_ten_channels_and_original_parity(self):
        import torch
        from agentprint.model import getRF
        from agentprint.v2 import build_model
        torch.set_num_threads(1)
        original_definition = dict(version=1, variant='single', width=1,
                                   dropout_2d=0.1, dropout_1d=0.3)
        torch.manual_seed(42)
        original = getRF(5).eval()
        torch.manual_seed(42)
        compatible = build_model(original_definition, in_channels=2).eval()
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, compatible.state_dict()[key]), key)
        values = torch.randn(2, 2, 2, 1800)
        self.assertTrue(torch.equal(original(values), compatible(values)))
        for channels in (2, 10):
            model = build_model(dict(original_definition, width=2,
                                     dropout_2d=0.05, dropout_1d=0.1), channels)
            scores = model(torch.randn(2, channels, 2, 1800))
            self.assertEqual(tuple(scores.shape), (2, 5))
            loss = torch.nn.functional.cross_entropy(scores, torch.tensor([0, 4]))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))

    def test_public_model_rejects_unknown_definition_fields(self):
        from agentprint.v2 import build_model
        with self.assertRaisesRegex(ValueError, 'unknown.*model'):
            build_model(dict(version=1, variant='single', width=2,
                             dropout_2d=0.05, dropout_1d=0.1, typo=True), 2)


if __name__ == '__main__':
    unittest.main()
