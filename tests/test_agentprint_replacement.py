"""Replacement multikernel acceptance at the public v2 model and CLI seams."""
import json
import unittest

import test_agentprint_v2 as v2_tests


class ReplacementCommands(unittest.TestCase):
    setUp = v2_tests.V2Commands.setUp
    command = v2_tests.V2Commands.command

    def test_raw_and_log1p_public_pipeline_saves_and_rebuilds_replacement(self):
        import torch
        from agentprint.v2 import build_model
        self.settings['model'] = {'variant': 'replace'}
        self.command('prepare')
        for transform in ('raw', 'log1p'):
            with self.subTest(transform=transform):
                self.settings['transform'] = transform
                self.command('train')
                runs = list((self.root / 'output').glob('v2-*'))
                run = next(path for path in runs if
                           torch.load(path / 'best.pt', map_location='cpu', weights_only=False)
                           ['config']['transform'] == transform)
                saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
                self.assertEqual(saved['model_definition'], dict(version=1, variant='replace',
                    width=2, dropout_2d=0.05, dropout_1d=0.1))
                self.assertEqual(saved['identity']['model_definition'], saved['model_definition'])
                restored = build_model(saved['model_definition'],
                                       saved['feature_definition']['in_channels'])
                restored.load_state_dict(saved['model'])
                self.assertEqual(saved['parameters'], sum(p.numel() for p in restored.parameters()))
                result = self.command('evaluate', '--checkpoint', str(run / 'best.pt'))
                summary = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(summary['experiment_id'], saved['experiment_id'])
                self.assertEqual(summary['transform'], transform)
                self.assertEqual(summary['support'], [1, 0, 0, 0, 0])
                self.assertTrue(torch.isfinite(torch.tensor(summary['loss'])))
                self.settings['model'] = {'variant': 'single'}
                mismatch = self.command('evaluate', '--checkpoint', str(run / 'best.pt'), ok=False)
                self.assertIn('incompatible checkpoint model', mismatch.stderr)
                self.settings['model'] = {'variant': 'replace'}


class ReplacementModel(unittest.TestCase):
    def test_public_factory_supports_two_and_ten_channels_with_finite_gradients(self):
        import torch
        from agentprint.v2 import build_model
        torch.set_num_threads(1)
        definition = dict(version=1, variant='replace', width=2,
                          dropout_2d=0.05, dropout_1d=0.1)
        for channels in (2, 10):
            with self.subTest(channels=channels):
                model = build_model(definition, in_channels=channels)
                values = torch.randn(2, channels, 2, 1800, requires_grad=True)
                scores = model(values)
                self.assertEqual(tuple(scores.shape), (2, 5))
                loss = torch.nn.functional.cross_entropy(scores, torch.tensor([0, 4]))
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(values.grad).all())
                self.assertGreater(values.grad.abs().sum().item(), 0)
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
