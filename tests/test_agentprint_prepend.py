"""Prepend multikernel acceptance at the public v2 model and CLI seams."""
import json
import unittest

import test_agentprint_v2 as v2_tests


class PrependCommands(unittest.TestCase):
    setUp = v2_tests.V2Commands.setUp
    command = v2_tests.V2Commands.command

    def test_raw_and_log1p_public_pipeline_saves_and_rebuilds_prepend(self):
        import torch
        from agentprint.v2 import build_model
        torch.set_num_threads(1)
        self.settings['model'] = {'variant': 'prepend'}
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
                self.assertEqual(saved['model_definition'], dict(version=1, variant='prepend',
                    width=2, dropout_2d=0.05, dropout_1d=0.1))
                self.assertEqual(saved['identity']['model_definition'], saved['model_definition'])
                restored = build_model(saved['model_definition'],
                                       saved['feature_definition']['in_channels'])
                restored.load_state_dict(saved['model'])
                parameters = sum(p.numel() for p in restored.parameters())
                self.assertEqual(saved['parameters'], parameters)
                self.assertEqual(json.loads((run / 'result.json').read_text())['parameters'],
                                 parameters)
                result = self.command('evaluate', '--checkpoint', str(run / 'best.pt'))
                summary = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(summary['experiment_id'], saved['experiment_id'])
                self.assertEqual(summary['transform'], transform)
                self.assertEqual(summary['support'], [1, 0, 0, 0, 0])
                self.assertTrue(torch.isfinite(torch.tensor(summary['loss'])))
                self.assertEqual(summary['loss'], saved['validation_result']['summary']['loss'])
                for other in ('single', 'replace'):
                    self.settings['model'] = {'variant': other}
                    mismatch = self.command('evaluate', '--checkpoint', str(run / 'best.pt'),
                                            ok=False)
                    self.assertIn('incompatible checkpoint model', mismatch.stderr)
                self.settings['model'] = {'variant': 'prepend'}


class PrependModel(unittest.TestCase):
    def test_public_factory_supports_two_and_ten_channels_with_finite_gradients(self):
        import torch
        from agentprint.v2 import build_model
        torch.set_num_threads(1)
        definition = dict(version=1, variant='prepend', width=2,
                          dropout_2d=0.05, dropout_1d=0.1)
        for channels in (2, 10):
            for transform in ('raw', 'log1p'):
                with self.subTest(channels=channels, transform=transform):
                    torch.manual_seed(42)
                    model = build_model(definition, in_channels=channels)
                    counts = torch.randint(0, 10, (2, channels, 2, 1800)).float()
                    values = (counts.log1p() if transform == 'log1p' else counts)
                    values.requires_grad_()
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
