"""V2 protocol boundaries use synthetic histories and public smoke commands."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import yaml

from agentprint.config import load_config
from test_agentprint import REPO, capture, frame
import test_agentprint_v2 as tracer


class V2StoppingProtocol(unittest.TestCase):
    def rows(self, count, f1=0.5, loss=1.0):
        return [dict(epoch=i, val_macro_f1=f1, val_loss=loss) for i in range(count)]

    def test_formal_requires_first_thirty_anchor_and_thirty_additional_epochs(self):
        from agentprint.v2_protocol import stop_status
        self.assertFalse(stop_status(self.rows(59), 'formal', 200)['stop'])
        status = stop_status(self.rows(60), 'formal', 200)
        self.assertTrue(status['converged'])
        self.assertEqual(status['reason'], 'validation_plateau')
        self.assertEqual(status['last_significant_epoch'], 29)
        self.assertEqual(status['epochs_without_significant_gain'], 30)
        # The anchor is the best of all baseline epochs, not the last baseline row.
        rows = self.rows(60, f1=0.4995)
        rows[4]['val_macro_f1'] = 0.5
        rows[40]['val_macro_f1'] = 0.5005
        self.assertTrue(stop_status(rows, 'formal', 200)['converged'])

    def test_exact_significant_gain_loss_thresholds_zero_loss_and_epoch_cap(self):
        from agentprint.v2_protocol import stop_status
        rows = self.rows(60)
        rows[40]['val_macro_f1'] = 0.501
        self.assertEqual(stop_status(rows, 'formal', 200)['epochs_without_significant_gain'], 19)
        self.assertFalse(stop_status(rows, 'formal', 200)['stop'])
        rows = self.rows(60)
        for row in rows[-10:]:
            row['val_loss'] = 0.99
        self.assertFalse(stop_status(rows, 'formal', 200)['stop'])
        self.assertTrue(stop_status(self.rows(60, loss=0), 'formal', 200)['converged'])
        rows = self.rows(200)
        rows[-1]['val_macro_f1'] = 0.9
        status = stop_status(rows, 'formal', 200)
        self.assertTrue(status['stop'])
        self.assertFalse(status['converged'])
        self.assertEqual(status['reason'], 'epoch_cap_reached')
        self.assertTrue(stop_status(self.rows(200), 'formal', 200)['converged'])
        self.assertEqual(stop_status(self.rows(2), 'smoke', 2)['reason'], 'smoke_budget_exhausted')
        self.assertFalse(stop_status(self.rows(60), 'smoke', 60)['converged'])

    def test_tuning_uses_joint_convergence_policy_while_smoke_stays_nonconverged(self):
        from agentprint.v2_protocol import stop_status
        self.assertTrue(stop_status(self.rows(60), 'tuning', 200)['converged'])
        self.assertFalse(stop_status(self.rows(60), 'smoke', 200)['converged'])

    def test_tuning_requires_all_five_training_classes(self):
        from agentprint.config import load_config
        from agentprint.training import validate_training
        config = load_config(REPO / 'configs/agentprint/common-v2.yaml')
        config['mode'] = 'tuning'
        samples = type('Samples', (), {'records': [{'label_id': 0}]})()
        with self.assertRaisesRegex(ValueError, 'missing_training_classes'):
            validate_training(config, samples)

    def test_history_must_be_complete_contiguous_and_finite_including_baseline(self):
        from agentprint.v2_protocol import stop_status
        for field, value in [('epoch', 5), ('val_macro_f1', float('nan')),
                             ('val_loss', float('inf')), ('val_loss', -1)]:
            rows = self.rows(60)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                stop_status(rows, 'formal', 200)

    def test_v2_schedule_uses_new_anchor_and_floor(self):
        from agentprint.v2_protocol import learning_rate
        settings = load_config(REPO / 'configs/agentprint/common-v2.yaml')['training']
        self.assertEqual(learning_rate(0, settings), 0.001)
        self.assertEqual(learning_rate(30, settings), 0.0005)
        self.assertEqual(learning_rate(60, settings), 0.00025)
        self.assertEqual(learning_rate(300, settings), 0.000001)
        self.assertAlmostEqual(learning_rate(1, settings), 0.000977159968434246)

    def test_formal_configuration_enforces_anchor_and_fixed_two_hundred_epoch_cap(self):
        from agentprint.v2_protocol import validate_formal
        config = load_config(REPO / 'configs/agentprint/common-v2.yaml')
        config.update(mode='formal')
        config['training']['epochs'] = 200
        validate_formal(config)
        for group, field, value in [('training', 'epochs', 60), ('training', 'epochs', 201),
                                    ('training', 'lr_decay', 0.2), ('training', 'batch_size', 2),
                                    ('training', 'min_learning_rate', 1e-7),
                                    ('model', 'width', 1), ('model', 'dropout_1d', 0.3)]:
            changed = deepcopy(config)
            changed[group][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'formal'):
                validate_formal(changed)


class V2ProtocolCommands(unittest.TestCase):
    command = tracer.V2Commands.command

    def setUp(self):
        tracer.V2Commands.setUp(self)
        self.settings['training'] = {'epochs': 2, 'batch_size': 2}

    def train(self, *args):
        result = self.command('train', *args)
        return Path(json.loads(result.stdout.splitlines()[-1])['run_dir'])

    def test_two_epoch_smoke_persists_nonconvergence_history_and_new_schedule(self):
        import torch
        self.command('prepare')
        run = self.train()
        saved = torch.load(run / 'last.pt', weights_only=False)
        status = saved['convergence']
        self.assertTrue(status['stop'])
        self.assertFalse(status['converged'])
        self.assertEqual(status['reason'], 'smoke_budget_exhausted')
        self.assertEqual(status['total_epochs'], 2)
        self.assertEqual(saved['rows'][0]['learning_rate'], 0.001)
        self.assertAlmostEqual(saved['rows'][1]['learning_rate'], 0.000977159968434246)
        self.assertEqual(json.loads((run / 'convergence.json').read_text()), status)
        protocol = json.loads((run / 'protocol.json').read_text())
        self.assertFalse(protocol['cudnn_benchmark'])
        self.assertEqual(protocol['convergence_policy']['max_total_epochs'], 200)
        self.assertEqual(protocol['learning_rate_rule'],
                         'max(min_learning_rate,learning_rate*lr_decay**(epoch/30))')

    def test_validation_pairs_transforms_only_within_same_model_and_training_cohort(self):
        import torch
        self.command('prepare')
        raw = self.train()
        first = self.command('evaluate', '--checkpoint', str(raw / 'best.pt'))
        raw_summary = json.loads(first.stdout.splitlines()[-1])
        self.assertFalse(raw_summary['selection_complete'])
        self.settings['transform'] = 'log1p'
        log = self.train()
        second = self.command('evaluate', '--checkpoint', str(log / 'best.pt'))
        log_summary = json.loads(second.stdout.splitlines()[-1])
        self.assertEqual(raw_summary['selection_artifact'], log_summary['selection_artifact'])
        selection = json.loads(Path(log_summary['selection_artifact']).read_text())
        self.assertTrue(selection['complete'])
        self.assertNotIn('transform', selection['cohort'])
        for name, run in [('raw', raw), ('log1p', log)]:
            saved = torch.load(run / 'best.pt', weights_only=False)
            self.assertEqual(selection['candidates'][name]['validation_macro_f1'], saved['best_macro_f1'])
            self.assertEqual(selection['candidates'][name]['experiment_id'], saved['experiment_id'])
        winner = max(('raw', 'log1p'), key=lambda name: selection['candidates'][name]['validation_macro_f1'])
        self.assertEqual(selection['selected_transform'], winner)
        # A different model cannot complete or overwrite this transform pair.
        original = Path(log_summary['selection_artifact']).read_bytes()
        self.settings['model'] = {'width': 1}
        other = self.train()
        third = self.command('evaluate', '--checkpoint', str(other / 'best.pt'))
        other_summary = json.loads(third.stdout.splitlines()[-1])
        self.assertNotEqual(other_summary['selection_artifact'], log_summary['selection_artifact'])
        self.assertFalse(other_summary['selection_complete'])
        self.assertEqual(Path(log_summary['selection_artifact']).read_bytes(), original)
        self.assertFalse((self.root / 'cache/scenario/test').exists())

    def test_complete_epoch_recovery_restores_adam_rng_shuffle_best_and_archives_ahead_logs(self):
        import numpy as np
        import torch
        for split in ('train', 'val'):
            for index, scenario in enumerate(('business_services', 'general_knowledge')):
                capture(self.root / 'data' / split / 'engine' / scenario / f'e{index}/network.pcap', [
                    (0, frame('10.0.0.2', '8.8.8.8', 2)),
                    (index + 1, frame('8.8.8.8', '10.0.0.2', 18)),
                    (20, frame('10.0.0.2', '8.8.8.8'))])
        self.command('prepare')
        continuous = self.train()
        self.config.write_text(yaml.safe_dump(self.settings), encoding='utf-8')
        known = set((self.root / 'output').iterdir())
        child = subprocess.Popen([sys.executable, '-m', 'agentprint', 'train',
                                  '--config', str(self.config), '--device', 'cpu'],
                                 cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, bufsize=1)
        try:
            first = json.loads(child.stdout.readline())
            self.assertEqual(first['epoch'], 0)
            child.terminate()
            child.communicate(timeout=30)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()
        run = next(path for path in (self.root / 'output').iterdir() if path not in known)
        committed = torch.load(run / 'last.pt', weights_only=False)
        self.assertEqual(committed['epoch'], 0)
        with (run / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(committed['rows'][0], epoch=99)) + '\n{partial')
        (run / 'best.pt').write_bytes(b'incomplete publication')
        self.assertEqual(self.train('--resume', str(run)), run)
        expected = torch.load(continuous / 'last.pt', weights_only=False)
        actual = torch.load(run / 'last.pt', weights_only=False)

        def equal(left, right):
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right))
            elif isinstance(left, np.ndarray):
                np.testing.assert_array_equal(left, right)
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    equal(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    equal(a, b)
            else:
                self.assertEqual(left, right)

        for field in ('model', 'optimizer', 'rng', 'best_epoch', 'best_macro_f1', 'convergence', 'global_step'):
            with self.subTest(field=field):
                equal(expected[field], actual[field])
        comparable = lambda rows: [{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in rows]
        self.assertEqual(comparable(expected['rows']), comparable(actual['rows']))
        self.assertEqual(actual['best_epoch'], max(actual['rows'], key=lambda row: row['val_macro_f1'])['epoch'])
        restored_best = torch.load(run / 'best.pt', weights_only=False)
        self.assertEqual(restored_best['epoch'], actual['best_epoch'])
        archives = list((run / 'archive').glob('*/metrics.jsonl'))
        self.assertEqual(len(archives), 1)
        self.assertIn('{partial', archives[0].read_text())
        before = (run / 'metrics.jsonl').read_bytes()
        checkpoint_before = (run / 'last.pt').read_bytes()
        self.train('--resume', str(run))
        self.assertEqual((run / 'metrics.jsonl').read_bytes(), before)
        self.assertEqual((run / 'last.pt').read_bytes(), checkpoint_before)
        # A nonfinite restored model stops at the next epoch; it does not reduce the run.
        for value in committed['model'].values():
            if value.is_floating_point():
                value.fill_(float('nan'))
        torch.save(committed, run / 'last.pt')
        failed_checkpoint = (run / 'last.pt').read_bytes()
        result = self.command('train', '--resume', str(run), ok=False)
        self.assertIn('non-finite training loss', result.stderr)
        self.assertEqual((run / 'last.pt').read_bytes(), failed_checkpoint)

    def test_training_refuses_changed_identities_and_corrupt_committed_history(self):
        import torch
        from agentprint.config import fingerprint
        self.command('prepare')
        run = self.train()
        original = (run / 'last.pt').read_bytes()
        for field in ('model_definition', 'feature_definition', 'fit_fingerprint', 'iat_cache',
                      'base_cache', 'training', 'seed', 'transform'):
            saved = torch.load(run / 'last.pt', weights_only=False)
            saved['identity'][field] = 'foreign-identity'
            saved['experiment_id'] = fingerprint(saved['identity'])
            torch.save(saved, run / 'last.pt')
            result = self.command('train', '--resume', str(run), ok=False)
            self.assertIn('identity', result.stderr)
            (run / 'last.pt').write_bytes(original)
        saved = torch.load(run / 'last.pt', weights_only=False)
        saved['rows'][0]['val_loss'] += 0.25
        torch.save(saved, run / 'last.pt')
        result = self.command('train', '--resume', str(run), ok=False)
        self.assertIn('committed metrics history differ', result.stderr)

    def test_formal_public_train_validates_anchor_and_class_coverage_without_running_epochs(self):
        self.command('prepare')
        self.settings['mode'] = 'formal'
        result = self.command('train', ok=False)
        self.assertIn('formal protocol requires', result.stderr)
        self.settings['training'] = {'epochs': 200}
        result = self.command('train', ok=False)
        self.assertIn('missing_training_classes', result.stderr)
        self.assertFalse((self.root / 'output').exists())


if __name__ == '__main__':
    unittest.main()
