"""Recovery is exercised through public commands and readable run artifacts."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from test_agentprint import REPO, capture, frame


class RecoveryCommands(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'smoke.yaml'
        self.settings = {
            'extends': str(REPO / 'configs/agentprint/common.yaml'),
            'task': 'scenario', 'transform': 'raw', 'mode': 'smoke',
            'paths': {'data_root': str(self.root / 'data'),
                      'cache_root': str(self.root / 'cache'),
                      'output_root': str(self.root / 'runs')},
            'training': {'epochs': 3, 'batch_size': 2},
        }
        self.write_config()
        self.env = dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='1',
                        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        for split in ('train', 'val'):
            for index, scenario in enumerate(('artifact_processing', 'business_services',
                                               'general_knowledge')):
                capture(self.root / 'data' / split / 'engine' / scenario /
                        f'episode{index}' / 'network.pcap', [
                            (0, frame('10.0.0.2', '8.8.8.8', 2)),
                            (1, frame('10.0.0.2', '1.1.1.1', 2)),
                            (index + 2, frame('8.8.8.8', '10.0.0.2', 18)),
                            (20, frame('10.0.0.2', '8.8.8.8'))])
        self.command('prepare')

    def write_config(self):
        self.config.write_text(yaml.safe_dump(self.settings), encoding='utf-8')

    def argv(self, *args):
        return [sys.executable, '-m', 'agentprint', *args, '--config', str(self.config)]

    def command(self, *args, ok=True):
        result = subprocess.run(self.argv(*args), cwd=REPO, env=self.env,
                                text=True, capture_output=True, timeout=180)
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result

    def train(self, *args):
        result = self.command('train', '--device', 'cpu', *args)
        return Path(json.loads(result.stdout.splitlines()[-1])['run_dir'])

    def rows(self, run):
        return [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]

    def events(self, run, stream):
        accumulator = EventAccumulator(str(run / 'tensorboard' / run.name / stream))
        accumulator.Reload()
        return accumulator

    def interrupted_train(self):
        existing = set((self.root / 'runs').iterdir()) if (self.root / 'runs').exists() else set()
        child = subprocess.Popen(self.argv('train', '--device', 'cpu'), cwd=REPO,
                                 env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
        interrupted = next(path for path in (self.root / 'runs').iterdir() if path not in existing)
        self.assertTrue((interrupted / 'last.pt').is_file())
        return interrupted

    def test_complete_epoch_interruption_resumes_to_total_and_matches_continuous_cpu(self):
        continuous = self.train()
        interrupted = self.interrupted_train()
        recovery = torch.load(interrupted / 'last.pt', weights_only=False)
        self.assertLess(recovery['epoch'], 2)
        self.assertIn('optimizer', recovery)
        self.assertIn('rng', recovery)
        self.assertIn('shuffle_generator', recovery['rng'])
        self.assertEqual(self.train('--resume', str(interrupted)), interrupted)
        expected, actual = self.rows(continuous), self.rows(interrupted)
        self.assertEqual([row['epoch'] for row in actual], [0, 1, 2])
        for left, right in zip(expected, actual):
            self.assertEqual({k: v for k, v in left.items() if k != 'elapsed_seconds'},
                             {k: v for k, v in right.items() if k != 'elapsed_seconds'})
        a = torch.load(continuous / 'last.pt', weights_only=False)
        b = torch.load(interrupted / 'last.pt', weights_only=False)
        self.assertEqual(a['best_epoch'], b['best_epoch'])
        self.assertEqual(b['global_step'], 6)
        for key in a['model']:
            self.assertTrue(torch.equal(a['model'][key], b['model'][key]), key)
        self.assertEqual([event.step for event in
                          self.events(interrupted, 'epochs').Scalars('val_macro_f1')], [0, 1, 2])
        before = (interrupted / 'metrics.jsonl').read_bytes()
        self.train('--resume', str(interrupted))
        self.assertEqual((interrupted / 'metrics.jsonl').read_bytes(), before)
        self.command('evaluate', '--checkpoint', str(interrupted / 'best.pt'), '--split', 'val')

    def test_repeat_last_archives_ahead_logs_and_restores_committed_best(self):
        run = self.interrupted_train()
        committed = (run / 'last.pt').read_bytes()
        self.train('--resume', str(run))
        # Repeat an authentic committed recovery point with newer observable logs.
        (run / 'last.pt').write_bytes(committed)
        (run / 'best.pt').write_bytes(b'incomplete publication')
        metrics = self.rows(run)
        with (run / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(metrics[-1], epoch=99)) + '\n')
            handle.write('{partial')
        self.train('--resume', str(run))
        self.assertEqual([row['epoch'] for row in self.rows(run)], [0, 1, 2])
        archives = list((run / 'archive').glob('*/metrics.jsonl'))
        self.assertEqual(len(archives), 1)
        self.assertIn('99', archives[0].read_text())
        self.assertIn('{partial', archives[0].read_text())
        saved_best = torch.load(run / 'best.pt', weights_only=False)
        self.assertEqual(saved_best['epoch'], torch.load(run / 'last.pt', weights_only=False)['best_epoch'])
        self.assertEqual([event.step for event in
                          self.events(run, 'epochs').Scalars('train_loss')], [0, 1, 2])

    def test_resume_checks_experiment_identity_and_records_allowed_path_changes(self):
        run = self.train()
        original = self.settings.copy()
        for field, value in (('seed', 7), ('preprocessing', {'bins': 1800,
                                                         'shard_size': 1024,
                                                         'splits': ['train', 'val']})):
            self.settings[field] = value
            self.write_config()
            error = self.command('train', '--resume', str(run), ok=False)
            self.assertIn(field, error.stderr)
            self.settings = original.copy()
        self.settings['training'] = dict(self.settings['training'], batch_size=1)
        self.write_config()
        self.assertIn('training', self.command('train', '--resume', str(run), ok=False).stderr)
        self.settings = original.copy()
        self.write_config()
        committed = (run / 'last.pt').read_bytes()
        for field in ('model_version', 'preprocessing_version', 'label_map_id',
                      'train_manifest_id', 'val_manifest_id'):
            checkpoint = torch.load(run / 'last.pt', weights_only=False)
            checkpoint[field] = 'different-identity'
            torch.save(checkpoint, run / 'last.pt')
            self.assertIn(field, self.command('train', '--resume', str(run), ok=False).stderr)
            (run / 'last.pt').write_bytes(committed)
        copied = self.root / 'relocated-cache'
        shutil.copytree(self.root / 'cache', copied)
        self.train('--resume', str(run), '--cache-root', str(copied),
                   '--output-root', str(self.root / 'elsewhere'))
        history = [json.loads(line) for line in (run / 'runtime_history.jsonl').read_text().splitlines()]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[-1]['config']['paths']['cache_root'], str(copied))
        self.assertIn('git_commit', history[-1]['environment'])
        self.assertIn('torch', history[-1]['environment']['packages'])
        self.assertTrue((run / 'summary.json').is_file())
        self.assertTrue((run / 'per_class.csv').is_file())
        self.assertTrue((run / 'confusion_matrix.csv').is_file())

    def test_periodic_loss_uses_global_steps_across_epoch_recovery(self):
        self.settings['training'] = {'epochs': 18, 'batch_size': 1}
        self.write_config()
        run = self.interrupted_train()
        self.train('--resume', str(run))
        steps = [json.loads(line) for line in (run / 'training_steps.jsonl').read_text().splitlines()]
        self.assertEqual([row['global_step'] for row in steps], [50])
        self.assertEqual(steps[0]['epoch'], 16)
        self.assertEqual([event.step for event in self.events(run, 'steps').Scalars('train_loss')], [50])
        last = torch.load(run / 'last.pt', weights_only=False)
        self.assertEqual(last['global_step'], 54)


if __name__ == '__main__':
    unittest.main()
