"""Integrated IAT/multikernel behavior at approved public command/model seams."""
from contextlib import redirect_stderr, redirect_stdout
import io
import csv
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from agentprint.__main__ import main
from agentprint.config import load_config
from test_agentprint import REPO
from test_agentprint_iat_fit import nano_capture


class IATMultikernelIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'integration.yaml'
        self.settings = dict(extends=str(REPO / 'configs/agentprint/common-v2.yaml'),
            mode='smoke', paths={name + '_root': str(self.root / name)
                                 for name in ('data', 'cache', 'output')},
            preprocessing={'splits': ['train', 'val']},
            training={'epochs': 1, 'batch_size': 2})
        for split in ('train', 'val'):
            for scenario, name, times in [
                    ('artifact_processing', 'A', [0, 10, 11, 13, 13, 30, 1000]),
                    ('business_services', 'B', [0, 20, 24, 29, 40, 1000])]:
                nano_capture(self.root / 'data' / split / 'engine' / scenario / name / 'network.pcap', times)

    def command(self, command, *args, ok=True):
        self.config.write_text(yaml.safe_dump(self.settings), encoding='utf-8')
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main([command, '--config', str(self.config), *args])
        self.assertEqual(status == 0, ok, stdout.getvalue() + stderr.getvalue())
        return json.loads(stdout.getvalue().splitlines()[-1]) if ok else stderr.getvalue()

    def prepare_iat(self):
        self.command('prepare')
        self.settings['feature'] = dict(iat_enabled=True,
            fit_artifact=str(self.root / 'fit.json'), iat_cache_root=str(self.root / 'iat'))
        self.command('fit-bins')
        self.command('prepare')

    def test_six_iat_commands_share_fit_and_cache_and_isolate_three_transform_pairs(self):
        import torch
        from agentprint.v2 import build_model, loader
        self.prepare_iat()
        fit = (self.root / 'fit.json').read_bytes()
        cache = (self.root / 'iat/scenario/train/metadata.json').read_bytes()
        selections, experiment_ids = set(), set()
        for variant in ('single', 'replace', 'prepend'):
            selection_path = None
            for transform in ('raw', 'log1p'):
                with self.subTest(variant=variant, transform=transform):
                    self.settings['model'] = {'variant': variant}
                    self.settings['transform'] = transform
                    self.command('prepare')
                    self.assertEqual((self.root / 'fit.json').read_bytes(), fit)
                    self.assertEqual((self.root / 'iat/scenario/train/metadata.json').read_bytes(), cache)
                    run = Path(self.command('train')['run_dir'])
                    saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
                    self.assertEqual(saved['model_definition']['variant'], variant)
                    self.assertEqual(saved['feature_definition']['in_channels'], 10)
                    self.assertEqual(saved['identity']['fit_fingerprint'], json.loads(fit)['fingerprint'])
                    self.assertNotIn(saved['experiment_id'], experiment_ids)
                    experiment_ids.add(saved['experiment_id'])
                    summary = self.command('evaluate', '--checkpoint', str(run / 'best.pt'))
                    self.assertEqual(summary['support'], [1, 1, 0, 0, 0])
                    self.assertEqual(summary['experiment_id'], saved['experiment_id'])
                    self.assertEqual(summary['selection_complete'], transform == 'log1p')
                    if selection_path is None:
                        selection_path = summary['selection_artifact']
                    self.assertEqual(summary['selection_artifact'], selection_path)
                    # Saved definitions reconstruct the exact evaluated logits, not only shapes.
                    rebuilt = build_model(saved['model_definition'], 10).eval()
                    rebuilt.load_state_dict(saved['model'])
                    samples, batches = loader(load_config(self.config), 'val')
                    try:
                        with torch.no_grad():
                            predictions = rebuilt(next(iter(batches))[0]).argmax(1).tolist()
                        with (run / 'evaluation-val/confusion_matrix.csv').open(newline='', encoding='utf-8') as handle:
                            rows = list(csv.reader(handle))[1:]
                        matrix = np.asarray([[int(x) for x in row[1:]] for row in rows])
                        expected = np.zeros((5, 5), dtype=int)
                        for truth, predicted in zip((0, 1), predictions):
                            expected[truth, predicted] += 1
                        np.testing.assert_array_equal(matrix, expected)
                    finally:
                        samples._iat_shard = samples._shard = None
            selections.add(selection_path)
        self.assertEqual(len(selections), 3)
        self.assertFalse((self.root / 'data/test').exists())
        self.assertFalse((self.root / 'cache/scenario/test').exists())

    def test_iat_prepend_epoch_interrupt_recovers_model_adam_rng_shuffle_and_best(self):
        import torch
        for index in range(4):
            nano_capture(self.root / 'data/train/engine/artifact_processing' /
                         f'extra{index}/network.pcap', [0, 10, 12 + index, 20, 1000])
        self.settings['training']['epochs'] = 2
        self.prepare_iat()
        self.settings['model'] = {'variant': 'prepend'}
        self.settings['transform'] = 'log1p'
        continuous = Path(self.command('train')['run_dir'])
        known = set((self.root / 'output').iterdir())
        # Observe public stdout and terminate the real process at its first committed
        # epoch; no training/recovery collaborator is replaced or monkeypatched.
        stop_after_epoch = r"""
import json, os, sys
from agentprint.__main__ import main
original = sys.stdout
class StopAfterCommittedEpoch:
    pending = ''
    def write(self, text):
        original.write(text)
        original.flush()
        self.pending += text
        if '\n' in self.pending:
            line, self.pending = self.pending.split('\n', 1)
            if json.loads(line).get('epoch') == 0:
                os._exit(71)
        return len(text)
    def flush(self):
        original.flush()
sys.stdout = StopAfterCommittedEpoch()
raise SystemExit(main(['train', '--config', sys.argv[1], '--device', 'cpu']))
"""
        interrupted = subprocess.run([sys.executable, '-c', stop_after_epoch, str(self.config)],
            cwd=REPO, capture_output=True, text=True, timeout=120)
        self.assertEqual(interrupted.returncode, 71, interrupted.stdout + interrupted.stderr)
        self.assertEqual(json.loads(interrupted.stdout)['epoch'], 0)
        run = next(path for path in (self.root / 'output').iterdir() if path not in known)
        initial = torch.load(run / 'last.pt', weights_only=False)
        self.assertEqual(initial['epoch'], 0)
        with (run / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write('{' + 'partial ahead log')
        (run / 'best.pt').write_bytes(b'incomplete publication')
        self.assertEqual(self.command('train', '--resume', str(run))['run_dir'], str(run))
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
            elif isinstance(left, (tuple, list)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    equal(a, b)
            else:
                self.assertEqual(left, right)

        for field in ('identity', 'model', 'optimizer', 'rng', 'epoch', 'global_step',
                      'best_epoch', 'best_macro_f1', 'convergence'):
            with self.subTest(field=field):
                equal(expected[field], actual[field])
        comparable = lambda rows: [{k: v for k, v in row.items() if k != 'elapsed_seconds'} for row in rows]
        self.assertEqual(comparable(expected['rows']), comparable(actual['rows']))
        self.assertEqual(torch.load(run / 'best.pt', weights_only=False)['epoch'], actual['best_epoch'])
        archives = list((run / 'archive').glob('*/metrics.jsonl'))
        self.assertEqual(len(archives), 1)
        self.assertIn('partial ahead log', archives[0].read_text(encoding='utf-8'))
        before = (run / 'last.pt').read_bytes()
        self.command('train', '--resume', str(run))
        self.assertEqual((run / 'last.pt').read_bytes(), before)
        self.command('evaluate', '--checkpoint', str(run / 'best.pt'))

    def test_twelve_public_model_configurations_have_logits_cross_entropy_and_gradients(self):
        import torch
        from agentprint.v2 import build_model
        torch.set_num_threads(4)
        for enabled in (False, True):
            for variant in ('single', 'replace', 'prepend'):
                for transform in ('raw', 'log1p'):
                    with self.subTest(iat=enabled, variant=variant, transform=transform):
                        channels = 10 if enabled else 2
                        values = torch.arange(2 * channels * 2 * 1800, dtype=torch.float32).reshape(2, channels, 2, 1800) % 11
                        if transform == 'log1p':
                            values = values.log1p()
                        model = build_model(dict(version=1, variant=variant, width=2,
                                                dropout_2d=0.05, dropout_1d=0.1), channels)
                        logits = model(values)
                        self.assertEqual(tuple(logits.shape), (2, 5))
                        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 4]))
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                            for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
