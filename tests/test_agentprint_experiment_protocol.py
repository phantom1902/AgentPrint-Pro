"""T05 public CLI protocol tests on bounded synthetic traffic, never research results."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import yaml

from test_agentprint import capture
from test_agentprint_primitive import annotation


REPO = Path(__file__).resolve().parents[1]


class ExperimentProtocol(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = self.root / 'config.yaml'
        self.settings = {
            'extends': str(REPO / 'configs/agentprint/common.yaml'),
            'mode': 'smoke',
            'paths': {'data_root': str(self.root / 'data'),
                      'cache_root': str(self.root / 'cache'),
                      'output_root': str(self.root / 'runs')},
            'training': {'epochs': 1, 'batch_size': 4}}
        self.save()

    def save(self):
        self.config.write_text(yaml.safe_dump(self.settings), encoding='utf-8')

    def command(self, name, *args, ok=True):
        result = subprocess.run([sys.executable, '-m', 'agentprint', name,
                                 '--config', str(self.config), *args], cwd=REPO,
                                capture_output=True, text=True,
                                env=dict(os.environ, PYTHONUTF8='1', OMP_NUM_THREADS='1',
                                         OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result

    def fixture(self):
        for split in ('train', 'val', 'test'):
            for label in ('artifact_processing', 'business_services'):
                target = self.root / 'data' / split / 'engine' / label / 'A001'
                capture(target / 'network.pcap')
                (target / 'messages.jsonl').write_text(
                    json.dumps(annotation(0, 'N001')) + '\n' +
                    json.dumps(dict(annotation(0, 'N047'), start_time='2023-11-14T22:28:20Z')) + '\n', encoding='utf-8')

    def prepare(self, task):
        self.settings.update(task=task, transform='raw')
        self.save()
        self.command('prepare')

    def checkpoint(self, task, transform, score, predicted=0):
        from agentprint import MODEL_VERSION, PREPROCESSING_VERSION
        from agentprint.cache import LABEL_MAP_ID
        from agentprint.config import load_config
        from agentprint.model import getRF
        model = getRF(5 if task == 'scenario' else 47)
        with torch.no_grad():
            for value in model.parameters():
                value.zero_()
            # Last BN is followed by ReLU and average pooling: fixed positive class score.
            last_bn = [module for module in model.modules()
                       if isinstance(module, torch.nn.BatchNorm1d)][-1]
            last_bn.bias[predicted] = 1
        run = self.root / 'runs' / f'{task}-{transform}'
        run.mkdir(parents=True, exist_ok=True)
        saved = {'model': model.state_dict(), 'config': load_config(self.config),
                 'epoch': 0, 'best_macro_f1': score,
                 'label_map_id': LABEL_MAP_ID, 'model_version': MODEL_VERSION,
                 'preprocessing_version': PREPROCESSING_VERSION}
        for split in ('train', 'val'):
            meta = json.loads((self.root / 'cache' / task / split / 'metadata.json').read_text())
            saved[f'{split}_manifest_id'] = meta['manifest_id']
        torch.save(saved, run / 'best.pt')
        return run

    def evaluate(self, run, split):
        result = self.command('evaluate', '--checkpoint', str(run / 'best.pt'), '--split', split)
        return json.loads(result.stdout.splitlines()[-1])

    def test_nested_split_is_a_clean_public_configuration_error(self):
        self.settings['preprocessing'] = {'splits': [['train']]}
        self.save()
        result = self.command('prepare', ok=False)
        self.assertIn('preprocessing.splits', result.stderr)
        self.assertNotIn('Traceback', result.stderr)

    def test_formal_parameters_cannot_silently_be_changed(self):
        self.settings['mode'] = 'formal'
        for field, value in [('seed', 7), ('epochs', 1), ('batch_size', 4),
                             ('learning_rate', .1), ('weight_decay', 0), ('lr_decay', .5)]:
            with self.subTest(field=field):
                settings = dict(self.settings)
                settings['training'] = {} if field == 'seed' else {field: value}
                if field == 'seed':
                    settings['seed'] = value
                self.config.write_text(yaml.safe_dump(settings), encoding='utf-8')
                result = self.command('prepare', ok=False)
                self.assertIn('formal', result.stderr)

    def test_four_public_train_and_evaluate_paths_share_raw_cache(self):
        self.fixture()
        for task in ('scenario', 'primitive'):
            self.prepare(task)
            cache = self.root / 'cache' / task
            before = {str(path): path.read_bytes() for path in cache.rglob('*') if path.is_file()}
            from agentprint.cache import CachedSamples
            raw = CachedSamples(self.root / 'cache', task, 'train', 'raw')
            logged = CachedSamples(self.root / 'cache', task, 'train', 'log1p')
            for index in range(len(raw)):
                values, label = raw[index]
                transformed, transformed_label = logged[index]
                self.assertEqual(values.dtype, np.float32)
                self.assertEqual(transformed.dtype, np.float32)
                self.assertEqual(label, transformed_label)
                np.testing.assert_array_equal(transformed, np.log1p(values))
            for transform in ('raw', 'log1p'):
                self.settings['transform'] = transform
                self.save()
                self.command('train')
                run = next(path for path in (self.root / 'runs').iterdir()
                           if path.name.startswith(f'{task}-{transform}-seed'))
                for split in ('val', 'test'):
                    summary = self.evaluate(run, split)
                    self.assertEqual(summary['task'], task)
                    self.assertEqual(summary['transform'], transform)
                    self.assertEqual(summary['coverage']['supported_class_count'], 2)
                protocol = json.loads((run / 'protocol.json').read_text())
                self.assertEqual(protocol['precision'], 'float32')
                self.assertFalse(protocol['formal_experiment'])
                self.assertEqual(protocol['seed'], 42)
            self.assertEqual(before, {str(path): path.read_bytes()
                                     for path in cache.rglob('*') if path.is_file()})

    def test_tied_validation_epochs_keep_the_earliest_checkpoint(self):
        self.fixture()
        self.prepare('scenario')
        # A tiny positive smoke-only LR keeps float32 scores unchanged while
        # still executing actual Adam/CE updates and repeated public validation.
        self.settings['training'].update(epochs=3, learning_rate=1e-30)
        self.save()
        self.command('train')
        run, = (self.root / 'runs').iterdir()
        rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
        scores = [row['val_macro_f1'] for row in rows]
        self.assertEqual(scores, [scores[0]] * 3)
        summary = self.evaluate(run, 'val')
        self.assertEqual(summary['epoch'], 0)

    def test_hand_computed_evaluation_fixed_support_and_validation_tie_selection(self):
        self.fixture()
        for task in ('scenario', 'primitive'):
            self.prepare(task)
            for transform, predicted in (('log1p', 2), ('raw', 0)):
                self.settings['transform'] = transform
                self.save()
                run = self.checkpoint(task, transform, .5, predicted)
                summary = self.evaluate(run, 'val')
                expected = [0, 1] if task == 'scenario' else [0, 46]
                self.assertEqual(summary['evaluated_classes'], expected)
                self.assertAlmostEqual(summary['macro_f1'], 0 if predicted == 2 else 1 / 3)
                self.assertAlmostEqual(summary['coverage']['class_fraction'], 2 / (5 if task == 'scenario' else 47))
                if task == 'primitive':
                    self.assertAlmostEqual(summary['fixed47_macro_f1'], 0 if predicted == 2 else 2 / 141)
                # Evaluate test as well; the saved validation score must decide even when
                # evaluated test performance differs between the transforms.
                self.evaluate(run, 'test')
            selection, = (self.root / 'runs').glob(f'selection-{task}-*.json')
            selected = json.loads(selection.read_text())
            self.assertTrue(selected['complete'])
            self.assertEqual(selected['selection_split'], 'val')
            self.assertEqual(selected['selected_transform'], 'raw')
            self.assertEqual(selected['candidates']['raw']['validation_macro_f1'], .5)
            self.assertTrue(selected['candidates']['raw']['validation_selected_main'])
            # A better validation score wins even though this log1p model predicts
            # an unsupported class and has strictly worse test results.
            self.settings['transform'] = 'log1p'
            self.save()
            run = self.checkpoint(task, 'log1p', .75, predicted=2)
            self.evaluate(run, 'test')
            selected = json.loads(selection.read_text())
            self.assertEqual(selected['selected_transform'], 'log1p')
            self.assertEqual(selected['candidates']['log1p']['validation_macro_f1'], .75)

    def test_missing_training_classes_stop_formal_before_creating_a_run(self):
        self.fixture()
        for task, missing in [('scenario', 'general_knowledge'), ('primitive', 'N002')]:
            self.prepare(task)
            self.settings['mode'] = 'formal'
            self.settings['training'] = {}
            self.save()
            result = self.command('train', ok=False)
            self.assertIn('missing_training_classes', result.stderr)
            self.assertIn(missing, result.stderr)
            self.assertIn('support', result.stderr)
            self.assertFalse((self.root / 'runs').exists())
            self.settings['mode'] = 'smoke'
            self.settings['training'] = {'epochs': 1, 'batch_size': 4}

    def test_evaluation_refuses_non_best_state_with_an_earlier_best_score(self):
        self.fixture()
        self.prepare('scenario')
        run = self.checkpoint('scenario', 'raw', .5)
        saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
        saved.update(epoch=1, best_epoch=0)
        torch.save(saved, run / 'last.pt')
        result = self.command('evaluate', '--checkpoint', str(run / 'last.pt'), ok=False)
        self.assertIn('validation-best checkpoint', result.stderr)
        self.assertFalse(list((self.root / 'runs').glob('selection-*.json')))

    def test_predeclared_overlay_files_keep_formal_and_smoke_distinct(self):
        from agentprint.config import load_config
        for task in ('scenario', 'primitive'):
            for transform in ('raw', 'log1p'):
                formal = load_config(REPO / f'configs/agentprint/{task}_{transform}.yaml')
                smoke = load_config(REPO / f'configs/agentprint/smoke_{task}_{transform}.yaml')
                self.assertEqual((formal['mode'], formal['seed'], formal['training']['epochs'],
                                  formal['training']['batch_size']), ('formal', 42, 30, 200))
                self.assertEqual((smoke['mode'], smoke['training']['epochs'],
                                  smoke['training']['batch_size']), ('smoke', 1, 4))


if __name__ == '__main__':
    unittest.main()

