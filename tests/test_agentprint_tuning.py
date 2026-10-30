"""Capacity variants must preserve the original point and valid gradient paths."""
import unittest
import hashlib
import json
from pathlib import Path
import tempfile
import time

import torch


class TuningContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def api(self):
        from agentprint import tuning
        return tuning

    def test_default_model_is_original_including_initialization_and_output(self):
        from agentprint.model import getRF
        tuning = self.api()
        torch.manual_seed(42)
        original = getRF(5).eval()
        torch.manual_seed(42)
        candidate = tuning.build_model(tuning.DEFAULT_PARAMS).eval()
        self.assertEqual(sum(p.numel() for p in candidate.parameters()), 891855)
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[key]), key)
        x = torch.randn(2, 2, 2, 1800)
        self.assertTrue(torch.equal(original(x), candidate(x)))

    def test_width_variants_have_five_logits_and_finite_gradients(self):
        tuning = self.api()
        counts = []
        for width in (0.5, 1.0, 2.0):
            model = tuning.build_model(dict(tuning.DEFAULT_PARAMS, width=width))
            logits = model(torch.randn(2, 2, 2, 1800))
            self.assertEqual(tuple(logits.shape), (2, 5))
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 4])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertTrue(counts[0] < counts[1] < counts[2])

    def test_rejects_out_of_scope_parameters(self):
        tuning = self.api()
        for change in ({'width': 3}, {'batch_size': 12}, {'learning_rate': float('nan')},
                       {'weight_decay': -1}, {'dropout_1d': 0.9}, {'task': 'primitive'}):
            with self.assertRaises(ValueError):
                tuning.validate_params(dict(tuning.DEFAULT_PARAMS, **change))

    def test_joint_stopping_minimum_and_cap(self):
        tuning = self.api()
        rows = [dict(epoch=i, val_macro_f1=0.5, val_loss=1.0) for i in range(60)]
        self.assertFalse(tuning.stop_status(rows[:59])['stop'])
        self.assertTrue(tuning.stop_status(rows)['converged'])
        improving = [dict(epoch=i, val_macro_f1=0.001*i, val_loss=200-i) for i in range(200)]
        self.assertEqual(tuning.stop_status(improving)['reason'], 'epoch_cap_reached')

    def fixture(self, cache):
        import numpy as np
        from agentprint.cache import write_split
        for split in ('train', 'val'):
            samples = []
            for i in range(5):
                x = np.zeros((2, 2, 1800), dtype=np.uint64)
                x[0, :, i*100:(i+1)*100] = i+1
                x[1] = x[0]*100
                samples.append((dict(sample_id=f'{split}:{i}', label_id=i), x))
            write_split(cache, 'scenario', split, samples, 'synthetic-tuning-test')

    def test_real_evaluator_records_best_and_is_idempotent_and_does_not_read_test(self):
        from agentprint.convergence import ConvergencePolicy
        tuning = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.fixture(base/'cache')
            policy = ConvergencePolicy(baseline_epochs=1, min_additional_epochs=1, patience=1,
                                       loss_window=1, max_total_epochs=2)
            result = tuning.evaluate_trial(tuning.DEFAULT_PARAMS, base/'trial', base/'cache',
                                           device='cpu', policy=policy)
            self.assertEqual(result['state'], 'complete')
            rows = [json.loads(s) for s in (base/'trial/metrics.jsonl').read_text().splitlines()]
            self.assertEqual(result['best_val_macro_f1'], max(r['val_macro_f1'] for r in rows))
            self.assertFalse(result['uses_test'])
            state = torch.load(base/'trial/last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(state['global_step'], 2)
            sha = hashlib.sha256((base/'trial/last.pt').read_bytes()).hexdigest()
            self.assertEqual(result, tuning.evaluate_trial(tuning.DEFAULT_PARAMS, base/'trial', base/'cache',
                                                          device='cpu', policy=policy))
            self.assertEqual(sha, hashlib.sha256((base/'trial/last.pt').read_bytes()).hexdigest())
            with self.assertRaises(ValueError):
                tuning.evaluate_trial(dict(tuning.DEFAULT_PARAMS, width=0.5), base/'trial',
                                      base/'cache', device='cpu', policy=policy)

    def test_expired_deadline_prunes_without_fabricating_a_score(self):
        tuning = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.fixture(base/'cache')
            result = tuning.evaluate_trial(tuning.DEFAULT_PARAMS, base/'trial', base/'cache',
                                           device='cpu', deadline=time.time()-1)
            self.assertEqual(result['state'], 'pruned')
            self.assertEqual(result['epochs'], 0)
            self.assertIsNone(result['best_val_macro_f1'])

    def test_resume_preserves_model_optimizer_rng_and_ignores_uncommitted_best(self):
        from agentprint.convergence import ConvergencePolicy
        from unittest.mock import patch
        tuning = self.api()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.fixture(base/'cache')
            policy = ConvergencePolicy(baseline_epochs=1, min_additional_epochs=2, patience=2,
                                       loss_window=1, max_total_epochs=3)
            expected = tuning.evaluate_trial(tuning.DEFAULT_PARAMS, base/'full', base/'cache', device='cpu', policy=policy)
            real_save = tuning.save_checkpoint
            def crash_after_commit(path, value):
                real_save(path, value)
                if Path(path).name == 'last.pt':
                    raise InterruptedError('simulated process interruption after epoch commit')
            with patch.object(tuning, 'save_checkpoint', side_effect=crash_after_commit):
                with self.assertRaises(InterruptedError):
                    tuning.evaluate_trial(tuning.DEFAULT_PARAMS, base/'resume', base/'cache', device='cpu', policy=policy)
            torch.save({'uncommitted': True}, base/'resume/best.pt')
            actual = tuning.evaluate_trial(tuning.DEFAULT_PARAMS, base/'resume', base/'cache', device='cpu', policy=policy)
            self.assertEqual(expected['best_val_macro_f1'], actual['best_val_macro_f1'])
            x = torch.load(base/'full/last.pt', map_location='cpu', weights_only=False)
            y = torch.load(base/'resume/last.pt', map_location='cpu', weights_only=False)
            for key, value in x['model'].items():
                self.assertTrue(torch.equal(value, y['model'][key]), key)
            for key, state in x['optimizer']['state'].items():
                for field, value in state.items():
                    self.assertTrue(torch.equal(value, y['optimizer']['state'][key][field]))
            self.assertTrue(torch.equal(x['rng']['torch'], y['rng']['torch']))
            self.assertTrue(torch.equal(x['rng']['shuffle_generator'], y['rng']['shuffle_generator']))
            best = torch.load(base/'resume/best.pt', map_location='cpu', weights_only=False)
            self.assertEqual(best['best_f1'], actual['best_val_macro_f1'])


if __name__ == '__main__':
    unittest.main()
