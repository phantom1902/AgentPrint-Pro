"""Convergence extension policy and real checkpoint continuation."""
import math
import unittest

from agentprint.convergence import ConvergencePolicy, assess, learning_rate


class ConvergenceRules(unittest.TestCase):
    def rows(self, n, f1=0.5, loss=1.0):
        return [dict(epoch=30+i, val_macro_f1=f1, val_loss=loss) for i in range(n)]

    def test_original_exponent_is_not_stretched_and_floor_is_respected(self):
        self.assertAlmostEqual(learning_rate(30), 0.0001)
        self.assertAlmostEqual(learning_rate(60), 0.00002)
        self.assertEqual(learning_rate(199), 0.000001)

    def test_minimum_thirty_additional_epochs_and_both_plateau_conditions(self):
        policy = ConvergencePolicy()
        self.assertFalse(assess(self.rows(29), 0.5, policy)['stop'])
        result = assess(self.rows(30), 0.5, policy)
        self.assertTrue(result['stop'])
        self.assertTrue(result['converged'])
        self.assertEqual(result['reason'], 'validation_plateau')
        improving = self.rows(30)
        for row in improving[20:]:
            row['val_loss'] = 0.98
        self.assertFalse(assess(improving, 0.5, policy)['stop'])
        improving = self.rows(30)
        improving[-1]['val_macro_f1'] = 0.502
        self.assertFalse(assess(improving, 0.5, policy)['stop'])

    def test_cumulative_small_gains_reset_patience_but_checkpoint_ties_do_not(self):
        values = self.rows(50)
        for i, row in enumerate(values):
            row['val_macro_f1'] = 0.5 + min(i+1, 31)*0.00005
        result = assess(values, 0.5, ConvergencePolicy())
        self.assertEqual(result['last_significant_epoch'], 49)
        self.assertEqual(result['epochs_without_significant_gain'], 30)
        self.assertTrue(result['converged'])
        self.assertEqual(assess(self.rows(30), 0.5, ConvergencePolicy())['epochs_without_significant_gain'], 30)

    def test_exact_thresholds_and_regressing_validation_loss(self):
        rows = self.rows(30)
        rows[10]['val_macro_f1'] = 0.501
        result = assess(rows, 0.5, ConvergencePolicy())
        self.assertEqual(result['epochs_without_significant_gain'], 19)
        self.assertFalse(result['stop'])
        rows = self.rows(30)
        for row in rows[-10:]:
            row['val_loss'] = 0.99
        self.assertFalse(assess(rows, 0.5, ConvergencePolicy())['stop'])
        for row in rows[-10:]:
            row['val_loss'] = 1.1
        self.assertTrue(assess(rows, 0.5, ConvergencePolicy())['converged'])

    def test_cap_does_not_claim_convergence_and_plateau_at_cap_does(self):
        rows = self.rows(170)
        rows[-1]['val_macro_f1'] = 0.9
        result = assess(rows, 0.5, ConvergencePolicy())
        self.assertTrue(result['stop'])
        self.assertFalse(result['converged'])
        self.assertEqual(result['reason'], 'epoch_cap_reached')
        self.assertTrue(assess(self.rows(170), 0.5, ConvergencePolicy())['converged'])

    def test_nonfinite_out_of_order_and_invalid_policy_are_rejected(self):
        for key, value in [('val_loss', float('nan')), ('val_macro_f1', float('inf')), ('val_loss', -1), ('val_macro_f1', 1.2), ('epoch', 45)]:
            rows = self.rows(30); rows[1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                assess(rows, 0.5, ConvergencePolicy())
        for kwargs in ({'min_additional_epochs': 0}, {'patience': 0}, {'loss_window': 0}, {'max_total_epochs': 40}, {'f1_min_delta': -1}):
            with self.assertRaises(ValueError):
                ConvergencePolicy(**kwargs)


if __name__ == '__main__':
    unittest.main()
