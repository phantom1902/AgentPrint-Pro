import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suite import scan, pool, task_label, save_predictions, aggregate, read
from make_fixture import make


class ProtocolTests(unittest.TestCase):
    def test_public_splits_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make(root)
            episodes = scan(root)
            self.assertEqual(len(episodes), 30)
            self.assertEqual(len({task_label(r, 'scenario') for r in episodes}), 5)
            p = root / 'val/fixture_engine/code_data_engineering/A1500'
            p.mkdir()
            (p / 'network.pcap').symlink_to(root / 'train/fixture_engine/code_data_engineering/A1500/network.pcap')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                scan(root)

    def test_selection_uses_validation_not_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            def row(split, name, family):
                return dict(id=name, split=split, family=family, episode='A4000' if family == 'attack_execution' else 'A1500')
            vr = [row('val', 'a', 'attack_execution')] * 3
            tr = [row('test', 'b', 'code_data_engineering')] * 3
            p = np.array([[.55, .45], [.55, .45], [0., 1.]])
            save_predictions(run / 'val', p, vr, ['benign', 'malicious'], 'risk')
            save_predictions(run / 'test', p, tr, ['benign', 'malicious'], 'risk')
            aggregate(run, 'risk', [vr[0], tr[0]])
            chosen = read(run / 'aggregation.json')
            self.assertEqual(chosen['selected_rule'], 'mean_soft')
            self.assertEqual(chosen['validation_accuracy']['majority_vote'], 0)
            self.assertEqual(read(run / 'episode/test/evaluation_metrics.json')['accuracy'], 0)

    def test_uniform_confidence_is_finite(self):
        p, _ = pool(np.full((3, 5), .2), [{'id': 'a'}] * 3, 'confidence_weighted_mean')
        np.testing.assert_allclose(p, np.full((1, 5), .2))


if __name__ == '__main__':
    unittest.main()
