import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import tempfile
import unittest

import numpy as np
import torch


class DFConvergence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, root):
        import sys
        from suite import ROOT as repo
        sys.path[:0]=[str(repo/'Benchmark/scripts'),str(repo/'Benchmark')]
        from run_ant_wf_scenario_smoke import build_model_specs,train_one_model,set_seed
        raw=root/'raw';raw.mkdir()
        rng=np.random.default_rng(123)
        for split in ('train','val','test'):
            np.savez_compressed(raw/(split+'.npz'),X=rng.normal(size=(5,5000)).astype('float32'),y=np.arange(5))
        (raw/'label_mapping.json').write_text(json.dumps({'id_to_label':{str(i):str(i) for i in range(5)}}))
        spec=replace(build_model_specs(repo/'Benchmark/all_default_configs')['DF'],train_epochs=2,batch_size=5)
        args=SimpleNamespace(device='cpu',feature_workers=1,learning_rate=None,batch_size=None,epochs=None,
                             optimizer=None,checkpoint_metric='accuracy',epoch_metrics_path=None)
        parent=root/'baseline'
        set_seed(42)
        train_one_model(spec,raw,parent,5,args)
        return raw,parent

    def policy(self):
        from agentprint.convergence import ConvergencePolicy
        return ConvergencePolicy(baseline_epochs=2,min_additional_epochs=2,max_total_epochs=4)

    def test_replays_parent_preserves_it_and_reports_cap_without_claiming_convergence(self):
        from tools.df_convergence import run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw,parent=self.fixture(root)
            before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.iterdir() if p.is_file()}
            output=root/'extension'
            result=run(raw,output,parent,device='cpu',policy=self.policy())
            self.assertTrue(result['baseline_parity_passed'])
            self.assertEqual(result['epochs'],4)
            self.assertFalse(result['stopping']['converged'])
            self.assertEqual(result['stopping']['reason'],'epoch_cap_reached')
            self.assertEqual(before,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in parent.iterdir() if p.is_file()})
            last=torch.load(output/'last.pt',weights_only=False)
            self.assertTrue(all(float(s['step'])==4 for s in last['optimizer']['state'].values()))
            self.assertEqual(result,run(raw,output,parent,device='cpu',policy=self.policy()))
            self.assertEqual(len(last['rows']),4)
            with np.load(raw/'train.npz') as data:
                changed_x, labels = data['X']+1, data['y']
            np.savez(raw/'train.npz',X=changed_x,y=labels)
            with self.assertRaisesRegex(ValueError,'identity'):
                run(raw,output,parent,device='cpu',policy=self.policy())

    def test_interrupted_resume_preserves_exact_model_optimizer_and_history(self):
        from tools.df_convergence import run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw,parent=self.fixture(root)
            run(raw,root/'whole',parent,device='cpu',policy=self.policy())
            # Extension must not need test data before its stop/selection decision.
            (raw/'test.npz').rename(raw/'test-reserved.npz')
            interrupted=run(raw,root/'resumed',parent,device='cpu',policy=self.policy(),epoch_limit=3)
            self.assertEqual(interrupted['state'],'interrupted')
            self.assertFalse((root/'resumed/result.json').exists())
            (raw/'test-reserved.npz').rename(raw/'test.npz')
            run(raw,root/'resumed',parent,device='cpu',policy=self.policy())
            a=torch.load(root/'whole/last.pt',weights_only=False)
            b=torch.load(root/'resumed/last.pt',weights_only=False)
            for key in a['model']:self.assertTrue(torch.equal(a['model'][key],b['model'][key]),key)
            for key in a['optimizer']['state']:
                for name in a['optimizer']['state'][key]:
                    self.assertTrue(torch.equal(a['optimizer']['state'][key][name],b['optimizer']['state'][key][name]),name)
            for left,right in zip(a['rows'],b['rows']):
                self.assertEqual({k:v for k,v in left.items() if k!='elapsed_seconds'},
                                 {k:v for k,v in right.items() if k!='elapsed_seconds'})

    def test_replay_mismatch_stops_before_extension(self):
        from tools.df_convergence import run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);raw,parent=self.fixture(root)
            metadata=json.loads((parent/'native_metrics.json').read_text())
            metadata['loss'][0]+=0.1
            (parent/'native_metrics.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(RuntimeError,'baseline replay mismatch'):
                run(raw,root/'rejected',parent,device='cpu',policy=self.policy())
            self.assertFalse((root/'rejected/last.pt').exists())
            self.assertFalse((root/'rejected/result.json').exists())
            # A separately identified fresh experiment must not pretend to replay the parent.
            fresh=run(raw,root/'fresh',parent,device='cpu',policy=self.policy(),fresh=True)
            self.assertEqual(fresh['identity']['initialization'],'fresh_deterministic_run')
            self.assertFalse(fresh['baseline_parity_passed'])


if __name__=='__main__':unittest.main()
