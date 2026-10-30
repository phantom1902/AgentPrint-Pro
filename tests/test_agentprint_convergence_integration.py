"""Real synthetic baseline and continuation, using independent output roots."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
import yaml
from test_agentprint import REPO, capture, frame


class ContinuationIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
        cls.config = cls.root/'config.yaml'
        cls.config.write_text(yaml.safe_dump({'extends':str(REPO/'configs/agentprint/smoke_scenario_raw.yaml'),
            'paths':{'data_root':str(cls.root/'data'),'cache_root':str(cls.root/'cache'),'output_root':str(cls.root/'baseline')},
            'training':{'epochs':2,'batch_size':2}}))
        for split in ('train','val'):
            for i,label in enumerate(('artifact_processing','business_services','general_knowledge')):
                capture(cls.root/'data'/split/'engine'/label/'episode'/'network.pcap',
                    [(0,frame('10.0.0.2','8.8.8.8',2)),(1,frame('10.0.0.2','1.1.1.1',2)),
                     (i+3,frame('8.8.8.8','10.0.0.2',18)),(20,frame('10.0.0.2','8.8.8.8'))])
        for cmd in ('prepare','train'):
            p=subprocess.run([sys.executable,'-m','agentprint',cmd,'--config',str(cls.config)],cwd=REPO,env=cls.env,text=True,capture_output=True)
            if p.returncode: raise RuntimeError(p.stdout+p.stderr)
        cls.parent=Path(json.loads(p.stdout.splitlines()[-1])['run_dir'])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def hashes(self):
        return {p.relative_to(self.parent).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.parent.rglob('*') if p.is_file()}

    def args(self, output):
        source = (
            'from agentprint.convergence import continue_training,ConvergencePolicy; '
            'continue_training('+repr(str(self.parent))+','+repr(str(output))+
            ',device="cpu",policy=ConvergencePolicy(baseline_epochs=2,min_additional_epochs=3,patience=2,loss_window=1,max_total_epochs=5))')
        return [sys.executable,'-c',source]

    def run_extension(self, output):
        p=subprocess.run(self.args(output),cwd=REPO,env=self.env,text=True,capture_output=True,timeout=180)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        return output/self.parent.name

    def test_parent_immutable_completed_noop_and_policy_cannot_change(self):
        before=self.hashes()
        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            output=Path(tmp); run=self.run_extension(output)
            state=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
            parent=torch.load(self.parent/'last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(state['epoch'],4)
            self.assertEqual(state['global_step'],10)
            self.assertTrue(all(float(v['step'])==10 for v in state['optimizer']['state'].values()))
            rows=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines()]
            base=[json.loads(s) for s in (self.parent/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual(rows[:2],base)
            self.assertEqual([r['epoch'] for r in rows],list(range(5)))
            self.assertEqual(json.loads((run/'convergence.json').read_text())['total_epochs'],5)
            protocol=json.loads((run/'protocol.json').read_text())
            self.assertFalse(protocol['formal_experiment'])
            self.assertEqual(protocol['experiment_kind'],'convergence_extension')
            best=torch.load(run/'best.pt',map_location='cpu',weights_only=False)
            self.assertEqual(best['best_macro_f1'],max(z['val_macro_f1'] for z in rows))
            self.assertTrue((run/'validation-best/summary.json').exists())
            self.assertFalse((run/'evaluation-test').exists())
            committed=(run/'last.pt').read_bytes(); self.run_extension(output)
            self.assertEqual(committed,(run/'last.pt').read_bytes())
            self.assertEqual(before,self.hashes())
            code=self.args(output)[-1].replace('patience=2','patience=3')
            p=subprocess.run([sys.executable,'-c',code],cwd=REPO,env=self.env,text=True,capture_output=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('policy',p.stderr)

    def test_formal_parent_missing_class_is_rejected_before_training(self):
        import copy
        import shutil
        with tempfile.TemporaryDirectory(dir=self.root) as tmp:
            parent=Path(tmp)/self.parent.name
            shutil.copytree(self.parent,parent)
            state=torch.load(parent/'last.pt',map_location='cpu',weights_only=False)
            state['config']=copy.deepcopy(state['config'])
            state['config']['mode']='formal'
            torch.save(state,parent/'last.pt')
            code=self.args(Path(tmp)/'extension')[-1].replace(repr(str(self.parent)),repr(str(parent)))
            p=subprocess.run([sys.executable,'-c',code],cwd=REPO,env=self.env,text=True,capture_output=True,timeout=180)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('missing_training_classes',p.stderr)

    def test_interrupted_continuation_matches_uninterrupted_model_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory(dir=self.root) as a,tempfile.TemporaryDirectory(dir=self.root) as b:
            expected=self.run_extension(Path(a))
            child=subprocess.Popen(self.args(Path(b)),cwd=REPO,env=self.env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=1)
            try:
                row=json.loads(child.stdout.readline()); self.assertEqual(row['epoch'],2)
                child.terminate(); child.communicate(timeout=30)
            finally:
                if child.poll() is None: child.kill(); child.communicate()
            actual=self.run_extension(Path(b))
            x=torch.load(expected/'last.pt',map_location='cpu',weights_only=False)
            y=torch.load(actual/'last.pt',map_location='cpu',weights_only=False)
            for k in x['model']: self.assertTrue(torch.equal(x['model'][k],y['model'][k]),k)
            self.assertEqual(x['global_step'],y['global_step'])
            self.assertTrue(torch.equal(x['rng']['torch'],y['rng']['torch']))
            self.assertTrue(torch.equal(x['rng']['shuffle_generator'],y['rng']['shuffle_generator']))
            for k,state in x['optimizer']['state'].items():
                for key,value in state.items():
                    self.assertTrue(torch.equal(value,y['optimizer']['state'][k][key]))
