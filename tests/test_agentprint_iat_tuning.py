import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentprint.config import load_config
from agentprint.iat_tuning import BASELINE_PARAMS, BOUNDS, build_trial_config, validate_iat_params


class IATTuningTests(unittest.TestCase):
    def test_parameter_endpoints_and_rejections(self):
        for side in (0, 1):
            validate_iat_params({k: v[side] for k, v in BOUNDS.items()})
        bad = [dict(BASELINE_PARAMS, width=2), {k: v for k, v in BASELINE_PARAMS.items() if k != 'lr_decay'}]
        for value in (True, float('nan'), float('inf'), -1, '0.1'):
            bad.append(dict(BASELINE_PARAMS, learning_rate=value))
        for params in bad:
            with self.subTest(params=params), self.assertRaises(ValueError):
                validate_iat_params(params)

    def test_fixed_workload_preserves_cache_and_fit(self):
        config = load_config('configs/agentprint/iat-full-20261102/replace-log1p.yaml')
        original = copy.deepcopy(config)
        candidate = build_trial_config(config, dict(BASELINE_PARAMS, lr_decay=0.7), 43, Path('isolated'))
        self.assertEqual(config, original)
        self.assertEqual(candidate['feature'], original['feature'])
        for name in ('data_root', 'cache_root'):
            self.assertEqual(candidate['paths'][name], original['paths'][name])
        self.assertEqual((candidate['mode'], candidate['transform'], candidate['model']['variant']), ('tuning', 'log1p', 'replace'))
        self.assertEqual((candidate['training']['epochs'], candidate['training']['batch_size'], candidate['model']['width']), (200, 200, 2))
        self.assertEqual(candidate['preprocessing']['splits'], ['train', 'val'])
        self.assertEqual(candidate['training']['lr_decay'], 0.7)

    def test_trial_preserves_requested_model_and_transform(self):
        for name, variant, transform in (
            ('single-log1p', 'single', 'log1p'),
            ('replace-raw', 'replace', 'raw'),
        ):
            with self.subTest(name=name):
                source = load_config(f'configs/agentprint/iat-full-20261102/{name}.yaml')
                original = copy.deepcopy(source)
                candidate = build_trial_config(source, BASELINE_PARAMS, 42, Path('isolated'))
                self.assertEqual((candidate['model']['variant'], candidate['transform']),
                                 (variant, transform))
                self.assertEqual(candidate['feature'], original['feature'])
                self.assertEqual(candidate['paths']['cache_root'], original['paths']['cache_root'])
                self.assertEqual(source, original)


if __name__ == '__main__':
    unittest.main()

class ProtocolTests(unittest.TestCase):
    def test_phase_baseline_and_budget_rules(self):
        from agentprint.iat_tuning import validate_phase
        validate_phase(BASELINE_PARAMS, 42, 'baseline', [])
        candidate = dict(BASELINE_PARAMS, lr_decay=0.7)
        baseline = dict(params=BASELINE_PARAMS, phase='baseline', seed=42, state='complete')
        for params, seed, phase, prior in [
            (candidate, 42, 'baseline', []), (candidate, 42, 'search', []),
            (candidate, 43, 'search', [baseline]), (candidate, 43, 'repeat', [baseline]),
            (BASELINE_PARAMS, 42, 'search', [baseline])]:
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                validate_phase(params, seed, phase, prior)
        searches = [dict(params=dict(candidate, learning_rate=0.0001+i*0.00001), seed=42,
                         phase='search', state='complete') for i in range(12)]
        with self.assertRaises(RuntimeError):
            validate_phase(candidate, 42, 'search', [baseline, *searches])
        with self.assertRaises(RuntimeError):
            validate_phase(searches[0]['params'], 44, 'repeat', [baseline, *searches,
                *[dict(phase='repeat', params=searches[0]['params'], seed=43) for _ in range(4)]])
        validate_phase(searches[0]['params'], 43, 'repeat', [baseline, searches[0]])

    def test_deadline_without_epoch_has_no_objective(self):
        import subprocess
        from agentprint.iat_tuning import execute_trial
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = build_trial_config(load_config('configs/agentprint/iat-full-20261102/replace-log1p.yaml'),
                                        BASELINE_PARAMS, 42, root / 'runs')
            with patch('agentprint.iat_tuning.command_result', side_effect=subprocess.TimeoutExpired('train', 1)):
                result = execute_trial(root, config, {}, 0, 'cpu')
            self.assertEqual(result['state'], 'pruned')
            self.assertIsNone(result['objective'])
            self.assertIsNone(result['latest_val_macro_f1'])
            self.assertTrue((root / 'train-command.json').exists())

    def test_deadline_uses_committed_epoch_not_log_tail(self):
        import subprocess
        from agentprint.iat_tuning import execute_trial
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = build_trial_config(load_config('configs/agentprint/iat-full-20261102/replace-log1p.yaml'),
                                        BASELINE_PARAMS, 42, root / 'runs')
            def timeout(*args):
                run_dir = root / 'runs' / 'v2-real'
                run_dir.mkdir(parents=True)
                (run_dir / 'last.pt').write_bytes(b'committed')
                (run_dir / 'metrics.jsonl').write_text('{"val_macro_f1": 0.99}')
                raise subprocess.TimeoutExpired('train', 1)
            with patch('agentprint.iat_tuning.command_result', side_effect=timeout), patch(
                'agentprint.iat_tuning.committed_result', return_value=dict(epochs=3,
                    latest_val_macro_f1=0.3, best_val_macro_f1=0.4)) as recovery:
                result = execute_trial(root, config, {'frozen': True}, 0, 'cpu')
            recovery.assert_called_once_with(root / 'runs' / 'v2-real', {'frozen': True})
            self.assertEqual(result['latest_val_macro_f1'], 0.3)
            self.assertEqual(result['state'], 'pruned')
            self.assertIsNone(result['objective'])
            self.assertEqual((root / 'runs' / 'v2-real' / 'last.pt').read_bytes(), b'committed')

    def test_public_train_then_validation_verifies_best(self):
        from agentprint.iat_tuning import execute_trial
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / 'runs' / 'v2-real'
            config = build_trial_config(load_config('configs/agentprint/iat-full-20261102/replace-log1p.yaml'),
                                        BASELINE_PARAMS, 42, root / 'runs')
            with patch('agentprint.iat_tuning.command_result', side_effect=[
                {'run_dir': str(run_dir)}, {'macro_f1': 0.4}]) as public, patch(
                'agentprint.iat_tuning.committed_result', return_value=dict(epochs=60,
                    latest_val_macro_f1=0.3, best_val_macro_f1=0.4, best_checkpoint=str(run_dir/'best.pt'))):
                result = execute_trial(root, config, {}, 10, 'cpu')
            self.assertEqual(result['objective'], 0.4)
            commands = [call.args[0] for call in public.call_args_list]
            self.assertEqual(commands[0][1:4], ['-m', 'agentprint', 'train'])
            self.assertEqual(commands[1][-2:], ['--split', 'val'])
            self.assertNotIn('test', commands[1])

    def test_committed_result_checks_identity_and_metrics(self):
        import sys
        from types import SimpleNamespace
        from agentprint.iat_tuning import committed_result
        saved = dict(identity={'frozen': 1}, epoch=1, rows=[{'val_macro_f1': 0.4}, {'val_macro_f1': 0.3}],
                     best_macro_f1=0.4, best_epoch=0, best_checkpoint='checkpoints/best.pt')
        with patch.dict(sys.modules, {'torch': SimpleNamespace(load=lambda *a, **k: saved)}), patch('agentprint.v2.validate_checkpoint'):
            self.assertEqual(committed_result('run', {'frozen': 1})['latest_val_macro_f1'], 0.3)
            with self.assertRaises(ValueError):
                committed_result('run', {'frozen': 2})
            saved['best_macro_f1'] = 0.99
            with self.assertRaises(ValueError):
                committed_result('run', {'frozen': 1})


class StudyTests(unittest.TestCase):
    def setUp(self):
        try:
            import optim_agent
        except ImportError:
            self.skipTest('optim-agent==0.1.1 optional dependency is not installed')
        self.oa = optim_agent
        from types import SimpleNamespace
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / '.optim-agent-runs' / 'study'
        self.params = Path(self.tmp.name) / 'params.json'
        self.params.write_text(__import__('json').dumps(BASELINE_PARAMS))
        self.args = SimpleNamespace(study_dir=str(self.root), trial_name='trial-00', params=str(self.params),
            phase='baseline', seed=42, config='configs/agentprint/iat-full-20261102/replace-log1p.yaml',
            device='cpu', rationale='test', invocation=[])
        sample = SimpleNamespace(metadata={'support': [1]*5, 'manifest_id': 'base'},
            fit_fingerprint='fit', iat_metadata={'scope': 'full', 'fingerprint': 'cache'})
        self.loader = patch('agentprint.v2.loader', return_value=(sample, []))
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def invoke(self):
        from agentprint.iat_tuning import run
        return run(self.args, self.oa.create_study)

    def test_real_ask_tell_idempotence_and_no_deadline_reset(self):
        import json
        complete = dict(state='complete', objective=0.4, latest_val_macro_f1=0.3)
        with patch('agentprint.iat_tuning.execute_trial', return_value=complete) as expensive:
            result = self.invoke()
            original_protocol = json.loads((self.root/'protocol.json').read_text())
            self.assertTrue((self.root/'trial-00'/'proposal.json').exists())
            self.assertTrue((self.root/'trial-00'/'ask.json').exists())
            self.assertEqual(self.invoke(), result)
            expensive.assert_called_once()
        self.assertEqual(original_protocol, json.loads((self.root/'protocol.json').read_text()))
        study = self.oa.create_study(direction='maximize', storage=self.root/'study.json')
        self.assertEqual(len(study.trials), 1)
        self.assertEqual(study.best_value, 0.4)
        self.args.trial_name = 'duplicate'
        self.args.phase = 'search'
        with self.assertRaises(ValueError):
            self.invoke()

    def test_pruned_and_failure_are_not_completed_objectives(self):
        with patch('agentprint.iat_tuning.execute_trial', return_value=dict(state='pruned',
            objective=None, epochs=3, latest_val_macro_f1=0.3)):
            self.invoke()
        study = self.oa.create_study(direction='maximize', storage=self.root/'study.json')
        self.assertEqual(study.trials[0].state, 'pruned')
        self.assertEqual(study.trials[0].value, 0.3)
        self.assertIsNone(study.best_value)
        self.args.trial_name = 'trial-01'
        self.args.phase = 'search'
        self.params.write_text(__import__('json').dumps(dict(BASELINE_PARAMS, lr_decay=0.7)))
        with patch('agentprint.iat_tuning.execute_trial', side_effect=RuntimeError('crash')):
            with self.assertRaisesRegex(RuntimeError, 'crash'):
                self.invoke()
        study = self.oa.create_study(direction='maximize', storage=self.root/'study.json')
        self.assertEqual(study.trials[1].state, 'failed')
        self.assertIsNone(study.trials[1].value)

    def test_manifest_drift_and_expired_budget_fail_closed(self):
        import json
        from types import SimpleNamespace
        with patch('agentprint.iat_tuning.execute_trial', return_value=dict(state='complete', objective=0.4)):
            self.invoke()
        self.args.trial_name = 'trial-01'
        self.args.phase = 'search'
        self.params.write_text(json.dumps(dict(BASELINE_PARAMS, lr_decay=0.7)))
        changed = SimpleNamespace(metadata={'support': [1]*5, 'manifest_id': 'changed'},
            fit_fingerprint='fit', iat_metadata={'scope': 'full', 'fingerprint': 'cache'})
        with patch('agentprint.v2.loader', return_value=(changed, [])), self.assertRaisesRegex(ValueError, 'identities'):
            self.invoke()
        protocol = json.loads((self.root/'protocol.json').read_text())
        with patch('agentprint.iat_tuning.time.time', return_value=protocol['deadline_unix'] + 1), self.assertRaisesRegex(RuntimeError, 'deadline'):
            self.invoke()

    def test_reconcile_result_after_crash_before_tell(self):
        with patch('agentprint.iat_tuning.execute_trial', return_value=dict(state='complete', objective=0.4)), patch(
                'agentprint.iat_tuning.tell_result', side_effect=RuntimeError('interrupted tell')):
            with self.assertRaises(RuntimeError):
                self.invoke()
        with patch('agentprint.iat_tuning.execute_trial', side_effect=AssertionError('must not retrain')):
            self.assertEqual(self.invoke()['objective'], 0.4)
        study = self.oa.create_study(direction='maximize', storage=self.root/'study.json')
        self.assertEqual(len(study.trials), 1)
        self.assertEqual(study.best_value, 0.4)


class CommandLifecycleTests(unittest.TestCase):
    def test_unsupported_platform_fails_before_launch(self):
        import time
        from agentprint.iat_tuning import command_result
        with tempfile.TemporaryDirectory() as tmp, patch('agentprint.iat_tuning.sys.platform', 'win32'), patch('agentprint.iat_tuning.subprocess.run') as launch:
            with self.assertRaisesRegex(RuntimeError, 'Linux'):
                command_result(['expensive'], Path(tmp) / 'log', time.time() + 10)
            launch.assert_not_called()

    @unittest.skipUnless(__import__('sys').platform == 'linux', 'Linux parent-death contract')
    def test_supervisor_sigkill_stops_writer_before_recovery(self):
        import os
        import signal
        import subprocess
        import sys
        import time
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = root / 'writer.py'
            writer.write_text('''import fcntl, os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
with (root / "writer.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / "pid").write_text(str(os.getpid()))
    while True:
        with (root / "checkpoint").open("a") as checkpoint:
            checkpoint.write("committed\\n")
        time.sleep(0.01)
''')
            supervisor = subprocess.Popen([sys.executable, '-c',
                'import sys,time; from pathlib import Path; from agentprint.iat_tuning import command_result,study_lock; '
                'root=Path(sys.argv[1]); '
                'exec("with study_lock(root):\\n command_result([sys.executable, str(root / \'writer.py\'), str(root)], root / \'log\', time.time()+30)")', str(root)])
            child_pid = None
            try:
                deadline = time.monotonic() + 10
                while not (root / 'checkpoint').exists() and time.monotonic() < deadline:
                    if supervisor.poll() is not None:
                        self.fail('supervisor exited before writer started')
                    time.sleep(0.01)
                self.assertTrue((root / 'checkpoint').exists())
                child_pid = int((root / 'pid').read_text())
                supervisor.kill()
                supervisor.wait(timeout=5)
                # Recover immediately: the new writer must acquire the exclusive
                # writer lock, proving the previous process cannot still write.
                recovered = subprocess.run([sys.executable, '-c',
                    'import fcntl,sys; from pathlib import Path; from agentprint.iat_tuning import study_lock; '
                    'root=Path(sys.argv[1]); '
                    'exec("with study_lock(root):\\n with (root / \'writer.lock\').open(\'a\') as lock:\\n  fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)")', str(root)],
                    capture_output=True, text=True, timeout=5)
                self.assertEqual(recovered.returncode, 0, recovered.stderr)
                committed = (root / 'checkpoint').read_bytes()
                time.sleep(0.15)
                self.assertEqual((root / 'checkpoint').read_bytes(), committed)
                status = Path('/proc') / str(child_pid) / 'status'
                self.assertTrue(not status.exists() or '\nState:\tZ' in status.read_text())
            finally:
                if supervisor.poll() is None:
                    supervisor.kill()
                    supervisor.wait(timeout=5)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    @unittest.skipUnless(__import__('sys').platform == 'linux', 'Linux public commands')
    def test_original_deadline_kills_command(self):
        import subprocess
        import sys
        import time
        from agentprint.iat_tuning import command_result
        with tempfile.TemporaryDirectory() as tmp:
            started = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                command_result([sys.executable, '-c', 'import time; time.sleep(30)'],
                               Path(tmp) / 'log', time.time() + 0.2)
            self.assertLess(time.monotonic() - started, 3)
