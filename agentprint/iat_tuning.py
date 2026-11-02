"""Sequential validation-only optim-agent trials for full-cache IAT variants."""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .cache import write_json
from .config import fingerprint, load_config

BASELINE_PARAMS = dict(learning_rate=0.001, weight_decay=0.001, lr_decay=0.5,
                       dropout_2d=0.05, dropout_1d=0.1)
BOUNDS = dict(learning_rate=(1e-5, 0.005), weight_decay=(0, 0.01),
              lr_decay=(0.1, 1.0), dropout_2d=(0, 0.3), dropout_1d=(0, 0.5))
LIMITS = dict(max_trials=17, max_search_trials=12, max_repeat_trials=4, wall_seconds=3600)


def validate_iat_params(params):
    if not isinstance(params, dict) or params.keys() != BOUNDS.keys():
        raise ValueError('parameters must contain exactly the five approved IAT values')
    for name, (low, high) in BOUNDS.items():
        value = params[name]
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not low <= value <= high):
            raise ValueError('out of bounds: ' + name)


def build_trial_config(config, params, seed, output_root):
    from .v2 import validate_config
    validate_iat_params(params)
    if type(seed) is not int or seed not in (42, 43, 44):
        raise ValueError('only seeds 42, 43, 44 are approved')
    if config.get('schema_version') != 2 or not config['feature']['iat_enabled']:
        raise ValueError('requires v2 enabled IAT and existing full-scope caches')
    candidate = deepcopy(config)
    candidate.update(task='scenario', transform=config['transform'], mode='tuning', seed=seed)
    candidate['preprocessing']['splits'] = ['train', 'val']
    candidate['paths']['output_root'] = str(Path(output_root).resolve())
    candidate['model'].update(variant=config['model']['variant'], width=2.0,
        dropout_2d=params['dropout_2d'], dropout_1d=params['dropout_1d'])
    candidate['training'].update(epochs=200, batch_size=200, optimizer='adam',
        min_learning_rate=1e-6, num_workers=0, numerical_threads=4,
        **{k: params[k] for k in ('learning_rate', 'weight_decay', 'lr_decay')})
    return validate_config(candidate)


def workload_identity(config):
    """Verify only train/val; loaders verify source/fit/cache fingerprints in full."""
    from .v2 import experiment_identity, loader
    train, _ = loader(config, 'train')
    val, _ = loader(config, 'val')
    for samples in (train, val):
        if (samples.iat_metadata.get('scope') != 'full'
                or any(n == 0 for n in samples.metadata['support'])):
            raise ValueError('requires full-scope caches with all five classes')
    identity = experiment_identity(config, train, val)
    # Search values and seed are bound separately to each trial, not to the study.
    for name in ('seed', 'training', 'model_definition'):
        identity.pop(name)
    return identity


@contextmanager
def study_lock(root):
    """OS releases the exclusive lock even if the runner is killed."""
    with (root / 'study.lock').open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0)
            lock.write(b'0')
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Keep the same flock alive until the public child has actually
            # exited, closing the brief parent-death signal delivery window.
            os.set_inheritable(lock.fileno(), True)
        try:
            yield
        finally:
            if os.name == 'nt':
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_phase(params, seed, phase, prior):
    if phase not in ('baseline', 'search', 'repeat'):
        raise ValueError('unknown trial phase')
    if phase == 'baseline':
        if params != BASELINE_PARAMS or seed != 42 or prior:
            raise ValueError('first baseline must use the exact approved point and seed 42')
    elif not prior or prior[0]['phase'] != 'baseline':
        raise ValueError('first trial must establish baseline')
    elif phase == 'search' and seed != 42:
        raise ValueError('search uses seed 42')
    elif phase == 'repeat':
        if seed not in (43, 44):
            raise ValueError('repeats use seeds 43 or 44')
        if not any(p['phase'] == 'search' and p['params'] == params
                   and p.get('state') == 'complete' for p in prior):
            raise ValueError('repeat requires a completed new seed-42 configuration')
    if len(prior) >= LIMITS['max_trials']:
        raise RuntimeError('17 evaluation slots exhausted')
    cap = {'search': 12, 'repeat': 4, 'baseline': 1}[phase]
    if sum(p['phase'] == phase for p in prior) >= cap:
        raise RuntimeError(phase + ' trial allocation exhausted')
    if any(p['params'] == params and p['seed'] == seed for p in prior):
        raise ValueError('duplicate parameter/seed combination')


def command_result(command, log, deadline):
    """Bound both public commands by the original study deadline."""
    if sys.platform != 'linux':
        raise RuntimeError('public tuning commands require Linux parent-death protection')
    import ctypes
    import signal
    libc = ctypes.CDLL(None, use_errno=True)
    parent_pid = os.getpid()

    def protect_child():
        # Training uses no worker children. Kill this public command even when
        # SIGKILL prevents the runner from cleaning it up or retaining its lock.
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            os._exit(127)
        # The runner may have died between fork and installing the signal.
        if os.getppid() != parent_pid:
            os._exit(127)

    remaining = deadline - time.time()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(command, 0)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with Path(log).open('a', encoding='utf-8') as output:
        subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                       timeout=remaining, check=True, env=env, preexec_fn=protect_child,
                       close_fds=False)
    for line in reversed(Path(log).read_text(encoding='utf-8').splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError('public command did not emit its final JSON result')


def committed_result(run_dir, identity):
    """Ignore metrics tails: only atomic last.pt establishes a complete epoch."""
    import torch
    from .v2 import validate_checkpoint
    saved = torch.load(Path(run_dir) / 'last.pt', map_location='cpu', weights_only=False)
    validate_checkpoint(saved)
    if saved['identity'] != identity or len(saved['rows']) != saved['epoch'] + 1:
        raise ValueError('committed checkpoint trial identity/history changed')
    rows = saved['rows']
    score = rows[-1]['val_macro_f1']
    best = saved['best_macro_f1']
    if (not rows or any(not math.isfinite(v) or not 0 <= v <= 1 for v in (score, best))
            or best != max(r['val_macro_f1'] for r in rows)):
        raise ValueError('invalid committed validation metrics')
    return dict(epochs=len(rows), latest_val_macro_f1=score,
                best_val_macro_f1=best, best_epoch=saved['best_epoch'],
                best_checkpoint=str(Path(run_dir) / saved['best_checkpoint']))


def execute_trial(root, config, identity, deadline, device):
    import yaml
    root = Path(root)
    config_path = root / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    runs_root = Path(config['paths']['output_root'])
    runs = sorted(runs_root.glob('v2-*')) if runs_root.exists() else []
    if len(runs) > 1:
        raise ValueError('ambiguous trial run directories')
    run_dir = runs[0] if runs else None
    train = [sys.executable, '-m', 'agentprint', 'train', '--config', str(config_path), '--device', device]
    if run_dir is not None:
        if (run_dir / 'last.pt').exists():
            committed_result(run_dir, identity)
            train += ['--resume', str(run_dir)]
        else:
            raise ValueError('interrupted first epoch has no committed checkpoint; use a new trial')
    try:
        if run_dir is None or not (run_dir / 'result.json').exists():
            write_json(root / 'train-command.json', dict(command=train, output=str(root / 'train.log')))
            trained = command_result(train, root / 'train.log', deadline)
            run_dir = Path(trained['run_dir'])
            if run_dir.parent.resolve() != runs_root.resolve():
                raise ValueError('public training returned an unexpected run directory')
        metrics = committed_result(run_dir, identity)
        evaluate = [sys.executable, '-m', 'agentprint', 'evaluate', '--config', str(config_path),
                    '--device', device, '--checkpoint', metrics['best_checkpoint'], '--split', 'val']
        write_json(root / 'evaluate-command.json', dict(command=evaluate, output=str(root / 'evaluate.log')))
        evaluated = command_result(evaluate, root / 'evaluate.log', deadline)
        score = evaluated['macro_f1']
        if not math.isfinite(score) or not math.isclose(score, metrics['best_val_macro_f1'], rel_tol=0, abs_tol=1e-12):
            raise ValueError('reloaded validation result differs from committed best score')
        return dict(metrics, state='complete', reason='public_validation_verified',
                    objective=score, evaluation=evaluated, run_dir=str(run_dir), uses_test=False)
    except subprocess.TimeoutExpired:
        runs = sorted(runs_root.glob('v2-*')) if runs_root.exists() else []
        if len(runs) > 1:
            raise ValueError('ambiguous deadline recovery runs')
        metrics = dict(epochs=0, latest_val_macro_f1=None, best_val_macro_f1=None)
        if runs and (runs[0] / 'last.pt').exists():
            metrics = committed_result(runs[0], identity)
        return dict(metrics, state='pruned', reason='study_deadline', objective=None,
                    run_dir=str(runs[0]) if runs else None, uses_test=False)


def tell_result(study, trial, result):
    if result['state'] == 'complete':
        study.tell(trial, result['objective'])
    elif result['state'] == 'pruned':
        if result['latest_val_macro_f1'] is not None:
            trial.report(result['latest_val_macro_f1'], step=result['epochs'] - 1)
        study.tell(trial, state='pruned')
    else:
        study.tell(trial, state='failed')


def run(args, study_factory=None):
    from .v2 import experiment_identity, loader
    root = Path(args.study_dir).resolve()
    if '.optim-agent-runs' not in root.parts:
        raise ValueError('study must be under ignored .optim-agent-runs')
    if not args.trial_name or Path(args.trial_name).name != args.trial_name or args.trial_name in ('.', '..'):
        raise ValueError('trial-name must be a single path component')
    params = read_json(args.params)
    validate_iat_params(params)
    source = load_config(args.config)
    trial_root = root / args.trial_name
    config = build_trial_config(source, params, args.seed, trial_root / 'runs')
    # Never allow output inside or above existing data/fit/cache artifacts.
    for location in [source['paths']['data_root'], source['paths']['cache_root'],
                     source['paths']['output_root'], source['feature']['iat_cache_root'],
                     str(Path(source['feature']['fit_artifact']).parent)]:
        other = Path(location).resolve()
        if root.is_relative_to(other) or other.is_relative_to(root):
            raise ValueError('study must not overlap existing artifacts')
    root.mkdir(parents=True, exist_ok=True)
    with study_lock(root):
        protocol_path = root / 'protocol.json'
        if not protocol_path.exists():
            validate_phase(params, args.seed, args.phase, [])
            started = time.time()
            write_json(protocol_path, dict(LIMITS, started_unix=started, deadline_unix=started + 3600,
                baseline=BASELINE_PARAMS, objective='best_validation_macro_f1', direction='maximize', uses_test=False))
        protocol = read_json(protocol_path)
        if (any(protocol.get(k) != v for k, v in LIMITS.items()) or protocol.get('baseline') != BASELINE_PARAMS
                or protocol.get('objective') != 'best_validation_macro_f1'
                or protocol.get('direction') != 'maximize' or protocol.get('uses_test') is not False
                or protocol['deadline_unix'] != protocol['started_unix'] + 3600):
            raise ValueError('approved study protocol changed')
        # The canonical baseline config freezes every non-search setting and all paths.
        canonical = build_trial_config(source, BASELINE_PARAMS, 42, root / 'canonical-output')
        workload = dict(config=canonical, identity=workload_identity(canonical))
        if 'workload' not in protocol:
            protocol['workload'] = workload
            write_json(protocol_path, protocol)
        elif protocol['workload'] != workload:
            raise ValueError('study full manifests, fit/cache identities or configuration changed')
        train, _ = loader(config, 'train')
        val, _ = loader(config, 'val')
        identity = experiment_identity(config, train, val)
        trial_identity = dict(config=config, experiment=identity, params=params, seed=args.seed)
        identity_id = fingerprint(trial_identity)
        proposals = []
        for path in sorted(root.glob('*/proposal.json'), key=lambda p: read_json(p)['order']):
            proposal = read_json(path)
            result_file = path.parent / 'result.json'
            proposal['state'] = read_json(result_file)['state'] if result_file.exists() else 'running'
            proposals.append(proposal)
        existing = trial_root / 'proposal.json'
        if existing.exists():
            proposal = read_json(existing)
            if proposal['identity_id'] != identity_id or proposal['phase'] != args.phase:
                raise ValueError('existing trial identity changed')
        else:
            if any(p['state'] == 'running' for p in proposals):
                raise ValueError('resume the pending trial before proposing another')
            validate_phase(params, args.seed, args.phase, proposals)
            if time.time() >= protocol['deadline_unix']:
                raise RuntimeError('3600-second study deadline exhausted')
            proposal = dict(params=params, seed=args.seed, phase=args.phase, rationale=args.rationale,
                identity=trial_identity, identity_id=identity_id, order=len(proposals),
                output=str(trial_root), proposed_unix=time.time(),
                command=[sys.executable, '-m', 'agentprint.iat_tuning', *args.invocation])
            write_json(existing, proposal)
        if study_factory is None:
            from importlib.metadata import version
            if version('optim-agent') != '0.1.1':
                raise RuntimeError('requires optim-agent==0.1.1')
            import optim_agent
            study_factory = optim_agent.create_study
        study = study_factory(direction='maximize', storage=root / 'study.json', seed=42)
        known = [dict(p['params'], seed=p['seed']) for p in proposals]
        if dict(params, seed=args.seed) not in known:
            known.append(dict(params, seed=args.seed))
        if len(study.trials) > 17 or any(t.params not in known for t in study.trials):
            raise ValueError('study contains evaluations without matching proposals')
        for prior in proposals:
            if prior['output'] == str(trial_root):
                continue
            recorded = [t for t in study.trials if t.params == dict(prior['params'], seed=prior['seed'])]
            if len(recorded) != 1 or recorded[0].state != prior['state']:
                raise ValueError('resume prior trial to reconcile its ask/tell before new work')
        oa_params = dict(params, seed=args.seed)
        matches = [trial for trial in study.trials if trial.params == oa_params]
        result_path = trial_root / 'result.json'
        if matches:
            if len(matches) != 1 or not result_path.exists():
                raise ValueError('ask/tell history and trial result disagree')
            result = read_json(result_path)
            if result.get('identity_id') != identity_id:
                raise ValueError('saved trial result identity changed')
            expected = result['objective'] if result['state'] == 'complete' else result.get('latest_val_macro_f1') if result['state'] == 'pruned' else None
            if matches[0].state != result['state'] or matches[0].value != expected:
                raise ValueError('ask/tell outcome differs from saved result')
            return result
        trial = study.ask(oa_params)
        # optim-agent 0.1.1 only persists terminal trials; this durable record bridges crashes.
        write_json(trial_root / 'ask.json', dict(number=trial.number, params=oa_params, identity_id=identity_id))
        if result_path.exists():
            result = read_json(result_path)
            if result.get('identity_id') != identity_id:
                raise ValueError('saved trial result identity changed')
        else:
            try:
                result = execute_trial(trial_root, config, identity, protocol['deadline_unix'], args.device)
            except Exception as exc:
                result = dict(state='failed', objective=None, error=repr(exc), uses_test=False)
                write_json(result_path, dict(result, identity_id=identity_id))
                tell_result(study, trial, result)
                raise
            result.update(identity_id=identity_id, finished_unix=time.time())
            write_json(result_path, result)
        tell_result(study, trial, result)
        return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study-dir', 'trial-name', 'params', 'phase', 'config', 'rationale'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    args.invocation = list(argv if argv is not None else sys.argv[1:])
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
