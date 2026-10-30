"""Explicit optim-agent trials for scenario/raw; production training is unchanged."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch
from torch import nn

from . import MODEL_VERSION, PREPROCESSING_VERSION
from .cache import LABEL_MAP_ID, write_json
from .convergence import ConvergencePolicy, assess
from .model import RF, getRF, make_layers
from .recovery import (append_json, archive_ahead_logs, capture_rng, environment,
                       atomic_write, restore_rng, save_checkpoint)
from .training import evaluate_model, loader, setup_execution


DEFAULT_PARAMS = dict(learning_rate=0.0005, weight_decay=0.001, decay_per_30=0.2,
                      batch_size=200, dropout_2d=0.1, dropout_1d=0.3, width=1.0)


def validate_params(params):
    if params.keys() != DEFAULT_PARAMS.keys():
        raise ValueError('parameters must match the approved seven-dimensional space')
    bounds = {'learning_rate': (1e-5, 0.005), 'weight_decay': (0, 0.01),
              'decay_per_30': (0.1, 1), 'dropout_2d': (0, 0.3), 'dropout_1d': (0, 0.5)}
    for key, (low, high) in bounds.items():
        value = params[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError('out of bounds: ' + key)
    if type(params['batch_size']) is not int or params['batch_size'] not in (32, 64, 128, 200, 256):
        raise ValueError('invalid batch_size')
    if isinstance(params['width'], bool) or params['width'] not in (0.5, 1.0, 2.0):
        raise ValueError('invalid width')


def build_model(params):
    validate_params(params)
    if params['width'] == 1:
        model = getRF(5)
    else:
        width = params['width']
        # Preserve RF's reshape topology: 64 channels become 32 with doubled length.
        model = RF.__new__(RF)
        nn.Module.__init__(model)
        model.first_layer_in_channel = 2
        model.first_layer_out_channel = int(32 * width)
        layers = []
        for i, o, pool in [(2, int(32*width), None), (int(32*width), int(32*width), (1, 3)),
                           (int(32*width), int(64*width), None), (int(64*width), int(64*width), (2, 2))]:
            layers.extend([nn.Conv2d(i, o, (3, 6), padding=(1, 1)), nn.BatchNorm2d(o), nn.ReLU()])
            if pool:
                layers.extend([nn.MaxPool2d(pool), nn.Dropout(0.1)])
        model.first_layer = nn.Sequential(*layers)
        model.features = make_layers([int(128*width), int(128*width), 'M',
                                      int(256*width), int(256*width), 'M', int(512*width), 5],
                                     in_channels=int(32*width))
        model.class_num = 5
        model.classifier = nn.AdaptiveAvgPool1d(1)
        model._initialize_weights()
    for module in model.first_layer.modules():
        if isinstance(module, nn.Dropout):
            module.p = params['dropout_2d']
    for module in model.features.modules():
        if isinstance(module, nn.Dropout):
            module.p = params['dropout_1d']
    return model


def stop_status(rows, policy=None):
    policy = policy or ConvergencePolicy()
    if len(rows) < policy.baseline_epochs:
        return dict(stop=False, converged=False, reason='running', total_epochs=len(rows))
    baseline_best = max(r['val_macro_f1'] for r in rows[:policy.baseline_epochs])
    return assess(rows[policy.baseline_epochs:], baseline_best, policy)


def evaluate_trial(params, root, cache_root, seed=42, device='cuda:0', deadline=None, policy=None):
    """Run/resume one explicit point; deadline pruning never becomes a complete trial."""
    validate_params(params)
    if type(seed) is not int or seed not in (42, 43, 44):
        raise ValueError('only approved seeds 42, 43, 44 are permitted')
    policy = policy or ConvergencePolicy()
    root, cache_root = Path(root).resolve(), Path(cache_root).resolve()
    if root.is_relative_to(cache_root) or cache_root.is_relative_to(root):
        raise ValueError('trial and cache paths must not overlap')
    setup_execution(seed)
    generator = torch.Generator().manual_seed(seed)
    config = dict(task='scenario', transform='raw', seed=seed,
                  paths={'cache_root': str(cache_root)}, training={'batch_size': params['batch_size']})
    train_samples, train_batches = loader(config, 'train', True, generator)
    val_samples, val_batches = loader(config, 'val')
    for samples in (train_samples, val_samples):
        if samples.metadata['preprocessing_version'] != PREPROCESSING_VERSION or any(n == 0 for n in samples.metadata['support']):
            raise ValueError('requires compatible cache with all five classes')
    identity = dict(params=params, seed=seed, task='scenario', transform='raw',
                    policy=asdict(policy), label_map_id=LABEL_MAP_ID,
                    model_version=MODEL_VERSION, tuning_version=1,
                    preprocessing_version=PREPROCESSING_VERSION,
                    train_manifest_id=train_samples.metadata['manifest_id'],
                    val_manifest_id=val_samples.metadata['manifest_id'])
    root.mkdir(parents=True, exist_ok=True)
    if (root/'identity.json').exists():
        if json.loads((root/'identity.json').read_text()) != identity:
            raise ValueError('trial identity changed')
    else:
        write_json(root/'identity.json', identity)
    if (root/'result.json').exists():
        return json.loads((root/'result.json').read_text())
    model = build_model(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    rows, start, global_step = [], 0, 0
    best_f1, best_epoch, best_checkpoint = -1.0, None, None
    if (root/'last.pt').exists():
        saved = torch.load(root/'last.pt', map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('checkpoint identity changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'], generator)
        start, global_step = saved['epoch']+1, saved['global_step']
        rows, best_f1, best_epoch = saved['rows'], saved['best_f1'], saved['best_epoch']
        best_checkpoint = saved['best_checkpoint']
        archive_ahead_logs(root, saved['epoch'], global_step)
    else:
        # A killed first epoch has no committed history; discard only its log tail.
        archive_ahead_logs(root, -1, 0)
    write_json(root/'environment.json', dict(environment(), device=device, precision='FP32', uses_test=False))
    status = stop_status(rows, policy)
    expired = False
    for epoch in range(start, policy.max_total_epochs):
        if status['stop']:
            break
        started = time.monotonic()
        lr = max(1e-6, params['learning_rate'] * params['decay_per_30'] ** (epoch/30))
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        total = count = 0
        for values, labels in train_batches:
            if deadline is not None and time.time() >= deadline:
                expired = True
                break
            values, labels = values.to(device=device, dtype=torch.float32), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('non-finite gradient')
            optimizer.step()
            global_step += 1
            total += float(loss.detach()) * len(labels)
            count += len(labels)
        if expired:
            break
        summary, per_class, matrix = evaluate_model(model, val_batches, device, 'scenario')
        row = dict(epoch=epoch, train_loss=total/count, val_loss=summary['loss'],
                   val_macro_f1=summary['macro_f1'], val_accuracy=summary['accuracy'],
                   learning_rate=lr, train_samples=len(train_samples), val_samples=len(val_samples),
                   global_step=global_step, elapsed_seconds=time.monotonic()-started)
        if any(not math.isfinite(v) for v in row.values()):
            raise RuntimeError('non-finite epoch metric')
        if summary['macro_f1'] > best_f1:
            best_f1, best_epoch = summary['macro_f1'], epoch
            best_checkpoint = f'checkpoints/best-epoch-{epoch:05d}.pt'
            (root/'checkpoints').mkdir(exist_ok=True)
            save_checkpoint(root/best_checkpoint, dict(model=model.state_dict(), identity=identity,
                            epoch=epoch, best_f1=best_f1, summary=summary,
                            per_class=per_class, matrix=matrix.tolist()))
        rows.append(row)
        append_json(root/'metrics.jsonl', row)
        status = stop_status(rows, policy)
        save_checkpoint(root/'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        identity=identity, epoch=epoch, rows=rows, global_step=global_step,
                        best_f1=best_f1, best_epoch=best_epoch, best_checkpoint=best_checkpoint,
                        rng=capture_rng(generator)))
        print(json.dumps(row, sort_keys=True), flush=True)
    result = dict(identity=identity, state='pruned' if expired else 'complete',
                  reason='study_deadline' if expired else status['reason'],
                  epochs=len(rows), best_val_macro_f1=best_f1 if rows else None,
                  best_epoch_one_based=best_epoch+1 if best_epoch is not None else None,
                  latest_val_macro_f1=rows[-1]['val_macro_f1'] if rows else None,
                  train_samples=len(train_samples), val_samples=len(val_samples),
                  parameters=sum(p.numel() for p in model.parameters()), uses_test=False,
                  elapsed_seconds=sum(r['elapsed_seconds'] for r in rows), stopping=status)
    if best_checkpoint is not None:
        from .metrics import write_metrics
        import numpy as np
        import shutil
        def copy_best(handle):
            with (root/best_checkpoint).open('rb') as source:
                shutil.copyfileobj(source, handle)
        atomic_write(root/'best.pt', copy_best)
        best = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
        write_metrics(root/'validation-best', dict(best['summary'], uses_test=False),
                      best['per_class'], np.asarray(best['matrix']))
    write_json(root/'result.json', result)
    return result


def main(argv=None):
    import argparse
    import fcntl
    import sys
    import optim_agent as oa

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', required=True)
    parser.add_argument('--trial-name', required=True)
    parser.add_argument('--params', required=True, help='JSON file with explicit seven-dimensional parameters')
    parser.add_argument('--rationale', required=True)
    parser.add_argument('--phase', choices=['baseline', 'search', 'repeat'], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args(argv)
    root = Path(args.study_dir).resolve()
    if '.optim-agent-runs' not in root.parts:
        raise ValueError('study must be in ignored .optim-agent-runs')
    if Path(args.trial_name).name != args.trial_name or args.trial_name in ('.', '..'):
        raise ValueError('trial-name must be a single path component')
    root.mkdir(parents=True, exist_ok=True)
    with (root/'study.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        params = json.loads(Path(args.params).read_text())
        validate_params(params)
        if args.phase == 'baseline' and (params != DEFAULT_PARAMS or args.seed != 42):
            raise ValueError('baseline must use the original point and seed 42')
        if args.phase != 'repeat' and args.seed != 42:
            raise ValueError('search uses seed 42')
        if args.phase == 'repeat' and args.seed not in (43, 44):
            raise ValueError('repeats use seeds 43 or 44')
        protocol_file = root/'protocol.json'
        if not protocol_file.exists():
            if args.phase != 'baseline':
                raise ValueError('first trial must establish baseline')
            now = time.time()
            write_json(protocol_file, dict(started_unix=now, deadline_unix=now+3600,
                       max_trials=17, max_search_trials=12, direction='maximize',
                       objective='best_validation_macro_f1', uses_test=False,
                       cache_root=str(Path(args.cache_root).resolve())))
        protocol = json.loads(protocol_file.read_text())
        if protocol['cache_root'] != str(Path(args.cache_root).resolve()):
            raise ValueError('study cache root changed')
        if time.time() >= protocol['deadline_unix']:
            raise RuntimeError('approved study wall-time budget exhausted')
        current_manifests = {s: json.loads((Path(args.cache_root)/'scenario'/s/'metadata.json').read_text())['manifest_id']
                             for s in ('train', 'val')}
        if 'manifests' not in protocol:
            protocol['manifests'] = current_manifests
            write_json(protocol_file, protocol)
        elif protocol['manifests'] != current_manifests:
            raise ValueError('study train/val manifests changed')
        study = oa.create_study(direction='maximize', storage=root/'study.json', seed=42)
        root_trial = root/args.trial_name
        proposal = dict(params=params, seed=args.seed, phase=args.phase, rationale=args.rationale,
                        command=[sys.executable, '-m', 'agentprint.tuning', *(argv if argv is not None else sys.argv[1:])])
        oa_params = dict(params, seed=args.seed)
        matches = [t for t in study.trials if t.params == oa_params]
        if matches:
            if (root_trial/'result.json').exists():
                print(json.dumps(json.loads((root_trial/'result.json').read_text()), sort_keys=True))
                return
            raise ValueError('duplicate point/seed already recorded')
        prior = list(root.glob('*/proposal.json'))
        existing = root_trial/'proposal.json'
        if existing.exists():
            old = json.loads(existing.read_text())
            if any(old[k] != proposal[k] for k in ('params', 'seed', 'phase')):
                raise ValueError('existing trial proposal changed')
        else:
            if len(study.trials) >= 17:
                raise RuntimeError('approved trial count exhausted')
            if args.phase == 'search' and sum(json.loads(p.read_text())['phase'] == 'search' for p in prior) >= 12:
                raise RuntimeError('12 new configurations exhausted')
            if args.phase == 'repeat' and sum(json.loads(p.read_text())['phase'] == 'repeat' for p in prior) >= 4:
                raise RuntimeError('four repeat slots exhausted')
            if args.phase == 'baseline' and prior:
                raise ValueError('baseline already proposed')
            write_json(existing, proposal)
        trial = study.ask(oa_params)
        try:
            result = evaluate_trial(params, root_trial, args.cache_root, args.seed,
                                    args.device, protocol['deadline_unix'])
        except Exception as exc:
            write_json(root_trial/'failure.json', dict(error=repr(exc), params=oa_params))
            study.tell(trial, state='failed')
            raise
        if result['state'] == 'pruned':
            if result['latest_val_macro_f1'] is not None:
                trial.report(result['latest_val_macro_f1'], step=result['epochs']-1)
            study.tell(trial, state='pruned')
        else:
            study.tell(trial, result['best_val_macro_f1'])
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
