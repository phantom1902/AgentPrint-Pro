"""Separate, user-approved extension of fixed-budget runs; never reads test data."""
from dataclasses import asdict, dataclass
import math
from statistics import mean


@dataclass(frozen=True)
class ConvergencePolicy:
    baseline_epochs: int = 30
    min_additional_epochs: int = 30
    patience: int = 20
    f1_min_delta: float = 0.001
    loss_window: int = 10
    loss_relative_delta: float = 0.01
    max_total_epochs: int = 200

    def __post_init__(self):
        for key in ('baseline_epochs', 'min_additional_epochs', 'patience', 'loss_window', 'max_total_epochs'):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(key + ' must be a positive integer')
        if self.max_total_epochs < self.baseline_epochs + self.min_additional_epochs:
            raise ValueError('epoch cap cannot precede minimum additional budget')
        for key in ('f1_min_delta', 'loss_relative_delta'):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(key + ' must be finite in (0,1]')


def learning_rate(epoch):
    if type(epoch) is not int or epoch < 0:
        raise ValueError('epoch must be a nonnegative integer')
    return max(1e-6, 5e-4 * 0.2 ** (epoch / 30))


def assess(rows, baseline_best, policy):
    if not math.isfinite(baseline_best) or not 0 <= baseline_best <= 1:
        raise ValueError('invalid baseline Macro-F1')
    anchor, last = baseline_best, policy.baseline_epochs - 1
    for ordinal, row in enumerate(rows):
        if type(row['epoch']) is not int or row['epoch'] != policy.baseline_epochs + ordinal:
            raise ValueError('continuation epochs must be contiguous after baseline')
        if (not math.isfinite(row['val_macro_f1']) or not 0 <= row['val_macro_f1'] <= 1
                or not math.isfinite(row['val_loss']) or row['val_loss'] < 0):
            raise ValueError('invalid validation metric')
        # Close equality avoids float representation changing the >= boundary.
        gain = row['val_macro_f1'] - anchor
        if gain >= policy.f1_min_delta or math.isclose(gain, policy.f1_min_delta, rel_tol=1e-12, abs_tol=1e-15):
            anchor, last = row['val_macro_f1'], row['epoch']
    epoch = rows[-1]['epoch'] if rows else policy.baseline_epochs - 1
    stale = epoch - last
    previous = recent = improvement = None
    enough = len(rows) >= 2 * policy.loss_window
    if enough:
        previous = mean(row['val_loss'] for row in rows[-2*policy.loss_window:-policy.loss_window])
        recent = mean(row['val_loss'] for row in rows[-policy.loss_window:])
        improvement = (previous - recent) / previous if previous > 0 else 0.0
    loss_flat = enough and improvement < policy.loss_relative_delta and not math.isclose(
        improvement, policy.loss_relative_delta, rel_tol=1e-12, abs_tol=1e-15)
    converged = len(rows) >= policy.min_additional_epochs and stale >= policy.patience and loss_flat
    capped = epoch + 1 >= policy.max_total_epochs
    return {'stop': bool(converged or capped), 'converged': bool(converged),
            'reason': 'validation_plateau' if converged else 'epoch_cap_reached' if capped else 'running',
            'total_epochs': epoch+1, 'additional_epochs': len(rows),
            'last_significant_epoch': last, 'epochs_without_significant_gain': stale,
            'significant_best_macro_f1': anchor, 'previous_mean_val_loss': previous,
            'recent_mean_val_loss': recent, 'relative_val_loss_improvement': improvement,
            'loss_plateau': bool(loss_flat), 'policy': asdict(policy)}


def continue_training(parent_run, output_root, device='cuda:0', policy=None):
    """Fork a completed baseline and resume exact model/Adam/RNG state."""
    from copy import deepcopy
    import hashlib
    import json
    import os
    from pathlib import Path
    import shutil
    import tempfile
    import time
    import uuid

    import torch
    from torch.utils.tensorboard import SummaryWriter
    from . import MODEL_VERSION, PREPROCESSING_VERSION
    from .cache import LABEL_MAP, LABEL_MAP_ID, write_json
    from .recovery import (append_json, archive_ahead_logs, capture_rng,
                           environment, publish_best, restore_rng, save_checkpoint)
    from .training import (evaluate_model, getRF, loader, protocol_metadata,
                           setup_execution, validate_training)
    from .metrics import write_metrics

    policy = policy or ConvergencePolicy()
    parent = Path(parent_run).resolve()
    output = Path(output_root).resolve()
    root = output / parent.name
    if output.is_relative_to(parent) or parent.is_relative_to(root):
        raise ValueError('continuation output must be independent of baseline')
    def read_rows(path):
        return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    baseline = torch.load(parent/'last.pt', map_location='cpu', weights_only=False)
    history = read_rows(parent/'metrics.jsonl')
    if (baseline['epoch'] != policy.baseline_epochs-1 or
            [r['epoch'] for r in history] != list(range(policy.baseline_epochs))):
        raise ValueError('parent must contain the complete baseline epoch budget')
    parent_best = Path(baseline['best_checkpoint'])
    if parent_best.is_absolute() or '..' in parent_best.parts:
        raise ValueError('invalid baseline checkpoint path')
    lineage = {'parent_run': str(parent), 'parent_last_sha256': sha(parent/'last.pt'),
               'parent_metrics_sha256': sha(parent/'metrics.jsonl'),
               'parent_best_sha256': sha(parent/parent_best), 'policy': asdict(policy),
               'baseline_best_macro_f1': baseline['best_macro_f1']}
    source_config = deepcopy(baseline['config'])
    config = deepcopy(source_config)
    config['mode'] = 'smoke'  # Explicit extension, not the fixed 30-epoch protocol.
    config['training']['epochs'] = policy.max_total_epochs
    config['paths']['output_root'] = str(output)
    cache = Path(config['paths']['cache_root']).resolve()
    if root.is_relative_to(cache) or cache.is_relative_to(root):
        raise ValueError('continuation output must not overlap cache')
    if root.exists():
        if json.loads((root/'lineage.json').read_text()) != lineage:
            raise ValueError('continuation lineage/policy differs from existing run')
    else:
        output.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.initializing-', dir=output))
        try:
            (staging/parent_best).parent.mkdir(parents=True)
            best = torch.load(parent/parent_best, map_location='cpu', weights_only=False)
            best.update(config=config, experiment_kind='convergence_extension', continuation=lineage)
            save_checkpoint(staging/parent_best, best)
            baseline.update(config=config, experiment_kind='convergence_extension', continuation=lineage)
            save_checkpoint(staging/'last.pt', baseline)
            shutil.copy2(parent/'metrics.jsonl', staging/'metrics.jsonl')
            if (parent/'training_steps.jsonl').exists():
                shutil.copy2(parent/'training_steps.jsonl', staging/'training_steps.jsonl')
            write_json(staging/'lineage.json', lineage)
            os.replace(staging, root)
        finally:
            if staging.exists(): shutil.rmtree(staging)
    saved = torch.load(root/'last.pt', map_location='cpu', weights_only=False)
    if saved.get('continuation') != lineage or saved['config'] != config:
        raise ValueError('continuation checkpoint policy/configuration changed')
    setup_execution(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    train_samples, train_batches = loader(config, 'train', True, generator)
    val_samples, val_batches = loader(config, 'val')
    validate_training(source_config, train_samples)
    identity = {'label_map_id': LABEL_MAP_ID, 'model_version': MODEL_VERSION,
                'preprocessing_version': PREPROCESSING_VERSION,
                'train_manifest_id': train_samples.metadata['manifest_id'],
                'val_manifest_id': val_samples.metadata['manifest_id']}
    if any(saved.get(k) != v for k, v in identity.items()):
        raise ValueError('continuation checkpoint/cache identity differs')
    if any(s.metadata['preprocessing_version'] != PREPROCESSING_VERSION for s in (train_samples, val_samples)):
        raise ValueError('incompatible cache preprocessing_version')
    archive_ahead_logs(root, saved['epoch'], saved['global_step'])
    rows = read_rows(root/'metrics.jsonl')
    if [r['epoch'] for r in rows] != list(range(saved['epoch']+1)):
        raise ValueError('checkpoint and committed metrics epochs differ')
    if rows[:policy.baseline_epochs] != history:
        raise ValueError('continuation baseline history changed')
    publish_best(root, saved['best_checkpoint'])
    protocol = protocol_metadata(config)
    protocol.update(experiment_kind='convergence_extension', formal_experiment=False,
                    convergence_policy=asdict(policy), learning_rate_rule='max(1e-6,5e-4*0.2**(epoch/30))',
                    parent_run=str(parent), uses_test=False)
    write_json(root/'protocol.json', protocol)
    write_json(root/'resolved_config.json', config)
    status = assess(rows[policy.baseline_epochs:], lineage['baseline_best_macro_f1'], policy)
    write_json(root/'convergence.json', status)
    if not status['stop']:
        model = getRF(len(LABEL_MAP[config['task']])).to(device)
        model.load_state_dict(saved['model'])
        settings = config['training']
        optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'], generator)
        best_f1, best_epoch, best_checkpoint = saved['best_macro_f1'], saved['best_epoch'], saved['best_checkpoint']
        global_step = saved['global_step']
        append_json(root/'runtime_history.jsonl', {'environment': environment(), 'device': str(device),
                    'resume_epoch': saved['epoch'], 'lineage': lineage, 'protocol': protocol})
        criterion = torch.nn.CrossEntropyLoss()
        with SummaryWriter(str(root/'tensorboard/epochs'), purge_step=saved['epoch']+1) as writer:
            for epoch in range(saved['epoch']+1, policy.max_total_epochs):
                started = time.monotonic()
                lr = learning_rate(epoch)
                for group in optimizer.param_groups: group['lr'] = lr
                model.train(); total = count = 0
                for values, labels in train_batches:
                    values, labels = values.to(device=device,dtype=torch.float32), labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(values), labels)
                    if not torch.isfinite(loss): raise RuntimeError('non-finite training loss')
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise RuntimeError('non-finite training gradient')
                    optimizer.step(); global_step += 1
                    total += float(loss.detach())*len(labels); count += len(labels)
                    if global_step % 50 == 0:
                        step = {'epoch':epoch,'global_step':global_step,'train_loss':float(loss.detach())}
                        append_json(root/'training_steps.jsonl', step)
                summary, per_class, matrix = evaluate_model(model,val_batches,device,config['task'])
                summary.update(exclusion_reasons=val_samples.metadata.get('exclusion_reasons',{}),
                               excluded_count=val_samples.metadata.get('excluded_count',0))
                row = {'epoch':epoch,'train_loss':total/count,'val_loss':summary['loss'],
                       'val_accuracy':summary['accuracy'],'val_macro_f1':summary['macro_f1'],
                       'learning_rate':lr,'elapsed_seconds':time.monotonic()-started,
                       'train_samples':len(train_samples),'val_samples':len(val_samples),'global_step':global_step}
                if 'fixed47_macro_f1' in summary: row['val_fixed47_macro_f1'] = summary['fixed47_macro_f1']
                if any(not math.isfinite(v) for v in row.values() if isinstance(v,float)):
                    raise RuntimeError('non-finite epoch metric')
                checkpoint = dict(identity,checkpoint_format=1,model=model.state_dict(),epoch=epoch,
                                  config=config,run_id=root.name,experiment_kind='convergence_extension',continuation=lineage)
                if summary['macro_f1'] > best_f1:
                    best_f1,best_epoch = summary['macro_f1'],epoch
                    best_checkpoint = f'checkpoints/best-epoch-{epoch:05d}-{uuid.uuid4().hex[:8]}.pt'
                    checkpoint.update(best_macro_f1=best_f1,best_epoch=best_epoch,validation_result={
                        'summary':dict(summary,best_epoch=best_epoch,task=config['task'],transform=config['transform'],
                                       formal_experiment=False,experiment_kind='convergence_extension'),
                        'per_class':per_class,'matrix':matrix.tolist()})
                    save_checkpoint(root/best_checkpoint,checkpoint)
                append_json(root/'metrics.jsonl',row)
                for key,value in row.items():
                    if key!='epoch': writer.add_scalar(key,value,epoch)
                writer.flush()
                rows.append(row)
                status = assess(rows[policy.baseline_epochs:],lineage['baseline_best_macro_f1'],policy)
                checkpoint.update(optimizer=optimizer.state_dict(),global_step=global_step,best_macro_f1=best_f1,
                    best_epoch=best_epoch,best_checkpoint=best_checkpoint,rng=capture_rng(generator),convergence=status)
                save_checkpoint(root/'last.pt',checkpoint)
                publish_best(root,best_checkpoint)
                write_json(root/'convergence.json',status)
                print(json.dumps(row,sort_keys=True),flush=True)
                if status['stop']: break
    best = torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    result = best['validation_result']
    report = dict(result['summary'],task=config['task'],transform=config['transform'],
                  formal_experiment=False,experiment_kind='convergence_extension',uses_test=False,
                  best_epoch=best['epoch'],validation_macro_f1=best['best_macro_f1'])
    write_metrics(root/'validation-best',report,result['per_class'],__import__('numpy').asarray(result['matrix']))
    if sha(parent/'last.pt') != lineage['parent_last_sha256'] or sha(parent/'metrics.jsonl') != lineage['parent_metrics_sha256']:
        raise RuntimeError('baseline changed during continuation')
    return root


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run',required=True)
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args(argv)
    root = continue_training(args.parent_run,args.output_root,args.device)
    print(json.dumps({'run_dir':str(root),'experiment_kind':'convergence_extension'}))


if __name__ == '__main__':
    main()
