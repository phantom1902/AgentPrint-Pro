"""FP32 training with complete-epoch recovery and file/TensorBoard history."""
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from . import MODEL_VERSION, PREPROCESSING_VERSION
from .cache import CachedSamples, LABEL_MAP, LABEL_MAP_ID, write_json
from .config import ConfigError, fingerprint
from .metrics import classification_metrics, write_metrics
from .model import getRF
from .recovery import (append_json, archive_ahead_logs, capture_rng, publish_best,
                       record_runtime, restore_rng, save_checkpoint, validate_resume)


class MTAMDataset(CachedSamples, Dataset):
    """PyTorch map-style adapter; preparation/cache modules do not import torch."""


def setup_execution(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def supported(config):
    if config['task'] not in LABEL_MAP or config['transform'] not in ('raw', 'log1p'):
        raise ConfigError('training/evaluation requires scenario/primitive and raw/log1p')


def validate_training(config, train_samples):
    """Stop incomplete formal ontologies before allocating the RF model or run."""
    support = [0] * len(LABEL_MAP[config['task']])
    for record in train_samples.records:
        support[record['label_id']] += 1
    missing = [name for name, index in LABEL_MAP[config['task']].items() if not support[index]]
    if config['mode'] == 'formal' and missing:
        raise ConfigError(json.dumps({'reason': 'missing_training_classes',
                                      'task': config['task'], 'support': support,
                                      'missing_classes': missing}, sort_keys=True))


def protocol_metadata(config):
    return {'formal_experiment': config['mode'] == 'formal', 'mode': config['mode'],
            'task': config['task'], 'transform': config['transform'], 'seed': config['seed'],
            'precision': 'float32', 'amp': False, 'tf32': False, 'cudnn_benchmark': False,
            'deterministic_algorithms': True, 'num_workers': 0,
            'formal_execution': 'serial', 'loss': 'unweighted_cross_entropy',
            'training': config['training'], 'checkpoint_selection_split': 'val',
            'checkpoint_tie_rule': 'earlier_epoch', 'transform_tie_rule': 'raw',
            'input_cache_dtype': 'uint64', 'input_conversion': 'float32_then_transform'}


def record_selection(config, saved, checkpoint):
    """Aggregate evaluated candidates using checkpoint validation scores only.

    Each transform's most recently explicitly evaluated checkpoint is its candidate.
    A cohort is complete after both transforms are evaluated, in either split/order.
    Dataset identities and all comparison settings must agree before pairing.
    """
    cohort = {field: saved['config'][field] for field in ('task', 'mode', 'seed', 'training')}
    cohort.update({field: saved[field] for field in ('train_manifest_id', 'val_manifest_id')})
    cohort.update(label_map_id=saved['label_map_id'], model_version=saved['model_version'],
                  preprocessing_version=saved['preprocessing_version'])
    root = Path(config['paths']['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"selection-{config['task']}-{fingerprint(cohort)[:16]}.json"
    selection: dict = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'cohort': cohort, 'selection_split': 'val', 'tie_rule': 'raw', 'candidates': {}}
    candidates = selection['candidates']
    score = float(saved['best_macro_f1'])
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ConfigError('checkpoint best validation Macro-F1 must be finite and in [0,1]')
    candidates[config['transform']] = {
        'checkpoint': str(Path(checkpoint).resolve()), 'epoch': saved['epoch'],
        'validation_macro_f1': score}
    chosen = max((name for name in ('raw', 'log1p') if name in candidates),
                 key=lambda name: candidates[name]['validation_macro_f1'])
    selection.update(complete=all(name in candidates for name in ('raw', 'log1p')),
                     selected_transform=chosen)
    for name, candidate in candidates.items():
        candidate['validation_selected_main'] = name == chosen
    write_json(path, selection)
    return path, selection


def loader(config, split, shuffle=False, generator=None):
    samples = MTAMDataset(config['paths']['cache_root'], config['task'], split,
                          config['transform'])
    if not samples:
        if split == 'train':
            validate_training(config, samples)
        raise ConfigError(f'{split} has no valid samples')
    return samples, DataLoader(samples, batch_size=config['training']['batch_size'],
                               shuffle=shuffle, num_workers=0, generator=generator)


def evaluate_model(model, batches, device, task):
    model.eval()
    loss_total, truth, predicted = 0.0, [], []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for values, labels in batches:
            values, labels = values.to(device=device, dtype=torch.float32), labels.to(device)
            scores = model(values)
            loss = criterion(scores, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite evaluation loss; execution stopped')
            loss_total += float(loss) * len(labels)
            truth.extend(labels.cpu().tolist())
            predicted.extend(scores.argmax(dim=1).cpu().tolist())
    summary, per_class, matrix = classification_metrics(truth, predicted, task)
    summary['loss'] = loss_total / len(truth)
    return summary, per_class, matrix


def train(config, device='cpu', resume=None):
    supported(config)
    setup_execution(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    train_samples, train_batches = loader(config, 'train', shuffle=True, generator=generator)
    validate_training(config, train_samples)
    val_samples, val_batches = loader(config, 'val')
    identity = {'label_map_id': LABEL_MAP_ID, 'model_version': MODEL_VERSION,
                'preprocessing_version': PREPROCESSING_VERSION,
                'train_manifest_id': train_samples.metadata['manifest_id'],
                'val_manifest_id': val_samples.metadata['manifest_id']}
    for samples in (train_samples, val_samples):
        if samples.metadata['preprocessing_version'] != PREPROCESSING_VERSION:
            raise ConfigError('incompatible cache preprocessing_version')
    saved = None
    if resume is None:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run_id = f"{config['task']}-{config['transform']}-seed{config['seed']}-{timestamp}-{uuid.uuid4().hex[:8]}"
        root = Path(config['paths']['output_root']) / run_id
        root.mkdir(parents=True, exist_ok=False)
    else:
        root = Path(resume)
        if not root.is_dir() or not (root / 'last.pt').is_file():
            raise ConfigError('resume requires an existing run directory with committed last.pt')
        saved = torch.load(root / 'last.pt', map_location='cpu', weights_only=False)
        validate_resume(saved, config, identity)
        if saved['run_id'] != root.name:
            raise ConfigError('incompatible recovery run_id')
    run_id = root.name
    model = getRF(len(LABEL_MAP[config['task']])).to(device)
    settings = config['training']
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    best_f1, best_epoch, best_checkpoint = -1.0, None, None
    start_epoch, global_step = 0, 0
    if saved is not None:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch, global_step = saved['epoch'] + 1, saved['global_step']
        best_f1, best_epoch = saved['best_macro_f1'], saved['best_epoch']
        best_checkpoint = saved['best_checkpoint']
        publish_best(root, best_checkpoint)
        archive_ahead_logs(root, saved['epoch'], global_step)
        restore_rng(saved['rng'], generator)
    record_runtime(root, config, device, start_epoch - 1 if saved is not None else None)
    write_json(root / 'protocol.json', protocol_metadata(config))
    log_root = root / 'tensorboard' / run_id
    with SummaryWriter(str(log_root / 'epochs'),
                       purge_step=start_epoch if saved is not None else None) as epochs_writer, \
         SummaryWriter(str(log_root / 'steps'),
                       purge_step=global_step + 1 if saved is not None else None) as steps_writer:
        for epoch in range(start_epoch, settings['epochs']):
            started = time.monotonic()
            learning_rate = settings['learning_rate'] * settings['lr_decay'] ** (epoch / settings['epochs'])
            for group in optimizer.param_groups:
                group['lr'] = learning_rate
            model.train()
            total, count = 0.0, 0
            for values, labels in train_batches:
                values, labels = values.to(device=device, dtype=torch.float32), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(values), labels)
                if not torch.isfinite(loss):
                    raise RuntimeError('non-finite training loss; execution stopped')
                loss.backward()
                optimizer.step()
                global_step += 1
                total += float(loss.detach()) * len(labels)
                count += len(labels)
                if global_step % 50 == 0:
                    step_row = {'epoch': epoch, 'global_step': global_step,
                                'train_loss': float(loss.detach())}
                    append_json(root / 'training_steps.jsonl', step_row)
                    steps_writer.add_scalar('train_loss', step_row['train_loss'], global_step)
                    steps_writer.flush()
            summary, per_class, matrix = evaluate_model(model, val_batches, device, config['task'])
            summary.update(exclusion_reasons=val_samples.metadata.get('exclusion_reasons', {}),
                           excluded_count=val_samples.metadata.get('excluded_count', 0))
            row = {'epoch': epoch, 'train_loss': total / count, 'val_loss': summary['loss'],
                   'val_accuracy': summary['accuracy'], 'val_macro_f1': summary['macro_f1'],
                   'learning_rate': learning_rate, 'elapsed_seconds': time.monotonic() - started,
                   'train_samples': len(train_samples), 'val_samples': len(val_samples),
                   'global_step': global_step}
            if 'fixed47_macro_f1' in summary:
                row['val_fixed47_macro_f1'] = summary['fixed47_macro_f1']
            checkpoint = dict(identity, checkpoint_format=1, model=model.state_dict(),
                              epoch=epoch, config=config, run_id=run_id)
            if summary['macro_f1'] > best_f1:
                best_f1, best_epoch = summary['macro_f1'], epoch
                best_checkpoint = f'checkpoints/best-epoch-{epoch:05d}-{uuid.uuid4().hex[:8]}.pt'
                checkpoint.update(best_macro_f1=best_f1, best_epoch=best_epoch,
                                  validation_result={
                                      'summary': dict(summary, best_epoch=best_epoch,
                                                      run_id=run_id, task=config['task'],
                                                      transform=config['transform']),
                                      'per_class': per_class, 'matrix': matrix.tolist()})
                save_checkpoint(root / best_checkpoint, checkpoint)
            # Logs precede the complete-epoch commit, so any uncommitted tail is
            # recoverable through archival/purge. Never overwrite a best snapshot.
            append_json(root / 'metrics.jsonl', row)
            for key, value in row.items():
                if key != 'epoch':
                    epochs_writer.add_scalar(key, value, epoch)
            epochs_writer.flush()
            checkpoint.update(optimizer=optimizer.state_dict(), global_step=global_step,
                              best_macro_f1=best_f1, best_epoch=best_epoch,
                              best_checkpoint=best_checkpoint, rng=capture_rng(generator))
            save_checkpoint(root / 'last.pt', checkpoint)
            publish_best(root, best_checkpoint)
            print(json.dumps(row, sort_keys=True), flush=True)
    return root


def evaluate(config, device, checkpoint, split='val'):
    supported(config)
    setup_execution(config['seed'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if 'best_epoch' in saved and saved['epoch'] != saved['best_epoch']:
        raise ConfigError('evaluation requires a validation-best checkpoint; use best.pt')
    for field, expected in [('label_map_id', LABEL_MAP_ID), ('model_version', MODEL_VERSION),
                            ('preprocessing_version', PREPROCESSING_VERSION)]:
        if saved.get(field) != expected:
            raise ConfigError(f'incompatible checkpoint {field}')
    for field in ('task', 'transform'):
        if saved['config'][field] != config[field]:
            raise ConfigError(f'incompatible checkpoint {field}')
    samples, batches = loader(config, split)
    if split == 'val' and saved['val_manifest_id'] != samples.metadata['manifest_id']:
        raise ConfigError('checkpoint validation manifest differs from requested cache')
    model = getRF(len(LABEL_MAP[config['task']])).to(device)
    model.load_state_dict(saved['model'])
    summary, per_class, matrix = evaluate_model(model, batches, device, config['task'])
    summary.update(split=split, epoch=saved['epoch'], task=config['task'],
                   transform=config['transform'], manifest_id=samples.metadata['manifest_id'],
                   validation_macro_f1=saved['best_macro_f1'],
                   formal_experiment=saved['config']['mode'] == 'formal',
                   exclusion_reasons=samples.metadata.get('exclusion_reasons', {}),
                   excluded_count=samples.metadata.get('excluded_count', 0))
    selection_path, selection = record_selection(config, saved, checkpoint)
    summary.update(selection_artifact=str(selection_path),
                   validation_selected_transform=selection['selected_transform'],
                   selection_complete=selection['complete'])
    write_metrics(Path(checkpoint).parent / f'evaluation-{split}', summary, per_class, matrix)
    return summary
