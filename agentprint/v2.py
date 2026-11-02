"""Version 2 extension boundary, independent of the genuine v1 protocol.

Supports explicit IAT features and single, replacement, and prepend models
without changing the historical v1 configuration or recovery protocol.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, cast
import uuid

from . import MODEL_VERSION, PREPROCESSING_VERSION
from .cache import LABEL_MAP_ID, write_json
from .config import ConfigError, SCHEMA, fingerprint


FEATURE_VERSION = 'agentprint-iat-features-v1'
MODEL_EXTENSION_VERSION = 'agentprint-multikernel-v1'
SCHEMA_V2: dict[str, Any] = dict(SCHEMA, feature={
    'version': int, 'iat_enabled': bool, 'fit_artifact': str, 'iat_cache_root': str},
    model={'version': int, 'variant': str, 'width': (int, float),
           'dropout_2d': (int, float), 'dropout_1d': (int, float)},
    training=dict(cast(dict[str, Any], SCHEMA['training']),
                  min_learning_rate=(int, float), numerical_threads=int))


def _fields(config, schema, prefix=''):
    if not isinstance(config, dict):
        raise ConfigError(f'{prefix} must be a mapping')
    for key in config.keys() - schema.keys():
        raise ConfigError(f'unknown configuration field: {prefix}{key}')
    for key, expected in schema.items():
        name = prefix + key
        if key not in config:
            raise ConfigError(f'missing configuration field: {name}')
        value = config[key]
        if isinstance(expected, dict):
            _fields(value, expected, name + '.')
        elif expected is bool:
            if type(value) is not bool:
                raise ConfigError(f'illegal type for {name}')
        elif isinstance(value, bool) or not isinstance(value, expected):
            raise ConfigError(f'illegal type for {name}')


def validate_config(config, overrides=None):
    """Validate the explicit v2 contract without injecting any v1 defaults."""
    _fields(config, SCHEMA_V2)
    if config['schema_version'] != 2 or config['task'] != 'scenario':
        raise ConfigError('v2 requires schema_version=2 and task=scenario')
    if config['transform'] not in ('raw', 'log1p') or config['mode'] not in ('smoke', 'formal', 'tuning'):
        raise ConfigError('v2 requires raw/log1p and smoke/formal/tuning')
    if not 0 <= config['seed'] < 2**32:
        raise ConfigError('seed must be an integer in [0, 2**32)')
    prep, feature, model, settings = (config[name] for name in
                                       ('preprocessing', 'feature', 'model', 'training'))
    if prep['bins'] != 1800 or prep['shard_size'] != 1024:
        raise ConfigError('base preprocessing requires bins=1800 and shard_size=1024')
    if (not prep['splits'] or any(not isinstance(s, str) for s in prep['splits'])
            or len(set(prep['splits'])) != len(prep['splits'])
            or any(s not in ('train', 'val', 'test') for s in prep['splits'])):
        raise ConfigError('preprocessing.splits must contain unique official split names')
    if feature['version'] != 1 or model['version'] != 1:
        raise ConfigError('unsupported v2 feature/model definition version')
    if feature['iat_enabled']:
        if not feature['fit_artifact'].strip() or not feature['iat_cache_root'].strip():
            raise ConfigError('enabled IAT requires feature.fit_artifact and feature.iat_cache_root')
    elif feature['fit_artifact'] or feature['iat_cache_root']:
        raise ConfigError('disabled IAT requires empty fit_artifact and iat_cache_root')
    if model['variant'] not in ('single', 'replace', 'prepend'):
        raise ConfigError('unknown model.variant')
    if model['width'] not in (0.5, 1.0, 2.0):
        raise ConfigError('model.width must be 0.5, 1, or 2')
    for name in ('dropout_2d', 'dropout_1d'):
        if not math.isfinite(model[name]) or not 0 <= model[name] < 1:
            raise ConfigError(f'illegal model.{name}')
    if settings['epochs'] < 1 or settings['batch_size'] < 1:
        raise ConfigError('training epochs and batch_size must be positive')
    for name in ('learning_rate', 'weight_decay', 'lr_decay', 'min_learning_rate'):
        if not math.isfinite(settings[name]) or settings[name] < 0:
            raise ConfigError(f'illegal training.{name}')
    if (not 0 < settings['min_learning_rate'] <= settings['learning_rate']
            or not 0 < settings['lr_decay'] <= 1):
        raise ConfigError('invalid training learning-rate schedule')
    if (settings['optimizer'] != 'adam' or settings['num_workers'] != 0
            or settings['numerical_threads'] != 4):
        raise ConfigError('v2 requires adam, num_workers=0, numerical_threads=4')
    for name, value in (overrides or {}).items():
        if name not in ('data_root', 'cache_root', 'output_root'):
            raise ConfigError(f'unsupported runtime override: {name}')
        if value is not None:
            config['paths'][name] = str(value)
    if any(not value.strip() for value in config['paths'].values()):
        raise ConfigError('paths must not be empty')
    return config


def supported(config):
    if config['model']['variant'] not in ('single', 'replace', 'prepend'):
        raise ConfigError('model variant is not implemented yet: ' + config['model']['variant'])


def feature_definition(config):
    return dict(version=FEATURE_VERSION, base_version=PREPROCESSING_VERSION,
                iat_enabled=config['feature']['iat_enabled'],
                in_channels=10 if config['feature']['iat_enabled'] else 2,
                directions=2, bins=1800, dtype='uint64',
                channel_order=['packets', 'ip_bytes'] +
                ([f'iat_bucket_{i}' for i in range(8)] if config['feature']['iat_enabled'] else []))


def build_model(definition, in_channels=2):
    """Public fixed-five-class model interface; disabled defaults preserve RF."""
    from .tuning import DEFAULT_PARAMS, build_model as build_rf
    _fields(definition, SCHEMA_V2['model'], 'model.')
    if definition['version'] != 1 or definition['variant'] not in ('single', 'replace', 'prepend'):
        raise ConfigError('model definition is not implemented')
    params = dict(DEFAULT_PARAMS, **{name: definition[name] for name in
                                   ('width', 'dropout_2d', 'dropout_1d')})
    if definition['variant'] == 'replace':
        from .multikernel import build_replacement_model
        return build_replacement_model(params, in_channels=in_channels)
    if definition['variant'] == 'prepend':
        from .prepend import build_prepend_model
        return build_prepend_model(params, in_channels=in_channels)
    return build_rf(params, in_channels=in_channels)


def prepare(config):
    supported(config)
    if config['feature']['iat_enabled']:
        from .iat_cache import prepare as prepare_iat
        return prepare_iat(config)
    from .preparation import prepare as prepare_base
    return prepare_base(config)


def loader(config, split, shuffle=False, generator=None):
    from .training import loader as base_loader
    supported(config)
    if config['feature']['iat_enabled']:
        from torch.utils.data import DataLoader, Dataset
        from .iat_cache import IATSamples
        class IATDataset(IATSamples, Dataset):
            """Torch adapter for the aggregate-only supplemental cache module."""
        samples = IATDataset(config, split)
        if not samples:
            raise ConfigError(f'{split} has no valid samples')
        batches: DataLoader = DataLoader(samples, batch_size=config['training']['batch_size'],
                             shuffle=shuffle, num_workers=0, generator=generator)
    else:
        # V2 binds the actual base manifest/source/array contract even when the
        # supplemental feature is disabled. Keep the genuine v1 loader intact.
        from .iat import selected_base_records
        selected_base_records(config, split)
        samples, batches = base_loader(config, split, shuffle, generator)
    if samples.metadata['preprocessing_version'] != PREPROCESSING_VERSION:
        raise ConfigError('incompatible cache preprocessing_version')
    return samples, batches


def experiment_identity(config, train_samples, val_samples):
    enabled = config['feature']['iat_enabled']
    if enabled and train_samples.fit_fingerprint != val_samples.fit_fingerprint:
        raise ConfigError('incompatible IAT train/validation fit fingerprints')
    return dict(schema_version=2, task=config['task'], mode=config['mode'], seed=config['seed'],
                transform=config['transform'], training=deepcopy(config['training']),
                label_map_id=LABEL_MAP_ID, model_version=MODEL_VERSION,
                model_extension_version=MODEL_EXTENSION_VERSION,
                model_definition=deepcopy(config['model']),
                feature_definition=feature_definition(config),
                fit_fingerprint=train_samples.fit_fingerprint if enabled else None,
                iat_cache={split: deepcopy(samples.iat_metadata) for split, samples in
                           (('train', train_samples), ('val', val_samples))} if enabled else None,
                base_cache={split: deepcopy(samples.metadata) for split, samples in
                            (('train', train_samples), ('val', val_samples))})


def validate_checkpoint(saved):
    """Reject inconsistent saved definitions before rebuilding or resuming."""
    from .v2_protocol import validate_formal
    if saved.get('checkpoint_format') != 2:
        raise ConfigError('incompatible v2 checkpoint format')
    identity = saved.get('identity', {})
    if saved.get('experiment_id') != fingerprint(identity):
        raise ConfigError('incompatible checkpoint identity fingerprint')
    config = validate_config(deepcopy(saved['config']))
    validate_formal(config)
    for name in ('task', 'mode', 'seed', 'transform', 'training'):
        if identity.get(name) != config[name]:
            raise ConfigError('incompatible checkpoint identity ' + name)
    expected = {'schema_version': 2, 'label_map_id': LABEL_MAP_ID,
                'model_version': MODEL_VERSION, 'model_extension_version': MODEL_EXTENSION_VERSION,
                'model_definition': config['model'], 'feature_definition': feature_definition(config)}
    if not config['feature']['iat_enabled']:
        expected.update(fit_fingerprint=None, iat_cache=None)
    else:
        fit_id, caches = identity.get('fit_fingerprint'), identity.get('iat_cache')
        if (not isinstance(fit_id, str) or len(fit_id) != 64
                or not isinstance(caches, dict) or set(caches) != {'train', 'val'}):
            raise ConfigError('incompatible checkpoint identity IAT fit/cache')
        for split, metadata in caches.items():
            if (not isinstance(metadata, dict) or metadata.get('fit_fingerprint') != fit_id
                    or metadata.get('base_cache_identity') != identity.get('base_cache', {}).get(split)
                    or metadata.get('scope') != ('smoke' if config['mode'] == 'smoke' else 'full')
                    or metadata.get('fingerprint') != fingerprint({k: v for k, v in metadata.items() if k != 'fingerprint'})):
                raise ConfigError('incompatible checkpoint identity IAT cache ' + split)
    for name, value in expected.items():
        if identity.get(name) != value:
            raise ConfigError('incompatible checkpoint identity ' + name)
    for name in ('model_definition', 'feature_definition'):
        if saved.get(name) != identity[name]:
            raise ConfigError('incompatible checkpoint identity ' + name)


def train(config, device='cpu', resume=None):
    import torch
    from .recovery import (append_json, archive_ahead_logs, capture_rng, publish_best,
                           record_runtime, restore_rng, save_checkpoint)
    from .training import evaluate_model, setup_execution, validate_training
    from .v2_protocol import learning_rate, stop_status, validate_formal
    supported(config)
    validate_formal(config)
    torch.set_num_threads(config['training']['numerical_threads'])
    setup_execution(config['seed'])
    generator = torch.Generator().manual_seed(config['seed'])
    train_samples, train_batches = loader(config, 'train', True, generator)
    validate_training(config, train_samples)
    val_samples, val_batches = loader(config, 'val')
    identity = experiment_identity(config, train_samples, val_samples)
    experiment_id = fingerprint(identity)
    saved = None
    if resume is None:
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        root = Path(config['paths']['output_root']) / f'v2-{experiment_id[:12]}-{timestamp}-{uuid.uuid4().hex[:8]}'
        root.mkdir(parents=True, exist_ok=False)
    else:
        root = Path(resume)
        if not (root / 'last.pt').is_file():
            raise ConfigError('resume requires an existing run directory with committed last.pt')
        saved = torch.load(root / 'last.pt', map_location='cpu', weights_only=False)
        validate_checkpoint(saved)
        if saved.get('identity') != identity:
            raise ConfigError('incompatible v2 recovery identity')
        if saved['run_id'] != root.name:
            raise ConfigError('incompatible recovery run_id')
    definition = deepcopy(config['model'])
    features = feature_definition(config)
    model = build_model(definition, features['in_channels']).to(device)
    settings = config['training']
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'],
                                 weight_decay=settings['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    best_f1, best_epoch, best_checkpoint = -1.0, None, None
    start_epoch, global_step, rows = 0, 0, []
    if saved is not None:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch, global_step = saved['epoch'] + 1, saved['global_step']
        best_f1, best_epoch = saved['best_macro_f1'], saved['best_epoch']
        best_checkpoint, rows = saved['best_checkpoint'], saved['rows']
        if (type(saved['epoch']) is not int or saved['epoch'] < 0
                or len(rows) != start_epoch or start_epoch > settings['epochs']):
            raise ConfigError('incompatible recovery epoch history / target total epochs')
        status = stop_status(rows, config['mode'], settings['epochs'])
        if saved.get('convergence', status) != status:
            raise ConfigError('incompatible recovery convergence history')
        publish_best(root, best_checkpoint)
        archive_ahead_logs(root, saved['epoch'], global_step)
        committed_rows = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()
                          if line.strip()]
        if committed_rows != rows:
            raise ConfigError('checkpoint and committed metrics history differ')
        restore_rng(saved['rng'], generator)
    record_runtime(root, config, device, start_epoch - 1 if saved is not None else None)
    write_json(root / 'identity.json', dict(identity=identity, experiment_id=experiment_id))
    status = stop_status(rows, config['mode'], settings['epochs'])
    write_json(root / 'protocol.json', dict(schema_version=2, precision='float32', amp=False,
        tf32=False, deterministic_algorithms=True, num_workers=0, numerical_threads=4,
        cudnn_benchmark=False, mode=config['mode'], formal_experiment=config['mode'] == 'formal',
        convergence_policy=status['policy'],
        learning_rate_rule='max(min_learning_rate,learning_rate*lr_decay**(epoch/30))',
        selection_split='val', checkpoint_tie_rule='earlier_epoch',
        loss='unweighted_cross_entropy', training=settings, uses_test=False))
    write_json(root / 'convergence.json', status)
    for epoch in range(start_epoch, settings['epochs']):
        if status['stop']:
            break
        started = time.monotonic()
        lr = learning_rate(epoch, settings)
        for group in optimizer.param_groups:
            group['lr'] = lr
        model.train()
        total, count = 0.0, 0
        for values, labels in train_batches:
            values, labels = values.to(device=device, dtype=torch.float32), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), labels)
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite training loss; execution stopped')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('non-finite gradient; execution stopped')
            optimizer.step()
            global_step += 1
            total += float(loss.detach()) * len(labels)
            count += len(labels)
        summary, per_class, matrix = evaluate_model(model, val_batches, device, 'scenario')
        row = dict(epoch=epoch, train_loss=total / count, val_loss=summary['loss'],
                   val_accuracy=summary['accuracy'], val_macro_f1=summary['macro_f1'],
                   learning_rate=lr, global_step=global_step,
                   elapsed_seconds=time.monotonic() - started,
                   train_samples=len(train_samples), val_samples=len(val_samples))
        if any(not math.isfinite(value) for value in row.values() if isinstance(value, float)):
            raise RuntimeError('non-finite epoch metric; execution stopped')
        checkpoint = dict(checkpoint_format=2, identity=identity, experiment_id=experiment_id,
                          config=deepcopy(config), model_definition=definition,
                          feature_definition=features, model=model.state_dict(),
                          epoch=epoch, run_id=root.name,
                          parameters=sum(p.numel() for p in model.parameters()))
        if summary['macro_f1'] > best_f1:
            best_f1, best_epoch = summary['macro_f1'], epoch
            best_checkpoint = f'checkpoints/best-epoch-{epoch:05d}-{uuid.uuid4().hex[:8]}.pt'
            checkpoint.update(best_macro_f1=best_f1, best_epoch=best_epoch,
                              validation_result={'summary': dict(summary, best_epoch=epoch),
                                                 'per_class': per_class, 'matrix': matrix.tolist()})
            save_checkpoint(root / best_checkpoint, checkpoint)
        rows.append(row)
        status = stop_status(rows, config['mode'], settings['epochs'])
        append_json(root / 'metrics.jsonl', row)
        checkpoint.update(optimizer=optimizer.state_dict(), global_step=global_step,
                          best_macro_f1=best_f1, best_epoch=best_epoch,
                          best_checkpoint=best_checkpoint, rows=rows, rng=capture_rng(generator),
                          convergence=status)
        save_checkpoint(root / 'last.pt', checkpoint)
        publish_best(root, best_checkpoint)
        write_json(root / 'convergence.json', status)
        print(json.dumps(row, sort_keys=True), flush=True)
    write_json(root / 'result.json', dict(experiment_id=experiment_id, epochs=len(rows),
        best_epoch=best_epoch, best_val_macro_f1=best_f1, parameters=sum(p.numel() for p in model.parameters()),
        converged=status['converged'], reason=status['reason'], uses_test=False))
    return root


def evaluate(config, device, checkpoint, split='val'):
    import torch
    from .metrics import write_metrics
    from .training import evaluate_model, setup_execution
    from .v2_protocol import record_selection
    supported(config)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_checkpoint(saved)
    if saved['epoch'] != saved['best_epoch']:
        raise ConfigError('evaluation requires a validation-best checkpoint; use best.pt')
    runtime = deepcopy(saved['config'])
    runtime['paths'] = deepcopy(config['paths'])
    for field in ('task', 'transform', 'model'):
        if config[field] != saved['config'][field]:
            raise ConfigError(f'incompatible checkpoint {field}')
    if any(config['feature'][name] != saved['config']['feature'][name]
           for name in ('version', 'iat_enabled')):
        raise ConfigError('incompatible checkpoint feature')
    runtime['feature'] = deepcopy(config['feature'])
    torch.set_num_threads(runtime['training']['numerical_threads'])
    setup_execution(runtime['seed'])
    samples, batches = loader(runtime, split)
    if split == 'val' and samples.metadata != saved['identity']['base_cache']['val']:
        raise ConfigError('checkpoint validation cache differs from requested cache')
    if runtime['feature']['iat_enabled']:
        # Verify the saved fit and both training identities for either eval split.
        # A synthetic test cache is independently bound to that same verified fit.
        train_samples, _ = loader(runtime, 'train')
        val_samples = samples if split == 'val' else loader(runtime, 'val')[0]
        if experiment_identity(runtime, train_samples, val_samples) != saved['identity']:
            raise ConfigError('checkpoint IAT fit/cache identity differs from requested cache')
    model = build_model(saved['model_definition'], saved['feature_definition']['in_channels']).to(device)
    model.load_state_dict(saved['model'])
    summary, per_class, matrix = evaluate_model(model, batches, device, 'scenario')
    selection_path, selection = record_selection(config['paths']['output_root'], saved, checkpoint)
    summary.update(split=split, epoch=saved['epoch'], task='scenario', transform=runtime['transform'],
                   experiment_id=saved['experiment_id'], manifest_id=samples.metadata['manifest_id'],
                   validation_macro_f1=saved['best_macro_f1'], selection_artifact=str(selection_path),
                   validation_selected_transform=selection['selected_transform'],
                   selection_complete=selection['complete'])
    write_metrics(Path(checkpoint).parent / f'evaluation-{split}', summary, per_class, matrix)
    return summary
