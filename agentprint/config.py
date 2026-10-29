"""Strict common-YAML plus experiment overlay configuration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import yaml


class ConfigError(ValueError):
    pass


SCHEMA = {
    'schema_version': int, 'task': str, 'transform': str, 'mode': str, 'seed': int,
    'paths': {'data_root': str, 'cache_root': str, 'output_root': str},
    'preprocessing': {'bins': int, 'shard_size': int, 'splits': list},
    'training': {'epochs': int, 'batch_size': int, 'learning_rate': (int, float),
                 'weight_decay': (int, float), 'optimizer': str,
                 'lr_decay': (int, float), 'num_workers': int},
}


def merge(base, overlay):
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read(path, stack):
    path = Path(path).resolve()
    if path in stack:
        raise ConfigError('cyclic configuration extends')
    with path.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError('configuration must be a YAML mapping')
    parent = config.pop('extends', None)
    if parent is not None:
        if not isinstance(parent, str):
            raise ConfigError('extends must be a path string')
        config = merge(_read(path.parent / parent, stack + [path]), config)
    return config


def _validate_fields(config, schema, prefix=''):
    if not isinstance(config, dict):
        raise ConfigError(f'{prefix} must be a mapping')
    for key in config.keys() - schema.keys():
        raise ConfigError(f'unknown configuration field: {prefix}{key}')
    for key, expected in schema.items():
        field = prefix + key
        if key not in config:
            raise ConfigError(f'missing configuration field: {field}')
        value = config[key]
        if isinstance(expected, dict):
            _validate_fields(value, expected, field + '.')
        elif isinstance(value, bool) or not isinstance(value, expected):
            raise ConfigError(f'illegal type for {field}')


def load_config(path, overrides=None):
    config = _read(path, [])
    _validate_fields(config, SCHEMA)
    if config['schema_version'] != 1:
        raise ConfigError('schema_version must be 1')
    for field, allowed in [('task', ('scenario', 'primitive')),
                           ('transform', ('raw', 'log1p')),
                           ('mode', ('formal', 'smoke'))]:
        if config[field] not in allowed:
            raise ConfigError(f'illegal {field}: {config[field]}')
    if not 0 <= config['seed'] < 2**32:
        raise ConfigError('seed must be an integer in [0, 2**32)')
    prep, train = config['preprocessing'], config['training']
    if prep['bins'] != 1800 or prep['shard_size'] != 1024:
        raise ConfigError('preprocessing requires bins=1800 and shard_size=1024')
    if (not prep['splits'] or len(set(prep['splits'])) != len(prep['splits'])
            or any(s not in ('train', 'val', 'test') for s in prep['splits'])):
        raise ConfigError('preprocessing.splits must contain unique official split names')
    if train['epochs'] < 1 or train['batch_size'] < 1:
        raise ConfigError('training epochs and batch_size must be positive')
    import math
    for field in ('learning_rate', 'lr_decay', 'weight_decay'):
        if not math.isfinite(train[field]) or train[field] < 0:
            raise ConfigError(f'illegal training.{field}')
    if train['learning_rate'] == 0 or not 0 < train['lr_decay'] <= 1:
        raise ConfigError('learning_rate must be positive and lr_decay in (0,1]')
    if train['optimizer'] != 'adam' or train['num_workers'] != 0:
        raise ConfigError('training requires optimizer=adam and num_workers=0')
    for key, value in (overrides or {}).items():
        if key not in ('data_root', 'cache_root', 'output_root'):
            raise ConfigError(f'unsupported runtime override: {key}')
        if value is not None:
            config['paths'][key] = str(value)
    for key, value in config['paths'].items():
        if not value.strip():
            raise ConfigError(f'paths.{key} must not be empty')
    return config


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()
