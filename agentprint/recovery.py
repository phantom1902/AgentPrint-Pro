"""Epoch commits, immutable best snapshots, and recovery-safe observable history.

The atomic replacement of last.pt commits an epoch. Logs can be ahead of that
commit after interruption; they are archived and TensorBoard purged on resume.
Best snapshots are immutable and published before last references them. The
convenience best.pt and CSV/JSON views are repairable from that committed pointer.
"""
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import uuid

import numpy as np
import torch
import yaml

from .config import ConfigError
from .metrics import write_metrics


def atomic_write(path, writer):
    """Replace a single durable file without exposing a partial new value."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path, value):
    atomic_write(path, lambda handle: torch.save(value, handle))


def save_text(path, text):
    atomic_write(path, lambda handle: handle.write(text.encode('utf-8')))


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def capture_rng(generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'shuffle_generator': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(saved, generator):
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['torch'])
    generator.set_state(saved['shuffle_generator'])
    # Device changes are permitted and recorded, without a bitwise guarantee.
    if torch.cuda.is_available() and len(saved['cuda']) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(saved['cuda'])


def validate_resume(saved, config, identity):
    if saved.get('checkpoint_format') != 1:
        raise ConfigError('incompatible recovery checkpoint format')
    for field, value in identity.items():
        if saved.get(field) != value:
            raise ConfigError(f'incompatible recovery {field}')
    for field, value in config.items():
        if field != 'paths' and saved['config'].get(field) != value:
            raise ConfigError(f'incompatible recovery configuration {field}')
    if not 0 <= saved['epoch'] < config['training']['epochs']:
        raise ConfigError('incompatible recovery epoch / target total epochs')


def archive_ahead_logs(root, epoch, global_step):
    """Preserve uncommitted/partial rows before replacing the effective history."""
    archive = None
    for name, field, maximum in [('metrics.jsonl', 'epoch', epoch),
                                 ('training_steps.jsonl', 'global_step', global_step)]:
        path = root / name
        if not path.exists():
            continue
        kept, ahead = [], []
        seen = set()
        for line in path.read_text(encoding='utf-8').splitlines(keepends=True):
            try:
                row = json.loads(line)
                step = row[field]
                retain = isinstance(step, int) and 0 <= step <= maximum and step not in seen
            except (ValueError, TypeError, KeyError):
                retain = False
            if retain:
                kept.append(line if line.endswith('\n') else line + '\n')
                seen.add(step)
            else:
                ahead.append(line)
        if ahead:
            if archive is None:
                stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                archive = root / 'archive' / f'{stamp}-{uuid.uuid4().hex[:8]}'
                archive.mkdir(parents=True, exist_ok=False)
            save_text(archive / name, ''.join(ahead))
            save_text(path, ''.join(kept))


def publish_best(root, relative):
    snapshot = root / relative
    if not snapshot.is_file():
        raise ConfigError(f'committed best checkpoint is missing: {relative}')
    atomic_write(root / 'best.pt', lambda handle: _copy(snapshot, handle))
    saved = torch.load(snapshot, map_location='cpu', weights_only=False)
    result = saved['validation_result']
    write_metrics(root, result['summary'], result['per_class'], np.asarray(result['matrix']))


def _copy(source, destination):
    with source.open('rb') as handle:
        shutil.copyfileobj(handle, destination)


def environment():
    packages: dict[str, str | None] = {}
    for name in ('torch', 'numpy', 'PyYAML', 'scikit-learn', 'tensorboard', 'dpkt'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'],
                                    cwd=Path(__file__).resolve().parents[1], capture_output=True,
                                    text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'git_commit': commit, 'git_dirty': dirty}


def record_runtime(root, config, device, resume_epoch):
    append_json(root / 'runtime_history.jsonl', {
        'utc': datetime.now(timezone.utc).isoformat(), 'run_id': root.name,
        'device': str(device), 'config': config, 'resume_epoch': resume_epoch,
        'environment': environment(), 'execution': {
            'precision': 'float32', 'amp': False, 'tf32': False,
            'cudnn_benchmark': False, 'deterministic_algorithms': True,
            'num_workers': 0, 'cross_device_bitwise_guarantee': False}})
    save_text(root / 'resolved_config.yaml', yaml.safe_dump(config, sort_keys=False))
