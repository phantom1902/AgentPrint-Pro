"""Versioned JSONL manifests and bounded, whole-sample NPY shards."""
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import FORMAT_VERSION, PREPROCESSING_VERSION
from .config import ConfigError, fingerprint


LABEL_MAP = {
    'scenario': {label: index for index, label in enumerate((
        'artifact_processing', 'business_services', 'general_knowledge',
        'scientific_evidence', 'security_testing'))},
    'primitive': {f'N{index + 1:03d}': index for index in range(47)},
}
LABEL_MAP_ID = fingerprint(LABEL_MAP)
CACHE_RECOVERY_VERSION = 1


def write_json(path, value):
    """Publish one complete JSON document, never a truncated recovery record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_split(cache_root, task, split, samples, config_fingerprint):
    """Commit shards with their records and cursor before publishing manifests.

    A callable samples(cursor) yields (record, raw MTAM or None, next_cursor).
    The cursor is an opaque source position supplied by preparation. Historical
    pair iterators remain supported for fresh writes, but cannot resume.
    JSON commits contain metadata only, never another copy of MTAM arrays.
    """
    root = Path(cache_root)
    base = root / task / split
    shards, commits_dir = base / 'shards', base / 'commits'
    shards.mkdir(parents=True, exist_ok=True)
    commits_dir.mkdir(exist_ok=True)
    commits = []
    committed_names = set()
    cursor = {'source_index': 0, 'sample_offset': 0}
    for number, path in enumerate(sorted(commits_dir.glob('*.json'))):
        if path.name != f'{number:05d}.json':
            raise ConfigError(f'incompatible cache commit sequence: {path}')
        commit = json.loads(path.read_text(encoding='utf-8'))
        if commit.get('config_fingerprint') != config_fingerprint:
            raise ConfigError(f'incompatible cache config in {path}')
        relative = commit['shard_relpath']
        if relative is not None:
            shard = root / relative
            if not shard.is_file():
                raise ConfigError(f'incompatible cache: committed shard is missing: {relative}')
            try:
                values = np.load(shard, mmap_mode='r', allow_pickle=False)
                if (values.dtype != np.uint64 or
                        values.shape != (len(commit['valid']), 2, 2, 1800) or
                        not 1 <= len(values) <= 1024):
                    raise ValueError('invalid shape or dtype')
                del values
            except (ValueError, OSError) as exc:
                raise ConfigError(f'incompatible committed shard {relative}: {exc}') from exc
            committed_names.add(Path(relative).name)
        cursor = commit['cursor']
        commits.append(commit)
    # A crash after array rename but before its JSON commit leaves an orphan.
    # Only this split's uncommitted files are removed; committed arrays stay put.
    for path in shards.iterdir():
        if path.name not in committed_names:
            path.unlink()
    records: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    shard_index = len(committed_names)

    def flush(complete=False):
        nonlocal shard_index
        relative = None
        if arrays:
            relative = f'{task}/{split}/shards/{shard_index:05d}.npy'
            temporary = (root / relative).with_suffix('.npy.tmp')
            with temporary.open('wb') as handle:
                np.save(handle, np.stack(arrays), allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, root / relative)
            for index, record in enumerate(records):
                record.update(shard_relpath=relative, row_index=index)
            shard_index += 1
        commit = {'config_fingerprint': config_fingerprint,
                  'shard_relpath': relative, 'valid': list(records),
                  'excluded': list(exclusions), 'cursor': cursor, 'complete': complete}
        write_json(commits_dir / f'{len(commits):05d}.json', commit)
        commits.append(commit)
        records.clear()
        exclusions.clear()
        arrays.clear()

    if not commits or not commits[-1]['complete']:
        if commits and not callable(samples):
            raise ConfigError('incompatible recovery source: a cursor-aware sample factory is required')
        iterator = samples(cursor) if callable(samples) else samples
        for item in iterator:
            if len(item) == 3:
                record, array, cursor = item
            else:
                record, array = item
            if array is None:
                exclusions.append(record)
                continue
            if array.dtype != np.uint64 or array.shape != (2, 2, 1800):
                raise ValueError('raw MTAM must have uint64 shape [2,2,1800]')
            arrays.append(array)
            records.append(record)
            if len(arrays) == 1024:
                flush()
        # This also commits a trailing exclusion-only run or an empty split.
        flush(complete=True)

    support = [0] * len(LABEL_MAP[task])
    reasons: Counter[str] = Counter()
    valid_count = excluded_count = 0
    for name in ('valid', 'excluded'):
        target = base / f'{name}.jsonl'
        temporary_manifest = target.with_suffix('.jsonl.tmp')
        with temporary_manifest.open('w', encoding='utf-8') as handle:
            for commit in commits:
                for record in commit[name]:
                    handle.write(json.dumps(record, sort_keys=True) + '\n')
                    if name == 'valid':
                        support[record['label_id']] += 1
                        valid_count += 1
                    else:
                        reasons[record['reason']] += 1
                        excluded_count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_manifest, target)
    metadata = {'format_version': FORMAT_VERSION,
                'preprocessing_version': PREPROCESSING_VERSION,
                'cache_recovery_version': CACHE_RECOVERY_VERSION,
                'config_fingerprint': config_fingerprint, 'label_map_id': LABEL_MAP_ID,
                'manifest_id': fingerprint(read_jsonl(base / 'valid.jsonl')),
                'task': task, 'split': split,
                'valid_count': valid_count, 'excluded_count': excluded_count,
                'support': support, 'exclusion_reasons': dict(reasons),
                'shards': shard_index, 'complete': True}
    write_json(base / 'metadata.json', metadata)
    return metadata


class CachedSamples:
    """Map-style dataset; torch DataLoader consumes tensors from float32 arrays."""
    def __init__(self, cache_root, task, split, transform='raw'):
        self.root = Path(cache_root)
        self.records = read_jsonl(self.root / task / split / 'valid.jsonl')
        self.metadata = json.loads((self.root / task / split / 'metadata.json').read_text())
        self.transform = transform
        self._shard: np.ndarray | None = None
        self._shard_name = None
        if transform not in ('raw', 'log1p'):
            raise ValueError(f'unknown input transform: {transform}')
        if not self.metadata['complete'] or self.metadata['label_map_id'] != LABEL_MAP_ID:
            raise ValueError('cache is incomplete or label map is incompatible')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        name = record['shard_relpath']
        if name != self._shard_name:
            self._shard = np.load(self.root / name, mmap_mode='r', allow_pickle=False)
            self._shard_name = name
        assert self._shard is not None
        values = self._shard[record['row_index']].astype(np.float32)
        if self.transform == 'log1p':
            np.log1p(values, out=values)
        return values, record['label_id']
