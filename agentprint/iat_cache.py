"""Independent, whole-shard IAT commits and sample-ID MTAM association.

Only aggregated uint64 arrays are persisted. The base cache remains authoritative
for Episode eligibility, source provenance, capture bounds and label identity.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .cache import CachedSamples, read_jsonl, write_json
from .config import ConfigError, fingerprint
from .iat import effective_episode, load_fit_artifact, selected_base_records


SHARD_SIZE = 256
CACHE_RULES = dict(version=1, dtype='uint64', shape=[8, 2, 1800], shard_size=SHARD_SIZE,
                   predecessor='all_effective_packets', first_packet='no_interval',
                   attribution='current_packet_time_and_direction', zero_bucket=0,
                   boundary_side='right', capture_bounds='original_base_record')


def _safe_root(config):
    root = Path(config['feature']['iat_cache_root']).resolve()
    fit = Path(config['feature']['fit_artifact']).resolve()
    protected = [Path(config['paths'][name]).resolve() for name in ('cache_root', 'data_root')]
    protected += [fit, fit.with_name(fit.name + '.progress')]
    if any(root == path or root in path.parents or path in root.parents for path in protected):
        raise ConfigError('IAT cache root must not overlap base cache, data, fit artifact or fit progress')
    return root


def base_records(config: dict[str, Any], split: str):
    try:
        return selected_base_records(config, split)
    except FileNotFoundError as exc:
        raise ConfigError('IAT requires an existing compatible base cache and sources: ' + str(exc)) from exc


def verified_fit(config: dict[str, Any]) -> dict[str, Any]:
    """Verify the completed fit against the selected training cache, never fit implicitly."""
    path = Path(config['feature']['fit_artifact'])
    if not path.is_file():
        raise ConfigError('IAT fit artifact is missing; run fit-bins explicitly')
    fit = load_fit_artifact(path)
    records, sources, metadata = base_records(config, 'train')
    expected = dict(scope='smoke' if config['mode'] == 'smoke' else 'full',
                    train_manifest=records, sources=sources, base_cache_identity=metadata)
    for name, value in expected.items():
        if fit.get(name) != value:
            raise ConfigError('incompatible IAT fit ' + name)
    return fit


def split_identity(fit, records, sources, metadata):
    return dict(kind='agentprint-iat-cache', format_version=1, rules=CACHE_RULES,
                fit_fingerprint=fit['fingerprint'], scope=fit['scope'],
                base_cache_identity=metadata, sources=sources,
                sample_ids=[row['sample_id'] for row in records])


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _counts(config, record, boundaries):
    episode = effective_episode(config['paths']['data_root'], record)
    values = np.zeros((8, 2, 1800), dtype=np.uint64)
    intervals = episode.intervals_ns
    buckets = 1 + np.searchsorted(np.asarray(boundaries, dtype=np.uint64), intervals, side='right')
    buckets[intervals == 0] = 0
    # Python integers preserve exact base slot arithmetic even at int64 limits.
    slots = np.fromiter((min((int(time) - record['start_ns']) * 1800 //
                            (record['end_ns'] - record['start_ns']), 1799)
                         for time in episode.current_times_ns), dtype=np.int64,
                        count=len(intervals))
    np.add.at(values, (buckets, episode.current_directions, slots), np.uint64(1))
    return values, episode.quality


def _read_commits(root, split, identity, records):
    commits = []
    offset = 0
    for number, path in enumerate(sorted((root / 'scenario' / split / 'commits').glob('*.json'))):
        commit = json.loads(path.read_text(encoding='utf-8'))
        rows = commit.get('records', [])
        relative = f'scenario/{split}/shards/{number:05d}.npy'
        expected_ids = [row['sample_id'] for row in records[offset:offset + SHARD_SIZE]]
        if (path.name != f'{number:05d}.json' or commit.get('identity') != fingerprint(identity)
                or commit.get('fingerprint') != fingerprint({k: v for k, v in commit.items() if k != 'fingerprint'})
                or not rows or [row.get('sample_id') for row in rows] != expected_ids
                or commit.get('shard_relpath') != relative
                or any(row.get('shard_relpath') != relative or row.get('row_index') != i
                       for i, row in enumerate(rows))):
            raise ConfigError('incompatible IAT shard commit identity/sequence')
        shard = root / relative
        if not shard.is_file() or _hash(shard) != commit.get('sha256'):
            raise ConfigError('incompatible IAT committed shard checksum')
        values = np.load(shard, mmap_mode='r', allow_pickle=False)
        if values.dtype != np.uint64 or values.shape != (len(rows), 8, 2, 1800):
            raise ConfigError('incompatible IAT committed shard array')
        del values
        commits.append(commit)
        offset += len(rows)
    return commits, offset


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    root = _safe_root(config)
    fit = verified_fit(config)
    datasets = {}
    # Preflight all requested base splits before publishing any IAT output.
    selected = {split: base_records(config, split) for split in config['preprocessing']['splits']}
    for split, (records, sources, metadata) in selected.items():
        identity = split_identity(fit, records, sources, metadata)
        base = root / 'scenario' / split
        identity_path = base / 'identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
                raise ConfigError('incompatible IAT cache source/base/fit identity')
        else:
            write_json(identity_path, identity)
        shards = base / 'shards'
        shards.mkdir(parents=True, exist_ok=True)
        commits, offset = _read_commits(root, split, identity, records)
        committed = {Path(row['shard_relpath']).name for row in commits}
        for path in shards.iterdir():
            if path.name not in committed and path.is_file() and (path.suffix == '.npy' or path.name.endswith('.npy.tmp')):
                path.unlink()
        while offset < len(records):
            group = records[offset:offset + SHARD_SIZE]
            arrays, rows = [], []
            relative = f'scenario/{split}/shards/{len(commits):05d}.npy'
            for i, record in enumerate(group):
                values, quality = _counts(config, record, fit['boundaries_ns'])
                arrays.append(values)
                rows.append(dict(sample_id=record['sample_id'], label_id=record['label_id'],
                                 shard_relpath=relative, row_index=i, quality=quality))
            target = root / relative
            temporary = target.with_suffix('.npy.tmp')
            with temporary.open('wb') as handle:
                np.save(handle, np.stack(arrays), allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            commit = dict(identity=fingerprint(identity), shard_relpath=relative, records=rows,
                          sha256=_hash(target))
            commit['fingerprint'] = fingerprint(commit)
            write_json(base / 'commits' / f'{len(commits):05d}.json', commit)
            commits.append(commit)
            offset += len(group)
        rows = [row for commit in commits for row in commit['records']]
        manifest = base / 'valid.jsonl'
        temporary = manifest.with_suffix('.jsonl.tmp')
        with temporary.open('w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest)
        result = dict(identity, complete=True, manifest_id=fingerprint(rows),
                      valid_count=len(rows), split=split, task='scenario',
                      shards=[dict(shard_relpath=c['shard_relpath'], sha256=c['sha256']) for c in commits])
        result['fingerprint'] = fingerprint(result)
        write_json(base / 'metadata.json', result)
        datasets['scenario/' + split] = result
    return dict(complete=True, datasets=datasets)


class IATSamples(CachedSamples):
    """Join independent shard layouts before float32 conversion and transform."""
    def __init__(self, config, split):
        self.iat_root = _safe_root(config)
        # Verify base metadata/shapes before trusting the historical cache adapter.
        records, sources, metadata = base_records(config, split)
        super().__init__(config['paths']['cache_root'], 'scenario', split, 'raw')
        fit = verified_fit(config)
        base = self.iat_root / 'scenario' / split
        try:
            self.iat_metadata = json.loads((base / 'metadata.json').read_text(encoding='utf-8'))
            rows = read_jsonl(base / 'valid.jsonl')
        except FileNotFoundError as exc:
            raise ConfigError('IAT supplemental cache is missing; run prepare explicitly') from exc
        identity = split_identity(fit, records, sources, metadata)
        payload = {k: v for k, v in self.iat_metadata.items() if k != 'fingerprint'}
        if (any(self.iat_metadata.get(k) != v for k, v in identity.items())
                or self.iat_metadata.get('fingerprint') != fingerprint(payload)
                or self.iat_metadata.get('complete') is not True
                or self.iat_metadata.get('manifest_id') != fingerprint(rows)
                or self.iat_metadata.get('valid_count') != len(rows)):
            raise ConfigError('incompatible IAT cache source/base/fit/sample_id/manifest identity')
        commits, offset = _read_commits(self.iat_root, split, identity, records)
        if (offset != len(records) or rows != [row for commit in commits for row in commit['records']]
                or self.iat_metadata['shards'] != [dict(shard_relpath=c['shard_relpath'], sha256=c['sha256']) for c in commits]):
            raise ConfigError('incompatible IAT cache sample_id or shard commit')
        self.iat_records = {row['sample_id']: row for row in rows}
        if len(self.iat_records) != len(rows) or set(self.iat_records) != {row['sample_id'] for row in self.records}:
            raise ConfigError('IAT cache sample_id is missing or duplicated')
        self.fit_fingerprint = fit['fingerprint']
        self.transform = config['transform']
        self._iat_shard: np.ndarray | None = None
        self._iat_name = None

    def __getitem__(self, index):
        record = self.records[index]
        if self._shard_name != record['shard_relpath']:
            self._shard = np.load(self.root / record['shard_relpath'], mmap_mode='r', allow_pickle=False)
            self._shard_name = record['shard_relpath']
        assert self._shard is not None
        if (self._shard.dtype != np.uint64 or self._shard.ndim != 4
                or self._shard.shape[1:] != (2, 2, 1800)
                or not 0 <= record['row_index'] < len(self._shard)):
            raise ConfigError('incompatible IAT base cache shard array')
        row = self.iat_records[record['sample_id']]
        if self._iat_name != row['shard_relpath']:
            self._iat_shard = np.load(self.iat_root / row['shard_relpath'], mmap_mode='r', allow_pickle=False)
            self._iat_name = row['shard_relpath']
        assert self._iat_shard is not None
        values = np.concatenate((self._shard[record['row_index']], self._iat_shard[row['row_index']])).astype(np.float32)
        if self.transform == 'log1p':
            np.log1p(values, out=values)
        return values, record['label_id']
