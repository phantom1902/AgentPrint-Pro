"""Shared integer-time Episode IAT semantics and explicit train-only bin fitting.

No dataset discovery occurs here. Valid base Scenario rows select both the
capture bounds/endpoints and the source identities to verify and read.
"""
from array import array
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import FORMAT_VERSION, PREPROCESSING_VERSION
from .cache import LABEL_MAP_ID, read_jsonl, write_json
from .config import ConfigError, fingerprint
from .parsing import decode_packet, direction, iter_records


FIT_RULES = dict(version=1, timestamp_unit='integer_nanoseconds',
                 sort='stable_in_memory', predecessor='all_effective_packets',
                 attribution='current_packet_time_and_direction', first_packet='no_interval',
                 seed=42, seed_derivation='sha256(seed:sample_id), first 128 bits',
                 sampler='numpy-PCG64-choice-without-replacement-v1',
                 max_samples_per_episode=4096, weighting='equal_episode',
                 quantile='exact_weighted_empirical_inverse_cdf',
                 quantile_numerators=[1, 2, 3, 4, 5, 6], quantile_denominator=7,
                 zero_bucket=0, positive_bucket_count=7,
                 boundary_side='right', duplicate_boundaries='retain', last_bucket='unbounded')


@dataclass
class EpisodeIAT:
    timestamps_ns: np.ndarray
    directions: np.ndarray
    intervals_ns: np.ndarray
    quality: dict[str, int]

    @property
    def current_times_ns(self):
        return self.timestamps_ns[1:]

    @property
    def current_directions(self):
        return self.directions[1:]


def effective_episode(data_root: str | Path, record: dict[str, Any]) -> EpisodeIAT:
    """Read all effective packets using the original base record's semantics.

    Stable sorting retains capture order at ties. Compact numeric buffers never
    retain raw packet contents. Allocation/sort failure aborts the Episode.
    """
    try:
        times, directions = array('q'), array('B')
        inversions = skipped = 0
        previous = None
        for time_ns, raw in iter_records(Path(data_root) / record['pcap_relpath']):
            packet = decode_packet(time_ns, raw)
            if packet is None or not record['start_ns'] <= time_ns <= record['end_ns']:
                continue
            orientation = direction(packet, record['runtime_endpoints'])
            if orientation is None:
                skipped += 1
                continue
            inversions += int(previous is not None and time_ns < previous)
            previous = time_ns
            times.append(time_ns)
            directions.append(orientation)
        timestamps = np.frombuffer(times, dtype=np.int64)
        orientations = np.frombuffer(directions, dtype=np.uint8)
        order = np.argsort(timestamps, kind='stable')
        timestamps, orientations = timestamps[order], orientations[order]
        intervals = np.diff(timestamps).astype(np.uint64)
        return EpisodeIAT(timestamps, orientations, intervals,
                          dict(effective_packets=len(timestamps),
                               timestamp_inversions=inversions, direction_skipped=skipped,
                               zero_intervals=int(np.count_nonzero(intervals == 0)),
                               positive_intervals=int(np.count_nonzero(intervals > 0))))
    except (MemoryError, OverflowError) as exc:
        raise RuntimeError('IAT integer-time sorting resource failure for ' + record['sample_id']) from exc


def _verify_source(data_root: str | Path, source: dict[str, Any]) -> None:
    root = Path(data_root)
    path = root / source['pcap_relpath']
    stat = path.stat()
    if stat.st_size != source['size_bytes'] or stat.st_mtime_ns != source['mtime_ns']:
        raise ConfigError('incompatible IAT source PCAP path/size/mtime: ' + source['pcap_relpath'])
    endpoints = [candidate for candidate in sorted(path.parent.iterdir())
                 if candidate.is_file() and candidate.suffix.lower() in
                 ('.json', '.jsonl', '.yaml', '.yml') and
                 ('endpoint' in candidate.stem.lower() or candidate.name in ('runtime.json', 'identity.json'))]
    actual = [{'relpath': item.relative_to(root).as_posix(),
               'sha256': hashlib.sha256(item.read_bytes()).hexdigest()} for item in endpoints]
    if actual != source['endpoint_metadata']:
        raise ConfigError('incompatible IAT source endpoint metadata: ' + source['pcap_relpath'])
    annotation = path.with_name('messages.jsonl')
    annotation_hash = hashlib.sha256(annotation.read_bytes()).hexdigest() if annotation.is_file() else None
    if annotation_hash != source.get('annotation_sha256'):
        raise ConfigError('incompatible IAT source annotation identity: ' + source['pcap_relpath'])


def selected_base_records(config: dict[str, Any], split: str = 'train') -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return (valid records, matching source entries, base metadata), by sample ID.

    Selected views publish their own manifest metadata while retaining original
    source inventory and references to original arrays.
    """
    root = Path(config['paths']['cache_root'])
    index = json.loads((root / 'index.json').read_text(encoding='utf-8'))
    metadata = json.loads((root / 'scenario' / split / 'metadata.json').read_text(encoding='utf-8'))
    contract = index.get('tasks', {}).get('scenario', {})
    for name, expected in (('format_version', FORMAT_VERSION),
                           ('preprocessing_version', PREPROCESSING_VERSION),
                           ('label_map_id', LABEL_MAP_ID)):
        if index.get(name) != expected or metadata.get(name) != expected:
            raise ConfigError('incompatible IAT base cache ' + name)
    if not contract.get('complete') or not metadata.get('complete'):
        raise ConfigError('IAT requires a complete base Scenario cache')
    if metadata.get('task') != 'scenario' or metadata.get('split') != split:
        raise ConfigError('incompatible IAT base cache task/split')
    if metadata.get('config_fingerprint') != contract.get('config_fingerprint'):
        raise ConfigError('incompatible IAT base cache preprocessing identity')
    records = read_jsonl(root / 'scenario' / split / 'valid.jsonl')
    manifest_id = fingerprint(records)
    if manifest_id != metadata['manifest_id']:
        raise ConfigError('incompatible IAT base cache manifest identity')
    if metadata.get('valid_count') != len(records):
        raise ConfigError('incompatible IAT base cache valid count')
    sources = contract.get('sources', index.get('sources', []))
    by_path = {source['pcap_relpath']: source for source in sources}
    records = sorted(records, key=lambda row: row['sample_id'])
    if len({row['sample_id'] for row in records}) != len(records):
        raise ConfigError('duplicate IAT base sample_id')
    selected = []
    for row in records:
        if row.get('split') != split or row.get('task') != 'scenario':
            raise ConfigError('IAT requires valid Scenario records from ' + split)
        source = by_path.get(row['pcap_relpath'])
        if source is None or source['split'] != split:
            raise ConfigError('IAT base cache source identity is missing')
        _verify_source(config['paths']['data_root'], source)
        if not (root / row['shard_relpath']).is_file():
            raise ConfigError('IAT base cache shard is missing: ' + row['shard_relpath'])
        values = np.load(root / row['shard_relpath'], mmap_mode='r', allow_pickle=False)
        if (values.dtype != np.uint64 or values.ndim != 4 or values.shape[1:] != (2, 2, 1800)
                or not 0 <= row['row_index'] < len(values)):
            raise ConfigError('incompatible IAT base cache shard array')
        del values
        selected.append(source)
    return records, selected, metadata


def _weighted_boundaries(episodes: list[dict[str, Any]]) -> list[int]:
    usable = [row['sampled_positive_ns'] for row in episodes if row['sampled_positive_ns']]
    if not usable:
        raise ConfigError('IAT fit pool has no positive intervals; no boundaries can be fitted')
    # A common integer denominator makes >= targets exact even at rational ties.
    denominator = math.lcm(*(len(values) for values in usable))
    streams = [zip(values, [denominator // len(values)] * len(values)) for values in usable]
    cumulative = 0
    boundaries: list[int] = []
    total = len(usable) * denominator
    for value, weight in heapq.merge(*streams):
        cumulative += weight
        while len(boundaries) < 6 and cumulative * 7 >= (len(boundaries) + 1) * total:
            boundaries.append(int(value))
        if len(boundaries) == 6:
            break
    return boundaries


def load_fit_artifact(path: str | Path) -> dict[str, Any]:
    """Validate a completed, self-fingerprinted shared fit artifact."""
    artifact = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(artifact, dict):
        raise ConfigError('incompatible IAT fit artifact document')
    payload = {key: value for key, value in artifact.items() if key != 'fingerprint'}
    boundaries = artifact.get('boundaries_ns', [])
    if (artifact.get('kind') != 'agentprint-iat-fit' or artifact.get('format_version') != 1
            or artifact.get('complete') is not True or artifact.get('rules') != FIT_RULES
            or artifact.get('scope') not in ('smoke', 'full')
            or not isinstance(boundaries, list) or len(boundaries) != 6
            or any(type(v) is not int or v <= 0 for v in boundaries)
            or sorted(boundaries) != boundaries or artifact.get('fingerprint') != fingerprint(payload)):
        raise ConfigError('incompatible IAT fit artifact identity/rules/boundaries')
    return artifact


def fit_bins(config: dict[str, Any]) -> dict[str, Any]:
    """Fit once or resume at committed Episodes, with no val/test PCAP reads."""
    if config['schema_version'] != 2 or config['task'] != 'scenario' or not config['feature']['iat_enabled']:
        raise ConfigError('fit-bins requires schema_version=2 and enabled IAT')
    records, sources, metadata = selected_base_records(config)
    artifact_path = Path(config['feature']['fit_artifact'])
    identity = dict(kind='agentprint-iat-fit', format_version=1, rules=FIT_RULES,
                    scope='smoke' if config['mode'] == 'smoke' else 'full',
                    train_manifest=records, sources=sources,
                    base_cache_identity=metadata)
    identity_id = fingerprint(identity)
    progress = artifact_path.with_name(artifact_path.name + '.progress')
    identity_path = progress / 'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ConfigError('incompatible IAT fit progress source/rules identity')
    else:
        write_json(identity_path, identity)
    episodes: list[dict[str, Any]] = []
    for record, source in zip(records, sources):
        name = hashlib.sha256(record['sample_id'].encode()).hexdigest() + '.json'
        path = progress / 'episodes' / name
        episode_identity = fingerprint(dict(fit_identity=identity_id, record=record, source=source))
        if path.exists():
            episode = json.loads(path.read_text(encoding='utf-8'))
            samples = episode.get('sampled_positive_ns', [])
            if (episode.get('identity') != episode_identity or len(samples) > 4096
                    or any(type(v) is not int or v <= 0 for v in samples)
                    or sorted(samples) != samples or episode.get('fingerprint') !=
                    fingerprint({key: value for key, value in episode.items() if key != 'fingerprint'})):
                raise ConfigError('incompatible IAT fit Episode commit: ' + record['sample_id'])
        else:
            values = effective_episode(config['paths']['data_root'], record)
            positive = values.intervals_ns[values.intervals_ns > 0]
            if len(positive) > 4096:
                seed = int.from_bytes(hashlib.sha256(('42:' + record['sample_id']).encode()).digest()[:16], 'big')
                positive = np.random.Generator(np.random.PCG64(seed)).choice(positive, 4096, replace=False)
            episode = dict(sample_id=record['sample_id'], identity=episode_identity,
                           quality=values.quality, sampled_positive_count=len(positive),
                           sampled_positive_ns=sorted(int(v) for v in positive),
                           fit_status='participating' if len(positive) else 'no_positive_intervals')
            episode['fingerprint'] = fingerprint(episode)
            write_json(path, episode)
        episodes.append(episode)
    boundaries = _weighted_boundaries(episodes)
    sample_bucket_counts = np.zeros(7, dtype=np.uint64)
    for episode in episodes:
        indices = np.searchsorted(boundaries, episode['sampled_positive_ns'], side='right')
        sample_bucket_counts += np.bincount(indices, minlength=7).astype(np.uint64)
    statistics = {name: sum(row['quality'][name] for row in episodes) for name in
                  ('effective_packets', 'timestamp_inversions', 'direction_skipped', 'zero_intervals', 'positive_intervals')}
    statistics.update(participating_episodes=sum(bool(row['sampled_positive_ns']) for row in episodes),
                      no_positive_episodes=sum(not row['sampled_positive_ns'] for row in episodes),
                      sampled_positive_intervals=sum(len(row['sampled_positive_ns']) for row in episodes),
                      fit_sample_positive_bucket_counts=[int(n) for n in sample_bucket_counts],
                      empty_positive_buckets=[i + 1 for i, count in enumerate(sample_bucket_counts) if count == 0])
    # The final artifact reports identities/counts, while samples remain only in
    # the bounded Episode commits. It never stores packet sequences or payloads.
    result = dict(identity, complete=True, boundaries_ns=boundaries, statistics=statistics,
                  episodes=[{key: value for key, value in row.items() if key != 'sampled_positive_ns'} for row in episodes])
    result['fingerprint'] = fingerprint(result)
    if artifact_path.exists() and load_fit_artifact(artifact_path) != result:
        raise ConfigError('incompatible existing IAT fit artifact')
    write_json(artifact_path, result)
    return result
