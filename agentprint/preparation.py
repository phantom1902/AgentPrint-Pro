"""Discover official episodes and compose parser, interval MTAM, and cache."""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import FORMAT_VERSION, PREPROCESSING_VERSION
from .cache import CACHE_RECOVERY_VERSION, LABEL_MAP, LABEL_MAP_ID, write_json, write_split
from .config import ConfigError, fingerprint
from .parsing import EpisodeExcluded, decode_packet, direction, iter_records, scan_episode
from .primitive import primitive_samples


def source_inventory(data_root, splits):
    root = Path(data_root)
    if not root.is_dir():
        raise ConfigError(f'data root is not a directory: {root}')
    result = []
    for path in sorted(root.rglob('*.pcap')):
        relative = path.relative_to(root)
        parts = relative.parts
        if len(parts) != 5 or parts[0] not in ('train', 'val', 'test') or parts[-1] != 'network.pcap':
            raise ConfigError(f'unsupported ANT capture layout: {relative.as_posix()}')
        split, engine, scenario, episode_id, _ = parts
        if scenario not in LABEL_MAP['scenario']:
            raise ConfigError(f'unknown scenario: {scenario}')
        if split not in splits:
            continue
        stat = path.stat()
        annotation = path.with_name('messages.jsonl')
        # Only small identity/endpoint metadata and annotations are content
        # hashed. PCAP compatibility intentionally uses path, size and mtime.
        endpoints = [candidate for candidate in sorted(path.parent.iterdir())
                     if candidate.is_file() and candidate.suffix.lower() in
                     ('.json', '.jsonl', '.yaml', '.yml') and
                     ('endpoint' in candidate.stem.lower() or
                      candidate.name in ('runtime.json', 'identity.json'))]
        result.append({'pcap_relpath': relative.as_posix(), 'size_bytes': stat.st_size,
                       'mtime_ns': stat.st_mtime_ns, 'split': split, 'engine': engine,
                       'scenario': scenario, 'episode_id': episode_id,
                       'annotation_relpath': annotation.relative_to(root).as_posix()
                       if annotation.is_file() else None,
                       'annotation_sha256': hashlib.sha256(annotation.read_bytes()).hexdigest()
                       if annotation.is_file() else None,
                       'endpoint_metadata': [
                           {'relpath': candidate.relative_to(root).as_posix(),
                            'sha256': hashlib.sha256(candidate.read_bytes()).hexdigest()}
                           for candidate in endpoints]})
    if not result:
        raise ConfigError('no network.pcap episodes found in configured official splits')
    return result


def build_mtam(path, start_ns, end_ns, endpoints):
    """Full interval; internal boundaries enter the new bin, endpoint the last."""
    values = np.zeros((2, 2, 1800), dtype=np.uint64)
    skipped = 0
    for time_ns, raw in iter_records(path):
        packet = decode_packet(time_ns, raw)
        if packet is None:
            continue
        if not start_ns <= time_ns <= end_ns:
            continue
        orientation = direction(packet, endpoints)
        if orientation is None:
            skipped += 1
            continue
        slot = min((time_ns - start_ns) * 1800 // (end_ns - start_ns), 1799)
        values[0, orientation, slot] += np.uint64(1)
        values[1, orientation, slot] += np.uint64(packet.ip_length)
    return values, skipped


def scenario_samples(data_root, sources):
    for source in sources:
        record = {key: source[key] for key in ('split', 'engine', 'scenario', 'episode_id', 'pcap_relpath')}
        record.update(sample_id='scenario:' + source['pcap_relpath'].removesuffix('/network.pcap'),
                      task='scenario', label_name=source['scenario'],
                      label_id=LABEL_MAP['scenario'][source['scenario']],
                      annotation_relpath=None, annotation_row=None)
        path = Path(data_root) / source['pcap_relpath']
        try:
            scan = scan_episode(path)
            record.update(start_ns=scan.start_ns, end_ns=scan.end_ns,
                          duration_ns=scan.end_ns - scan.start_ns,
                          runtime_endpoints=scan.runtime_endpoints,
                          endpoint_source=scan.endpoint_source, quality=scan.quality)
            values, skipped = build_mtam(path, scan.start_ns, scan.end_ns, scan.runtime_endpoints)
            record['quality']['direction_skipped'] = skipped
            record['packet_count'] = [int(n) for n in values[0].sum(axis=1)]
            record['ip_bytes'] = [int(n) for n in values[1].sum(axis=1)]
            if not sum(record['packet_count']):
                raise EpisodeExcluded('no_direction_packets', 'no usable runtime packets')
            yield record, values
        except EpisodeExcluded as exc:
            record.update(exc.context)
            record.update(reason=exc.reason, detail=exc.detail)
            yield record, None


def resumed_samples(data_root, sources, generator, task, cursor):
    """Skip committed PCAPs, replay only an in-progress episode when necessary.

    source_index is split-local. sample_offset counts all generator rows,
    including exclusions, so an episode spanning shards resumes without gaps.
    One-row lookahead marks the last row as source-complete before it can be
    committed, without reading the next PCAP or adding an Episode cache.
    """
    start = cursor['source_index']
    for index in range(start, len(sources)):
        skip = cursor['sample_offset'] if index == start else 0
        rows = iter(generator(data_root, [sources[index]]))
        current = next(rows, None)
        ordinal = 0
        while current is not None:
            following = next(rows, None)
            if ordinal >= skip:
                next_cursor = ({'source_index': index + 1, 'sample_offset': 0}
                               if following is None else
                               {'source_index': index, 'sample_offset': ordinal + 1})
                record, array = current
                yield record, array, next_cursor
            current = following
            ordinal += 1


def prepare(config):
    root = Path(config['paths']['cache_root'])
    sources = source_inventory(config['paths']['data_root'], config['preprocessing']['splits'])
    # Input transforms/training paths do not change the integer-statistics cache.
    prep_config = {'task': config['task'], 'preprocessing': config['preprocessing']}
    config_id = fingerprint(prep_config)
    root.mkdir(parents=True, exist_ok=True)
    contract = {'config_fingerprint': config_id, 'sources': sources,
                'source_inventory_id': fingerprint(sources), 'complete': False}
    index_path = root / 'index.json'
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding='utf-8'))
        for key, expected in (('format_version', FORMAT_VERSION),
                              ('preprocessing_version', PREPROCESSING_VERSION),
                              ('cache_recovery_version', CACHE_RECOVERY_VERSION),
                              ('label_map_id', LABEL_MAP_ID)):
            if index.get(key) != expected:
                raise ConfigError(f'incompatible cache {key}: {index.get(key)!r} != {expected!r}')
        previous = index['tasks'].get(config['task'])
        if previous:
            if previous['config_fingerprint'] != config_id:
                raise ConfigError('incompatible cache configuration for task ' + config['task'])
            if previous['source_inventory_id'] != contract['source_inventory_id']:
                raise ConfigError('incompatible cache source inventory: PCAP path/size/mtime or '
                                  'annotation/endpoint metadata content changed')
    else:
        if (root / config['task']).exists():
            raise ConfigError('incompatible cache: task artifacts exist without an index')
        index = {'format_version': FORMAT_VERSION, 'preprocessing_version': PREPROCESSING_VERSION,
                 'cache_recovery_version': CACHE_RECOVERY_VERSION,
                 'config_fingerprint': config_id, 'label_map_id': LABEL_MAP_ID,
                 'sources': sources, 'source_inventory_id': fingerprint(sources),
                 'tasks': {}, 'datasets': {}, 'complete': False}
    write_json(root / 'label_map.json', LABEL_MAP)
    index['tasks'][config['task']] = contract
    index['complete'] = False
    write_json(root / 'index.json', index)
    for split in config['preprocessing']['splits']:
        selected = [source for source in sources if source['split'] == split]
        generator = primitive_samples if config['task'] == 'primitive' else scenario_samples
        def samples(cursor):
            return resumed_samples(config['paths']['data_root'], selected,
                                   generator, config['task'], cursor)
        metadata = write_split(root, config['task'], split, samples, config_id)
        index['datasets'][f"{config['task']}/{split}"] = metadata
        write_json(root / 'index.json', index)
    index['tasks'][config['task']]['complete'] = True
    index['complete'] = all(task['complete'] for task in index['tasks'].values())
    write_json(root / 'index.json', index)
    return index
