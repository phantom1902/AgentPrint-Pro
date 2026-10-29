"""Discover official episodes and compose parser, interval MTAM, and cache."""
from pathlib import Path

import numpy as np

from . import FORMAT_VERSION, PREPROCESSING_VERSION
from .cache import LABEL_MAP, LABEL_MAP_ID, write_json, write_split
from .config import ConfigError, fingerprint
from .parsing import EpisodeExcluded, decode_packet, direction, iter_records, scan_episode


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
        result.append({'pcap_relpath': relative.as_posix(), 'size_bytes': stat.st_size,
                       'mtime_ns': stat.st_mtime_ns, 'split': split, 'engine': engine,
                       'scenario': scenario, 'episode_id': episode_id})
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


def prepare(config):
    if config['task'] != 'scenario':
        raise ConfigError('T01 prepare supports scenario only; primitive preparation requires T02')
    root = Path(config['paths']['cache_root'])
    # Recovery is deliberately deferred to T04; never silently overwrite outputs.
    if (root / 'index.json').exists() or (root / config['task']).exists():
        raise ConfigError('cache already exists; T01 does not yet support cache recovery or overwrite')
    sources = source_inventory(config['paths']['data_root'], config['preprocessing']['splits'])
    # Input transforms/training paths do not change the integer-statistics cache.
    prep_config = {'task': config['task'], 'preprocessing': config['preprocessing']}
    config_id = fingerprint(prep_config)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'label_map.json', LABEL_MAP)
    index = {'format_version': FORMAT_VERSION, 'preprocessing_version': PREPROCESSING_VERSION,
             'config_fingerprint': config_id, 'label_map_id': LABEL_MAP_ID,
             'sources': sources, 'source_inventory_id': fingerprint(sources),
             'datasets': {}, 'complete': False}
    write_json(root / 'index.json', index)
    for split in config['preprocessing']['splits']:
        selected = [source for source in sources if source['split'] == split]
        metadata = write_split(root, config['task'], split,
                               scenario_samples(config['paths']['data_root'], selected), config_id)
        index['datasets'][f"{config['task']}/{split}"] = metadata
        write_json(root / 'index.json', index)
    index['complete'] = True
    write_json(root / 'index.json', index)
    return index
