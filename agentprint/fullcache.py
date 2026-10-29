"""Read-only delivery audit: python -m agentprint.fullcache --help.

This utility consumes preparation's public artifacts, small annotation metadata,
and cached arrays. It never reads PCAP payload or evaluates a model. Successful
verification and eligibility for formal training are separate report fields.
"""
import argparse
from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

from . import FORMAT_VERSION, PREPROCESSING_VERSION
from .cache import CACHE_RECOVERY_VERSION, LABEL_MAP, LABEL_MAP_ID, read_jsonl, write_json
from .config import fingerprint
from .parsing import ENDPOINT_RULE_VERSION
from .preparation import source_inventory
from .primitive import annotation_records


SPLITS = ('train', 'val', 'test')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compatibility(value, context):
    for key, expected in (('format_version', FORMAT_VERSION),
                          ('preprocessing_version', PREPROCESSING_VERSION),
                          ('cache_recovery_version', CACHE_RECOVERY_VERSION),
                          ('label_map_id', LABEL_MAP_ID)):
        require(value.get(key) == expected, f'{context}: incompatible {key}')
    require(value.get('complete') is True, f'{context}: incomplete preparation')


def quality_summary(task, valid, excluded):
    """Coverage is valid / (valid + excluded); null means zero denominator.

    Annotation-less episodes contribute one explicitly unknown-label exclusion,
    not an invented number of primitive rows or an invented fixed label.
    """
    classes = {}
    for label in [*LABEL_MAP[task], '__unknown__']:
        good = [r for r in valid if r['label_name'] == label]
        bad = [r for r in excluded if
               (r.get('label_name') if r.get('label_name') in LABEL_MAP[task]
                else '__unknown__') == label]
        total = len(good) + len(bad)
        classes[label] = {'label_id': LABEL_MAP[task].get(label), 'valid': len(good),
                          'support': len(good), 'excluded': len(bad), 'total': total,
                          'coverage': len(good) / total if total else None,
                          'excluded_fraction': len(bad) / total if total else None,
                          'excluded_reasons': dict(Counter(r['reason'] for r in bad))}
        if label == '__unknown__':
            classes[label]['excluded_labels'] = dict(Counter(
                r.get('label_name') or '__unavailable__' for r in bad))
    total = len(valid) + len(excluded)
    return {'valid': len(valid), 'support': len(valid), 'excluded': len(excluded),
            'total': total, 'coverage': len(valid) / total if total else None,
            'excluded_fraction': len(excluded) / total if total else None,
            'excluded_reasons': dict(Counter(r['reason'] for r in excluded)),
            'classes': classes}


def verify_split(root, data_root, task, split, sources, contract, index, scenario_rows):
    name = f'{task}/{split}'
    base = root / name
    metadata = read_json(base / 'metadata.json')
    compatibility(metadata, name)
    require(metadata == index['datasets'].get(name), f'{name}: index/metadata mismatch')
    require(metadata['task'] == task and metadata['split'] == split,
            f'{name}: task/split mismatch')
    require(metadata['config_fingerprint'] == contract['config_fingerprint'],
            f'{name}: config fingerprint mismatch')
    valid, excluded = (read_jsonl(base / f'{kind}.jsonl') for kind in ('valid', 'excluded'))
    require(metadata['manifest_id'] == fingerprint(valid), f'{name}: valid manifest hash mismatch')
    require(metadata['valid_count'] == len(valid) and metadata['excluded_count'] == len(excluded),
            f'{name}: manifest count mismatch')
    paths = sorted((base / 'commits').glob('*.json'))
    require(bool(paths), f'{name}: missing completion commits')
    commits = [read_json(path) for path in paths]
    for number, (path, commit) in enumerate(zip(paths, commits)):
        require(path.name == f'{number:05d}.json', f'{name}: missing/duplicate commit sequence')
        require(commit['config_fingerprint'] == contract['config_fingerprint'],
                f'{name}: commit config mismatch')
        require(commit['complete'] is (number == len(commits) - 1),
                f'{name}: incomplete or premature completion commit')
        for row in commit['valid']:
            require(row['shard_relpath'] == commit['shard_relpath'],
                    f'{name}: commit shard reference mismatch')
    require(commits[-1]['cursor'] == {'source_index': len(sources), 'sample_offset': 0},
            f'{name}: incomplete source cursor')
    for kind, rows in (('valid', valid), ('excluded', excluded)):
        require([r for c in commits for r in c[kind]] == rows,
                f'{name}: {kind} manifest differs from committed records')
    expected: dict[str, dict] = {}
    annotation_starts = {}
    annotation_start_counts = {}
    for source in sources:
        if task == 'scenario':
            rows = [dict(source, sample_id='scenario:' + source['pcap_relpath'].removesuffix('/network.pcap'),
                         task=task, label_name=source['scenario'], label_id=LABEL_MAP[task][source['scenario']],
                         annotation_relpath=None, annotation_row=None)]
        else:
            rows = annotation_records(data_root, source)
            counts = Counter(r['start_ns'] for r in rows if 'start_ns' in r)
            annotation_start_counts[source['pcap_relpath']] = counts
            annotation_starts[source['pcap_relpath']] = sorted(counts)
        expected.update((row['sample_id'], row) for row in rows)
    actual = valid + excluded
    ids = [r['sample_id'] for r in actual]
    require(len(ids) == len(set(ids)) and set(ids) == set(expected),
            f'{name}: manifest duplicate/omitted source or annotation rows')
    references: dict[str, list[dict]] = {}
    for row in actual:
        identity = row['sample_id']
        origin = expected[identity]
        for key in ('task', 'split', 'engine', 'scenario', 'episode_id', 'pcap_relpath',
                    'annotation_relpath', 'annotation_row', 'label_name', 'label_id'):
            require(row.get(key) == origin.get(key), f'{name}: provenance mismatch {identity}: {key}')
        if task == 'primitive' and 'start_ns' in origin:
            require(row.get('start_ns') == origin['start_ns'], f'{name}: annotation time mismatch {identity}')
        if 'reason' in row:
            require(bool(row['reason']) and 'shard_relpath' not in row and 'row_index' not in row,
                    f'{name}: invalid excluded record {identity}')
            continue
        if task == 'primitive':
            require('reason' not in origin and type(origin.get('start_ns')) is int,
                    f'{name}: invalid annotation origin {identity}: {origin.get("reason", "missing_exact_start")}')
        require(row['label_name'] in LABEL_MAP[task] and
                row['label_id'] == LABEL_MAP[task][row['label_name']], f'{name}: invalid label {identity}')
        start, end = row['start_ns'], row['end_ns']
        require(type(start) is int and type(end) is int and end > start and row['duration_ns'] == end - start,
                f'{name}: invalid interval {identity}')
        evidence = row['endpoint_source']
        require(evidence['rule_version'] == ENDPOINT_RULE_VERSION and evidence['kind'] == 'inferred'
                and evidence['capture_complete'] is True and evidence['selected_ipv4']
                and evidence['syn_candidates'] == [evidence['selected_ipv4']]
                and evidence['participation_leaders'] == [evidence['selected_ipv4']]
                and row['runtime_endpoints'] == [evidence['selected_ipv4'], *evidence['associated_ipv6']],
                f'{name}: invalid endpoint provenance {identity}')
        if task == 'primitive':
            require(annotation_start_counts[row['pcap_relpath']][start] == 1,
                    f'{name}: duplicate_annotation_start {identity}')
            episode = scenario_rows[row['pcap_relpath']]
            require('reason' not in episode and episode['start_ns'] <= start < end <= episode['end_ns']
                    and evidence == episode['endpoint_source'], f'{name}: capture provenance mismatch {identity}')
            boundaries = annotation_starts[row['pcap_relpath']]
            next_index = bisect_right(boundaries, start)
            expected_end = boundaries[next_index] if next_index < len(boundaries) else episode['end_ns']
            require(end == expected_end, f'{name}: interval end mismatch {identity}')
        relative = row['shard_relpath']
        require(Path(relative).as_posix().startswith(name + '/shards/') and
                (root / relative).resolve().parent == (base / 'shards').resolve(),
                f'{name}: invalid shard path {relative}')
        references.setdefault(relative, []).append(row)
    committed = [c['shard_relpath'] for c in commits if c['shard_relpath'] is not None]
    found = {p.relative_to(root).as_posix() for p in (base / 'shards').iterdir()}
    require(len(committed) == len(set(committed)) == metadata['shards'] and
            set(committed) == set(references) == found, f'{name}: duplicate/missing/orphan shard')
    payload_bytes = file_bytes = 0
    for relative, rows in references.items():
        values = np.load(root / relative, mmap_mode='r', allow_pickle=False)
        require(values.dtype == np.uint64 and values.shape == (len(rows), 2, 2, 1800)
                and 1 <= len(rows) <= 1024, f'{name}: invalid shard dtype/shape {relative}')
        require((root / relative).stat().st_size == values.offset + values.nbytes,
                f'{name}: invalid shard file byte count {relative}')
        require(sorted(r['row_index'] for r in rows) == list(range(len(rows))),
                f'{name}: duplicate/omitted shard rows {relative}')
        for row in rows:
            array = values[row['row_index']]
            for channel, field in ((0, 'packet_count'), (1, 'ip_bytes')):
                require([int(n) for n in array[channel].sum(axis=1)] == row[field],
                        f'{name}: array {field} mismatch {row["sample_id"]}')
            require(sum(row['packet_count']) > 0, f'{name}: no direction packets in valid row')
        payload_bytes += values.nbytes
        file_bytes += (root / relative).stat().st_size
        del values
    summary = quality_summary(task, valid, excluded)
    require(metadata['support'] == [summary['classes'][label]['support'] for label in LABEL_MAP[task]]
            and metadata['exclusion_reasons'] == summary['excluded_reasons'], f'{name}: quality metadata mismatch')
    summary.update(manifest_id=metadata['manifest_id'], config_fingerprint=metadata['config_fingerprint'],
                   shard_count=len(references), array_payload_bytes=payload_bytes, array_file_bytes=file_bytes)
    if task == 'scenario':
        scenario_rows.update((r['pcap_relpath'], r) for r in actual)
    return summary


def verify(cache_root, data_root, runtime_record=None):
    """Return a verified quality report, or raise for an incomplete/inconsistent delivery."""
    root = Path(cache_root)
    index = read_json(root / 'index.json')
    compatibility(index, 'index')
    require(read_json(root / 'label_map.json') == LABEL_MAP, 'incompatible label map')
    require(set(index['tasks']) == set(LABEL_MAP), 'incomplete: both tasks are required')
    require(set(index['datasets']) == {f'{t}/{s}' for t in LABEL_MAP for s in SPLITS},
            'incomplete: both tasks require all official splits')
    sources = source_inventory(data_root, SPLITS)
    source_id = fingerprint(sources)
    require(index['source_inventory_id'] == source_id and index['sources'] == sources,
            'source inventory changed: PCAP path/size/mtime or annotation/endpoint identity')
    report = {'verified': True, 'verification_scope': 'metadata_annotations_and_cached_arrays_no_pcap_payload',
              'cache_root': str(root.resolve()), 'data_root': str(Path(data_root).resolve()),
              'format_version': FORMAT_VERSION, 'preprocessing_version': PREPROCESSING_VERSION,
              'label_map_id': LABEL_MAP_ID, 'source_inventory': {
                  'source_inventory_id': source_id, 'capture_count': len(sources),
                  'capture_bytes': sum(s['size_bytes'] for s in sources),
                  'annotation_count': sum(s['annotation_relpath'] is not None for s in sources),
                  'identity_policy': 'PCAP path/size/mtime_ns; annotation and endpoint metadata SHA256',
                  'by_split': {s: {'captures': sum(r['split'] == s for r in sources),
                                   'annotations': sum(r['split'] == s and r['annotation_relpath'] is not None
                                                      for r in sources)} for s in SPLITS}},
              'datasets': {}, 'formal_training': {}}
    if runtime_record:
        path = Path(runtime_record)
        raw = path.read_bytes()
        record = json.loads(raw)
        require(record.get('status') in ('prepared_pending_verification', 'verified', 'complete'),
                'runtime record: preparation is incomplete/running')
        require({c['task'] for c in record['commands']} == set(LABEL_MAP) and
                all(c['returncode'] == 0 for c in record['commands']), 'runtime record: failed/incomplete commands')
        require('cache_root' not in record or Path(record['cache_root']).resolve() == root.resolve(),
                'runtime record: cache root mismatch')
        report['preparation_execution'] = {'path': str(path.resolve()),
                                           'sha256': hashlib.sha256(raw).hexdigest(), 'record': record}
    else:
        report['preparation_execution'] = None
    scenario_rows: dict[str, dict] = {}
    for task in LABEL_MAP:
        contract = index['tasks'][task]
        require(contract.get('complete') is True, f'{task}: incomplete preparation')
        expected_config = fingerprint({'task': task, 'preprocessing':
                                       {'bins': 1800, 'shard_size': 1024, 'splits': list(SPLITS)}})
        require(contract['config_fingerprint'] == expected_config, f'{task}: incompatible full configuration')
        require(contract['source_inventory_id'] == source_id and contract['sources'] == sources,
                f'{task}: source inventory mismatch')
        for split in SPLITS:
            report['datasets'][f'{task}/{split}'] = verify_split(
                root, data_root, task, split, [r for r in sources if r['split'] == split],
                contract, index, scenario_rows)
        train = report['datasets'][f'{task}/train']['classes']
        missing = [label for label in LABEL_MAP[task] if not train[label]['support']]
        report['formal_training'][task] = {'eligible': not missing, 'missing_train_classes': missing,
                                           'reason': 'missing_fixed_train_classes' if missing else None}
    require(source_inventory(data_root, SPLITS) == sources and read_json(root / 'index.json') == index,
            'source inventory or cache index changed during verification; rerun after preparation stops')
    report['coverage_definition'] = 'valid/(valid+excluded); null for no observations; counts are samples, with one unknown exclusion per annotation-less/empty-log episode'
    report['storage'] = {'cache_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
                         'array_payload_bytes': sum(d['array_payload_bytes'] for d in report['datasets'].values()),
                         'array_file_bytes': sum(d['array_file_bytes'] for d in report['datasets'].values()),
                         'disk_free_bytes_at_verification': shutil.disk_usage(root).free}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('cache-root', 'data-root', 'output'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--runtime-record', help='Separate preparation execution JSON; never inferred from cache')
    args = parser.parse_args(argv)
    started = time.monotonic()
    try:
        output = Path(args.output).resolve()
        require(not any(output.is_relative_to(Path(p).resolve()) for p in (args.cache_root, args.data_root)),
                'report output must be outside the read-only raw and cache roots')
        report = verify(args.cache_root, args.data_root, args.runtime_record)
    except (ValueError, OSError, KeyError, TypeError, EOFError) as exc:
        report = {'verified': False, 'errors': [str(exc)], 'cache_root': args.cache_root}
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
                              capture_output=True, text=True, check=False)
    report['verification_execution'] = {'completed_utc': datetime.now(timezone.utc).isoformat(),
                                        'elapsed_seconds': time.monotonic() - started,
                                        'code_sha': revision.stdout.strip() or None,
                                        'python': sys.version, 'platform': platform.platform(),
                                        'numpy': np.__version__}
    # An invalid output location must not turn a failed read-only audit into a write in its inputs.
    if any(Path(args.output).resolve().is_relative_to(Path(p).resolve())
           for p in (args.cache_root, args.data_root)):
        print(json.dumps(report), file=sys.stderr)
        return 1
    write_json(args.output, report)
    print(json.dumps({'verified': report['verified'], 'report': str(Path(args.output).resolve())}))
    return 0 if report['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
