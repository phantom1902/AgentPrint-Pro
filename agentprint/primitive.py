"""Given log intervals and their shared Episode-to-MTAM streaming pass.

Annotations supply operational boundaries, not inferred physical action times.
No clock offsets, message bodies, or saved segment captures are used.
"""
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .cache import LABEL_MAP
from .parsing import EpisodeExcluded, decode_packet, direction, iter_records, scan_episode


ISO_TIME = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})')
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def annotation_inventory(data_root, sources):
    """Attach complete small-log identities; .incomplete files are never inputs."""
    for source in sources:
        relative = Path(source['pcap_relpath']).with_name('messages.jsonl')
        path = Path(data_root) / relative
        source['annotation_relpath'] = relative.as_posix() if path.is_file() else None
        source['annotation_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def time_ns(value):
    """ISO timestamp to exact nanoseconds, requiring an explicit UTC offset."""
    match = ISO_TIME.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError('start_time must be an ISO timestamp with timezone and at most 9 fractional digits')
    base, fraction, offset = match.groups()
    instant = datetime.fromisoformat(base + ('+00:00' if offset == 'Z' else offset))
    delta = instant.astimezone(timezone.utc) - EPOCH
    return (delta.days * 86400 + delta.seconds) * 1000000000 + int((fraction or '').ljust(9, '0'))


def annotation_records(data_root, source):
    """Retain only label, exact start and provenance from each source JSONL row."""
    relative = source.get('annotation_relpath')
    if relative is None:
        candidate = Path(source['pcap_relpath']).with_name('messages.jsonl')
        if (Path(data_root) / candidate).is_file():
            relative = candidate.as_posix()
    base = {key: source[key] for key in ('split', 'engine', 'scenario', 'episode_id', 'pcap_relpath')}
    identity = 'primitive:' + source['pcap_relpath'].removesuffix('/network.pcap')
    base.update(task='primitive', annotation_relpath=relative,
                annotation_row=None, label_name=None, label_id=None, sample_id=identity)
    if relative is None:
        return [dict(base, reason='missing_annotations', detail='complete messages.jsonl is absent')]
    records = []
    with (Path(data_root) / relative).open(encoding='utf-8') as handle:
        for row_number, line in enumerate(handle, 1):
            record = dict(base, annotation_row=row_number, sample_id=f'{identity}:row{row_number}')
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError('annotation must be a JSON object')
            except (json.JSONDecodeError, ValueError):
                records.append(dict(record, reason='malformed_annotation', detail='annotation row is not a JSON object'))
                continue
            primitive, macro = value.get('primitive_label_group_id'), value.get('macro_group_id')
            label = primitive if primitive is not None else macro
            if isinstance(label, str):
                record['label_name'] = label
            if primitive is not None and macro is not None and primitive != macro:
                record.update(reason='annotation_label_conflict', detail='current primitive and macro group identifiers disagree')
            elif not isinstance(label, str) or label not in LABEL_MAP['primitive']:
                record.update(reason='invalid_primitive_label', detail='expected current N001 through N047 label')
            else:
                record['label_id'] = LABEL_MAP['primitive'][label]
            try:
                record['start_ns'] = time_ns(value.get('start_time'))
            except ValueError:
                record.update(reason='invalid_annotation_time', detail='start_time cannot be aligned as an exact timestamp')
            records.append(record)
    return records or [dict(base, reason='empty_annotations', detail='complete messages.jsonl contains no rows')]


def primitive_samples(data_root, sources):
    """One scan plus one aggregation pass per Episode, shared by all intervals."""
    for source in sources:
        records = annotation_records(data_root, source)
        if all('start_ns' not in record for record in records):
            for record in records:
                yield record, None
            continue
        path = Path(data_root) / source['pcap_relpath']
        try:
            scan = scan_episode(path)
        except EpisodeExcluded as exc:
            for record in records:
                # Scan context times describe the capture, not this annotation.
                # Preserve the known log start rather than relabel it as the
                # capture start or invent a segment duration after scan failure.
                record.update({key: value for key, value in exc.context.items()
                               if key not in ('start_ns', 'end_ns', 'duration_ns')})
                record.update(reason=exc.reason, detail=exc.detail)
                yield record, None
            continue
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            record.update(runtime_endpoints=scan.runtime_endpoints,
                          endpoint_source=scan.endpoint_source, quality=dict(scan.quality))
            if 'start_ns' in record:
                groups[record['start_ns']].append(record)
        starts = sorted(groups)
        arrays = {}
        # Every known timestamp is a boundary, even when its label is invalid.
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else scan.end_ns
            for record in groups[start]:
                record.update(end_ns=end, duration_ns=end - start)
                if len(groups[start]) > 1:
                    record.update(reason='duplicate_annotation_start', detail='all rows at this start and their interval are excluded')
                elif start < scan.start_ns or start > scan.end_ns or end > scan.end_ns:
                    record.update(reason='annotation_outside_capture', detail='full annotation interval is outside capture bounds; no clock offset inferred')
                elif end <= start:
                    record.update(reason='zero_segment_duration', detail='annotation interval has zero duration')
                if 'reason' not in record:
                    arrays[start] = np.zeros((2, 2, 1800), dtype=np.uint64)
                    record['quality']['direction_skipped'] = 0
        try:
            if arrays:
                for timestamp, raw in iter_records(path):
                    index = bisect_right(starts, timestamp) - 1
                    if index < 0:
                        continue
                    start = starts[index]
                    if start not in arrays:
                        continue
                    record = groups[start][0]
                    # bisect selects the actual final group, never an earlier
                    # valid interval when the capture-end group is excluded.
                    if timestamp > record['end_ns']:
                        continue
                    packet = decode_packet(timestamp, raw)
                    if packet is None:
                        continue
                    orientation = direction(packet, scan.runtime_endpoints)
                    if orientation is None:
                        record['quality']['direction_skipped'] += 1
                        continue
                    slot = min((timestamp - start) * 1800 // record['duration_ns'], 1799)
                    arrays[start][0, orientation, slot] += np.uint64(1)
                    arrays[start][1, orientation, slot] += np.uint64(packet.ip_length)
        except EpisodeExcluded as exc:
            # Do not publish partial samples if the stream changed or failed.
            for record in records:
                record.update(reason=exc.reason, detail=exc.detail)
                yield record, None
            continue
        for record in sorted(records, key=lambda row: (row.get('start_ns', -1), row['annotation_row'] or 0)):
            values = arrays.get(record.get('start_ns')) if 'reason' not in record else None
            if values is not None:
                record['packet_count'] = [int(n) for n in values[0].sum(axis=1)]
                record['ip_bytes'] = [int(n) for n in values[1].sum(axis=1)]
                if not sum(record['packet_count']):
                    record.update(reason='no_direction_packets', detail='annotation interval has no usable runtime packets')
                    values = None
            yield record, values
