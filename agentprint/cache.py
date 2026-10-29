"""Versioned JSONL manifests and bounded, whole-sample NPY shards."""
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import FORMAT_VERSION, PREPROCESSING_VERSION
from .config import fingerprint


LABEL_MAP = {
    'scenario': {label: index for index, label in enumerate((
        'artifact_processing', 'business_services', 'general_knowledge',
        'scientific_evidence', 'security_testing'))},
    'primitive': {f'N{index + 1:03d}': index for index in range(47)},
}
LABEL_MAP_ID = fingerprint(LABEL_MAP)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
                    encoding='utf-8')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_split(cache_root, task, split, samples, config_fingerprint):
    """Consume (record, raw MTAM or None) pairs without retaining all tensors."""
    root = Path(cache_root)
    base = root / task / split
    (base / 'shards').mkdir(parents=True, exist_ok=False)
    valid_path, excluded_path = base / 'valid.jsonl', base / 'excluded.jsonl'
    records: list[dict[str, Any]] = []
    arrays: list[np.ndarray] = []
    shard_index = 0
    support = [0] * len(LABEL_MAP[task])
    reasons: Counter[str] = Counter()
    valid_count, excluded_count = 0, 0
    with valid_path.open('w', encoding='utf-8') as valid, excluded_path.open('w', encoding='utf-8') as excluded:
        def flush():
            nonlocal shard_index
            if not arrays:
                return
            relative = f'{task}/{split}/shards/{shard_index:05d}.npy'
            # A shard is a complete [n,2,2,1800] array, never a packet fragment.
            np.save(root / relative, np.stack(arrays), allow_pickle=False)
            for index, record in enumerate(records):
                record.update(shard_relpath=relative, row_index=index)
                valid.write(json.dumps(record, sort_keys=True) + '\n')
            valid.flush()
            arrays.clear()
            records.clear()
            shard_index += 1
        for record, array in samples:
            if array is None:
                excluded.write(json.dumps(record, sort_keys=True) + '\n')
                reasons[record['reason']] += 1
                excluded_count += 1
                continue
            if array.dtype != np.uint64 or array.shape != (2, 2, 1800):
                raise ValueError('raw MTAM must have uint64 shape [2,2,1800]')
            arrays.append(array)
            records.append(record)
            support[record['label_id']] += 1
            valid_count += 1
            if len(arrays) == 1024:
                flush()
        flush()
    manifest_id = fingerprint(read_jsonl(valid_path))
    metadata = {'format_version': FORMAT_VERSION,
                'preprocessing_version': PREPROCESSING_VERSION,
                'config_fingerprint': config_fingerprint, 'label_map_id': LABEL_MAP_ID,
                'manifest_id': manifest_id, 'task': task, 'split': split,
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
