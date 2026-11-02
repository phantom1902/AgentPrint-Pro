"""Read-only full-cache/GPU checks against the completed formal experiment."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import importlib.metadata
import torch

from agentprint.config import load_config
from agentprint.v2 import experiment_identity, loader


def main():
    root = Path('/root/autodl-tmp/ant-workspace')
    config = load_config(sys.argv[1])
    state = json.loads((root / 'reports/iat-full-20261102/state.json').read_text())
    assert state['status'] == 'completed'
    anchor = next(x for x in state['configurations'] if x['name'] == 'replace-log1p')
    saved = torch.load(Path(anchor['run_dir']) / 'best.pt', map_location='cpu', weights_only=False)
    datasets = {split: loader(config, split)[0] for split in ('train', 'val')}
    identity = experiment_identity(config, datasets['train'], datasets['val'])
    previous = dict(saved['identity'], mode='tuning')
    assert identity == previous, 'workload differs from completed formal baseline'
    assert [len(datasets[s]) for s in ('train', 'val')] == [2697, 202]
    assert torch.cuda.is_available()
    assert importlib.metadata.version('optim-agent') == '0.1.1'
    assert shutil.disk_usage(root).free > 8 * 2**30
    report = dict(
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        torch=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), free_disk_bytes=shutil.disk_usage(root).free,
        sample_counts={s: len(datasets[s]) for s in datasets},
        base_manifests={s: identity['base_cache'][s]['manifest_id'] for s in datasets},
        iat_cache_fingerprints={s: identity['iat_cache'][s]['fingerprint'] for s in datasets},
        fit_fingerprint=identity['fit_fingerprint'], matches_formal_workload=True,
        anchor_result=anchor['result'], uses_test=False)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
