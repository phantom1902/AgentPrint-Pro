"""Bounded six-configuration IAT acceptance on existing train/val MTAM.

Run only after committing the implementation. This entry never discovers raw
Episodes, rebuilds base MTAM, reads test splits, or performs formal training.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import traceback

import yaml

from .cache import LABEL_MAP, read_jsonl, write_json
from .config import ConfigError, fingerprint, load_config


def _revision():
    source = Path(__file__).resolve().parent.parent
    def git(*args):
        return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ConfigError('commit tracked implementation changes before acceptance selection')
    git('ls-files', '--error-unmatch', 'agentprint/iat_smoke.py')
    return git('rev-parse', 'HEAD')


def _select_view(config, work, revision):
    """Publish ordinary cache manifests referring to unchanged original arrays."""
    from .iat import selected_base_records
    base = Path(config['paths']['cache_root']).resolve()
    index = json.loads((base / 'index.json').read_text(encoding='utf-8'))
    sources = index.get('tasks', {}).get('scenario', {}).get('sources', index.get('sources', []))
    by_path = {source['pcap_relpath']: source for source in sources}
    view = work / 'base-view'
    selected = {}
    original = {}
    for split, limit in [('train', 4), ('val', 2)]:
        records = read_jsonl(base / 'scenario' / split / 'valid.jsonl')
        metadata = json.loads((base / 'scenario' / split / 'metadata.json').read_text(encoding='utf-8'))
        if metadata.get('manifest_id') != fingerprint(records) or metadata.get('valid_count') != len(records):
            raise ConfigError('incompatible original base manifest identity: ' + split)
        if len({r['sample_id'] for r in records}) != len(records):
            raise ConfigError('duplicate original sample_id: ' + split)
        if any(row.get('label_id') not in range(5) for row in records):
            raise ConfigError('incompatible original Scenario label ontology: ' + split)
        chosen = []
        for label, label_id in LABEL_MAP['scenario'].items():
            candidates = [row for row in records if row['label_id'] == label_id]
            for row in candidates:
                source = by_path.get(row['pcap_relpath'])
                if (row.get('task') != 'scenario' or row.get('split') != split
                        or row.get('label_name') != label or source is None or source.get('split') != split):
                    raise ConfigError('incompatible selection record/source: ' + row['sample_id'])
            chosen.extend(sorted(candidates, key=lambda r:
                (by_path[r['pcap_relpath']]['size_bytes'], r['sample_id']))[:limit])
        if not chosen:
            raise ConfigError('no valid acceptance samples: ' + split)
        selected[split] = chosen
        original[split] = dict(manifest_id=metadata['manifest_id'], valid_count=metadata['valid_count'])
        target = view / 'scenario' / split
        target.mkdir(parents=True)
        (target / 'valid.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in chosen), encoding='utf-8')
        (target / 'excluded.jsonl').write_text('', encoding='utf-8')
        shards = sorted({row['shard_relpath'] for row in chosen})
        for relative in shards:
            shard = (base / relative).resolve()
            destination = view / relative
            if not shard.is_relative_to(base) or not destination.resolve().is_relative_to(view):
                raise ConfigError('base shard escapes cache root')
            if not shard.is_file():
                raise ConfigError('missing original base MTAM shard: ' + relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if os.name == 'nt':
                os.link(shard, destination)  # No-copy link without Windows symlink privileges.
            else:
                destination.symlink_to(shard)
        metadata.update(manifest_id=fingerprint(chosen), valid_count=len(chosen), excluded_count=0,
            support=[sum(row['label_id'] == i for row in chosen) for i in range(5)],
            exclusion_reasons={}, shards=len(shards))
        write_json(target / 'metadata.json', metadata)
    write_json(view / 'index.json', index)
    runtime = deepcopy(config)
    runtime['paths']['cache_root'] = str(view)
    # Public source/cache validation reads only selected train/val Episodes.
    for split in ('train', 'val'):
        selected_base_records(runtime, split)
    selection = dict(status='selected_only', git_revision=revision, scope='smoke', uses_test=False,
        selection_rule='PCAP size ascending then sample_id; max 4 train / 2 val per class',
        original_base=str(base), original_manifests=original, selected=selected)
    write_json(work / 'selection.json', selection)
    return runtime, selection


def _configs(config, work):
    settings = load_config(Path(__file__).resolve().parent.parent / 'configs/agentprint/common-v2.yaml')
    settings['paths'] = dict(data_root=config['paths']['data_root'], cache_root=str(work / 'base-view'),
                             output_root=str(work / 'runs'))
    settings['feature'] = dict(version=1, iat_enabled=True,
        fit_artifact=str(work / 'fit.json'), iat_cache_root=str(work / 'iat-cache'))
    result = []
    for variant in ('single', 'replace', 'prepend'):
        for transform in ('raw', 'log1p'):
            candidate = deepcopy(settings)
            candidate['model']['variant'], candidate['transform'] = variant, transform
            path = work / 'configs' / f'{variant}-{transform}.yaml'
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding='utf-8')
            result.append(path)
    return result


def _command(args, log):
    from .__main__ import main
    output, errors = io.StringIO(), io.StringIO()
    with redirect_stdout(output), redirect_stderr(errors):
        code = main(args)
    with log.open('a', encoding='utf-8') as handle:
        handle.write('$ python -m agentprint ' + ' '.join(args) + '\n')
        handle.write(output.getvalue() + errors.getvalue())
    if code:
        raise RuntimeError('acceptance command failed; see ' + str(log))
    return json.loads(output.getvalue().splitlines()[-1])


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config_path, work_root, workspace_root, select_only=False):
    workspace, work = Path(workspace_root).resolve(), Path(work_root).resolve()
    repository = Path(__file__).resolve().parent.parent
    if not work.is_relative_to(workspace) or work == workspace or work.is_relative_to(repository):
        raise ConfigError('acceptance work root must be outside Git and inside the designated data workspace')
    config = load_config(config_path)
    for name in ('data_root', 'cache_root'):
        source = Path(config['paths'][name]).resolve()
        if source.is_relative_to(work) or work.is_relative_to(source):
            raise ConfigError('acceptance work root overlaps source ' + name)
    revision = _revision()  # Pin committed implementation before selecting any Episode.
    work.mkdir(parents=True, exist_ok=False)
    temporary = work / 'tmp'
    temporary.mkdir()
    for name in ('TMPDIR', 'TEMP', 'TMP'):
        os.environ[name] = str(temporary)
    os.environ['XDG_CACHE_HOME'] = str(work / 'runtime-cache')
    os.environ['MPLCONFIGDIR'] = str(work / 'matplotlib')
    tempfile.tempdir = str(temporary)
    summary = dict(status='running', git_revision=revision, scope='smoke', uses_test=False,
                   formal_training=False, full_preprocessing=False, configurations=[])
    try:
        runtime, selection = _select_view(config, work, revision)
        configs = _configs(runtime, work)
        if select_only:
            return selection
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        summary['hardware'] = dict(device=device, torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
            logical_cpus=os.cpu_count(), platform=platform.platform(), numerical_threads=4)
        write_json(work / 'summary.json', summary)
        log = work / 'commands.log'
        first = ['--config', str(configs[0])]
        _command(['fit-bins', *first], log)
        _command(['prepare', *first], log)
        fit = json.loads((work / 'fit.json').read_text(encoding='utf-8'))
        summary['fit_fingerprint'] = fit['fingerprint']
        summary['iat_cache'] = {split: json.loads((work / 'iat-cache/scenario' / split / 'metadata.json').read_text(encoding='utf-8'))
                                for split in ('train', 'val')}
        for path in configs:
            common = ['--config', str(path), '--device', device]
            trained = _command(['train', *common], log)
            root = Path(trained['run_dir'])
            evaluated = _command(['evaluate', *common, '--checkpoint', str(root / 'best.pt'), '--split', 'val'], log)
            before = {name: _sha(root / name) for name in ('last.pt', 'best.pt', 'metrics.jsonl', 'result.json')}
            _command(['train', *common, '--resume', str(root)], log)
            after = {name: _sha(root / name) for name in before}
            if before != after:
                raise RuntimeError('completed-epoch resume unexpectedly changed trained artifacts: ' + str(root))
            result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
            if result['epochs'] != 2 or result['converged']:
                raise RuntimeError('acceptance two-epoch smoke budget violated')
            summary['configurations'].append(dict(name=path.stem, config=str(path), run_dir=str(root),
                result=result, validation=evaluated, resume_unchanged=True, artifact_sha256=after))
            write_json(work / 'summary.json', summary)
        summary['status'] = 'passed'
        write_json(work / 'summary.json', summary)
        return summary
    except Exception as exc:
        summary.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        write_json(work / 'summary.json', summary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='configuration referencing existing base MTAM')
    parser.add_argument('--work-root', required=True, help='new output directory; must not already exist')
    parser.add_argument('--workspace-root', required=True, help='designated data-disk workspace boundary')
    parser.add_argument('--select-only', action='store_true', help='inspect the bounded selection without fitting or training')
    args = parser.parse_args(argv)
    previous = {key: os.environ.get(key) for key in ('TMPDIR', 'TEMP', 'TMP', 'XDG_CACHE_HOME', 'MPLCONFIGDIR')}
    old_tempdir = tempfile.tempdir
    try:
        result = run(args.config, args.work_root, args.workspace_root, args.select_only)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ConfigError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print('agentprint IAT acceptance: ' + str(exc), file=sys.stderr)
        return 1
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tempfile.tempdir = old_tempdir


if __name__ == '__main__':
    raise SystemExit(main())
