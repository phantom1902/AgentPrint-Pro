#!/usr/bin/env python3
"""Bounded real-ANT CPU acceptance via public agentprint commands.

No installs, SSH, raw copies, annotation edits, test split, or production-loader
limits. Run this with the intended Python, from the matching server checkout.
All generated files belong in a fresh workspace outside Git; see --help.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
SCENARIOS = ('artifact_processing', 'business_services', 'general_knowledge',
             'scientific_evidence', 'security_testing')
CAPS = {'train': {'episodes': 20, 'annotation_rows': 64},
        'val': {'episodes': 10, 'annotation_rows': 32}}
CONFIGS = [(task, transform) for task in ('scenario', 'primitive')
           for transform in ('raw', 'log1p')]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()
            if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n',
                         encoding='utf-8')
    os.replace(temporary, path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def git_state():
    def git(*args):
        return subprocess.run(['git', *args], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    return {'commit': git('rev-parse', 'HEAD'), 'status_porcelain': git('status', '--porcelain'),
            'branch': git('rev-parse', '--abbrev-ref', 'HEAD')}


def inventory(data):
    """Only inspect official train/val metadata; never open capture payloads here."""
    sources = []
    for split in CAPS:
        for pcap in sorted((data / split).glob('*/*/*/network.pcap')):
            relative = pcap.relative_to(data)
            _, engine, scenario, episode, _ = relative.parts
            require(scenario in SCENARIOS, f'unknown scenario: {relative}')
            annotation = pcap.with_name('messages.jsonl')
            small = [path for path in sorted(pcap.parent.iterdir())
                     if path.is_file() and (path.name == 'messages.jsonl' or
                         (path.suffix.lower() in ('.json', '.jsonl', '.yaml', '.yml') and
                          ('endpoint' in path.stem.lower() or
                           path.name in ('runtime.json', 'identity.json'))))]
            files = []
            for path in [pcap, *small]:
                stat = path.stat()
                entry = {'relpath': path.relative_to(data).as_posix(),
                         'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                         'kind': 'pcap' if path == pcap else 'metadata'}
                if path != pcap:
                    entry['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                files.append(entry)
            if annotation.is_file():
                with annotation.open('rb') as handle:
                    count = sum(1 for _ in handle)
            else:
                count = 0
            sources.append({'split': split, 'engine': engine, 'scenario': scenario,
                            'episode_id': episode, 'pcap_relpath': relative.as_posix(),
                            'annotation_relpath': annotation.relative_to(data).as_posix()
                            if annotation.is_file() else None,
                            'annotation_rows': count, 'size_bytes': pcap.stat().st_size,
                            'files': files})
    return sources


def choose(sources):
    """Maximize feasible scenario coverage first, then prefer smaller captures.

    The minimum annotation cost per class reserves enough budget for the other
    feasible classes. Additional whole episodes are added round-robin. Every
    physical annotation line consumes budget, including malformed/excluded rows.
    """
    selected, summary, limitations = [], {}, []
    for split, cap in CAPS.items():
        split_sources = [source for source in sources if source['split'] == split]
        # Subset selection targets complete annotated Episodes so the same
        # selection can exercise both tasks. This is a sampling constraint,
        # never a new production scenario-quality exclusion.
        available = [source for source in split_sources if source['annotation_rows'] > 0]
        groups = {name: sorted([source for source in available if source['scenario'] == name],
                              key=lambda source: (source['size_bytes'], source['pcap_relpath']))
                  for name in SCENARIOS}
        minima = {name: min(row['annotation_rows'] for row in group)
                  for name, group in groups.items() if group}
        coverage, cost = [], 0
        for name in sorted(minima, key=lambda name: (minima[name], name)):
            if cost + minima[name] <= cap['annotation_rows']:
                coverage.append(name)
                cost += minima[name]
        chosen, used = [], 0
        for name in sorted(coverage):
            remaining = [other for other in coverage if other > name]
            room = cap['annotation_rows'] - used - sum(minima[other] for other in remaining)
            candidate = next(row for row in groups[name] if row['annotation_rows'] <= room)
            chosen.append(candidate)
            used += candidate['annotation_rows']
            groups[name].remove(candidate)
        while len(chosen) < cap['episodes']:
            progress = False
            for name in SCENARIOS:
                if len(chosen) == cap['episodes']:
                    break
                candidate = next((row for row in groups[name]
                                  if used + row['annotation_rows'] <= cap['annotation_rows']), None)
                if candidate is not None:
                    chosen.append(candidate)
                    used += candidate['annotation_rows']
                    groups[name].remove(candidate)
                    progress = True
            if not progress:
                break
        selected.extend(chosen)
        covered = sorted({row['scenario'] for row in chosen})
        summary[split] = {'available_episodes': len(split_sources),
                          'eligible_nonempty_annotation_episodes': len(available),
                          'available_scenarios': sorted(minima),
                          'selected_episodes': len(chosen), 'selected_annotation_rows': used,
                          'selected_pcap_bytes': sum(row['size_bytes'] for row in chosen),
                          'selected_scenarios': covered,
                          'missing_scenarios': sorted(set(SCENARIOS) - set(covered)), 'caps': cap}
        if len(covered) < 5 or len(chosen) < cap['episodes']:
            limitations.append(f'{split}: selected {len(chosen)} episodes across {len(covered)}/5 '
                                f'scenarios within the complete-annotation row budget; '
                                'valid sample counts remain subject to unchanged quality rules')
    return sorted(selected, key=lambda row: row['pcap_relpath']), summary, limitations


def select(data, work):
    import yaml
    require(data.is_dir(), f'raw data root does not exist: {data}')
    require(not work.exists(), f'workspace already exists; use a fresh path: {work}')
    sources, summary, limitations = choose(inventory(data))
    work.mkdir(parents=True)
    selection = {'schema_version': 1, 'created_utc': utc(), 'seed': 42,
                 'algorithm': 'complete-episode-budgeted-size-v1',
                 'data_root': str(data), 'source_identity': 'PCAP path/size/mtime; small metadata SHA256',
                 'pcap_identity_limitation': 'cannot detect PCAP replacement with unchanged size and mtime',
                 'source_git': git_state(), 'split_summary': summary, 'sources': sources,
                 'limitations': limitations, 'uses_test': False, 'modifies_annotations': False}
    selection['selection_id'] = digest(selection)
    write(work / 'selection.json', selection)
    require(all(summary[split]['selected_episodes'] for split in CAPS),
            'no valid samples can be prepared: selection has no episodes in train or val')
    for source in sources:
        for file in source['files']:
            link = work / 'selected-data' / file['relpath']
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(data / file['relpath'])
    for task, transform in CONFIGS:
        config = {'extends': str(REPO / f'configs/agentprint/smoke_{task}_{transform}.yaml'),
                  'seed': 42, 'mode': 'smoke',
                  'preprocessing': {'splits': ['train', 'val']},
                  'training': {'epochs': 1, 'batch_size': 4},
                  'paths': {'data_root': str(work / 'selected-data'),
                            'cache_root': str(work / 'cache'), 'output_root': str(work / 'runs')}}
        target = work / 'configs' / f'{task}_{transform}.yaml'
        target.parent.mkdir(exist_ok=True)
        target.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    write(work / 'summary.json', {'status': 'selected_not_accepted',
                                 'selection_id': selection['selection_id'], 'limitations': limitations})
    return selection


def verify_selection(work, selection):
    expected = dict(selection)
    identity = expected.pop('selection_id')
    require(digest(expected) == identity, 'selection identity differs from recorded contents')
    expected_pcaps = set()
    for source in selection['sources']:
        require(source['split'] in CAPS, 'selection contains a non-train/val source')
        expected_pcaps.add(source['pcap_relpath'])
        for file in source['files']:
            original = Path(selection['data_root']) / file['relpath']
            link = work / 'selected-data' / file['relpath']
            require(link.is_symlink() and link.resolve() == original.resolve(),
                    f'selected file is not the original individual-file symlink: {link}')
            stat = link.stat()
            require((stat.st_size, stat.st_mtime_ns) == (file['size_bytes'], file['mtime_ns']),
                    f'source changed size/mtime: {link}')
            if 'sha256' in file:
                require(hashlib.sha256(link.read_bytes()).hexdigest() == file['sha256'],
                        f'source metadata content changed: {link}')
    actual = {path.relative_to(work / 'selected-data').as_posix()
              for path in (work / 'selected-data').rglob('*.pcap')}
    require(actual == expected_pcaps, 'selected data tree contains unexpected/missing captures')
    for split, cap in CAPS.items():
        sources = [row for row in selection['sources'] if row['split'] == split]
        require(0 < len(sources) <= cap['episodes'], f'{split} episode selection cap violated')
        require(sum(row['annotation_rows'] for row in sources) <= cap['annotation_rows'],
                f'{split} whole-annotation selection budget violated')


def environment(threads):
    # Importing here retains framework memory throughout all four training runs.
    # A short-lived process releases it before preparation/training can begin.
    probe = subprocess.run([sys.executable, str(REPO / 'tools/agentprint_cpu_environment.py')],
                           cwd=REPO, capture_output=True, text=True, check=True,
                           env=dict(os.environ, OMP_NUM_THREADS=str(threads),
                                    MKL_NUM_THREADS=str(threads), CUDA_VISIBLE_DEVICES=''))
    packages = json.loads(probe.stdout)
    return {'recorded_utc': utc(), 'python': sys.version, 'executable': sys.executable,
            'platform': platform.platform(), 'git': git_state(), 'packages': packages,
            'cpu_count': os.cpu_count(), 'cpu_threads': threads,
            'subprocess_environment': {'OMP_NUM_THREADS': str(threads),
                                       'MKL_NUM_THREADS': str(threads), 'CUDA_VISIBLE_DEVICES': ''},
            'device': 'cpu', 'installs_performed': False}


def command(work, label, args, threads):
    """Keep exact argv, stdout/stderr and wall time even on command failure."""
    log_dir = work / 'logs'
    log_dir.mkdir(exist_ok=True)
    stdout, stderr = log_dir / f'{label}.stdout.log', log_dir / f'{label}.stderr.log'
    argv = [sys.executable, '-m', 'agentprint', *args]
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1')
    started, stamp = time.monotonic(), utc()
    with stdout.open('w', encoding='utf-8') as out, stderr.open('w', encoding='utf-8') as err:
        result = subprocess.run(argv, cwd=REPO, env=env, stdout=out, stderr=err, text=True)
    record = {'label': label, 'argv': argv, 'cwd': str(REPO), 'started_utc': stamp,
              'elapsed_seconds': time.monotonic() - started, 'returncode': result.returncode,
              'stdout': str(stdout), 'stderr': str(stderr),
              'environment': {name: env[name] for name in
                              ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')}}
    with (work / 'commands.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record, sort_keys=True) + '\n')
    print(json.dumps(record, sort_keys=True), flush=True)
    require(result.returncode == 0, f'public command failed: {label}; read {stderr}')
    return [json.loads(line) for line in stdout.read_text(encoding='utf-8').splitlines()
            if line.strip().startswith('{')][-1]


def verify_cache(work, selection):
    import numpy as np
    verify_selection(work, selection)
    cache = work / 'cache'
    index = read(cache / 'index.json')
    require(index['complete'], 'cache index is incomplete')
    expected_labels = {'scenario': {name: i for i, name in enumerate(SCENARIOS)},
                       'primitive': {f'N{i + 1:03d}': i for i in range(47)}}
    require(read(cache / 'label_map.json') == expected_labels, 'fixed label map differs')
    label_id = digest(expected_labels)
    sources = {row['pcap_relpath']: row for row in selection['sources']}
    report = {'status': 'passed', 'selection_id': selection['selection_id'], 'datasets': {}}
    for task, _ in CONFIGS[::2]:
        require(index['tasks'][task]['complete'], f'{task} cache contract incomplete')
        require({row['pcap_relpath'] for row in index['tasks'][task]['sources']} == set(sources),
                f'{task} cache source inventory differs from selection')
        for split, cap in CAPS.items():
            base = cache / task / split
            metadata, valid, excluded = read(base / 'metadata.json'), rows(base / 'valid.jsonl'), rows(base / 'excluded.jsonl')
            name = f'{task}/{split}'
            require(metadata['complete'] and metadata['label_map_id'] == label_id,
                    f'{name}: incomplete/incompatible metadata')
            require(metadata == index['datasets'][name], f'{name}: index metadata differs')
            require(metadata['manifest_id'] == digest(valid), f'{name}: manifest identity differs')
            require(len(valid) == metadata['valid_count'] and len(excluded) == metadata['excluded_count'],
                    f'{name}: count differs from metadata')
            require(0 < len(valid) <= cap['episodes' if task == 'scenario' else 'annotation_rows'],
                    f'{name}: no valid samples or acceptance cap exceeded')
            require(len({row['sample_id'] for row in valid}) == len(valid), f'{name}: duplicate sample IDs')
            require(not any('row_index' in row or 'shard_relpath' in row for row in excluded),
                    f'{name}: exclusion has a fabricated cache index')
            support, shard_rows = [0] * len(expected_labels[task]), {}
            for record in valid:
                source = sources.get(record['pcap_relpath'])
                require(source is not None and source['split'] == split and record['split'] == split
                        and record['task'] == task, f'{name}: record lacks selection lineage')
                require(record['label_id'] == expected_labels[task][record['label_name']],
                        f'{name}: label mapping differs')
                for field in ('start_ns', 'end_ns', 'duration_ns'):
                    require(type(record[field]) is int, f'{name}: timestamp is not integer nanoseconds')
                require(record['end_ns'] > record['start_ns'] and
                        record['duration_ns'] == record['end_ns'] - record['start_ns'],
                        f'{name}: full interval duration differs')
                if task == 'primitive':
                    require(record['annotation_relpath'] == source['annotation_relpath'] and
                            1 <= record['annotation_row'] <= source['annotation_rows'],
                            f'{name}: original annotation provenance differs')
                shard = record['shard_relpath']
                require(Path(shard).parts[:3] == (task, split, 'shards') and '..' not in Path(shard).parts,
                        f'{name}: shard path escapes its task/split')
                shard_rows.setdefault(shard, []).append(record)
                support[record['label_id']] += 1
            require(support == metadata['support'], f'{name}: class support differs')
            require(dict(Counter(row['reason'] for row in excluded)) == metadata['exclusion_reasons'],
                    f'{name}: exclusion reasons differ')
            for shard, records in shard_rows.items():
                values = np.load(cache / shard, mmap_mode='r', allow_pickle=False)
                require(values.dtype == np.uint64 and values.shape == (len(records), 2, 2, 1800)
                        and len(values) <= 1024, f'{name}: shard shape/dtype differs')
                require(sorted(record['row_index'] for record in records) == list(range(len(values))),
                        f'{name}: shard row references differ')
                for record in records:
                    array = values[record['row_index']]
                    require([int(n) for n in array[0].sum(axis=1)] == record['packet_count'] and
                            [int(n) for n in array[1].sum(axis=1)] == record['ip_bytes'] and
                            sum(record['packet_count']) > 0, f'{name}: shard statistics differ')
                del values
            require(len(shard_rows) == metadata['shards'], f'{name}: shard count differs')
            report['datasets'][name] = {'manifest_id': metadata['manifest_id'],
                                       'valid_count': len(valid), 'excluded_count': len(excluded),
                                       'support': support, 'exclusion_reasons': metadata['exclusion_reasons'],
                                       'shards': sorted(shard_rows)}
    write(work / 'cache-verification.json', report)
    return report


def verify_metrics(directory, task, expected_count):
    summary = read(directory / 'summary.json')
    classes = 5 if task == 'scenario' else 47
    with (directory / 'per_class.csv').open(newline='', encoding='utf-8') as handle:
        per_class = list(csv.DictReader(handle))
    with (directory / 'confusion_matrix.csv').open(newline='', encoding='utf-8') as handle:
        matrix = list(csv.reader(handle))
    require(len(per_class) == classes and len(matrix) == classes + 1 and
            all(len(row) == classes + 1 for row in matrix), f'{directory}: fixed-class CSV shape differs')
    require(sum(int(row['support']) for row in per_class) == expected_count and
            sum(sum(int(value) for value in row[1:]) for row in matrix[1:]) == expected_count and
            summary['sample_count'] == expected_count, f'{directory}: evaluation count differs')
    return summary


def verify_runs(work, selection):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    import yaml
    cache = verify_cache(work, selection)
    results = read(work / 'run-results.json')
    require(len(results) == 4 and {(row['task'], row['transform']) for row in results} == set(CONFIGS),
            'four independent task/transform runs are required')
    report = {'status': 'passed', 'selection_id': selection['selection_id'], 'runs': []}
    for result in results:
        task, transform, root = result['task'], result['transform'], Path(result['run_dir'])
        config = yaml.safe_load((root / 'resolved_config.yaml').read_text(encoding='utf-8'))
        require(config['mode'] == 'smoke' and config['seed'] == 42 and config['task'] == task and
                config['transform'] == transform and config['training']['epochs'] == 1 and
                config['training']['batch_size'] == 4 and config['preprocessing']['splits'] == ['train', 'val'],
                f'{root}: independent CPU configuration differs')
        history = rows(root / 'runtime_history.jsonl')
        require(history and all(row['device'] == 'cpu' for row in history), f'{root}: runtime is not CPU')
        require(all(row['environment']['git_commit'] == selection['source_git']['commit'] for row in history),
                f'{root}: runtime commit differs from selection')
        metrics = rows(root / 'metrics.jsonl')
        require(len(metrics) == 1 and metrics[0]['epoch'] == 0, f'{root}: expected one committed epoch')
        counts = {split: cache['datasets'][f'{task}/{split}']['valid_count'] for split in CAPS}
        require(metrics[0]['train_samples'] == counts['train'] and metrics[0]['val_samples'] == counts['val'],
                f'{root}: training counts differ')
        for name in ('best.pt', 'last.pt'):
            saved = torch.load(root / name, map_location='cpu', weights_only=False)
            require(saved['epoch'] == 0 and saved['config'] == config, f'{root}/{name}: checkpoint differs')
            for split in CAPS:
                require(saved[f'{split}_manifest_id'] == cache['datasets'][f'{task}/{split}']['manifest_id'],
                        f'{root}/{name}: manifest reference differs')
            require(saved['model'], f'{root}/{name}: empty model state')
            if name == 'last.pt':
                require(all(key in saved for key in ('optimizer', 'rng', 'best_checkpoint')),
                        f'{root}: incomplete recovery state')
                require((root / saved['best_checkpoint']).is_file(), f'{root}: missing best snapshot')
            del saved
        verify_metrics(root, task, counts['val'])
        evaluation = verify_metrics(root / 'evaluation-val', task, counts['val'])
        require(evaluation['split'] == 'val' and not evaluation['formal_experiment'],
                f'{root}: evaluation split/protocol differs')
        event_files = list((root / 'tensorboard' / root.name).rglob('events.out.tfevents.*'))
        require(event_files, f'{root}: TensorBoard events are missing')
        epoch_events = EventAccumulator(str(root / 'tensorboard' / root.name / 'epochs')).Reload()
        for tag in ('train_loss', 'val_loss', 'val_accuracy', 'val_macro_f1', 'learning_rate'):
            events = epoch_events.Scalars(tag)
            require(len(events) == 1 and events[0].step == 0, f'{root}: TensorBoard {tag} differs')
        report['runs'].append({'task': task, 'transform': transform, 'run_dir': str(root),
                               'train_count': counts['train'], 'val_count': counts['val'],
                               'tensorboard_files': [str(path) for path in event_files],
                               'epoch_elapsed_seconds': metrics[0]['elapsed_seconds'],
                               'val_summary': evaluation})
    selections = list((work / 'runs').glob('selection-*.json'))
    require(len(selections) == 2 and all(read(path)['complete'] for path in selections),
            'both transform comparison artifacts must be complete')
    write(work / 'artifact-verification.json', report)
    return report


def execute(work, selection, threads, prepare_only=False):
    require(not (work / 'execution.json').exists(),
            'workspace already has an execution attempt; preserve it and use a fresh workspace')
    verify_selection(work, selection)
    require(git_state()['commit'] == selection['source_git']['commit'],
            'checkout commit changed since selection; use a fresh workspace')
    write(work / 'execution.json', {'started_utc': utc(), 'prepare_only': prepare_only})
    env = environment(threads)
    write(work / 'environment.json', env)
    require(all(package['import'] == 'ok' for package in env['packages'].values()),
            'dependency import check failed; see environment.json; no packages were installed')
    started, results = time.monotonic(), []
    for task, transform in CONFIGS:
        if prepare_only and transform == 'log1p':
            continue
        config = str(work / 'configs' / f'{task}_{transform}.yaml')
        command(work, f'{task}-{transform}-prepare', ['prepare', '--config', config], threads)
        # Both tasks must pass real preparation before any training is started.
    cache = verify_cache(work, selection)
    write(work / 'summary.json', {'status': 'prepared_not_accepted',
                                 'selection_id': selection['selection_id'],
                                 'datasets': cache['datasets'], 'limitations': selection['limitations']})
    if prepare_only:
        return
    for task, transform in CONFIGS:
        config = str(work / 'configs' / f'{task}_{transform}.yaml')
        trained = command(work, f'{task}-{transform}-train',
                          ['train', '--config', config, '--device', 'cpu'], threads)
        root = Path(trained['run_dir'])
        command(work, f'{task}-{transform}-evaluate',
                ['evaluate', '--config', config, '--device', 'cpu', '--split', 'val',
                 '--checkpoint', str(root / 'best.pt')], threads)
        results.append({'task': task, 'transform': transform, 'run_dir': str(root)})
        write(work / 'run-results.json', results)
    verify_runs(work, selection)
    write(work / 'summary.json', {'status': 'cpu_acceptance_passed', 'completed_utc': utc(),
                                 'selection_id': selection['selection_id'],
                                 'git_commit': selection['source_git']['commit'],
                                 'elapsed_seconds': time.monotonic() - started,
                                 'datasets': cache['datasets'], 'runs': results,
                                 'limitations': selection['limitations'] + [
                                     'Small-subset smoke scores do not estimate model performance.',
                                     'This does not complete full preparation or formal GPU experiments.']})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('select', 'prepare', 'run', 'verify-cache', 'verify'),
                        help='select only; prepare both tasks only; run all four serially; '
                             'verify-cache or verify existing artifacts')
    parser.add_argument('--data-root', type=Path,
                        default=Path('/root/autodl-tmp/ant-workspace/data/raw/ANT'))
    parser.add_argument('--work-root', required=True, type=Path,
                        help='fresh independent acceptance directory outside the Git checkout')
    parser.add_argument('--cpu-threads', type=int, default=2,
                        help='recorded OMP/MKL threads for serial CPU subprocesses (default: 2)')
    args = parser.parse_args(argv)
    data, work = args.data_root.resolve(), args.work_root.resolve()
    # Reject dangerous nesting before creating any output or following links.
    require(not work.is_relative_to(REPO) and not REPO.is_relative_to(work),
            'acceptance workspace must be outside the source checkout')
    require(not work.is_relative_to(data) and not data.is_relative_to(work),
            'acceptance workspace must not overlap the raw data root')
    require(args.cpu_threads > 0, 'cpu-threads must be positive')
    may_report = False
    fresh_workspace = not work.exists()
    try:
        if args.action == 'select' or not work.exists() and args.action in ('prepare', 'run'):
            selection = select(data, work)
            may_report = True
        else:
            require(work.is_dir(), f'acceptance workspace does not exist: {work}')
            selection = read(work / 'selection.json')
            may_report = True
            require(data == Path(selection['data_root']), 'data root differs from saved selection')
        if args.action in ('prepare', 'run'):
            execute(work, selection, args.cpu_threads, prepare_only=args.action == 'prepare')
        elif args.action == 'verify-cache':
            verify_cache(work, selection)
        elif args.action == 'verify':
            verify_runs(work, selection)
        print(json.dumps({'action': args.action, 'work_root': str(work),
                          'summary': read(work / 'summary.json')}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, ImportError, subprocess.SubprocessError) as exc:
        if may_report or fresh_workspace and work.is_dir():
            write(work / 'summary.json', {'status': 'incomplete', 'failed_action': args.action,
                                         'reason': str(exc), 'recorded_utc': utc()})
        print(f'CPU acceptance incomplete: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
