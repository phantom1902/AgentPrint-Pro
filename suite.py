#!/usr/bin/env python3
"""ANT risk/scenario suite: public split PCAPs -> predictions -> episode metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'Benchmark'), str(ROOT / 'Benchmark/scripts')]
SPLITS = ('train', 'val', 'test')
WF = ('DF', 'AWF', 'TikTok', 'VarCNN', 'NetCLR', 'TMWF', 'ARES', 'BAPM')
MODELS = ('FlowLens', 'YaTC', 'NetMamba', 'TFusion', 'TrafficFormer') + WF
WEIGHTS = {'YaTC': 'yatc-pretrained.pth', 'NetMamba': 'netmamba-pretrained.pth',
           'TrafficFormer': 'trafficformer-pretrained.bin'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')


def task_label(episode, task):
    if task == 'risk':
        return 'malicious' if episode['family'] in {'attack_code_generation', 'attack_execution'} else 'benign'
    match = re.fullmatch(r'A(\d{2})\d{2}', episode['episode'])
    groups = read(ROOT / 'configs/scenario.json')['groups']
    matches = [g['id'] for g in groups if match and int(match[1]) in g['tasks']]
    if len(matches) != 1:
        raise ValueError('Unmapped scenario episode: ' + str(episode))
    return matches[0]


def scan(data):
    from ant_pcap_native import discover_episodes
    episodes = discover_episodes(data, SPLITS)
    if not episodes:
        raise ValueError('Expected <data>/<train|val|test>/<engine>/<family>/Axxxx/network.pcap')
    result, seen, targets = [], set(), set()
    for ep in episodes:
        key = ep.engine + '/' + ep.episode_id
        target = str(ep.pcap_path.resolve())
        if key in seen or target in targets:
            raise ValueError('Duplicate or cross-split episode: ' + key)
        seen.add(key)
        targets.add(target)
        if not ep.pcap_path.is_file() or ep.pcap_path.stat().st_size < 24:
            raise ValueError('Missing/invalid PCAP: ' + str(ep.pcap_path))
        result.append(dict(split=ep.split, engine=ep.engine, family=ep.scenario,
                           episode=ep.episode_id, id=key, pcap=target))
    for task in ('risk', 'scenario'):
        expected = {'benign', 'malicious'} if task == 'risk' else {g['id'] for g in read(ROOT / 'configs/scenario.json')['groups']}
        for split in SPLITS:
            observed = {task_label(r, task) for r in result if r['split'] == split}
            if observed != expected:
                raise ValueError(f'{task}/{split}: expected {sorted(expected)}, found {sorted(observed)}')
    return result


def prepare(args):
    from collections import Counter
    episodes = scan(args.data.resolve())
    payload = dict(data_root=str(args.data.resolve()), episodes=episodes)
    path = args.output / 'dataset.json'
    if path.exists() and read(path) != payload:
        raise ValueError('Output contains a different dataset manifest; use a new output directory')
    write(path, payload)
    counts = {}
    for task in ('risk', 'scenario'):
        counts[task] = {}
        for split in SPLITS:
            rows = [r for r in episodes if r['split'] == split]
            sources = [dict(pcap=r['pcap'], source=r['split'] + '_' + r['id'].replace('/', '_'),
                            default_label=task_label(r, task)) for r in rows]
            write(args.output / 'manifests' / task / (split + '.json'), {'sources': sources})
            counts[task][split] = dict(Counter(s['default_label'] for s in sources))
    write(args.output / 'dataset_counts.json', counts)
    print(json.dumps(counts, indent=2))


def execute(command, cwd, log, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    print('Running; log: ' + str(log), flush=True)
    write(log.with_suffix('.command.json'), dict(argv=[str(x) for x in command], cwd=str(cwd)))
    with log.open('w') as handle:
        subprocess.run([str(x) for x in command], cwd=cwd, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def save_predictions(dest, soft, rows, labels, task):
    import numpy as np
    from Benchmark.evaluation import evaluate_predictions
    soft = np.asarray(soft, dtype=float)
    if soft.shape != (len(rows), len(labels)) or not len(rows) or not np.isfinite(soft).all():
        raise ValueError('Invalid prediction matrix or row alignment: ' + str(dest))
    if (soft < -1e-6).any() or not np.allclose(soft.sum(axis=1), 1, atol=1e-4):
        raise ValueError('Expected normalized class probabilities')
    dest.mkdir(parents=True, exist_ok=True)
    truth = [labels.index(task_label(r, task)) for r in rows]
    np.save(dest / 'soft_predictions.npy', soft.astype(np.float32))
    (dest / 'test_labels.txt').write_text('\n'.join(map(str, truth)) + '\n')
    (dest / 'hard_predictions.txt').write_text('\n'.join(map(str, soft.argmax(1))) + '\n')
    write(dest / 'label_mapping.json', {'id_to_label': {str(i): x for i, x in enumerate(labels)}})
    write(dest / 'rows.json', rows)
    return evaluate_predictions(dest / 'hard_predictions.txt', dest / 'test_labels.txt',
                                dest / 'evaluation_metrics.json', dest / 'soft_predictions.npy',
                                dest / 'label_mapping.json')


def pool(soft, rows, rule):
    import numpy as np
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r['id'], []).append(i)
    probabilities, episodes = [], []
    for indices in groups.values():
        p = soft[indices]
        if rule == 'mean_soft':
            q = p.mean(0)
        elif rule == 'majority_vote':
            q = np.bincount(p.argmax(1), minlength=p.shape[1]) / len(p)
        elif rule == 'max_confidence':
            q = p[p.max(1).argmax()]
        elif rule == 'confidence_weighted_mean':
            entropy = -(p * np.log(np.clip(p, 1e-12, 1))).sum(1)
            weights = np.maximum(0, 1 - entropy / np.log(p.shape[1]))
            q = np.average(p, axis=0, weights=weights) if weights.sum() > 1e-12 else p.mean(0)
        else:
            raise ValueError(rule)
        probabilities.append(q)
        episodes.append(rows[indices[0]])
    return np.asarray(probabilities), episodes


def aggregate(run, task, expected):
    import numpy as np
    val = run / 'val'
    labels = list(read(val / 'label_mapping.json')['id_to_label'].values())
    soft, rows = np.load(val / 'soft_predictions.npy'), read(val / 'rows.json')
    rules = ('mean_soft', 'majority_vote', 'confidence_weighted_mean', 'max_confidence')
    scores = {}
    for rule in rules:
        p, eps = pool(soft, rows, rule)
        truth = np.asarray([labels.index(task_label(r, task)) for r in eps])
        scores[rule] = float((p.argmax(1) == truth).mean())
    # Stable, declared tie break. Test predictions are read only after selection.
    selected = max(rules, key=lambda r: scores[r])
    write(run / 'aggregation.json', dict(selected_rule=selected, validation_accuracy=scores,
                                        selection='validation episode accuracy', tie_break=list(rules)))
    coverage = {}
    for split in ('val', 'test'):
        part = run / split
        if list(read(part / 'label_mapping.json')['id_to_label'].values()) != labels:
            raise ValueError('Validation/test class columns differ')
        p, eps = pool(np.load(part / 'soft_predictions.npy'), read(part / 'rows.json'), selected)
        save_predictions(run / 'episode' / split, p, eps, labels, task)
        observed = {r['id'] for r in eps}
        full = {r['id'] for r in expected if r['split'] == split}
        coverage[split] = {'expected': len(full), 'evaluated': len(observed),
                           'missing_episode_ids': sorted(full - observed)}
        if not observed <= full:
            raise ValueError('Predictions contain episodes outside the reserved split')
    write(run / 'coverage.json', coverage)


def summarize(args):
    rows = []
    for task in ('risk', 'scenario'):
        for model in MODELS:
            run = args.output / task / model
            status = read(run / 'suite_status.json') if (run / 'suite_status.json').exists() else {}
            p = run / 'episode/test/evaluation_metrics.json'
            metrics = read(p) if p.exists() and status.get('status') == 'completed' else {}
            cp = run / 'coverage.json'
            coverage = read(cp).get('test', {}) if cp.exists() else {}
            rows.append(dict(task=task, baseline=model, status=status.get('status', 'not_run'),
                             expected_episodes=coverage.get('expected'),
                             evaluated_episodes=coverage.get('evaluated'),
                             accuracy=metrics.get('accuracy'), macro_f1=metrics.get('macro_f1'),
                             malicious_f1=metrics.get('malicious_f1'), recall=metrics.get('malicious_recall'),
                             fpr=metrics.get('malicious_fpr'),
                             balanced_accuracy=metrics.get('balanced_accuracy')))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output / 'results.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'summarize', 'check'])
    parser.add_argument('--data', type=Path, default=ROOT / 'Benchmark/data/ANT')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    parser.add_argument('--task', choices=['risk', 'scenario'], default='risk')
    parser.add_argument('--model', choices=MODELS, default='FlowLens')
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, help='Explicit budget override; recorded as a non-default run')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.data, args.output, args.weights = args.data.resolve(), args.output.resolve(), args.weights.resolve()
    if args.command in ('prepare', 'check'):
        if args.command == 'prepare':
            prepare(args)
        else:
            episodes = scan(args.data)
            print('Validated episodes:', len(episodes))
            for model, name in WEIGHTS.items():
                print(model, 'weight present' if (args.weights / name).is_file() else 'weight missing', name)
        return
    if args.command == 'summarize':
        summarize(args)
        return
    if not (args.output / 'dataset.json').exists():
        prepare(args)
    dataset = read(args.output / 'dataset.json')
    if Path(dataset['data_root']) != args.data:
        raise ValueError('Dataset differs from the prepared manifest')
    run = args.output / args.task / args.model
    if (run / 'suite_status.json').exists() and read(run / 'suite_status.json').get('status') != 'prepared':
        raise ValueError('This run already has a status file; use a new --output to avoid mixing runs')
    if args.model in WEIGHTS and not args.prepare_only and not (args.weights / WEIGHTS[args.model]).is_file():
        raise FileNotFoundError('Supply pretrained checkpoint: ' + str(args.weights / WEIGHTS[args.model]))
    status = dict(status='running', arguments=vars(args), python=sys.version)
    write(run / 'suite_status.json', status)
    try:
        from engines import run_model
        run_model(args, run, dataset['episodes'])
        status['status'] = 'prepared' if args.prepare_only else 'completed'
        if not args.prepare_only and args.model not in WF:
            aggregate(run, args.task, dataset['episodes'])
    except Exception as exc:
        status.update(status='failed', error=repr(exc))
        raise
    finally:
        write(run / 'suite_status.json', status)
    print('status=' + status['status'] + ' output=' + str(run))


if __name__ == '__main__':
    main()
