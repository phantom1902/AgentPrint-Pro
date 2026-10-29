#!/usr/bin/env python3
"""Run the risk/scenario matrix sequentially, using a separate NetMamba environment."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from suite import ROOT, MODELS, WEIGHTS, read


def interpreter(value):
    found = shutil.which(value)
    if found is None:
        raise FileNotFoundError('Python interpreter not found: ' + value)
    return str(Path(found).absolute())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'Benchmark/data/ANT')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs')
    p.add_argument('--weights', type=Path, default=ROOT / 'weights')
    p.add_argument('--python-main', default=sys.executable)
    p.add_argument('--python-netmamba', help='Python from the NetMamba CUDA environment')
    p.add_argument('--tasks', nargs='+', choices=['risk', 'scenario'], default=['risk', 'scenario'])
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    main_python = interpreter(args.python_main)
    if 'NetMamba' in args.models and not args.python_netmamba:
        p.error('Pass --python-netmamba for the separate NetMamba environment')
    net_python = interpreter(args.python_netmamba) if 'NetMamba' in args.models else main_python
    if not args.prepare_only:
        for model in args.models:
            if model in WEIGHTS and not (args.weights / WEIGHTS[model]).is_file():
                p.error('Missing weight: ' + str(args.weights / WEIGHTS[model]))
    common = ['--data', str(args.data.resolve()), '--output', str(args.output.resolve()),
              '--weights', str(args.weights.resolve()), '--device', args.device,
              '--workers', str(args.workers)]
    for flag, value in [('--epochs', args.epochs), ('--batch-size', args.batch_size)]:
        if value is not None:
            common += [flag, str(value)]
    if args.prepare_only:
        common += ['--prepare-only']
    subprocess.run([main_python, str(ROOT / 'suite.py'), 'prepare'] + common, check=True)
    for task in args.tasks:
        for model in args.models:
            status = args.output / task / model / 'suite_status.json'
            if status.exists() and read(status).get('status') == 'completed':
                print('Already completed:', task, model, flush=True)
                continue
            py = net_python if model == 'NetMamba' else main_python
            subprocess.run([py, str(ROOT / 'suite.py'), 'run', '--task', task,
                            '--model', model] + common, check=True)
    subprocess.run([main_python, str(ROOT / 'suite.py'), 'summarize',
                    '--output', str(args.output.resolve())], check=True)


if __name__ == '__main__':
    main()
