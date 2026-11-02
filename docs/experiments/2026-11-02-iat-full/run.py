"""Run the authorized full train/val IAT pipeline through public commands.

Launch once with the project's Python from the committed repository root.
State and command logs survive SSH disconnection. No real test evaluation.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


WORKSPACE = Path('/root/autodl-tmp/ant-workspace')
REPOSITORY = WORKSPACE / 'src/AgentPrint-Pro-IAT'
REPORT = WORKSPACE / 'reports/iat-full-20261102'
CONFIGS = REPOSITORY / 'configs/agentprint/iat-full-20261102'


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    sys.path.insert(0, str(REPOSITORY))
    REPORT.mkdir(parents=True, exist_ok=True)
    state_path = REPORT / 'state.json'
    if state_path.exists():
        raise RuntimeError('existing pipeline state; inspect before any manual restart')
    revision = subprocess.check_output(
        ['git', '-C', str(REPOSITORY), 'rev-parse', 'HEAD'], text=True).strip()
    tracked_changes = subprocess.check_output(
        ['git', '-C', str(REPOSITORY), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if tracked_changes:
        raise RuntimeError('tracked implementation/configuration must be committed')
    for key in ('TMPDIR', 'TEMP', 'TMP'):
        os.environ[key] = str(WORKSPACE / 'tmp/iat-full-20261102')
    os.environ['XDG_CACHE_HOME'] = str(WORKSPACE / 'runtime-cache/iat-full-20261102')
    os.environ['MPLCONFIGDIR'] = str(WORKSPACE / 'runtime-cache/matplotlib-iat-full-20261102')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '4'
    for key in ('TMPDIR', 'XDG_CACHE_HOME', 'MPLCONFIGDIR'):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    state = dict(status='running', started_utc=now(), git_revision=revision,
                 pid=os.getpid(), scope='full_train_val', uses_test=False,
                 configurations=[], commands=[], stage='preflight')

    def publish():
        state['updated_utc'] = now()
        temporary = state_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(state, sort_keys=True, indent=2) + '\n', encoding='utf-8')
        os.replace(temporary, state_path)

    def command(arguments, log_name):
        argv = [sys.executable, '-u', '-m', 'agentprint', *arguments]
        log = REPORT / log_name
        entry = dict(argv=argv, started_utc=now(), log=str(log))
        state['commands'].append(entry)
        publish()
        print(json.dumps(dict(event='start', **entry)), flush=True)
        with log.open('w', encoding='utf-8') as handle:
            result = subprocess.run(argv, cwd=REPOSITORY, stdout=handle, stderr=subprocess.STDOUT)
        entry.update(exit_code=result.returncode, ended_utc=now())
        publish()
        if result.returncode:
            raise RuntimeError(f'command exit {result.returncode}; see {log}')
        lines = [line for line in log.read_text(encoding='utf-8').splitlines() if line.strip()]
        return json.loads(lines[-1])

    publish()
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('authorized full training requires the provisioned GPU')
        from agentprint.config import load_config
        from agentprint.iat import selected_base_records
        from agentprint.v2_protocol import validate_formal
        paths = [CONFIGS / f'{variant}-{transform}.yaml'
                 for variant in ('single', 'replace', 'prepend')
                 for transform in ('raw', 'log1p')]
        for path in paths:
            validate_formal(load_config(path))
        config = load_config(paths[0])
        counts = {split: len(selected_base_records(config, split)[0]) for split in ('train', 'val')}
        if counts != {'train': 2697, 'val': 202}:
            raise RuntimeError(f'unexpected full base manifest counts: {counts}')
        cache_bytes = sum(counts.values()) * 8 * 2 * 1800 * 8
        free = shutil.disk_usage(WORKSPACE).free
        if free < cache_bytes + 2 * 1024**3:
            raise RuntimeError('insufficient data-disk space for IAT cache and training headroom')
        state['preflight'] = dict(sample_counts=counts, expected_iat_array_bytes=cache_bytes,
                                 free_disk_bytes=free, gpu=torch.cuda.get_device_name(0),
                                 torch=torch.__version__, cuda=torch.version.cuda,
                                 cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])
        state['stage'] = 'fit-bins'
        command(['fit-bins', '--config', str(paths[0])], 'fit-bins.log')
        state['stage'] = 'prepare'
        command(['prepare', '--config', str(paths[0])], 'prepare.log')
        for path in paths:
            state['stage'] = f'train:{path.stem}'
            trained = command(['train', '--config', str(path), '--device', 'cuda'],
                              path.stem + '-train.log')
            root = Path(trained['run_dir'])
            state['stage'] = f'evaluate:{path.stem}'
            evaluated = command(['evaluate', '--config', str(path), '--device', 'cuda',
                                 '--checkpoint', str(root / 'best.pt'), '--split', 'val'],
                                path.stem + '-val.log')
            result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
            state['configurations'].append(dict(name=path.stem, run_dir=str(root),
                                                 result=result, validation=evaluated))
            publish()
        state.update(status='completed', stage='completed', ended_utc=now())
        publish()
        return 0
    except Exception as exc:
        state.update(status='failed', error=str(exc), ended_utc=now())
        publish()
        raise


if __name__ == '__main__':
    sys.exit(main())
