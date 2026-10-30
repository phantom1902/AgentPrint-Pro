#!/usr/bin/env python3
"""Serial continuation of the four recorded baselines, using train/val only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    baseline, root = args.baseline_root.resolve(), args.output_root.resolve()
    if root.is_relative_to(baseline) or baseline.is_relative_to(root):
        raise ValueError('experiment roots must be independent')
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return execute(baseline, root, args.device)


def execute(baseline, root, device):
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    from agentprint.cache import write_json
    from agentprint.convergence import ConvergencePolicy, assess
    from dataclasses import asdict
    original = json.loads((baseline/'execution.json').read_text())
    expected = {(t,x) for t in ('scenario','primitive') for x in ('raw','log1p')}
    if original['status'] != 'passed' or {(r['task'],r['transform']) for r in original['runs']} != expected:
        raise ValueError('requires four completed baseline runs')
    record = {'status':'running','baseline_root':str(baseline),'output_root':str(root),
              'started_utc':datetime.now(timezone.utc).isoformat(),'uses_test':False,
              'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
              'policy':asdict(ConvergencePolicy()),'runs':[],'selections':{}}
    def save(): write_json(root/'execution.json',record)
    save()
    env = dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',
               CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(root/'tmp'))
    (root/'tmp').mkdir(exist_ok=True); (root/'logs').mkdir(exist_ok=True)
    try:
        for task in ('scenario','primitive'):
            for transform in ('raw','log1p'):
                source = next(r for r in original['runs'] if (r['task'],r['transform'])==(task,transform))
                parent = Path(source['run_dir'])
                if not parent.is_relative_to(baseline): raise ValueError('parent outside baseline root')
                entry = {'task':task,'transform':transform,'parent_run':str(parent),'status':'running'}
                record['runs'].append(entry); save()
                argv=[sys.executable,'-m','agentprint.convergence','--parent-run',str(parent),
                      '--output-root',str(root),'--device',device]
                entry['argv']=argv
                start=time.monotonic()
                with (root/'logs'/(task+'-'+transform+'.log')).open('a') as log:
                    result=subprocess.run(argv,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
                entry.update(returncode=result.returncode,elapsed_seconds=time.monotonic()-start)
                if result.returncode: raise RuntimeError(task+'/'+transform+' continuation failed; inspect logs')
                run=root/parent.name
                state=json.loads((run/'convergence.json').read_text())
                lineage=json.loads((run/'lineage.json').read_text())
                rows=[json.loads(s) for s in (run/'metrics.jsonl').read_text().splitlines() if s.strip()]
                recomputed=assess(rows[30:],lineage['baseline_best_macro_f1'],ConvergencePolicy())
                if state != recomputed or not state['stop']: raise ValueError('stop assessment mismatch')
                validation=json.loads((run/'validation-best/summary.json').read_text())
                if validation['macro_f1'] != max(r['val_macro_f1'] for r in rows):
                    raise ValueError('best validation result mismatch')
                entry.update(status='validation_plateau' if state['converged'] else 'epoch_cap_reached',
                             run_dir=str(run),convergence=state,validation=validation)
                save()
            candidates=[r for r in record['runs'] if r['task']==task]
            chosen=max(candidates,key=lambda r:r['validation']['macro_f1'])
            record['selections'][task]={'selection_split':'val','selected_transform':chosen['transform'],
                'validation_macro_f1':chosen['validation']['macro_f1'],'checkpoint':str(Path(chosen['run_dir'])/'best.pt'),
                'all_candidates_reached_plateau':all(r['convergence']['converged'] for r in candidates)}
            save()
        record['status']='completed' if all(r['convergence']['converged'] for r in record['runs']) else 'completed_with_epoch_caps'
    except Exception as exc:
        record.update(status='failed',error=repr(exc))
        raise
    finally:
        record['completed_utc']=datetime.now(timezone.utc).isoformat(); save()
    print(json.dumps({'status':record['status'],'output_root':str(root)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
