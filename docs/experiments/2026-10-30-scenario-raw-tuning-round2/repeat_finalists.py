"""Wait for last search, then replicate its top-two completed new points."""
import json
import os
from pathlib import Path
import subprocess
import time

REPO=Path('/root/autodl-tmp/ant-workspace/src/AgentPrint-Pro')
ROOT=REPO/'.optim-agent-runs/scenario-raw-20261030-round2'
DEV=Path('/root/autodl-tmp/ant-workspace/tuning-development-20261030-round2')
PY='/root/autodl-tmp/ant-workspace/envs/optim-agent/bin/python'
protocol=json.loads((ROOT/'protocol.json').read_text())
while not (ROOT/'trial-07/result.json').exists():
    if (ROOT/'trial-07/failure.json').exists():raise RuntimeError('last search failed')
    if time.time()>protocol['deadline_unix']:raise RuntimeError('deadline while awaiting search')
    time.sleep(2)
candidates=[]
for directory in ROOT.glob('trial-*'):
    proposal=json.loads((directory/'proposal.json').read_text())
    if proposal['phase']!='search':continue
    result=json.loads((directory/'result.json').read_text())
    if result['state']=='complete':
        candidates.append(dict(trial=directory.name,params=proposal['params'],score=result['best_val_macro_f1']))
candidates.sort(key=lambda row:-row['score'])
assert len(candidates)>=2
finalists=candidates[:2]
(ROOT/'finalist-plan.json').write_text(json.dumps(dict(selected_by='top_two_completed_new_seed42_macro_f1',
    finalists=finalists,order='leading candidate seeds43/44 then runnerup seeds43/44',
    incumbent='scenario-raw-20261030/trial-07,09,10',created_unix=time.time()),indent=2))
print('FINALISTS',json.dumps(finalists),flush=True)
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8')
for index,(candidate,seed) in enumerate([(finalists[0],43),(finalists[0],44),(finalists[1],43),(finalists[1],44)],start=8):
    if time.time()>=protocol['deadline_unix']:
        print('DEADLINE_NO_NEW_TRIAL',index,flush=True)
        break
    name=f'trial-{index:02d}'
    params=ROOT/'params'/f'{index:02d}.json'
    params.write_text(json.dumps(candidate['params'],indent=2))
    command=[PY,'-m','agentprint.tuning','--study-dir',str(ROOT),'--trial-name',name,
             '--params',str(params),'--rationale',f"replicate-round2-finalist-{candidate['trial']}-seed{seed}",
             '--phase','repeat','--seed',str(seed),'--cache-root',protocol['cache_root']]
    print('START',name,'seed',seed,'source',candidate['trial'],'remaining',round(protocol['deadline_unix']-time.time()),flush=True)
    with (DEV/(name+'.log')).open('w') as log:
        result=subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'{name} failed with {result.returncode}')
    r=json.loads((ROOT/name/'result.json').read_text())
    print('DONE',name,r['state'],r['epochs'],r['best_val_macro_f1'],flush=True)
print('REPLICATION_SCHEDULER_FINISHED',flush=True)
