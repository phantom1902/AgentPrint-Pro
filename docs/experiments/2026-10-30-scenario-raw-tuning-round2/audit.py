"""Audit round-2 artifacts and compare replicated challengers with incumbent."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys
import time

import optim_agent as oa
import torch

REPO = Path('/root/autodl-tmp/ant-workspace/src/AgentPrint-Pro')
sys.path.insert(0, str(REPO))
from agentprint.cache import write_json
from agentprint.tuning import build_model, stop_status
from agentprint.training import evaluate_model, loader, setup_execution

ROOT = REPO/'.optim-agent-runs/scenario-raw-20261030-round2'
OLD = REPO/'.optim-agent-runs/scenario-raw-20261030'
protocol = json.loads((ROOT/'protocol.json').read_text())
old_audit = json.loads((OLD/'audit.json').read_text())
assert old_audit['passed']
assert protocol['manifests'] == old_audit['study_protocol']['manifests']
assert not subprocess.check_output(['git','diff','167118c','HEAD','--','agentprint'], cwd=REPO)
study = oa.create_study(direction='maximize', storage=ROOT/'study.json', seed=42)
assert len(study.trials) <= 17
records = []
for directory in sorted(ROOT.glob('trial-*')):
    proposal = json.loads((directory/'proposal.json').read_text())
    result = json.loads((directory/'result.json').read_text())
    rows = [json.loads(s) for s in (directory/'metrics.jsonl').read_text().splitlines()]
    params = proposal['params']
    assert result['identity']['params'] == params
    assert result['identity']['seed'] == proposal['seed']
    assert not result['uses_test']
    assert rows and len(rows) == result['epochs']
    assert [r['epoch'] for r in rows] == list(range(len(rows)))
    for row in rows:
        assert all(math.isfinite(v) for v in row.values())
        expected_lr = max(1e-6, params['learning_rate']*params['decay_per_30']**(row['epoch']/30))
        assert row['learning_rate'] == expected_lr
        assert row['train_samples'] == 2697 and row['val_samples'] == 202
    best_index = max(range(len(rows)), key=lambda i: rows[i]['val_macro_f1'])
    assert result['best_val_macro_f1'] == rows[best_index]['val_macro_f1']
    assert result['best_epoch_one_based'] == best_index+1
    if result['state'] == 'complete':
        assert stop_status(rows)['stop']
        assert not any(stop_status(rows[:n])['stop'] for n in range(1,len(rows)))
    saved = torch.load(directory/'last.pt', map_location='cpu', weights_only=False)
    assert saved['identity'] == result['identity']
    steps = math.ceil(2697/params['batch_size'])*len(rows)
    assert saved['global_step'] == steps
    assert all(float(v['step']) == steps for v in saved['optimizer']['state'].values())
    best = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
    assert best['epoch'] == best_index and best['best_f1'] == result['best_val_macro_f1']
    matches = [t for t in study.trials if t.params == dict(params,seed=proposal['seed'])]
    assert len(matches) == 1 and matches[0].state == result['state']
    if result['state'] == 'complete':assert matches[0].value == result['best_val_macro_f1']
    environment = json.loads((directory/'environment.json').read_text())
    previous = json.loads((OLD/'trial-00/environment.json').read_text())
    for key in ('packages','python','platform','precision','device'):
        assert environment[key] == previous[key], key
    record = dict(trial=directory.name, seed=proposal['seed'], phase=proposal['phase'],
                  params=params, macro_f1=result['best_val_macro_f1'], accuracy=best['summary']['accuracy'],
                  best_epoch=best_index+1, epochs=len(rows), parameters=result['parameters'],
                  state=result['state'], stop_reason=result['reason'],
                  checkpoint=str(directory/'best.pt'),
                  checkpoint_sha256=hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest())
    records.append(record)
    del best,saved

fields = ['epoch','global_step','learning_rate','train_loss','val_loss','val_accuracy',
          'val_macro_f1','train_samples','val_samples']
x = [json.loads(s) for s in (OLD/'trial-00/metrics.jsonl').read_text().splitlines()]
y = [json.loads(s) for s in (ROOT/'trial-00/metrics.jsonl').read_text().splitlines()]
assert len(x) == len(y) == 90
assert all(a[k] == b[k] for a,b in zip(x,y) for k in fields)

groups = {}
for row in records:
    if row['phase'] != 'baseline' and row['state'] == 'complete':
        groups.setdefault(json.dumps(row['params'],sort_keys=True), []).append(row)
finalists = []
for key, candidates in groups.items():
    if {r['seed'] for r in candidates} != {42,43,44}:continue
    candidates.sort(key=lambda r:r['seed'])
    scores = [r['macro_f1'] for r in candidates]
    finalists.append(dict(params=json.loads(key),mean_macro_f1=mean(scores),
                          sample_std_macro_f1=stdev(scores),mean_accuracy=mean(r['accuracy'] for r in candidates),
                          trials=[r['trial'] for r in candidates],scores=scores,origin='round2'))
seed42 = sorted([r for r in records if r['phase']=='search' and r['state']=='complete'],key=lambda r:-r['macro_f1'])
if len(finalists) == 2:
    assert {json.dumps(r['params'],sort_keys=True) for r in seed42[:2]} == {json.dumps(r['params'],sort_keys=True) for r in finalists}
incumbent = dict(old_audit['finalists'][0],origin='round1')
assert incumbent['params'] == old_audit['selected_params']
for trial_name, score in zip(incumbent['trials'], incumbent['scores']):
    record = next(r for r in old_audit['trials'] if r['trial']==trial_name)
    assert record['macro_f1'] == score
    assert hashlib.sha256(Path(record['checkpoint']).read_bytes()).hexdigest() == record['checkpoint_sha256']

setup_execution(42)
reloads = [r for r in records if r['phase']=='repeat' or r['params'] in [f['params'] for f in finalists]]
reloads.extend(r for r in old_audit['trials'] if r['trial'] in incumbent['trials'])
for row in reloads:
    config = dict(task='scenario',transform='raw', paths={'cache_root':protocol['cache_root']},
                  training={'batch_size':row['params']['batch_size']})
    samples,batches = loader(config,'val')
    assert samples.metadata['manifest_id'] == protocol['manifests']['val']
    model = build_model(row['params']).to('cuda:0')
    saved = torch.load(row['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'])
    summary,_,_ = evaluate_model(model,batches,'cuda:0','scenario')
    assert summary['macro_f1'] == row['macro_f1'] and summary['accuracy'] == row['accuracy']
    del saved,model

ranked = sorted([incumbent,*finalists],key=lambda f:-f['mean_macro_f1'])
complete_comparison = len(finalists) == 2
selected = ranked[0] if complete_comparison else incumbent
source_records = records if selected['origin']=='round2' else old_audit['trials']
selected_run = next(r for r in source_records if r['params']==selected['params'] and r['seed']==42)
report = dict(passed=True,uses_test=False,objective='three_seed_mean_best_validation_macro_f1',
              direction='maximize',selection_status='complete_three_seed_comparison' if complete_comparison else 'incomplete_replication_incumbent_retained',
              selected_params=selected['params'],selected_origin=selected['origin'],
              selected_mean_macro_f1=selected['mean_macro_f1'],
              selected_sample_std_macro_f1=selected['sample_std_macro_f1'],
              selected_seed42_macro_f1=selected_run['macro_f1'],selected_seed42_checkpoint=selected_run['checkpoint'],
              incumbent=incumbent,finalists=finalists,ranked=ranked,
              seed42_best=seed42[0] if seed42 else None,trials=records,trial_count=len(records),
              failed=sum(t.state=='failed' for t in study.trials),pruned=sum(t.state=='pruned' for t in study.trials),
              baseline_parity=dict(epochs=90,fields_per_epoch=9,passed=True),
              source_environment_manifest_parity=True,reloaded_checkpoints=len(reloads),
              search_wall_seconds=max((ROOT/r['trial']/'result.json').stat().st_mtime for r in records)-protocol['started_unix'],
              phase_counts={p:sum(r['phase']==p for r in records) for p in ('baseline','search','repeat')},
              study_protocol=protocol,audit_unix=time.time())
write_json(ROOT/'audit.json',report)
write_json(ROOT/'best-params.json',selected['params'])
print(json.dumps({k:v for k,v in report.items() if k!='trials'},indent=2))
