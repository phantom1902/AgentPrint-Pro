"""Independent read-only audit and final summary of explicit ask/tell trials."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys
import time

import optim_agent as oa
import torch

REPO = Path('/root/autodl-tmp/ant-workspace/src/AgentPrint-Pro')
sys.path.insert(0, str(REPO))
from agentprint.cache import write_json
from agentprint.tuning import build_model, stop_status
from agentprint.training import evaluate_model, loader, setup_execution

ROOT = REPO/'.optim-agent-runs/scenario-raw-20261030'
protocol = json.loads((ROOT/'protocol.json').read_text())
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
        assert not any(stop_status(rows[:n])['stop'] for n in range(1, len(rows)))
    saved = torch.load(directory/'last.pt', map_location='cpu', weights_only=False)
    assert saved['identity'] == result['identity']
    steps = math.ceil(2697/params['batch_size'])*len(rows)
    assert saved['global_step'] == steps
    assert all(float(v['step']) == steps for v in saved['optimizer']['state'].values())
    best = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
    assert best['epoch'] == best_index and best['best_f1'] == result['best_val_macro_f1']
    point = dict(params, seed=proposal['seed'])
    matches = [t for t in study.trials if t.params == point]
    assert len(matches) == 1
    assert matches[0].state == result['state']
    if result['state'] == 'complete':
        assert matches[0].value == result['best_val_macro_f1']
    record = dict(trial=directory.name, seed=proposal['seed'], phase=proposal['phase'],
                  params=params, macro_f1=result['best_val_macro_f1'],
                  accuracy=best['summary']['accuracy'], best_epoch=best_index+1,
                  epochs=len(rows), parameters=result['parameters'],
                  state=result['state'], stop_reason=result['reason'],
                  checkpoint=str(directory/'best.pt'),
                  checkpoint_sha256=hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest())
    records.append(record)
    del best, saved

original = Path('/root/autodl-tmp/ant-workspace/runs/convergence-four-e057ace-20261030/scenario-raw-seed42-20261030T041108Z-dc362779/metrics.jsonl')
old = [json.loads(s) for s in original.read_text().splitlines()]
new = [json.loads(s) for s in (ROOT/'trial-00/metrics.jsonl').read_text().splitlines()]
assert len(old) == len(new) == 90
fields = ['epoch', 'global_step', 'learning_rate', 'train_loss', 'val_loss',
          'val_accuracy', 'val_macro_f1', 'train_samples', 'val_samples']
assert all(x[k] == y[k] for x, y in zip(old, new) for k in fields)

groups = {}
for r in records:
    if r['state'] != 'complete':
        continue
    groups.setdefault(json.dumps(r['params'], sort_keys=True), []).append(r)
finalists = []
for key, candidates in groups.items():
    if {r['seed'] for r in candidates} != {42, 43, 44}:
        continue
    candidates.sort(key=lambda r: r['seed'])
    scores = [r['macro_f1'] for r in candidates]
    finalist = dict(params=json.loads(key), mean_macro_f1=mean(scores),
                    sample_std_macro_f1=stdev(scores),
                    mean_accuracy=mean(r['accuracy'] for r in candidates),
                    trials=[r['trial'] for r in candidates], scores=scores)
    finalists.append(finalist)

finalists.sort(key=lambda r: -r['mean_macro_f1'])
seed42 = sorted([r for r in records if r['seed'] == 42 and r['state'] == 'complete'],
                key=lambda r: -r['macro_f1'])
if len(finalists) == 2:
    assert {json.dumps(r['params'], sort_keys=True) for r in seed42[:2]} == {json.dumps(r['params'], sort_keys=True) for r in finalists}
setup_execution(42)
for r in records:
    if r['phase'] == 'repeat' or r['params'] in [f['params'] for f in finalists]:
        config = dict(task='scenario', transform='raw',
                      paths={'cache_root': protocol['cache_root']},
                      training={'batch_size': r['params']['batch_size']})
        samples, batches = loader(config, 'val')
        assert samples.metadata['manifest_id'] == protocol['manifests']['val']
        model = build_model(r['params']).to('cuda:0')
        saved = torch.load(r['checkpoint'], map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        summary, _, _ = evaluate_model(model, batches, 'cuda:0', 'scenario')
        assert summary['macro_f1'] == r['macro_f1']
        assert summary['accuracy'] == r['accuracy']
        del saved, model

selected = finalists[0] if len(finalists) == 2 else None
selected_seed42 = next(r for r in records if r['params'] == selected['params'] and r['seed'] == 42) if selected else None
replication = []
for candidate in seed42[:2]:
    completed = [r for r in records if r['params'] == candidate['params'] and r['state'] == 'complete']
    replication.append(dict(params=candidate['params'], completed_seeds=[r['seed'] for r in completed],
                            scores=[r['macro_f1'] for r in completed],
                            completed_seed_mean=mean(r['macro_f1'] for r in completed)))
report = dict(passed=True, uses_test=False, direction='maximize',
              objective='three_seed_mean_best_validation_macro_f1',
              baseline_macro_f1=records[0]['macro_f1'],
              seed42_best=seed42[0], finalists=finalists,
              selection_status='complete_three_seed_comparison' if selected else 'incomplete_finalist_replication',
              replication=replication,
              selected_params=selected['params'] if selected else None,
              selected_seed42_checkpoint=selected_seed42['checkpoint'] if selected else None,
              selected_seed42_macro_f1=selected_seed42['macro_f1'] if selected else None,
              trials=records, trial_count=len(records),
              failed=sum(t.state == 'failed' for t in study.trials),
              pruned=sum(t.state == 'pruned' for t in study.trials),
              baseline_parity=dict(epochs=90, fields_per_epoch=9, passed=True),
              search_wall_seconds=max((ROOT/r['trial']/'result.json').stat().st_mtime for r in records)-protocol['started_unix'],
              phase_counts={phase: sum(r['phase'] == phase for r in records) for phase in ('baseline','search','repeat')},
              audit_unix=time.time(), study_protocol=protocol)
write_json(ROOT/'audit.json', report)
if selected:
    write_json(ROOT/'best-params.json', selected['params'])
print(json.dumps({k: v for k, v in report.items() if k != 'trials'}, indent=2))
