"""Audit saved trial evidence without starting training or reading test data."""
import json
from pathlib import Path
import statistics
import sys

import optim_agent
import torch

from agentprint.iat_tuning import committed_result
from agentprint.v2_protocol import learning_rate, stop_status


def read(path):
    return json.loads(path.read_text())


def main():
    root = Path(sys.argv[1])
    protocol = read(root / 'protocol.json')
    study = optim_agent.create_study(direction='maximize', storage=root / 'study.json', seed=42)
    proposals = sorted(root.glob('*/proposal.json'), key=lambda p: read(p)['order'])
    assert len(proposals) == len(study.trials) <= 17
    rows = []
    for path, trial in zip(proposals, study.trials):
        proposal, result = read(path), read(path.parent / 'result.json')
        assert trial.params == dict(proposal['params'], seed=proposal['seed'])
        assert trial.state == result['state']
        assert result['identity_id'] == proposal['identity_id']
        assert result['uses_test'] is False
        if result['state'] == 'complete':
            assert trial.value == result['objective'] == result['best_val_macro_f1']
        elif result['state'] == 'pruned':
            assert trial.value == result['latest_val_macro_f1']
        else:
            assert trial.value is None
        item = dict(name=path.parent.name, phase=proposal['phase'], seed=proposal['seed'],
                    params=proposal['params'], state=result['state'],
                    objective=result.get('objective'), epochs=result.get('epochs'),
                    best_epoch=result.get('best_epoch'), run_dir=result.get('run_dir'),
                    finished_unix=result.get('finished_unix'))
        if result.get('epochs', 0):
            run = Path(result['run_dir'])
            saved = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
            committed = committed_result(run, proposal['identity']['experiment'])
            history = saved['rows']
            assert len(history) == result['epochs']
            assert committed['best_val_macro_f1'] == result['best_val_macro_f1']
            config = saved['config']
            for index, row in enumerate(history):
                assert row['epoch'] == index
                assert row['learning_rate'] == learning_rate(index, config['training'])
                assert row['global_step'] == (index + 1) * 14
                if index < len(history) - 1:
                    assert not stop_status(history[:index + 1], 'tuning', 200)['stop']
            status = stop_status(history, 'tuning', 200)
            item.update(converged=status['converged'], stop_reason=status['reason'])
            if result['state'] == 'complete':
                assert status['stop']
                evaluation = result['evaluation']
                assert evaluation['split'] == 'val' and evaluation['sample_count'] == 202
                assert evaluation['macro_f1'] == result['objective']
                item.update(accuracy=evaluation['accuracy'], checkpoint=result['best_checkpoint'])
        rows.append(item)
    complete = [r for r in rows if r['state'] == 'complete']
    seed42 = [r for r in complete if r['seed'] == 42]
    finalists = []
    for candidate in sorted([r for r in seed42 if r['phase'] == 'search'],
                            key=lambda r: r['objective'], reverse=True)[:2]:
        matching = [r for r in complete if r['params'] == candidate['params']]
        finalists.append(dict(params=candidate['params'], seeds=[r['seed'] for r in matching],
                              mean=statistics.mean(r['objective'] for r in matching),
                              three_seed_complete={r['seed'] for r in matching} == {42, 43, 44}))
    ended = max(r.get('finished_unix') or protocol['started_unix'] for r in rows)
    report = dict(trials=rows, trial_count=len(rows),
        counts={s: sum(r['state'] == s for r in rows) for s in ('complete', 'pruned', 'failed')},
        baseline=rows[0], best_seed42=max(seed42, key=lambda r: r['objective']),
        finalists=finalists, replicated_comparison_complete=all(f['three_seed_complete'] for f in finalists),
        elapsed_to_last_result=ended - protocol['started_unix'],
        deadline_unix=protocol['deadline_unix'], uses_test=False)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
