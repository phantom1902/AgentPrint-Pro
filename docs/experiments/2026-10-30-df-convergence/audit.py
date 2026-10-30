from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support

REPO=Path('/root/autodl-tmp/ant-workspace/src/AgentPrint-Pro')
ROOT=Path('/root/autodl-tmp/ant-workspace/runs/df-scenario-convergence-fresh-20261030')
PARENT=Path('/root/autodl-tmp/ant-workspace/runs/df-scenario-default-20261030')
sys.path.insert(0,str(REPO))
from tools.df_convergence import evaluate,digest
from run_ant_wf_scenario_smoke import build_model_specs,instantiate_model,tensors_for_split,make_loader
from agentprint.convergence import ConvergencePolicy,assess

result=json.loads((ROOT/'result.json').read_text())
identity=json.loads((ROOT/'identity.json').read_text())
assert identity['initialization']=='fresh_deterministic_run'
assert not result['baseline_parity_passed']
assert result['identity']==identity
policy=ConvergencePolicy(**identity['policy'])
rows=[json.loads(s) for s in (ROOT/'metrics.jsonl').read_text().splitlines()]
assert len(rows)==result['epochs'] and [r['epoch'] for r in rows]==list(range(len(rows)))
assert all(r['learning_rate']==.002 for r in rows)
baseline_f1=max(r['val_macro_f1'] for r in rows[:30])
assert assess(rows[30:],baseline_f1,policy)==result['stopping']
assert result['stopping']['stop']
assert not any(assess(rows[30:n],baseline_f1,policy)['stop'] for n in range(30,len(rows)))
assert len(rows)>=60
best_index=max(range(len(rows)),key=lambda i:(rows[i]['val_accuracy'],i))
assert best_index+1==result['best_epoch']
best=torch.load(ROOT/'best.pt',map_location='cpu',weights_only=False)
last=torch.load(ROOT/'last.pt',map_location='cpu',weights_only=False)
assert best['epoch']==best_index and last['epoch']==len(rows)-1
assert last['rows']==rows
assert best['identity']==last['identity']==identity
assert last['best_accuracy']==rows[best_index]['val_accuracy']
steps=(2703//128)*len(rows)
assert all(float(v['step'])==steps for v in last['optimizer']['state'].values())
assert digest(PARENT/'WFlib-DF/native_metrics.json')==identity['parent_metrics_sha256']
assert digest(PARENT/'WFlib-DF/checkpoint-best.pt')==identity['parent_checkpoint_sha256']
raw=PARENT/'wflib_raw'
for name,expected in identity['input_hashes'].items():assert digest(raw/name)==expected
assert digest(raw/'test.npz')==result['test_cache_sha256']
mapping=json.loads((raw/'label_mapping.json').read_text())
labels=[mapping['id_to_label'][str(i)] for i in range(5)]
spec=build_model_specs(REPO/'Benchmark/all_default_configs')['DF']
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True)
model=instantiate_model(spec,5).to('cuda:0')
model.load_state_dict(best['model'])
metrics={}
for split,field in [('val','validation'),('test','test')]:
    X,y,_=tensors_for_split(raw,split,spec,1)
    summary,soft=evaluate(model,make_loader(X,y,128,False),'cuda:0')
    for key,value in result[field].items():assert abs(summary[key]-value)<1e-12,(key,summary[key],value)
    predictions=soft.argmax(1)
    if split=='test':
        assert np.allclose(soft,np.load(ROOT/'test_soft_predictions.npy'),atol=1e-7,rtol=1e-6)
        assert np.array_equal(y.numpy(),np.load(ROOT/'test_labels.npy'))
    p,r,f,s=precision_recall_fscore_support(y.numpy(),predictions,labels=np.arange(5),zero_division=0)
    metrics[split]=dict(**summary,confusion_matrix=confusion_matrix(y.numpy(),predictions,labels=np.arange(5)).tolist(),
        per_class=[dict(label=labels[i],precision=float(p[i]),recall=float(r[i]),f1=float(f[i]),support=int(s[i])) for i in range(5)])
prior=json.loads((PARENT/'audit.json').read_text())
report=dict(passed=True,audit_unix=time.time(),experiment='independent fresh deterministic DF run',
    baseline_replay_failed=True,baseline_parity_passed=False,original_parent_preserved=True,
    epochs=len(rows),best_epoch=result['best_epoch'],stopping=result['stopping'],metrics=metrics,
    highest_val_macro_f1=result['highest_val_macro_f1'],
    checkpoint_metric=result['checkpoint_metric'],baseline30_current_run=rows[29],
    baseline30_current_best_accuracy=max(r['val_accuracy'] for r in rows[:30]),
    baseline30_current_best_macro_f1=baseline_f1,
    baseline30_original_metrics=prior['metrics'],
    optimizer='Adamax',learning_rate=.002,batch_size=128,seed=42,optimizer_steps=steps,
    parameters=result['parameters'],samples=dict(train=result['train_samples'],val=result['val_samples'],test=result['test_samples']),
    elapsed_seconds=result['elapsed_seconds'],
    hashes={name:digest(ROOT/name) for name in ('best.pt','last.pt','metrics.jsonl','result.json')},
    environment=dict(torch=torch.__version__,numpy=np.__version__,python=sys.version,gpu=torch.cuda.get_device_name(0),
        precision='FP32',deterministic_algorithms=True,TF32=False,cudnn_benchmark=False,CUBLAS_WORKSPACE_CONFIG=':4096:8'),
    identity=identity)
(ROOT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('identity','baseline30_original_metrics')},indent=2))
