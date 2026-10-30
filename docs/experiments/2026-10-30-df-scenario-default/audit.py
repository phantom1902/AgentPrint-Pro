"""Independently audit the unchanged DF default baseline and its input coverage."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,precision_recall_fscore_support

REPO=Path('/root/autodl-tmp/ant-workspace/src/AgentPrint-Pro')
OUT=Path('/root/autodl-tmp/ant-workspace/runs/df-scenario-default-20261030')
DEV=Path('/root/autodl-tmp/ant-workspace/df-development-20261030')
sys.path[:0]=[str(REPO),str(REPO/'Benchmark'),str(REPO/'Benchmark/scripts')]
from ant_pcap_native import discover_episodes
from run_ant_wf_scenario_smoke import build_model_specs,instantiate_model,tensors_for_split,classification_metrics
import suite

summary=json.loads((OUT/'summary.json').read_text())
assert len(summary['results'])==1 and summary['results'][0]['status']=='completed'
run=OUT/'WFlib-DF'
native=json.loads((run/'native_metrics.json').read_text())
metrics=json.loads((run/'evaluation_metrics.json').read_text())
assert native['epochs']==30 and native['batch_size']==128 and native['learning_rate']==.002
assert native['optimizer']=='Adamax' and native['checkpoint_metric']=='accuracy'
assert len(native['loss'])==len(native['val_accuracy'])==len(native['val_macro_f1'])==30
assert all(math.isfinite(v) for k in ('loss','val_accuracy','val_macro_f1','val_balanced_accuracy') for v in native[k])
best_index=max(range(30),key=lambda i:(native['val_accuracy'][i],i))
assert native['best_epoch']==best_index+1
assert native['best_val_accuracy']==native['val_accuracy'][best_index]
assert native['best_val_macro_f1']==native['val_macro_f1'][best_index]
manifest=json.loads((OUT/'wflib_raw/raw_manifest.json').read_text())
mapping=json.loads((OUT/'wflib_raw/label_mapping.json').read_text())
names=[mapping['id_to_label'][str(i)] for i in range(5)]
assert set(names)=={g['id'] for g in suite.read(REPO/'configs/scenario.json')['groups']}
eps=discover_episodes(Path('/root/autodl-tmp/ant-workspace/data/raw/ANT'),('train','val','test'))
assert len(eps)==3114
coverage={};input_statistics={};tensors={};hashes={};all_keys=set()
for split in ('train','val','test'):
    path=OUT/'wflib_raw'/f'{split}.npz'
    saved=np.load(path)
    X,y=saved['X'],saved['y']
    rows=manifest['splits'][split]['sources']
    assert X.shape==(len(rows),5000) and y.shape==(len(rows),)
    assert np.isfinite(X).all() and np.isfinite(y).all()
    observed={str(Path(r['pcap']).resolve()) for r in rows}
    expected={str(ep.pcap_path.resolve()) for ep in eps if ep.split==split}
    assert len(observed)==len(rows) and observed<=expected
    for row,label in zip(rows,y):
        assert row['label']==names[int(label)]
        assert suite.task_label(dict(family=row['scenario'],episode=row['episode']),'scenario')==row['label']
        key=(row['engine'],row['episode'])
        assert key not in all_keys
        all_keys.add(key)
    coverage[split]=dict(expected=len(expected),evaluated=len(rows),missing=sorted(expected-observed),
                         support=dict(Counter(names[int(v)] for v in y)))
    directions=np.sign(X)
    lengths=np.count_nonzero(directions,axis=1)
    input_statistics[split]=dict(shape=list(X.shape),nonzero_direction_slots=dict(
        minimum=int(lengths.min()),median=float(np.median(lengths)),maximum=int(lengths.max())),
        all_zero_rows=int((lengths==0).sum()),
        note='Zero slots include the first packet at relative time zero, equal-time packets, and padding; nonzero count is not packet count')
    with path.open('rb') as handle:hashes[str(path)]=hashlib.file_digest(handle,'sha256').hexdigest()

spec=build_model_specs(REPO/'Benchmark/all_default_configs')['DF']
model=instantiate_model(spec,5).to('cuda:0')
checkpoint=torch.load(run/'checkpoint-best.pt',map_location='cuda:0',weights_only=False)
assert checkpoint['epoch']==native['best_epoch'] and checkpoint['checkpoint_metric']=='accuracy'
model.load_state_dict(checkpoint['model_state_dict']);model.eval()
recomputed={}
for split in ('val','test'):
    X,y,_=tensors_for_split(OUT/'wflib_raw',split,spec,4)
    logits=[]
    with torch.no_grad():
        for batch in X.split(128):logits.append(torch.softmax(model(batch.to('cuda:0')),dim=1).cpu().numpy())
    soft=np.concatenate(logits)
    pred=soft.argmax(1);truth=y.numpy()
    stats=classification_metrics(truth.tolist(),pred.tolist(),5)
    precision,recall,f1,support=precision_recall_fscore_support(truth,pred,labels=np.arange(5),zero_division=0)
    recomputed[split]=dict(**stats,confusion_matrix=confusion_matrix(truth,pred,labels=np.arange(5)).tolist(),
         per_class=[dict(label=names[i],precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(support[i])) for i in range(5)])
    if split=='val':
        for k,v in stats.items():assert abs(v-native['best_val_'+k])<1e-12
    else:
        original=np.load(run/'soft_predictions.npy')
        assert np.allclose(original,soft,atol=1e-7,rtol=1e-6)
        assert np.array_equal(original.argmax(1),pred)
        for k,v in stats.items():assert abs(v-metrics[k])<1e-12
        assert metrics['evaluated_sample_count']==len(truth)
    if split=='val':np.save(OUT/'validation_soft_predictions.npy',soft)

for path in (run/'checkpoint-best.pt',REPO/'Benchmark/2024_CCS_WFlib_Early_Stage_WF/WFlib/models/DF.py',
             REPO/'Benchmark/scripts/run_ant_wf_scenario_smoke.py',REPO/'Benchmark/2024_CCS_WFlib_Early_Stage_WF/WFlib/tools/data_processor.py'):
    with path.open('rb') as handle:hashes[str(path)]=hashlib.file_digest(handle,'sha256').hexdigest()
launch=json.loads((DEV/'launch.json').read_text())
assert not subprocess.check_output(['git','diff',launch['source_commit'],'HEAD','--','Benchmark','suite.py','engines.py'],cwd=REPO)
report=dict(passed=True,audit_unix=time.time(),spec=asdict(spec),seed=42,parameters=sum(p.numel() for p in model.parameters()),
    coverage=coverage,input_statistics=input_statistics,selected_checkpoint_epoch=native['best_epoch'],
    checkpoint_metric='validation accuracy; latest epoch wins ties',metrics=recomputed,
    highest_val_macro_f1_over_all_epochs=max(native['val_macro_f1']),
    highest_val_macro_f1_epoch=native['val_macro_f1'].index(max(native['val_macro_f1']))+1,
    train_drop_last=True,training_samples_per_epoch=(coverage['train']['evaluated']//128)*128,
    environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0)),
    execution=launch,hashes=hashes,wall_seconds=(OUT/'summary.json').stat().st_mtime-launch['started_unix'],
    native_training_and_evaluation_seconds=summary['results'][0]['elapsed_seconds'])
(OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
