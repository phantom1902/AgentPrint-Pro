"""Replay and verify a DF baseline, then extend it to a validation plateau."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from suite import ROOT
sys.path[:0]=[str(ROOT/'Benchmark/scripts'),str(ROOT/'Benchmark')]
from run_ant_wf_scenario_smoke import (build_model_specs,classification_metrics,
    instantiate_model,make_loader,make_optimizer,set_seed,tensors_for_split)
from agentprint.convergence import ConvergencePolicy,assess
from agentprint.recovery import capture_rng,restore_rng,save_checkpoint,save_text


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def evaluate(model,batches,device):
    model.eval()
    truth=[];predicted=[];probabilities=[];total=0.0
    with torch.no_grad():
        for values,labels in batches:
            scores=model(values.to(device))
            loss=torch.nn.functional.cross_entropy(scores,labels.to(device))
            if not torch.isfinite(loss):raise RuntimeError('non-finite validation/test loss')
            total+=float(loss)*len(labels)
            truth.extend(labels.tolist())
            predicted.extend(scores.argmax(1).cpu().tolist())
            probabilities.append(scores.softmax(1).cpu().numpy())
    if not truth:raise ValueError('empty evaluation input')
    result=classification_metrics(truth,predicted,5)
    result['loss']=total/len(truth)
    return result,np.concatenate(probabilities)


def run(raw_root,output,parent_root,device='cuda:0',seed=42,policy=None,epoch_limit=None):
    """Resume atomic complete epochs; epoch_limit is only for interruption tests."""
    policy=policy or ConvergencePolicy()
    raw_root,output,parent_root=map(lambda p:Path(p).resolve(),(raw_root,output,parent_root))
    if output==parent_root or output.is_relative_to(parent_root) or parent_root.is_relative_to(output):
        raise ValueError('output must be independent of parent')
    if output.is_relative_to(raw_root) or raw_root.is_relative_to(output):
        raise ValueError('output must not overlap input')
    parent=json.loads((parent_root/'native_metrics.json').read_text())
    if (parent['model']!='DF' or parent['feature']!='DIR' or parent['epochs']!=policy.baseline_epochs
            or parent['checkpoint_metric']!='accuracy'):
        raise ValueError('parent must be a complete fixed-budget DF/accuracy baseline')
    if any(len(parent[key])!=policy.baseline_epochs for key in ('loss','val_accuracy','val_macro_f1','val_balanced_accuracy')):
        raise ValueError('incomplete parent trajectory')
    identity=dict(parent=str(parent_root),parent_metrics_sha256=digest(parent_root/'native_metrics.json'),
        parent_checkpoint_sha256=digest(parent_root/'checkpoint-best.pt'),seed=seed,
        policy=asdict(policy),input_hashes={name:digest(raw_root/name) for name in ('train.npz','val.npz','label_mapping.json')},
        batch_size=parent['batch_size'],learning_rate=parent['learning_rate'],optimizer=parent['optimizer'],
        checkpoint_metric='accuracy',input='DIR-5000',version=1)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'identity.json').exists():
        if json.loads((output/'identity.json').read_text())!=identity:raise ValueError('experiment identity changed')
    else:save_text(output/'identity.json',json.dumps(identity,indent=2))
    if (output/'result.json').exists():return json.loads((output/'result.json').read_text())
    set_seed(seed)
    generator=torch.Generator().manual_seed(seed)
    spec=build_model_specs(ROOT/'Benchmark/all_default_configs')['DF']
    X_train,y_train,_=tensors_for_split(raw_root,'train',spec,1)
    X_val,y_val,_=tensors_for_split(raw_root,'val',spec,1)
    model=instantiate_model(spec,5).to(device)
    optimizer=make_optimizer(parent['optimizer'],model.parameters(),parent['learning_rate'])
    train_batches=make_loader(X_train,y_train,parent['batch_size'],True)
    val_batches=make_loader(X_val,y_val,parent['batch_size'],False)
    rows=[];best_accuracy=-1.0;best_relative=None;parity=False
    if (output/'last.pt').exists():
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('checkpoint identity changed')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng'],generator)
        rows=saved['rows'];best_accuracy=saved['best_accuracy'];best_relative=saved['best_relative']
        parity=saved['baseline_parity_passed']
    def stopping():
        if len(rows)<policy.baseline_epochs:
            return dict(stop=False,converged=False,reason='replaying_baseline',total_epochs=len(rows))
        return assess(rows[policy.baseline_epochs:],max(r['val_macro_f1'] for r in rows[:policy.baseline_epochs]),policy)
    for epoch in range(len(rows),policy.max_total_epochs):
        if stopping()['stop']:break
        if epoch_limit is not None and epoch>=epoch_limit:return dict(state='interrupted',epochs=len(rows))
        started=time.monotonic();model.train();losses=[]
        for values,labels in train_batches:
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(values.to(device)),labels.to(device))
            if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('non-finite gradient')
            optimizer.step();losses.append(float(loss.detach().cpu()))
        metrics,_=evaluate(model,val_batches,device)
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),val_loss=metrics['loss'],
            val_accuracy=metrics['accuracy'],val_macro_f1=metrics['macro_f1'],
            val_balanced_accuracy=metrics['balanced_accuracy'],learning_rate=parent['learning_rate'],
            elapsed_seconds=time.monotonic()-started)
        if not all(math.isfinite(v) for v in row.values()):raise RuntimeError('non-finite epoch metrics')
        if epoch<policy.baseline_epochs:
            comparisons={'train_loss':'loss','val_accuracy':'val_accuracy','val_macro_f1':'val_macro_f1','val_balanced_accuracy':'val_balanced_accuracy'}
            for current,old in comparisons.items():
                if row[current]!=parent[old][epoch]:
                    raise RuntimeError(f'baseline replay mismatch at epoch {epoch+1}, {current}: {row[current]} != {parent[old][epoch]}')
        if metrics['accuracy']>=best_accuracy:
            best_accuracy=metrics['accuracy']
            best_relative=f'checkpoints/epoch-{epoch+1:03d}.pt'
            save_checkpoint(output/best_relative,dict(model=model.state_dict(),epoch=epoch,
                validation=metrics,identity=identity))
        rows.append(row)
        if len(rows)==policy.baseline_epochs:
            original=torch.load(parent_root/'checkpoint-best.pt',map_location='cpu',weights_only=False)
            reconstructed=torch.load(output/best_relative,map_location='cpu',weights_only=False)
            if original['epoch']!=reconstructed['epoch']+1:raise RuntimeError('baseline best epoch differs')
            for key,value in original['model_state_dict'].items():
                if not torch.equal(value,reconstructed['model'][key]):raise RuntimeError('baseline tensor replay differs: '+key)
            parity=True
            save_text(output/'baseline-parity.json',json.dumps(dict(passed=True,epochs=len(rows),fields_per_epoch=4,
                checkpoint_tensors_identical=True,reconstruction='seed42 model/Adamax/RNG replay; no warm start')))
        if epoch>=policy.baseline_epochs and not parity:raise RuntimeError('cannot extend before verified replay')
        save_checkpoint(output/'last.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            identity=identity,epoch=epoch,rows=rows,best_accuracy=best_accuracy,best_relative=best_relative,
            baseline_parity_passed=parity,rng=capture_rng(generator)))
        save_text(output/'metrics.jsonl',''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
        print(json.dumps(dict(row,baseline_parity_passed=parity,stopping=stopping())),flush=True)
    status=stopping()
    if not status['stop']:return dict(state='interrupted',epochs=len(rows))
    best=torch.load(output/best_relative,map_location='cpu',weights_only=False)
    model.load_state_dict(best['model'])
    # No further training after this point; test is used only for final reporting.
    X_test,y_test,_=tensors_for_split(raw_root,'test',spec,1)
    test,soft=evaluate(model,make_loader(X_test,y_test,parent['batch_size'],False),device)
    np.save(output/'test_soft_predictions.npy',soft)
    np.save(output/'test_labels.npy',y_test.numpy())
    save_checkpoint(output/'best.pt',best)
    result=dict(state='complete',epochs=len(rows),stopping=status,baseline_parity_passed=parity,
        best_epoch=best['epoch']+1,validation=best['validation'],test=test,
        highest_val_macro_f1=max(r['val_macro_f1'] for r in rows),
        checkpoint_metric='validation accuracy, latest tie',identity=identity,
        parameters=sum(p.numel() for p in model.parameters()),
        train_samples=len(y_train),val_samples=len(y_val),test_samples=len(y_test),
        test_cache_sha256=digest(raw_root/'test.npz'),
        elapsed_seconds=sum(r['elapsed_seconds'] for r in rows))
    save_text(output/'result.json',json.dumps(result,indent=2))
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--parent-root',required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    print(json.dumps(run(args.raw_root,args.output,args.parent_root,device=args.device),indent=2))


if __name__=='__main__':main()
