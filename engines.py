"""Adapters preserve the existing model implementations; no historical cache is required."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace

import numpy as np

from suite import ROOT, SPLITS, WF, WEIGHTS, read, write, task_label, execute, save_predictions


def seed(value):
    random.seed(value)
    np.random.seed(value)
    try:
        import torch
        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
    except ImportError:
        pass


def row_lookup(episodes):
    return {str(Path(r['pcap']).resolve()): r for r in episodes}


def source_row(value, lookup):
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    return lookup[str(p.resolve())]


def metadata_rows(path, lookup):
    with Path(path).open() as f:
        return [source_row(row['SOURCE_FILE'], lookup) for row in csv.DictReader(f)]


def labels_for(task, episodes):
    return sorted({task_label(r, task) for r in episodes})


def config(args, run):
    c = read(ROOT / 'configs' / args.task / (args.model + '.json'))
    c['runtime'] = {'python': sys.executable, 'output_root': str(args.output), 'timeout_seconds': None}
    c.setdefault('data', {})['label_mode'] = 'binary' if args.task == 'risk' else 'auto'
    a = c['algorithm_config']
    if args.model == 'TFusion':
        a['detection_model'] = 'rforest'
        if args.task == 'scenario':
            a['detector_params'] = {'n_estimators': 20, 'max_depth': 50}
    a['early_stop'] = {}
    conversion = a.setdefault('conversion', {})
    conversion.update(cache_dir=str(run / 'cache/flows'), image_cache_dir=str(run / 'cache/images'),
                      require_feature_cache=False, workers=args.workers)
    if args.model in WEIGHTS:
        a['checkpoint' if args.model == 'NetMamba' else 'pretrained_model_path'] = str(args.weights / WEIGHTS[args.model])
    ft = a.get('finetune', {})
    if ft:
        ft.update(device=args.device, num_workers=args.workers)
        if args.epochs is not None:
            ft['max_epoch'] = args.epochs
        if args.batch_size is not None:
            ft['batch_size'] = args.batch_size
    a['cache'] = {'enabled': True, 'dir': str(run / 'cache/tfusion')}
    a['device'] = args.device
    write(run / 'resolved_config.json', c)
    return c


def wf(args, run, episodes):
    from run_ant_wf_scenario_smoke import build_model_specs, write_raw_npz, train_one_model
    raw = run / 'prepared'
    raw.mkdir(parents=True, exist_ok=True)
    spec = build_model_specs(ROOT / 'Benchmark/all_default_configs')[args.model]
    selected = {}
    for split in SPLITS:
        selected[split] = [SimpleNamespace(pcap_path=Path(r['pcap']), engine=r['engine'],
                            episode_id=r['episode'], scenario=r['family'],
                            labels=SimpleNamespace(value=lambda _field, row=r: task_label(row, args.task)))
                           for r in episodes if r['split'] == split]
    manifest = write_raw_npz(selected, raw, spec.raw_seq_len, None, 'suite_label')
    if args.prepare_only:
        return
    ns = SimpleNamespace(device=args.device, require_cuda=args.device.startswith('cuda'),
                         feature_workers=args.workers, learning_rate=None, batch_size=args.batch_size,
                         epochs=args.epochs, optimizer=None, checkpoint_metric='accuracy',
                         epoch_metrics_path=run / 'training_progress.json')
    train_one_model(spec, raw, run / 'native', len(labels_for(args.task, episodes)), ns)
    lookup = row_lookup(episodes)
    rows = [source_row(r['pcap'], lookup) for r in manifest['splits']['test']['sources']]
    labels = list(read(raw / 'label_mapping.json')['id_to_label'].values())
    save_predictions(run / 'episode/test', np.load(run / 'native/soft_predictions.npy'), rows, labels, args.task)
    expected = {r['id'] for r in episodes if r['split'] == 'test'}
    write(run / 'coverage.json', {'test': {'expected': len(expected), 'evaluated': len(rows),
          'missing_episode_ids': sorted(expected - {r['id'] for r in rows})}})


def integrated(args, run, episodes):
    from Benchmark.data import prepare_data
    from Benchmark.algorithms import run_algorithm
    c = config(args, run)
    manifests = args.output / 'manifests' / args.task
    prepared = prepare_data(args.model, [manifests / 'train.json'], [manifests / 'test.json'],
                            c, run, [manifests / 'val.json'])
    if args.prepare_only:
        return
    result = run_algorithm(prepared, c, run)
    lookup = row_lookup(episodes)
    labels = labels_for(args.task, episodes)
    if args.model == 'FlowLens':
        def flow_rows(result, sources):
            with (result.output_dir / 'five_tuple_predictions.csv').open() as f:
                return [source_row(sources[int(re.match(r'test(\d+):', r['FLOW_ID'])[1])].pcap, lookup)
                        for r in csv.DictReader(f)]
        rows = flow_rows(result, prepared.test_packets)
        save_predictions(run / 'test', np.load(result.soft_predictions), rows, labels, args.task)
        c['algorithm_config']['eval_model_path'] = str(result.output_dir / 'flowlens_model.joblib')
        vp = replace(prepared, test_packets=prepared.validation_packets, validation_packets=None)
        vr = run_algorithm(vp, c, run / 'validation_inference')
        save_predictions(run / 'val', np.load(vr.soft_predictions), flow_rows(vr, vp.test_packets), labels, args.task)
        return
    mapping = read(prepared.label_mapping_path)['id_to_label']
    labels = [mapping[str(i)] for i in range(len(mapping))]
    save_predictions(run / 'test', np.load(result.soft_predictions),
                     metadata_rows(prepared.work_dir / 'test_row_metadata.csv', lookup), labels, args.task)
    if args.model == 'TrafficFormer':
        soft = np.load(result.output_dir / 'validation_soft_predictions.npy')
        rows = metadata_rows(prepared.work_dir / 'dev_row_metadata.csv', lookup)
    else:
        # eval.py reads its test directory; this view points only to validation images.
        view = run / 'validation_view'
        view.mkdir()
        (view / 'test').symlink_to(prepared.work_dir / 'valid', target_is_directory=True)
        (view / 'train').symlink_to(prepared.work_dir / 'train', target_is_directory=True)
        (view / 'label_mapping.json').write_text(prepared.label_mapping_path.read_text())
        c['algorithm_config']['eval_checkpoint_path'] = str(result.output_dir / 'checkpoint-best.pth')
        # Native eval.py does not accept training-only options such as --lr.
        c['algorithm_config']['extra_args'] = ['--no_amp'] if c['algorithm_config']['finetune'].get('no_amp') else []
        vp = replace(prepared, work_dir=view, test_path=view / 'test')
        vr = run_algorithm(vp, c, run / 'validation_inference')
        soft = np.load(vr.soft_predictions)
        rows = metadata_rows(prepared.work_dir / 'valid_row_metadata.csv', lookup)
    save_predictions(run / 'val', soft, rows, labels, args.task)


def tfusion(args, run, episodes):
    import torch
    import joblib
    from Benchmark import algorithms as a
    from Benchmark.data import prepare_data
    sys.path.insert(0, str(ROOT / 'Benchmark/2025-CCS-TFusion'))
    from model.detection import DetectorFactory
    c = config(args, run)
    alg = c['algorithm_config']
    # Native parser needs binary labels for benign-only encoder training, in both tasks.
    manifests = args.output / 'manifests/risk'
    c.setdefault('data', {})['label_mode'] = 'binary'
    prepared = prepare_data('TFusion', [manifests / 'train.json'], [manifests / 'test.json'],
                            c, run, [manifests / 'val.json'])
    raw, meta, flows, tensors = {}, {}, {}, {}
    for split, sources in zip(SPLITS, [prepared.training_packets, prepared.validation_packets, prepared.test_packets]):
        raw[split], meta[split] = a._tfusion_packet_sources_to_raw_flows(sources, alg, split)
        if not raw[split]:
            raise ValueError('No TFusion flows in ' + split)
    stats = a._tfusion_feature_stats(raw['train'])
    for split in SPLITS:
        flows[split], tensors[split], _, _ = a._tfusion_prepare_finalized_cached(
            raw[split], meta[split], stats, int(alg['data_construct_param']['segment_len']),
            alg, split, run / 'native_input' / (split + '.json'))
    if args.prepare_only:
        return
    model, stage1 = a._tfusion_stage1_model(args.device)
    train = alg['training']
    benign = [f for f in flows['train'] if f['label'] == 'benign']
    batches = a._tfusion_build_contrast_batches(benign, int(train['contrast_batch_size']), int(train['seed']))
    if batches is None:
        raise ValueError('No benign flows for TFusion encoder pretraining')
    a._tfusion_pretrain_stage1(stage1, batches, args.device, run / 'stage1_model.pt',
                              float(train['learning_rate']), float(train['weight_decay']),
                              args.epochs if args.epochs is not None else int(train['max_epoch']))
    model.eval()
    stage1.eval()
    features = {s: a._tfusion_embeddings(model, tensors[s], args.device,
                 args.batch_size or int(alg['infer_batch_size'])) for s in SPLITS}
    lookup = row_lookup(episodes)
    rows = {s: [source_row(f['source_file'], lookup) for f in flows[s]] for s in SPLITS}
    labels = labels_for(args.task, episodes)
    y = np.asarray([labels.index(task_label(r, args.task)) for r in rows['train']])
    detector = DetectorFactory.create('rforest')
    params = alg.get('detector_params', {})
    detector.model.set_params(**params, random_state=args.seed, n_jobs=args.workers)
    detector.train(features['train'], y)
    joblib.dump(detector.model, run / 'detector.joblib')
    write(run / 'preprocessing.json', {'stats': stats, 'encoder_training': 'train benign flows only',
                                      'segment_len': alg['data_construct_param']['segment_len'],
                                      'classifier_parameters': detector.model.get_params()})
    for split in ('val', 'test'):
        save_predictions(run / split, detector.model.predict_proba(features[split]), rows[split], labels, args.task)


def yatc(args, run, episodes):
    project = ROOT / 'Benchmark/2023-AAAI-YaTC'
    prepared = run / 'prepared'
    # Preserve the actual adapter settings used in the existing experiments.
    execute([sys.executable, project / 'adapt_ant_pcap.py', '--ant-root', args.data,
             '--out-dir', prepared, '--label-field', 'Risk' if args.task == 'risk' else 'Scenario',
             '--scope', 'flow', '--protocols', 'tcp,udp'], ROOT, run / 'prepare.log')
    if args.prepare_only:
        return
    hp = read(ROOT / 'Benchmark/all_default_configs/2023-AAAI-YaTC.json')['performance_hyperparameters']['model_training_hyperparameters']
    labels = labels_for(args.task, episodes)
    cmd = [sys.executable, 'fine-tune.py', '--data_path', prepared / 'images',
           '--nb_classes', len(labels), '--device', args.device, '--num_workers', args.workers,
           '--batch_size', args.batch_size or hp['batch_size'], '--seed', args.seed]
    if args.task == 'scenario':
        cmd += ['--scenario_task_groups', ROOT / 'configs/scenario.json']
    env = os.environ.copy()
    env.update(ANT_UNIFIED_OUTPUT_DIR=str(run / 'unified'), WANDB_MODE='offline', WANDB_DISABLED='true')
    train_cmd = cmd + ['--output_dir', run / 'native', '--log_dir', run / 'tensorboard',
                       '--epochs', args.epochs or hp['training_budget_epochs'],
                       '--finetune', args.weights / WEIGHTS['YaTC'], '--blr', hp['base_learning_rate'],
                       '--weight_decay', hp['weight_decay'], '--layer_decay', hp['layer_decay'],
                       '--drop_path', hp['drop_path'], '--warmup_epochs', hp['warmup_epochs'],
                       '--smoothing', hp['label_smoothing'], '--reprob', hp['random_erase_probability']]
    execute(train_cmd, project, run / 'train.log', env)
    execute(cmd + ['--eval', '--eval_split', 'valid', '--resume', run / 'native/checkpoint-best.pth',
                   '--output_dir', run / 'validation_inference'], project, run / 'validation.log')
    lookup = row_lookup(episodes)
    metadata = [json.loads(l) for l in (prepared / 'row_metadata.jsonl').read_text().splitlines()]
    for split in ('val', 'test'):
        ordered = [r for r in metadata if r['split'] == split]
        ordered.sort(key=lambda r: (labels.index(task_label(source_row(r['source_pcap'], lookup), args.task)), r['image']))
        rows = [source_row(r['source_pcap'], lookup) for r in ordered]
        if split == 'val':
            output = read(run / 'validation_inference/val_stats.json')
            p = output['soft_predictions']
            truth = output['test_labels']
        else:
            p = np.load(run / 'unified/soft_predictions.npy')
            truth = [int(x) for x in (run / 'unified/test_labels.txt').read_text().splitlines()]
        if truth != [labels.index(task_label(r, args.task)) for r in rows]:
            raise ValueError('YaTC prediction/metadata ordering mismatch')
        save_predictions(run / split, p, rows, labels, args.task)


def run_model(args, run, episodes):
    seed(args.seed)
    if args.model in WF:
        wf(args, run, episodes)
    elif args.model == 'TFusion':
        tfusion(args, run, episodes)
    elif args.model == 'YaTC':
        yatc(args, run, episodes)
    else:
        integrated(args, run, episodes)
