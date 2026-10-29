import argparse
import datetime
import json
import numpy as np
import os
import re
import sys
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

import timm

# assert timm.__version__ == "0.3.2"  # version check
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import os
import PIL

from torchvision import datasets, transforms

import util.lr_decay as lrd
import util.misc as misc
from util.repro import set_seed_deterministic, worker_init_fn
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_YaTC

from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('YaTC fine-tuning for traffic classification', add_help=False)
    # 64
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='TraFormer_YaTC', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=40, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
#20
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--finetune', default='./output_dir/pretrained-model.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--data_path', default='./data/ISCXVPN2016_MFR', type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number of the classification types')
    parser.add_argument('--scenario_task_groups', default='',
                        help='Relabel cached images by episode task ID without changing image bytes')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int,
                        help='Base random seed (per-rank offset applied for DDP).')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='仅评估：加载 --resume 权重，在 --eval_split 上跑一遍（不训练）')
    parser.add_argument('--eval_split', default='test', choices=['valid', 'test'],
                        help='与 --eval 联用：在验证集或测试集上评估（默认 test）')
    parser.add_argument('--early_stop', default=0, type=int,
                        help='Early stopping patience (0=disabled). Stop if val acc does not improve for N epochs.')
    parser.add_argument('--max_train_steps', default=0, type=int,
                        help='若 >0，全局训练步数达到后退出（显存探测）；0 为正常训练。')
    parser.add_argument('--step_bench_warmup', default=10, type=int,
                        help='step timing benchmark 丢弃的前 N 个 warmup step。')
    parser.add_argument('--bench_first_epoch_only', action='store_true',
                        help='仅跑第 1 个 epoch，打印 [EPOCH_BENCH] 后退出（不跑 test）。')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def build_dataset(data_split, args):
    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    assert data_split in ["train", "valid", "test"]
    root = os.path.join(args.data_path, data_split)
    dataset = datasets.ImageFolder(root, transform=transform)
    if args.scenario_task_groups:
        groups = json.loads(Path(args.scenario_task_groups).read_text())['groups']
        classes = sorted(group['id'] for group in groups)
        class_to_idx = {label: index for index, label in enumerate(classes)}
        task_to_label = {int(task): group['id'] for group in groups for task in group['tasks']}
        remapped = []
        for image, _ in dataset.samples:
            match = re.search(r'_A(\d{2})\d{2}_flow\d+', Path(image).stem)
            if not match or int(match.group(1)) not in task_to_label:
                raise ValueError(f'Cannot map image to a five-class scenario: {image}')
            remapped.append((image, class_to_idx[task_to_label[int(match.group(1))]]))
        dataset.samples = sorted(remapped, key=lambda item: (item[1], item[0]))
        dataset.imgs = dataset.samples
        dataset.targets = [target for _, target in dataset.samples]
        dataset.classes = classes
        dataset.class_to_idx = class_to_idx
    if len(dataset.classes) != args.nb_classes:
        raise ValueError(f'Class count {len(dataset.classes)} does not match --nb_classes={args.nb_classes}')

    print(dataset)

    return dataset

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # 可复现：Python/NumPy/PyTorch/CUDA + cudnn 确定性；DataLoader 使用 generator 与 worker_init_fn
    rank = misc.get_rank()
    process_seed = args.seed + rank
    rng_generator = set_seed_deterministic(process_seed)

    if args.eval:
        if not args.resume:
            print("错误: --eval 必须与 --resume 联用，指定微调 checkpoint（例如 output/<DATASET>/checkpoint-best.pth）")
            sys.exit(1)
        dataset_eval = build_dataset(data_split=args.eval_split, args=args)
        sampler_eval = torch.utils.data.SequentialSampler(dataset_eval)
        data_loader_eval = torch.utils.data.DataLoader(
            dataset_eval, sampler=sampler_eval,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        )
        model = models_YaTC.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
        )
        model.to(device)
        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=1e-4)
        loss_scaler = NativeScaler()
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
        per_path = os.path.join(args.output_dir, "per_class_f1.csv") if args.output_dir else None
        test_stats = evaluate(
            data_loader_eval, model, device,
            class_names=list(dataset_eval.classes) if per_path else None,
            per_class_csv_path=per_path,
        )
        split_name = "测试集" if args.eval_split == "test" else "验证集"
        print(f"\n=== 仅评估 ({split_name}, {len(dataset_eval)} 张) ===")
        print(f"Acc@1: {test_stats['acc1']:.4f}")
        print(f"Macro-F1 (average=macro): {test_stats['macro_f1']:.4f}")
        print(f"Weighted-F1 (average=weighted): {test_stats['weighted_f1']:.4f}")
        if args.output_dir:
            serializable = {}
            for k, v in test_stats.items():
                if hasattr(v, 'tolist'):
                    serializable[k] = v.tolist()
                elif isinstance(v, float):
                    serializable[k] = v
                else:
                    serializable[k] = v
            out_name = "test_stats.json" if args.eval_split == "test" else "val_stats.json"
            out_json = os.path.join(args.output_dir, out_name)
            with open(out_json, "w") as f:
                json.dump(serializable, f, indent=2)
            print(f"已写入: {out_json}")
        return

    dataset_train = build_dataset(data_split='train', args=args)
    dataset_val = build_dataset(data_split='valid', args=args)

    labels = dataset_val.classes

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True,
            seed=args.seed,
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(
            dataset_train, generator=rng_generator,
        )
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
        generator=rng_generator,
    )

    # 验证集为顺序采样，不传 generator，避免与训练 DataLoader 共享 RNG 状态
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = models_YaTC.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        # MAE's norm is replaced by a new fc_norm in the original fine-tuning model.
        missing_encoder = [key for key in msg.missing_keys
                           if not key.startswith(('head.', 'fc_norm.'))]
        if missing_encoder:
            raise ValueError(f"Pretrained YaTC encoder is incomplete: {missing_encoder}")
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            with open(Path(args.output_dir) / 'pretrained_load.json', 'w') as f:
                json.dump({'checkpoint': str(Path(args.finetune).resolve()),
                           'matched_model_keys': len(state_dict) - len(msg.missing_keys),
                           'missing_keys': msg.missing_keys,
                           'unexpected_keys': msg.unexpected_keys}, f, indent=2)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if device.type == "cuda" and getattr(args, "max_train_steps", 0):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    print(f"Start training for {args.epochs} epochs")
    if args.early_stop > 0:
        print(f"Early stopping enabled: patience = {args.early_stop}")
    start_time = time.time()
    max_accuracy = float('-inf')
    max_f1 = 0.0
    best_epoch = None
    best_metrics_path = Path(args.output_dir) / 'best_metrics.json'
    if args.resume and best_metrics_path.is_file():
        previous = json.loads(best_metrics_path.read_text())
        max_accuracy = previous['validation_accuracy']
        max_f1 = previous.get('validation_macro_f1', 0.0)
        best_epoch = previous['epoch']
    epochs_no_improve = 0
    mem_bench_stop = False

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        t_tr = time.perf_counter()
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args,
            global_step_base=epoch * len(data_loader_train),
        )
        train_sec = time.perf_counter() - t_tr

        if train_stats.get("mem_bench_stop"):
            mem_bench_stop = True
            break

        t_va = time.perf_counter()
        val_stats = evaluate(data_loader_val, model, device)
        val_sec = time.perf_counter() - t_va
        if getattr(args, "bench_first_epoch_only", False):
            print(
                f"[EPOCH_BENCH] epoch_train_sec={train_sec:.4f} "
                f"epoch_val_sec={val_sec:.4f} epoch_id={epoch}"
            )
            print("[EPOCH_BENCH] bench_first_epoch_only：已结束。")
            break

        print(f"Accuracy of the network on the {len(dataset_val)} val images: {val_stats['acc1']:.4f}")
        print(f"Macro-F1 (val): {val_stats['macro_f1']:.4f} | Weighted-F1 (val): {val_stats['weighted_f1']:.4f}")

        if log_writer is not None:
            ep = epoch
            log_writer.add_scalar('train/loss', train_stats['loss'], ep)
            log_writer.add_scalar('train/lr', train_stats.get('lr', 0.0), ep)
            if train_stats.get('acc1') is not None:
                log_writer.add_scalar('train/acc', train_stats['acc1'], ep)
            log_writer.add_scalar('val/loss', val_stats['loss'], ep)
            log_writer.add_scalar('val/acc', val_stats['acc1'], ep)
            log_writer.add_scalar('val/acc5', val_stats.get('acc5', 0.0), ep)
            log_writer.add_scalar('val/macro_f1', val_stats['macro_f1'], ep)
            log_writer.add_scalar('val/macro_precision', val_stats.get('macro_pre', 0.0), ep)
            log_writer.add_scalar('val/macro_recall', val_stats.get('macro_rec', 0.0), ep)
            log_writer.add_scalar('val/weighted_f1', val_stats.get('weighted_f1', 0.0), ep)
            log_writer.flush()

        if val_stats["acc1"] > max_accuracy:
            max_accuracy = val_stats["acc1"]
            best_epoch = epoch
            epochs_no_improve = 0
            if args.output_dir:
                misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name="best")
                if misc.is_main_process():
                    best_metrics_path.write_text(json.dumps({
                        'epoch': epoch, 'validation_accuracy': max_accuracy,
                        'validation_macro_f1': val_stats['macro_f1'],
                    }, indent=2) + '\n')
        else:
            epochs_no_improve += 1
        max_f1 = max(max_f1, val_stats["macro_f1"])
        print(f'Max Val Accuracy: {max_accuracy:.4f}')
        print(f'Max Val Macro-F1: {max_f1:.4f}')

        log_stats = {**{f'train_{k}': float(v) for k, v in train_stats.items()
                       if isinstance(v, (int, float, np.number))},
                     **{f'val_{k}': float(v) for k, v in val_stats.items()
                        if isinstance(v, (int, float, np.number))},
                     'epoch': epoch,
                     'train_seconds': train_sec, 'val_seconds': val_sec,
                     'best_val_accuracy': max_accuracy, 'best_epoch': best_epoch,
                     'n_parameters': n_parameters}
        if args.output_dir:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, name="last")
            if misc.is_main_process():
                with open(Path(args.output_dir) / 'epoch_metrics.jsonl', 'a') as f:
                    f.write(json.dumps(log_stats) + '\n')
        if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
            print(f'Early stopping at epoch {epoch}: no improvement for {args.early_stop} epochs')
            break

    if log_writer is not None:
        log_writer.close()

    if mem_bench_stop:
        print("[MEM_BENCH] 已跳过后续验证与测试")
        return

    if getattr(args, "bench_first_epoch_only", False):
        return

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.output_dir:
        best_ckpt = os.path.join(args.output_dir, "checkpoint-best.pth")
        if os.path.exists(best_ckpt):
            checkpoint = torch.load(best_ckpt, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Loaded best checkpoint for final test evaluation")

    dataset_test = build_dataset(data_split='test', args=args)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )
    per_path = os.path.join(args.output_dir, "per_class_f1.csv") if args.output_dir else None
    test_stats = evaluate(
        data_loader_test, model, device,
        class_names=list(dataset_test.classes) if per_path else None,
        per_class_csv_path=per_path,
    )
    print(f"\n=== Final Test Results ===")
    print(f"Test Accuracy (Acc@1): {test_stats['acc1']:.4f}")
    print(f"Test Macro-F1 (sklearn average=macro): {test_stats['macro_f1']:.4f}")
    print(f"Test Weighted-F1 (sklearn average=weighted): {test_stats['weighted_f1']:.4f}")
    if args.output_dir:
        serializable = {}
        for k, v in test_stats.items():
            if hasattr(v, 'tolist'):
                serializable[k] = v.tolist()
            elif isinstance(v, float):
                serializable[k] = v
            else:
                serializable[k] = v
        with open(os.path.join(args.output_dir, "test_stats.json"), "w") as f:
            json.dump(serializable, f, indent=2)
        unified_dir = os.environ.get("ANT_UNIFIED_OUTPUT_DIR")
        if unified_dir:
            Path(unified_dir).mkdir(parents=True, exist_ok=True)
            hard = [str(int(x)) for x in test_stats.get("hard_predictions", [])]
            labels_out = [str(int(x)) for x in test_stats.get("test_labels", [])]
            soft = np.asarray(test_stats.get("soft_predictions", []), dtype=np.float32)
            with open(os.path.join(unified_dir, "hard_predictions.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(hard) + ("\n" if hard else ""))
            with open(os.path.join(unified_dir, "test_labels.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(labels_out) + ("\n" if labels_out else ""))
            np.save(os.path.join(unified_dir, "soft_predictions.npy"), soft)
            class_names = list(dataset_test.classes)
            with open(os.path.join(unified_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {"id_to_label": {str(i): name for i, name in enumerate(class_names)}},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            prepared_metadata = Path(args.data_path).parent / 'row_metadata.jsonl'
            if prepared_metadata.is_file():
                metadata = {}
                with prepared_metadata.open() as f:
                    for line in f:
                        row = json.loads(line)
                        if row['split'] == 'test':
                            key = str(Path(row['image']).resolve())
                            if key in metadata:
                                raise ValueError(f'Duplicate test image metadata: {key}')
                            metadata[key] = row
                with open(Path(unified_dir) / 'test_row_metadata.jsonl', 'w') as f:
                    for image, label_id in dataset_test.samples:
                        row = dict(metadata[str(Path(image).resolve())])
                        row['label_id'] = label_id
                        row['label'] = dataset_test.classes[label_id]
                        f.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
