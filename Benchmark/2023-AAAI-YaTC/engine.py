import csv
import math
import os
import sys
import time
from typing import Iterable, List, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np


def _per_class_prf1_from_cm(cm: np.ndarray, eps: float = 1e-12):
    """sklearn confusion_matrix: 行=真实类 i，列=预测类 j。"""
    n = cm.shape[0]
    pre = np.zeros(n)
    rec = np.zeros(n)
    f1 = np.zeros(n)
    sup = cm.sum(axis=1).astype(np.int64)
    for i in range(n):
        tp = float(cm[i, i])
        col = cm[:, i].sum()
        row = cm[i, :].sum()
        pre[i] = tp / (col + eps) if col > 0 else 0.0
        rec[i] = tp / (row + eps) if row > 0 else 0.0
        if pre[i] + rec[i] > eps:
            f1[i] = 2 * pre[i] * rec[i] / (pre[i] + rec[i] + eps)
        else:
            f1[i] = 0.0
    return pre, rec, f1, sup


def _write_per_class_csv(cm: np.ndarray, path: str, class_names: List[str]) -> None:
    pre, rec, f1, sup = _per_class_prf1_from_cm(cm)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "precision", "recall", "f1", "support"])
        for i, name in enumerate(class_names):
            w.writerow([name, f"{pre[i]:.6f}", f"{rec[i]:.6f}", f"{f1[i]:.6f}", int(sup[i])])


def pretrain_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    model_without_ddp=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('steps', misc.SmoothedValue(window_size=1, fmt='{value:.0f}'))
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    steps_of_one_epoch = len(data_loader)

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        steps = steps_of_one_epoch * epoch + data_iter_step
        metric_logger.update(steps=int(steps))

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
        if args.output_dir and steps % args.save_steps_freq == 0 and epoch > 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, name='step'+str(steps))


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None, global_step_base: int = 0):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    pred_all = []
    target_all = []

    step_times_ms: list = []

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if device.type == "cuda":
            torch.cuda.synchronize()
        _t_step_begin = time.perf_counter()

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        _, pred = outputs.topk(1, 1, True, True)
        pred = pred.t()
        # 仅无 Mixup 时累积，用于 epoch 末训练集准确率（软标签下 pred/target 不可比）
        if mixup_fn is None:
            pred_all.extend(pred[0].cpu())
            target_all.extend(targets.cpu())

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss_step', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/lr_step', max_lr, epoch_1000x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times_ms.append((time.perf_counter() - _t_step_begin) * 1000.0)

        if getattr(args, "max_train_steps", 0) and args.max_train_steps > 0:
            g = global_step_base + data_iter_step + 1
            if g >= args.max_train_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    pa = torch.cuda.max_memory_allocated(device) / (1024**3)
                    pr = torch.cuda.max_memory_reserved(device) / (1024**3)
                    print(
                        f"[MEM_BENCH] peak_allocated_gb={pa:.4f} peak_reserved_gb={pr:.4f}"
                    )
                _warm = getattr(args, "step_bench_warmup", 10)
                measured = step_times_ms[_warm:]
                if measured:
                    import statistics as _st
                    _mean = sum(measured) / len(measured)
                    _std = _st.pstdev(measured) if len(measured) > 1 else 0.0
                    _bs = getattr(args, "batch_size", "?")
                    print(
                        f"[STEP_BENCH] step_time_ms_mean={_mean:.4f} step_time_ms_std={_std:.4f} "
                        f"n_measured={len(measured)} n_warmup={_warm} batch_size={_bs}"
                    )
                print(f"[MEM_BENCH] stopped after {g} train steps")
                out = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
                out["mem_bench_stop"] = True
                return out

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    out = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # 单卡且无 Mixup 时记录训练集准确率；DDP 下各进程仅见子集，此处不计算以免误导
    if mixup_fn is None and len(pred_all) > 0 and misc.get_world_size() == 1:
        out['acc1'] = float(accuracy_score(target_all, pred_all))
    return out


@torch.no_grad()
def evaluate(data_loader, model, device, class_names: Optional[List[str]] = None,
             per_class_csv_path: Optional[str] = None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    pred_all = []
    target_all = []
    soft_all = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()

        pred_all.extend(pred[0].cpu())
        target_all.extend(target.cpu())
        # Keep small logit differences when exporting AMP predictions for AUC/aggregation.
        soft_all.extend(torch.softmax(output.float(), dim=1).detach().cpu())

        topk = (1, min(5, output.shape[1]))
        acc1, acc5 = accuracy(output, target, topk=topk)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item()/100, n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item()/100, n=batch_size)

    y_true = np.asarray(target_all, dtype=np.int64)
    y_pred = np.asarray(pred_all, dtype=np.int64)

    weighted = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    macro_pre, macro_rec, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.4f} Acc@5 {top5.global_avg:.4f} loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    print(
        '* Macro Pre {mp:.4f} Rec {mr:.4f} F1 {mf:.4f} (sklearn average=macro)'
        .format(mp=macro_pre, mr=macro_rec, mf=macro_f1))
    print(
        '* Weighted Pre {wp:.4f} Rec {wr:.4f} F1 {wf:.4f} (sklearn average=weighted)'
        .format(wp=weighted[0], wr=weighted[1], wf=weighted[2]))

    test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    test_state['macro_pre'] = float(macro_pre)
    test_state['macro_rec'] = float(macro_rec)
    test_state['macro_f1'] = float(macro_f1)
    test_state['weighted_pre'] = float(weighted[0])
    test_state['weighted_rec'] = float(weighted[1])
    test_state['weighted_f1'] = float(weighted[2])
    test_state['cm'] = cm
    test_state['hard_predictions'] = y_pred.tolist()
    test_state['test_labels'] = y_true.tolist()
    test_state['soft_predictions'] = [x.tolist() for x in soft_all]

    if per_class_csv_path and class_names is not None and misc.is_main_process():
        if len(class_names) != cm.shape[0]:
            print(f"警告: class_names 长度 ({len(class_names)}) 与混淆矩阵阶数 ({cm.shape[0]}) 不一致，跳过 per-class CSV")
        else:
            _write_per_class_csv(cm, per_class_csv_path, class_names)
            print(f"Per-class breakdown 已保存: {per_class_csv_path}")

    return test_state
