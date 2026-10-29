"""
This script provides an exmaple to wrap UER-py for classification.
"""
import os
import sys
import json
import time
sys.path.append(os.getcwd())
import random
import argparse
import torch
import torch.nn as nn
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        temp_output = output
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits
            #return temp_output, logits


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return max(2, len(labels_set))


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        print("Initialize with pretrained model.")
        pretrained = torch.load(args.pretrained_model_path, map_location="cpu")
        current = model.state_dict()
        compatible = {}
        skipped = []
        for name, value in pretrained.items():
            if name in current and current[name].shape == value.shape:
                compatible[name] = value
            else:
                skipped.append(name)
        model.load_state_dict(compatible, strict=False)
        if skipped:
            print("Skipped incompatible pretrained parameters: {}".format(", ".join(skipped)))
    else:
        print("Initialize with normal distribution.")
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else:  # Sentence-pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)
 
            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))

    return dataset

def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def _compute_auc(y_true, y_soft, labels_num):
    if not y_true or not y_soft:
        return None
    try:
        soft = np.asarray(y_soft, dtype=np.float32)
        if labels_num == 2:
            if len(set(y_true)) < 2 or soft.shape[1] < 2:
                return None
            return float(roc_auc_score(y_true, soft[:, 1]))
        return float(roc_auc_score(y_true, soft, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def evaluate(args, dataset, print_confusion_matrix=False, hard_prediction_path=None, soft_prediction_path=None, label_path=None):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    y_true, y_pred, y_soft = [], [], []
    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        src_batch = src_batch.to(args.device)
        # print(src_batch[0][113],args.tokenizer.convert_ids_to_tokens([src_batch.cpu().numpy()[0][113]]))
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
        prob = nn.Softmax(dim=1)(logits)
        pred = torch.argmax(prob, dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
            y_true.append(int(gold[j].cpu()))
            y_pred.append(int(pred[j].cpu()))
            y_soft.append(prob[j].detach().cpu().numpy())
        correct += torch.sum(pred == gold).item()
        
    
    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        cf_array = confusion.numpy()
        # with open("./results/confusion_matrix",'w') as f:
        #     for cf_a in cf_array:
        #         f.write(str(cf_a)+'\n')
        print("Report precision, recall, and f1:")
        eps = 1e-9
        for i in range(confusion.size()[0]):
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            if (p + r) == 0:
                f1 = 0
            else:
                f1 = 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
        

    accuracy = correct / len(dataset) if dataset else 0.0
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    micro_precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    auc = _compute_auc(y_true, y_soft, args.labels_num)

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(accuracy, correct, len(dataset)))
    print("Macro precision: {:.4f}, Micro precision: {:.4f}, Weighted precision: {:.4f}".format(
        macro_precision, micro_precision, weighted_precision))
    print("Macro recall: {:.4f}, Micro recall: {:.4f}, Weighted recall: {:.4f}".format(
        macro_recall, micro_recall, weighted_recall))
    print("Macro f1: {:.4f}, Micro f1: {:.4f}, Weighted f1: {:.4f}".format(
        macro_f1, micro_f1, weighted_f1))
    if auc is not None:
        print("AUC: {:.4f}".format(auc))

    if hard_prediction_path is not None:
        hard_dir = os.path.dirname(hard_prediction_path)
        if hard_dir:
            os.makedirs(hard_dir, exist_ok=True)
        with open(hard_prediction_path, mode="w", encoding="utf-8") as f:
            f.write("\n".join(str(x) for x in y_pred) + "\n")
    if label_path is not None:
        label_dir = os.path.dirname(label_path)
        if label_dir:
            os.makedirs(label_dir, exist_ok=True)
        with open(label_path, mode="w", encoding="utf-8") as f:
            f.write("\n".join(str(x) for x in y_true) + "\n")
    if soft_prediction_path is not None:
        soft_dir = os.path.dirname(soft_prediction_path)
        if soft_dir:
            os.makedirs(soft_dir, exist_ok=True)
        np.save(soft_prediction_path, np.asarray(y_soft, dtype=np.float32))

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "micro_precision": float(micro_precision),
        "weighted_precision": float(weighted_precision),
        "macro_recall": float(macro_recall),
        "micro_recall": float(micro_recall),
        "weighted_recall": float(weighted_recall),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "weighted_f1": float(weighted_f1),
        "auc": auc,
        "confusion": confusion,
        "total": int(len(dataset)),
        "correct": int(correct),
    }


def _threshold_metrics_ok(metrics, names, threshold):
    for name in names:
        value = metrics.get(name)
        if value is None or float(value) <= threshold:
            return False
    return True


def _json_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key != "confusion"
    }


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")
    
    parser.add_argument("--earlystop", type=int, default=5, help="early stop rounds.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
    parser.add_argument("--prediction_path", default=None,
                        help="Optional path for hard test predictions, one integer label per line.")
    parser.add_argument("--soft_prediction_path", default=None,
                        help="Optional .npy path for soft test predictions.")
    parser.add_argument("--label_path", default=None,
                        help="Optional path for test labels aligned with predictions.")
    parser.add_argument("--dev_prediction_path", default=None,
                        help="Optional path for hard validation predictions, one integer label per line.")
    parser.add_argument("--dev_soft_prediction_path", default=None,
                        help="Optional .npy path for soft validation predictions.")
    parser.add_argument("--dev_label_path", default=None,
                        help="Optional path for validation labels aligned with predictions.")
    parser.add_argument(
        "--epochwise_output_dir",
        default=None,
        help="Optional directory for per-epoch checkpoints and test predictions.",
    )
    parser.add_argument(
        "--early_stop_metric_threshold",
        type=float,
        default=0.99,
        help="Stop training and run one test-set evaluation when all selected validation metrics exceed this threshold.",
    )
    parser.add_argument(
        "--early_stop_test_threshold",
        type=float,
        default=None,
        help="Report whether selected test metrics also exceed this threshold. Defaults to validation threshold.",
    )
    parser.add_argument(
        "--early_stop_metrics",
        default="accuracy,macro_f1,auc",
        help="Comma-separated metric names used for validation/test high-score early stopping.",
    )
    
    #MOE Model Options
    parser.add_argument("--is_moe", action="store_true", help="adopt moe layer.")
    parser.add_argument("--vocab_size", type=int, required=False, help="Number of vocab.")
    parser.add_argument("--moebert_expert_dim", type=int, required=False, default=3072, help="Dim of expert,default is ffn.")
    parser.add_argument("--moebert_expert_num", type=int, required=False, help="Number of expert.")
    parser.add_argument("--moebert_route_method", choices=["gate-token", "gate-sentence", "hash-random", "hash-balance","proto"], default="hash-random",
                        help="moebert route method.")
    parser.add_argument("--moebert_route_hash_list", default=None, type=str, help="Path of moebert hash list file.")
    parser.add_argument("--moebert_load_balance", type=float, default=0.0, help="gate loss weight.")
    
    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    if args.train_path is None:
        label_source_path = args.dev_path or args.test_path
        if label_source_path is None:
            raise ValueError("Either train_path, dev_path, or test_path is required to infer label count.")
        args.labels_num = count_labels_num(label_source_path)
    else:
        args.labels_num = count_labels_num(args.train_path)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    if args.train_path is None:
        args.model = model
        print("No train data, only evaluate..")
        if args.test_path is not None:
            evaluate(
                args,
                read_dataset(args, args.test_path),
                True,
                hard_prediction_path=args.prediction_path,
                soft_prediction_path=args.soft_prediction_path,
                label_path=args.label_path,
            )
        elif args.dev_path is not None:
            evaluate(
                args,
                read_dataset(args, args.dev_path),
                True,
                hard_prediction_path=args.prediction_path,
                soft_prediction_path=args.soft_prediction_path,
                label_path=args.label_path,
            )
        if args.dev_path is not None and (
            args.dev_prediction_path is not None
            or args.dev_soft_prediction_path is not None
            or args.dev_label_path is not None
        ):
            evaluate(
                args,
                read_dataset(args, args.dev_path),
                False,
                hard_prediction_path=args.dev_prediction_path,
                soft_prediction_path=args.dev_soft_prediction_path,
                label_path=args.dev_label_path,
            )
        return
    
    # Training phase.
    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size
    
    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])
    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[3] for example in trainset])
    else:
        soft_tgt = None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0
    best_result_round = 0
    devset = read_dataset(args, args.dev_path)
    testset = read_dataset(args, args.test_path) if args.test_path is not None else None
    # print("Start training.")
    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        epoch_started = time.time()
        model.train()
        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        
        result = evaluate(args, devset)
        test_result = None
        if args.epochwise_output_dir is not None:
            epoch_dir = os.path.join(args.epochwise_output_dir, "epoch_{:03d}".format(epoch))
            os.makedirs(epoch_dir, exist_ok=True)
            save_model(model, os.path.join(epoch_dir, "checkpoint.bin"))
            with open(os.path.join(epoch_dir, "epoch_runtime.json"), "w", encoding="utf-8") as handle:
                dev_confusion = result["confusion"]
                dev_total = int(dev_confusion.sum().item())
                dev_correct = int(torch.trace(dev_confusion).item())
                json.dump(
                    {
                        "epoch": epoch,
                        "elapsed_seconds": time.time() - epoch_started,
                        "dev_metrics": _json_metrics(result),
                        "dev_accuracy": float(dev_correct / dev_total) if dev_total else 0.0,
                    },
                    handle,
                    indent=2,
                )
        early_stop_metrics = [name.strip() for name in args.early_stop_metrics.split(",") if name.strip()]
        test_threshold = args.early_stop_metric_threshold if args.early_stop_test_threshold is None else args.early_stop_test_threshold
        if (
            testset is not None
            and args.early_stop_metric_threshold >= 0
            and early_stop_metrics
            and _threshold_metrics_ok(result, early_stop_metrics, args.early_stop_metric_threshold)
        ):
            print("Validation high-score threshold reached; stopping training and evaluating test set once.")
            save_model(model, args.output_model_path)
            test_result = evaluate(args, testset, True)
            test_threshold_passed = _threshold_metrics_ok(test_result, early_stop_metrics, test_threshold)
            if args.epochwise_output_dir is not None:
                epoch_dir = os.path.join(args.epochwise_output_dir, "epoch_{:03d}".format(epoch))
                with open(os.path.join(epoch_dir, "epoch_runtime.json"), "r", encoding="utf-8") as handle:
                    epoch_payload = json.load(handle)
                epoch_payload["test_metrics_on_threshold"] = _json_metrics(test_result)
                epoch_payload["test_threshold_passed"] = test_threshold_passed
                with open(os.path.join(epoch_dir, "epoch_runtime.json"), "w", encoding="utf-8") as handle:
                    json.dump(epoch_payload, handle, indent=2)
            if args.epochwise_output_dir is not None:
                with open(os.path.join(args.epochwise_output_dir, "early_stop.json"), "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "epoch": epoch,
                            "reason": "validation_threshold_reached",
                            "validation_threshold": args.early_stop_metric_threshold,
                            "test_threshold": test_threshold,
                            "test_threshold_passed": test_threshold_passed,
                            "metrics": early_stop_metrics,
                            "dev_metrics": _json_metrics(result),
                            "test_metrics": _json_metrics(test_result),
                            "note": "Training stops on validation metrics only; test metrics are recorded after the stop decision.",
                        },
                        handle,
                        indent=2,
                    )
            break
        if epoch == 1 or result["macro_f1"] > best_result:
            best_result = result["macro_f1"]
            best_result_round = epoch
            save_model(model, args.output_model_path)
        elif epoch - best_result_round >= args.earlystop:
            print("early stopping...")
            break

    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))
        evaluate(
            args,
            testset,
            True,
            hard_prediction_path=args.prediction_path,
            soft_prediction_path=args.soft_prediction_path,
            label_path=args.label_path,
        )
    if args.dev_path is not None and (
        args.dev_prediction_path is not None
        or args.dev_soft_prediction_path is not None
        or args.dev_label_path is not None
    ):
        print("Validation set prediction export.")
        evaluate(
            args,
            devset,
            False,
            hard_prediction_path=args.dev_prediction_path,
            soft_prediction_path=args.dev_soft_prediction_path,
            label_path=args.dev_label_path,
        )


if __name__ == "__main__":
    main()
