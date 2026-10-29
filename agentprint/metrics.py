"""Metrics use the fixed true-label support of the evaluated manifest."""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .cache import LABEL_MAP, write_json


def classification_metrics(truth, predicted, task):
    if not truth:
        raise ValueError('cannot evaluate an empty split')
    classes = list(range(len(LABEL_MAP[task])))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=classes, zero_division=0)
    active = np.flatnonzero(support).tolist()
    summary = {'accuracy': float(accuracy_score(truth, predicted)),
               'macro_f1': float(f1[active].mean()),
               'support': support.tolist(), 'evaluated_classes': active,
               'sample_count': len(truth)}
    if task == 'primitive':
        summary['fixed47_macro_f1'] = float(f1.mean())
    per_class = [{'label_id': index, 'label_name': name,
                  'support': int(support[index]), 'precision': float(precision[index]),
                  'recall': float(recall[index]), 'f1': float(f1[index])}
                 for name, index in LABEL_MAP[task].items()]
    return summary, per_class, confusion_matrix(truth, predicted, labels=classes)


def write_metrics(directory, summary, per_class, matrix):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'summary.json', summary)
    with (root / 'per_class.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_class[0]))
        writer.writeheader()
        writer.writerows(per_class)
    with (root / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as handle:
        matrix_writer = csv.writer(handle)
        matrix_writer.writerow(['true_label_id'] + list(range(len(matrix))))
        for index, row in enumerate(matrix):
            matrix_writer.writerow([index] + row.tolist())
