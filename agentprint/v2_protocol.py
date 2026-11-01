"""Auditable v2 stopping and selection rules, independent of the v1 schedule."""
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any

from .cache import write_json
from .config import ConfigError, fingerprint
from .convergence import ConvergencePolicy, assess


POLICY = ConvergencePolicy()


def validate_formal(config: dict[str, Any]) -> None:
    """Formal runs use the approved anchor and fixed cap; smoke stays bounded by config."""
    if config['mode'] != 'formal':
        return
    training = dict(epochs=200, batch_size=200, learning_rate=0.001,
                    weight_decay=0.001, lr_decay=0.5, min_learning_rate=1e-6)
    model = dict(width=2, dropout_2d=0.05, dropout_1d=0.1)
    if (config['seed'] != 42
            or any(config['training'][key] != value for key, value in training.items())
            or any(config['model'][key] != value for key, value in model.items())):
        raise ConfigError('v2 formal protocol requires seed=42, epochs=200, batch_size=200, '
                          'learning_rate=0.001, weight_decay=0.001, lr_decay=0.5, '
                          'min_learning_rate=0.000001, width=2, dropout_2d=0.05, dropout_1d=0.1')


def learning_rate(epoch: int, settings: dict[str, Any]) -> float:
    if type(epoch) is not int or epoch < 0:
        raise ValueError('epoch must be a nonnegative integer')
    return max(settings['min_learning_rate'],
               settings['learning_rate'] * settings['lr_decay'] ** (epoch / 30))


def stop_status(rows: list[dict[str, Any]], mode: str, epochs: int) -> dict[str, Any]:
    """Assess complete zero-based history with the original 30-epoch anchor."""
    if mode not in ('formal', 'smoke') or epochs < 1 or (mode == 'formal' and epochs != 200):
        raise ValueError('invalid v2 stopping mode / epoch budget')
    if len(rows) > epochs:
        raise ValueError('history exceeds epoch budget')
    for epoch, row in enumerate(rows):
        if type(row['epoch']) is not int or row['epoch'] != epoch:
            raise ValueError('complete epoch history must be contiguous from zero')
        if (not math.isfinite(row['val_macro_f1']) or not 0 <= row['val_macro_f1'] <= 1
                or not math.isfinite(row['val_loss']) or row['val_loss'] < 0):
            raise ValueError('invalid validation metric in epoch history')
    baseline = rows[:POLICY.baseline_epochs]
    baseline_best = max((row['val_macro_f1'] for row in baseline), default=0.0)
    status = assess(rows[POLICY.baseline_epochs:], baseline_best, POLICY)
    if len(rows) < POLICY.baseline_epochs:
        status.update(total_epochs=len(rows), additional_epochs=0,
                      last_significant_epoch=len(rows) - 1,
                      epochs_without_significant_gain=0)
    if mode == 'smoke':
        exhausted = len(rows) >= epochs
        status.update(stop=exhausted, converged=False,
                      reason='smoke_budget_exhausted' if exhausted else 'running')
    return status


def record_selection(output_root: str, saved: dict[str, Any], checkpoint: str | Path
                     ) -> tuple[Path, dict[str, Any]]:
    """Pair transforms using saved validation scores and every other identity field."""
    cohort = deepcopy(saved['identity'])
    transform = cohort.pop('transform')
    if transform not in ('raw', 'log1p'):
        raise ConfigError('invalid selection transform')
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'selection-v2-{fingerprint(cohort)[:16]}.json'
    selection = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {
        'cohort': cohort, 'selection_split': 'val', 'tie_rule': 'raw', 'candidates': {}}
    if selection.get('cohort') != cohort or selection.get('selection_split') != 'val':
        raise ConfigError('incompatible v2 selection cohort')
    candidates = selection['candidates']
    candidates[transform] = dict(checkpoint=str(Path(checkpoint).resolve()), epoch=saved['epoch'],
                                 experiment_id=saved['experiment_id'],
                                 validation_macro_f1=saved['best_macro_f1'])
    for name, candidate in candidates.items():
        if (name not in ('raw', 'log1p')
                or candidate['experiment_id'] != fingerprint(dict(cohort, transform=name))):
            raise ConfigError('incompatible v2 selection candidate identity')
        score = candidate['validation_macro_f1']
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ConfigError('checkpoint best validation Macro-F1 must be finite and in [0,1]')
    chosen = max((name for name in ('raw', 'log1p') if name in candidates),
                 key=lambda name: candidates[name]['validation_macro_f1'])
    selection.update(complete=all(name in candidates for name in ('raw', 'log1p')),
                     selected_transform=chosen)
    for name, candidate in candidates.items():
        candidate['validation_selected_main'] = name == chosen
    write_json(path, selection)
    return path, selection
