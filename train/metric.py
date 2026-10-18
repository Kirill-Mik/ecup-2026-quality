"""Метрика: «по обеим категориям считается F1, после чего усредняются».

Формулировка допускает два прочтения; считаем все и оптимизируем
компромиссную версию A (macro-F1 внутри категории, усреднённый по
категориям) — она почти оптимальна для обоих прочтений.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CAT_BAD, CAT_FLAM  # noqa: E402


def metric_a(y_true, y_pred, cats) -> float:
    """macro-F1 внутри каждой категории, среднее по категориям."""
    scores = []
    for cat in (CAT_BAD, CAT_FLAM):
        m = cats == cat
        if m.sum() == 0:
            continue
        scores.append(f1_score(y_true[m], y_pred[m], average="macro"))
    return float(np.mean(scores))


def metric_b(y_true, y_pred, cats, positive=0) -> float:
    """бинарный F1 (positive=«бан», label 0) в каждой категории, среднее."""
    scores = []
    for cat in (CAT_BAD, CAT_FLAM):
        m = cats == cat
        if m.sum() == 0:
            continue
        scores.append(f1_score(y_true[m], y_pred[m], pos_label=positive))
    return float(np.mean(scores))


def metric_c(y_true, y_pred, cats=None) -> float:
    """глобальный macro-F1 по классам без разреза на категории."""
    return float(f1_score(y_true, y_pred, average="macro"))


def report(y_true, y_pred, cats, tag=""):
    a = metric_a(y_true, y_pred, cats)
    b0 = metric_b(y_true, y_pred, cats, positive=0)
    b1 = metric_b(y_true, y_pred, cats, positive=1)
    c = metric_c(y_true, y_pred)
    print(f"[{tag}] A(percat-macro)={a:.4f}  B(f1-ban)={b0:.4f}  "
          f"B(f1-ok)={b1:.4f}  C(global-macro)={c:.4f}")
    return a
