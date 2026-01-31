from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, tpr: float = 0.95) -> float:
    fpr, tpr_curve, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr_curve, tpr, side="left")
    if idx >= len(fpr):
        return float(fpr[-1])
    return float(fpr[idx])


def sensitivity_at_specificity(
    y_true: np.ndarray, y_score: np.ndarray, specificity: float = 0.95
) -> float:
    fpr, tpr_curve, _ = roc_curve(y_true, y_score)
    spec_curve = 1.0 - fpr
    mask = spec_curve >= specificity
    if np.any(mask):
        return float(tpr_curve[mask].max())
    return float(tpr_curve[np.argmax(spec_curve)])


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "auroc": compute_auroc(y_true, y_score),
        "auprc": compute_auprc(y_true, y_score),
        "fpr_at_95_tpr": fpr_at_tpr(y_true, y_score, tpr=0.95),
        "sens_at_95_spec": sensitivity_at_specificity(y_true, y_score, specificity=0.95),
    }

