from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
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


def evaluate_from_scores(score_csv: str, label_col: str = "label", score_col: str = "score") -> Dict[str, float]:
    df = pd.read_csv(score_csv)
    y_true = df[label_col].astype(int).to_numpy()
    y_score = df[score_col].astype(float).to_numpy()
    return evaluate_scores(y_true, y_score)


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, tuple[float, float]]:
    rng = np.random.default_rng(seed)
    metrics = {key: [] for key in ["auroc", "auprc", "fpr_at_95_tpr", "sens_at_95_spec"]}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        res = evaluate_scores(y_true[idx], y_score[idx])
        for key, val in res.items():
            metrics[key].append(val)
    return {
        key: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        for key, vals in metrics.items()
    }

