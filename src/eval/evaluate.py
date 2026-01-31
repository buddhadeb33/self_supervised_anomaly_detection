from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .metrics import evaluate_scores


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
    return {key: (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) for key, vals in metrics.items()}

