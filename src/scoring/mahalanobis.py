from __future__ import annotations

import numpy as np


def fit_mahalanobis(train_features: np.ndarray):
    mean = train_features.mean(axis=0)
    cov = np.cov(train_features, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    return mean, cov_inv


def score_mahalanobis(features: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    delta = features - mean
    scores = np.einsum("ij,jk,ik->i", delta, cov_inv, delta)
    return scores

