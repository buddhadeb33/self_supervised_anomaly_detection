from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM


def fit_one_class_svm(train_features: np.ndarray, nu: float = 0.1, gamma: str = "scale"):
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(train_features)
    return model


def score_one_class_svm(model: OneClassSVM, features: np.ndarray) -> np.ndarray:
    scores = -model.decision_function(features)
    return scores

