from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def fit_knn(train_features: np.ndarray, k: int = 5) -> NearestNeighbors:
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(train_features)
    return knn


def score_knn(knn: NearestNeighbors, features: np.ndarray) -> np.ndarray:
    distances, _ = knn.kneighbors(features, return_distance=True)
    return distances.mean(axis=1)

