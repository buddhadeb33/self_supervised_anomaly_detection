from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

from .ssl import MAEWrapper, MoCoV2, SimCLR


def load_ssl_encoder(
    checkpoint_path: str,
    method: str,
    backbone: str = "resnet50",
    projection_dim: int = 128,
    hidden_dim: int = 2048,
    use_projector: bool = False,
) -> torch.nn.Module:
    if method == "simclr":
        model = SimCLR(backbone=backbone, projection_dim=projection_dim, hidden_dim=hidden_dim)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        return model if use_projector else model.backbone
    if method == "moco":
        model = MoCoV2(
            backbone=backbone,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        return model.encoder_q if use_projector else model.encoder_q[0]
    raise ValueError(f"Unsupported method: {method}")


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    feats = []
    labels = []
    for batch in dataloader:
        images, batch_labels = batch[:2]
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]
        if normalize:
            outputs = F.normalize(outputs, dim=1)
        feats.append(outputs.cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def fit_knn(train_features: np.ndarray, k: int = 5) -> NearestNeighbors:
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(train_features)
    return knn


def score_knn(knn: NearestNeighbors, features: np.ndarray) -> np.ndarray:
    distances, _ = knn.kneighbors(features, return_distance=True)
    return distances.mean(axis=1)


def fit_mahalanobis(train_features: np.ndarray):
    mean = train_features.mean(axis=0)
    cov = np.cov(train_features, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    return mean, cov_inv


def score_mahalanobis(features: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    delta = features - mean
    scores = np.einsum("ij,jk,ik->i", delta, cov_inv, delta)
    return scores


def fit_one_class_svm(train_features: np.ndarray, nu: float = 0.1, gamma: str = "scale"):
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(train_features)
    return model


def score_one_class_svm(model: OneClassSVM, features: np.ndarray) -> np.ndarray:
    scores = -model.decision_function(features)
    return scores


@torch.no_grad()
def mae_reconstruction_scores(
    model: MAEWrapper,
    dataloader: Iterable,
    device: torch.device,
) -> np.ndarray:
    model.to(device)
    model.eval()
    scores = []
    for batch in dataloader:
        images = batch[0].to(device, non_blocking=True)
        loss, pred, mask = model(images)
        if pred is None or mask is None or not hasattr(model.model, "patchify"):
            batch_scores = loss.detach().cpu().numpy()
            if batch_scores.ndim == 0:
                batch_scores = np.repeat(batch_scores, images.size(0))
            scores.append(batch_scores)
            continue

        target = model.model.patchify(images)
        per_patch_loss = ((pred - target) ** 2).mean(dim=-1)
        mask = mask.float()
        per_image = (per_patch_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        scores.append(per_image.detach().cpu().numpy())

    return np.concatenate(scores, axis=0)

