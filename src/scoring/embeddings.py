from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..ssl import MoCoV2, SimCLR


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

