from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from ..ssl import MAEWrapper


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

