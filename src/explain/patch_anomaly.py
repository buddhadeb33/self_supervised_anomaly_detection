from __future__ import annotations

import torch
import torch.nn.functional as F

from ..ssl import MAEWrapper


@torch.no_grad()
def mae_patch_anomaly_map(model: MAEWrapper, images: torch.Tensor) -> torch.Tensor:
    """
    Generate a patch-level anomaly heatmap from MAE reconstruction error.
    Returns a tensor of shape (B, H, W) normalized to [0,1].
    """
    model.eval()
    loss, pred, mask = model(images)
    if pred is None or mask is None or not hasattr(model.model, "patchify"):
        raise ValueError("MAE model does not expose patchify/pred for patch maps.")

    target = model.model.patchify(images)
    per_patch_loss = ((pred - target) ** 2).mean(dim=-1)
    grid_size = model.model.patch_embed.grid_size
    patch_h, patch_w = grid_size
    per_patch_loss = per_patch_loss.reshape(images.size(0), patch_h, patch_w)
    per_patch_loss = per_patch_loss.unsqueeze(1)
    heatmap = F.interpolate(
        per_patch_loss,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    heatmap = heatmap.squeeze(1)
    heatmap = (heatmap - heatmap.amin(dim=(-2, -1), keepdim=True)) / (
        heatmap.amax(dim=(-2, -1), keepdim=True) - heatmap.amin(dim=(-2, -1), keepdim=True) + 1e-8
    )
    return heatmap

