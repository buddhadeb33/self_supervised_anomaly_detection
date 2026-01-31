from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .ssl import MAEWrapper


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        image: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> torch.Tensor:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        output = self.model(image)
        if output.ndim == 1:
            output = output.unsqueeze(0)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        score = output[:, class_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(0).squeeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


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

