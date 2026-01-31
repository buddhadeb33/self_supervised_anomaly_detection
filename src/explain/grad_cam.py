from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


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

