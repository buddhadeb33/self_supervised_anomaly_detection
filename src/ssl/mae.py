from __future__ import annotations

import torch
import torch.nn as nn


class MAEWrapper(nn.Module):
    def __init__(self, model_name: str = "mae_vit_base_patch16", mask_ratio: float = 0.75):
        super().__init__()
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("timm is required for MAE. Install via `pip install timm`.") from exc

        self.model = timm.create_model(model_name, pretrained=False)
        self.mask_ratio = mask_ratio

    def forward(self, images: torch.Tensor):
        outputs = self.model(images, mask_ratio=self.mask_ratio)
        if isinstance(outputs, tuple):
            loss = outputs[0]
            pred = outputs[1] if len(outputs) > 1 else None
            mask = outputs[2] if len(outputs) > 2 else None
            return loss, pred, mask
        return outputs, None, None

