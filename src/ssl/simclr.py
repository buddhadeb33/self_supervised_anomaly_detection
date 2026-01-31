import torch
import torch.nn as nn
from torchvision import models


class SimCLR(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        projection_dim: int = 128,
        hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        if backbone == "resnet50":
            base = models.resnet50(weights=None)
        elif backbone == "resnet18":
            base = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.projector = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        proj = self.projector(feats)
        return proj

