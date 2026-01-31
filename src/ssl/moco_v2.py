from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MoCoV2(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        projection_dim: int = 128,
        hidden_dim: int = 2048,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.2,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.momentum = momentum
        self.queue_size = queue_size

        self.encoder_q = self._build_encoder(backbone, projection_dim, hidden_dim)
        self.encoder_k = self._build_encoder(backbone, projection_dim, hidden_dim)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _build_encoder(self, backbone: str, projection_dim: int, hidden_dim: int) -> nn.Module:
        if backbone == "resnet50":
            base = models.resnet50(weights=None)
        elif backbone == "resnet18":
            base = models.resnet18(weights=None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        dim = base.fc.in_features
        base.fc = nn.Identity()
        projector = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )
        return nn.Sequential(base, projector)

    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1.0 - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor):
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        self._dequeue_and_enqueue(k)
        return logits, labels

