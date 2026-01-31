from __future__ import annotations

import os
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Normalized temperature-scaled cross entropy loss for SimCLR.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)

    sim = torch.matmul(z, z.T) / temperature
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, float("-inf"))

    labels = (torch.arange(2 * batch_size, device=z.device) + batch_size) % (
        2 * batch_size
    )
    loss = F.cross_entropy(sim, labels)
    return loss


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


def _extract_views(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    images = batch[0]
    if isinstance(images, (tuple, list)) and len(images) == 2:
        return images[0], images[1]
    raise ValueError("Batch does not contain two augmented views.")


def _save_checkpoint(model, optimizer, epoch: int, output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(ckpt, os.path.join(output_dir, f"{name}_epoch_{epoch}.pt"))


def train_simclr(
    model,
    dataloader: Iterable,
    optimizer,
    device: torch.device,
    epochs: int = 100,
    log_every: int = 50,
    output_dir: str = "checkpoints",
) -> None:
    model.to(device)
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            x1, x2 = _extract_views(batch)
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            z1 = model(x1)
            z2 = model(x2)
            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % log_every == 0:
                avg = running_loss / log_every
                print(f"[SimCLR] epoch={epoch} step={step} loss={avg:.4f}")
                running_loss = 0.0

        _save_checkpoint(model, optimizer, epoch, output_dir, "simclr")


def train_moco(
    model,
    dataloader: Iterable,
    optimizer,
    device: torch.device,
    epochs: int = 100,
    log_every: int = 50,
    output_dir: str = "checkpoints",
) -> None:
    model.to(device)
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            x1, x2 = _extract_views(batch)
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            logits, labels = model(x1, x2)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % log_every == 0:
                avg = running_loss / log_every
                print(f"[MoCo] epoch={epoch} step={step} loss={avg:.4f}")
                running_loss = 0.0

        _save_checkpoint(model, optimizer, epoch, output_dir, "moco")


def train_mae(
    model,
    dataloader: Iterable,
    optimizer,
    device: torch.device,
    epochs: int = 100,
    log_every: int = 50,
    output_dir: str = "checkpoints",
) -> None:
    model.to(device)
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            images = batch[0]
            images = images.to(device, non_blocking=True)
            loss, _, _ = model(images)
            if not torch.is_tensor(loss):
                loss = loss[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % log_every == 0:
                avg = running_loss / log_every
                print(f"[MAE] epoch={epoch} step={step} loss={avg:.4f}")
                running_loss = 0.0

        _save_checkpoint(model, optimizer, epoch, output_dir, "mae")


# =============================================================================
# DINOv2 Frozen Encoder
# =============================================================================


class DINOv2Encoder(nn.Module):
    """
    DINOv2 frozen encoder for feature extraction.
    Uses pretrained DINOv2 models from torch hub.

    Supported model sizes:
    - dinov2_vits14: ViT-S/14 (21M params, 384 dim)
    - dinov2_vitb14: ViT-B/14 (86M params, 768 dim)
    - dinov2_vitl14: ViT-L/14 (300M params, 1024 dim)
    - dinov2_vitg14: ViT-G/14 (1.1B params, 1536 dim) - requires significant memory
    """

    SUPPORTED_MODELS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        freeze: bool = True,
        return_patch_tokens: bool = False,
    ) -> None:
        """
        Args:
            model_name: One of dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
            freeze: Whether to freeze the model weights (default: True)
            return_patch_tokens: If True, return patch tokens instead of CLS token
        """
        super().__init__()
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.embed_dim = self.SUPPORTED_MODELS[model_name]
        self.return_patch_tokens = return_patch_tokens

        # Load from torch hub
        self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            x: Input images of shape (B, 3, H, W). H and W should be divisible by 14.

        Returns:
            If return_patch_tokens=False: CLS token features of shape (B, embed_dim)
            If return_patch_tokens=True: Patch tokens of shape (B, N_patches, embed_dim)
        """
        if self.return_patch_tokens:
            # Get patch tokens (excluding CLS)
            features = self.model.forward_features(x)
            if isinstance(features, dict):
                # Some versions return a dict
                patch_tokens = features.get("x_norm_patchtokens", features.get("x_patchtokens"))
                if patch_tokens is None:
                    # Fallback: get all tokens and exclude CLS
                    all_tokens = features.get("x_norm_clstoken", features.get("x"))
                    if all_tokens is not None and all_tokens.dim() == 3:
                        patch_tokens = all_tokens[:, 1:, :]  # Exclude CLS token
                    else:
                        patch_tokens = self.model(x)
            else:
                # Tensor output: (B, N+1, D) where first token is CLS
                patch_tokens = features[:, 1:, :]
            return patch_tokens
        else:
            # Get CLS token (global feature)
            return self.model(x)

    @property
    def output_dim(self) -> int:
        return self.embed_dim


class DINOv2PatchExtractor(nn.Module):
    """
    Extract patch-level features from DINOv2 for PatchCore-style anomaly detection.
    Reshapes patch tokens to spatial feature maps.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = DINOv2Encoder(
            model_name=model_name,
            freeze=freeze,
            return_patch_tokens=True,
        )
        self.patch_size = 14  # DINOv2 uses 14x14 patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature map from images.

        Args:
            x: Input images of shape (B, 3, H, W)

        Returns:
            Feature map of shape (B, embed_dim, H', W') where H'=H/14, W'=W/14
        """
        B, C, H, W = x.shape
        patch_tokens = self.encoder(x)  # (B, N_patches, embed_dim)

        # Compute spatial dimensions
        h = H // self.patch_size
        w = W // self.patch_size

        # Reshape to spatial feature map
        # patch_tokens: (B, h*w, D) -> (B, D, h, w)
        feat_map = patch_tokens.permute(0, 2, 1).reshape(B, -1, h, w)
        return feat_map

    @property
    def output_dim(self) -> int:
        return self.encoder.output_dim


def load_dinov2_encoder(
    model_name: str = "dinov2_vitb14",
    return_patch_tokens: bool = False,
) -> DINOv2Encoder:
    """
    Convenience function to load a pretrained DINOv2 encoder.

    Args:
        model_name: One of dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
        return_patch_tokens: If True, return patch tokens instead of CLS token

    Returns:
        Frozen DINOv2 encoder
    """
    return DINOv2Encoder(
        model_name=model_name,
        freeze=True,
        return_patch_tokens=return_patch_tokens,
    )

