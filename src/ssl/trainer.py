from __future__ import annotations

import os
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F

from .losses import nt_xent_loss


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

