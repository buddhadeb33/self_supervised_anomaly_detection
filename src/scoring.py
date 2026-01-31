from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM

from .ssl import DINOv2Encoder, DINOv2PatchExtractor, MAEWrapper, MoCoV2, SimCLR


def load_ssl_encoder(
    checkpoint_path: Optional[str],
    method: str,
    backbone: str = "resnet50",
    projection_dim: int = 128,
    hidden_dim: int = 2048,
    use_projector: bool = False,
) -> torch.nn.Module:
    """
    Load an SSL encoder from a checkpoint or pretrained weights.

    Args:
        checkpoint_path: Path to checkpoint file. Not required for dinov2 methods.
        method: One of 'simclr', 'moco', 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'
        backbone: Backbone architecture for simclr/moco
        projection_dim: Projection dimension for simclr/moco
        hidden_dim: Hidden dimension for simclr/moco projector
        use_projector: Whether to include the projection head

    Returns:
        Encoder model
    """
    # DINOv2 models - no checkpoint needed, loaded from torch hub
    if method.startswith("dinov2"):
        return DINOv2Encoder(model_name=method, freeze=True, return_patch_tokens=False)

    # SimCLR
    if method == "simclr":
        model = SimCLR(backbone=backbone, projection_dim=projection_dim, hidden_dim=hidden_dim)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
        return model if use_projector else model.backbone

    # MoCo-v2
    if method == "moco":
        model = MoCoV2(
            backbone=backbone,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
        )
        if checkpoint_path:
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


def fit_knn(train_features: np.ndarray, k: int = 5) -> NearestNeighbors:
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    knn.fit(train_features)
    return knn


def score_knn(knn: NearestNeighbors, features: np.ndarray) -> np.ndarray:
    distances, _ = knn.kneighbors(features, return_distance=True)
    return distances.mean(axis=1)


def fit_mahalanobis(train_features: np.ndarray):
    mean = train_features.mean(axis=0)
    cov = np.cov(train_features, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    return mean, cov_inv


def score_mahalanobis(features: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    delta = features - mean
    scores = np.einsum("ij,jk,ik->i", delta, cov_inv, delta)
    return scores


def fit_one_class_svm(train_features: np.ndarray, nu: float = 0.1, gamma: str = "scale"):
    model = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    model.fit(train_features)
    return model


def score_one_class_svm(model: OneClassSVM, features: np.ndarray) -> np.ndarray:
    scores = -model.decision_function(features)
    return scores


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


# =============================================================================
# PatchCore Implementation
# =============================================================================


class PatchCoreFeatureExtractor(nn.Module):
    """
    Extract patch-level features from a CNN backbone (ResNet).
    Uses intermediate layer features for richer patch representations.
    """

    def __init__(
        self,
        backbone: nn.Module,
        layers: List[str] = ["layer2", "layer3"],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self._features: dict = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.backbone.named_modules():
            if name in self.layers:
                module.register_forward_hook(self._get_hook(name))

    def _get_hook(self, name: str):
        def hook(module, input, output):
            self._features[name] = output.detach()

        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._features = {}
        _ = self.backbone(x)

        # Collect features from specified layers
        feature_maps = []
        target_size = None
        for layer_name in self.layers:
            feat = self._features[layer_name]
            if target_size is None:
                target_size = feat.shape[-2:]
            # Upsample to match the largest feature map
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            feature_maps.append(feat)

        # Concatenate along channel dimension
        combined = torch.cat(feature_maps, dim=1)
        return combined


def _random_coreset_selection(
    features: np.ndarray,
    coreset_ratio: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """
    Random coreset selection for memory bank.
    For large datasets, this is faster than greedy k-center.
    """
    rng = np.random.default_rng(seed)
    n_samples = features.shape[0]
    n_coreset = max(1, int(n_samples * coreset_ratio))
    indices = rng.choice(n_samples, size=n_coreset, replace=False)
    return features[indices]


def _greedy_coreset_selection(
    features: np.ndarray,
    coreset_ratio: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """
    Greedy k-center coreset selection for memory bank.
    Provides better coverage than random selection.
    """
    rng = np.random.default_rng(seed)
    n_samples = features.shape[0]
    n_coreset = max(1, int(n_samples * coreset_ratio))

    if n_coreset >= n_samples:
        return features

    # Initialize with a random point
    selected_indices = [rng.integers(0, n_samples)]
    min_distances = np.full(n_samples, np.inf)

    for _ in range(n_coreset - 1):
        # Update minimum distances to selected set
        last_selected = features[selected_indices[-1]]
        distances = np.linalg.norm(features - last_selected, axis=1)
        min_distances = np.minimum(min_distances, distances)

        # Select the point with maximum minimum distance
        next_idx = np.argmax(min_distances)
        selected_indices.append(next_idx)

    return features[selected_indices]


@torch.no_grad()
def extract_patch_features(
    extractor: PatchCoreFeatureExtractor,
    dataloader: Iterable,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Extract patch features for PatchCore.
    Returns: (patch_features, labels, spatial_shape)
    - patch_features: (N * H * W, C) array of all patch features
    - labels: (N,) array of image labels
    - spatial_shape: (H, W) spatial dimensions of feature map
    """
    extractor.to(device)
    extractor.eval()

    all_features = []
    all_labels = []
    spatial_shape = None

    for batch in dataloader:
        images, batch_labels = batch[:2]
        images = images.to(device, non_blocking=True)
        feat_map = extractor(images)  # (B, C, H, W)

        if spatial_shape is None:
            spatial_shape = (feat_map.shape[2], feat_map.shape[3])

        # Reshape to (B, H*W, C)
        B, C, H, W = feat_map.shape
        feat_map = feat_map.permute(0, 2, 3, 1).reshape(B, H * W, C)
        feat_map = F.normalize(feat_map, dim=-1)

        all_features.append(feat_map.cpu().numpy())
        all_labels.append(batch_labels.numpy())

    features = np.concatenate(all_features, axis=0)  # (N, H*W, C)
    labels = np.concatenate(all_labels, axis=0)  # (N,)

    # Flatten to (N * H * W, C) for memory bank
    N, HW, C = features.shape
    features_flat = features.reshape(N * HW, C)

    return features_flat, labels, spatial_shape


class PatchCoreScorer:
    """
    PatchCore anomaly scorer using a memory bank of normal patch features.
    """

    def __init__(
        self,
        memory_bank: np.ndarray,
        k: int = 9,
    ) -> None:
        self.memory_bank = memory_bank
        self.k = k
        self.knn = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
        self.knn.fit(memory_bank)

    def score_patches(self, patch_features: np.ndarray) -> np.ndarray:
        """
        Score patch features using kNN distance to memory bank.
        Returns distances of shape (N_patches,).
        """
        distances, _ = self.knn.kneighbors(patch_features, return_distance=True)
        # Use the maximum of k-nearest distances (PatchCore uses k=1 typically)
        return distances[:, 0]

    def score_images(
        self,
        patch_features: np.ndarray,
        n_images: int,
        spatial_shape: Tuple[int, int],
        aggregation: str = "max",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score images by aggregating patch-level scores.
        Returns: (image_scores, patch_score_maps)
        - image_scores: (N,) array of image-level anomaly scores
        - patch_score_maps: (N, H, W) array of patch-level score maps
        """
        patch_scores = self.score_patches(patch_features)

        H, W = spatial_shape
        n_patches_per_image = H * W

        # Reshape to (N, H, W)
        patch_score_maps = patch_scores.reshape(n_images, H, W)

        # Aggregate to image-level score
        if aggregation == "max":
            image_scores = patch_score_maps.reshape(n_images, -1).max(axis=1)
        elif aggregation == "mean":
            image_scores = patch_score_maps.reshape(n_images, -1).mean(axis=1)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        return image_scores, patch_score_maps


def fit_patchcore(
    extractor: PatchCoreFeatureExtractor,
    dataloader: Iterable,
    device: torch.device,
    coreset_ratio: float = 0.1,
    coreset_method: str = "random",
    k: int = 9,
    seed: int = 42,
) -> Tuple[PatchCoreScorer, Tuple[int, int]]:
    """
    Fit PatchCore by building a memory bank from normal training patches.
    Returns: (scorer, spatial_shape)
    """
    print("[PatchCore] Extracting patch features from training data...")
    patch_features, _, spatial_shape = extract_patch_features(extractor, dataloader, device)

    print(f"[PatchCore] Extracted {patch_features.shape[0]} patches with dim {patch_features.shape[1]}")

    # Coreset selection
    print(f"[PatchCore] Selecting coreset with {coreset_method} method (ratio={coreset_ratio})...")
    if coreset_method == "random":
        memory_bank = _random_coreset_selection(patch_features, coreset_ratio, seed)
    elif coreset_method == "greedy":
        memory_bank = _greedy_coreset_selection(patch_features, coreset_ratio, seed)
    else:
        raise ValueError(f"Unknown coreset method: {coreset_method}")

    print(f"[PatchCore] Memory bank size: {memory_bank.shape[0]} patches")

    scorer = PatchCoreScorer(memory_bank, k=k)
    return scorer, spatial_shape


@torch.no_grad()
def score_patchcore(
    extractor: PatchCoreFeatureExtractor,
    scorer: PatchCoreScorer,
    dataloader: Iterable,
    device: torch.device,
    spatial_shape: Tuple[int, int],
    aggregation: str = "max",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Score test images using PatchCore.
    Returns: (image_scores, labels, patch_score_maps)
    """
    extractor.to(device)
    extractor.eval()

    all_patch_features = []
    all_labels = []

    for batch in dataloader:
        images, batch_labels = batch[:2]
        images = images.to(device, non_blocking=True)
        feat_map = extractor(images)  # (B, C, H, W)

        B, C, H, W = feat_map.shape
        feat_map = feat_map.permute(0, 2, 3, 1).reshape(B, H * W, C)
        feat_map = F.normalize(feat_map, dim=-1)

        all_patch_features.append(feat_map.cpu().numpy())
        all_labels.append(batch_labels.numpy())

    patch_features = np.concatenate(all_patch_features, axis=0)  # (N, H*W, C)
    labels = np.concatenate(all_labels, axis=0)

    N, HW, C = patch_features.shape
    patch_features_flat = patch_features.reshape(N * HW, C)

    image_scores, patch_score_maps = scorer.score_images(
        patch_features_flat, N, spatial_shape, aggregation
    )

    return image_scores, labels, patch_score_maps


# =============================================================================
# Dual-Space Scoring (Global + Patch)
# =============================================================================


def dual_space_score(
    global_scores: np.ndarray,
    patch_scores: np.ndarray,
    alpha: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """
    Combine global and patch-level anomaly scores.

    Args:
        global_scores: (N,) array of global embedding-based scores
        patch_scores: (N,) array of patch-level scores (e.g., max of patch map)
        alpha: weight for global scores (1-alpha for patch scores)
        normalize: whether to normalize scores to [0, 1] before combining

    Returns:
        (N,) array of combined scores
    """
    if normalize:
        # Min-max normalization
        global_min, global_max = global_scores.min(), global_scores.max()
        if global_max > global_min:
            global_scores = (global_scores - global_min) / (global_max - global_min)

        patch_min, patch_max = patch_scores.min(), patch_scores.max()
        if patch_max > patch_min:
            patch_scores = (patch_scores - patch_min) / (patch_max - patch_min)

    return alpha * global_scores + (1 - alpha) * patch_scores


# =============================================================================
# Diffusion-Based Anomaly Scoring
# =============================================================================


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule for diffusion beta values.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def _linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear schedule for diffusion beta values.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


class SimpleDiffusionUNet(nn.Module):
    """
    Simplified U-Net for diffusion-based anomaly detection.
    Designed to be lightweight for medical imaging experiments.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),
        time_emb_dim: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        ch = base_channels
        self.init_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        prev_ch = ch
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.GroupNorm(8, prev_ch),
                    nn.SiLU(),
                    nn.Conv2d(prev_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                )
            )
            self.downsample.append(nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1))
            prev_ch = out_ch

        # Middle
        self.middle = nn.Sequential(
            nn.GroupNorm(8, prev_ch),
            nn.SiLU(),
            nn.Conv2d(prev_ch, prev_ch, 3, padding=1),
            nn.GroupNorm(8, prev_ch),
            nn.SiLU(),
            nn.Conv2d(prev_ch, prev_ch, 3, padding=1),
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.upsample.append(nn.ConvTranspose2d(prev_ch, prev_ch, 4, stride=2, padding=1))
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.GroupNorm(8, prev_ch + out_ch),  # Skip connection
                    nn.SiLU(),
                    nn.Conv2d(prev_ch + out_ch, out_ch, 3, padding=1),
                    nn.GroupNorm(8, out_ch),
                    nn.SiLU(),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                )
            )
            prev_ch = out_ch

        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, prev_ch),
            nn.SiLU(),
            nn.Conv2d(prev_ch, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Noisy image (B, C, H, W)
            t: Timesteps (B,)

        Returns:
            Predicted noise (B, C, H, W)
        """
        # Time embedding
        t_emb = self.time_mlp(t.float().unsqueeze(-1))  # (B, time_emb_dim)

        # Initial convolution
        h = self.init_conv(x)

        # Encoder
        skip_connections = []
        for block, down in zip(self.encoder_blocks, self.downsample):
            h = block(h)
            skip_connections.append(h)
            h = down(h)

        # Middle
        h = self.middle(h)

        # Decoder
        for up, block, skip in zip(self.upsample, self.decoder_blocks, reversed(skip_connections)):
            h = up(h)
            # Handle size mismatch
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h)

        return self.out_conv(h)


class DiffusionAnomalyScorer(nn.Module):
    """
    Diffusion-based anomaly scorer.
    Trains a denoising diffusion model on normal images and uses
    reconstruction error as an anomaly score.

    Based on: "Diffusion Models for Medical Anomaly Detection" and
    "AnoDDPM: Anomaly Detection with Denoising Diffusion Probabilistic Models"
    """

    def __init__(
        self,
        image_size: int = 224,
        in_channels: int = 3,
        timesteps: int = 1000,
        beta_schedule: str = "cosine",
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.timesteps = timesteps

        # Beta schedule
        if beta_schedule == "cosine":
            betas = _cosine_beta_schedule(timesteps)
        else:
            betas = _linear_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

        # Denoising model
        self.model = SimpleDiffusionUNet(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=base_channels,
        )

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: add noise to x_0 at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute training loss for denoising.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        x_noisy = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_noisy, t)

        loss = F.mse_loss(predicted_noise, noise, reduction="none")
        return loss.mean(dim=[1, 2, 3])  # Per-image loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass: sample random timesteps and compute loss.
        """
        B = x.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=x.device)
        return self.p_losses(x, t).mean()

    @torch.no_grad()
    def denoise_step(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Single denoising step.
        """
        predicted_noise = self.model(x, t)

        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)

        # Mean
        model_mean = sqrt_recip_alpha_t * (
            x - (1 - alpha_t) / sqrt_one_minus_alpha_cumprod_t * predicted_noise
        )

        # Add noise if not final step
        if t[0] > 0:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        return model_mean

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples by iterative denoising.
        """
        x = torch.randn(n_samples, self.in_channels, self.image_size, self.image_size, device=device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
            x = self.denoise_step(x, t_batch)

        return x

    @torch.no_grad()
    def compute_reconstruction_error(
        self,
        x: torch.Tensor,
        n_steps: int = 250,
        start_t: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute reconstruction error for anomaly scoring.

        Args:
            x: Input images (B, C, H, W)
            n_steps: Number of denoising steps for reconstruction
            start_t: Starting timestep for noising (higher = more noise)

        Returns:
            Tuple of (reconstruction_error, reconstructed_images)
        """
        if start_t is None:
            start_t = self.timesteps // 4  # Add moderate noise

        B = x.shape[0]

        # Add noise at timestep start_t
        t = torch.full((B,), start_t, device=x.device, dtype=torch.long)
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)

        # Denoise
        x_recon = x_noisy
        step_size = max(1, start_t // n_steps)
        for timestep in range(start_t, -1, -step_size):
            t_batch = torch.full((B,), timestep, device=x.device, dtype=torch.long)
            x_recon = self.denoise_step(x_recon, t_batch)

        # Compute reconstruction error
        error = F.mse_loss(x_recon, x, reduction="none")
        error = error.mean(dim=[1, 2, 3])  # Per-image error

        return error, x_recon

    @torch.no_grad()
    def compute_patch_reconstruction_error(
        self,
        x: torch.Tensor,
        patch_size: int = 16,
        n_steps: int = 250,
        start_t: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute patch-level reconstruction error for localization.

        Returns:
            Tuple of (patch_errors, reconstructed_images)
            patch_errors: (B, H', W') tensor of per-patch errors
        """
        error, x_recon = self.compute_reconstruction_error(x, n_steps, start_t)

        # Compute per-pixel error
        pixel_error = (x - x_recon) ** 2
        pixel_error = pixel_error.mean(dim=1, keepdim=True)  # (B, 1, H, W)

        # Average over patches
        B, _, H, W = pixel_error.shape
        h_patches = H // patch_size
        w_patches = W // patch_size

        # Reshape to patches and average
        pixel_error = pixel_error[:, :, :h_patches * patch_size, :w_patches * patch_size]
        patch_error = F.avg_pool2d(pixel_error, kernel_size=patch_size)
        patch_error = patch_error.squeeze(1)  # (B, h_patches, w_patches)

        return patch_error, x_recon


def train_diffusion_scorer(
    model: DiffusionAnomalyScorer,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 50,
    log_every: int = 50,
    output_dir: str = "checkpoints",
) -> None:
    """
    Train the diffusion-based anomaly scorer.
    """
    import os

    model.to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for step, batch in enumerate(dataloader, start=1):
            images = batch[0]
            images = images.to(device, non_blocking=True)

            loss = model(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % log_every == 0:
                avg = running_loss / log_every
                print(f"[Diffusion] epoch={epoch} step={step} loss={avg:.6f}")
                running_loss = 0.0

        # Save checkpoint
        os.makedirs(output_dir, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(output_dir, f"diffusion_epoch_{epoch}.pt"))


@torch.no_grad()
def score_diffusion(
    model: DiffusionAnomalyScorer,
    dataloader: Iterable,
    device: torch.device,
    n_steps: int = 250,
    start_t: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score images using diffusion-based reconstruction error.

    Returns:
        Tuple of (scores, labels)
    """
    model.to(device)
    model.eval()

    all_scores = []
    all_labels = []

    for batch in dataloader:
        images, labels = batch[:2]
        images = images.to(device, non_blocking=True)

        errors, _ = model.compute_reconstruction_error(images, n_steps, start_t)

        all_scores.append(errors.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


@torch.no_grad()
def score_diffusion_with_patches(
    model: DiffusionAnomalyScorer,
    dataloader: Iterable,
    device: torch.device,
    patch_size: int = 16,
    n_steps: int = 250,
    start_t: Optional[int] = None,
    aggregation: str = "max",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Score images using diffusion-based patch reconstruction error.

    Returns:
        Tuple of (image_scores, labels, patch_score_maps)
    """
    model.to(device)
    model.eval()

    all_scores = []
    all_labels = []
    all_patch_maps = []

    for batch in dataloader:
        images, labels = batch[:2]
        images = images.to(device, non_blocking=True)

        patch_errors, _ = model.compute_patch_reconstruction_error(
            images, patch_size, n_steps, start_t
        )

        # Aggregate to image-level score
        B = patch_errors.shape[0]
        if aggregation == "max":
            image_scores = patch_errors.view(B, -1).max(dim=1)[0]
        else:
            image_scores = patch_errors.view(B, -1).mean(dim=1)

        all_scores.append(image_scores.cpu().numpy())
        all_labels.append(labels.numpy())
        all_patch_maps.append(patch_errors.cpu().numpy())

    return (
        np.concatenate(all_scores),
        np.concatenate(all_labels),
        np.concatenate(all_patch_maps),
    )

