from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torchvision import transforms as T


class TwoCropsTransform:
    def __init__(self, base_transform: Callable) -> None:
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


def get_ssl_transform(image_size: int = 224) -> Callable:
    return T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def get_mae_transform(image_size: int = 224) -> Callable:
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


# =============================================================================
# CutPaste Synthetic Anomaly Augmentation
# =============================================================================


class CutPasteTransform:
    """
    CutPaste augmentation for self-supervised anomaly detection.
    Generates synthetic anomalies by cutting and pasting image patches.

    Based on: "CutPaste: Self-Supervised Learning for Anomaly Detection and Localization"
    (Li et al., CVPR 2021)

    This implementation adapts CutPaste for medical imaging with:
    - Anatomically-aware patch sizes (configurable for chest X-rays)
    - Multiple paste modes (normal, scar, perlin noise)
    - Optional blending for more realistic anomalies
    """

    def __init__(
        self,
        patch_ratio_range: Tuple[float, float] = (0.02, 0.15),
        aspect_ratio_range: Tuple[float, float] = (0.3, 3.3),
        rotation_range: Tuple[float, float] = (-45, 45),
        mode: str = "cutpaste",
        blend_alpha: float = 0.0,
        jitter_strength: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            patch_ratio_range: Range of patch area as ratio of image area
            aspect_ratio_range: Range of patch aspect ratios
            rotation_range: Range of patch rotation angles in degrees
            mode: 'cutpaste' (standard), 'scar' (thin linear patches), or 'perlin' (noise)
            blend_alpha: Alpha for blending (0 = hard paste, 1 = fully transparent)
            jitter_strength: Color jitter strength for pasted patches
            seed: Random seed for reproducibility
        """
        self.patch_ratio_range = patch_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.rotation_range = rotation_range
        self.mode = mode
        self.blend_alpha = blend_alpha
        self.jitter_strength = jitter_strength
        self.rng = np.random.default_rng(seed)

        # Color jitter for pasted patches
        self.color_jitter = T.ColorJitter(
            brightness=jitter_strength,
            contrast=jitter_strength,
            saturation=jitter_strength * 0.5,
            hue=jitter_strength * 0.1,
        )

    def _get_patch_params(
        self, img_w: int, img_h: int
    ) -> Tuple[int, int, int, int, float]:
        """
        Sample patch parameters (x, y, w, h, rotation).
        """
        img_area = img_w * img_h

        # Sample patch area
        patch_ratio = self.rng.uniform(*self.patch_ratio_range)
        patch_area = int(img_area * patch_ratio)

        # Sample aspect ratio
        aspect_ratio = self.rng.uniform(*self.aspect_ratio_range)

        # Compute patch dimensions
        patch_h = int(np.sqrt(patch_area / aspect_ratio))
        patch_w = int(patch_area / patch_h)

        # Clamp to image size
        patch_h = min(patch_h, img_h - 1)
        patch_w = min(patch_w, img_w - 1)

        # Sample position (ensure patch fits in image)
        x = self.rng.integers(0, img_w - patch_w)
        y = self.rng.integers(0, img_h - patch_h)

        # Sample rotation
        rotation = self.rng.uniform(*self.rotation_range)

        return x, y, patch_w, patch_h, rotation

    def _get_scar_params(
        self, img_w: int, img_h: int
    ) -> Tuple[int, int, int, int, float]:
        """
        Sample scar-like patch parameters (thin, elongated patches).
        """
        # Scar-like patches are thin and elongated
        scar_length = self.rng.integers(int(img_w * 0.1), int(img_w * 0.4))
        scar_width = self.rng.integers(2, max(3, int(img_w * 0.03)))

        x = self.rng.integers(0, img_w - scar_width)
        y = self.rng.integers(0, img_h - scar_length)

        rotation = self.rng.uniform(-90, 90)

        return x, y, scar_width, scar_length, rotation

    def _apply_cutpaste(self, image: Image.Image) -> Image.Image:
        """
        Apply standard CutPaste augmentation.
        """
        img_w, img_h = image.size

        # Get source patch
        src_x, src_y, patch_w, patch_h, rotation = self._get_patch_params(img_w, img_h)
        patch = image.crop((src_x, src_y, src_x + patch_w, src_y + patch_h))

        # Apply transforms to patch
        patch = patch.rotate(rotation, expand=False, resample=Image.BILINEAR)
        patch = self.color_jitter(patch)

        # Get destination position (different from source)
        max_attempts = 10
        for _ in range(max_attempts):
            dst_x = self.rng.integers(0, max(1, img_w - patch_w))
            dst_y = self.rng.integers(0, max(1, img_h - patch_h))
            # Ensure some distance from source
            if abs(dst_x - src_x) > patch_w // 2 or abs(dst_y - src_y) > patch_h // 2:
                break

        # Create output image
        output = image.copy()

        # Blend or paste
        if self.blend_alpha > 0:
            # Create alpha mask
            mask = Image.new("L", patch.size, int(255 * (1 - self.blend_alpha)))
            output.paste(patch, (dst_x, dst_y), mask)
        else:
            output.paste(patch, (dst_x, dst_y))

        return output

    def _apply_scar(self, image: Image.Image) -> Image.Image:
        """
        Apply scar-like CutPaste augmentation.
        """
        img_w, img_h = image.size

        # Get scar parameters
        src_x, src_y, scar_w, scar_h, rotation = self._get_scar_params(img_w, img_h)

        # Extract scar patch
        patch = image.crop((src_x, src_y, src_x + scar_w, src_y + scar_h))
        patch = patch.rotate(rotation, expand=True, resample=Image.BILINEAR)
        patch = self.color_jitter(patch)

        # Get destination
        new_w, new_h = patch.size
        dst_x = self.rng.integers(0, max(1, img_w - new_w))
        dst_y = self.rng.integers(0, max(1, img_h - new_h))

        # Paste
        output = image.copy()
        output.paste(patch, (dst_x, dst_y))

        return output

    def _apply_perlin_noise(self, image: Image.Image) -> Image.Image:
        """
        Apply Perlin noise-based synthetic anomaly.
        Creates a localized noise pattern that simulates texture anomalies.
        """
        img_w, img_h = image.size

        # Get patch location
        x, y, patch_w, patch_h, _ = self._get_patch_params(img_w, img_h)

        # Generate noise pattern
        noise = self.rng.random((patch_h, patch_w)) * 2 - 1

        # Smooth the noise
        noise_img = Image.fromarray((noise * 127 + 128).astype(np.uint8))
        noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=2))

        # Apply noise to patch region
        output = image.copy()
        patch = image.crop((x, y, x + patch_w, y + patch_h))

        # Convert to numpy for blending
        patch_np = np.array(patch).astype(float)
        noise_np = np.array(noise_img).astype(float)

        if patch_np.ndim == 3:
            noise_np = noise_np[:, :, np.newaxis]

        # Blend noise with patch
        blend_factor = self.rng.uniform(0.2, 0.5)
        blended = patch_np * (1 - blend_factor) + noise_np * blend_factor
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Create new patch image
        if patch_np.ndim == 3:
            new_patch = Image.fromarray(blended, mode="RGB")
        else:
            new_patch = Image.fromarray(blended, mode="L")

        output.paste(new_patch, (x, y))

        return output

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply CutPaste augmentation to an image.
        """
        if self.mode == "cutpaste":
            return self._apply_cutpaste(image)
        elif self.mode == "scar":
            return self._apply_scar(image)
        elif self.mode == "perlin":
            return self._apply_perlin_noise(image)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class CutPasteSSLTransform:
    """
    Transform that creates pairs of (normal, anomalous) images for SSL training.
    Returns (original, cutpaste_augmented) for contrastive learning.
    """

    def __init__(
        self,
        base_transform: Optional[Callable] = None,
        cutpaste_modes: Tuple[str, ...] = ("cutpaste", "scar"),
        patch_ratio_range: Tuple[float, float] = (0.02, 0.15),
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            base_transform: Transform to apply to both images (e.g., resize, normalize)
            cutpaste_modes: Tuple of CutPaste modes to randomly choose from
            patch_ratio_range: Range of patch sizes
            seed: Random seed
        """
        self.base_transform = base_transform
        self.cutpaste_modes = cutpaste_modes
        self.rng = np.random.default_rng(seed)

        # Create CutPaste transforms for each mode
        self.cutpaste_transforms = {
            mode: CutPasteTransform(
                patch_ratio_range=patch_ratio_range,
                mode=mode,
                seed=seed,
            )
            for mode in cutpaste_modes
        }

    def __call__(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Apply transform and return (normal, augmented, label).
        Label: 0 = normal, 1 = cutpaste anomaly, 2 = scar anomaly, etc.
        """
        # Randomly select anomaly type (or keep normal)
        if self.rng.random() < 0.5:
            # Return normal image
            anomaly_label = 0
            augmented = image
        else:
            # Apply random CutPaste mode
            mode = self.rng.choice(self.cutpaste_modes)
            mode_idx = self.cutpaste_modes.index(mode)
            anomaly_label = mode_idx + 1
            augmented = self.cutpaste_transforms[mode](image)

        # Apply base transform
        if self.base_transform:
            original = self.base_transform(image)
            augmented = self.base_transform(augmented)
        else:
            original = T.ToTensor()(image)
            augmented = T.ToTensor()(augmented)

        return original, augmented, anomaly_label


class CutPasteThreeCropsTransform:
    """
    Transform that creates three views: original, CutPaste, and CutPaste-Scar.
    Useful for contrastive learning with multiple anomaly types.
    """

    def __init__(
        self,
        base_transform: Optional[Callable] = None,
        image_size: int = 224,
        seed: Optional[int] = None,
    ) -> None:
        self.image_size = image_size
        self.seed = seed

        if base_transform is None:
            self.base_transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.base_transform = base_transform

        self.cutpaste = CutPasteTransform(mode="cutpaste", seed=seed)
        self.scar = CutPasteTransform(mode="scar", seed=seed)

    def __call__(
        self, image: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (original, cutpaste, scar) tensors.
        """
        cutpaste_img = self.cutpaste(image)
        scar_img = self.scar(image)

        return (
            self.base_transform(image),
            self.base_transform(cutpaste_img),
            self.base_transform(scar_img),
        )


def get_cutpaste_transform(
    image_size: int = 224,
    mode: str = "cutpaste",
    include_normal: bool = True,
) -> Callable:
    """
    Get a CutPaste transform for anomaly-aware SSL training.

    Args:
        image_size: Output image size
        mode: CutPaste mode ('cutpaste', 'scar', 'perlin')
        include_normal: If True, randomly returns normal or augmented images

    Returns:
        Transform callable
    """
    base = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    cutpaste = CutPasteTransform(mode=mode)

    if include_normal:
        def transform(image):
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                return base(image), 0  # normal
            else:
                return base(cutpaste(image)), 1  # anomaly
        return transform
    else:
        def transform(image):
            return base(cutpaste(image))
        return transform


# =============================================================================
# Anatomically-Aware Augmentations for Chest X-rays
# =============================================================================


class AnatomicalCutPaste(CutPasteTransform):
    """
    CutPaste augmentation adapted for chest X-rays.
    Focuses on lung regions and uses medically-plausible anomaly patterns.
    """

    def __init__(
        self,
        lung_region: Tuple[float, float, float, float] = (0.1, 0.15, 0.9, 0.85),
        nodule_size_range: Tuple[float, float] = (0.01, 0.05),
        opacity_range: Tuple[float, float] = (0.02, 0.12),
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            lung_region: Approximate lung region as (x1, y1, x2, y2) ratios
            nodule_size_range: Size range for nodule-like anomalies
            opacity_range: Size range for opacity-like anomalies
            seed: Random seed
        """
        super().__init__(seed=seed)
        self.lung_region = lung_region
        self.nodule_size_range = nodule_size_range
        self.opacity_range = opacity_range

    def _get_lung_position(self, img_w: int, img_h: int) -> Tuple[int, int]:
        """
        Sample a position within the lung region.
        """
        x1 = int(img_w * self.lung_region[0])
        y1 = int(img_h * self.lung_region[1])
        x2 = int(img_w * self.lung_region[2])
        y2 = int(img_h * self.lung_region[3])

        x = self.rng.integers(x1, x2)
        y = self.rng.integers(y1, y2)

        return x, y

    def _create_nodule(self, image: Image.Image) -> Image.Image:
        """
        Create a nodule-like synthetic anomaly.
        """
        img_w, img_h = image.size
        output = image.copy()

        # Nodule size
        size_ratio = self.rng.uniform(*self.nodule_size_range)
        size = int(min(img_w, img_h) * size_ratio)

        # Position in lung region
        x, y = self._get_lung_position(img_w, img_h)

        # Create circular mask
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([0, 0, size - 1, size - 1], fill=255)

        # Apply Gaussian blur to mask for soft edges
        mask = mask.filter(ImageFilter.GaussianBlur(radius=size // 4))

        # Extract patch and modify intensity
        x1, y1 = max(0, x - size // 2), max(0, y - size // 2)
        x2, y2 = min(img_w, x1 + size), min(img_h, y1 + size)

        if x2 - x1 > 0 and y2 - y1 > 0:
            patch = image.crop((x1, y1, x2, y2))
            patch_np = np.array(patch).astype(float)

            # Reduce intensity (simulate nodule)
            intensity_factor = self.rng.uniform(0.3, 0.7)
            patch_np = patch_np * intensity_factor

            # Resize mask if needed
            actual_size = (x2 - x1, y2 - y1)
            if actual_size != (size, size):
                mask = mask.resize(actual_size, Image.BILINEAR)

            mask_np = np.array(mask).astype(float) / 255.0
            if patch_np.ndim == 3:
                mask_np = mask_np[:, :, np.newaxis]

            # Blend
            original_patch = np.array(image.crop((x1, y1, x2, y2))).astype(float)
            blended = original_patch * (1 - mask_np) + patch_np * mask_np
            blended = np.clip(blended, 0, 255).astype(np.uint8)

            output.paste(Image.fromarray(blended), (x1, y1))

        return output

    def _create_opacity(self, image: Image.Image) -> Image.Image:
        """
        Create an opacity-like synthetic anomaly (ground-glass or consolidation).
        """
        img_w, img_h = image.size
        output = image.copy()

        # Opacity size (larger than nodules)
        size_ratio = self.rng.uniform(*self.opacity_range)
        size_w = int(img_w * size_ratio)
        size_h = int(img_h * size_ratio * self.rng.uniform(0.7, 1.3))

        # Position in lung region
        x, y = self._get_lung_position(img_w, img_h)

        # Create irregular mask using noise
        mask = np.zeros((size_h, size_w), dtype=np.float32)
        center = (size_w // 2, size_h // 2)
        for i in range(size_h):
            for j in range(size_w):
                dist = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
                max_dist = np.sqrt(center[0] ** 2 + center[1] ** 2)
                mask[i, j] = max(0, 1 - dist / max_dist)

        # Add noise to mask
        noise = self.rng.random((size_h, size_w)) * 0.3
        mask = mask * (1 + noise - 0.15)
        mask = np.clip(mask, 0, 1)

        # Blur mask
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=max(1, size_w // 6)))
        mask = np.array(mask_img).astype(float) / 255.0

        # Apply opacity effect
        x1, y1 = max(0, x - size_w // 2), max(0, y - size_h // 2)
        x2, y2 = min(img_w, x1 + size_w), min(img_h, y1 + size_h)

        if x2 - x1 > 0 and y2 - y1 > 0:
            patch = image.crop((x1, y1, x2, y2))
            patch_np = np.array(patch).astype(float)

            # Increase intensity (simulate opacity/consolidation)
            intensity_factor = self.rng.uniform(1.1, 1.4)
            opacity_patch = patch_np * intensity_factor

            # Resize mask if needed
            actual_h, actual_w = y2 - y1, x2 - x1
            if (actual_w, actual_h) != (size_w, size_h):
                mask_resized = Image.fromarray((mask * 255).astype(np.uint8))
                mask_resized = mask_resized.resize((actual_w, actual_h), Image.BILINEAR)
                mask = np.array(mask_resized).astype(float) / 255.0

            if patch_np.ndim == 3:
                mask = mask[:, :, np.newaxis]

            # Blend
            blended = patch_np * (1 - mask) + opacity_patch * mask
            blended = np.clip(blended, 0, 255).astype(np.uint8)

            output.paste(Image.fromarray(blended), (x1, y1))

        return output

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Apply anatomically-aware CutPaste augmentation.
        """
        choice = self.rng.choice(["nodule", "opacity", "cutpaste", "scar"])

        if choice == "nodule":
            return self._create_nodule(image)
        elif choice == "opacity":
            return self._create_opacity(image)
        elif choice == "scar":
            return self._apply_scar(image)
        else:
            return self._apply_cutpaste(image)

