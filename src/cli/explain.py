import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms as T

from ..explain import GradCAM, mae_patch_anomaly_map
from ..ssl import MAEWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate explainability maps.")
    parser.add_argument("--method", choices=["gradcam", "mae"], required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--target-layer", default="layer4")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_image(path: str) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    image = _load_image(args.image_path).to(device)

    if args.method == "gradcam":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(ckpt["model"])
        model.to(device)
        target_layer = getattr(model, args.target_layer)
        cam = GradCAM(model, target_layer).generate(image)
        np.save(args.output, cam.detach().cpu().numpy())
    else:
        model = MAEWrapper()
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for MAE explainability.")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(device)
        heatmap = mae_patch_anomaly_map(model, image)
        np.save(args.output, heatmap.detach().cpu().numpy())


if __name__ == "__main__":
    main()

