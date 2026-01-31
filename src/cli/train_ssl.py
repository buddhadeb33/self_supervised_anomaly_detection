import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..augment import TwoCropsTransform, get_mae_transform, get_ssl_transform
from ..data import NIHChestXrayDataset
from ..ssl import MAEWrapper, MoCoV2, SimCLR, train_mae, train_moco, train_simclr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SSL baselines on NIH ChestXray14.")
    parser.add_argument("--method", choices=["simclr", "moco", "mae"], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--mask-ratio", type=float, default=0.75)
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(args.train_csv)

    if args.method in {"simclr", "moco"}:
        transform = TwoCropsTransform(get_ssl_transform(args.image_size))
    else:
        transform = get_mae_transform(args.image_size)

    dataset = NIHChestXrayDataset(train_df, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    if args.method == "simclr":
        model = SimCLR(
            backbone=args.backbone,
            projection_dim=args.projection_dim,
            hidden_dim=args.hidden_dim,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train_simclr(model, loader, optimizer, device, args.epochs, output_dir=args.output_dir)
    elif args.method == "moco":
        model = MoCoV2(
            backbone=args.backbone,
            projection_dim=args.projection_dim,
            hidden_dim=args.hidden_dim,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train_moco(model, loader, optimizer, device, args.epochs, output_dir=args.output_dir)
    else:
        model = MAEWrapper(mask_ratio=args.mask_ratio)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        train_mae(model, loader, optimizer, device, args.epochs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()

