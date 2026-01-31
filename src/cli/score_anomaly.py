import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..augment import get_mae_transform
from ..data import NIHChestXrayDataset
from ..scoring import (
    extract_embeddings,
    fit_knn,
    fit_mahalanobis,
    fit_one_class_svm,
    load_ssl_encoder,
    mae_reconstruction_scores,
    score_knn,
    score_mahalanobis,
    score_one_class_svm,
)
from ..ssl import MAEWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute anomaly scores for NIH ChestXray14.")
    parser.add_argument("--method", choices=["knn", "mahalanobis", "ocsvm", "mae"], required=True)
    parser.add_argument("--train-csv", help="Normal-only train CSV for fitting.")
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--checkpoint", help="Path to SSL or MAE checkpoint.")
    parser.add_argument("--ssl-method", choices=["simclr", "moco"], default="simclr")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, help="Output CSV with scores.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    test_df = pd.read_csv(args.test_csv)
    transform = get_mae_transform(args.image_size)
    test_ds = NIHChestXrayDataset(test_df, transform=transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.method == "mae":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for MAE scoring.")
        model = MAEWrapper()
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        scores = mae_reconstruction_scores(model, test_loader, device)
    else:
        if not args.train_csv or not args.checkpoint:
            raise ValueError("--train-csv and --checkpoint are required for embedding scoring.")
        train_df = pd.read_csv(args.train_csv)
        train_ds = NIHChestXrayDataset(train_df, transform=transform)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        encoder = load_ssl_encoder(
            args.checkpoint,
            method=args.ssl_method,
            backbone=args.backbone,
            projection_dim=args.projection_dim,
            hidden_dim=args.hidden_dim,
        )
        train_feats, _ = extract_embeddings(encoder, train_loader, device)
        test_feats, test_labels = extract_embeddings(encoder, test_loader, device)

        if args.method == "knn":
            model = fit_knn(train_feats)
            scores = score_knn(model, test_feats)
        elif args.method == "mahalanobis":
            mean, cov_inv = fit_mahalanobis(train_feats)
            scores = score_mahalanobis(test_feats, mean, cov_inv)
        else:
            model = fit_one_class_svm(train_feats)
            scores = score_one_class_svm(model, test_feats)

        test_df["label"] = test_labels

    test_df["score"] = scores
    test_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

