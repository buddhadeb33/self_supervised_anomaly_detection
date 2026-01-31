from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import models, transforms as T

from .augment import TwoCropsTransform, get_mae_transform, get_ssl_transform
from .data import (
    NIHChestXrayDataset,
    SplitConfig,
    create_nih_splits,
    load_nih_metadata,
    save_splits,
)
from .eval import bootstrap_ci, evaluate_scores
from .explain import GradCAM, mae_patch_anomaly_map
from .scoring import (
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
from .ssl import MAEWrapper, MoCoV2, SimCLR, train_mae, train_moco, train_simclr
from .utils import load_config


def _device_from_arg(device_str: str) -> torch.device:
    return torch.device(device_str if torch.cuda.is_available() else "cpu")


def _save_json(output: str | None, payload: Dict[str, Any]) -> None:
    if not output:
        return
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def cmd_create_splits(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    df = load_nih_metadata(args.csv, image_root=args.image_root)
    config = SplitConfig(
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        normal_only_train=args.normal_only_train,
        train_val_list=args.train_val_list,
        test_list=args.test_list,
    )
    train_df, val_df, test_df = create_nih_splits(df, config)
    save_splits(train_df, val_df, test_df, args.output_dir)


def cmd_train_ssl(args: argparse.Namespace) -> None:
    device = _device_from_arg(args.device)
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


def cmd_score(args: argparse.Namespace) -> None:
    device = _device_from_arg(args.device)
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
        labels = test_df["is_abnormal"].astype(int).to_numpy() if "is_abnormal" in test_df else None
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
        test_feats, labels = extract_embeddings(encoder, test_loader, device)

        if args.method == "knn":
            model = fit_knn(train_feats, k=args.k)
            scores = score_knn(model, test_feats)
        elif args.method == "mahalanobis":
            mean, cov_inv = fit_mahalanobis(train_feats)
            scores = score_mahalanobis(test_feats, mean, cov_inv)
        else:
            model = fit_one_class_svm(train_feats, nu=args.nu)
            scores = score_one_class_svm(model, test_feats)

    test_df["score"] = scores
    if labels is not None:
        test_df["label"] = labels
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    test_df.to_csv(args.output, index=False)


def cmd_evaluate(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.scores_csv)
    y_true = df[args.label_col].astype(int).to_numpy()
    y_score = df[args.score_col].astype(float).to_numpy()
    metrics = evaluate_scores(y_true, y_score)
    output = {"metrics": metrics}
    if args.bootstrap:
        output["ci"] = bootstrap_ci(y_true, y_score, n_boot=args.n_boot)
    print(json.dumps(output, indent=2))
    _save_json(args.output, output)


def cmd_cross_eval(args: argparse.Namespace) -> None:
    results = {}
    for entry in args.inputs:
        if "=" not in entry:
            raise ValueError("Inputs must be name=path/to/scores.csv")
        name, path = entry.split("=", 1)
        df = pd.read_csv(path)
        y_true = df[args.label_col].astype(int).to_numpy()
        y_score = df[args.score_col].astype(float).to_numpy()
        results[name] = evaluate_scores(y_true, y_score)
    print(json.dumps(results, indent=2))
    _save_json(args.output, results)


def cmd_explain(args: argparse.Namespace) -> None:
    device = _device_from_arg(args.device)
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = Image.open(args.image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    if args.method == "gradcam":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(ckpt["model"])
        model.to(device)
        target_layer = getattr(model, args.target_layer)
        cam = GradCAM(model, target_layer).generate(image_tensor)
        np.save(args.output, cam.detach().cpu().numpy())
    else:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required for MAE explainability.")
        model = MAEWrapper()
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.to(device)
        heatmap = mae_patch_anomaly_map(model, image_tensor)
        np.save(args.output, heatmap.detach().cpu().numpy())


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def run_pipeline(config: Dict[str, Any]) -> None:
    if "splits" in config:
        args = argparse.Namespace(**config["splits"])
        cmd_create_splits(args)
    if "ssl" in config:
        args = argparse.Namespace(**config["ssl"])
        cmd_train_ssl(args)
    if "scoring" in config:
        args = argparse.Namespace(**config["scoring"])
        cmd_score(args)
    if "eval" in config:
        args = argparse.Namespace(**config["eval"])
        cmd_evaluate(args)


def cmd_run_experiment(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    base = config.get("base", {})
    sweeps = config.get("sweep", [])

    if not sweeps:
        run_pipeline(base)
        return

    for sweep in sweeps:
        run_config = deepcopy(base)
        run_config = _deep_update(run_config, sweep)
        run_pipeline(run_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CXR SSL anomaly detection CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    splits = subparsers.add_parser("create_splits", help="Create NIH ChestXray14 splits.")
    splits.add_argument("--csv", required=True)
    splits.add_argument("--image-root", required=True)
    splits.add_argument("--output-dir", required=True)
    splits.add_argument("--val-fraction", type=float, default=0.1)
    splits.add_argument("--test-fraction", type=float, default=0.2)
    splits.add_argument("--seed", type=int, default=42)
    splits.add_argument("--normal-only-train", action="store_true")
    splits.add_argument("--train-val-list", default=None)
    splits.add_argument("--test-list", default=None)
    splits.set_defaults(func=cmd_create_splits)

    train = subparsers.add_parser("train_ssl", help="Train SSL baselines.")
    train.add_argument("--method", choices=["simclr", "moco", "mae"], required=True)
    train.add_argument("--train-csv", required=True)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--image-size", type=int, default=224)
    train.add_argument("--backbone", default="resnet50")
    train.add_argument("--projection-dim", type=int, default=128)
    train.add_argument("--hidden-dim", type=int, default=2048)
    train.add_argument("--mask-ratio", type=float, default=0.75)
    train.add_argument("--output-dir", default="checkpoints")
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--device", default="cuda")
    train.set_defaults(func=cmd_train_ssl)

    score = subparsers.add_parser("score", help="Compute anomaly scores.")
    score.add_argument("--method", choices=["knn", "mahalanobis", "ocsvm", "mae"], required=True)
    score.add_argument("--train-csv")
    score.add_argument("--test-csv", required=True)
    score.add_argument("--checkpoint")
    score.add_argument("--ssl-method", choices=["simclr", "moco"], default="simclr")
    score.add_argument("--backbone", default="resnet50")
    score.add_argument("--projection-dim", type=int, default=128)
    score.add_argument("--hidden-dim", type=int, default=2048)
    score.add_argument("--batch-size", type=int, default=64)
    score.add_argument("--image-size", type=int, default=224)
    score.add_argument("--num-workers", type=int, default=4)
    score.add_argument("--device", default="cuda")
    score.add_argument("--k", type=int, default=5)
    score.add_argument("--nu", type=float, default=0.1)
    score.add_argument("--output", required=True)
    score.set_defaults(func=cmd_score)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate anomaly scores.")
    evaluate.add_argument("--scores-csv", required=True)
    evaluate.add_argument("--label-col", default="label")
    evaluate.add_argument("--score-col", default="score")
    evaluate.add_argument("--bootstrap", action="store_true")
    evaluate.add_argument("--n-boot", type=int, default=1000)
    evaluate.add_argument("--output", default=None)
    evaluate.set_defaults(func=cmd_evaluate)

    cross_eval = subparsers.add_parser("cross_eval", help="Aggregate metrics across datasets.")
    cross_eval.add_argument("--inputs", nargs="+", required=True)
    cross_eval.add_argument("--label-col", default="label")
    cross_eval.add_argument("--score-col", default="score")
    cross_eval.add_argument("--output", default=None)
    cross_eval.set_defaults(func=cmd_cross_eval)

    explain = subparsers.add_parser("explain", help="Generate explainability maps.")
    explain.add_argument("--method", choices=["gradcam", "mae"], required=True)
    explain.add_argument("--image-path", required=True)
    explain.add_argument("--output", required=True)
    explain.add_argument("--checkpoint", default=None)
    explain.add_argument("--target-layer", default="layer4")
    explain.add_argument("--device", default="cuda")
    explain.set_defaults(func=cmd_explain)

    run = subparsers.add_parser("run_experiment", help="Run config-driven experiment.")
    run.add_argument("--config", required=True)
    run.set_defaults(func=cmd_run_experiment)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

