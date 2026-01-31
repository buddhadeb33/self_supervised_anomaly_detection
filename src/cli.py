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
    CheXpertDataset,
    MIMICCXRDataset,
    NIHChestXrayDataset,
    SplitConfig,
    create_dataset,
    create_nih_splits,
    load_chexpert_metadata,
    load_dataset_metadata,
    load_mimic_cxr_metadata,
    load_nih_metadata,
    print_cross_dataset_summary,
    save_splits,
)
from .eval import (
    bootstrap_ci,
    evaluate_calibration,
    evaluate_scores,
    evaluate_scores_with_calibration,
    plot_reliability_diagram,
)
from .explain import GradCAM, mae_patch_anomaly_map
from .scoring import (
    DiffusionAnomalyScorer,
    PatchCoreFeatureExtractor,
    PatchCoreScorer,
    dual_space_score,
    extract_embeddings,
    extract_patch_features,
    fit_knn,
    fit_mahalanobis,
    fit_one_class_svm,
    fit_patchcore,
    load_ssl_encoder,
    mae_reconstruction_scores,
    score_diffusion,
    score_diffusion_with_patches,
    score_knn,
    score_mahalanobis,
    score_one_class_svm,
    score_patchcore,
    train_diffusion_scorer,
)
from .ssl import (
    DINOv2Encoder,
    DINOv2PatchExtractor,
    MAEWrapper,
    MoCoV2,
    SimCLR,
    train_mae,
    train_moco,
    train_simclr,
)
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


def cmd_cross_dataset_eval(args: argparse.Namespace) -> None:
    """
    Cross-dataset evaluation: train on source dataset, evaluate on target datasets.
    Supports PatchCore and dual-space scoring.
    """
    device = _device_from_arg(args.device)
    transform = get_mae_transform(args.image_size)

    # Load source (training) data
    print(f"Loading source dataset: {args.source_dataset}")
    if args.source_dataset == "nih":
        source_df = pd.read_csv(args.source_csv)
    elif args.source_dataset == "chexpert":
        source_df = load_chexpert_metadata(
            args.source_csv,
            image_root=args.source_image_root,
            treat_uncertain_as=args.treat_uncertain_as,
        )
        if args.normal_only_train:
            source_df = source_df[source_df["is_normal"]].copy()
    else:
        raise ValueError(f"Unknown source dataset: {args.source_dataset}")

    source_ds = create_dataset(args.source_dataset, source_df, transform=transform)
    source_loader = DataLoader(
        source_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Load encoder
    print(f"Loading encoder: {args.ssl_method}")
    if args.ssl_method.startswith("dinov2"):
        encoder = DINOv2Encoder(model_name=args.ssl_method, freeze=True)
        backbone = encoder
    else:
        encoder = load_ssl_encoder(
            args.checkpoint,
            method=args.ssl_method,
            backbone=args.backbone,
            projection_dim=args.projection_dim,
            hidden_dim=args.hidden_dim,
        )
        backbone = encoder

    # Extract global embeddings from source
    print("Extracting embeddings from source dataset...")
    train_feats, _ = extract_embeddings(backbone, source_loader, device)

    # Fit global scoring model
    print(f"Fitting global scoring model: {args.global_method}")
    if args.global_method == "knn":
        global_model = fit_knn(train_feats, k=args.k)
    elif args.global_method == "mahalanobis":
        global_model = fit_mahalanobis(train_feats)
    else:
        global_model = fit_one_class_svm(train_feats, nu=args.nu)

    # Optionally fit PatchCore
    patchcore_scorer = None
    patchcore_spatial_shape = None
    if args.use_patchcore:
        print("Fitting PatchCore model...")
        if args.ssl_method.startswith("dinov2"):
            patch_extractor = DINOv2PatchExtractor(model_name=args.ssl_method, freeze=True)
        else:
            patch_extractor = PatchCoreFeatureExtractor(backbone, layers=["layer2", "layer3"])

        patchcore_scorer, patchcore_spatial_shape = fit_patchcore(
            patch_extractor,
            source_loader,
            device,
            coreset_ratio=args.coreset_ratio,
            coreset_method=args.coreset_method,
            k=args.patchcore_k,
        )

    # Evaluate on each target dataset
    all_results = {}

    for target_spec in args.targets:
        # Parse target specification: "name:dataset:csv:image_root"
        parts = target_spec.split(":")
        if len(parts) < 3:
            raise ValueError(
                f"Target must be 'name:dataset:csv_path' or 'name:dataset:csv_path:image_root'. Got: {target_spec}"
            )
        target_name = parts[0]
        target_dataset = parts[1]
        target_csv = parts[2]
        target_image_root = parts[3] if len(parts) > 3 else None

        print(f"\nEvaluating on target: {target_name} ({target_dataset})")

        # Load target dataset
        if target_dataset == "nih":
            target_df = pd.read_csv(target_csv)
        elif target_dataset == "chexpert":
            target_df = load_chexpert_metadata(
                target_csv,
                image_root=target_image_root,
                treat_uncertain_as=args.treat_uncertain_as,
            )
        elif target_dataset == "mimic":
            target_df = load_mimic_cxr_metadata(
                labels_csv=args.mimic_labels_csv,
                metadata_csv=target_csv,
                image_root=target_image_root,
                treat_uncertain_as=args.treat_uncertain_as,
            )
        else:
            raise ValueError(f"Unknown target dataset: {target_dataset}")

        target_ds = create_dataset(target_dataset, target_df, transform=transform)
        target_loader = DataLoader(
            target_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # Extract embeddings and score
        test_feats, labels = extract_embeddings(backbone, target_loader, device)

        # Global scores
        if args.global_method == "knn":
            global_scores = score_knn(global_model, test_feats)
        elif args.global_method == "mahalanobis":
            global_scores = score_mahalanobis(test_feats, global_model[0], global_model[1])
        else:
            global_scores = score_one_class_svm(global_model, test_feats)

        # PatchCore scores
        patch_scores = None
        if patchcore_scorer is not None:
            print(f"  Computing PatchCore scores...")
            if args.ssl_method.startswith("dinov2"):
                patch_extractor = DINOv2PatchExtractor(model_name=args.ssl_method, freeze=True)
            else:
                patch_extractor = PatchCoreFeatureExtractor(backbone, layers=["layer2", "layer3"])

            image_scores, _, patch_maps = score_patchcore(
                patch_extractor,
                patchcore_scorer,
                target_loader,
                device,
                patchcore_spatial_shape,
            )
            patch_scores = image_scores

        # Combine scores (dual-space)
        if args.use_dual_space and patch_scores is not None:
            final_scores = dual_space_score(
                global_scores,
                patch_scores,
                alpha=args.dual_alpha,
                normalize=True,
            )
        elif patch_scores is not None:
            final_scores = patch_scores
        else:
            final_scores = global_scores

        # Evaluate
        metrics = evaluate_scores_with_calibration(labels, final_scores)

        # Bootstrap CI
        if args.bootstrap:
            ci = bootstrap_ci(labels, final_scores, n_boot=args.n_boot)
            metrics["ci"] = {k: list(v) for k, v in ci.items()}

        all_results[target_name] = {
            "metrics": metrics,
            "n_samples": len(labels),
            "n_abnormal": int(labels.sum()),
            "n_normal": int((1 - labels).sum()),
        }

        # Also store individual scoring results
        if patch_scores is not None:
            all_results[target_name]["global_only"] = evaluate_scores(labels, global_scores)
            all_results[target_name]["patch_only"] = evaluate_scores(labels, patch_scores)

        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  AUPRC: {metrics['auprc']:.4f}")
        print(f"  ECE: {metrics['ece']:.4f}")

        # Save scores
        if args.save_scores:
            scores_df = target_df.copy()
            scores_df["global_score"] = global_scores
            if patch_scores is not None:
                scores_df["patch_score"] = patch_scores
            scores_df["score"] = final_scores
            scores_df["label"] = labels
            scores_path = os.path.join(args.output_dir, f"{target_name}_scores.csv")
            os.makedirs(os.path.dirname(scores_path) or ".", exist_ok=True)
            scores_df.to_csv(scores_path, index=False)
            print(f"  Saved scores to {scores_path}")

    # Save combined results
    output_path = os.path.join(args.output_dir, "cross_dataset_results.json")
    _save_json(output_path, all_results)
    print(f"\nSaved results to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("CROSS-DATASET EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Dataset':<20} {'AUROC':>10} {'AUPRC':>10} {'FPR@95':>10} {'ECE':>10}")
    print("-" * 60)
    for name, res in all_results.items():
        m = res["metrics"]
        print(
            f"{name:<20} {m['auroc']:>10.4f} {m['auprc']:>10.4f} "
            f"{m['fpr_at_95_tpr']:>10.4f} {m['ece']:>10.4f}"
        )
    print("=" * 80)


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


def cmd_train_diffusion(args: argparse.Namespace) -> None:
    """Train diffusion model for anomaly detection."""
    device = _device_from_arg(args.device)
    train_df = pd.read_csv(args.train_csv)

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

    model = DiffusionAnomalyScorer(
        image_size=args.image_size,
        in_channels=3,
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        base_channels=args.base_channels,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_diffusion_scorer(
        model,
        loader,
        optimizer,
        device,
        epochs=args.epochs,
        output_dir=args.output_dir,
    )


def cmd_score_diffusion(args: argparse.Namespace) -> None:
    """Score images using diffusion reconstruction error."""
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

    # Load model
    model = DiffusionAnomalyScorer(
        image_size=args.image_size,
        in_channels=3,
        timesteps=args.timesteps,
        base_channels=args.base_channels,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    # Score
    if args.use_patches:
        scores, labels, patch_maps = score_diffusion_with_patches(
            model,
            test_loader,
            device,
            patch_size=args.patch_size,
            n_steps=args.n_steps,
            start_t=args.start_t,
        )
    else:
        scores, labels = score_diffusion(
            model,
            test_loader,
            device,
            n_steps=args.n_steps,
            start_t=args.start_t,
        )

    # Save
    test_df["score"] = scores
    test_df["label"] = labels
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    test_df.to_csv(args.output, index=False)
    print(f"Saved scores to {args.output}")


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

    # Cross-dataset evaluation
    cross_ds = subparsers.add_parser(
        "cross_dataset_eval",
        help="Cross-dataset evaluation: train on source, evaluate on targets.",
    )
    cross_ds.add_argument(
        "--source-dataset",
        choices=["nih", "chexpert"],
        default="nih",
        help="Source dataset for training",
    )
    cross_ds.add_argument("--source-csv", required=True, help="Source dataset CSV")
    cross_ds.add_argument("--source-image-root", help="Source dataset image root")
    cross_ds.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="Target datasets: 'name:dataset:csv_path:image_root' (image_root optional)",
    )
    cross_ds.add_argument("--mimic-labels-csv", help="MIMIC-CXR labels CSV (for MIMIC targets)")
    cross_ds.add_argument(
        "--ssl-method",
        choices=["simclr", "moco", "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"],
        default="simclr",
    )
    cross_ds.add_argument("--checkpoint", help="SSL checkpoint (not needed for DINOv2)")
    cross_ds.add_argument("--backbone", default="resnet50")
    cross_ds.add_argument("--projection-dim", type=int, default=128)
    cross_ds.add_argument("--hidden-dim", type=int, default=2048)
    cross_ds.add_argument(
        "--global-method",
        choices=["knn", "mahalanobis", "ocsvm"],
        default="knn",
        help="Global scoring method",
    )
    cross_ds.add_argument("--k", type=int, default=5, help="k for kNN")
    cross_ds.add_argument("--nu", type=float, default=0.1, help="nu for OCSVM")
    cross_ds.add_argument("--use-patchcore", action="store_true", help="Enable PatchCore scoring")
    cross_ds.add_argument("--coreset-ratio", type=float, default=0.1)
    cross_ds.add_argument("--coreset-method", choices=["random", "greedy"], default="random")
    cross_ds.add_argument("--patchcore-k", type=int, default=9)
    cross_ds.add_argument("--use-dual-space", action="store_true", help="Enable dual-space scoring")
    cross_ds.add_argument("--dual-alpha", type=float, default=0.5, help="Weight for global scores")
    cross_ds.add_argument("--treat-uncertain-as", choices=["positive", "negative"], default="negative")
    cross_ds.add_argument("--normal-only-train", action="store_true")
    cross_ds.add_argument("--batch-size", type=int, default=32)
    cross_ds.add_argument("--image-size", type=int, default=224)
    cross_ds.add_argument("--num-workers", type=int, default=4)
    cross_ds.add_argument("--device", default="cuda")
    cross_ds.add_argument("--bootstrap", action="store_true")
    cross_ds.add_argument("--n-boot", type=int, default=1000)
    cross_ds.add_argument("--save-scores", action="store_true")
    cross_ds.add_argument("--output-dir", default="results/cross_dataset")
    cross_ds.set_defaults(func=cmd_cross_dataset_eval)

    # Diffusion training
    diffusion_train = subparsers.add_parser(
        "train_diffusion",
        help="Train diffusion model for anomaly detection.",
    )
    diffusion_train.add_argument("--train-csv", required=True)
    diffusion_train.add_argument("--batch-size", type=int, default=16)
    diffusion_train.add_argument("--epochs", type=int, default=50)
    diffusion_train.add_argument("--lr", type=float, default=1e-4)
    diffusion_train.add_argument("--image-size", type=int, default=224)
    diffusion_train.add_argument("--timesteps", type=int, default=1000)
    diffusion_train.add_argument("--base-channels", type=int, default=64)
    diffusion_train.add_argument("--beta-schedule", choices=["cosine", "linear"], default="cosine")
    diffusion_train.add_argument("--output-dir", default="checkpoints/diffusion")
    diffusion_train.add_argument("--num-workers", type=int, default=4)
    diffusion_train.add_argument("--device", default="cuda")
    diffusion_train.set_defaults(func=cmd_train_diffusion)

    # Diffusion scoring
    diffusion_score = subparsers.add_parser(
        "score_diffusion",
        help="Score images using diffusion reconstruction error.",
    )
    diffusion_score.add_argument("--test-csv", required=True)
    diffusion_score.add_argument("--checkpoint", required=True)
    diffusion_score.add_argument("--batch-size", type=int, default=16)
    diffusion_score.add_argument("--image-size", type=int, default=224)
    diffusion_score.add_argument("--timesteps", type=int, default=1000)
    diffusion_score.add_argument("--base-channels", type=int, default=64)
    diffusion_score.add_argument("--n-steps", type=int, default=250, help="Denoising steps for reconstruction")
    diffusion_score.add_argument("--start-t", type=int, default=None, help="Starting noise level")
    diffusion_score.add_argument("--use-patches", action="store_true", help="Use patch-level scoring")
    diffusion_score.add_argument("--patch-size", type=int, default=16)
    diffusion_score.add_argument("--output", required=True)
    diffusion_score.add_argument("--num-workers", type=int, default=4)
    diffusion_score.add_argument("--device", default="cuda")
    diffusion_score.set_defaults(func=cmd_score_diffusion)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

