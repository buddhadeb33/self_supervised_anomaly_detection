# Self-Supervised Anomaly Detection on Chest X-rays

This repo provides an end-to-end pipeline for self-supervised pretraining, anomaly scoring,
evaluation, and explainability on NIH ChestXray14 with **dual-space scoring** and **cross-dataset generalization**.

## Key Features

- **SSL Methods**: SimCLR, MoCo-v2, MAE, DINOv2 (frozen)
- **Anomaly Scoring**: kNN, Mahalanobis, OCSVM, MAE reconstruction, PatchCore, Diffusion-based
- **Dual-Space Scoring**: Combine global embeddings + patch-level features
- **Cross-Dataset Evaluation**: NIH → CheXpert / MIMIC-CXR
- **Calibration Metrics**: ECE, MCE, Brier score, reliability diagrams
- **Explainability**: Grad-CAM, MAE patch maps, PatchCore heatmaps
- **Synthetic Anomalies**: CutPaste augmentation for anomaly-aware training

## Quickstart

### 1. Create splits
```bash
python -m src.cli create_splits --csv /path/Data_Entry_2017.csv --image-root /path/images --output-dir splits --normal-only-train
```

### 2. Train SSL (SimCLR/MoCo/MAE)
```bash
python -m src.cli train_ssl --method simclr --train-csv splits/train.csv --output-dir checkpoints/simclr
```

### 3. Score anomalies (basic)
```bash
python -m src.cli score --method knn --train-csv splits/train.csv --test-csv splits/test.csv --checkpoint checkpoints/simclr/simclr_epoch_100.pt --ssl-method simclr --output scores.csv
```

### 4. Evaluate with calibration
```bash
python -m src.cli evaluate --scores-csv scores.csv --bootstrap
```

### 5. Cross-dataset evaluation (NIH → CheXpert)
```bash
python -m src.cli cross_dataset_eval \
  --source-dataset nih --source-csv splits/train.csv \
  --targets "chexpert:chexpert:/path/CheXpert-v1.0/valid.csv:/path/CheXpert-v1.0" \
  --ssl-method dinov2_vitb14 \
  --use-patchcore --use-dual-space \
  --output-dir results/cross_dataset
```

### 6. Train diffusion model (stretch goal)
```bash
python -m src.cli train_diffusion --train-csv splits/train.csv --output-dir checkpoints/diffusion
python -m src.cli score_diffusion --test-csv splits/test.csv --checkpoint checkpoints/diffusion/diffusion_epoch_50.pt --output scores_diffusion.csv
```

### 7. Explain predictions
```bash
python -m src.cli explain --method mae --image-path /path/image.png --checkpoint checkpoints/mae/mae_epoch_1.pt --output heatmap.npy
```

### 8. Run ablation config
```bash
python -m src.cli run_experiment --config configs/nih_ablation.yaml
```

## Project Structure

```
src/
├── data.py       # Dataset loaders (NIH, CheXpert, MIMIC-CXR) and splits
├── augment.py    # SSL augmentations + CutPaste synthetic anomalies
├── ssl.py        # SimCLR, MoCo-v2, MAE, DINOv2 encoders
├── scoring.py    # kNN, Mahalanobis, OCSVM, PatchCore, Diffusion scoring
├── eval.py       # Metrics (AUROC, AUPRC, ECE) and calibration
├── explain.py    # Grad-CAM and patch anomaly maps
├── cli.py        # Single CLI entrypoint
├── utils.py      # Utilities

configs/
├── nih_ablation.yaml       # Ablation experiments
├── nih_simclr.yaml         # SimCLR training
├── nih_moco.yaml           # MoCo-v2 training
├── nih_mae.yaml            # MAE training
├── nih_diffusion.yaml      # Diffusion training
├── cross_dataset_eval.yaml # Cross-dataset evaluation

paper/
├── outline.md              # Paper structure
├── experiments.md          # Experiment design
├── related_work.md         # Literature review
├── roadmap.md              # Timeline
```

## Scoring Methods

| Method | Type | Description |
|--------|------|-------------|
| kNN | Global | k-nearest neighbor distance in embedding space |
| Mahalanobis | Global | Mahalanobis distance from normal distribution |
| OCSVM | Global | One-class SVM decision boundary |
| PatchCore | Patch | Memory bank of patch features with coreset selection |
| MAE Recon | Patch | Masked autoencoder reconstruction error |
| Diffusion | Patch | Denoising diffusion reconstruction error |
| Dual-Space | Fusion | Weighted combination of global + patch scores |

## Scope & Exploration Ideas

- **Dataset scope**: NIH-only baseline, NIH→CheXpert generalization, or multi-dataset pretraining
- **SSL choices**: SimCLR vs MoCo-v2 vs MAE vs DINOv2; backbone size vs accuracy trade-off
- **Anomaly scoring**: Global vs patch-based vs dual-space fusion
- **Protocol changes**: Normal-only training vs all-image pretraining; patient-level splits
- **Explainability**: Grad-CAM vs MAE patch maps vs PatchCore heatmaps
- **Calibration**: ECE optimization, temperature scaling
- **Deployment**: Reduce model size (ResNet-18, ViT-Tiny), measure latency

## References

See `paper/related_work.md` for star-graded references and reading order.

## Course Project Form

- Proposal submission form: [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSeTXIdRfQ4WR0rUublF3YzMJfVF0OPdChagthdBFs74gBZobA/viewform)
- Slide deck: [Computer Vision Project](https://docs.google.com/presentation/d/1Vcgb1uLly2Mi90zD3hlfZbOynRS9jjKEfOmhO473nO4/edit?slide=id.p12#slide=id.p12)
