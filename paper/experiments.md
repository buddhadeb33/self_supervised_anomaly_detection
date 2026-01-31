# Experiments & Ablations

## Datasets
- NIH ChestXray14: normal = "No Finding"; abnormal = any pathology label.
- Cross-dataset: CheXpert or MIMIC-CXR (same normal/abnormal protocol).
- Split policy: patient-wise splits, no patient leakage across train/val/test.
- Preprocessing: resize to 224, normalize; optional lung cropping as ablation.

## Baselines
- ImageNet ResNet-50 + kNN
- Deep SVDD (if time permits)
- CutPaste or PaDiM (optional)
- PatchCore (if compute allows) for strong patch-level baseline.

## SSL Variants
- SimCLR (ResNet-50)
- MoCo-v2 (ResNet-50)
- MAE (ViT base)
- Optional: smaller backbones (ResNet-18, ViT-Tiny) for deployment.

## Scoring Variants
- kNN (k=5,10)
- Mahalanobis
- One-class SVM (nu=0.1)
- MAE reconstruction error
- Dual-space: combine global score + patch map score (weighted sum).

## Ablations
1. Normal-only vs all-image SSL pretraining
2. Image resolution (224 vs 384)
3. Global-only vs dual-space scoring
4. Projection head depth (1 vs 2 layers)
5. Augmentation strength (weak vs strong)
6. Patch masking ratio (MAE)

## Metrics
- AUROC, AUPRC
- FPR@95%TPR
- Sensitivity@95%Specificity
- Report mean ± 95% CI via bootstrap.

## Table Template
| Method | AUROC | AUPRC | FPR@95%TPR | Sens@95%Spec |
|---|---:|---:|---:|---:|
| SimCLR + kNN |  |  |  |  |
| MoCo + Mahalanobis |  |  |  |  |
| MAE Recon |  |  |  |  |

## Figure List
1. ROC curves (NIH test)
2. Cross-dataset ROC (NIH→CheXpert)
3. Heatmaps for normal vs abnormal
4. Calibration plot with threshold selection

