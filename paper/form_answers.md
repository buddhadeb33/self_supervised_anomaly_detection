# Computer Vision Course Project Form Answers

## Team Composition
- Team size: 2 (no third member).
- TA: Madhumitha V (ai23resch11004@iith.ac.in)

### Team Member 1
- Name: Buddhadeb Mondal
- Roll no.: cs24mtech15014
- Email: cs24mtech15014@iith.ac.in

### Team Member 2
- Name: Nabendu Bhuiya
- Roll no.: cs24mtech15011
- Email: cs24mtech15011@iith.ac.in

### Team Member 3 (optional)
- Name: N/A
- Roll no.: N/A
- Reason for third member: N/A (team size within limit)

## Project Details
- Project Name: Self-Supervised Anomaly Detection on Chest X-rays
- Project Category: Medical anomaly detection

## TLDR (one line)
Learn normal chest anatomy via self-supervision and detect unseen abnormalities using dual-space anomaly scoring with cross-dataset validation.

## Abstract (brief project description)
We propose a self-supervised anomaly detection pipeline for chest X-rays that learns normal anatomy without explicit anomaly labels. The model is pretrained using SSL (SimCLR/MoCo/MAE) on normal images and anomalies are detected using dual-space scores that combine global embeddings with patch-level reconstruction maps. We evaluate on NIH ChestXray14 using patient-wise splits and report AUROC/AUPRC with ablations, and we test cross-dataset generalization on CheXpert/MIMIC-CXR. Explainability is included via Grad-CAM and MAE patch heatmaps, while keeping the approach feasible on a single GPU.

## Compute Feasibility Note (if asked)
The project is designed for single-GPU training with efficient backbones (ResNet-18/50, ViT-Tiny/Base) and short ablations, satisfying the course compute constraints.

## Value-Add / Novelty Note (if asked)
We emphasize dual-space scoring (global + patch), explicit normal-only protocols, and cross-dataset generalization with explainability as core contributions.

