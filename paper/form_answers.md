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



## Project Details
- Project Name: Self-Supervised Anomaly Detection on Chest X-rays
- Project Category: Medical anomaly detection

## TLDR (one line)
Learn normal chest anatomy via self-supervision and detect unseen abnormalities using dual-space anomaly scoring with cross-dataset validation.

## Abstract (brief project description)
We propose an end-to-end self-supervised anomaly detection pipeline for chest X-rays that learns normal anatomy without explicit anomaly labels. The approach pretrains visual encoders using SSL (SimCLR/MoCo/MAE) on normal images, then computes anomaly scores using a dual-space strategy that combines global embedding distance with patch-level reconstruction errors. We follow a patient-wise split protocol on NIH ChestXray14 and report AUROC/AUPRC along with FPR@95%TPR and sensitivity at 95% specificity, including ablations over pretraining data (normal-only vs all-image), scoring methods (kNN, Mahalanobis, OCSVM, MAE recon), and model size (ResNet-18/50, ViT-Tiny/Base). To assess robustness, we evaluate cross-dataset generalization on CheXpert or MIMIC-CXR without re-training. Explainability is integrated via Grad-CAM and MAE patch heatmaps to support clinical plausibility analysis. The system is designed for single-GPU feasibility, and results are packaged with reproducible configs and an experimental runner.

## Compute Feasibility Note (if asked)
The project is designed for single-GPU training with efficient backbones (ResNet-18/50, ViT-Tiny/Base) and short ablations, satisfying the course compute constraints.

## Value-Add / Novelty Note (if asked)
Our value-add is not a new backbone, but a **protocol + scoring + evaluation package** that is missing in many CXR anomaly detection baselines. Concretely:

- **Dual-space anomaly scoring**: combine global SSL embedding distance with patch-level reconstruction error, instead of relying on a single score.
- **Protocol rigor**: patient-wise splits, normal-only training, and explicit ablations over normal-only vs all-image pretraining.
- **Cross-dataset generalization as a primary result**: NIH → CheXpert/MIMIC evaluation without re-training to quantify domain shift.
- **Explainability baked in**: Grad-CAM and MAE patch maps to support clinical plausibility discussion.
- **Efficiency focus**: report accuracy vs model size (ResNet-18/50, ViT-Tiny/Base) to show feasibility under course compute limits.
- **Reproducibility**: unified CLI and config-driven runner that makes experiments and ablations easy to repeat.

