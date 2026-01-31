# Detailed Roadmap

## Phase 0: Setup (Week 0)
- Environment setup, dependency check, dataset access planning.
- Define normal/abnormal protocol and patient-wise split rules.

## Phase 1: Data & Protocol (Week 1)
- Parse NIH metadata and generate splits.
- Sanity checks: label distributions, patient leakage audit.
- Output artifacts: `splits/train.csv`, `splits/val.csv`, `splits/test.csv`.

## Phase 2: SSL Pretraining (Weeks 2–3)
- Train SimCLR (ResNet-50) on NIH normal-only images.
- Train MoCo-v2 (ResNet-50) for comparison.
- Train MAE (ViT base) for reconstruction baseline.
- Output artifacts: SSL checkpoints + training logs.

## Phase 3: Anomaly Scoring (Week 4)
- Extract embeddings for train/test.
- Fit kNN, Mahalanobis, and OCSVM.
- Compute MAE reconstruction scores.
- Output artifacts: `scores/*.csv`.

## Phase 4: Evaluation + Ablations (Weeks 5–6)
- Metrics: AUROC, AUPRC, FPR@95%TPR, Sens@95%Spec.
- Run ablations: normal-only vs all-image pretraining; resolution; scoring variants.
- Output artifacts: results JSON, plots (ROC, bar charts).

## Phase 5: Cross-Dataset Generalization (Week 7)
- Train on NIH, evaluate on CheXpert or MIMIC-CXR.
- Compare performance drop and stability.

## Phase 6: Explainability (Week 8)
- Generate Grad-CAM heatmaps for select cases.
- Generate MAE patch anomaly maps.
- Qualitative analysis with pathology overlap discussion.

## Phase 7: Packaging (Week 9)
- Finalize paper sections.
- Prepare reproducibility checklist and code release.

## TA Validation Checklist (Short)
- Confirm project scope aligns with compute constraints (single GPU, efficient models).
- Validate the value-add: dual-space scoring + cross-dataset generalization.
- Approve baseline list (SimCLR/MoCo/MAE + at least one strong patch baseline).
- Agree on evaluation protocol (patient-wise splits, normal-only training).
- Review planned deliverables: metrics + ablations + explainability figures.
