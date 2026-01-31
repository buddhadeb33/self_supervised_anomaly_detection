# Paper Outline (Draft)

## Title
Self-Supervised Anomaly Detection on Chest X-rays via Dual-Space Scoring

## Abstract
- Motivation: scarce anomaly labels, weak labels.
- Method: SSL backbone + anomaly scoring (global + patch-level).
- Results: NIH baseline + cross-dataset generalization.

## 1. Introduction
- Clinical motivation and anomaly rarity
- Challenges of weak/noisy labels
- Contributions:
  1) SSL representation for normal anatomy
  2) Dual-space scoring with global + patch maps
  3) Cross-dataset generalization analysis

## 2. Related Work
- SSL in medical imaging (BioViL, ConVIRT)
- Anomaly detection (Deep SVDD, PaDiM, PatchCore)
- CXR-specific AD (SSD)

## 3. Methodology
- Data protocol and normal-only training
- SSL pretraining (SimCLR/MoCo/MAE)
- Anomaly scoring (kNN/Mahalanobis/OCSVM/MAE recon)
- Explainability (Grad-CAM + patch maps)

## 4. Experiments
- Dataset splits and preprocessing
- Evaluation metrics
- Baselines and ablations
- Cross-dataset generalization

## 5. Results
- Main tables and plots
- Qualitative heatmaps

## 6. Discussion
- Clinical plausibility
- Limitations
- Future work

## 7. Conclusion

