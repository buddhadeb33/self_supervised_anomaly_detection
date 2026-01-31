# Paper Outline (Draft)

## Title
Self-Supervised Anomaly Detection on Chest X-rays via Dual-Space Scoring

## Abstract
- Motivation: scarce anomaly labels, weak labels, and domain shift.
- Method: SSL backbone + dual-space anomaly scoring (global embeddings + patch maps).
- Results: NIH baseline, ablations, and cross-dataset generalization.

## 1. Introduction
- Clinical motivation and anomaly rarity
- Challenges of weak/noisy labels
- Contributions:
  1) SSL representation for normal anatomy with normal-only protocol
  2) Dual-space anomaly scoring with global + patch maps
  3) Cross-dataset generalization and calibration analysis
  4) Lightweight deployment trade-off discussion

## 2. Related Work
- SSL in medical imaging (BioViL, ConVIRT)
- Anomaly detection (Deep SVDD, PaDiM, PatchCore)
- CXR-specific AD (SSD)
- Weak labels and robust evaluation protocols

## 3. Methodology
- Data protocol and normal-only training
- SSL pretraining (SimCLR/MoCo/MAE)
- Anomaly scoring (kNN/Mahalanobis/OCSVM/MAE recon)
- Explainability (Grad-CAM + patch maps)
- Implementation details (augmentations, losses, and training budget)

## 4. Experiments
- Dataset splits and preprocessing
- Evaluation metrics
- Baselines and ablations
- Cross-dataset generalization
- Efficiency analysis (FLOPs, params, inference latency)

## 5. Results
- Main tables and plots
- Qualitative heatmaps
- Calibration curves and thresholding examples

## 6. Discussion
- Clinical plausibility
- Limitations
- Future work
- Ethics and deployment risks

## 7. Conclusion

