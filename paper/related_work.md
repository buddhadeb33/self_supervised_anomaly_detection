# Related Work (Star-Graded)

Legend: ★★★ seminal/required, ★★ highly relevant, ★ useful background

## Starter Reading (Links)
- ★★★ SimCLR: A Simple Framework for Contrastive Learning of Visual Representations (Chen et al., 2020) — https://arxiv.org/abs/2002.05709
- ★★★ Deep SVDD: Deep One-Class Classification (Ruff et al., 2018) — https://arxiv.org/abs/1802.06360
- ★★ MoCo-v2: Improved Baselines with Momentum Contrastive Learning (He et al., 2020) — https://arxiv.org/abs/2003.04297
- ★★ MAE: Masked Autoencoders Are Scalable Vision Learners (He et al., 2021) — https://arxiv.org/abs/2111.06377
- ★★ PatchCore: Towards Total Recall in Industrial Anomaly Detection (Roth et al., 2021) — https://arxiv.org/abs/2106.08265

## Core SSL + AD (General)
- ★★★ Golan & El-Yaniv, 2018: Geometric transformations for AD
- ★★★ Deep SVDD (Ruff et al., ICML 2018)
- ★★ SimCLR (Chen et al., 2020)
- ★★ MoCo-v2 (He et al., 2020)
- ★★ MAE (He et al., 2021)
- ★★ CutPaste (Li et al., 2021)
- ★★ PatchCore (Roth et al., 2022)
- ★ PaDiM (Defard et al., 2020)
- ★ DINO (Caron et al., 2021)

## Medical Imaging + CXR
- ★★★ SSD: Self-Supervised Learning for AD in Chest X-rays (Sehwag et al., MICCAI 2021)
- ★★★ Azizi et al., 2021: SSL works better for medical imaging (Nature Medicine)
- ★★ BioViL (Boecking et al., 2022)
- ★★ ConVIRT (Zhang et al., 2020)
- ★★ CheXzero (Tiu et al., 2022)
- ★ Li et al., 2021: synthetic anomaly generation (CVPR)

## CXR Domain Shift / External Validation
- ★★ Zech et al., 2018: Generalizability of chest X-ray models across hospitals
- ★★ CheXphoto (Irvin et al., 2019): real-world photo domain shift benchmark
- ★★ CheXpert (Irvin et al., 2019): large-scale dataset for external validation
- ★★ MIMIC-CXR (Johnson et al., 2019): multi-institution dataset for cross-site tests
- ★ PadChest (Bustos et al., 2020): alternative distribution for robustness checks
- ★ NIH ChestXray14 (Wang et al., 2017): legacy dataset; label noise considerations

## Closest Overlap (Last 2–3 Years)
No recent CXR paper appears to combine **dual-space anomaly scoring** with **cross-dataset
evaluation** as a single, explicit framework. The closest overlaps are partial:

- **MedIAnomaly (2024)** provides a multi-dataset medical anomaly benchmark (includes CXR
  among other modalities) and emphasizes standardized evaluation, but it does not focus on
  dual-space scoring or a dedicated CXR cross-dataset protocol.  
  Link: https://arxiv.org/pdf/2404.04518
- **CXR Foundation models (2024)** offer strong CXR representations and cross-site utility,
  but they do not define an anomaly detection pipeline or dual-space scoring.  
  Link: https://developers.google.com/health-ai-developer-foundations/cxr-foundation  
  Link: https://github.com/google-health/cxr-foundation

## How to Keep This Safely Non-Overcrowded
- **Scope it tightly**: CXR anomaly detection with normal-only training and patient-wise
  splits as a strict protocol.
- **Show a gap**: report cross-dataset performance (NIH → CheXpert/MIMIC) and quantify the
  drop; few CXR AD papers make this a primary result.
- **Make the dual-space design central**: ablate global vs patch-only vs fused scoring.
- **Limit the baselines but make them strong**: include one patch baseline (PatchCore/PaDiM)
  and one SSL baseline (SimCLR/MoCo/MAE).
- **Be compute-honest**: justify single-GPU feasibility and report efficiency trade-offs.

## Evaluation/Robustness
- ★ Uncertainty calibration for medical AI
- ★ Cross-dataset shift in chest X-ray (various surveys)

