# Project Positioning and Publication Scope

## How This Project Differs From Prior Work
This project is intentionally built as an end-to-end, reproducible pipeline with explicit
protocols for normal-only training, patient-wise splits, and multi-method scoring. The
distinctive aspects compared to common baselines are:

- **Dual-space scoring as a first-class design choice**: combines global embedding distance
  with patch-level reconstruction maps, rather than only one of the two.
- **Experiment-first tooling**: a single CLI and config-driven runner to support ablations
  and variant comparisons without major refactors.
- **Cross-dataset generalization focus**: planned NIH -> CheXpert/MIMIC evaluation as a
  standard experiment, not an afterthought.
- **Explainability baked in**: Grad-CAM and patch maps are part of the workflow, which is
  not always standardized in prior anomaly detection baselines.
- **Lightweight deployment exploration**: explicit trade-offs with smaller backbones and
  practical constraints (memory, latency).

## Compared to Key Reference Directions (High-Level)
- **SSD (CXR-specific SSL AD)**: we emphasize dual-space scoring and a stronger ablation grid
  (normal-only vs all-image pretraining, scoring combinations, backbone size trade-offs).
- **General SSL baselines (SimCLR, MoCo, MAE)**: we position them as modular blocks that can
  be swapped in and evaluated under the same clinical protocol and scoring suite.
- **PatchCore/PaDiM style methods**: we align to their patch-level philosophy but integrate
  it with a normal-only SSL pipeline and explicit cross-dataset tests.

## Comparative Table (Specific Papers + Claims)
| Paper | Claimed Contribution | Our Difference | Evidence to Show |
|---|---|---|---|
| Golan & El-Yaniv 2018 | Geometric-transform pretext for AD | Use SSL encoders + dual-space scoring for CXR | AUROC/AUPRC, ablations |
| Deep SVDD 2018 | One-class objective in feature space | Use SSL features + multiple scoring heads | Compare vs OCSVM/Mahalanobis |
| SimCLR 2020 | Contrastive SSL with strong augmentations | Medical protocol + anomaly scoring | Normal-only vs all-image |
| MoCo-v2 2020 | Momentum contrastive learning | Same but evaluated under CXR AD protocol | NIH baseline + ablations |
| MAE 2021 | Masked reconstruction for representation | Use recon error as anomaly score | Recon vs embedding scores |
| SSD 2021 (CXR) | SSL for CXR anomaly detection | Dual-space scoring + cross-dataset focus | NIH->CheXpert results |
| PatchCore 2021/22 | Patch-level anomaly detection | Combine patch + global SSL scoring | Patch maps + global ROC |
| BioViL 2022 | Multimodal CXR SSL | Use same protocol to compare representations | Transfer performance |
| CheXzero 2022 | Zero-shot CXR diagnosis | Position as strong SSL baseline | NIH-only + cross-dataset |

## Scope to Explore (Paper-Level Opportunities)
- **Protocol scope**: normal-only vs all-image SSL; label noise handling.
- **Model scope**: ResNet-18/50, ViT-Tiny/Base, MAE vs contrastive.
- **Scoring scope**: kNN, Mahalanobis, OCSVM, recon error, dual-space fusion.
- **Generalization scope**: NIH→CheXpert/MIMIC transfer; domain-shift analysis.
- **Explainability scope**: Grad-CAM vs patch maps; clinical plausibility studies.
- **Efficiency scope**: latency/params/energy trade-offs for deployment.

## Publication Scope: Multiple Papers From One Project
The work can be separated into distinct paper-style contributions:

1. **Core Method Paper**: dual-space scoring with SSL pretraining; NIH baseline + ablations.
2. **Generalization Paper**: cross-dataset performance (NIH -> CheXpert/MIMIC), domain shift,
   and stabilization techniques.
3. **Efficiency Paper**: performance vs. compute (ResNet-18/50, ViT-Tiny/Base), latency and
   resource analysis for deployment.
4. **Explainability Paper**: structured comparison of Grad-CAM vs patch maps for clinical
   plausibility; qualitative error taxonomy.
5. **Synthetic Anomaly Paper**: anatomically constrained pseudo-lesions to improve robustness
   (if synthetic generation is added later).

## Application Areas
- **Triage and screening**: prioritize abnormal scans when radiology workload is high.
- **Quality control**: flag corrupted or out-of-distribution scans at ingestion.
- **Cross-site deployment**: domain shift diagnostics when models are moved across hospitals.
- **Label efficiency**: reduce manual annotation by using SSL features + anomaly scoring.
- **Clinical research**: identify rare or evolving patterns in longitudinal datasets.

## Course Guideline Alignment (Computer Vision Project.pdf)
Reference checklist aligned to course constraints and best practices.

| Guideline | Status | Notes / Evidence |
|---|---|---|
| Team size <= 2 | ✅ | Current team has 2 members; no exception needed. |
| Avoid high computation | ✅ | Single-GPU baseline; ResNet-18/50, ViT-Tiny options. |
| Feasibility on standard machines | ✅ | Configs are lightweight; ablation runner supports short runs. |
| Avoid overly trendy topics | ✅ | Niche focus: CXR anomaly detection with SSL protocol emphasis. |
| Literature survey depth | ⚠️ | Related work list exists; add more CXR domain-shift papers. |
| Value-add / gap identification | ✅ | Dual-space scoring + protocol + generalization emphasis. |
| Implement SOTA baselines | ⚠️ | PatchCore/PaDiM still optional; schedule one strong baseline. |
| Integrity policy (originality) | ✅ | Ensure experiments and writing are original; cite all sources. |
| TA/Instructor validation | ⚠️ | Plan to share value-add and baseline list for approval. |

## Gaps To Close (Action Items)
- Add at least one strong patch-based baseline (PatchCore or PaDiM).
- Include 1–2 CXR domain shift/generalization papers in related work.
- Confirm with TA that dual-space scoring + cross-dataset focus is non-trivial.

## Notes on Evidence Needed
- If claims are about **clinical utility**, include qualitative panels with annotations.
- If claims are about **generalization**, report NIH -> CheXpert/MIMIC drop and confidence.
- If claims are about **deployment**, include parameter counts, FLOPs, and latency.

## Where This Fits in the Repo
- Experiments: `paper/experiments.md`
- References: `paper/related_work.md`
- Roadmap: `paper/roadmap.md`

