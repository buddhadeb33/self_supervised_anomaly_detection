# Self-Supervised Anomaly Detection on Chest X-rays

This repo provides an end-to-end pipeline for self-supervised pretraining, anomaly scoring,
evaluation, and explainability on NIH ChestXray14.

## Quickstart
1. Create splits:
   - `python -m src.cli create_splits --csv /path/Data_Entry_2017.csv --image-root /path/images --output-dir splits --normal-only-train`
2. Train SSL:
   - `python -m src.cli train_ssl --method simclr --train-csv splits/train.csv --output-dir checkpoints/simclr`
3. Score anomalies:
   - `python -m src.cli score --method knn --train-csv splits/train.csv --test-csv splits/test.csv --checkpoint checkpoints/simclr/simclr_epoch_1.pt --ssl-method simclr --output scores.csv`
4. Evaluate:
   - `python -m src.cli evaluate --scores-csv scores.csv --bootstrap`
5. Explain:
   - `python -m src.cli explain --method mae --image-path /path/image.png --checkpoint checkpoints/mae/mae_epoch_1.pt --output heatmap.npy`
6. Run ablation config:
   - `python -m src.cli run_experiment --config configs/nih_ablation.yaml`

## Project Structure
- `src/data.py`: dataset loaders and splits
- `src/augment.py`: SSL augmentations
- `src/ssl.py`: SimCLR, MoCo-v2, MAE, trainers
- `src/scoring.py`: kNN, Mahalanobis, one-class SVM, MAE reconstruction
- `src/eval.py`: metrics and evaluation harness
- `src/explain.py`: Grad-CAM and patch anomaly maps
- `src/cli.py`: single CLI entrypoint
- `paper/`: paper outline, experiments, roadmap, references
- `notebooks/`: guided experiment kickoff

## Scope & Exploration Ideas
- Dataset scope: NIH-only baseline, NIH→CheXpert generalization, or multi-dataset pretraining.
- SSL choices: SimCLR vs MoCo-v2 vs MAE; backbone size vs accuracy trade-off.
- Anomaly scoring: kNN vs Mahalanobis vs OCSVM; global vs patch-based scoring.
- Protocol changes: normal-only training vs all-image pretraining; patient-level splits.
- Explainability: Grad-CAM vs MAE patch maps; compare with clinical plausibility.
- Deployment: reduce model size (ResNet-18), measure latency, try quantization.

## References

- See `paper/related_work.md` for star-graded references and reading order.

## Course Project Form

- Proposal submission form: [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSeTXIdRfQ4WR0rUublF3YzMJfVF0OPdChagthdBFs74gBZobA/viewform)
- Slide deck: [Computer Vision Project](https://docs.google.com/presentation/d/1Vcgb1uLly2Mi90zD3hlfZbOynRS9jjKEfOmhO473nO4/edit?slide=id.p12#slide=id.p12)
