# Self-Supervised Anomaly Detection on Chest X-rays

This repo provides an end-to-end pipeline for self-supervised pretraining, anomaly scoring,
evaluation, and explainability on NIH ChestXray14.

## Quickstart
1. Create splits:
   - `python -m src.cli.create_splits --csv /path/Data_Entry_2017.csv --image-root /path/images --output-dir splits --normal-only-train`
2. Train SSL:
   - `python -m src.cli.train_ssl --method simclr --train-csv splits/train.csv --output-dir checkpoints/simclr`
3. Score anomalies:
   - `python -m src.cli.score_anomaly --method knn --train-csv splits/train.csv --test-csv splits/test.csv --checkpoint checkpoints/simclr/simclr_epoch_1.pt --ssl-method simclr --output scores.csv`
4. Evaluate:
   - `python -m src.cli.evaluate --scores-csv scores.csv --bootstrap`
5. Explain:
   - `python -m src.cli.explain --method mae --image-path /path/image.png --checkpoint checkpoints/mae/mae_epoch_1.pt --output heatmap.npy`

## Project Structure
- `src/data`: dataset loaders and splits
- `src/augment`: SSL augmentations
- `src/ssl`: SimCLR, MoCo-v2, MAE, trainers
- `src/scoring`: kNN, Mahalanobis, one-class SVM, MAE reconstruction
- `src/eval`: metrics and evaluation harness
- `src/explain`: Grad-CAM and patch anomaly maps
- `paper/`: paper outline, experiments, and figure scripts

