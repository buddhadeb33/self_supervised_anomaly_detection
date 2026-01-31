import argparse

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc


def parse_args():
    parser = argparse.ArgumentParser(description="Plot ROC curve from scores CSV.")
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.scores_csv)
    y_true = df[args.label_col].astype(int).to_numpy()
    y_score = df[args.score_col].astype(float).to_numpy()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUROC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()

