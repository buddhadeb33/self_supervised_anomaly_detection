import argparse
import json

import pandas as pd

from ..eval.evaluate import bootstrap_ci
from ..eval.metrics import evaluate_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate anomaly scores.")
    parser.add_argument("--scores-csv", required=True)
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.scores_csv)
    y_true = df[args.label_col].astype(int).to_numpy()
    y_score = df[args.score_col].astype(float).to_numpy()

    metrics = evaluate_scores(y_true, y_score)
    output = {"metrics": metrics}
    if args.bootstrap:
        output["ci"] = bootstrap_ci(y_true, y_score, n_boot=args.n_boot)

    print(json.dumps(output, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)


if __name__ == "__main__":
    main()

