import argparse
import json

import pandas as pd

from ..eval.metrics import evaluate_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate metrics across datasets.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Pairs like name=path/to/scores.csv",
    )
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {}
    for entry in args.inputs:
        if "=" not in entry:
            raise ValueError("Inputs must be name=path/to/scores.csv")
        name, path = entry.split("=", 1)
        df = pd.read_csv(path)
        y_true = df[args.label_col].astype(int).to_numpy()
        y_score = df[args.score_col].astype(float).to_numpy()
        results[name] = evaluate_scores(y_true, y_score)

    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()

