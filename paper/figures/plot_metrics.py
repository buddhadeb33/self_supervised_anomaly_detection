import argparse
import json

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot metrics from JSON outputs.")
    parser.add_argument("--input", required=True, help="JSON file with metrics per method.")
    parser.add_argument("--metric", default="auroc")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    labels = list(data.keys())
    values = [data[key][args.metric] for key in labels]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel(args.metric.upper())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()

