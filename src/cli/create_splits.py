import argparse
import os

from ..data import create_nih_splits, load_nih_metadata, save_splits
from ..data.splits import SplitConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create NIH ChestXray14 splits.")
    parser.add_argument("--csv", required=True, help="Path to Data_Entry_2017.csv")
    parser.add_argument("--image-root", required=True, help="Path to images directory")
    parser.add_argument("--output-dir", required=True, help="Directory to save splits")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normal-only-train", action="store_true")
    parser.add_argument("--train-val-list", default=None)
    parser.add_argument("--test-list", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_nih_metadata(args.csv, image_root=args.image_root)
    config = SplitConfig(
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        normal_only_train=args.normal_only_train,
        train_val_list=args.train_val_list,
        test_list=args.test_list,
    )
    train_df, val_df, test_df = create_nih_splits(df, config)
    save_splits(train_df, val_df, test_df, args.output_dir)


if __name__ == "__main__":
    main()

