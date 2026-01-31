from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .utils import read_lines, seed_everything

NO_FINDING_LABEL = "No Finding"


def _parse_labels(label_str: str) -> list[str]:
    if pd.isna(label_str):
        return []
    return [label.strip() for label in str(label_str).split("|") if label.strip()]


def _is_normal(labels: list[str]) -> bool:
    return len(labels) == 1 and labels[0] == NO_FINDING_LABEL


def load_nih_metadata(csv_path: str, image_root: Optional[str] = None) -> pd.DataFrame:
    """
    Load NIH ChestXray14 metadata from the official CSV and compute basic fields.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Image Index": "image_index"})
    df["finding_labels"] = df["Finding Labels"].apply(_parse_labels)
    df["is_normal"] = df["finding_labels"].apply(_is_normal)
    df["is_abnormal"] = ~df["is_normal"]
    df["patient_id"] = df["Patient ID"].astype(str)
    if image_root:
        df["image_path"] = df["image_index"].apply(lambda x: os.path.join(image_root, x))
    return df


class NIHChestXrayDataset(Dataset):
    """
    Minimal dataset for NIH ChestXray14 anomaly detection.
    Expects a DataFrame with at least image_path and is_abnormal columns.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
        return_metadata: bool = False,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = 1 if row["is_abnormal"] else 0
        if self.return_metadata:
            meta = {
                "image_index": row.get("image_index"),
                "patient_id": row.get("patient_id"),
                "finding_labels": row.get("finding_labels"),
            }
            return image, label, meta
        return image, label


@dataclass
class SplitConfig:
    val_fraction: float = 0.1
    test_fraction: float = 0.2
    seed: int = 42
    normal_only_train: bool = True
    train_val_list: Optional[str] = None
    test_list: Optional[str] = None


def _assign_by_list(
    df: pd.DataFrame,
    train_val_list: Optional[str],
    test_list: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not train_val_list and not test_list:
        return pd.DataFrame(), pd.DataFrame()

    train_val_images = set(read_lines(train_val_list)) if train_val_list else set()
    test_images = set(read_lines(test_list)) if test_list else set()

    df = df.copy()
    df["split_hint"] = "unassigned"
    if test_images:
        df.loc[df["image_index"].isin(test_images), "split_hint"] = "test"
    if train_val_images:
        df.loc[df["image_index"].isin(train_val_images), "split_hint"] = "trainval"

    patient_test = set(df.loc[df["split_hint"] == "test", "patient_id"].unique())
    df.loc[df["patient_id"].isin(patient_test), "split_hint"] = "test"

    test_df = df[df["split_hint"] == "test"].copy()
    trainval_df = df[df["split_hint"] != "test"].copy()
    return trainval_df, test_df


def _patient_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seed_everything(seed)
    patient_ids = df["patient_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)
    val_size = max(1, int(len(patient_ids) * val_fraction))
    val_patients = set(patient_ids[:val_size])
    val_df = df[df["patient_id"].isin(val_patients)].copy()
    train_df = df[~df["patient_id"].isin(val_patients)].copy()
    return train_df, val_df


def create_nih_splits(
    df: pd.DataFrame,
    config: SplitConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create patient-level train/val/test splits.
    If train/test lists are provided, use them and then split trainval into train/val.
    """
    if config.train_val_list or config.test_list:
        trainval_df, test_df = _assign_by_list(
            df,
            config.train_val_list,
            config.test_list,
        )
    else:
        seed_everything(config.seed)
        patient_ids = df["patient_id"].unique()
        rng = np.random.default_rng(config.seed)
        rng.shuffle(patient_ids)
        test_size = max(1, int(len(patient_ids) * config.test_fraction))
        test_patients = set(patient_ids[:test_size])
        test_df = df[df["patient_id"].isin(test_patients)].copy()
        trainval_df = df[~df["patient_id"].isin(test_patients)].copy()

    train_df, val_df = _patient_split(trainval_df, config.val_fraction, config.seed)
    if config.normal_only_train:
        train_df = train_df[train_df["is_normal"]].copy()
    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

