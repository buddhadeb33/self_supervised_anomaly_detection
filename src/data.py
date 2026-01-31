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


# =============================================================================
# CheXpert Dataset Support
# =============================================================================

CHEXPERT_NO_FINDING_LABEL = "No Finding"
CHEXPERT_PATHOLOGY_COLUMNS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def load_chexpert_metadata(
    csv_path: str,
    image_root: Optional[str] = None,
    treat_uncertain_as: str = "negative",
) -> pd.DataFrame:
    """
    Load CheXpert metadata and compute normal/abnormal labels.

    CheXpert uses: 1.0 = positive, 0.0 = negative, -1.0 = uncertain, NaN = not mentioned

    Args:
        csv_path: Path to train.csv or valid.csv from CheXpert
        image_root: Root directory for images (prepended to Path column)
        treat_uncertain_as: How to treat -1 labels ('positive', 'negative', or 'ignore')

    Returns:
        DataFrame with is_normal and is_abnormal columns
    """
    df = pd.read_csv(csv_path)

    # Handle uncertain labels
    for col in CHEXPERT_PATHOLOGY_COLUMNS:
        if col in df.columns:
            if treat_uncertain_as == "positive":
                df[col] = df[col].replace(-1.0, 1.0)
            elif treat_uncertain_as == "negative":
                df[col] = df[col].replace(-1.0, 0.0)
            # Fill NaN with 0 (not mentioned = negative)
            df[col] = df[col].fillna(0.0)

    # Compute is_abnormal: any pathology label is positive (excluding "No Finding")
    pathology_cols = [c for c in CHEXPERT_PATHOLOGY_COLUMNS if c in df.columns and c != CHEXPERT_NO_FINDING_LABEL]
    df["is_abnormal"] = (df[pathology_cols] == 1.0).any(axis=1)
    df["is_normal"] = ~df["is_abnormal"]

    # Extract patient ID from path (CheXpert format: patient<id>/study<n>/view.jpg)
    df["patient_id"] = df["Path"].apply(lambda x: x.split("/")[1] if "/" in str(x) else str(x))
    df["image_index"] = df["Path"].apply(lambda x: os.path.basename(str(x)))

    # Construct full image path
    if image_root:
        df["image_path"] = df["Path"].apply(lambda x: os.path.join(image_root, str(x)))
    else:
        df["image_path"] = df["Path"]

    return df


class CheXpertDataset(Dataset):
    """
    Dataset for CheXpert anomaly detection.
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
                "path": row.get("Path"),
            }
            return image, label, meta
        return image, label


# =============================================================================
# MIMIC-CXR Dataset Support
# =============================================================================

MIMIC_PATHOLOGY_COLUMNS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def load_mimic_cxr_metadata(
    labels_csv: str,
    metadata_csv: str,
    image_root: Optional[str] = None,
    treat_uncertain_as: str = "negative",
) -> pd.DataFrame:
    """
    Load MIMIC-CXR metadata and labels.

    Args:
        labels_csv: Path to mimic-cxr-2.0.0-chexpert.csv (CheXpert-style labels)
        metadata_csv: Path to mimic-cxr-2.0.0-metadata.csv
        image_root: Root directory for MIMIC-CXR-JPG images
        treat_uncertain_as: How to treat -1 labels ('positive', 'negative', or 'ignore')

    Returns:
        DataFrame with is_normal, is_abnormal, and image_path columns
    """
    labels_df = pd.read_csv(labels_csv)
    meta_df = pd.read_csv(metadata_csv)

    # Merge labels with metadata
    df = pd.merge(
        meta_df,
        labels_df,
        on=["subject_id", "study_id"],
        how="left",
    )

    # Handle uncertain labels
    for col in MIMIC_PATHOLOGY_COLUMNS:
        if col in df.columns:
            if treat_uncertain_as == "positive":
                df[col] = df[col].replace(-1.0, 1.0)
            elif treat_uncertain_as == "negative":
                df[col] = df[col].replace(-1.0, 0.0)
            df[col] = df[col].fillna(0.0)

    # Compute is_abnormal
    pathology_cols = [c for c in MIMIC_PATHOLOGY_COLUMNS if c in df.columns]
    df["is_abnormal"] = (df[pathology_cols] == 1.0).any(axis=1)
    df["is_normal"] = ~df["is_abnormal"]

    # Patient ID
    df["patient_id"] = df["subject_id"].astype(str)
    df["image_index"] = df["dicom_id"].astype(str) + ".jpg"

    # Construct image path: p<first 2 digits of subject>/p<subject_id>/s<study_id>/<dicom_id>.jpg
    if image_root:
        def build_path(row):
            subject_str = str(row["subject_id"])
            prefix = "p" + subject_str[:2]
            patient_dir = "p" + subject_str
            study_dir = "s" + str(row["study_id"])
            filename = str(row["dicom_id"]) + ".jpg"
            return os.path.join(image_root, prefix, patient_dir, study_dir, filename)

        df["image_path"] = df.apply(build_path, axis=1)
    else:
        df["image_path"] = df["dicom_id"].astype(str) + ".jpg"

    return df


class MIMICCXRDataset(Dataset):
    """
    Dataset for MIMIC-CXR anomaly detection.
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
                "subject_id": row.get("subject_id"),
                "study_id": row.get("study_id"),
            }
            return image, label, meta
        return image, label


# =============================================================================
# Cross-Dataset Evaluation Utilities
# =============================================================================


def load_dataset_metadata(
    dataset_name: str,
    csv_path: str,
    image_root: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load dataset metadata by name.

    Args:
        dataset_name: One of 'nih', 'chexpert', 'mimic'
        csv_path: Path to main CSV file
        image_root: Root directory for images
        **kwargs: Additional arguments for specific dataset loaders

    Returns:
        DataFrame with standardized columns: image_path, is_normal, is_abnormal, patient_id
    """
    if dataset_name.lower() == "nih":
        return load_nih_metadata(csv_path, image_root)
    elif dataset_name.lower() == "chexpert":
        return load_chexpert_metadata(csv_path, image_root, **kwargs)
    elif dataset_name.lower() == "mimic":
        labels_csv = kwargs.get("labels_csv")
        if labels_csv is None:
            raise ValueError("MIMIC-CXR requires labels_csv argument")
        return load_mimic_cxr_metadata(labels_csv, csv_path, image_root, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_dataset(
    dataset_name: str,
    dataframe: pd.DataFrame,
    transform: Optional[Callable] = None,
    return_metadata: bool = False,
) -> Dataset:
    """
    Create a dataset object by name.

    Args:
        dataset_name: One of 'nih', 'chexpert', 'mimic'
        dataframe: Metadata DataFrame
        transform: Image transform
        return_metadata: Whether to return metadata in __getitem__

    Returns:
        Dataset object
    """
    if dataset_name.lower() == "nih":
        return NIHChestXrayDataset(dataframe, transform, return_metadata)
    elif dataset_name.lower() == "chexpert":
        return CheXpertDataset(dataframe, transform, return_metadata)
    elif dataset_name.lower() == "mimic":
        return MIMICCXRDataset(dataframe, transform, return_metadata)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def compute_dataset_statistics(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for a dataset.

    Args:
        df: DataFrame with is_normal and patient_id columns

    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_images": len(df),
        "normal_images": int(df["is_normal"].sum()),
        "abnormal_images": int(df["is_abnormal"].sum()),
        "normal_ratio": float(df["is_normal"].mean()),
        "unique_patients": int(df["patient_id"].nunique()),
    }
    stats["abnormal_ratio"] = 1.0 - stats["normal_ratio"]
    return stats


def print_cross_dataset_summary(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    source_name: str = "Source",
    target_name: str = "Target",
) -> None:
    """
    Print a summary comparing source and target datasets for cross-dataset evaluation.
    """
    source_stats = compute_dataset_statistics(source_df)
    target_stats = compute_dataset_statistics(target_df)

    print(f"\n{'='*60}")
    print(f"Cross-Dataset Evaluation: {source_name} -> {target_name}")
    print(f"{'='*60}")

    print(f"\n{source_name} (Training):")
    print(f"  Total images: {source_stats['total_images']:,}")
    print(f"  Normal: {source_stats['normal_images']:,} ({source_stats['normal_ratio']:.1%})")
    print(f"  Abnormal: {source_stats['abnormal_images']:,} ({source_stats['abnormal_ratio']:.1%})")
    print(f"  Unique patients: {source_stats['unique_patients']:,}")

    print(f"\n{target_name} (Testing):")
    print(f"  Total images: {target_stats['total_images']:,}")
    print(f"  Normal: {target_stats['normal_images']:,} ({target_stats['normal_ratio']:.1%})")
    print(f"  Abnormal: {target_stats['abnormal_images']:,} ({target_stats['abnormal_ratio']:.1%})")
    print(f"  Unique patients: {target_stats['unique_patients']:,}")
    print(f"{'='*60}\n")

