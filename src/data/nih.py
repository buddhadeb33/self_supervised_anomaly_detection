import os
from typing import Callable, Dict, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

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

