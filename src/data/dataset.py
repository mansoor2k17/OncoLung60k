"""
PyTorch Dataset for OncoLung60K and LungHist700.

Both datasets follow the same CSV schema:
    filepath, label, patient_id[, fold]

Where:
    filepath:   relative path to the image (under data_root)
    label:      integer class label (0..N-1)
    patient_id: anonymised patient identifier
    fold:       optional, fold assignment for CV splits
"""
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HistoPathologyDataset(Dataset):
    """Histopathology patch dataset with patient-level metadata.

    Args:
        df: DataFrame with at least columns [filepath, label].
        data_root: Root directory where filepath is relative to.
        transform: Optional transform applied to PIL images.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        required = {"filepath", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self.data_root / row["filepath"]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])

    @property
    def labels(self):
        return self.df["label"].values

    @property
    def patient_ids(self):
        if "patient_id" not in self.df.columns:
            raise ValueError("DataFrame has no 'patient_id' column")
        return self.df["patient_id"].values
