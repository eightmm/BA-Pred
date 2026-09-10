from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class ProcessedComplexDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str | None = None):
        manifest = Path(manifest)
        self.df = pd.read_csv(manifest)
        if split is not None:
            self.df = self.df[self.df["split"].astype(str) == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No samples for split={split!r} in {manifest}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = Path(self.df.iloc[idx]["graph_path"])
        return torch.load(path, weights_only=False)
