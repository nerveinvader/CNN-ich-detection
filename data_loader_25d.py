"""
File name:		data_loader_25D.py
Description:	Loads DCM files from a metadata parquet file.
Created:		22nd/JULY/2025
Author:			nerveinvader
"""
# Imports
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dcm_to_tensor import DCMLoader


# Class
class CQ500DataLoader25D(Dataset):
    """
    Iterable dataset that yields (x, y) pairs.
    x = (3, H, W) windows for slice i, with slice context \
    [i-1, i, i+1].
    y = series-level ICH label.
    Optionally caches complete series volumes in RAM.
    """
    def __init__(
        self,
        metadata_path: str | Path,
        indices: Optional[Sequence[int]] = None,
        transform = None,
        cache: bool = False,
        replicate_edge: bool = True,
    ) -> None:
        super().__init__()
        self.df = pd.read_parquet(metadata_path)

        ## Restric to desired subset of patients (int idx list)
        grouped = self.df.groupby("name", sort = False)
        self.patients: List[Tuple[str, pd.DataFrame]] = list(grouped)
        if indices is not None:
            self.patients = [self.patients[i] for i in indices]

        ## Flatten into sample idx list (patient_idx, slice_idx)
        self.sample_index: List[Tuple[int, int]] = []
        for p_idx, (_, pdf) in enumerate(self.patients):
            pdf_sorted = pdf.sort_values("instance_num")
            n_slices = len(pdf_sorted)
            for s_idx in range(n_slices):
                self.sample_index.append((p_idx, s_idx))

        self.transform = transform
        self.cache_enabled = cache
        self.replicate_edge = replicate_edge
        self._series_cache: Dict[str, torch.Tensor] = {}

        ## Cache if caching is set to True
        if self.cache_enabled:
            self._populate_cache()

    def __len__(self) -> int:
        return len(self.sample_index)

    def _populate_cache(self) -> None:
        """
        Preload all series volumes into RAM (lazy-converted to torch.Tensor).
        """
        print("[CQ500DataLoader25D] Caching series data …")
        for patient_name, pdf in self.patients:
            pdf_sorted = pdf.sort_values("instance_num")
            dcm_loader = DCMLoader()
            slices = [dcm_loader.dcm_to_tensor(path = p) for p in pdf_sorted["path"].tolist()]
            vol_np = np.stack(slices, axis=0)  # (S, 3, H, W)
            self._series_cache[patient_name] = torch.from_numpy(vol_np)
        print(f"→ Cached {len(self._series_cache)} series.")

    ## Public API
    def enable_cache(self) -> None:
        """Enable caching automatically."""
        if not self.cache_enabled:
            self.cache_enabled = True
            self._populate_cache()
    def disable_cache(self) -> None:
        """Disable caching automatically."""
        if self.cache_enabled:
            self.cache_enabled = False
            self._series_cache.clear()

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        patient_idx, slice_idx = self.sample_index[idx]
        patient_name, pdf = self.patients[patient_idx]
        label = torch.tensor(pdf["ICH-majority"].iloc[0], dtype=torch.float32)

        # Load the volume form cache or disk on-the-fly
        if self.cache_enabled and patient_name in self._series_cache:
            volume = self._series_cache[patient_name]   # (S, 3, H, W) torch.Tensor
        else:
            pdf_sorted = pdf.sort_values("instance_num")
            paths = pdf_sorted["path"].tolist()
            dcm_loader = DCMLoader()
            slices = [dcm_loader.dcm_to_tensor(path = p) for p in paths]
            volume = torch.from_numpy(np.stack(slices, axis = 0))   # (S, 3, H, W)

        n_slices = volume.shape[0]

        # Determine neighboring indices with optional edge replication
        def safe_index(i: int) -> int:
            if self.replicate_edge:
                return max(0, min(i, n_slices - 1))
            return i
        prev_idx = safe_index(slice_idx - 1)
        next_idx = safe_index(slice_idx + 1)

        x_stack = torch.stack(
            [volume[prev_idx], volume[slice_idx], volume[next_idx]], dim = 0
        )   # (3, 3, H, W)
        ## Merge HU channel and slice context dims (slice_ctx, hu_ch, H, W).\
        ## The convention is to keep the slice context and drop the HU channel.\
        ## Because we windowed into HU spaces, each slice context acts as one channel.\
        x = x_stack.mean(dim = 1)   # (3, H, W) - dim=1 for HU channel

        if self.transform:
            x = self.transform(x)

        return x, label
