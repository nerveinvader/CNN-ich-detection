"""
File name:		data_loader_25D.py
Description:	Loads DCM files from a metadata parquet file.
Created:		22nd/JULY/2025
Author:			nerveinvader
"""
# Imports
from pathlib import Path
from typing import List, Sequence, Tuple, Dict, Optional

import os
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
            if isinstance(indices[0], int):
                self.patients = [self.patients[i] for i in indices]
            elif isinstance(indices[0], str):
                name_to_idx = {name: (name, pdf) for name, pdf in self.patients}
                self.patients = [name_to_idx[name] for name in indices if name in name_to_idx]
            else:
                raise ValueError(
                    "The argument for indices must be a list of int or str (patient names)"
                )

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

        pdf_sorted = pdf.sort_values("instance_num")
        n_slices = len(pdf_sorted)

        # Determine neighboring indices with optional edge replication
        def safe_index(i: int) -> int:
            if self.replicate_edge:
                return max(0, min(i, n_slices - 1))
            return i
        prev_idx = safe_index(slice_idx - 1)
        curr_idx = safe_index(slice_idx)
        next_idx = safe_index(slice_idx + 1)

        if self.cache_enabled and patient_name in self._series_cache:
            volume = self._series_cache[patient_name]   # (S, 3, H, W) torch.Tensor
            x_stack = torch.stack(
                [volume[prev_idx], volume[curr_idx], volume[next_idx]], dim=0
            )  # (3, 3, H, W)
        else:
            # Only load the three required slices from disk
            paths = pdf_sorted["path"].tolist()
            dcm_loader = DCMLoader()
            needed_indices = [prev_idx, curr_idx, next_idx]
            x_stack = torch.stack(
                [dcm_loader.dcm_to_tensor(path=paths[i]) for i in needed_indices], dim=0
            )  # (3, 3, H, W)

        # Merge HU channel and slice context dims (slice_ctx, hu_ch, H, W)
        # The convention is to keep the slice context and drop the HU channel.
        # Because we windowed into HU spaces, each slice context acts as one channel.
        # x = x_stack.mean(dim=1)   # (3, H, W) - dim=1 for HU channel
        x = x_stack.permute(1, 0, 2, 3).reshape(-1, x_stack.shape[2], x_stack.shape[3])  # (9, H, W)

        if self.transform:
            x = self.transform(x)

        return x, label

    def preprocess(
        self, output_dir: str,
        chunk_size: int = 100,
        max_ram_gb: int = 12
    ) -> None:
        """
        Preprocess DICOMs: convert to tensors and save to disk \
        in small chunks to avoid exceeding RAM.
        Args:
            output_dir: Directory to save tensors (will be created if not exists).
            chunk_size: Number of slices to process and save at once.
            max_ram_gb: Maximum RAM (in GB) to use for holding tensors in memory at once.
        """
        os.makedirs(output_dir, exist_ok=True)
        dcm_loader = DCMLoader()
        tensor_buffer = []
        meta_buffer = []
        buffer_bytes = 0
        max_bytes = max_ram_gb * 1024 ** 3
        chunk_idx = 0
        print(f"[Preprocess] Saving tensors to {output_dir} \
            in chunks of {chunk_size} slices, max RAM {max_ram_gb} GB...")
        for patient_name, pdf in self.patients:
            pdf_sorted = pdf.sort_values("instance_num")
            for _, row in pdf_sorted.iterrows():
                dcm_path = row["path"]
                instance_num = row["instance_num"]
                tensor = dcm_loader.dcm_to_tensor(dcm_path)
                tensor = tensor.to(torch.float16)  # Convert to float16 for efficient storage
                tensor_buffer.append(tensor)
                meta_buffer.append((patient_name, instance_num, dcm_path))
                buffer_bytes += tensor.element_size() * tensor.nelement()
                # Save chunk if buffer is large enough
                if len(tensor_buffer) >= chunk_size or buffer_bytes >= max_bytes:
                    chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx:05d}.pt")
                    meta_file = os.path.join(output_dir, f"chunk_{chunk_idx:05d}_meta.csv")
                    torch.save(torch.stack(tensor_buffer), chunk_file)
                    pd.DataFrame(
                        meta_buffer, columns=["patient_name", "instance_num", "dcm_path"]
                    ).to_csv(meta_file, index=False)
                    print(f"  Saved {len(tensor_buffer)} slices to {chunk_file}")
                    tensor_buffer.clear()
                    meta_buffer.clear()
                    buffer_bytes = 0
                    chunk_idx += 1
        # Save any remaining tensors
        if tensor_buffer:
            chunk_file = os.path.join(output_dir, f"chunk_{chunk_idx:05d}.pt")
            meta_file = os.path.join(output_dir, f"chunk_{chunk_idx:05d}_meta.csv")
            torch.save(torch.stack(tensor_buffer), chunk_file)
            pd.DataFrame(
                meta_buffer, columns=["patient_name", "instance_num", "dcm_path"]
            ).to_csv(meta_file, index=False)
            print(f"  Saved {len(tensor_buffer)} slices to {chunk_file}")
        print("[Preprocess] Done.")
