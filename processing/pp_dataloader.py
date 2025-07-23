"""
Preprocessed Dataloader
File name:		pp_dataloader.py
Description:	Feed tensor as input to the Model.
Created:		5th/JULY/2025
Author:			nerveinvader
"""

# Import
import torch
import pandas as pd

# Class
class PreprocessedTensorDataset(torch.utils.data.Dataset):
    """
    Load preprocessed tensors from .pt files \
    and feed to model.
    """
    def __init__(self, tensor_files, meta_files):
        self.tensors = []
        self.labels = []
        for tfile, mfile in zip(tensor_files, meta_files):
            self.tensors.append(torch.load(tfile))  # shape: (N, 9, H, W)
            meta = pd.read_csv(mfile)
            self.labels.extend(meta["label"].tolist())  # or however you store labels
        self.tensors = torch.cat(self.tensors, dim=0)
    def __len__(self):
        return len(self.tensors)
    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]
