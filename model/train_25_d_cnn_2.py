"""
train_25d_cnn.py — v4
====================
Usage example
-------------
> bash
python train_25d_cnn.py \
    --meta /kaggle/input/metadata/cq500ct_metadata.parquet \
    --arch efficientnet_b2
    --pos_weight 5.2 --epochs 15 --batch 16
"""
from __future__ import annotations

#import os
import argparse
import json
import math
from pathlib import Path
from typing import Optional, List, Sequence

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import models

import torchmetrics as tm
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,  # PR‑AUC
    BinaryAccuracy,
    BinaryRecall,            # Sensitivity
    BinarySpecificity,       # Specificity
)

from data_loader_25d import CQ500DataLoader25D

# -----------------------------------------------------
# Model builder – patches first conv to accept 3/9 channels
# -----------------------------------------------------

def build_backbone(
        name: str = "resnet18",
        in_channels: int = 9,
        pretrained: bool = True
) -> nn.Module:
    """ Build a Model Backbone """
    model = getattr(models, name)(weights="IMAGENET1K_V1" if pretrained else None)
    if in_channels != 3:
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        # Reuse ImageNet weights if divisible by 3
        if in_channels % 3 == 0:
            with torch.no_grad():
                repeat = in_channels // 3
                new_conv.weight[:] = old_conv.weight.repeat(1, repeat, 1, 1) / repeat
        model.conv1 = new_conv
    # Replace classifier head
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, 1)
    return model

# -----------------------------------------------------
# Trainer class
# -----------------------------------------------------

class Trainer:
    """ Model Trainer """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        patience: int = 3,
        pos_weight: Optional[float] = None,
        out_dir: Path = Path("runs/exp"),
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scaler = GradScaler()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.patience = patience
        self.best_auc = -math.inf
        self.no_improve = 0
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        pos_w = None if pos_weight is None else torch.tensor([pos_weight], device=device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        base = {
            "AUROC": BinaryAUROC(),
            "PR_AUC": BinaryAveragePrecision(),
            "Accuracy": BinaryAccuracy(),
            "Sensitivity": BinaryRecall(),
            "Specificity": BinarySpecificity(),
        }
        self.train_metrics = tm.MetricCollection(base).to(device)
        self.val_metrics = self.train_metrics.clone(prefix="val_")

    # --------------------
    def _run_loader(self, loader: DataLoader, train: bool):
        metrics = self.train_metrics if train else self.val_metrics
        metrics.reset()
        mode = "train" if train else "val"
        self.model.train(train)
        total_loss = 0.0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.float().unsqueeze(1).to(self.device, non_blocking=True)
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = self.model(x)
                loss = self.criterion(logits, y)
            if train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            metrics.update(logits.sigmoid(), y)
            total_loss += loss.item() * x.size(0)
        stats = metrics.compute()
        stats[f"{mode}_loss"] = total_loss / len(loader.dataset)
        return stats

    def fit(self, epochs: int = 10):
        """ Fit the Model """
        for ep in range(1, epochs + 1):
            tr = self._run_loader(self.train_loader, train=True)
            vl = self._run_loader(self.val_loader, train=False)
            self._print_epoch(ep, tr | vl)
            auc = vl["val_AUROC"].item()
            if auc > self.best_auc:
                self.best_auc = auc
                self.no_improve = 0
                torch.save(self.model.state_dict(), self.out_dir / "best.pt")
            else:
                self.no_improve += 1
            if self.no_improve >= self.patience:
                print(f"Early stopping at epoch {ep}, best AUROC {self.best_auc:.4f}")
                break

    @staticmethod
    def _print_epoch(ep: int, stats):
        flat = {k: (v if isinstance(v, float) else v.item()) for k, v in stats.items()}
        print(
            json.dumps({"epoch": ep, **{k: f"{v:.4f}" for k, v in flat.items()}} \
            , separators=(",", ":"))
        )

# -----------------------------------------------------
# Helpers to read patients split
# -----------------------------------------------------

def read_name_file(path: str | Path) -> List[str]:
    """ Extract names from parquet split file """
    df = pd.read_parquet(path)
    if "name" not in df.columns:
        raise ValueError(f"{path} must contain a 'name' column")
    return df["name"].astype(str).tolist()

def indices_from_names(meta_df: pd.DataFrame, names: Sequence[str]) -> list[int]:
    """ Make indices from names """
    idx = meta_df.index[meta_df["name"].astype(str).isin(names)].tolist()
    missing = set(names) - set(meta_df.loc[idx, "name"].astype(str))
    if missing:
        raise ValueError(f"Names not found in metadata: {missing}")
    return idx


# -----------------------------------------------------
# CLI
# -----------------------------------------------------

def parse_args():
    """ Argument Parser """
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True, help="Path to .parquet files")
    p.add_argument("--train_names", required=True, help="Path to train.parquet files")
    p.add_argument("--val_names", required=True, help="Path to val.parquet files")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--in_channels", type=int, default=None, \
                   help="Override channel count")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--pos_weight", type=float)
    p.add_argument("--arch", "--backbone", dest="backbone", default="resnet18", \
                   help="CNN backbone (alias --arch)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    """ Main Function """
    args = parse_args()

    main_meta_df = pd.read_parquet(args.meta)
    train_n = read_name_file(args.train_names)
    val_n = read_name_file(args.val_names)
    overlap = set(train_n) & set(val_n)
    if overlap:
        raise ValueError(f"Leakage - Patients present in both train and val splits: {overlap}")

    train_idx = indices_from_names(main_meta_df, train_n)
    val_idx = indices_from_names(main_meta_df, val_n)

    train_ds = CQ500DataLoader25D(
        metadata_path="", indices=train_idx
    )
    val_ds = CQ500DataLoader25D(
        metadata_path="", indices=val_idx
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_backbone(args.backbone)

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        pos_weight=args.pos_weight,
    )
    trainer.fit(args.epochs)


if __name__ == "__main__":
    main()
