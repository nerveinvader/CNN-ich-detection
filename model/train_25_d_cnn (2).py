from __future__ import annotations

"""
train_25d_cnn.py — v3
====================
Major tweak: **`--arch` is now an accepted alias for `--backbone`.**

Why?
-----
My chat reply mentioned `--arch`, while the previous script exposed `--backbone`.
Both are now interchangeable so your CLI examples continue to work.

Usage example
-------------
```bash
python train_25d_cnn.py \
    --data_dir /kaggle/input/cq500 --meta metadata.parquet \
    --arch efficientnet_b2 \  # or --backbone efficientnet_b2
    --in_channels 9 --pos_weight 5.2 --epochs 15
```

No other logic changed.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional

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

from cq500_25d_dataloader import CQ500SliceTripletDataset

# -----------------------------------------------------
# Model builder – patches first conv to accept 3/9 channels
# -----------------------------------------------------

def build_backbone(name: str = "resnet18", in_channels: int = 9, pretrained: bool = True) -> nn.Module:
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
    def _print_epoch(ep, stats):
        flat = {k: (v if isinstance(v, float) else v.item()) for k, v in stats.items()}
        print(json.dumps({"epoch": ep, **{k: f"{v:.4f}" for k, v in flat.items()}}, separators=(",", ":")))

# -----------------------------------------------------
# CLI
# -----------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--in_channels", type=int, default=9)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--pos_weight", type=float)
    p.add_argument("--arch", "--backbone", dest="backbone", default="resnet18", help="CNN backbone (alias --arch)")
    return p.parse_args()


def main():
    args = parse_args()
    train_ds = CQ500SliceTripletDataset(args.data_dir, args.meta, split="train", in_channels=args.in_channels)
    val_ds = CQ500SliceTripletDataset(args.data_dir, args.meta, split="val", in_channels=args.in_channels)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_backbone(args.backbone, in_channels=args.in_channels)

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
