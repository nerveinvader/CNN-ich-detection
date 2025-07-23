from __future__ import annotations

"""
train_25d_cnn.py — v2
====================
* Added **pos_weight** support for class‑imbalance in `BCEWithLogitsLoss`.
* Added **PR‑AUC (Average Precision)**, **Sensitivity (Recall)** and **Specificity** metrics alongside AUROC & Accuracy.
* Metrics handled through a single helper that prints an easy‑to‑read summary after every epoch.
* Minor refactor of the `Trainer` class for clarity.

Run with e.g.:
```
python train_25d_cnn.py \
  --data_dir /kaggle/input/cq500 --meta metadata.parquet \
  --in_channels 9 --pos_weight 5.2 --epochs 15
```
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import models, transforms

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
# Model: first‑conv patcher so ResNet (or any 2‑D CNN) 
# can take arbitrary channel counts (3 or 9)
# -----------------------------------------------------

def build_backbone(name: str = "resnet18", in_channels: int = 9, pretrained: bool = True) -> nn.Module:
    model = getattr(models, name)(weights="IMAGENET1K_V1" if pretrained else None)
    if in_channels != 3:
        # Replace first conv
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        # Kaiming init & copy if feasible
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if in_channels % 3 == 0:
            with torch.no_grad():
                repeat = in_channels // 3
                new_conv.weight[:] = old_conv.weight.repeat(1, repeat, 1, 1) / repeat
        model.conv1 = new_conv
    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model

# -----------------------------------------------------
# Trainer
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
        out_dir: Path = Path("runs/exp")
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.scaler = GradScaler()
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.patience = patience
        self.best_auc = -math.inf
        self.epochs_without_improve = 0
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        pos_w_tensor = None if pos_weight is None else torch.tensor([pos_weight], device=device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)

        # Metrics collection
        self.train_metrics = tm.MetricCollection({
            "AUROC": BinaryAUROC(thresholds=None),
            "PR-AUC": BinaryAveragePrecision(),
            "Accuracy": BinaryAccuracy(),
            "Sensitivity": BinaryRecall(),
            "Specificity": BinarySpecificity()
        }).to(device)
        self.val_metrics = self.train_metrics.clone(prefix="val_")

    # ----------------------
    # One epoch helpers
    # ----------------------
    def _forward(self, batch):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True).float().unsqueeze(1)
        with autocast():
            logits = self.model(x)
            loss = self.criterion(logits, y)
        return loss, logits, y

    def _step_epoch(self, loader: DataLoader, train: bool):
        metrics = self.train_metrics if train else self.val_metrics
        metrics.reset()
        pbar = loader if not train else loader
        mode_str = "Train" if train else "Val  "
        self.model.train(train)
        total_loss = 0.0
        for x, y in pbar:
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            loss, logits, targets = self._forward((x, y))
            if train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            metrics.update(logits.sigmoid(), targets)
            total_loss += loss.item() * x.size(0)
        epoch_loss = total_loss / len(loader.dataset)
        epoch_metrics = metrics.compute()
        epoch_metrics[f"{mode_str}_loss"] = epoch_loss
        return epoch_metrics

    # ----------------------
    # Training loop
    # ----------------------
    def fit(self, epochs: int = 10):
        for epoch in range(1, epochs + 1):
            train_stats = self._step_epoch(self.train_loader, train=True)
            val_stats = self._step_epoch(self.val_loader, train=False)
            self._log_epoch(epoch, train_stats | val_stats)

            current_auc = val_stats["val_AUROC"].item()
            if current_auc > self.best_auc:
                self.best_auc = current_auc
                self.epochs_without_improve = 0
                torch.save(self.model.state_dict(), self.out_dir / "best.pt")
            else:
                self.epochs_without_improve += 1

            if self.epochs_without_improve >= self.patience:
                print(f"Early stopping at epoch {epoch} (no AUROC improvement for {self.patience} epochs)")
                break

    # ----------------------
    # Pretty print helper
    # ----------------------
    @staticmethod
    def _log_epoch(epoch: int, stats: Dict[str, torch.Tensor]):
        stats_fmt = {k: f"{v:.4f}" for k, v in stats.items()}
        print(json.dumps({"epoch": epoch, **stats_fmt}, indent=None))

# -----------------------------------------------------
# CLI / main
# -----------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--in_channels", type=int, default=9, help="3 or 9 depending on dataset loader")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--pos_weight", type=float, default=None, help="Ratio of neg/pos samples. If None no weighting is used.")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Any torchvision model")
    return parser.parse_args()


def main():
    args = parse_args()

    train_ds = CQ500SliceTripletDataset(
        root=args.data_dir,
        metadata_path=args.meta,
        split="train",
        in_channels=args.in_channels,
        cache=False,
    )
    val_ds = CQ500SliceTripletDataset(
        root=args.data_dir,
        metadata_path=args.meta,
        split="val",
        in_channels=args.in_channels,
        cache=False,
    )

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
        out_dir=Path("runs/exp-001"),
    )

    trainer.fit(epochs=args.epochs)


if __name__ == "__main__":
    main()
