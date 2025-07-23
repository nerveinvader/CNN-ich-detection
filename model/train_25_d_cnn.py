from __future__ import annotations

"""
train_25d_cnn.py
================
End‑to‑end training script for a 2.5‑D intracranial‑hemorrhage detector on CQ500 slice
triplets.

Assumptions
-----------
* `cq500_25d_dataloader.py` is in PYTHONPATH and its `CQ500SliceTripletDataset`
  returns a *single* tensor per sample shaped **[C, H, W]** plus a target `float32`
  scalar. `C` may be 3 (merged HU windows) or 9 (3 windows × 3 slices). The code
  auto‑infers it from the first batch.
* CUDA‑capable GPU available.
* `torch>=2.3` and `torchvision`, `torchmetrics`, `tqdm` installed.

Execution example
-----------------
$ python train_25d_cnn.py --metadata metadata.parquet --root /path/to/dicom/root \
                          --split_file split.csv --epochs 30 --batch_size 32

See `python train_25d_cnn.py -h` for all CLI flags.
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as tvm
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from torchmetrics.classification import BinaryAUROC, BinaryAccuracy, BinaryF1Score, BinaryRecall, BinarySpecificity
from tqdm import tqdm

from cq500_25d_dataloader import CQ500SliceTripletDataset

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------

class Slice25DCNN(nn.Module):
    """ResNet18 backbone whose first conv is widened for an arbitrary channel count."""

    def __init__(self, in_channels: int, pretrained: bool = True):
        super().__init__()
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # widen first conv
        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        # copy / repeat weights if pretrained and channel mismatch
        if pretrained:
            with torch.no_grad():
                if in_channels == 3:
                    backbone.conv1.weight.copy_(old_conv.weight)
                elif in_channels > 3:
                    repeat = math.floor(in_channels / 3)
                    remainder = in_channels % 3
                    w = old_conv.weight.repeat(1, repeat, 1, 1)
                    if remainder:
                        w = torch.cat([w, old_conv.weight[:, :remainder]], dim=1)
                    backbone.conv1.weight.copy_(w[:, :in_channels])
                else:  # in_channels < 3
                    backbone.conv1.weight.copy_(old_conv.weight[:, :in_channels])

        # replace classifier head
        num_feats = backbone.fc.in_features
        backbone.fc = nn.Linear(num_feats, 1)  # binary logit
        self.net = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, H, W] → [B]
        return self.net(x).squeeze(1)

# -----------------------------------------------------------------------------
# Train / eval helpers
# -----------------------------------------------------------------------------

def step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    criterion: nn.Module,
    scaler: GradScaler,
    optimizer: optim.Optimizer | None = None,
    device: torch.device = torch.device("cpu"),
):
    """Run one train or val step. If *optimizer* is None we do inference‑only."""
    x, y = batch
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    with (autocast() if scaler is not None else torch.no_grad()):
        logits = model(x)
        loss = criterion(logits, y)

    if optimizer is not None:  # training
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return logits.detach().cpu(), y.detach().cpu(), loss.detach().cpu()


def epoch_loop(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    scaler: GradScaler,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    metrics: Dict[str, torchmetrics.Metric] | None = None,
    desc: str = "",
):
    model.train(mode=optimizer is not None)
    pbar = tqdm(loader, desc=desc, leave=False)
    losses = []

    if metrics:
        for m in metrics.values():
            m.reset()

    for batch in pbar:
        logits, targets, loss = step(model, batch, criterion, scaler, optimizer, device)
        losses.append(loss.item())

        if metrics:
            for name, metric in metrics.items():
                metric.update(torch.sigmoid(logits), targets.int())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    log = {"loss": sum(losses) / len(losses)}
    if metrics:
        log.update({name: metric.compute().item() for name, metric in metrics.items()})
    return log

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--metadata", type=Path, required=True, help="Path to metadata.parquet")
    parser.add_argument("--root", type=Path, required=True, help="Root folder with DICOMs")
    parser.add_argument("--split_file", type=Path, help="Optional CSV with 'patient' and 'fold' columns")
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data for validation if split file not given")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--output", type=Path, default=Path("checkpoints"), help="Folder for model checkpoints")
    parser.add_argument("--patience", type=int, default=3, help="Early‑stopping patience on AUROC")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed‑precision")
    return parser.parse_args()

# -----------------------------------------------------------------------------


def main():
    args = parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    # ---------------- Dataset ----------------
    ds = CQ500SliceTripletDataset(metadata_path=args.metadata, dicom_root=args.root)
    if args.split_file and args.split_file.exists():
        import pandas as pd
        fold_df = pd.read_csv(args.split_file)
        train_idx = fold_df[fold_df["fold"] == "train"].index.tolist()
        val_idx = fold_df[fold_df["fold"] == "val"].index.tolist()
        train_ds = torch.utils.data.Subset(ds, train_idx)
        val_ds = torch.utils.data.Subset(ds, val_idx)
    else:
        val_size = int(len(ds) * args.val_split)
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # infer channel count
    sample, _ = train_ds[0]
    in_channels = sample.shape[0]

    # ---------------- Model ----------------
    model = Slice25DCNN(in_channels=in_channels, pretrained=True).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=not args.no_amp)

    # metrics
    train_metrics = {
        "auroc": BinaryAUROC(thresholds=None),
        "acc": BinaryAccuracy(),
    }
    val_metrics = {
        "auroc": BinaryAUROC(thresholds=None),
        "acc": BinaryAccuracy(),
        "f1": BinaryF1Score(),
        "sens": BinaryRecall(),
        "spec": BinarySpecificity(),
    }

    best_auroc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_log = epoch_loop(
            model,
            train_loader,
            criterion,
            scaler,
            device,
            optimizer=optimizer,
            metrics=train_metrics,
            desc="train",
        )
        val_log = epoch_loop(
            model,
            val_loader,
            criterion,
            scaler,
            device,
            optimizer=None,
            metrics=val_metrics,
            desc="valid",
        )

        print("Train:", {k: f"{v:.4f}" for k, v in train_log.items()})
        print("Valid:", {k: f"{v:.4f}" for k, v in val_log.items()})

        current_auroc = val_log["auroc"]
        if current_auroc > best_auroc:
            best_auroc = current_auroc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "auroc": best_auroc,
            }, args.output / "best_model.pt")
            print(f"\u2714 Saved new best model with AUROC={best_auroc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {args.patience} epochs without AUROC improvement.")
                break

    print(f"Training finished. Best validation AUROC={best_auroc:.4f}")


if __name__ == "__main__":
    main()
