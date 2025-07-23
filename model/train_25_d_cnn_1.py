"""
Model pipeline
"""
# 2p5d_cnn_train.py
import torch.nn as nn
import torch, torchvision as tv
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.classification import (
    BinaryAUROC, BinaryAveragePrecision,
    BinaryRecall, BinarySpecificity
)
from tqdm import tqdm


# ---------- 1. Model ----------------------------------------------------------
def _replace_first_conv(m: nn.Module, in_ch: int) -> None:
    """Replace the first conv to accept `in_ch` channels; keeps pretrained weights."""
    old = m.conv1
    new = nn.Conv2d(in_ch, old.out_channels,
                    kernel_size=old.kernel_size,
                    stride=old.stride,
                    padding=old.padding,
                    bias=old.bias is not None)
    # repeat / average weights to new conv (simple heuristic)
    with torch.no_grad():
        repeat = in_ch // old.in_channels
        new.weight.copy_(old.weight.repeat(1, repeat, 1, 1) / repeat)
    m.conv1 = new


_BACKBONES = {
    "resnet18": lambda ic: tv.models.resnet18(weights="IMAGENET1K_V1"),
    "resnet34": lambda ic: tv.models.resnet34(weights="IMAGENET1K_V1"),
    "densenet121": lambda ic: tv.models.densenet121(weights="IMAGENET1K_V1"),
    # add more here...
}


class TwoPointFiveD(nn.Module):
    """ Backbone + Linear head for binary ICH classification """
    def __init__(self, backbone_name: str = "resnet18",
                 in_channels: int = 9):            # 3 HU ch × 3 slices
        super().__init__()
        backbone = _BACKBONES[backbone_name](in_channels)
        _replace_first_conv(backbone, in_channels)

        if hasattr(backbone, "fc"):               # ResNet-style
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif hasattr(backbone, "classifier"):     # DenseNet-style
            feat_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError("Add support for this backbone")

        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, 1)  # logits

    def forward(self, x):
        """ Set the model and return Classifier """
        x = self.backbone(x)
        return self.classifier(x).squeeze(1)      # (N,) logits


# ---------- 2. Metrics helpers -----------------------------------------------
def make_metric_dict(device):
    """ Make metrics dictionary """
    return {
        "auroc": BinaryAUROC().to(device),
        "prauc": BinaryAveragePrecision().to(device),
        "sens":  BinaryRecall(threshold=0.5).to(device),        # sensitivity
        "spec":  BinarySpecificity(threshold=0.5).to(device)    # specificity
    }


def update_metrics(metrics, preds, targets):
    """ Update the metrics """
    for m in metrics.values():
        m.update(preds, targets.int())


def compute_and_reset(metrics):
    """ Compute Metrics """
    out = {k: float(v.compute()) for k, v in metrics.items()}
    for v in metrics.values():
        v.reset()
    return out


# ---------- 3. Train / Val loops ---------------------------------------------
@torch.no_grad()
def validate(model, loader, loss_fn, device, metrics):
    """ Validate """
    model.eval()
    loop = tqdm(loader, desc="val", leave=False)
    total_loss = 0.0
    for x, y in loop:
        x, y = x.to(device, non_blocking=True), y.float().to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * y.size(0)

        probs = torch.sigmoid(logits)
        update_metrics(metrics, probs, y)
    stats = compute_and_reset(metrics)
    stats["loss"] = total_loss / len(loader.dataset)
    return stats


def train_one_epoch(model, loader, optimizer, scaler, loss_fn,
                    device, metrics, epoch):
    """ Train one Epoch """
    model.train()
    loop = tqdm(loader, desc=f"train {epoch}")
    for x, y in loop:
        x, y = x.to(device, non_blocking=True), y.float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probs = torch.sigmoid(logits.detach())
        update_metrics(metrics, probs, y)

        loop.set_postfix(loss=loss.item())
    return compute_and_reset(metrics)


# ---------- 4. Fit routine with early-stopping -------------------------------
def fit(model, train_loader, val_loader,
        epochs=20, patience=3, lr=3e-4, weight_decay=1e-4,
        save_path="best_auc.pt"):
    """ Fit the model """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler()

    best_auc, epochs_no_improve = 0.0, 0
    for ep in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, opt, scaler,
                                        loss_fn, device,
                                        make_metric_dict(device), ep)
        val_metrics = validate(model, val_loader, loss_fn, device,
                               make_metric_dict(device))

        print(f"\nEpoch {ep}: "
              f"AUROC {val_metrics['auroc']:.4f}  "
              f"PRAUC {val_metrics['prauc']:.4f}  "
              f"Sens {val_metrics['sens']:.4f}  "
              f"Spec {val_metrics['spec']:.4f}")

        cur_auc = val_metrics["auroc"]
        if cur_auc > best_auc:
            best_auc = cur_auc
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            print("  ↑ saved new best weights")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early-stopping (no AUROC improvement for {patience} epochs)")
                break


# ---------- 5. Example usage --------------------------------------------------
if __name__ == "__main__":
    # Assume you already have:
    #   train_idx.parquet  val_idx.parquet  full_metadata.parquet
    # And a DataLoader class `CQ500DataLoader25D` returning
    #   x: (9, H, W) float32  |  y: binary label 0/1  (ICH-majority)

    from data_loader_25d import CQ500DataLoader25D   # ← adjust import

    train_set = CQ500DataLoader25D("full_metadata.parquet",
                                   indices="train_idx.parquet")
    val_set   = CQ500DataLoader25D("full_metadata.parquet",
                                   indices="val_idx.parquet")

    current_train_loader = DataLoader(train_set, batch_size=32,
                              shuffle=True, num_workers=4, pin_memory=True)
    current_val_loader   = DataLoader(val_set, batch_size=32,
                              shuffle=False, num_workers=4, pin_memory=True)

    current_model = TwoPointFiveD(backbone_name="resnet18", in_channels=9)
    fit(current_model, current_train_loader, current_val_loader, epochs=30, patience=3)
