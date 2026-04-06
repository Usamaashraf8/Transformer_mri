"""
Training script for the transformer-based brain tumor classifier.

Schedule:
  - Epoch 1: train classifier head only
  - Remaining epochs: fine-tune full backbone with lower LR
"""

import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import config
from dataset import get_dataloaders
from model import build_model, unfreeze_backbone


def get_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if config.DEVICE == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _classifier_key():
    return "head" if config.MODEL_NAME == "swin_t" else "heads.head"


def build_optimizer(model, backbone_trainable):
    classifier_key = _classifier_key()

    if not backbone_trainable:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params,
            lr=config.HEAD_LR,
            weight_decay=config.WEIGHT_DECAY,
        )

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if classifier_key in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.BACKBONE_LR},
            {"params": head_params, "lr": config.HEAD_LR},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )


def build_scheduler(optimizer, epochs_remaining):
    if epochs_remaining <= 0:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs_remaining,
        eta_min=config.BACKBONE_LR * 0.1,
    )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRAD_CLIP_NORM)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

        if step % 50 == 0 or step == len(loader):
            print(
                f"    Step [{step:03d}/{len(loader):03d}] "
                f"Loss: {loss.item():.4f}  Acc: {correct / total:.4f}"
            )

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def plot_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs, label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(config.CHECKPOINT_DIR, "training_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"Training curves saved to {path}")


def main():
    torch.manual_seed(config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders()

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    backbone_trainable = config.FREEZE_BACKBONE_EPOCHS == 0
    optimizer = build_optimizer(model, backbone_trainable=backbone_trainable)
    scheduler = build_scheduler(
        optimizer,
        epochs_remaining=max(config.EPOCHS - config.FREEZE_BACKBONE_EPOCHS, 0),
    ) if backbone_trainable else None

    best_val_acc = 0.0
    best_val_loss = math.inf
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, config.EPOCHS + 1):
        if epoch == config.FREEZE_BACKBONE_EPOCHS + 1 and config.FREEZE_BACKBONE_EPOCHS > 0:
            unfreeze_backbone(model)
            optimizer = build_optimizer(model, backbone_trainable=True)
            scheduler = build_scheduler(
                optimizer,
                epochs_remaining=config.EPOCHS - config.FREEZE_BACKBONE_EPOCHS,
            )
            print("Backbone unfrozen for full fine-tuning.")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        current_lrs = ", ".join(f"{group['lr']:.2e}" for group in optimizer.param_groups)
        print(
            f"Epoch [{epoch:02d}/{config.EPOCHS}] "
            f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f} | "
            f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f} | "
            f"LRs: {current_lrs}"
        )

        if (vl_acc > best_val_acc) or (vl_acc == best_val_acc and vl_loss < best_val_loss):
            best_val_acc = vl_acc
            best_val_loss = vl_loss
            path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), path)
            print(f"  -> Best model saved (val acc: {best_val_acc:.4f}, val loss: {best_val_loss:.4f})")

    plot_curves(train_losses, val_losses, train_accs, val_accs)
    print(
        f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f} | "
        f"Best Val Loss: {best_val_loss:.4f}"
    )


if __name__ == "__main__":
    main()
