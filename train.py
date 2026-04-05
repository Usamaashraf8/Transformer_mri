"""
Training script for the Vision Transformer on Brain Tumor MRI dataset.

Features:
  - AdamW optimizer with cosine annealing LR schedule
  - Best model saved to checkpoints/best_model.pth
  - Training & validation loss / accuracy logged per epoch
  - Loss and accuracy curves saved to checkpoints/training_curves.png
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import config
from dataset import get_dataloaders
from model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


def plot_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses,   label="Val Loss")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs,   label="Val Acc")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()

    plt.tight_layout()
    path = os.path.join(config.CHECKPOINT_DIR, "training_curves.png")
    plt.savefig(path)
    plt.close()
    print(f"Training curves saved to {path}")


def main():
    torch.manual_seed(config.SEED)
    device = torch.device(config.DEVICE)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders()

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, config.EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)

        print(
            f"Epoch [{epoch:02d}/{config.EPOCHS}] "
            f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f} | "
            f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f}"
        )

        # save best model
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), path)
            print(f"  → Best model saved (val acc: {best_val_acc:.4f})")

    plot_curves(train_losses, val_losses, train_accs, val_accs)
    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
