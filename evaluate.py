"""
Evaluation script — loads the best saved model and runs inference
on the held-out test set.

Outputs:
  - Overall accuracy, precision, recall, F1 (per class + macro)
  - Confusion matrix saved to checkpoints/confusion_matrix.png
  - Full classification report printed to console
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import config
from dataset import get_dataloaders
from model import build_model


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(preds, labels):
    cm  = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=config.CLASSES)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set", fontsize=14)
    plt.tight_layout()

    path = os.path.join(config.CHECKPOINT_DIR, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"Confusion matrix saved to {path}")


def main():
    device = torch.device(config.DEVICE)

    # ── Load model ────────────────────────────────────────────────────────────
    model      = build_model().to(device)
    ckpt_path  = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded weights from {ckpt_path}")

    # ── Test data ─────────────────────────────────────────────────────────────
    _, _, test_loader = get_dataloaders()

    # ── Predict ───────────────────────────────────────────────────────────────
    preds, labels = get_predictions(model, test_loader, device)

    # ── Report ────────────────────────────────────────────────────────────────
    acc = (preds == labels).mean()
    print(f"\nTest Accuracy: {acc:.4f}\n")
    print(classification_report(labels, preds, target_names=config.CLASSES))

    plot_confusion_matrix(preds, labels)


if __name__ == "__main__":
    main()
