"""
Entry point — runs training followed by evaluation.

Usage:
    python main.py
"""

from train import main as train_main
from evaluate import main as eval_main

if __name__ == "__main__":
    print("=" * 60)
    print("  Brain Tumor MRI — Vision Transformer (from scratch)")
    print("=" * 60)

    print("\n[1/2] Training ...\n")
    train_main()

    print("\n[2/2] Evaluating on test set ...\n")
    eval_main()
