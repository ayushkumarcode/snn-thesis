"""
Few-Shot Learning Curves: How data-efficient is the SNN vs ANN?

Trains both SNN and ANN with progressively reduced training sets
(100%, 50%, 25%, 10%, 5%) and plots learning curves. Directly tests
the thesis narrative that SNNs need more data.

Usage:
    python -m experiments.few_shot_learning_curves [--fold 1] [--encoding direct]

    Best run on CSF3 for GPU acceleration (many training runs).
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    RESULTS_DIR, NUM_FOLDS, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, PATIENCE, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import get_encoder
from src.models.snn_model import SpikingCNN
from src.models.ann_model import ConvANN
from src.train import train_snn_epoch, eval_snn, train_ann_epoch, eval_ann

DATA_FRACTIONS = [1.0, 0.5, 0.25, 0.1, 0.05]


def subsample_loader(loader, fraction, seed=42):
    """Create a subsampled version of a DataLoader."""
    dataset = loader.dataset
    n_total = len(dataset)
    n_keep = max(1, int(n_total * fraction))

    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=rng)[:n_keep].tolist()

    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )


def train_and_eval(model_type, fold, encoding, fraction, device, seed=42):
    """Train model with a fraction of data and return best test accuracy."""
    train_loader, test_loader = get_fold_dataloaders(fold, BATCH_SIZE)

    # Subsample training data
    if fraction < 1.0:
        train_loader = subsample_loader(train_loader, fraction, seed=seed)

    n_train = len(train_loader.dataset)
    print(f"  {model_type.upper()} | Fraction={fraction:.0%} | "
          f"Training samples={n_train}")

    if model_type == "snn":
        model = SpikingCNN().to(device)
        encoder = get_encoder(encoding)
    else:
        model = ConvANN().to(device)
        encoder = None

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = torch.nn.CrossEntropyLoss() if model_type == "ann" else None

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        if model_type == "snn":
            train_loss, _ = train_snn_epoch(model, train_loader, optimizer, encoder, device)
            test_loss, test_acc, _, _ = eval_snn(model, test_loader, encoder, device)
        else:
            train_loss, _ = train_ann_epoch(model, train_loader, optimizer, criterion, device)
            test_loss, test_acc, _, _ = eval_ann(model, test_loader, criterion, device)

        scheduler.step(test_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    print(f"    Best acc: {best_acc:.4f} (stopped at epoch {epoch})")
    return best_acc


def run_learning_curves(fold=1, encoding="direct", device=None):
    """Run few-shot experiment for one fold."""
    if device is None:
        device = get_device()

    results = {"fold": fold, "encoding": encoding, "fractions": DATA_FRACTIONS}
    snn_accs = {}
    ann_accs = {}

    for fraction in DATA_FRACTIONS:
        print(f"\n{'='*40}")
        print(f"Data fraction: {fraction:.0%}")
        print(f"{'='*40}")

        snn_acc = train_and_eval("snn", fold, encoding, fraction, device)
        ann_acc = train_and_eval("ann", fold, "none", fraction, device)

        snn_accs[str(fraction)] = snn_acc
        ann_accs[str(fraction)] = ann_acc

        gap = ann_acc - snn_acc
        print(f"  Gap (ANN - SNN): {gap:.4f} ({gap*100:.1f} pp)")

    results["snn_accuracies"] = snn_accs
    results["ann_accuracies"] = ann_accs
    results["gaps"] = {
        str(f): ann_accs[str(f)] - snn_accs[str(f)] for f in DATA_FRACTIONS
    }

    # Save
    save_dir = RESULTS_DIR / "few_shot"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"learning_curves_fold{fold}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_learning_curves(results, save_dir / f"learning_curves_fold{fold}.png")

    # Summary
    print(f"\n{'='*60}")
    print(f"Few-Shot Learning Curves (Fold {fold})")
    print(f"{'='*60}")
    print(f"{'Fraction':>10} {'SNN':>10} {'ANN':>10} {'Gap':>10}")
    print(f"{'-'*40}")
    for f in DATA_FRACTIONS:
        s = snn_accs[str(f)]
        a = ann_accs[str(f)]
        g = a - s
        print(f"{f:>10.0%} {s:>10.4f} {a:>10.4f} {g:>10.4f}")

    return results


def plot_learning_curves(results, save_path):
    """Plot SNN vs ANN accuracy vs data fraction."""
    fractions = results["fractions"]
    snn_accs = [results["snn_accuracies"][str(f)] * 100 for f in fractions]
    ann_accs = [results["ann_accuracies"][str(f)] * 100 for f in fractions]
    gaps = [results["gaps"][str(f)] * 100 for f in fractions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: accuracy curves
    frac_pct = [f * 100 for f in fractions]
    ax1.plot(frac_pct, snn_accs, 'o-', linewidth=2, color='#1976d2',
             label='SNN (direct)', markersize=8)
    ax1.plot(frac_pct, ann_accs, 's-', linewidth=2, color='#d32f2f',
             label='ANN', markersize=8)
    ax1.set_xlabel("Training Data (%)", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax1.set_title("Learning Curves: SNN vs ANN")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_xticks(frac_pct)
    ax1.set_xticklabels([f"{f:.0f}%" for f in frac_pct])

    # Right: gap
    ax2.bar(range(len(fractions)), gaps, color='#7b1fa2', alpha=0.7)
    ax2.set_xticks(range(len(fractions)))
    ax2.set_xticklabels([f"{f:.0%}" for f in fractions])
    ax2.set_xlabel("Training Data Fraction", fontsize=12)
    ax2.set_ylabel("ANN - SNN Gap (pp)", fontsize=12)
    ax2.set_title("Accuracy Gap vs Data Size")
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-shot learning curves")
    parser.add_argument("--fold", type=int, default=1,
                        help="Fold to evaluate (default: 1)")
    parser.add_argument("--encoding", default="direct")
    args = parser.parse_args()

    download_esc50()
    run_learning_curves(fold=args.fold, encoding=args.encoding)
