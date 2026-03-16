"""
Encoding Transfer Matrix: Train with X, test with Y.

Loads models trained with each of the 7 encodings and evaluates them
using every other encoding at inference time. Produces a 7×7 accuracy
matrix revealing encoding-architecture coupling.

Key question: Does the SNN learn encoding-specific circuits or
general audio features?

Usage:
    python -m experiments.encoding_transfer_matrix [--fold 1]

    Default: fold 1 only (for speed).
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
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, BATCH_SIZE, get_device
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import get_encoder, ENCODERS
from src.models.snn_model import SpikingCNN


# All 7 encodings (population excluded — uses different output shape)
ENCODINGS = ["direct", "rate", "phase", "latency", "delta", "burst"]
# Note: population encoding uses a different model output (500 neurons),
# so it's excluded from the transfer matrix. 6×6 matrix instead.


@torch.no_grad()
def eval_with_encoding(model, loader, encoder, device):
    """Evaluate model using a given encoder."""
    model.eval()
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encoder(data).to(device)
        _, mem_out = model(spk_input)
        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return correct / total


def run_transfer_matrix(fold=1, device=None):
    """Build the encoding transfer matrix for a given fold."""
    if device is None:
        device = get_device()

    _, test_loader = get_fold_dataloaders(fold, BATCH_SIZE)

    n = len(ENCODINGS)
    matrix = np.zeros((n, n))
    detailed_results = {}

    for i, train_enc in enumerate(ENCODINGS):
        # Load model trained with this encoding
        model_path = RESULTS_DIR / "snn" / train_enc / f"best_fold{fold}.pt"
        if not model_path.exists():
            print(f"  Model not found for {train_enc} fold {fold}, skipping row")
            matrix[i, :] = np.nan
            continue

        model = SpikingCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"\nTrained with: {train_enc}")

        for j, test_enc in enumerate(ENCODINGS):
            encoder = get_encoder(test_enc)
            acc = eval_with_encoding(model, test_loader, encoder, device)
            matrix[i, j] = acc
            detailed_results[f"{train_enc}_to_{test_enc}"] = acc
            marker = " ← DIAGONAL" if i == j else ""
            print(f"  Test with {test_enc:>8s}: {acc:.4f}{marker}")

    # Compute statistics
    diagonal = np.diag(matrix)
    off_diagonal_mean = (matrix.sum() - diagonal.sum()) / (n * n - n) if n > 1 else 0
    transfer_ratio = off_diagonal_mean / np.nanmean(diagonal) if np.nanmean(diagonal) > 0 else 0

    results = {
        "fold": fold,
        "encodings": ENCODINGS,
        "matrix": matrix.tolist(),
        "diagonal_accuracies": {enc: float(matrix[i, i]) for i, enc in enumerate(ENCODINGS)},
        "diagonal_mean": float(np.nanmean(diagonal)),
        "off_diagonal_mean": float(off_diagonal_mean),
        "transfer_ratio": float(transfer_ratio),
        "detailed": detailed_results,
    }

    # Save
    save_dir = RESULTS_DIR / "snn" / "encoding_transfer"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"transfer_matrix_fold{fold}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot heatmap
    plot_transfer_matrix(matrix, ENCODINGS, save_dir / f"transfer_matrix_fold{fold}.png", fold)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Encoding Transfer Matrix Summary (Fold {fold})")
    print(f"{'='*60}")
    print(f"  Diagonal mean (matched train/test): {np.nanmean(diagonal):.4f}")
    print(f"  Off-diagonal mean (mismatched):     {off_diagonal_mean:.4f}")
    print(f"  Transfer ratio (off-diag/diag):     {transfer_ratio:.4f}")
    print(f"  If ratio ≈ 1.0: encoding-invariant features (general)")
    print(f"  If ratio << 1.0: encoding-specific circuits (coupled)")

    return results


def plot_transfer_matrix(matrix, encodings, save_path, fold):
    """Plot the encoding transfer matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))

    mask = np.isnan(matrix)
    sns.heatmap(
        matrix * 100,
        annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=encodings, yticklabels=encodings,
        ax=ax, mask=mask,
        vmin=0, vmax=60,
        cbar_kws={"label": "Accuracy (%)"},
    )

    ax.set_xlabel("Test Encoding", fontsize=12)
    ax.set_ylabel("Train Encoding", fontsize=12)
    ax.set_title(f"Encoding Transfer Matrix (Fold {fold})\n"
                 f"Diagonal = matched, Off-diagonal = mismatched", fontsize=11)

    # Highlight diagonal
    for i in range(len(encodings)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                    edgecolor='blue', linewidth=2))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encoding transfer matrix")
    parser.add_argument("--fold", type=int, default=1,
                        help="Fold to evaluate (default: 1)")
    args = parser.parse_args()

    download_esc50()
    run_transfer_matrix(fold=args.fold)
