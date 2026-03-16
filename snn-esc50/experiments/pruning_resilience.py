"""
Pruning Resilience: Weight sparsity × activation sparsity interaction.

Applies magnitude pruning at various sparsity levels to both SNN and ANN,
comparing accuracy retention. Analyzes the compound sparsity effect unique
to SNNs (weight sparsity + activation sparsity).

Usage:
    python -m experiments.pruning_resilience [--fold 1] [--encoding direct]
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.utils.prune as prune
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, BATCH_SIZE, NUM_FOLDS, get_device
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import get_encoder
from src.models.snn_model import SpikingCNN
from src.models.ann_model import ConvANN

SPARSITY_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]


def apply_global_pruning(model, sparsity):
    """Apply global unstructured L1 pruning to all weight parameters."""
    if sparsity <= 0:
        return
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            parameters_to_prune.append((module, "weight"))
    if parameters_to_prune:
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )


def count_zero_weights(model):
    """Count fraction of zero weights."""
    total = 0
    zeros = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            total += param.numel()
            zeros += (param == 0).sum().item()
    return zeros / total if total > 0 else 0


@torch.no_grad()
def eval_snn_pruned(model, loader, encoder, device):
    """Evaluate pruned SNN."""
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


@torch.no_grad()
def eval_ann_pruned(model, loader, device):
    """Evaluate pruned ANN."""
    model.eval()
    correct = 0
    total = 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        logits = model(data)
        predicted = logits.argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    return correct / total


def run_pruning_experiment(fold=1, encoding="direct", device=None):
    """Run pruning resilience experiment for one fold."""
    if device is None:
        device = get_device()

    _, test_loader = get_fold_dataloaders(fold, BATCH_SIZE)
    encoder = get_encoder(encoding)

    snn_results = {}
    ann_results = {}

    for sparsity in SPARSITY_LEVELS:
        print(f"\n--- Sparsity: {sparsity:.0%} ---")

        # SNN
        snn_path = RESULTS_DIR / "snn" / encoding / f"best_fold{fold}.pt"
        if snn_path.exists():
            model_snn = SpikingCNN().to(device)
            model_snn.load_state_dict(
                torch.load(snn_path, map_location=device, weights_only=True)
            )
            apply_global_pruning(model_snn, sparsity)
            actual_sparsity = count_zero_weights(model_snn)
            snn_acc = eval_snn_pruned(model_snn, test_loader, encoder, device)
            snn_results[sparsity] = {
                "accuracy": snn_acc,
                "actual_weight_sparsity": actual_sparsity,
            }
            print(f"  SNN: {snn_acc:.4f} (weight sparsity: {actual_sparsity:.3f})")
            del model_snn

        # ANN
        ann_path = RESULTS_DIR / "ann" / "none" / f"best_fold{fold}.pt"
        if ann_path.exists():
            model_ann = ConvANN().to(device)
            model_ann.load_state_dict(
                torch.load(ann_path, map_location=device, weights_only=True)
            )
            apply_global_pruning(model_ann, sparsity)
            actual_sparsity = count_zero_weights(model_ann)
            ann_acc = eval_ann_pruned(model_ann, test_loader, device)
            ann_results[sparsity] = {
                "accuracy": ann_acc,
                "actual_weight_sparsity": actual_sparsity,
            }
            print(f"  ANN: {ann_acc:.4f} (weight sparsity: {actual_sparsity:.3f})")
            del model_ann

    # Compute retention
    snn_baseline = snn_results.get(0.0, {}).get("accuracy", 0)
    ann_baseline = ann_results.get(0.0, {}).get("accuracy", 0)

    results = {
        "fold": fold,
        "encoding": encoding,
        "sparsity_levels": SPARSITY_LEVELS,
        "snn_results": {str(k): v for k, v in snn_results.items()},
        "ann_results": {str(k): v for k, v in ann_results.items()},
        "snn_baseline": snn_baseline,
        "ann_baseline": ann_baseline,
    }

    # Save
    save_dir = RESULTS_DIR / "snn" / "pruning"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"pruning_fold{fold}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_pruning(results, save_dir / f"pruning_fold{fold}.png")

    # Summary
    print(f"\n{'='*50}")
    print(f"Pruning Resilience (Fold {fold})")
    print(f"{'='*50}")
    for sp in SPARSITY_LEVELS:
        snn_acc = snn_results.get(sp, {}).get("accuracy", 0)
        ann_acc = ann_results.get(sp, {}).get("accuracy", 0)
        snn_ret = (snn_acc / snn_baseline * 100) if snn_baseline > 0 else 0
        ann_ret = (ann_acc / ann_baseline * 100) if ann_baseline > 0 else 0
        print(f"  {sp:5.0%}: SNN {snn_acc:.4f} ({snn_ret:.1f}% retained) | "
              f"ANN {ann_acc:.4f} ({ann_ret:.1f}% retained)")

    return results


def run_all_folds(encoding="direct"):
    """Run pruning experiment across all 5 folds."""
    device = get_device()
    all_results = {}

    for fold in range(1, NUM_FOLDS + 1):
        result = run_pruning_experiment(fold, encoding, device)
        all_results[f"fold_{fold}"] = result

    # Compute mean across folds
    save_dir = RESULTS_DIR / "snn" / "pruning"
    save_dir.mkdir(parents=True, exist_ok=True)

    snn_means = {}
    ann_means = {}
    for sp in SPARSITY_LEVELS:
        snn_accs = [all_results[f"fold_{f}"]["snn_results"].get(str(sp), {}).get("accuracy", 0)
                    for f in range(1, NUM_FOLDS + 1)]
        ann_accs = [all_results[f"fold_{f}"]["ann_results"].get(str(sp), {}).get("accuracy", 0)
                    for f in range(1, NUM_FOLDS + 1)]
        snn_means[sp] = {"mean": float(np.mean(snn_accs)), "std": float(np.std(snn_accs))}
        ann_means[sp] = {"mean": float(np.mean(ann_accs)), "std": float(np.std(ann_accs))}

    summary = {
        "encoding": encoding,
        "sparsity_levels": SPARSITY_LEVELS,
        "snn_means": {str(k): v for k, v in snn_means.items()},
        "ann_means": {str(k): v for k, v in ann_means.items()},
    }
    with open(save_dir / "pruning_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def plot_pruning(results, save_path):
    """Plot pruning resilience comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sparsities = SPARSITY_LEVELS
    snn_accs = [results["snn_results"].get(str(sp), {}).get("accuracy", 0) * 100
                for sp in sparsities]
    ann_accs = [results["ann_results"].get(str(sp), {}).get("accuracy", 0) * 100
                for sp in sparsities]
    sp_pct = [sp * 100 for sp in sparsities]

    ax.plot(sp_pct, snn_accs, 'o-', linewidth=2, color='#1976d2', label='SNN', markersize=8)
    ax.plot(sp_pct, ann_accs, 's-', linewidth=2, color='#d32f2f', label='ANN', markersize=8)

    ax.set_xlabel("Weight Pruning Sparsity (%)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Pruning Resilience: SNN vs ANN (Fold {results['fold']})")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sp_pct)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pruning resilience experiment")
    parser.add_argument("--fold", type=int, default=None,
                        help="Specific fold (1-5). If omitted, runs all 5.")
    parser.add_argument("--encoding", default="direct")
    args = parser.parse_args()

    download_esc50()

    if args.fold:
        run_pruning_experiment(args.fold, args.encoding)
    else:
        run_all_folds(args.encoding)
