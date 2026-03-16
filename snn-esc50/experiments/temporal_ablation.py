"""
Temporal Ablation: How many timesteps does the SNN actually need?

Evaluates the trained SNN at inference using only the first T timesteps
(T = 1, 2, 3, 5, 7, 10, 15, 20, 25) WITHOUT retraining. This maps
directly to latency-energy tradeoffs for deployment.

Usage:
    python -m experiments.temporal_ablation [--fold FOLD] [--encoding direct]

    Defaults to all 5 folds with direct encoding.
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

from src.config import RESULTS_DIR, NUM_FOLDS, NUM_STEPS, BATCH_SIZE, get_device
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import get_encoder
from src.models.snn_model import SpikingCNN


TIMESTEP_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 25]


@torch.no_grad()
def eval_snn_truncated(model, loader, encoder, device, max_steps):
    """Evaluate SNN using only the first `max_steps` timesteps."""
    model.eval()
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encoder(data).to(device)

        # Only use first max_steps timesteps
        spk_input_truncated = spk_input[:max_steps]

        # Manual forward pass with truncated timesteps
        mem1 = model.lif1.init_leaky()
        mem2 = model.lif2.init_leaky()
        mem3 = model.lif3.init_leaky()
        mem4 = model.lif4.init_leaky()

        mem_out_acc = torch.zeros(data.size(0), model.fc2.out_features, device=device)

        for step in range(max_steps):
            x_t = spk_input_truncated[step]
            cur1 = model.pool1(model.bn1(model.conv1(x_t)))
            spk1, mem1 = model.lif1(cur1, mem1)
            cur2 = model.pool2(model.bn2(model.conv2(spk1)))
            spk2, mem2 = model.lif2(cur2, mem2)
            pooled = model.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)
            cur3 = model.fc1(flat)
            spk3, mem3 = model.lif3(cur3, mem3)
            cur4 = model.fc2(spk3)
            spk4, mem4 = model.lif4(cur4, mem4)
            mem_out_acc += mem4

        predicted = mem_out_acc.argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return correct / total


def run_temporal_ablation(encoding="direct", folds=None, device=None):
    """Run temporal ablation across all timestep values and folds."""
    if device is None:
        device = get_device()
    if folds is None:
        folds = list(range(1, NUM_FOLDS + 1))

    encoder = get_encoder(encoding)
    results = {}

    for fold in folds:
        print(f"\n--- Fold {fold} ---")
        _, test_loader = get_fold_dataloaders(fold, BATCH_SIZE)

        # Load trained model
        model_path = RESULTS_DIR / "snn" / encoding / f"best_fold{fold}.pt"
        if not model_path.exists():
            print(f"  Model not found: {model_path}, skipping")
            continue

        model = SpikingCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        fold_results = {}
        for T in TIMESTEP_VALUES:
            acc = eval_snn_truncated(model, test_loader, encoder, device, T)
            fold_results[T] = acc
            pct_of_full = (acc / fold_results.get(25, acc)) * 100 if 25 in fold_results else 0
            print(f"  T={T:2d}: {acc:.4f} ({pct_of_full:.1f}% of full)")

        results[f"fold_{fold}"] = fold_results

    # Compute mean across folds
    all_Ts = TIMESTEP_VALUES
    mean_accs = {}
    std_accs = {}
    for T in all_Ts:
        accs_at_T = [results[f"fold_{f}"][T] for f in folds if f"fold_{f}" in results]
        if accs_at_T:
            mean_accs[T] = float(np.mean(accs_at_T))
            std_accs[T] = float(np.std(accs_at_T))

    results["mean"] = mean_accs
    results["std"] = std_accs
    results["encoding"] = encoding
    results["timestep_values"] = all_Ts

    # Save
    save_dir = RESULTS_DIR / "snn" / "temporal_ablation"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"ablation_{encoding}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {save_dir / f'ablation_{encoding}.json'}")

    # Plot
    plot_temporal_ablation(results, save_dir / f"ablation_{encoding}.png")

    # Print summary
    full_acc = mean_accs.get(25, 0)
    print(f"\n{'='*50}")
    print(f"Temporal Ablation Summary ({encoding})")
    print(f"{'='*50}")
    for T in all_Ts:
        pct = (mean_accs[T] / full_acc * 100) if full_acc > 0 else 0
        energy_saving = (1 - T / 25) * 100
        print(f"  T={T:2d}: {mean_accs[T]:.4f} ± {std_accs[T]:.4f} "
              f"({pct:.1f}% of full, {energy_saving:.0f}% energy saving)")

    return results


def plot_temporal_ablation(results, save_path):
    """Plot accuracy vs timesteps curve."""
    Ts = results["timestep_values"]
    means = [results["mean"][T] for T in Ts]
    stds = [results["std"][T] for T in Ts]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Accuracy curve
    ax1.errorbar(Ts, [m * 100 for m in means], yerr=[s * 100 for s in stds],
                 marker='o', capsize=4, linewidth=2, color='#1976d2', label='Accuracy')
    ax1.set_xlabel("Number of Timesteps (T)", fontsize=12)
    ax1.set_ylabel("Accuracy (%)", fontsize=12, color='#1976d2')
    ax1.set_xticks(Ts)
    ax1.grid(True, alpha=0.3)

    # Energy saving on secondary axis
    ax2 = ax1.twinx()
    energy_savings = [(1 - T / 25) * 100 for T in Ts]
    ax2.bar(Ts, energy_savings, alpha=0.15, color='#388e3c', width=1.2, label='Energy saving')
    ax2.set_ylabel("Energy Saving vs T=25 (%)", fontsize=12, color='#388e3c')
    ax2.set_ylim(0, 100)

    # Mark 90% of full accuracy
    full_acc = results["mean"][25] * 100
    ax1.axhline(y=full_acc * 0.9, color='red', linestyle='--', alpha=0.5,
                label=f'90% of full ({full_acc * 0.9:.1f}%)')

    ax1.legend(loc='lower right')
    ax1.set_title("Temporal Ablation: Accuracy vs Timesteps (SNN)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal ablation experiment")
    parser.add_argument("--encoding", default="direct", choices=["direct", "rate", "phase", "latency", "delta", "burst"])
    parser.add_argument("--fold", type=int, default=None, help="Specific fold (1-5)")
    args = parser.parse_args()

    download_esc50()

    folds = [args.fold] if args.fold else None
    run_temporal_ablation(encoding=args.encoding, folds=folds)
