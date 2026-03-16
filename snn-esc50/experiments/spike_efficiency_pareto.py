"""
Spike Efficiency Pareto: Accuracy vs spike count frontier.

Trains SNN with L1 spike regularization at varying strengths to map
the full Pareto frontier of accuracy vs total spike count. This is
critical for neuromorphic hardware energy budgets.

Usage:
    python -m experiments.spike_efficiency_pareto [--fold 1] [--encoding direct]

    Best run on CSF3 for GPU acceleration.
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    RESULTS_DIR, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, PATIENCE, NUM_STEPS, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import get_encoder
from src.models.snn_model import SpikingCNN

LAMBDA_VALUES = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]


def train_snn_epoch_with_spike_reg(model, loader, optimizer, encoder, device, lam):
    """Train SNN with L1 spike regularization."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_spikes = 0

    criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encoder(data).to(device)

        optimizer.zero_grad()
        spk_out, mem_out = model(spk_input)

        # Classification loss (standard CE on membrane potentials)
        ce_loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            ce_loss += criterion(mem_out[step], targets)

        # Spike regularization: L1 on total spike count
        spike_count = spk_out.sum()
        spike_reg = lam * spike_count

        loss = ce_loss + spike_reg
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_spikes += spike_count.item()
        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    avg_spikes = total_spikes / total  # spikes per sample
    return avg_loss, accuracy, avg_spikes


@torch.no_grad()
def eval_snn_with_spikes(model, loader, encoder, device):
    """Evaluate SNN and count total spikes."""
    model.eval()
    correct = 0
    total = 0
    total_spikes = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encoder(data).to(device)
        spk_out, mem_out = model(spk_input)

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
        total_spikes += spk_out.sum().item()

    accuracy = correct / total
    avg_spikes = total_spikes / total
    spike_rate = total_spikes / (total * NUM_STEPS * spk_out.shape[-1])
    return accuracy, avg_spikes, spike_rate


def train_with_lambda(fold, encoding, lam, device):
    """Train one SNN with a given lambda and return results."""
    print(f"\n  Lambda={lam:.1e}")
    train_loader, test_loader = get_fold_dataloaders(fold, BATCH_SIZE)
    encoder = get_encoder(encoding)

    model = SpikingCNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_acc = 0.0
    best_spikes = 0.0
    best_spike_rate = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_spikes = train_snn_epoch_with_spike_reg(
            model, train_loader, optimizer, encoder, device, lam
        )

        test_acc, test_spikes, test_spike_rate = eval_snn_with_spikes(
            model, test_loader, encoder, device
        )
        scheduler.step(-test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_spikes = test_spikes
            best_spike_rate = test_spike_rate
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    print(f"    Acc: {best_acc:.4f} | Spikes/sample: {best_spikes:.0f} | "
          f"Spike rate: {best_spike_rate:.4f} | Stopped: ep{epoch}")
    return {
        "lambda": lam,
        "accuracy": best_acc,
        "spikes_per_sample": best_spikes,
        "spike_rate": best_spike_rate,
        "stopped_epoch": epoch,
    }


def run_pareto(fold=1, encoding="direct", device=None):
    """Run the full Pareto experiment."""
    if device is None:
        device = get_device()

    print(f"\nSpike Efficiency Pareto (Fold {fold}, {encoding})")
    print(f"Lambdas: {LAMBDA_VALUES}")

    pareto_points = []
    for lam in LAMBDA_VALUES:
        result = train_with_lambda(fold, encoding, lam, device)
        pareto_points.append(result)

    results = {
        "fold": fold,
        "encoding": encoding,
        "lambda_values": LAMBDA_VALUES,
        "pareto_points": pareto_points,
    }

    # Save
    save_dir = RESULTS_DIR / "snn" / "spike_efficiency"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"pareto_fold{fold}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_pareto(pareto_points, save_dir / f"pareto_fold{fold}.png", fold)

    # Summary
    print(f"\n{'='*60}")
    print(f"Pareto Frontier (Fold {fold})")
    print(f"{'='*60}")
    print(f"{'Lambda':>10} {'Accuracy':>10} {'Spikes/sample':>15} {'Spike Rate':>12}")
    for p in pareto_points:
        print(f"{p['lambda']:>10.1e} {p['accuracy']:>10.4f} "
              f"{p['spikes_per_sample']:>15.0f} {p['spike_rate']:>12.4f}")

    return results


def plot_pareto(points, save_path, fold):
    """Plot accuracy vs spikes Pareto frontier."""
    accs = [p["accuracy"] * 100 for p in points]
    spikes = [p["spikes_per_sample"] for p in points]
    lambdas = [p["lambda"] for p in points]

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(spikes, accs, c=range(len(points)), cmap='coolwarm',
                         s=100, zorder=5, edgecolors='black')
    ax.plot(spikes, accs, '--', alpha=0.5, color='gray')

    # Label each point with its lambda
    for i, (s, a, l) in enumerate(zip(spikes, accs, lambdas)):
        ax.annotate(f"λ={l:.0e}", (s, a), textcoords="offset points",
                    xytext=(5, 8), fontsize=8, alpha=0.8)

    ax.set_xlabel("Output Spikes per Sample", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(f"Spike Efficiency Pareto Frontier (Fold {fold})")
    ax.grid(True, alpha=0.3)

    # Mark the "sweet spot" if there is one
    if len(accs) >= 2:
        baseline_acc = accs[0]
        for i, (a, s) in enumerate(zip(accs, spikes)):
            if a >= baseline_acc * 0.9 and s < spikes[0] * 0.5:
                ax.annotate("Sweet spot!", (s, a), textcoords="offset points",
                            xytext=(10, -15), fontsize=10, color='green',
                            fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='green'))
                break

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spike efficiency Pareto")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--encoding", default="direct")
    args = parser.parse_args()

    download_esc50()
    run_pareto(fold=args.fold, encoding=args.encoding)
