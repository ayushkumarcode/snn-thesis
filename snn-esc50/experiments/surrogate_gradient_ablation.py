"""
surrogate_gradient_ablation.py -- Ablation study across 8 surrogate gradient functions.

Tests all 8 snnTorch surrogate gradients on fold 1, direct encoding.
Reference: Zenke & Vogels (2021) -- shape matters less than slope/scale.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/surrogate_gradient_ablation.py [--fold N] [--seed N] [--epochs N]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import get_encoder


# ============================================================
# SNN model (parameterised surrogate gradient)
# ============================================================

class SpikingCNN(nn.Module):
    """SpikingCNN with configurable surrogate gradient."""

    def __init__(self, num_classes=50, beta=0.95, num_steps=25, spike_grad=None):
        super().__init__()
        self.num_steps = num_steps
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_out_rec = []
        mem_out_rec = []

        for step in range(self.num_steps):
            x_t = x[step]
            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)
            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)
            cur3 = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)

        return torch.stack(spk_out_rec), torch.stack(mem_out_rec)


def train_epoch(model, loader, optimizer, encoder, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    ce = nn.CrossEntropyLoss()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        spike_inputs = encoder(inputs)
        optimizer.zero_grad()
        spk_out, mem_out = model(spike_inputs)
        rate = spk_out.mean(dim=0)
        loss = ce(rate, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = rate.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def eval_model(model, loader, encoder, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            spike_inputs = encoder(inputs)
            spk_out, mem_out = model(spike_inputs)
            rate = spk_out.mean(dim=0)
            loss = ce(rate, labels)
            total_loss += loss.item()
            preds = rate.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def train_with_surrogate(surrogate_name, spike_grad, fold, seed, epochs, patience, device, encoder):
    """Train one model with given surrogate and return best test accuracy."""
    torch.manual_seed(seed)

    train_loader, test_loader = get_fold_dataloaders(fold, batch_size=BATCH_SIZE)
    model = SpikingCNN(spike_grad=spike_grad).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, encoder, device)
        te_loss, te_acc = eval_model(model, test_loader, encoder, device)
        scheduler.step(te_loss)

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  [{surrogate_name}] Ep {epoch:3d}/{epochs} | "
                  f"tr={tr_acc:.3f} te={te_acc:.3f} best={best_acc:.3f} ({elapsed:.0f}s)")

        if no_improve >= patience:
            print(f"  [{surrogate_name}] Early stop at epoch {epoch}, best={best_acc:.4f}")
            break

    return best_acc, best_epoch, epoch


def main():
    parser = argparse.ArgumentParser(description="Surrogate gradient ablation")
    parser.add_argument("--fold", type=int, default=1, help="Fold to use (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs (default: 50)")
    parser.add_argument("--patience", type=int, default=10, help="Early stop patience")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    download_esc50()

    encoder = get_encoder("direct")

    # Define all 8 surrogates from snnTorch
    surrogates = {
        "fast_sigmoid": surrogate.fast_sigmoid(slope=25),
        "atan":         surrogate.atan(alpha=2.0),
        "sigmoid":      surrogate.sigmoid(slope=25),
        "ste":          surrogate.straight_through_estimator(),
        "triangular":   surrogate.triangular(),
        "spike_rate_escape": surrogate.spike_rate_escape(beta=1.0, slope=25),
        "lso":          surrogate.LSO(slope=0.1),
        "sfs":          surrogate.SFS(slope=25),
    }

    out_dir = RESULTS_DIR / "snn" / "surrogate_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print(f"\n{'='*60}")
    print(f"Surrogate Gradient Ablation — Fold {args.fold}, Seed {args.seed}")
    print(f"{'='*60}\n")

    for surrogate_name, spike_grad in surrogates.items():
        print(f"\nTraining with surrogate: {surrogate_name}")
        best_acc, best_epoch, total_epochs = train_with_surrogate(
            surrogate_name, spike_grad,
            args.fold, args.seed, args.epochs, args.patience, device, encoder
        )
        results[surrogate_name] = {
            "best_accuracy": best_acc,
            "best_epoch": best_epoch,
            "total_epochs": total_epochs,
            "fold": args.fold,
            "seed": args.seed,
        }
        print(f"  -> {surrogate_name}: {best_acc*100:.2f}% at epoch {best_epoch}")

    # Save results
    out_file = out_dir / f"ablation_fold{args.fold}_seed{args.seed}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"SURROGATE GRADIENT ABLATION SUMMARY (Fold {args.fold}, Seed {args.seed})")
    print(f"{'='*60}")
    print(f"{'Surrogate':<25} {'Accuracy':>10} {'Best Epoch':>12}")
    print(f"{'-'*50}")
    sorted_results = sorted(results.items(), key=lambda x: x[1]["best_accuracy"], reverse=True)
    for name, r in sorted_results:
        print(f"{name:<25} {r['best_accuracy']*100:>9.2f}% {r['best_epoch']:>12d}")


if __name__ == "__main__":
    main()
