"""
Population coding experiment: output population code with 50×10=500 neurons.

Each of the 50 classes is represented by 10 output neurons. During inference,
the class with the highest total spike count across its 10 neurons wins.

Uses SF.mse_count_loss(population_code=True) and SF.accuracy_rate(population_code=True).
Input encoding: rate coding (same as baseline rate experiment).

Usage:
    python experiments/population_coding.py [--folds 4] [--epochs 50]
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
from snntorch import surrogate

from src.config import (
    NUM_CLASSES, BETA, NUM_STEPS, RESULTS_DIR,
    LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_rate


# Population coding parameters
POP_N = 10          # neurons per class
POP_CLASSES = NUM_CLASSES * POP_N   # 500 output neurons


class SpikingCNNPop(nn.Module):
    """Same architecture as SpikingCNN but with 500-neuron population output."""

    def __init__(self, beta: float = BETA, num_steps: int = NUM_STEPS,
                 spike_grad=None):
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

        # Population output: 500 neurons instead of 50
        self.fc2 = nn.Linear(256, POP_CLASSES)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_out_rec = []

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

        return torch.stack(spk_out_rec)   # (T, B, 500)


def train_pop_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_rate(data).to(device)

        optimizer.zero_grad()
        spk_out = model(spk_input)  # (T, B, 500)

        loss = loss_fn(spk_out, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        acc = SF.accuracy_rate(spk_out, targets,
                               population_code=True, num_classes=NUM_CLASSES)
        correct += acc * targets.size(0)
        total += targets.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_pop(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_rate(data).to(device)
        spk_out = model(spk_input)

        loss = loss_fn(spk_out, targets)
        total_loss += loss.item()
        acc = SF.accuracy_rate(spk_out, targets,
                               population_code=True, num_classes=NUM_CLASSES)
        correct += acc * targets.size(0)
        total += targets.size(0)

    return total_loss / len(loader), correct / total


def train_fold(fold, num_epochs, device):
    print(f"\n{'='*60}")
    print(f"  Population Coding | Fold {fold}/5")
    print(f"  Output neurons: {POP_CLASSES} ({NUM_CLASSES} classes × {POP_N} each)")
    print(f"{'='*60}")

    train_loader, test_loader = get_fold_dataloaders(fold, BATCH_SIZE)

    model = SpikingCNNPop().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0,
                                population_code=True, num_classes=NUM_CLASSES)

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    save_dir = RESULTS_DIR / "snn" / "population"
    save_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_pop_epoch(model, train_loader, optimizer,
                                          loss_fn, device)
        te_loss, te_acc = eval_pop(model, test_loader, loss_fn, device)
        scheduler.step(te_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / f"best_fold{fold}.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
                  f"Test Loss: {te_loss:.4f} Acc: {te_acc:.4f} | "
                  f"Best: {best_acc:.4f} (ep {best_epoch})")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start
    print(f"  Fold {fold} done in {elapsed:.1f}s | Best: {best_acc:.4f}")

    result = {
        "fold": fold,
        "model": "snn",
        "encoding": "population",
        "pop_n": POP_N,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "time_seconds": elapsed,
        "history": history,
    }
    with open(save_dir / f"result_fold{fold}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Folds to run (default: all 5)")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    download_esc50()
    device = get_device()
    print(f"Device: {device}")

    results = []
    for fold in args.folds:
        r = train_fold(fold, args.epochs, device)
        results.append(r)

    if len(results) > 1:
        accs = [r["best_acc"] for r in results]
        mean = sum(accs) / len(accs)
        std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
        print(f"\n{'='*60}")
        print(f"  Population Coding | {len(results)}-Fold Results")
        print(f"  Per-fold: {[f'{a:.4f}' for a in accs]}")
        print(f"  Mean: {mean:.4f} ± {std:.4f}")
        print(f"{'='*60}")

        summary = {
            "model": "snn", "encoding": "population",
            "folds_run": args.folds, "fold_accuracies": accs,
            "mean_accuracy": mean, "std_accuracy": std,
        }
        save_dir = RESULTS_DIR / "snn" / "population"
        with open(save_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
