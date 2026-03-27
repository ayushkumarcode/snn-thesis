"""
reduced_t_training.py -- Train rhythm SNN from scratch at low timesteps.

Instead of truncating a T=25 model, actually TRAIN at T=3,5,7,10,15.
This should give much better accuracy at low T since the model learns
to use fewer timesteps optimally.

Usage:
    python -m experiments.reduced_t_training --num-steps 5 --device cuda
    python -m experiments.reduced_t_training --device cuda  # sweep all T values
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    NUM_CLASSES, BETA, RESULTS_DIR, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, PATIENCE, NUM_EPOCHS, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct

from experiments.combo_experiment import RhythmLIF


class ReducedTSNN(nn.Module):
    """Rhythm SNN with configurable timesteps."""

    def __init__(self, num_steps):
        super().__init__()
        self.num_steps = num_steps
        sg = surrogate.spike_rate_escape(beta=1, slope=25)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.fc2 = nn.Linear(256, NUM_CLASSES)
        self.dropout = nn.Dropout(0.3)

        self.n1 = RhythmLIF(32, BETA, sg, learn_beta=True)
        self.n2 = RhythmLIF(64, BETA, sg, learn_beta=True)
        self.n3 = RhythmLIF(256, BETA, sg, learn_beta=True)
        self.n4 = RhythmLIF(NUM_CLASSES, BETA, sg, learn_beta=True)

    def forward(self, x):
        device = x.device
        m1 = self.n1.init_mem(device)
        m2 = self.n2.init_mem(device)
        m3 = self.n3.init_mem(device)
        m4 = self.n4.init_mem(device)

        spk_rec, mem_rec = [], []
        total_spikes = 0.0

        for step in range(self.num_steps):
            x_t = x[step]

            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, m1 = self.n1(cur1, m1, step)

            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, m2 = self.n2(cur2, m2, step)

            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            cur3 = self.fc1(flat)
            spk3, m3 = self.n3(cur3, m3, step)
            spk3 = self.dropout(spk3)

            cur4 = self.fc2(spk3)
            spk4, m4 = self.n4(cur4, m4, step)

            spk_rec.append(spk4)
            mem_rec.append(m4)
            total_spikes = total_spikes + spk4.sum()

        return (torch.stack(spk_rec), torch.stack(mem_rec),
                total_spikes.item() if isinstance(total_spikes, torch.Tensor)
                else total_spikes)


def train_epoch(model, loader, optimizer, device, num_steps):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == 'cuda'

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data, num_steps=num_steps).to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            spk_out, mem_out, _ = model(spk_input)
            T, B, C = mem_out.shape
            loss = F.cross_entropy(
                mem_out.reshape(T * B, C),
                targets.unsqueeze(0).expand(T, -1).reshape(-1),
            )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_model(model, loader, device, num_steps):
    model.eval()
    correct = 0
    total = 0
    use_amp = device.type == 'cuda'

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data, num_steps=num_steps).to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            _, mem_out, _ = model(spk_input)

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return correct / total


def run_fold(fold, device, num_steps):
    print(f"\n  Fold {fold}: T={num_steps}")
    train_loader, test_loader = get_fold_dataloaders(fold, BATCH_SIZE)

    model = ReducedTSNN(num_steps=num_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    save_dir = RESULTS_DIR / "energy" / f"reduced_t_{num_steps}"
    save_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, num_steps)
        test_acc = eval_model(model, test_loader, device, num_steps)
        scheduler.step(train_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(),
                       save_dir / f"best_fold{fold}.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | Train: {train_acc:.4f} | "
                  f"Test: {test_acc:.4f} | Best: {best_acc:.4f}")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start
    # Energy is proportional to T (linear in timesteps)
    energy_ratio = num_steps / 25.0
    print(f"  Fold {fold}: {best_acc:.4f} (ep{best_epoch}), "
          f"energy={energy_ratio:.2f}x of T=25")

    return {
        "fold": fold, "num_steps": num_steps,
        "best_acc": best_acc, "best_epoch": best_epoch,
        "energy_ratio": energy_ratio,
        "energy_reduction_x": 25.0 / num_steps,
        "time_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-steps", type=int, default=None,
                        help="If set, train only at this T. Otherwise sweep.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    download_esc50()

    if args.num_steps:
        t_values = [args.num_steps]
    else:
        t_values = [3, 5, 7, 10, 15]

    folds = [args.fold] if args.fold else list(range(1, 6))

    all_results = []
    for T in t_values:
        print(f"\n{'='*60}")
        print(f"  Reduced-T Training: T={T} (rhythm SNN)")
        print(f"{'='*60}")

        fold_results = []
        for fold in folds:
            result = run_fold(fold, device, T)
            fold_results.append(result)

        if len(fold_results) == 5:
            accs = [r["best_acc"] for r in fold_results]
            print(f"\n  T={T} 5-Fold: {np.mean(accs):.4f}±{np.std(accs):.4f}, "
                  f"{25/T:.1f}x energy reduction")

        all_results.extend(fold_results)

    save_dir = RESULTS_DIR / "energy" / "reduced_t_sweep"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {save_dir / 'all_results.json'}")


if __name__ == "__main__":
    main()
