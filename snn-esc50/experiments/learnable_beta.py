"""
Experiment 9: Learnable beta standalone ablation.

ONLY change from baseline: learn_beta=True on all LIF neurons.
Everything else identical to baseline SpikingCNN with fast_sigmoid.
This isolates the effect of heterogeneous membrane dynamics.

Usage:
    python -m experiments.learnable_beta              # all 5 folds
    python -m experiments.learnable_beta --fold 1     # single fold
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
    NUM_CLASSES, NUM_STEPS, BETA,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct


class LearnableBetaCNN(nn.Module):
    """Baseline SpikingCNN with ONLY learn_beta=True. No other changes."""

    def __init__(self, num_classes=NUM_CLASSES, beta=BETA, num_steps=NUM_STEPS):
        super().__init__()
        self.num_steps = num_steps

        # Keep fast_sigmoid (baseline default) to isolate beta effect
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        # NO dropout — keeping identical to baseline except learn_beta
        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

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


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data).to(device)

        optimizer.zero_grad()
        spk_out, mem_out = model(spk_input)

        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss += criterion(mem_out[step], targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data).to(device)
        spk_out, mem_out = model(spk_input)

        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss += criterion(mem_out[step], targets)
        total_loss += loss.item()

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


def run_fold(fold, device, num_epochs=NUM_EPOCHS):
    print(f"\n{'='*60}")
    print(f"  Learnable Beta Ablation | Fold {fold}/5 | Device: {device}")
    print(f"  ONLY change: learn_beta=True (all else = baseline)")
    print(f"{'='*60}")

    train_loader, test_loader = get_fold_dataloaders(fold, BATCH_SIZE)
    model = LearnableBetaCNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    start = time.time()
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = eval_model(model, test_loader, device)
        scheduler.step(test_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
            save_dir = RESULTS_DIR / "experiments" / "learnable_beta"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / f"best_fold{fold}.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | Train: {train_acc:.4f} | Test: {test_acc:.4f} | Best: {best_acc:.4f} (ep{best_epoch})")

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

