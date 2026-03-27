"""
layerwise_spike_reg.py -- Train SNN with aggressive per-layer spike regularization.

Targets <6% spike rate (Dampfhoffer 2023 threshold for energy parity with ANN).
Applies separate L1 penalties to conv1, conv2, and fc1 hidden spikes.

Usage:
    python -m experiments.layerwise_spike_reg --device cuda
    python -m experiments.layerwise_spike_reg --lambda-conv 1e-3 --lambda-fc 5e-4 --device cuda
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
    NUM_CLASSES, NUM_STEPS, BETA, RESULTS_DIR, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, PATIENCE, NUM_EPOCHS, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct

from experiments.combo_experiment import RhythmLIF


class SpikeRegSNN(nn.Module):
    """SNN with per-layer spike tracking for regularization."""

    def __init__(self, num_steps=NUM_STEPS):
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

        # Rhythm LIF neurons with learnable beta
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
        # Per-layer spike accumulators for regularization
        layer_spikes = {"conv1": 0.0, "conv2": 0.0, "fc1": 0.0, "out": 0.0}

        for step in range(self.num_steps):
            x_t = x[step]

            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, m1 = self.n1(cur1, m1, step)
            layer_spikes["conv1"] = layer_spikes["conv1"] + spk1.mean()

            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, m2 = self.n2(cur2, m2, step)
            layer_spikes["conv2"] = layer_spikes["conv2"] + spk2.mean()

            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            cur3 = self.fc1(flat)
            spk3, m3 = self.n3(cur3, m3, step)
            layer_spikes["fc1"] = layer_spikes["fc1"] + spk3.mean()

            spk3 = self.dropout(spk3)

            cur4 = self.fc2(spk3)
            spk4, m4 = self.n4(cur4, m4, step)
            layer_spikes["out"] = layer_spikes["out"] + spk4.mean()

            spk_rec.append(spk4)
            mem_rec.append(m4)

        # Normalize by timesteps to get average spike rate
        for k in layer_spikes:
            layer_spikes[k] = layer_spikes[k] / self.num_steps

        return torch.stack(spk_rec), torch.stack(mem_rec), layer_spikes


def train_epoch(model, loader, optimizer, device, lambda_conv, lambda_fc):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_rates = {"conv1": 0, "conv2": 0, "fc1": 0, "out": 0}
    use_amp = device.type == 'cuda'

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data).to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            spk_out, mem_out, layer_spikes = model(spk_input)

            T, B, C = mem_out.shape
            ce_loss = F.cross_entropy(
                mem_out.reshape(T * B, C),
                targets.unsqueeze(0).expand(T, -1).reshape(-1),
            )

            # Layerwise spike regularization
            spike_loss = (lambda_conv * (layer_spikes["conv1"] +
                                          layer_spikes["conv2"]) +
                          lambda_fc * layer_spikes["fc1"])

            loss = ce_loss + spike_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        for k in total_rates:
            total_rates[k] += layer_spikes[k].item()

    n_batches = len(loader)
    avg_rates = {k: v / n_batches for k, v in total_rates.items()}
    return total_loss / n_batches, correct / total, avg_rates


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    total_rates = {"conv1": 0, "conv2": 0, "fc1": 0, "out": 0}
    use_amp = device.type == 'cuda'

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data).to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            spk_out, mem_out, layer_spikes = model(spk_input)

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        for k in total_rates:
            total_rates[k] += layer_spikes[k].item()

    n_batches = len(loader)
    avg_rates = {k: v / n_batches for k, v in total_rates.items()}
    overall_rate = sum(avg_rates.values()) / len(avg_rates)
    return correct / total, avg_rates, overall_rate


def run_fold(fold, device, lambda_conv, lambda_fc, num_steps=NUM_STEPS):
    print(f"\n  Fold {fold}: lambda_conv={lambda_conv}, lambda_fc={lambda_fc}")
    train_loader, test_loader = get_fold_dataloaders(fold, BATCH_SIZE)

    model = SpikeRegSNN(num_steps=num_steps).to(device)
    param_count = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    save_dir = (RESULTS_DIR / "energy" /
                f"spike_reg_lc{lambda_conv}_lf{lambda_fc}")
    save_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc, train_rates = train_epoch(
            model, train_loader, optimizer, device, lambda_conv, lambda_fc)
        test_acc, test_rates, overall_rate = eval_model(
            model, test_loader, device)
        scheduler.step(train_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(),
                       save_dir / f"best_fold{fold}.pt")
            best_rates = test_rates.copy()
            best_overall = overall_rate
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            rates_str = " ".join(f"{k}={v:.4f}" for k, v in test_rates.items())
            print(f"  Ep {epoch:3d} | Train: {train_acc:.4f} | "
                  f"Test: {test_acc:.4f} | {rates_str} | "
                  f"overall={overall_rate:.4f}")
