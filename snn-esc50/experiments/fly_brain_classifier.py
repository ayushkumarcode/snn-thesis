"""
fly_brain_classifier.py -- Fruit fly olfactory circuit for audio classification.

Dasgupta et al. (Science 2017): Random projection + winner-take-all.
The fly classifies ~50 odors using:
  1. Input (50 projection neurons)
  2. Random sparse projection to 2000 Kenyon cells (40x expansion)
  3. Winner-take-all: keep top 5% active
  4. Trained readout (2000 -> 50)

For our audio: mel spectrogram features -> random projection -> WTA -> linear.
Extremely energy-efficient: random weights are FIXED, only readout is trained.

Usage:
    python -m experiments.fly_brain_classifier --device cuda
    python -m experiments.fly_brain_classifier --fold 1 --expansion 40 --device cuda
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    NUM_CLASSES, RESULTS_DIR, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, PATIENCE, NUM_EPOCHS, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders


class FlyBrainClassifier(nn.Module):
    """Fly olfactory circuit adapted for audio classification.

    Architecture:
        1. Global avg pool spectrogram -> feature vector (64 mel bins)
        2. Random sparse projection (64 -> expansion_dim, FIXED)
        3. Winner-take-all: keep top k% neurons active
        4. Trained linear readout (expansion_dim -> 50 classes)
    """

    def __init__(self, input_dim=64, expansion=40, wta_ratio=0.05,
                 connection_prob=0.1):
        super().__init__()
        self.expansion_dim = input_dim * expansion
        self.wta_k = max(1, int(self.expansion_dim * wta_ratio))

        # FIXED random sparse projection (not trained!)
        # Each Kenyon cell receives input from ~10% of projection neurons
        random_weights = torch.zeros(self.expansion_dim, input_dim)
        for i in range(self.expansion_dim):
            # Random sparse connections
            n_connections = max(1, int(input_dim * connection_prob))
            indices = torch.randperm(input_dim)[:n_connections]
            random_weights[i, indices] = torch.randn(n_connections)
        self.register_buffer('random_proj', random_weights)

        # Only the readout is trained
        self.readout = nn.Linear(self.expansion_dim, NUM_CLASSES)

        # Batch norm on projected features
        self.bn = nn.BatchNorm1d(self.expansion_dim)

    def forward(self, x):
        # x: (batch, 1, 64, 216) mel spectrogram
        # Global average pool over time dimension
        features = x.squeeze(1).mean(dim=-1)  # (batch, 64)

        # Random projection (FIXED, no gradients)
        projected = F.linear(features, self.random_proj)  # (batch, expansion_dim)
        projected = self.bn(projected)
        projected = F.relu(projected)

        # Winner-take-all: keep only top-k activations
        topk_vals, topk_idx = projected.topk(self.wta_k, dim=1)
        sparse = torch.zeros_like(projected)
        sparse.scatter_(1, topk_idx, topk_vals)

        # Trained readout
        logits = self.readout(sparse)
        return logits


class FlyBrainSNN(nn.Module):
    """Spiking version of fly brain for SpiNNaker compatibility.

    Uses LIF neurons in the Kenyon cell layer.
    """

    def __init__(self, input_dim=64, expansion=40, wta_ratio=0.05,
                 connection_prob=0.1, num_steps=25):
        super().__init__()
        import snntorch as snn
        from snntorch import surrogate

        self.expansion_dim = input_dim * expansion
        self.wta_k = max(1, int(self.expansion_dim * wta_ratio))
        self.num_steps = num_steps

        # FIXED random projection
        random_weights = torch.zeros(self.expansion_dim, input_dim)
        for i in range(self.expansion_dim):
            n_connections = max(1, int(input_dim * connection_prob))
            indices = torch.randperm(input_dim)[:n_connections]
            random_weights[i, indices] = torch.randn(n_connections)
        self.register_buffer('random_proj', random_weights)

        # LIF neuron for Kenyon cells
        sg = surrogate.fast_sigmoid(slope=25)
        self.lif = snn.Leaky(beta=0.95, spike_grad=sg)

        # Trained readout
        self.readout = nn.Linear(self.expansion_dim, NUM_CLASSES)
        self.lif_out = snn.Leaky(beta=0.95, spike_grad=sg)

    def forward(self, x):
        # x: (T, batch, 1, 64, 216) direct encoded spectrogram
        device = x.device
        mem1 = self.lif.init_leaky()
        mem2 = self.lif_out.init_leaky()

        spk_rec, mem_rec = [], []

        for step in range(self.num_steps):
            x_t = x[step]  # (batch, 1, 64, 216)
            features = x_t.squeeze(1).mean(dim=-1)  # (batch, 64)

            # Random projection
            projected = F.linear(features, self.random_proj)

            # LIF neuron (spikes = WTA naturally via threshold)
            spk1, mem1 = self.lif(projected, mem1)

            # WTA: keep only top-k spikes
            topk_vals, topk_idx = spk1.topk(self.wta_k, dim=1)
            sparse_spk = torch.zeros_like(spk1)
            sparse_spk.scatter_(1, topk_idx, topk_vals)

            # Readout
            cur2 = self.readout(sparse_spk)
            spk2, mem2 = self.lif_out(cur2, mem2)

            spk_rec.append(spk2)
            mem_rec.append(mem2)

        return torch.stack(spk_rec), torch.stack(mem_rec)


def train_epoch_ann(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == 'cuda'

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            logits = model(data)
            loss = F.cross_entropy(logits, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = logits.argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_ann(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        logits = model(data)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)
    return correct / total


def train_epoch_snn(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == 'cuda'
    from src.encoding import encode_direct

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data).to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            spk_out, mem_out = model(spk_input)
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
def eval_snn(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    from src.encoding import encode_direct

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data).to(device)
        _, mem_out = model(spk_input)
        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    return correct / total


def run_fold(fold, device, expansion, wta_ratio, mode="ann"):
    print(f"\n  Fold {fold} ({mode}): expansion={expansion}x, WTA={wta_ratio}")
    train_loader, test_loader = get_fold_dataloaders(fold, BATCH_SIZE)

    if mode == "ann":
        model = FlyBrainClassifier(
            expansion=expansion, wta_ratio=wta_ratio).to(device)
        train_fn = train_epoch_ann
        eval_fn = eval_ann
    else:
        model = FlyBrainSNN(
            expansion=expansion, wta_ratio=wta_ratio).to(device)
        train_fn = train_epoch_snn
        eval_fn = eval_snn

    # Only readout weights are trained
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total_params:,} total")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_fn(model, train_loader, optimizer,
                                         device)
        test_acc = eval_fn(model, test_loader, device)
        scheduler.step(train_loss)

        if test_acc > best_acc:
