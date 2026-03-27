"""
spike_budget_snn.py -- Energy-aware spike budgeting with feedback controller.

Based on arXiv 2602.12236: Adaptive spike scheduler enforces dataset-specific
energy constraints during training. Uses proportional feedback controller with
learnable LIF parameters (beta, threshold) to auto-tune for target spike budget.

Key insight: Can reduce spikes by 47% while IMPROVING accuracy on frame-based
datasets. The controller adjusts threshold dynamically to meet a target spike
rate per layer.

Usage:
    python -m experiments.spike_budget_snn --target-rate 0.06 --device cuda
    python -m experiments.spike_budget_snn --device cuda  # sweep targets
"""

import argparse
import json
import math
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


class AdaptiveLIF(nn.Module):
    """LIF with learnable beta and adaptive threshold via feedback controller."""

    def __init__(self, size, target_rate=0.06, kp=0.1):
        super().__init__()
        sg = surrogate.spike_rate_escape(beta=1, slope=25)
        self.spike_grad = sg
        self.size = size
        self.target_rate = target_rate
        self.kp = kp
        beta_init = math.log(BETA / (1 - BETA))
        self.beta_raw = nn.Parameter(torch.full((size,), beta_init))
        self.base_threshold = nn.Parameter(torch.ones(size))
        self.amplitude = nn.Parameter(torch.full((size,), 0.1))
        self.frequency = nn.Parameter(torch.empty(size).uniform_(0.5, 5.0))
        self.phase = nn.Parameter(torch.zeros(size))

    def init_state(self, device=None):
        if device is None:
            device = self.beta_raw.device
        return {
            "mem": torch.zeros(1, device=device),
            "running_rate": torch.full((self.size,), self.target_rate, device=device),
            "adaptive_thresh": self.base_threshold.data.clone(),
        }

    def forward(self, x, state, step):
        mem = state["mem"]
        if mem.device != x.device:
            mem = mem.to(x.device)
            state["running_rate"] = state["running_rate"].to(x.device)
            state["adaptive_thresh"] = state["adaptive_thresh"].to(x.device)
        beta = torch.sigmoid(self.beta_raw)
        shape = [1] * (x.dim() - 1) + [-1]
        if x.dim() == 4:
            shape = [1, -1, 1, 1]
        elif x.dim() == 2:
            shape = [1, -1]
        b = beta.view(shape)
        osc = (self.amplitude * torch.sin(
            2 * math.pi * self.frequency * step / NUM_STEPS + self.phase
        )).view(shape)
        thresh = state["adaptive_thresh"].view(shape)
        mem = b * mem + x + osc
        spk = self.spike_grad((mem - thresh) / thresh.abs().clamp(min=0.1))
        mem = mem * (1 - spk.detach())
        if x.dim() == 4:
            current_rate = spk.detach().mean(dim=(0, 2, 3))
        else:
            current_rate = spk.detach().mean(dim=0)
        ema_alpha = 0.1
        state["running_rate"] = (1 - ema_alpha) * state["running_rate"] + ema_alpha * current_rate
        error = state["running_rate"] - self.target_rate
        state["adaptive_thresh"] = (self.base_threshold + self.kp * error).clamp(min=0.5)
        state["mem"] = mem
        return spk, state


class SpikeBudgetSNN(nn.Module):
    """SNN with spike budget control per layer."""

    def __init__(self, target_rate=0.06, kp=0.1):
        super().__init__()
        self.num_steps = NUM_STEPS
        self.target_rate = target_rate
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
        self.n1 = AdaptiveLIF(32, target_rate, kp)
        self.n2 = AdaptiveLIF(64, target_rate, kp)
        self.n3 = AdaptiveLIF(256, target_rate, kp)
        self.n4 = AdaptiveLIF(NUM_CLASSES, target_rate, kp)

    def forward(self, x):
        device = x.device
        s1 = self.n1.init_state(device)
        s2 = self.n2.init_state(device)
        s3 = self.n3.init_state(device)
        s4 = self.n4.init_state(device)
        spk_rec, mem_rec = [], []
        layer_rates = {"conv1": 0, "conv2": 0, "fc1": 0, "out": 0}
        for step in range(self.num_steps):
            x_t = x[step]
            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, s1 = self.n1(cur1, s1, step)
            layer_rates["conv1"] += spk1.mean().item()
            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, s2 = self.n2(cur2, s2, step)
            layer_rates["conv2"] += spk2.mean().item()
            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)
            cur3 = self.fc1(flat)
            spk3, s3 = self.n3(cur3, s3, step)
            layer_rates["fc1"] += spk3.mean().item()
            spk3 = self.dropout(spk3)
            cur4 = self.fc2(spk3)
            spk4, s4 = self.n4(cur4, s4, step)
            layer_rates["out"] += spk4.mean().item()
            spk_rec.append(spk4)
            mem_rec.append(s4["mem"])
        for k in layer_rates:
            layer_rates[k] /= self.num_steps
        return torch.stack(spk_rec), torch.stack(mem_rec), layer_rates


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == 'cuda'
    all_rates = {"conv1": 0, "conv2": 0, "fc1": 0, "out": 0}
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data).to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            _, mem_out, rates = model(spk_input)
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
        for k in all_rates:
            all_rates[k] += rates[k]
    n = len(loader)
    return total_loss / n, correct / total, {k: v/n for k, v in all_rates.items()}


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_rates = {"conv1": 0, "conv2": 0, "fc1": 0, "out": 0}
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data).to(device)
        _, mem_out, rates = model(spk_input)
        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
        for k in all_rates:
            all_rates[k] += rates[k]
