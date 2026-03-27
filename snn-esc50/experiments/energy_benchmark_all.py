"""
energy_benchmark_all.py -- NeuroBench energy analysis on ALL models.

Runs NeuroBench metrics (SynapticOperations, ActivationSparsity) on
every trained model and computes energy estimates for each.

Usage:
    python -m experiments.energy_benchmark_all --device cuda
    python -m experiments.energy_benchmark_all --fold 1 --device cuda
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
    NUM_CLASSES, NUM_STEPS, RESULTS_DIR, BATCH_SIZE, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct
from src.models.snn_model import SpikingCNN
from src.models.ann_model import ConvANN

ENERGY_PER_AC_PJ = 0.9   # pJ per accumulate (neuromorphic)
ENERGY_PER_MAC_PJ = 4.6  # pJ per multiply-accumulate (CMOS)


@torch.no_grad()
def measure_snn_energy(model, loader, device, num_steps=NUM_STEPS,
                       encode_fn=None):
    """Measure SNN energy by counting actual spike-driven operations."""
    model.eval()
    if encode_fn is None:
        encode_fn = encode_direct

    total_acs = 0
    total_samples = 0
    total_spikes_per_layer = {}
    correct = 0

    # Operation counts per layer (fan-out)
    fan_out = {
        "conv1": 32 * 9,   # 32 output channels * 3x3 kernel
        "conv2": 64 * 9,   # 64 output channels * 3x3 kernel
        "fc1": 256,         # 256 output neurons
        "fc2": NUM_CLASSES, # 50 output neurons
    }

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        B = data.shape[0]
        spk_input = encode_fn(data, num_steps=num_steps).to(device)

        # Forward pass with spike counting
        mem1 = model.lif1.init_leaky()
        mem2 = model.lif2.init_leaky()
        mem3 = model.lif3.init_leaky()
        mem4 = model.lif4.init_leaky()

        batch_spikes = {"conv1": 0, "conv2": 0, "fc1": 0, "fc2": 0}
        mem_out_list = []

        for step in range(num_steps):
            x_t = spk_input[step]
            cur1 = model.pool1(model.bn1(model.conv1(x_t)))
            spk1, mem1 = model.lif1(cur1, mem1)
            batch_spikes["conv1"] += spk1.sum().item()

            cur2 = model.pool2(model.bn2(model.conv2(spk1)))
            spk2, mem2 = model.lif2(cur2, mem2)
            batch_spikes["conv2"] += spk2.sum().item()

            pooled = model.avg_pool(spk2)
            flat = pooled.view(B, -1)

            cur3 = model.fc1(flat)
            spk3, mem3 = model.lif3(cur3, mem3)
            batch_spikes["fc1"] += spk3.sum().item()

            cur4 = model.fc2(spk3)
            spk4, mem4 = model.lif4(cur4, mem4)
            batch_spikes["fc2"] += spk4.sum().item()
            mem_out_list.append(mem4)

        mem_out = torch.stack(mem_out_list)
        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()

        # Compute ACs: spikes * fan-out for each layer
        for layer in batch_spikes:
            acs = batch_spikes[layer] * fan_out[layer]
            total_acs += acs
            total_spikes_per_layer[layer] = (
                total_spikes_per_layer.get(layer, 0) +
                batch_spikes[layer])

        total_samples += B

    avg_acs = total_acs / total_samples
    avg_energy_nj = avg_acs * ENERGY_PER_AC_PJ / 1000  # pJ -> nJ

    # Spike rates per layer
    spike_rates = {}
    total_neurons = {"conv1": 32*32*108, "conv2": 64*16*54,
