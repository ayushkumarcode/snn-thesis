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

