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

