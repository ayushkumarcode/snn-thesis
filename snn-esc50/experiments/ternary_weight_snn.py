"""
ternary_weight_snn.py -- SNN with ternary weights {-1, 0, +1}.

Each synapse is just add/subtract (no multiply). Uses STE (straight-
through estimator) for gradients through quantization. Massive energy
savings: ternary multiply = 1 cycle vs float multiply = 3-5 cycles.

Usage:
    python -m experiments.ternary_weight_snn --device cuda
    python -m experiments.ternary_weight_snn --fold 1 --device cuda
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


class TernaryQuantize(torch.autograd.Function):
    """Ternary quantization with STE gradient."""

    @staticmethod
    def forward(ctx, weight, threshold=0.05):
        # Threshold-based ternary: w > thresh -> +1, w < -thresh -> -1, else 0
        scale = weight.abs().mean()
        ternary = torch.zeros_like(weight)
        ternary[weight > threshold * scale] = scale
        ternary[weight < -threshold * scale] = -scale
        return ternary

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient straight through
        return grad_output, None


class TernaryConv2d(nn.Module):
    """Conv2d with ternary weights during forward pass."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.threshold = 0.05

    def forward(self, x):
        tw = TernaryQuantize.apply(self.conv.weight, self.threshold)
        return F.conv2d(x, tw, self.conv.bias, self.conv.stride,
                        self.conv.padding)


class TernaryLinear(nn.Module):
    """Linear with ternary weights during forward pass."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)
        self.threshold = 0.05

    def forward(self, x):
        tw = TernaryQuantize.apply(self.linear.weight, self.threshold)
        return F.linear(x, tw, self.linear.bias)


class TernarySNN(nn.Module):
    """SNN with ternary quantized weights."""

    def __init__(self, num_steps=NUM_STEPS):
        super().__init__()
        self.num_steps = num_steps
        sg = surrogate.spike_rate_escape(beta=1, slope=25)

        self.conv1 = TernaryConv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = TernaryConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))
        self.fc1 = TernaryLinear(64 * 4 * 9, 256)
        self.fc2 = TernaryLinear(256, NUM_CLASSES)
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
