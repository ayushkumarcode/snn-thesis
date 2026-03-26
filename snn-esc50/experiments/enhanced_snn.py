"""
Experiment 1: Enhanced SNN with quick fixes.

Changes from baseline SpikingCNN:
  - Dropout(0.3) before fc2 (ANN has this, SNN doesn't)
  - learn_beta=True on all LIF neurons (heterogeneous membrane dynamics)
  - learn_threshold=True on all LIF neurons
  - spike_rate_escape surrogate (proven best in our ablation: 46.00% vs 44.75%)

Usage:
    python -m experiments.enhanced_snn                  # all 5 folds
    python -m experiments.enhanced_snn --fold 1         # single fold
    python -m experiments.enhanced_snn --device cuda    # specify device
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
    NUM_CLASSES, NUM_STEPS, N_MELS, BETA,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct


class EnhancedSpikingCNN(nn.Module):
    """SpikingCNN with dropout, learnable beta/threshold, and SRE surrogate."""

    def __init__(self, num_classes=NUM_CLASSES, beta=BETA, num_steps=NUM_STEPS):
        super().__init__()
        self.num_steps = num_steps

        spike_grad = surrogate.spike_rate_escape(beta=1, slope=25)

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                               learn_beta=True, learn_threshold=True)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
