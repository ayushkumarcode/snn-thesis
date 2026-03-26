"""
info_bottleneck_snn.py -- Information Bottleneck SNN experiment.

Compress spike representations via variational information bottleneck.
Based on "Learning to Time-Decode via Information Bottleneck" (NeurIPS, 2024).

Adds a variational information bottleneck (VIB) after the hidden layer (lif3):
  - mu = Linear(256, 256)
  - logvar = Linear(256, 256)
  - z = mu + exp(0.5 * logvar) * N(0,1)  (reparameterisation trick)
  - KL_loss = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
  - Total loss = CE_loss + beta_ib * KL_loss

During training, z replaces spk3 as input to FC2. The KL term encourages
the 256-dim hidden spike representation to be maximally compressed while
retaining task-relevant information. This acts as a powerful regulariser
on the small 1600-sample ESC-50 dataset.

During evaluation: use mu directly (no sampling, no stochasticity).

Also uses: learn_beta=True, Dropout(0.3), spike_rate_escape surrogate.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/info_bottleneck_snn.py --fold 1
    python experiments/info_bottleneck_snn.py --beta-ib 1e-3    # all 5 folds
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


# ============================================================
# Information Bottleneck SNN Model
# ============================================================

class InfoBottleneckSNN(nn.Module):
    """SpikingCNN with variational information bottleneck on hidden layer.

    After lif3 produces 256-dim hidden spikes, a VIB layer compresses
    the representation into a stochastic latent z. The KL divergence
    regulariser encourages z to be as close to N(0,1) as possible
    while still allowing accurate classification -- this forces the
    network to learn maximally compressed features.
    """

    def __init__(self, num_classes=NUM_CLASSES, beta=BETA, num_steps=NUM_STEPS):
        super().__init__()
        self.num_steps = num_steps

        spike_grad = surrogate.spike_rate_escape(beta=1, slope=25)

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))
