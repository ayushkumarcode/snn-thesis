"""
tet_training.py -- Temporal Efficient Training (TET) loss for SNNs.

Implements the TET loss from Deng et al. (ICLR 2022):
    L_TET = (1/T) * sum_t CE(mem_t, y) + lambda * (1/T) * sum_t (CE(mem_t, y) - L_mean)^2

The first term is the standard per-timestep CE loss (same as baseline).
The second term penalises variance across timesteps, encouraging the SNN
to make consistent predictions at EVERY timestep, not just at the end.
This flattens the temporal loss landscape and improves gradient flow.

Model uses learn_beta=True, learn_threshold=True, spike_rate_escape surrogate.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/tet_training.py --fold 1 --device mps
    python experiments/tet_training.py                      # all 5 folds
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    NUM_FOLDS, RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct


# ============================================================
# TET-SNN Model (learnable beta + threshold, spike_rate_escape)
# ============================================================

class TETSpikingCNN(nn.Module):
    """SpikingCNN with learnable beta/threshold and spike_rate_escape surrogate.

    Architecture matches the baseline SpikingCNN exactly:
    Conv2d(1,32) -> BN -> MaxPool(2) -> LIF -> Conv2d(32,64) -> BN -> MaxPool(2)
    -> LIF -> AvgPool(4,6) -> FC(2304,256) -> LIF -> FC(256,50) -> LIF

    Differences from baseline:
        - learn_beta=True: beta (membrane decay) is a learnable parameter
        - learn_threshold=True: threshold is a learnable parameter
        - spike_rate_escape surrogate gradient (best from ablation study)
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        beta: float = BETA,
        num_steps: int = NUM_STEPS,
    ):
        super().__init__()
        self.num_steps = num_steps

        spike_grad = surrogate.spike_rate_escape(beta=1.0, slope=25)

        # Convolutional feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        # After two MaxPool2d(2) on input (64, 216): (16, 54)
        # AvgPool2d(4,6) on (16,54) -> (4, 9)
        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # Fully connected classifier: 64 * 4 * 9 = 2304
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over all timesteps.

        Args:
            x: Input of shape (num_steps, batch, 1, n_mels, time_frames).

        Returns:
            spk_out: Output spikes, shape (num_steps, batch, num_classes).
            mem_out: Output membrane potentials, shape (num_steps, batch, num_classes).
        """
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
