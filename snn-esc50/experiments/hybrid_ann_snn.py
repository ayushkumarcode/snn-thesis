"""
hybrid_ann_snn.py -- ANN-to-SNN hybrid initialization with fine-tuning.

Research question: Can transferring learned ANN weights to an enhanced SNN
(with learnable beta, threshold, dropout, and spike_rate_escape surrogate)
close the ANN-SNN accuracy gap?

Method:
  - Load trained ANN weights from results/ann/none/best_fold{fold}.pt
  - Map ANN conv/bn/fc layers to corresponding SNN layers
  - Create EnhancedSpikingCNN with learn_beta=True, learn_threshold=True,
    Dropout(0.3), spike_rate_escape surrogate gradient
  - Fine-tune for 20 epochs at lower learning rate (1e-4)
  - 5-fold CV, save results to results/experiments/hybrid_ann_snn/

Weight mapping (ConvANN -> SpikingCNN):
  features.0  (Conv2d)     -> conv1
  features.1  (BatchNorm)  -> bn1
  features.4  (Conv2d)     -> conv2
  features.5  (BatchNorm)  -> bn2
  classifier.0 (Linear)    -> fc1
  classifier.3 (Linear)    -> fc2

Usage:
  cd snn-esc50
  source .venv/bin/activate
  python experiments/hybrid_ann_snn.py
  python experiments/hybrid_ann_snn.py --fold 1 --epochs 20
  python experiments/hybrid_ann_snn.py --fold 0  # run all 5 folds
"""

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from src.config import (
    NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    BATCH_SIZE, WEIGHT_DECAY, PATIENCE, NUM_FOLDS,
    RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct
from src.models.ann_model import ConvANN


# ============================================================
# Enhanced SNN with learnable beta, threshold, and dropout
# ============================================================

class EnhancedSpikingCNN(nn.Module):
    """SpikingCNN with learnable membrane decay (beta), learnable threshold,
    dropout regularization, and spike_rate_escape surrogate gradient.

    Designed for ANN->SNN weight transfer: same conv/bn/fc dimensions as
    ConvANN, but with LIF neurons and temporal dynamics.

    Args:
        num_classes: Number of output classes.
        beta: Initial membrane potential decay rate.
        num_steps: Number of simulation timesteps.
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

        # -- Convolutional feature extraction --
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

        # -- Fully connected classifier with dropout --
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.dropout = nn.Dropout(0.3)
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

