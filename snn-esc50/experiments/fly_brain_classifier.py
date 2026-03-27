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

