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

