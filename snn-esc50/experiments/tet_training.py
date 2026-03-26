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
