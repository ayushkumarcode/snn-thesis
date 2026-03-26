"""
learnable_delays.py -- Learnable Synaptic Delays for temporal pattern matching.

Adds per-output-neuron learnable delays to FC layers. Each output neuron
reads from a past timestep in its input history, enabling the network to
learn temporal alignment patterns in audio spectrograms.

Implementation:
    - DelayedLinear wraps nn.Linear with a delay buffer
    - Each output neuron j has a learnable delay d_j in [0, max_delay]
    - During training: continuous delays, soft interpolation between timesteps
    - During inference: delays rounded to nearest integer
    - Applied to FC1 (2304->256) and FC2 (256->50) only

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/learnable_delays.py --fold 1 --device mps
    python experiments/learnable_delays.py  # all 5 folds
    python experiments/learnable_delays.py --max-delay 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct


# ============================================================
# Delayed Linear Layer
# ============================================================

class DelayedLinear(nn.Module):
    """Linear layer with per-output-neuron learnable synaptic delays.

    Maintains a circular buffer of past inputs. Each output neuron j reads
    from buffer[delay_j] rather than the current input, allowing the network
    to learn temporal offsets for pattern matching.

    During training, delays are continuous and soft interpolation is used
    between adjacent integer timesteps for gradient flow. During eval,
    delays are rounded to the nearest integer.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        max_delay: Maximum allowed delay in timesteps.
        bias: Whether to include bias.
    """

    def __init__(self, in_features: int, out_features: int,
                 max_delay: int = 5, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_delay = max_delay

        # Standard linear weights
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Learnable delay per output neuron
        # Initialize to small random values near 0.5 so frac is nonzero from
        # the start, ensuring gradient flow through the interpolation weights.
        # Stored as raw parameter, clamped to [0, max_delay] during forward.
        self.delay_raw = nn.Parameter(
            torch.empty(out_features).uniform_(0.1, 0.9)
        )
