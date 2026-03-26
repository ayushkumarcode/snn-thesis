"""
stochastic_resonance_training.py -- Trainable Stochastic Resonance SNN.

We already showed SR exists at sigma=0.02 (+0.25pp in inference). This script
makes sigma a LEARNABLE per-neuron parameter, optimized jointly with the network.

Custom SRLIF neuron:
    v[t] = beta * v[t-1] * (1 - spk[t-1]) + I[t] + sigma * randn()
    sigma is nn.Parameter per neuron, initialized at 0.02 (proven SR sweet spot)

Also includes SR+Rhythm variant: noise + oscillatory modulation combined.

Based on Entropy 2025: "Trainable Stochastic Resonance in Neural Networks".

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/stochastic_resonance_training.py --fold 1
    python experiments/stochastic_resonance_training.py --with-rhythm --fold 1
    python experiments/stochastic_resonance_training.py               # all 5 folds, SR only
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
# SRLIF Neuron: LIF with Trainable Stochastic Resonance
# ============================================================

class SRLIF(nn.Module):
    """Leaky Integrate-and-Fire neuron with learnable noise amplitude.

    Membrane dynamics:
        v[t] = beta * v[t-1] * (1 - spk[t-1]) + I[t] + sigma * N(0,1)

    sigma is an nn.Parameter per neuron (per channel for conv, per unit for FC).
