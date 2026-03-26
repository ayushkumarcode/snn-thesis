"""
rhythm_snn.py -- Oscillatory Modulation (Rhythm-SNN) for ESC-50.

Implements learnable oscillatory modulation of membrane potentials, inspired by
Zhao et al. (Nature Communications 2025) "Rhythm-SNN":
    v[t] = beta * v[t-1] + I[t] + A * sin(2*pi*f*t/T + phi)

where A (amplitude), f (frequency), phi (phase) are LEARNABLE per-neuron
parameters. This adds a biologically-motivated oscillatory current that
can help temporal feature extraction in audio classification.

Model also uses learn_beta=True, spike_rate_escape surrogate, Dropout(0.3).

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/rhythm_snn.py --fold 1 --device mps
    python experiments/rhythm_snn.py                      # all 5 folds
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
# RhythmLIF Neuron
# ============================================================

class RhythmLIF(nn.Module):
    """Leaky Integrate-and-Fire neuron with learnable oscillatory modulation.

    Membrane dynamics:
        v[t] = beta * v[t-1] * (1 - spk[t-1]) + I[t] + A * sin(2*pi*f*t/T + phi)

    When v[t] >= threshold, a spike is emitted and v is reset (soft reset via
    multiplication by (1 - spk), same as snnTorch Leaky default).

