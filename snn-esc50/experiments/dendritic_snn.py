"""
dendritic_snn.py -- Dendritic Spiking Neurons for ESC-50 classification.

Replaces standard LIF neurons with multi-compartment dendritic neurons.
Each DendriticLIF neuron has K dendritic branches with distinct learnable
time constants (fast/medium/slow), enabling multi-timescale temporal
feature extraction from audio spectrograms.

Branch dynamics:
    mem_branch_k[t] = beta_k * mem_branch_k[t-1] + w_k * input[t]
    mem_soma[t] = sum_k(w_k * mem_branch_k[t])
    spike if mem_soma > threshold, then reset all branches

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/dendritic_snn.py --fold 1 --device mps
    python experiments/dendritic_snn.py  # all 5 folds
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
    NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct


# ============================================================
# Dendritic LIF Neuron
# ============================================================

class DendriticLIF(nn.Module):
    """Multi-compartment dendritic LIF neuron with learnable branch dynamics.

    Each neuron has K dendritic branches. Each branch has:
      - A learnable decay rate (beta_k), initialized to different timescales
      - A learnable gating weight (w_k) controlling its contribution to the soma

    The soma integrates weighted branch outputs and generates spikes
    using a surrogate gradient when the membrane potential exceeds threshold.

