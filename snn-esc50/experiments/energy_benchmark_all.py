"""
energy_benchmark_all.py -- NeuroBench energy analysis on ALL models.

Runs NeuroBench metrics (SynapticOperations, ActivationSparsity) on
every trained model and computes energy estimates for each.

Usage:
    python -m experiments.energy_benchmark_all --device cuda
    python -m experiments.energy_benchmark_all --fold 1 --device cuda
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
    NUM_CLASSES, NUM_STEPS, RESULTS_DIR, BATCH_SIZE, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct
from src.models.snn_model import SpikingCNN
from src.models.ann_model import ConvANN

ENERGY_PER_AC_PJ = 0.9   # pJ per accumulate (neuromorphic)
ENERGY_PER_MAC_PJ = 4.6  # pJ per multiply-accumulate (CMOS)


@torch.no_grad()
def measure_snn_energy(model, loader, device, num_steps=NUM_STEPS,
                       encode_fn=None):
    """Measure SNN energy by counting actual spike-driven operations."""
    model.eval()
    if encode_fn is None:
        encode_fn = encode_direct

    total_acs = 0
    total_samples = 0
    total_spikes_per_layer = {}
    correct = 0

    # Operation counts per layer (fan-out)
    fan_out = {
        "conv1": 32 * 9,   # 32 output channels * 3x3 kernel
        "conv2": 64 * 9,   # 64 output channels * 3x3 kernel
        "fc1": 256,         # 256 output neurons
        "fc2": NUM_CLASSES, # 50 output neurons
    }

