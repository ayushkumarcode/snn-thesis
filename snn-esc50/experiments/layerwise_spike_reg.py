"""
layerwise_spike_reg.py -- Train SNN with aggressive per-layer spike regularization.

Targets <6% spike rate (Dampfhoffer 2023 threshold for energy parity with ANN).
Applies separate L1 penalties to conv1, conv2, and fc1 hidden spikes.

Usage:
    python -m experiments.layerwise_spike_reg --device cuda
    python -m experiments.layerwise_spike_reg --lambda-conv 1e-3 --lambda-fc 5e-4 --device cuda
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
import snntorch as snn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    NUM_CLASSES, NUM_STEPS, BETA, RESULTS_DIR, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, PATIENCE, NUM_EPOCHS, get_device,
