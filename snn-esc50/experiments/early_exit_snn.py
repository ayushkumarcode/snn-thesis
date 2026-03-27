"""
early_exit_snn.py -- Confidence-based early exit for energy reduction.

Tests how many timesteps are actually needed per sample by checking
output confidence at each timestep. Easy samples exit early (T=3-5),
hard samples use full T=25. Directly reduces energy proportionally.

Usage:
    python -m experiments.early_exit_snn --device cuda
    python -m experiments.early_exit_snn --model-type rhythm --device cuda
    python -m experiments.early_exit_snn --fold 1 --device cuda
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
    NUM_CLASSES, NUM_STEPS, BETA, RESULTS_DIR, BATCH_SIZE, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct
from src.models.snn_model import SpikingCNN

# Import combo model for rhythm/dendritic
from experiments.combo_experiment import ComboSpikingCNN


# ============================================================
# Early exit inference
# ============================================================

@torch.no_grad()
def early_exit_inference(model, loader, device, thresholds,
                         stability_k=2, model_type="baseline"):
    """Run inference with confidence-based early exit.

    At each timestep, check max softmax probability of accumulated
    membrane potentials. If confidence > threshold for k consecutive
    steps, exit early.

    Returns dict mapping threshold -> results.
    """
    model.eval()
    use_amp = device.type == 'cuda'

