"""
extreme_pruning.py -- Iterative magnitude pruning to 95%/99% sparsity.

We showed 90% pruning retains 93.2% of accuracy. Push further with
iterative pruning + fine-tuning between rounds on the rhythm model.

Usage:
    python -m experiments.extreme_pruning --device cuda
    python -m experiments.extreme_pruning --fold 1 --device cuda
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
import torch.nn.utils.prune as prune

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    NUM_CLASSES, NUM_STEPS, RESULTS_DIR, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct
from experiments.combo_experiment import ComboSpikingCNN


def build_args(rhythm=True):
    """Build args for ComboSpikingCNN."""
    class Args:
        pass
    a = Args()
    a.rhythm = rhythm
    a.dendritic = False
    a.delays = False
    a.learn_beta = True
    a.learn_threshold = False
    a.dropout = True
    a.sre = True
    a.kd = False
    a.hybrid_init = False
    a.tet = False
    a.cochleagram = False
    a.l1_reg = 0.0
    a.branches = 3
    a.max_delay = 5
    return a


def apply_global_pruning(model, target_sparsity):
    """Apply global L1 unstructured pruning."""
    params = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params.append((module, "weight"))
    if params:
        prune.global_unstructured(
            params, pruning_method=prune.L1Unstructured,
            amount=target_sparsity)


def remove_pruning(model):
    """Make pruning permanent."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass


def count_sparsity(model):
    """Count actual weight sparsity."""
    total = 0
    zeros = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            total += param.numel()
