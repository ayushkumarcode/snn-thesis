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

    # Collect all samples first
    all_data, all_targets = [], []
    for data, targets in loader:
        all_data.append(data)
        all_targets.append(targets)
    all_data = torch.cat(all_data, dim=0).to(device)
    all_targets = torch.cat(all_targets, dim=0).to(device)
    N = all_data.shape[0]

    # Full forward pass collecting per-timestep membrane potentials
    # Process in batches to avoid OOM
    all_mems = []  # will be (T, N, C)
    bs = BATCH_SIZE
    for start in range(0, N, bs):
        end = min(start + bs, N)
        batch = all_data[start:end]
        spk_input = encode_direct(batch).to(device)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
            if model_type == "baseline":
                _, mem_out = model(spk_input)
            else:
                _, mem_out, _ = model(spk_input)

        all_mems.append(mem_out.cpu())

    # Concatenate: (T, N, C)
    all_mems = torch.cat(all_mems, dim=1).float()
