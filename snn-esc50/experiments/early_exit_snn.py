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
    T = all_mems.shape[0]

    results = {}
    for thresh in thresholds:
        # For each sample, find earliest exit timestep
        exit_steps = torch.full((N,), T, dtype=torch.long)
        predictions = torch.zeros(N, dtype=torch.long)
        stable_counts = torch.zeros(N, dtype=torch.long)
        prev_preds = torch.full((N,), -1, dtype=torch.long)

        cum_mem = torch.zeros(N, NUM_CLASSES)
        for t in range(T):
            cum_mem = cum_mem + all_mems[t]
            probs = F.softmax(cum_mem, dim=1)
            max_conf, pred_t = probs.max(dim=1)

            # Check stability
            same_pred = (pred_t == prev_preds)
            stable_counts = torch.where(same_pred, stable_counts + 1,
                                        torch.ones_like(stable_counts))
            prev_preds = pred_t

            # Exit condition: confidence > threshold AND stable for k steps
            can_exit = (max_conf > thresh) & (stable_counts >= stability_k)
            # Only exit samples that haven't exited yet
            newly_exiting = can_exit & (exit_steps == T)
            exit_steps[newly_exiting] = t + 1
            predictions[newly_exiting] = pred_t[newly_exiting]
