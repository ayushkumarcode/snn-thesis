"""
combined_energy.py -- Combined Tier 1 energy reduction.

Stacks: early exit + layerwise spike regularization + silence gating.
Projected ~20x energy reduction over baseline SNN.

This script trains a spike-regularized rhythm SNN, then evaluates it
with early exit and silence gating applied at inference time.

Usage:
    python -m experiments.combined_energy --device cuda
    python -m experiments.combined_energy --fold 1 --device cuda
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
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct
from experiments.combo_experiment import RhythmLIF


class CombinedEnergySNN(nn.Module):
    """SNN optimized for energy: spike reg + silence gating support."""

    def __init__(self, num_steps=NUM_STEPS, silence_threshold=0.01):
        super().__init__()
        self.num_steps = num_steps
        self.silence_threshold = silence_threshold
        sg = surrogate.spike_rate_escape(beta=1, slope=25)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.fc2 = nn.Linear(256, NUM_CLASSES)
        self.dropout = nn.Dropout(0.3)

        self.n1 = RhythmLIF(32, BETA, sg, learn_beta=True)
        self.n2 = RhythmLIF(64, BETA, sg, learn_beta=True)
        self.n3 = RhythmLIF(256, BETA, sg, learn_beta=True)
        self.n4 = RhythmLIF(NUM_CLASSES, BETA, sg, learn_beta=True)

    def forward(self, x, silence_gate=False, early_exit_thresh=None,
                stability_k=2):
        """Forward with optional silence gating and early exit.

        Returns (spk_out, mem_out, layer_spikes, exit_info)
        """
        device = x.device
        m1 = self.n1.init_mem(device)
        m2 = self.n2.init_mem(device)
        m3 = self.n3.init_mem(device)
        m4 = self.n4.init_mem(device)

        spk_rec, mem_rec = [], []
        layer_spikes = {"conv1": 0.0, "conv2": 0.0, "fc1": 0.0}
        active_steps = 0
        exit_step = self.num_steps

        # For early exit tracking
        cum_mem = None
        stable_count = 0
        prev_pred = -1

        for step in range(self.num_steps):
            x_t = x[step]

            # Silence gating: skip if input energy is below threshold
            if silence_gate:
                energy = x_t.abs().mean()
                if energy < self.silence_threshold:
                    # Still decay membrane potentials (leaky behavior)
                    # but don't compute any layers
                    if len(mem_rec) > 0:
                        mem_rec.append(mem_rec[-1])
                        spk_rec.append(torch.zeros_like(spk_rec[-1]))
                    continue

            active_steps += 1

            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, m1 = self.n1(cur1, m1, step)
            layer_spikes["conv1"] = layer_spikes["conv1"] + spk1.mean()

            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, m2 = self.n2(cur2, m2, step)
            layer_spikes["conv2"] = layer_spikes["conv2"] + spk2.mean()

            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            cur3 = self.fc1(flat)
            spk3, m3 = self.n3(cur3, m3, step)
            layer_spikes["fc1"] = layer_spikes["fc1"] + spk3.mean()

            spk3 = self.dropout(spk3)

            cur4 = self.fc2(spk3)
            spk4, m4 = self.n4(cur4, m4, step)

            spk_rec.append(spk4)
            mem_rec.append(m4)

            # Early exit check
            if early_exit_thresh is not None and not self.training:
                if cum_mem is None:
                    cum_mem = m4.clone()
                else:
                    cum_mem = cum_mem + m4
                probs = F.softmax(cum_mem, dim=1)
                max_conf, pred = probs.max(dim=1)

                # Check stability across batch
                mean_conf = max_conf.mean().item()
                curr_pred = pred[0].item()

                if curr_pred == prev_pred:
                    stable_count += 1
