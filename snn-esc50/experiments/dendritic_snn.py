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

    Args:
        size: Number of neurons (channels for conv, hidden_size for FC).
        num_branches: Number of dendritic branches (K).
        beta_inits: Initial beta values per branch. Defaults to [0.7, 0.9, 0.99]
                    for fast/medium/slow timescales.
        threshold: Spike threshold.
        spike_grad: Surrogate gradient function.
    """

    def __init__(
        self,
        size: int,
        num_branches: int = 3,
        beta_inits: list[float] | None = None,
        threshold: float = 1.0,
        spike_grad=None,
    ):
        super().__init__()
        self.size = size
        self.num_branches = num_branches
        self.threshold = threshold

        if spike_grad is None:
            spike_grad = surrogate.spike_rate_escape(beta=1.0, slope=25)
        self.spike_grad = spike_grad

        # Default branch timescales: fast, medium, slow
        if beta_inits is None:
            if num_branches == 1:
                beta_inits = [0.9]
            elif num_branches == 2:
                beta_inits = [0.7, 0.99]
            elif num_branches == 3:
                beta_inits = [0.7, 0.9, 0.99]
            else:
                # Linearly space from 0.7 to 0.99
                beta_inits = [0.7 + (0.99 - 0.7) * i / (num_branches - 1)
                              for i in range(num_branches)]

        # Learnable branch decay rates (stored as logits for sigmoid → (0,1))
        # inverse sigmoid: logit = log(beta / (1 - beta))
        beta_logits = []
        for b in beta_inits:
            b_clamped = max(min(b, 0.999), 0.001)
            beta_logits.append(torch.log(torch.tensor(b_clamped / (1.0 - b_clamped))))
        # Shape: (K, size) — each neuron in each branch has its own beta
        self.beta_logits = nn.Parameter(
            torch.stack(beta_logits).unsqueeze(1).expand(-1, size).clone()
        )  # (K, size)

        # Learnable gating weights per branch, initialized to 1/K
        self.gate_logits = nn.Parameter(
            torch.zeros(num_branches, size)
        )  # softmax → uniform init

    @property
    def betas(self):
        """Get branch decay rates in (0, 1) via sigmoid."""
        return torch.sigmoid(self.beta_logits)  # (K, size)

    @property
    def gates(self):
        """Get branch gating weights via softmax over branches."""
        return torch.softmax(self.gate_logits, dim=0)  # (K, size)

    def init_dendritic(self, batch_size=None, device=None):
        """Initialize branch membrane states to zero.

        Returns a list of K zero tensors, one per branch.
        If batch_size is None, returns a list of scalar zeros (will broadcast).
        """
        if batch_size is not None and device is not None:
            return [torch.zeros(batch_size, self.size, device=device)
                    for _ in range(self.num_branches)]
        else:
            return [torch.zeros(1, device=self.beta_logits.device)
                    for _ in range(self.num_branches)]

    def forward(self, input_current, branch_mems):
        """Single timestep forward pass.

        Args:
            input_current: Current input to the neuron. Shape depends on usage:
                - FC layers: (batch, size)
                - Conv layers: (batch, channels, H, W) — size=channels
            branch_mems: List of K branch membrane tensors, same shape as input_current.

        Returns:
            spk: Output spikes (same shape as input_current).
            new_branch_mems: Updated list of K branch membrane tensors.
        """
        betas = self.betas  # (K, size)
        gates = self.gates  # (K, size)

        is_spatial = input_current.dim() > 2  # conv layer input: (B, C, H, W)

        new_branch_mems = []
        soma = torch.zeros_like(input_current)

        for k in range(self.num_branches):
            beta_k = betas[k]  # (size,)
            gate_k = gates[k]  # (size,)

            if is_spatial:
                # Reshape for broadcasting: (1, C, 1, 1)
                beta_k = beta_k.view(1, -1, 1, 1)
                gate_k = gate_k.view(1, -1, 1, 1)

                # Expand branch_mems if they were initialized as scalar
                if branch_mems[k].dim() < input_current.dim():
                    branch_mems[k] = torch.zeros_like(input_current)
            else:
                # FC: (1, size) for broadcasting over batch
                beta_k = beta_k.unsqueeze(0)
                gate_k = gate_k.unsqueeze(0)

                if branch_mems[k].dim() < input_current.dim():
                    branch_mems[k] = torch.zeros_like(input_current)

            # Branch membrane dynamics: leak + input
            mem_k = beta_k * branch_mems[k] + gate_k * input_current
            new_branch_mems.append(mem_k)

            # Weighted contribution to soma
            soma = soma + gate_k * mem_k

        # Spike generation with surrogate gradient
        spk = self.spike_grad(soma - self.threshold)

        # Reset: on spike, reset all branch membranes
        # spk is binary (0 or 1), so (1 - spk) zeroes out where spike occurred
        reset_branch_mems = []
        for k in range(self.num_branches):
            if is_spatial:
                reset = new_branch_mems[k] * (1.0 - spk)
            else:
                reset = new_branch_mems[k] * (1.0 - spk)
            reset_branch_mems.append(reset)

        return spk, reset_branch_mems


# ============================================================
# Dendritic SNN Model
# ============================================================

class DendriticSpikingCNN(nn.Module):
    """Convolutional SNN with dendritic LIF neurons for ESC-50.

    Same architecture as SpikingCNN but with DendriticLIF replacing Leaky:
    Conv2d(1,32) -> BN -> MaxPool(2) -> DendLIF
    Conv2d(32,64) -> BN -> MaxPool(2) -> DendLIF
    AvgPool(4,6) -> FC(2304,256) -> DendLIF -> Dropout -> FC(256,50) -> DendLIF

    Args:
        num_classes: Number of output classes.
        num_steps: Number of simulation timesteps.
        num_branches: Number of dendritic branches per neuron.
        spike_grad: Surrogate gradient function.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_steps: int = NUM_STEPS,
        num_branches: int = 3,
        spike_grad=None,
    ):
