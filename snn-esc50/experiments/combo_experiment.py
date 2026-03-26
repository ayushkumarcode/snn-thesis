"""
Combinatorial experiment runner: stack multiple SNN techniques via flags.

Usage:
    # Rhythm + KD:
    python -m experiments.combo_experiment --rhythm --kd --device cuda

    # Rhythm + hybrid init + TET:
    python -m experiments.combo_experiment --rhythm --hybrid-init --tet --device cuda

    # Kitchen sink:
    python -m experiments.combo_experiment --rhythm --dendritic --delays --kd --tet --cochleagram --device cuda

    # Single fold test:
    python -m experiments.combo_experiment --rhythm --kd --fold 1 --device cuda

Techniques available:
    --learn-beta       Learnable beta per neuron (learn_beta=True)
    --learn-threshold  Learnable threshold per neuron
    --dropout          Dropout(0.3) before fc2
    --sre              spike_rate_escape surrogate (default: fast_sigmoid)
    --rhythm           Oscillatory modulation (Nature Comms 2025)
    --dendritic        Multi-compartment dendritic neurons (K=3 branches)
    --delays           Learnable synaptic delays on FC layers
    --kd               Knowledge distillation from ANN teacher
    --hybrid-init      Initialize SNN weights from trained ANN
    --tet              Temporal Efficient Training loss
    --cochleagram      Gammatone filterbank instead of mel spectrogram
    --l1-reg LAMBDA    L1 spike rate regularization

All techniques are composable. Results saved to results/experiments/combo_<hash>/
"""

import argparse
import hashlib
import json
import math
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
    NUM_CLASSES, NUM_STEPS, BETA, N_MELS,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    RESULTS_DIR, get_device, SAMPLE_RATE, DURATION, N_FFT, HOP_LENGTH,
)
from src.dataset import download_esc50, get_fold_dataloaders, ESC50Dataset
from src.encoding import encode_direct
from src.models.ann_model import ConvANN


# ============================================================
# Custom neuron modules
# ============================================================

class RhythmLIF(nn.Module):
    """LIF neuron with learnable oscillatory modulation."""

    def __init__(self, size, beta=BETA, spike_grad=None, learn_beta=True):
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)
        self.spike_grad = spike_grad
        self.size = size

        # Learnable beta via sigmoid
        beta_init = math.log(beta / (1 - beta))
        self.beta_raw = nn.Parameter(torch.full((size,), beta_init))

        # Oscillation parameters
        self.amplitude = nn.Parameter(torch.full((size,), 0.1))
        self.frequency = nn.Parameter(torch.empty(size).uniform_(0.5, 5.0))
        self.phase = nn.Parameter(torch.zeros(size))

        self.threshold = nn.Parameter(torch.ones(size)) if learn_beta else None
        self._threshold_val = 1.0

    def init_mem(self, device=None):
        if device is None:
            device = self.beta_raw.device
        return torch.zeros(1, device=device)

    def forward(self, x, mem, step):
        if mem.device != x.device:
            mem = mem.to(x.device)
        beta = torch.sigmoid(self.beta_raw)
        thresh = self.threshold if self.threshold is not None else self._threshold_val

        # Reshape for broadcasting
        shape = [1] * (x.dim() - 1) + [-1]
        if x.dim() == 4:  # conv: (batch, channels, h, w)
            shape = [1, -1, 1, 1]
        elif x.dim() == 2:  # fc: (batch, features)
            shape = [1, -1]

        b = beta.view(shape)
        t = thresh.view(shape) if isinstance(thresh, nn.Parameter) else thresh
        osc = (self.amplitude * torch.sin(
            2 * math.pi * self.frequency * step / NUM_STEPS + self.phase
        )).view(shape)

        mem = b * mem + x + osc
        spk = self.spike_grad((mem - t) / t)
        mem = mem * (1 - spk.detach())
        return spk, mem


class DendriticLIF(nn.Module):
    """Multi-compartment dendritic neuron with K branches."""

    def __init__(self, size, K=3, spike_grad=None):
        super().__init__()
        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)
        self.spike_grad = spike_grad
        self.K = K
        self.size = size

        # Branch betas (fast, medium, slow)
        init_betas = [0.7, 0.9, 0.99] if K == 3 else [0.7 + 0.3 * i / (K - 1) for i in range(K)]
        self.branch_beta_raw = nn.ParameterList([
            nn.Parameter(torch.full((size,), math.log(b / (1 - b)))) for b in init_betas
        ])
        # Branch gating weights
        self.gate_logits = nn.Parameter(torch.zeros(K, size))
        self.threshold = nn.Parameter(torch.ones(size))

    def init_mem(self, device=None):
        if device is None:
            device = self.branch_beta_raw[0].device
        return [torch.zeros(1, device=device) for _ in range(self.K)]

    def forward(self, x, branch_mems, step=None):
        branch_mems = [m.to(x.device) if m.device != x.device else m for m in branch_mems]
        shape = [1] * (x.dim() - 1) + [-1]
        if x.dim() == 4:
            shape = [1, -1, 1, 1]
        elif x.dim() == 2:
            shape = [1, -1]

        gates = F.softmax(self.gate_logits, dim=0)  # (K, size)
        soma = torch.zeros_like(x)
        new_mems = []

        for k in range(self.K):
            beta_k = torch.sigmoid(self.branch_beta_raw[k]).view(shape)
            gate_k = gates[k].view(shape)
            mem_k = beta_k * branch_mems[k] + gate_k * x
            new_mems.append(mem_k)
            soma = soma + mem_k

        thresh = self.threshold.view(shape)
        spk = self.spike_grad((soma - thresh) / thresh)

        # Reset all branches on spike
        reset = spk.detach()
        new_mems = [m * (1 - reset) for m in new_mems]

        return spk, new_mems

