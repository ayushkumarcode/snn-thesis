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


class DelayedLinear(nn.Module):
    """Linear layer with per-output-neuron learnable delays."""

    def __init__(self, in_features, out_features, max_delay=5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.max_delay = max_delay
        self.delay_raw = nn.Parameter(torch.empty(out_features).uniform_(0.1, 0.9))
        self.buffer = None

    def init_buffer(self, batch_size, in_features, device):
        self.buffer = torch.zeros(self.max_delay + 1, batch_size, in_features, device=device)

    def forward(self, x):
        if self.buffer is None or self.buffer.shape[1] != x.shape[0]:
            self.init_buffer(x.shape[0], x.shape[1], x.device)

        # Shift buffer and add new input
        self.buffer = torch.roll(self.buffer, 1, dims=0)
        self.buffer[0] = x

        # Soft delay lookup (differentiable)
        delays = torch.sigmoid(self.delay_raw) * self.max_delay  # (out_features,)
        delay_floor = delays.long().clamp(0, self.max_delay - 1)
        delay_frac = delays - delay_floor.float()

        # Gather delayed inputs for each output neuron
        out = torch.zeros(x.shape[0], self.linear.out_features, device=x.device)
        for j in range(self.linear.out_features):
            d = delay_floor[j]
            f = delay_frac[j]
            delayed_input = (1 - f) * self.buffer[d] + f * self.buffer[min(d + 1, self.max_delay)]
            out[:, j] = F.linear(delayed_input, self.linear.weight[j:j+1], self.linear.bias[j:j+1]).squeeze()

        return out

    def reset(self):
        self.buffer = None


# ============================================================
# Cochleagram preprocessing
# ============================================================

def gammatone_filterbank(sr, n_fft, n_filters=64, fmin=50):
    """Create gammatone filterbank matrix."""
    import librosa
    fmax = sr / 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # ERB-spaced center frequencies
    ear_q = 9.26449
    min_bw = 24.7
    cf_low = -(ear_q * min_bw) + np.exp(np.log(fmin + ear_q * min_bw) +
              np.arange(0, n_filters) * (-np.log(fmax + ear_q * min_bw) +
              np.log(fmin + ear_q * min_bw)) / (n_filters - 1)) * 0
    cfs = np.exp(np.linspace(np.log(fmin + ear_q * min_bw),
                              np.log(fmax + ear_q * min_bw), n_filters)) - ear_q * min_bw

    fb = np.zeros((n_filters, len(freqs)))
    for i, cf in enumerate(cfs):
        erb = 24.7 * (4.37 * cf / 1000 + 1)
        b = 1.019 * 2 * np.pi * erb
        resp = ((2 * np.pi * np.abs(freqs - cf)) ** 2 + b ** 2) ** (-2)
        fb[i] = resp / (resp.max() + 1e-10)

    return fb.astype(np.float32)


def wav_to_cochleagram(filepath, fb_matrix):
    """Convert audio to gammatone cochleagram."""
    import librosa
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
    expected_len = SAMPLE_RATE * DURATION
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    cochleagram = fb_matrix @ S
    cochleagram = librosa.power_to_db(cochleagram + 1e-10, ref=np.max)

    # Normalize to [0, 1]
    mn, mx = cochleagram.min(), cochleagram.max()
    if mx - mn > 0:
        cochleagram = (cochleagram - mn) / (mx - mn)
    else:
        cochleagram = np.zeros_like(cochleagram)

    return cochleagram.astype(np.float32)


# ============================================================
# Combo model builder
# ============================================================

class ComboSpikingCNN(nn.Module):
    """Configurable SNN that can combine any techniques."""

    def __init__(self, args):
        super().__init__()
        self.num_steps = NUM_STEPS
        self.args = args
        self.use_delays = args.delays
        self.use_dendritic = args.dendritic
        self.use_rhythm = args.rhythm

        sg = surrogate.spike_rate_escape(beta=1, slope=25) if args.sre else surrogate.fast_sigmoid(slope=25)

        # Conv layers (shared across all configs)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # FC layers
        if args.delays:
            self.fc1 = DelayedLinear(64 * 4 * 9, 256, max_delay=args.max_delay)
            self.fc2 = DelayedLinear(256, NUM_CLASSES, max_delay=args.max_delay)
        else:
            self.fc1 = nn.Linear(64 * 4 * 9, 256)
            self.fc2 = nn.Linear(256, NUM_CLASSES)

        # Dropout
        self.dropout = nn.Dropout(0.3) if args.dropout else nn.Identity()

        # Neuron layers
        if args.dendritic:
            self.n1 = DendriticLIF(32, K=args.branches, spike_grad=sg)
            self.n2 = DendriticLIF(64, K=args.branches, spike_grad=sg)
            self.n3 = DendriticLIF(256, K=args.branches, spike_grad=sg)
            self.n4 = DendriticLIF(NUM_CLASSES, K=args.branches, spike_grad=sg)
        elif args.rhythm:
            self.n1 = RhythmLIF(32, BETA, sg, args.learn_beta)
            self.n2 = RhythmLIF(64, BETA, sg, args.learn_beta)
            self.n3 = RhythmLIF(256, BETA, sg, args.learn_beta)
            self.n4 = RhythmLIF(NUM_CLASSES, BETA, sg, args.learn_beta)
        else:
            self.n1 = snn.Leaky(beta=BETA, spike_grad=sg,
                                 learn_beta=args.learn_beta, learn_threshold=args.learn_threshold)
            self.n2 = snn.Leaky(beta=BETA, spike_grad=sg,
                                 learn_beta=args.learn_beta, learn_threshold=args.learn_threshold)
            self.n3 = snn.Leaky(beta=BETA, spike_grad=sg,
                                 learn_beta=args.learn_beta, learn_threshold=args.learn_threshold)
            self.n4 = snn.Leaky(beta=BETA, spike_grad=sg,
                                 learn_beta=args.learn_beta, learn_threshold=args.learn_threshold)

    def _init_states(self, device):
        if self.use_dendritic:
            return [n.init_mem(device) for n in [self.n1, self.n2, self.n3, self.n4]]
        elif self.use_rhythm:
            return [n.init_mem(device) for n in [self.n1, self.n2, self.n3, self.n4]]
        else:
            return [n.init_leaky() for n in [self.n1, self.n2, self.n3, self.n4]]

    def forward(self, x):
        device = x.device
        states = self._init_states(device)
        m1, m2, m3, m4 = states

        if self.use_delays:
            self.fc1.reset()
            self.fc2.reset()
