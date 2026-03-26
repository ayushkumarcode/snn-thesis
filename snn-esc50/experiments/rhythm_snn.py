"""
rhythm_snn.py -- Oscillatory Modulation (Rhythm-SNN) for ESC-50.

Implements learnable oscillatory modulation of membrane potentials, inspired by
Zhao et al. (Nature Communications 2025) "Rhythm-SNN":
    v[t] = beta * v[t-1] + I[t] + A * sin(2*pi*f*t/T + phi)

where A (amplitude), f (frequency), phi (phase) are LEARNABLE per-neuron
parameters. This adds a biologically-motivated oscillatory current that
can help temporal feature extraction in audio classification.

Model also uses learn_beta=True, spike_rate_escape surrogate, Dropout(0.3).

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/rhythm_snn.py --fold 1 --device mps
    python experiments/rhythm_snn.py                      # all 5 folds
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    NUM_FOLDS, RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct


# ============================================================
# RhythmLIF Neuron
# ============================================================

class RhythmLIF(nn.Module):
    """Leaky Integrate-and-Fire neuron with learnable oscillatory modulation.

    Membrane dynamics:
        v[t] = beta * v[t-1] * (1 - spk[t-1]) + I[t] + A * sin(2*pi*f*t/T + phi)

    When v[t] >= threshold, a spike is emitted and v is reset (soft reset via
    multiplication by (1 - spk), same as snnTorch Leaky default).

    Parameters:
        neuron_shape: Shape of the neuron population (e.g., (32,) for conv channels,
                      (256,) for FC hidden layer). Oscillation params have this shape.
        beta: Initial membrane decay rate (learnable).
        threshold: Spike threshold (fixed at 1.0).
        num_steps: Total simulation timesteps T (for oscillation period).
        spike_grad: Surrogate gradient function for backward pass.
    """

    def __init__(
        self,
        neuron_shape: tuple,
        beta: float = BETA,
        threshold: float = 1.0,
        num_steps: int = NUM_STEPS,
        spike_grad=None,
    ):
        super().__init__()
        self.threshold = threshold
        self.num_steps = num_steps

        if spike_grad is None:
            spike_grad = surrogate.spike_rate_escape(beta=1.0, slope=25)
        self.spike_grad = spike_grad

        # Learnable membrane decay
        # Store as raw parameter, sigmoid it in forward for (0, 1) constraint
        self.beta_raw = nn.Parameter(torch.full(neuron_shape, math.log(beta / (1 - beta))))

        # Learnable oscillation parameters (per-neuron)
        self.amplitude = nn.Parameter(torch.full(neuron_shape, 0.1))
        self.frequency = nn.Parameter(torch.empty(neuron_shape).uniform_(0.5, 5.0))
        self.phase = nn.Parameter(torch.zeros(neuron_shape))

    @property
    def beta(self):
        """Membrane decay constrained to (0, 1) via sigmoid."""
        return torch.sigmoid(self.beta_raw)

    def _oscillation(self, t: int) -> torch.Tensor:
        """Compute oscillatory modulation at timestep t.

        Returns tensor with shape matching neuron_shape, broadcastable
        to the membrane potential tensor.
        """
        angle = 2.0 * math.pi * self.frequency * t / self.num_steps + self.phase
        return self.amplitude * torch.sin(angle)

    def init_rhythm(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize membrane potential to zeros.

        Returns:
            mem: Zeros with shape (batch_size, *neuron_shape).
        """
        return torch.zeros(batch_size, *self.amplitude.shape, device=device)

    def forward(
        self, input_current: torch.Tensor, mem: torch.Tensor, t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single timestep forward.

        Args:
            input_current: Synaptic input, shape (batch, *neuron_shape, ...).
            mem: Previous membrane potential, same shape as input_current.
            t: Current timestep index (0-indexed).

        Returns:
            spk: Output spikes (binary), same shape as mem.
            mem: Updated membrane potential.
        """
        beta = self.beta  # (neuron_shape)
        osc = self._oscillation(t)  # (neuron_shape)

        # Broadcast beta and osc to match input_current dimensions
        # For conv layers: input_current is (batch, channels, H, W), neuron_shape is (channels,)
        # For FC layers: input_current is (batch, hidden), neuron_shape is (hidden,)
        ndim_extra = input_current.dim() - 1 - len(self.amplitude.shape)
        beta_bc = beta
        osc_bc = osc
        for _ in range(ndim_extra):
            beta_bc = beta_bc.unsqueeze(-1)
            osc_bc = osc_bc.unsqueeze(-1)

        # Leaky integration with oscillatory modulation
        mem = beta_bc * mem + input_current + osc_bc

        # Spike generation with surrogate gradient
        spk = self.spike_grad(mem - self.threshold)

        # Soft reset: mem *= (1 - spk)
        mem = mem * (1.0 - spk.detach())

        return spk, mem


# ============================================================
# Rhythm-SNN Model
# ============================================================

class RhythmSpikingCNN(nn.Module):
    """Convolutional SNN with RhythmLIF neurons for ESC-50.

    Same architecture as baseline SpikingCNN but all 4 LIF neurons
    are replaced with RhythmLIF neurons that add learnable oscillatory
    modulation to membrane dynamics. Includes Dropout(0.3) for
    regularisation.

    Architecture:
        Conv2d(1,32) -> BN -> MaxPool(2) -> RhythmLIF
        Conv2d(32,64) -> BN -> MaxPool(2) -> RhythmLIF
        AvgPool(4,6) -> Flatten
        FC(2304,256) -> RhythmLIF -> Dropout(0.3)
