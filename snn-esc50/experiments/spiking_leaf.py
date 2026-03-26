"""
spiking_leaf.py -- Spiking-LEAF: Learnable auditory frontend on raw waveforms.

Based on Song et al. ICASSP 2024: learnable Gabor filterbank + PCEN frontend
that processes raw audio waveforms instead of precomputed mel spectrograms.
The filterbank is jointly trained end-to-end with the SNN/ANN classifier.

Architecture:
  1. Raw waveform (batch, 1, 110250)
  2. Gabor filterbank: 64 1D convolutions, log-spaced 50-11025 Hz, kernel=401
  3. Per-Channel Energy Normalization (PCEN): learnable alpha, delta, r, s
  4. Output: (batch, 64, ~216) -- same shape as mel spectrograms
  5. SpikingCNN / ConvANN backbone on top

For SNN: encode_direct on (batch, 1, 64, ~216) then feed to SNN backbone.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/spiking_leaf.py --fold 1
    python experiments/spiking_leaf.py --model snn --fold 1 --device mps
    python experiments/spiking_leaf.py --model both       # all 5 folds, SNN+ANN
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    NUM_FOLDS, RESULTS_DIR, SAMPLE_RATE, DURATION, HOP_LENGTH,
    ESC50_AUDIO_DIR, ESC50_META_PATH, get_device,
)
from src.dataset import download_esc50


# ============================================================
# Raw Waveform Dataset
# ============================================================

WAVEFORM_LENGTH = SAMPLE_RATE * DURATION  # 110250 samples


class ESC50WaveformDataset(Dataset):
    """ESC-50 dataset that returns raw audio waveforms instead of spectrograms.

    Each sample is a raw waveform of shape (1, 110250), normalized to [-1, 1].
    """

    def __init__(self, folds: list[int], precompute: bool = True):
        self.precompute = precompute
        meta = pd.read_csv(ESC50_META_PATH)
        self.meta = meta[meta["fold"].isin(folds)].reset_index(drop=True)

        self.data = []
        self.labels = []

        if precompute:
            print(f"Loading {len(self.meta)} raw waveforms from folds {folds}...")
            for _, row in self.meta.iterrows():
                filepath = ESC50_AUDIO_DIR / row["filename"]
                waveform = self._load_waveform(str(filepath))
                self.data.append(waveform)
                self.labels.append(row["target"])
            self.data = np.array(self.data, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)

    @staticmethod
    def _load_waveform(filepath: str) -> np.ndarray:
        """Load raw audio, pad to WAVEFORM_LENGTH, normalize to [-1, 1]."""
        y, _ = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
        if len(y) < WAVEFORM_LENGTH:
            y = np.pad(y, (0, WAVEFORM_LENGTH - len(y)))
        elif len(y) > WAVEFORM_LENGTH:
            y = y[:WAVEFORM_LENGTH]
        # Normalize to [-1, 1]
        max_val = np.abs(y).max()
        if max_val > 0:
            y = y / max_val
        return y[np.newaxis, :]  # (1, 110250)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.precompute:
            waveform = self.data[idx]
            label = self.labels[idx]
        else:
            row = self.meta.iloc[idx]
            filepath = ESC50_AUDIO_DIR / row["filename"]
            waveform = self._load_waveform(str(filepath))
            label = row["target"]
        return (
            torch.tensor(waveform, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def get_waveform_dataloaders(test_fold: int, batch_size: int = BATCH_SIZE):
    """Create train/test DataLoaders for raw waveform data."""
    train_folds = [f for f in range(1, NUM_FOLDS + 1) if f != test_fold]
    train_dataset = ESC50WaveformDataset(folds=train_folds)
    test_dataset = ESC50WaveformDataset(folds=[test_fold])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


# ============================================================
# Gabor Filterbank
# ============================================================

class GaborFilterbank(nn.Module):
    """Learnable 1D Gabor filterbank for audio processing.

    Initializes 64 1D convolution filters as Gabor wavelets with
    log-spaced center frequencies from 50 Hz to 11025 Hz.
    Center frequencies and bandwidths are learnable parameters.

    Args:
        n_filters: Number of filters (default: 64 to match N_MELS).
        kernel_size: Filter length in samples (401 ~ 18ms at 22050 Hz).
        stride: Stride for convolution (HOP_LENGTH=512 to get ~216 frames).
        sample_rate: Audio sample rate.
        f_min: Minimum center frequency (Hz).
        f_max: Maximum center frequency (Hz).
    """

    def __init__(
        self,
        n_filters: int = 64,
        kernel_size: int = 401,
        stride: int = HOP_LENGTH,
        sample_rate: int = SAMPLE_RATE,
        f_min: float = 50.0,
        f_max: float = 11025.0,
    ):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.sample_rate = sample_rate

        # Learnable center frequencies (in Hz, log-spaced initialization)
        center_freqs = torch.logspace(
            math.log10(f_min), math.log10(f_max), n_filters
        )
        # Store as normalized angular frequency: omega = 2*pi*f / sr
        self.center_freq = nn.Parameter(
            2.0 * math.pi * center_freqs / sample_rate
        )

        # Learnable bandwidth (sigma of Gaussian envelope)
        # Initialize proportional to center frequency (constant Q)
        init_sigma = sample_rate / (2.0 * math.pi * center_freqs) * 2.0
        init_sigma = init_sigma.clamp(min=2.0, max=kernel_size / 2.0)
        self.log_sigma = nn.Parameter(torch.log(init_sigma))

        # Time axis for kernel computation (centered at 0)
        t = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, dtype=torch.float32)
        self.register_buffer("t", t)

    def _build_kernels(self) -> torch.Tensor:
        """Build Gabor wavelet kernels from learnable parameters.

        Returns:
            kernels: Shape (n_filters, 1, kernel_size).
        """
        sigma = torch.exp(self.log_sigma).unsqueeze(1)  # (n_filters, 1)
        omega = self.center_freq.unsqueeze(1)  # (n_filters, 1)
        t = self.t.unsqueeze(0)  # (1, kernel_size)

        # Gaussian envelope
        gaussian = torch.exp(-0.5 * (t / sigma) ** 2)
        # Cosine carrier
        carrier = torch.cos(omega * t)
        # Gabor wavelet = Gaussian * cosine
        kernels = gaussian * carrier

        # Normalize each filter to unit energy
        norm = kernels.norm(dim=1, keepdim=True).clamp(min=1e-8)
        kernels = kernels / norm

        return kernels.unsqueeze(1)  # (n_filters, 1, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gabor filterbank to raw waveform.

        Args:
            x: Raw waveform, shape (batch, 1, waveform_length).

        Returns:
            Filtered output, shape (batch, n_filters, time_frames).
        """
        kernels = self._build_kernels()
        # Pad input for 'same'-like convolution at the given stride
        pad = self.kernel_size // 2
        out = F.conv1d(x, kernels, stride=self.stride, padding=pad)
        # Take absolute value (energy envelope)
        out = torch.abs(out)
        return out


# ============================================================
# PCEN (Per-Channel Energy Normalization)
# ============================================================

class PCEN(nn.Module):
    """Learnable Per-Channel Energy Normalization.

    PCEN(x) = (x / (eps + M)^alpha + delta)^r - delta^r
    where M is a smoothed version of x using a learnable IIR filter.

    All parameters (alpha, delta, r, s) are learnable per channel.

    Args:
        n_channels: Number of frequency channels.
        eps: Small constant to prevent division by zero.
        init_s: Initial smoothing coefficient (0 < s < 1).
        init_alpha: Initial compression exponent.
        init_delta: Initial offset.
        init_r: Initial root compression exponent.
    """

    def __init__(
        self,
        n_channels: int = 64,
        eps: float = 1e-6,
        init_s: float = 0.025,
        init_alpha: float = 0.98,
        init_delta: float = 2.0,
        init_r: float = 0.5,
    ):
        super().__init__()
        self.eps = eps

        # Learnable per-channel parameters (stored in log/logit space for constraints)
        self.log_s = nn.Parameter(torch.full((n_channels,), math.log(init_s / (1 - init_s))))
        self.log_alpha = nn.Parameter(torch.full((n_channels,), math.log(init_alpha)))
        self.log_delta = nn.Parameter(torch.full((n_channels,), math.log(init_delta)))
        self.log_r = nn.Parameter(torch.full((n_channels,), math.log(init_r)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PCEN normalization.

        Args:
            x: Input of shape (batch, n_channels, time_frames).

        Returns:
            Normalized output, same shape as input.
        """
        # Constrain parameters
        s = torch.sigmoid(self.log_s).unsqueeze(0).unsqueeze(-1)  # (1, C, 1)
        alpha = torch.exp(self.log_alpha).unsqueeze(0).unsqueeze(-1)
        delta = torch.exp(self.log_delta).unsqueeze(0).unsqueeze(-1)
        r = torch.exp(self.log_r).unsqueeze(0).unsqueeze(-1)

        # IIR smoothing: M[t] = (1-s) * M[t-1] + s * x[t]
        batch, channels, time_frames = x.shape
        M = torch.zeros(batch, channels, 1, device=x.device, dtype=x.dtype)
        M_list = []
        for t in range(time_frames):
            M = (1.0 - s) * M + s * x[:, :, t:t+1]
            M_list.append(M)
        M_full = torch.cat(M_list, dim=2)  # (batch, channels, time_frames)

        # PCEN formula
        smooth = (x / (self.eps + M_full).pow(alpha) + delta).pow(r) - delta.pow(r)
        return smooth


# ============================================================
# Spiking-LEAF Frontend
# ============================================================

class SpikingLEAF(nn.Module):
    """Learnable auditory frontend: Gabor filterbank + PCEN.

    Converts raw waveform to a (batch, 1, 64, ~216) representation
    that can be fed into the standard SpikingCNN / ConvANN backbone.
    """

    def __init__(
        self,
        n_filters: int = 64,
        kernel_size: int = 401,
        stride: int = HOP_LENGTH,
    ):
        super().__init__()
        self.gabor = GaborFilterbank(
            n_filters=n_filters, kernel_size=kernel_size, stride=stride,
        )
        self.pcen = PCEN(n_channels=n_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process raw waveform through learnable frontend.

        Args:
            x: Raw waveform, shape (batch, 1, waveform_length).

        Returns:
            Pseudo-spectrogram, shape (batch, 1, n_filters, time_frames).
        """
        # Gabor filterbank: (batch, 1, L) -> (batch, 64, T)
        filtered = self.gabor(x)
        # PCEN normalization
        normalized = self.pcen(filtered)
        # Add channel dim to match spectrogram format: (batch, 1, 64, T)
        return normalized.unsqueeze(1)


# ============================================================
# Spiking-LEAF SNN Model
# ============================================================

class SpikingLEAF_SNN(nn.Module):
    """End-to-end learnable audio SNN: Gabor + PCEN + SpikingCNN.

    The frontend and classifier are jointly optimized.
    Uses learn_beta=True, spike_rate_escape, and Dropout(0.3).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        beta: float = BETA,
        num_steps: int = NUM_STEPS,
    ):
        super().__init__()
        self.num_steps = num_steps

        # Learnable frontend
        self.frontend = SpikingLEAF()

        spike_grad = surrogate.spike_rate_escape(beta=1.0, slope=25)

        # SNN backbone (same as SpikingCNN)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Raw waveform, shape (batch, 1, waveform_length).

        Returns:
            spk_out: (num_steps, batch, num_classes)
            mem_out: (num_steps, batch, num_classes)
        """
        # Frontend: raw waveform -> pseudo-spectrogram (batch, 1, 64, T)
        features = self.frontend(x)

        # Truncate or pad time dimension to match expected 216 frames
        T_expected = 216
        T_actual = features.shape[3]
        if T_actual > T_expected:
            features = features[:, :, :, :T_expected]
        elif T_actual < T_expected:
            pad_size = T_expected - T_actual
            features = F.pad(features, (0, pad_size))

        # Direct encoding: repeat across timesteps
        # (num_steps, batch, 1, 64, 216)
        spike_input = features.unsqueeze(0).repeat(self.num_steps, *([1] * features.dim()))

        # Initialize membrane potentials
        device = features.device
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_out_rec = []
        mem_out_rec = []

        for step in range(self.num_steps):
            x_t = spike_input[step]

            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            cur3 = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3 = self.dropout(spk3)

            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)

        return torch.stack(spk_out_rec), torch.stack(mem_out_rec)


# ============================================================
# Spiking-LEAF ANN Model
# ============================================================

class SpikingLEAF_ANN(nn.Module):
    """Learnable audio ANN: Gabor + PCEN + ConvANN backbone.

    Same frontend, but ReLU-based classifier for comparison.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # Same learnable frontend
        self.frontend = SpikingLEAF()

        # ANN backbone (same as ConvANN)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AvgPool2d(kernel_size=(4, 6)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Raw waveform, shape (batch, 1, waveform_length).

        Returns:
            logits: (batch, num_classes)
        """
        features = self.frontend(x)

        # Truncate/pad time dimension
        T_expected = 216
        T_actual = features.shape[3]
        if T_actual > T_expected:
            features = features[:, :, :, :T_expected]
        elif T_actual < T_expected:
            pad_size = T_expected - T_actual
            features = F.pad(features, (0, pad_size))

        h = self.features(features)
        flat = h.view(h.size(0), -1)
        return self.classifier(flat)


# ============================================================
# Training and Evaluation
# ============================================================

def train_snn_epoch(model, loader, optimizer, device):
    """Train one SNN epoch with per-timestep CE on membrane potentials."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
