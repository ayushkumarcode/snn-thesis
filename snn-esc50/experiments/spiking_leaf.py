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
