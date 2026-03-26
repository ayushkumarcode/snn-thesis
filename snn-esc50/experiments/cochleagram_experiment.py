"""
cochleagram_experiment.py -- Replace mel spectrogram with gammatone filterbank cochleagram.

Computes a "gammatonegram": STFT magnitude filtered through a 64-channel gammatone
filterbank (ERB-spaced from 50 Hz to Nyquist). This is a biologically-motivated
alternative to the mel spectrogram, modelling the basilar membrane frequency response
more accurately (asymmetric filters, level-dependent bandwidth).

Hypothesis: SNNs may benefit more from cochleagram input than ANNs, since cochleagram
better matches the auditory periphery that spiking neurons evolved to process.

Both SNN (EnhancedSpikingCNN with learn_beta, learn_threshold, SRE, dropout) and
ANN (ConvANN) are trained on cochleagrams for comparison against mel baselines.

Output shape is (1, 64, 216) -- same as mel spectrogram -- so existing model
architectures work without modification.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/cochleagram_experiment.py                    # both models, all 5 folds
    python experiments/cochleagram_experiment.py --fold 1           # single fold
    python experiments/cochleagram_experiment.py --model snn        # SNN only
    python experiments/cochleagram_experiment.py --model ann        # ANN only
    python experiments/cochleagram_experiment.py --device cuda       # specify device
"""

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import Dataset, DataLoader
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_CLASSES, NUM_STEPS, N_MELS, BETA,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    SAMPLE_RATE, DURATION, N_FFT, HOP_LENGTH,
    RESULTS_DIR, get_device,
    ESC50_AUDIO_DIR, ESC50_META_PATH, NUM_FOLDS,
)
from src.dataset import download_esc50
from src.encoding import encode_direct


# ============================================================
# Gammatone filterbank implementation
# ============================================================

def erb(fc):
    """Equivalent Rectangular Bandwidth (Glasberg & Moore, 1990).

    ERB(fc) = 24.7 * (4.37 * fc/1000 + 1)

    Args:
        fc: Centre frequency in Hz (scalar or array).

    Returns:
        ERB bandwidth in Hz.
    """
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def gammatone_filterbank(sr, n_fft, n_filters=64, f_min=50.0, f_max=None):
    """Create a gammatone filterbank matrix for applying to STFT magnitude.

    Each row is one gammatone filter's frequency response, evaluated at the
    STFT frequency bins. The filters are 4th-order gammatone with ERB-spaced
    centre frequencies.

    The gammatone frequency response magnitude for a 4th-order filter centred
    at fc with bandwidth b is:
        |H(f)| = 1 / (1 + ((f - fc) / b)^2)^2

    This is the magnitude-squared of the transfer function of the cascade of
    4 first-order bandpass filters, which is a good approximation of the
    cochlear filter shape (Patterson et al., 1992).

    Args:
        sr: Sample rate in Hz.
        n_fft: FFT size.
        n_filters: Number of filters (channels). Default 64.
        f_min: Lowest centre frequency in Hz. Default 50.
        f_max: Highest centre frequency in Hz. Default sr/2.

    Returns:
        filterbank: np.ndarray of shape (n_filters, 1 + n_fft // 2).
    """
    if f_max is None:
        f_max = sr / 2.0

    # ERB-rate scale: convert Hz to ERB-rate, space linearly, convert back
    # ERB-rate(f) = 21.4 * log10(4.37 * f/1000 + 1)  (Glasberg & Moore)
    erb_lo = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_hi = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    erb_points = np.linspace(erb_lo, erb_hi, n_filters)

    # Convert back to Hz
    center_freqs = (10.0 ** (erb_points / 21.4) - 1.0) * 1000.0 / 4.37

    # STFT frequency bins
    n_freqs = 1 + n_fft // 2
    freqs = np.linspace(0, sr / 2.0, n_freqs)

    filterbank = np.zeros((n_filters, n_freqs), dtype=np.float32)

    for i, fc in enumerate(center_freqs):
        # Bandwidth parameter: 1.019 * ERB(fc) -- standard gammatone
        b = 1.019 * erb(fc)

        # 4th-order gammatone magnitude response
        # |H(f)|^2 = 1 / (1 + ((f - fc) / b)^2)^4
        # We use the amplitude (not power) so take sqrt:
        # |H(f)| = 1 / (1 + ((f - fc) / b)^2)^2
        filterbank[i] = 1.0 / (1.0 + ((freqs - fc) / b) ** 2) ** 2

    # Normalise each filter to have unit peak (preserves relative energy)
    for i in range(n_filters):
        peak = filterbank[i].max()
        if peak > 0:
            filterbank[i] /= peak

    return filterbank, center_freqs


def wav_to_cochleagram(filepath, sr=SAMPLE_RATE, duration=DURATION,
                       n_fft=N_FFT, hop_length=HOP_LENGTH,
                       n_filters=64, f_min=50.0, f_max=None):
    """Load audio and compute gammatone cochleagram.

    Pipeline:
      1. Load and pad audio to exactly `duration` seconds
      2. Compute STFT magnitude
      3. Apply gammatone filterbank (matrix multiply)
      4. Convert to log scale (dB)
      5. Output shape: (n_filters, time_frames)

    The time_frames dimension matches mel spectrogram exactly because
    we use the same n_fft and hop_length.

    Args:
        filepath: Path to WAV file.
        sr: Sample rate.
        duration: Audio duration in seconds.
        n_fft: FFT window size.
        hop_length: STFT hop length.
        n_filters: Number of gammatone channels.
        f_min: Lowest centre frequency.
        f_max: Highest centre frequency (default: Nyquist).

    Returns:
        cochleagram: np.ndarray of shape (n_filters, time_frames).
    """
    # Load audio
    y, _ = librosa.load(filepath, sr=sr, duration=duration)

    # Pad to exactly duration seconds if shorter
    expected_len = sr * duration
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))

    # STFT magnitude (same windowing as mel spectrogram)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # S shape: (1 + n_fft//2, time_frames) = (513, 216)

    # Build gammatone filterbank
    fb, center_freqs = gammatone_filterbank(
        sr=sr, n_fft=n_fft, n_filters=n_filters,
        f_min=f_min, f_max=f_max,
    )
    # fb shape: (n_filters, 513)

    # Apply filterbank: cochleagram = fb @ S
    # Result shape: (n_filters, time_frames) = (64, 216)
    cochleagram = fb @ S

    # Convert to log scale (power to dB, same as mel pipeline)
    cochleagram_db = librosa.power_to_db(cochleagram ** 2, ref=np.max)

    return cochleagram_db


def normalise_spectrogram(spec):
    """Min-max normalise to [0, 1]."""
    min_val = spec.min()
