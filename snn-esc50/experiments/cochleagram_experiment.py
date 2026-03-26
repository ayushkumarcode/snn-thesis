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
