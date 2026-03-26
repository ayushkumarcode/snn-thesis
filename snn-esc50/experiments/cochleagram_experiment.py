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


