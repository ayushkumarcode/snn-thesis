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
