"""
tet_training.py -- Temporal Efficient Training (TET) loss for SNNs.

Implements the TET loss from Deng et al. (ICLR 2022):
    L_TET = (1/T) * sum_t CE(mem_t, y) + lambda * (1/T) * sum_t (CE(mem_t, y) - L_mean)^2

The first term is the standard per-timestep CE loss (same as baseline).
The second term penalises variance across timesteps, encouraging the SNN
to make consistent predictions at EVERY timestep, not just at the end.
This flattens the temporal loss landscape and improves gradient flow.

Model uses learn_beta=True, learn_threshold=True, spike_rate_escape surrogate.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/tet_training.py --fold 1 --device mps
    python experiments/tet_training.py                      # all 5 folds
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
