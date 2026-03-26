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
