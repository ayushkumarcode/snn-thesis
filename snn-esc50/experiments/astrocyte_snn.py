"""
astrocyte_snn.py -- Astrocyte-Augmented SNN experiment.

Slow glial modulation of neural excitability for homeostatic regulation.
Based on "Astrocyte-gated multi-timescale plasticity" (PMC, 2025).

Each LIF layer is wrapped with an "astrocyte" module that:
  1. Monitors average spike rate via exponential moving average
     (tau_astro ~ 100 timesteps, much slower than neural dynamics)
  2. Modulates the effective threshold based on activity level:
     - High activity -> raise threshold (inhibit runaway firing)
     - Low activity  -> lower threshold (rescue dead neurons)
  3. Creates homeostatic regulation that prevents both over-firing
     and under-firing -- particularly useful for small datasets
     where training signal is sparse.

Implementation:
  - AstrocyteLIF wraps snn.Leaky and adds threshold modulation
  - astro_state: EMA of population spike rate (per-channel/neuron)
  - Effective threshold = base_threshold * (1 + gain * (astro_state - target_rate))
  - tau_astro and gain are learnable; target_rate is learnable (init 0.1)

Also uses: learn_beta=True, Dropout(0.3), spike_rate_escape surrogate.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/astrocyte_snn.py --fold 1
    python experiments/astrocyte_snn.py              # all 5 folds
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_CLASSES, NUM_STEPS, BETA,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct


# ============================================================
# Astrocyte LIF Module
# ============================================================
