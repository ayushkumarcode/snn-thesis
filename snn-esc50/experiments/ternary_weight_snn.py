"""
ternary_weight_snn.py -- SNN with ternary weights {-1, 0, +1}.

Each synapse is just add/subtract (no multiply). Uses STE (straight-
through estimator) for gradients through quantization. Massive energy
savings: ternary multiply = 1 cycle vs float multiply = 3-5 cycles.

Usage:
    python -m experiments.ternary_weight_snn --device cuda
    python -m experiments.ternary_weight_snn --fold 1 --device cuda
"""

import argparse
import json
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
    NUM_CLASSES, NUM_STEPS, BETA, RESULTS_DIR, BATCH_SIZE,
