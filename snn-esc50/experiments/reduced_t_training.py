"""
reduced_t_training.py -- Train rhythm SNN from scratch at low timesteps.

Instead of truncating a T=25 model, actually TRAIN at T=3,5,7,10,15.
This should give much better accuracy at low T since the model learns
to use fewer timesteps optimally.

Usage:
    python -m experiments.reduced_t_training --num-steps 5 --device cuda
    python -m experiments.reduced_t_training --device cuda  # sweep all T values
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
    NUM_CLASSES, BETA, RESULTS_DIR, BATCH_SIZE,
