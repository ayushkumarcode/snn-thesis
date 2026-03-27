"""
extreme_pruning.py -- Iterative magnitude pruning to 95%/99% sparsity.

We showed 90% pruning retains 93.2% of accuracy. Push further with
iterative pruning + fine-tuning between rounds on the rhythm model.

Usage:
    python -m experiments.extreme_pruning --device cuda
    python -m experiments.extreme_pruning --fold 1 --device cuda
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
import torch.nn.utils.prune as prune

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    NUM_CLASSES, NUM_STEPS, RESULTS_DIR, BATCH_SIZE,
    LEARNING_RATE, WEIGHT_DECAY, get_device,
)
