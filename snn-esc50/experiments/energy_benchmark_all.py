"""
energy_benchmark_all.py -- NeuroBench energy analysis on ALL models.

Runs NeuroBench metrics (SynapticOperations, ActivationSparsity) on
every trained model and computes energy estimates for each.

Usage:
    python -m experiments.energy_benchmark_all --device cuda
    python -m experiments.energy_benchmark_all --fold 1 --device cuda
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    NUM_CLASSES, NUM_STEPS, RESULTS_DIR, BATCH_SIZE, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct
