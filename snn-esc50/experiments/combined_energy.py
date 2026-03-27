"""
combined_energy.py -- Combined Tier 1 energy reduction.

Stacks: early exit + layerwise spike regularization + silence gating.
Projected ~20x energy reduction over baseline SNN.

This script trains a spike-regularized rhythm SNN, then evaluates it
with early exit and silence gating applied at inference time.

Usage:
    python -m experiments.combined_energy --device cuda
    python -m experiments.combined_energy --fold 1 --device cuda
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
