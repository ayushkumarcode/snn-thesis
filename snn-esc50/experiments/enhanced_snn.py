"""
Experiment 1: Enhanced SNN with quick fixes.

Changes from baseline SpikingCNN:
  - Dropout(0.3) before fc2 (ANN has this, SNN doesn't)
  - learn_beta=True on all LIF neurons (heterogeneous membrane dynamics)
  - learn_threshold=True on all LIF neurons
  - spike_rate_escape surrogate (proven best in our ablation: 46.00% vs 44.75%)

Usage:
    python -m experiments.enhanced_snn                  # all 5 folds
    python -m experiments.enhanced_snn --fold 1         # single fold
    python -m experiments.enhanced_snn --device cuda    # specify device
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
