"""
spike_budget_snn.py -- Energy-aware spike budgeting with feedback controller.

Based on arXiv 2602.12236: Adaptive spike scheduler enforces dataset-specific
energy constraints during training. Uses proportional feedback controller with
learnable LIF parameters (beta, threshold) to auto-tune for target spike budget.

Key insight: Can reduce spikes by 47% while IMPROVING accuracy on frame-based
datasets. The controller adjusts threshold dynamically to meet a target spike
rate per layer.

Usage:
    python -m experiments.spike_budget_snn --target-rate 0.06 --device cuda
    python -m experiments.spike_budget_snn --device cuda  # sweep targets
"""

import argparse
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
