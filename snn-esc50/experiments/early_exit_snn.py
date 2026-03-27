"""
early_exit_snn.py -- Confidence-based early exit for energy reduction.

Tests how many timesteps are actually needed per sample by checking
output confidence at each timestep. Easy samples exit early (T=3-5),
hard samples use full T=25. Directly reduces energy proportionally.

Usage:
    python -m experiments.early_exit_snn --device cuda
    python -m experiments.early_exit_snn --model-type rhythm --device cuda
    python -m experiments.early_exit_snn --fold 1 --device cuda
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
