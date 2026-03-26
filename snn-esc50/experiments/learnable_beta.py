"""
Experiment 9: Learnable beta standalone ablation.

ONLY change from baseline: learn_beta=True on all LIF neurons.
Everything else identical to baseline SpikingCNN with fast_sigmoid.
This isolates the effect of heterogeneous membrane dynamics.

Usage:
    python -m experiments.learnable_beta              # all 5 folds
    python -m experiments.learnable_beta --fold 1     # single fold
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
