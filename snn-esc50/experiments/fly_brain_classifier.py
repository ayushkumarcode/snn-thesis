"""
fly_brain_classifier.py -- Fruit fly olfactory circuit for audio classification.

Dasgupta et al. (Science 2017): Random projection + winner-take-all.
The fly classifies ~50 odors using:
  1. Input (50 projection neurons)
  2. Random sparse projection to 2000 Kenyon cells (40x expansion)
  3. Winner-take-all: keep top 5% active
  4. Trained readout (2000 -> 50)

For our audio: mel spectrogram features -> random projection -> WTA -> linear.
Extremely energy-efficient: random weights are FIXED, only readout is trained.

Usage:
    python -m experiments.fly_brain_classifier --device cuda
    python -m experiments.fly_brain_classifier --fold 1 --expansion 40 --device cuda
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
