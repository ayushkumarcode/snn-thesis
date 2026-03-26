"""
dendritic_snn.py -- Dendritic Spiking Neurons for ESC-50 classification.

Replaces standard LIF neurons with multi-compartment dendritic neurons.
Each DendriticLIF neuron has K dendritic branches with distinct learnable
time constants (fast/medium/slow), enabling multi-timescale temporal
feature extraction from audio spectrograms.

Branch dynamics:
    mem_branch_k[t] = beta_k * mem_branch_k[t-1] + w_k * input[t]
    mem_soma[t] = sum_k(w_k * mem_branch_k[t])
    spike if mem_soma > threshold, then reset all branches

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/dendritic_snn.py --fold 1 --device mps
    python experiments/dendritic_snn.py  # all 5 folds
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
