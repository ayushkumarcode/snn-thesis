"""
hybrid_ann_snn.py -- ANN-to-SNN hybrid initialization with fine-tuning.

Research question: Can transferring learned ANN weights to an enhanced SNN
(with learnable beta, threshold, dropout, and spike_rate_escape surrogate)
close the ANN-SNN accuracy gap?

Method:
  - Load trained ANN weights from results/ann/none/best_fold{fold}.pt
  - Map ANN conv/bn/fc layers to corresponding SNN layers
  - Create EnhancedSpikingCNN with learn_beta=True, learn_threshold=True,
    Dropout(0.3), spike_rate_escape surrogate gradient
  - Fine-tune for 20 epochs at lower learning rate (1e-4)
  - 5-fold CV, save results to results/experiments/hybrid_ann_snn/

Weight mapping (ConvANN -> SpikingCNN):
  features.0  (Conv2d)     -> conv1
  features.1  (BatchNorm)  -> bn1
  features.4  (Conv2d)     -> conv2
  features.5  (BatchNorm)  -> bn2
  classifier.0 (Linear)    -> fc1
  classifier.3 (Linear)    -> fc2

Usage:
  cd snn-esc50
  source .venv/bin/activate
  python experiments/hybrid_ann_snn.py
  python experiments/hybrid_ann_snn.py --fold 1 --epochs 20
  python experiments/hybrid_ann_snn.py --fold 0  # run all 5 folds
"""

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from src.config import (
    NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    BATCH_SIZE, WEIGHT_DECAY, PATIENCE, NUM_FOLDS,
    RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct
from src.models.ann_model import ConvANN


# ============================================================
# Enhanced SNN with learnable beta, threshold, and dropout
# ============================================================

class EnhancedSpikingCNN(nn.Module):
    """SpikingCNN with learnable membrane decay (beta), learnable threshold,
    dropout regularization, and spike_rate_escape surrogate gradient.

    Designed for ANN->SNN weight transfer: same conv/bn/fc dimensions as
    ConvANN, but with LIF neurons and temporal dynamics.

    Args:
        num_classes: Number of output classes.
        beta: Initial membrane potential decay rate.
        num_steps: Number of simulation timesteps.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        beta: float = BETA,
        num_steps: int = NUM_STEPS,
    ):
        super().__init__()
        self.num_steps = num_steps

        spike_grad = surrogate.spike_rate_escape(beta=1.0, slope=25)

