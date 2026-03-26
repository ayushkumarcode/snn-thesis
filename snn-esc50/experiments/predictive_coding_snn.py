"""
predictive_coding_snn.py -- Predictive Coding SNN experiment.

Suppress predictable spikes and transmit only prediction errors.
Based on "Predictive Coding Light" (Nature Communications, 2025).

Architecture: Same conv backbone, but after the hidden FC layer (lif3),
a prediction module tries to predict the *next* timestep's hidden spikes
from the *current* timestep's spikes. Only the prediction ERROR (residual)
is transmitted to FC2. This naturally reduces redundant spikes and acts
as a temporal regulariser on the small ESC-50 dataset.

Implementation details:
  - predict_fc = Linear(256, 256) predicts spk3[t] from spk3[t-1]
  - error[t] = spk3[t] - predict_fc(spk3[t-1])  (for t >= 1)
  - error[0] = spk3[0]  (no prediction available at first step)
  - FC2 receives error instead of spk3
  - Loss = CE_loss + lambda_pred * MSE(predict_fc(spk3[t-1]), spk3[t])
  - The MSE term encourages the network to produce predictable hidden
    representations, compressing redundancy across timesteps.

Also uses: learn_beta=True, Dropout(0.3), spike_rate_escape surrogate.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/predictive_coding_snn.py --fold 1
    python experiments/predictive_coding_snn.py              # all 5 folds
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
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct


# ============================================================
# Predictive Coding SNN Model
# ============================================================

class PredictiveSNN(nn.Module):
    """SpikingCNN with predictive coding on the hidden FC layer.

    After lif3 produces hidden spikes (256-dim), a prediction module
    estimates spk3[t] from spk3[t-1]. Only the prediction error is
    sent to FC2. The prediction loss encourages temporal redundancy
    reduction -- fewer unique spikes need to be transmitted.
    """

    def __init__(self, num_classes=NUM_CLASSES, beta=BETA, num_steps=NUM_STEPS):
        super().__init__()
        self.num_steps = num_steps

        spike_grad = surrogate.spike_rate_escape(beta=1, slope=25)

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))
