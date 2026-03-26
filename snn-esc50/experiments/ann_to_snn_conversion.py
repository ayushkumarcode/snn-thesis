"""
ann_to_snn_conversion.py -- ANN-to-SNN conversion with threshold calibration.

Based on Bojkovic AISTATS 2024 and Rathi et al. ICLR 2020:
  1. Load trained ANN from results/ann/none/best_fold{fold}.pt
  2. Run all training data through ANN, record max activation per layer
  3. Set SNN threshold per layer = percentile of max activations
  4. Convert: replace ReLU with IF neurons (beta=1.0, no leak)
  5. Evaluate converted SNN at different timestep budgets: T=1,4,8,16,25,50,100
  6. Report accuracy-vs-T curve

Key insight: IF neurons (no leak, beta=1.0) are used because in the conversion
framework, ReLU activations map to spike rates, and leakage would distort
the mapping. snn.Leaky(beta=1.0) gives a perfect integrator.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/ann_to_snn_conversion.py --fold 1
    python experiments/ann_to_snn_conversion.py --percentile 99.5 --max-timesteps 200
    python experiments/ann_to_snn_conversion.py                    # all 5 folds
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    BATCH_SIZE, NUM_FOLDS, RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.models.ann_model import ConvANN


# ============================================================
# Activation Recording Hook
# ============================================================

class ActivationRecorder:
    """Records max activations at specified layers during ANN forward pass.

    Attaches hooks to the 4 pre-activation points in ConvANN:
      - After conv1+bn1 (before ReLU)
      - After conv2+bn2 (before ReLU)
      - After fc1 (before ReLU)
      - After fc2 (output, before softmax)
    """

    def __init__(self):
        self.max_activations = {}
        self.all_activations = {}
        self.hooks = []

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            with torch.no_grad():
                vals = output.detach().cpu()
                if name not in self.all_activations:
                    self.all_activations[name] = []
                self.all_activations[name].append(vals)
        return hook_fn

    def register_hooks(self, model: ConvANN):
        """Register forward hooks on the 4 key layers of ConvANN.

        ConvANN.features: [Conv2d, BN, ReLU, MaxPool, Conv2d, BN, ReLU, MaxPool, AvgPool]
                           0       1    2     3        4       5    6     7        8
        ConvANN.classifier: [Linear, ReLU, Dropout, Linear]
                             0       1      2        3
        """
        features = model.features
        classifier = model.classifier

        # Hook after BN1 (index 1) -- pre-ReLU for conv block 1
        h1 = features[1].register_forward_hook(self._make_hook("conv1_bn"))
        # Hook after BN2 (index 5) -- pre-ReLU for conv block 2
        h2 = features[5].register_forward_hook(self._make_hook("conv2_bn"))
        # Hook after FC1 (index 0) -- pre-ReLU for FC block 1
        h3 = classifier[0].register_forward_hook(self._make_hook("fc1"))
        # Hook after FC2 (index 3) -- output logits
        h4 = classifier[3].register_forward_hook(self._make_hook("fc2"))

        self.hooks = [h1, h2, h3, h4]

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def compute_thresholds(self, percentile: float = 99.9) -> dict[str, float]:
        """Compute per-layer thresholds from recorded activations.

        For each layer, compute the given percentile of the maximum activation
        values across the entire dataset. This determines the SNN threshold:
        any ANN activation at the percentile maps to a spike rate of 1.0.

        Args:
            percentile: Percentile of activations to use as threshold (0-100).

        Returns:
            Dict mapping layer name to threshold value.
        """
        thresholds = {}
        for name, act_list in self.all_activations.items():
            # Concatenate all batches: (total_samples, ...)
            all_acts = torch.cat(act_list, dim=0)
            # ReLU clips negatives, so only consider positive activations
            positive = torch.clamp(all_acts, min=0.0)
            # Compute percentile across all values
            threshold = float(np.percentile(positive.numpy().flatten(), percentile))
            thresholds[name] = max(threshold, 1e-6)  # avoid zero threshold
        return thresholds


# ============================================================
# Converted SNN Model
# ============================================================

class ConvertedSNN(nn.Module):
    """ANN-to-SNN converted model with calibrated thresholds.

    Uses IF neurons (beta=1.0, no leak) which act as perfect integrators.
    The threshold at each layer is calibrated from ANN activation statistics
    so that the ANN-to-SNN rate mapping is preserved.

    The ANN weights are directly transferred without modification.
    The SNN accumulates input over T timesteps, and the fire rate at
    each layer approximates the normalized ReLU activation.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        thresholds: dict[str, float] = None,
    ):
        super().__init__()
        self.thresholds = thresholds or {}

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        thresh1 = self.thresholds.get("conv1_bn", 1.0)
        self.if1 = snn.Leaky(beta=1.0, threshold=thresh1, spike_grad=spike_grad,
                              learn_beta=False, learn_threshold=False)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        thresh2 = self.thresholds.get("conv2_bn", 1.0)
        self.if2 = snn.Leaky(beta=1.0, threshold=thresh2, spike_grad=spike_grad,
                              learn_beta=False, learn_threshold=False)

        # Pooling
