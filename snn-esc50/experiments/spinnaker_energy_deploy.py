"""
spinnaker_energy_deploy.py -- Deploy energy-optimized SNN on SpiNNaker.

Extracts hidden features from trained model, then deploys FC2 on SpiNNaker.

Two phases:
  Phase 1 (--extract-only): Run in .venv to extract features
  Phase 2 (default): Run in .venv-spinnaker for SpiNNaker inference

Usage:
    source .venv/bin/activate
    python experiments/spinnaker_energy_deploy.py --extract-only --fold 4

    source .venv-spinnaker/bin/activate
    python experiments/spinnaker_energy_deploy.py --fold 4 --num-samples 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import RESULTS_DIR, NUM_STEPS, NUM_CLASSES


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def extract_features(fold, model_type, num_samples):
    """Extract hidden spike features from trained model."""
    import torch
    from src.config import BATCH_SIZE, get_device
    from src.dataset import download_esc50, get_fold_dataloaders
    from src.encoding import encode_direct

    device = get_device()
    download_esc50()
    _, test_loader = get_fold_dataloaders(fold, BATCH_SIZE)

    if model_type == "baseline":
        from src.models.snn_model import SpikingCNN
        model = SpikingCNN().to(device)
        path = RESULTS_DIR / "snn" / "direct" / f"best_fold{fold}.pt"
        model.load_state_dict(
            torch.load(path, map_location=device, weights_only=True))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.eval()
    print(f"[{ts()}] Loaded model from {path}")

