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

    all_data, all_labels = [], []
    for data, targets in test_loader:
        all_data.append(data)
        all_labels.append(targets)
    all_data = torch.cat(all_data, dim=0)[:num_samples].to(device)
    all_labels = torch.cat(all_labels, dim=0)[:num_samples]

    hidden_features = []
    predictions = []

    with torch.no_grad():
        for i in range(0, num_samples, BATCH_SIZE):
            batch = all_data[i:i+BATCH_SIZE]
            B = batch.shape[0]
            spk_input = encode_direct(batch).to(device)

            mem1 = model.lif1.init_leaky()
            mem2 = model.lif2.init_leaky()
            mem3 = model.lif3.init_leaky()
            mem4 = model.lif4.init_leaky()

            hidden_spk = []
            mem_out = []
            for step in range(NUM_STEPS):
                x_t = spk_input[step]
                cur1 = model.pool1(model.bn1(model.conv1(x_t)))
                spk1, mem1 = model.lif1(cur1, mem1)
                cur2 = model.pool2(model.bn2(model.conv2(spk1)))
