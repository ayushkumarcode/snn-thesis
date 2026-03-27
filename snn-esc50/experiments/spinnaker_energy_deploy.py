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
                spk2, mem2 = model.lif2(cur2, mem2)
                pooled = model.avg_pool(spk2)
                flat = pooled.view(B, -1)
                cur3 = model.fc1(flat)
                spk3, mem3 = model.lif3(cur3, mem3)
                hidden_spk.append(spk3.cpu())
                cur4 = model.fc2(spk3)
                spk4, mem4 = model.lif4(cur4, mem4)
                mem_out.append(mem4)

            hidden_spk = torch.stack(hidden_spk)
            mem_out_t = torch.stack(mem_out)
            preds = mem_out_t.sum(dim=0).argmax(dim=1).cpu()
            hidden_features.append(hidden_spk.numpy())
            predictions.append(preds.numpy())

    hidden_features = np.concatenate(
        [h.transpose(1, 0, 2) for h in hidden_features], axis=0)
    predictions = np.concatenate(predictions, axis=0)
    labels = all_labels[:num_samples].numpy()

    fc2_weight = model.fc2.weight.data.cpu().numpy()
    connections = []
    for post in range(NUM_CLASSES):
        for pre in range(256):
            w = float(fc2_weight[post, pre])
            if abs(w) > 0.01:
                connections.append([pre, post, w, 1.0])
    connections = np.array(connections)

    save_dir = RESULTS_DIR / "spinnaker_weights" / f"energy_fold{fold}"
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "hidden_spike_features.npy", hidden_features)
    np.save(save_dir / "hidden_labels.npy", labels)
    np.save(save_dir / "snn_predictions.npy", predictions)
    np.save(save_dir / "fc2_connections.npy", connections)

    snn_acc = (predictions == labels).mean()
    print(f"[{ts()}] Saved {num_samples} samples to {save_dir}")
    print(f"  snnTorch acc: {snn_acc:.4f}, features: {hidden_features.shape}")
    print(f"  FC2 connections: {len(connections)}")


def run_spinnaker_inference(fold, num_samples, weight_scale=1.0):
    """Run FC2 inference on SpiNNaker hardware."""
    import pyNN.spiNNaker as sim

    weights_dir = RESULTS_DIR / "spinnaker_weights" / f"energy_fold{fold}"
    results_dir = RESULTS_DIR / "spinnaker_results" / f"energy_fold{fold}"
    results_dir.mkdir(parents=True, exist_ok=True)

    features = np.load(weights_dir / "hidden_spike_features.npy")
    labels = np.load(weights_dir / "hidden_labels.npy")
    snn_preds = np.load(weights_dir / "snn_predictions.npy")
    fc2_conns = np.load(weights_dir / "fc2_connections.npy")

    N, T, H = features.shape
    num_samples = min(num_samples, N)
    print(f"[{ts()}] SpiNNaker FC2: {num_samples} samples, scale={weight_scale}")

    exc_mask = fc2_conns[:, 2] > 0
    inh_mask = fc2_conns[:, 2] < 0
    exc_conns = fc2_conns[exc_mask].copy()
    exc_conns[:, 2] *= weight_scale
    inh_conns = fc2_conns[inh_mask].copy()
    inh_conns[:, 2] = np.abs(inh_conns[:, 2]) * weight_scale
    print(f"  Exc: {len(exc_conns)}, Inh: {len(inh_conns)}")

    LIF_PARAMS = {
        "cm": 1.0, "tau_m": 20.0, "tau_refrac": 0.1,
        "v_reset": 0.0, "v_rest": 0.0, "v_thresh": 1.0,
        "tau_syn_E": 5.0, "tau_syn_I": 5.0,
    }

    results = []
    correct = 0
    start_time = time.time()

    for idx in range(num_samples):
        sample = features[idx]
        true_label = int(labels[idx])
        spike_times = [[] for _ in range(H)]
        for t in range(T):
            for n in range(H):
