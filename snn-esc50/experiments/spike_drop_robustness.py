#!/usr/bin/env python3
"""
spike_drop_robustness.py -- SNN resilience to random spike packet drops.

Simulates SpiNNaker's real routing fabric behaviour where spike packets can be
lost due to congestion, buffer overflow, or routing faults.  This is distinct
from neuron ablation (permanent death) or adversarial perturbation (optimised
noise): here we randomly zero a fraction of *individual spike events* across
all timesteps, modelling transient communication failures.

Research question: How gracefully does FC2-only SNN inference degrade when a
fraction of inter-neuron spike packets are dropped?  This directly quantifies
the gap between ideal software simulation and noisy neuromorphic hardware.

Method:
  - Load pre-computed hidden spike features (shape: N, 25, 256) produced by
    the full SNN (conv layers + FC1 + lif3) and stored as binary spike trains.
  - For each drop rate in {0%, 5%, 10%, 20%, 30%, 50%}:
    - Randomly zero out that fraction of all 1-valued spikes.
    - Run FC2-only snnTorch inference (fc2 + lif4) on the corrupted features.
    - Record accuracy.
  - Repeat with 5 random seeds per drop rate for robustness.
  - Plot accuracy vs drop rate, overlaying actual SpiNNaker accuracy if available.

Usage:
  source .venv/bin/activate
  cd snn-esc50/
  python experiments/spike_drop_robustness.py
  python experiments/spike_drop_robustness.py --fold 4
  python experiments/spike_drop_robustness.py --fold 4 --num-samples 400
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR, BETA, NUM_STEPS
from src.models.snn_model import SpikingCNN

import snntorch as snn
from snntorch import surrogate


# ============================================================
# Spike drop utility
# ============================================================

def apply_spike_drops(features: np.ndarray, drop_rate: float,
                      rng: np.random.RandomState) -> np.ndarray:
    """Randomly zero out a fraction of all 1-valued spikes.

    This simulates lost spike packets in a neuromorphic routing fabric.
    Only spikes that are currently 1 can be dropped; zero entries are
    never modified (no spurious spike injection).

    Args:
        features: Binary spike array, shape (N, T, H), dtype float.
        drop_rate: Fraction of 1-spikes to zero out, in [0, 1].
        rng: Numpy RandomState for reproducibility.

    Returns:
        Corrupted copy of features with the specified fraction of spikes
        removed.
    """
    if drop_rate <= 0.0:
        return features.copy()

    corrupted = features.copy()

    # Find all indices where spikes are present
    spike_indices = np.argwhere(corrupted > 0.5)  # binary: >0.5 means spike
    num_spikes = len(spike_indices)

    if num_spikes == 0:
        return corrupted

    # Determine how many spikes to drop
    num_to_drop = int(round(num_spikes * drop_rate))
    if num_to_drop == 0:
        return corrupted

    # Randomly select spikes to zero out
    drop_selection = rng.choice(num_spikes, size=num_to_drop, replace=False)
    drop_coords = spike_indices[drop_selection]

    # Zero out selected spikes
    corrupted[drop_coords[:, 0], drop_coords[:, 1], drop_coords[:, 2]] = 0.0

    return corrupted


# ============================================================
# FC2-only snnTorch inference
# ============================================================

@torch.no_grad()
def fc2_inference(features: np.ndarray, labels: np.ndarray,
                  fc2_weight: torch.Tensor, fc2_bias: torch.Tensor,
                  beta: float, device: torch.device) -> float:
    """Run FC2-only snnTorch inference on pre-computed hidden spike features.

    For each sample, iterates over T=25 timesteps:
      cur = fc2(spikes[t])
      spk, mem = lif4(cur, mem)

    Decoding: sum membrane potentials over time, argmax over classes.

    Args:
        features: Spike features, shape (N, T, 256), binary float.
        labels: Ground truth labels, shape (N,), int.
        fc2_weight: FC2 weight tensor, shape (50, 256).
        fc2_bias: FC2 bias tensor, shape (50,).
        beta: LIF membrane decay rate.
        device: Torch device.

    Returns:
        Accuracy as a float in [0, 1].
    """
    N, T, H = features.shape
    num_classes = fc2_weight.shape[0]

    # Build a minimal FC2 + LIF4 module
    fc2 = nn.Linear(H, num_classes).to(device)
    fc2.weight.data.copy_(fc2_weight)
    fc2.bias.data.copy_(fc2_bias)

    spike_grad = surrogate.fast_sigmoid(slope=25)
    lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad).to(device)

    # Convert features to tensor
    feat_tensor = torch.tensor(features, dtype=torch.float32, device=device)

    correct = 0
    for i in range(N):
        mem = lif4.init_leaky()
        mem_sum = torch.zeros(num_classes, device=device)

        for t in range(T):
            spk_t = feat_tensor[i, t, :]  # (256,)
            cur = fc2(spk_t.unsqueeze(0))  # (1, 50)
            spk_out, mem = lif4(cur, mem)
            mem_sum += mem.squeeze(0)

        pred = mem_sum.argmax().item()
        if pred == labels[i]:
            correct += 1

    return correct / N


# ============================================================
# Plotting
# ============================================================

def plot_spike_drop_robustness(drop_rates: list[float],
                               mean_accs: list[float],
                               std_accs: list[float],
                               spinnaker_acc: float | None,
                               clean_ref_acc: float | None,
                               out_path: Path):
    """Plot accuracy vs spike drop rate.

    Args:
        drop_rates: List of drop rates (fractions).
        mean_accs: Mean accuracy at each drop rate.
        std_accs: Std accuracy at each drop rate.
        spinnaker_acc: If available, actual SpiNNaker accuracy for comparison.
        clean_ref_acc: Clean snnTorch FC2 accuracy (drop_rate=0).
        out_path: Path to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = [r * 100 for r in drop_rates]
    y = [a * 100 for a in mean_accs]
    yerr = [s * 100 for s in std_accs]

    ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=4,
                color="#2196F3", label="snnTorch FC2 (with drops)")

    if spinnaker_acc is not None:
        ax.axhline(y=spinnaker_acc * 100, color="#FF5722", linestyle="--",
                   linewidth=2, alpha=0.8,
                   label=f"SpiNNaker actual ({spinnaker_acc:.1%})")

    if clean_ref_acc is not None:
        ax.axhline(y=clean_ref_acc * 100, color="#4CAF50", linestyle=":",
                   linewidth=1.5, alpha=0.6,
                   label=f"snnTorch clean ({clean_ref_acc:.1%})")

    ax.set_xlabel("Spike Drop Rate (%)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("SNN Resilience to Random Spike Packet Drops\n"
                 "(simulating SpiNNaker routing fabric losses)",
                 fontsize=13)
    ax.legend(fontsize=10, loc="best")
    ax.set_xlim(-2, max(x) + 2)
    ax.set_ylim(0, max(y + [spinnaker_acc * 100 if spinnaker_acc else 0]) * 1.15)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SNN resilience to random spike packet drops "
                    "(simulating SpiNNaker routing fabric behaviour)"
    )
    parser.add_argument(
        "--fold", type=int, default=4,
        help="Fold to evaluate (default: 4)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=400,
        help="Number of samples to use (default: 400)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=5,
        help="Number of random seeds per drop rate (default: 5)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (default: auto-detect)"
    )
    args = parser.parse_args()

    fold = args.fold
    num_samples = args.num_samples
    num_seeds = args.num_seeds

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        from src.config import get_device
        device = get_device()

    # Drop rates to sweep
    drop_rates = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]

    print("=" * 65)
    print("Spike Drop Robustness: FC2-only SNN Inference")
    print("=" * 65)
    print(f"  Fold         : {fold}")
    print(f"  Num samples  : {num_samples}")
    print(f"  Drop rates   : {[f'{r:.0%}' for r in drop_rates]}")
    print(f"  Seeds/rate   : {num_seeds}")
    print(f"  Device       : {device}")
    print()

    # --------------------------------------------------------
    # Load pre-computed hidden spike features
    # --------------------------------------------------------
    # Try per-fold directory first, then fall back to root
    fold_dir = RESULTS_DIR / "spinnaker_weights" / f"fold{fold}"
    root_dir = RESULTS_DIR / "spinnaker_weights"

    if (fold_dir / "hidden_spike_features.npy").exists():
        features_path = fold_dir / "hidden_spike_features.npy"
        labels_path = fold_dir / "hidden_labels.npy"
        print(f"  Loading features from fold directory: {fold_dir.name}/")
    elif (root_dir / "hidden_spike_features.npy").exists():
        features_path = root_dir / "hidden_spike_features.npy"
        labels_path = root_dir / "hidden_labels.npy"
        print(f"  Loading features from root weights directory")
    else:
        print(f"FATAL: No hidden_spike_features.npy found for fold {fold}.")
        print(f"  Checked: {fold_dir}")
        print(f"  Checked: {root_dir}")
        print("  Run the feature extraction script first.")
        sys.exit(1)

    if not labels_path.exists():
        print(f"FATAL: Labels not found: {labels_path}")
        sys.exit(1)

    features = np.load(features_path)
    labels = np.load(labels_path)

    # Limit to num_samples
    if features.shape[0] > num_samples:
        features = features[:num_samples]
        labels = labels[:num_samples]

    N, T, H = features.shape
    print(f"  Features     : {features.shape} (N={N}, T={T}, H={H})")
    print(f"  Labels       : {labels.shape}, unique classes: {len(np.unique(labels))}")

    total_spikes = int(np.sum(features > 0.5))
    total_entries = features.size
    spike_density = total_spikes / total_entries
    print(f"  Total spikes : {total_spikes} / {total_entries} "
          f"({spike_density:.1%} density)")
    print()

    # --------------------------------------------------------
    # Load FC2 weights from SpikingCNN model
    # --------------------------------------------------------
    model_path = RESULTS_DIR / "snn" / "direct" / f"best_fold{fold}.pt"
    if not model_path.exists():
        print(f"FATAL: Model not found: {model_path}")
        sys.exit(1)

    model = SpikingCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu",
                                     weights_only=True))
    fc2_weight = model.fc2.weight.data.clone().to(device)  # (50, 256)
    fc2_bias = model.fc2.bias.data.clone().to(device)      # (50,)

    print(f"  Loaded FC2 weights from: {model_path.name}")
    print(f"  FC2 shape: weight {tuple(fc2_weight.shape)}, "
          f"bias {tuple(fc2_bias.shape)}")
    print()

    # --------------------------------------------------------
    # Run spike drop sweep
    # --------------------------------------------------------
    results_per_rate = {}

    print(f"{'Drop Rate':>10}  {'Mean Acc':>10}  {'Std':>8}  "
          f"{'Seeds':>30}")
    print("-" * 65)

    for drop_rate in drop_rates:
        seed_accs = []

        for seed_idx in range(num_seeds):
            seed = fold * 10000 + seed_idx * 100 + int(drop_rate * 100)
            rng = np.random.RandomState(seed)

            # Apply spike drops
            corrupted = apply_spike_drops(features, drop_rate, rng)

            # Run FC2-only inference
            acc = fc2_inference(corrupted, labels, fc2_weight, fc2_bias,
                                BETA, device)
            seed_accs.append(acc)

        mean_acc = np.mean(seed_accs)
        std_acc = np.std(seed_accs)
        results_per_rate[f"{drop_rate:.2f}"] = {
            "drop_rate": drop_rate,
            "accuracies": [float(a) for a in seed_accs],
            "mean": float(mean_acc),
            "std": float(std_acc),
        }

        seed_str = ", ".join(f"{a:.1%}" for a in seed_accs)
        print(f"{drop_rate:>9.0%}   {mean_acc:>9.2%}  {std_acc:>7.2%}  "
              f"[{seed_str}]")

    # --------------------------------------------------------
    # Load SpiNNaker actual accuracy for comparison
    # --------------------------------------------------------
    spinnaker_acc = None
    fivefold_path = RESULTS_DIR / "spinnaker_results" / "5fold_summary.json"
    if fivefold_path.exists():
        try:
            with open(fivefold_path) as f:
                summary_5fold = json.load(f)
            fold_accs = summary_5fold.get("fold_accuracies_spinnaker", [])
            if fold <= len(fold_accs):
                spinnaker_acc = fold_accs[fold - 1]  # 0-indexed
                print(f"\n  SpiNNaker actual accuracy (fold {fold}): "
                      f"{spinnaker_acc:.1%}")
        except (json.JSONDecodeError, KeyError, IndexError):
            pass

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------
    clean_acc = results_per_rate["0.00"]["mean"]
    worst_rate = drop_rates[-1]
    worst_acc = results_per_rate[f"{worst_rate:.2f}"]["mean"]
    absolute_drop = clean_acc - worst_acc
    relative_drop = absolute_drop / clean_acc if clean_acc > 0 else float("inf")

    print()
    print("=" * 65)
    print("Analysis")
    print("=" * 65)
    print(f"  Clean FC2 accuracy      : {clean_acc:.2%}")
    print(f"  At {worst_rate:.0%} drop rate       : {worst_acc:.2%}")
    print(f"  Absolute degradation    : {absolute_drop:.2%}")
    print(f"  Relative degradation    : {relative_drop:.1%} of clean accuracy lost")

    if spinnaker_acc is not None:
        # Find which drop rate most closely matches SpiNNaker accuracy
        mean_accs_list = [results_per_rate[f"{r:.2f}"]["mean"]
                          for r in drop_rates]
        closest_idx = int(np.argmin(
            [abs(a - spinnaker_acc) for a in mean_accs_list]
        ))
        equiv_rate = drop_rates[closest_idx]
        equiv_acc = mean_accs_list[closest_idx]

        print(f"\n  SpiNNaker equivalent drop rate: ~{equiv_rate:.0%} "
              f"(SpiNNaker={spinnaker_acc:.1%}, "
              f"simulated@{equiv_rate:.0%}={equiv_acc:.1%})")
        print("  Interpretation: The SpiNNaker hardware gap corresponds to "
              f"approximately {equiv_rate:.0%} effective spike loss.")

    # --------------------------------------------------------
    # Save JSON results
    # --------------------------------------------------------
    out_dir = RESULTS_DIR / "spinnaker_results" / "spike_drop"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_json = {
        "fold": fold,
        "num_samples": N,
        "num_seeds": num_seeds,
        "beta": BETA,
        "num_steps": T,
        "hidden_size": H,
        "spike_density": float(spike_density),
        "total_spikes": total_spikes,
        "drop_rates": drop_rates,
        "results": results_per_rate,
        "spinnaker_actual_accuracy": spinnaker_acc,
        "analysis": {
            "clean_accuracy": float(clean_acc),
            "worst_drop_rate": worst_rate,
            "worst_accuracy": float(worst_acc),
            "absolute_degradation": float(absolute_drop),
            "relative_degradation": float(relative_drop),
        },
    }

    if spinnaker_acc is not None:
        results_json["analysis"]["spinnaker_equivalent_drop_rate"] = equiv_rate
        results_json["analysis"]["spinnaker_equivalent_simulated_acc"] = float(
            equiv_acc
        )

    json_path = out_dir / f"spike_drop_fold{fold}.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved: {json_path}")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    mean_accs = [results_per_rate[f"{r:.2f}"]["mean"] for r in drop_rates]
    std_accs = [results_per_rate[f"{r:.2f}"]["std"] for r in drop_rates]

    plot_path = out_dir / f"spike_drop_fold{fold}.png"
    plot_spike_drop_robustness(
        drop_rates, mean_accs, std_accs,
        spinnaker_acc, clean_acc, plot_path,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
