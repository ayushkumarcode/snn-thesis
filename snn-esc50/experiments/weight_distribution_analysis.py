"""
weight_distribution_analysis.py -- Compare learned weight distributions between SNN and ANN.

Extracts weight tensors from matching layers (conv1, conv2, fc1, fc2), computes
distribution statistics (mean, std, kurtosis, sparsity, effective rank, norms),
and produces side-by-side histograms and a summary JSON.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/weight_distribution_analysis.py [--fold N]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

from src.config import RESULTS_DIR, get_device
from src.models.snn_model import SpikingCNN
from src.models.ann_model import ConvANN


# ============================================================
# Layer name mapping
# ============================================================

# SNN has conv1, conv2, fc1, fc2 directly as attributes.
# ANN wraps them in Sequential: features[0]=conv1, features[4]=conv2,
#   classifier[0]=fc1, classifier[3]=fc2.
LAYER_NAMES = ["conv1", "conv2", "fc1", "fc2"]

SNN_WEIGHT_KEYS = {
    "conv1": "conv1.weight",
    "conv2": "conv2.weight",
    "fc1":   "fc1.weight",
    "fc2":   "fc2.weight",
}

ANN_WEIGHT_KEYS = {
    "conv1": "features.0.weight",
    "conv2": "features.4.weight",
    "fc1":   "classifier.0.weight",
    "fc2":   "classifier.3.weight",
}


# ============================================================
# Statistics computation
# ============================================================

def compute_weight_stats(weights: np.ndarray, layer_name: str) -> dict:
    """Compute distribution statistics for a weight tensor.

    Args:
        weights: Flattened numpy array of weight values.
        layer_name: Name of the layer (for logging).

    Returns:
        Dictionary of statistics.
    """
    flat = weights.flatten()

    # Basic statistics
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    w_min = float(np.min(flat))
    w_max = float(np.max(flat))

    # Kurtosis (Fisher definition: normal = 0)
    kurt = float(kurtosis(flat, fisher=True))

    # Sparsity: fraction of weights with |w| < 0.01
    sparsity = float(np.mean(np.abs(flat) < 0.01))

    # L1 and L2 norms
    l1_norm = float(np.sum(np.abs(flat)))
    l2_norm = float(np.sqrt(np.sum(flat ** 2)))

    # Effective rank: ratio of nuclear norm to spectral norm
    # Reshape to 2D (fan_out, fan_in) for SVD
    if weights.ndim == 4:
        # Conv: (out_channels, in_channels, kH, kW) -> (out_channels, in_channels * kH * kW)
        mat = weights.reshape(weights.shape[0], -1)
    elif weights.ndim == 2:
        # Linear: already (out_features, in_features)
        mat = weights
    else:
        mat = weights.reshape(weights.shape[0], -1)

    svd_values = np.linalg.svd(mat, compute_uv=False)
    nuclear_norm = float(np.sum(svd_values))
    spectral_norm = float(svd_values[0])
    effective_rank = float(nuclear_norm / spectral_norm) if spectral_norm > 0 else 0.0

    return {
        "layer": layer_name,
        "shape": list(weights.shape),
        "num_params": int(flat.size),
        "mean": mean,
        "std": std,
        "min": w_min,
        "max": w_max,
        "kurtosis": kurt,
        "sparsity_001": sparsity,
        "l1_norm": l1_norm,
        "l2_norm": l2_norm,
        "nuclear_norm": nuclear_norm,
        "spectral_norm": spectral_norm,
        "effective_rank": effective_rank,
    }


# ============================================================
# Weight extraction
# ============================================================

def extract_weights(state_dict: dict, key_map: dict) -> dict:
    """Extract weight tensors from a state dict using a key mapping.

    Args:
        state_dict: Model state dict.
        key_map: Mapping from layer name to state dict key.

    Returns:
        Dictionary of layer_name -> numpy weight array.
    """
    weights = {}
    for layer_name, key in key_map.items():
        if key in state_dict:
            weights[layer_name] = state_dict[key].cpu().numpy()
        else:
            print(f"  WARNING: key '{key}' not found in state dict")
    return weights


# ============================================================
# Plotting
# ============================================================

def plot_histograms(snn_weights: dict, ann_weights: dict, fold: int,
                    save_dir: Path):
    """Create a 2x4 subplot figure: SNN row, ANN row, 4 layer columns.

    Also saves individual side-by-side histogram figures per layer.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    fig.suptitle(f"Weight Distributions — SNN vs ANN (Fold {fold})", fontsize=14)

    for col, layer_name in enumerate(LAYER_NAMES):
        snn_w = snn_weights[layer_name].flatten()
        ann_w = ann_weights[layer_name].flatten()

        # Determine shared bin range
        all_vals = np.concatenate([snn_w, ann_w])
        vmin, vmax = np.percentile(all_vals, [0.5, 99.5])
        bins = np.linspace(vmin, vmax, 80)

        # SNN row (top)
        ax_snn = axes[0, col]
        ax_snn.hist(snn_w, bins=bins, color="steelblue", alpha=0.8,
                    edgecolor="none", density=True)
        ax_snn.set_title(f"SNN — {layer_name}", fontsize=10)
        ax_snn.set_ylabel("Density" if col == 0 else "")
        ax_snn.tick_params(labelsize=7)
        ax_snn.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

        # ANN row (bottom)
        ax_ann = axes[1, col]
        ax_ann.hist(ann_w, bins=bins, color="coral", alpha=0.8,
                    edgecolor="none", density=True)
        ax_ann.set_title(f"ANN — {layer_name}", fontsize=10)
        ax_ann.set_xlabel("Weight value")
        ax_ann.set_ylabel("Density" if col == 0 else "")
        ax_ann.tick_params(labelsize=7)
        ax_ann.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    grid_path = save_dir / f"weight_distributions_grid_fold{fold}.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grid figure saved: {grid_path}")

    # Individual side-by-side histograms per layer
    for layer_name in LAYER_NAMES:
        snn_w = snn_weights[layer_name].flatten()
        ann_w = ann_weights[layer_name].flatten()

        all_vals = np.concatenate([snn_w, ann_w])
        vmin, vmax = np.percentile(all_vals, [0.5, 99.5])
        bins = np.linspace(vmin, vmax, 80)

        fig_layer, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4))
        fig_layer.suptitle(
            f"Weight Distribution — {layer_name} (Fold {fold})", fontsize=12
        )

        ax_l.hist(snn_w, bins=bins, color="steelblue", alpha=0.8,
                  edgecolor="none", density=True)
        ax_l.set_title(f"SNN ({snn_w.size:,} params)")
        ax_l.set_xlabel("Weight value")
        ax_l.set_ylabel("Density")
        ax_l.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

        ax_r.hist(ann_w, bins=bins, color="coral", alpha=0.8,
                  edgecolor="none", density=True)
        ax_r.set_title(f"ANN ({ann_w.size:,} params)")
        ax_r.set_xlabel("Weight value")
        ax_r.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

        fig_layer.tight_layout(rect=[0, 0, 1, 0.93])
        layer_path = save_dir / f"weight_dist_{layer_name}_fold{fold}.png"
        fig_layer.savefig(layer_path, dpi=150, bbox_inches="tight")
        plt.close(fig_layer)
        print(f"  Layer figure saved: {layer_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Weight distribution analysis: SNN vs ANN"
    )
    parser.add_argument(
        "--fold", type=int, default=1,
        help="Fold to analyse (default: 1)"
    )
    args = parser.parse_args()

    fold = args.fold
    device = get_device()
    print("=" * 60)
    print("Weight Distribution Analysis: SNN vs ANN")
    print("=" * 60)
    print(f"  Fold   : {fold}")
    print(f"  Device : {device}")
    print()

    # Output directory
    save_dir = RESULTS_DIR / "analysis" / "weight_distributions"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load model weights
    # --------------------------------------------------------
    snn_path = RESULTS_DIR / "snn" / "direct" / f"best_fold{fold}.pt"
    ann_path = RESULTS_DIR / "ann" / "none" / f"best_fold{fold}.pt"

    if not snn_path.exists():
        print(f"ERROR: SNN model not found: {snn_path}")
        sys.exit(1)
    if not ann_path.exists():
        print(f"ERROR: ANN model not found: {ann_path}")
        sys.exit(1)

    print(f"Loading SNN: {snn_path}")
    snn_state = torch.load(snn_path, map_location="cpu", weights_only=True)
    print(f"Loading ANN: {ann_path}")
    ann_state = torch.load(ann_path, map_location="cpu", weights_only=True)

    snn_weights = extract_weights(snn_state, SNN_WEIGHT_KEYS)
    ann_weights = extract_weights(ann_state, ANN_WEIGHT_KEYS)

    # --------------------------------------------------------
    # 2. Compute statistics
    # --------------------------------------------------------
    print()
    print("Computing weight statistics...")

    snn_stats = {}
    ann_stats = {}
    for layer_name in LAYER_NAMES:
        snn_stats[layer_name] = compute_weight_stats(
            snn_weights[layer_name], layer_name
        )
        ann_stats[layer_name] = compute_weight_stats(
            ann_weights[layer_name], layer_name
        )

    # Print comparison table
    print()
    print(f"{'Layer':<8} {'Model':<5} {'Mean':>9} {'Std':>9} {'Kurt':>9} "
          f"{'Sparsity':>9} {'EffRank':>9} {'L1':>10} {'L2':>10}")
    print("-" * 90)
    for layer_name in LAYER_NAMES:
        ss = snn_stats[layer_name]
        sa = ann_stats[layer_name]
        print(f"{layer_name:<8} {'SNN':<5} {ss['mean']:>+9.5f} {ss['std']:>9.5f} "
              f"{ss['kurtosis']:>9.3f} {ss['sparsity_001']:>9.3f} "
              f"{ss['effective_rank']:>9.2f} {ss['l1_norm']:>10.2f} {ss['l2_norm']:>10.4f}")
        print(f"{'':8} {'ANN':<5} {sa['mean']:>+9.5f} {sa['std']:>9.5f} "
              f"{sa['kurtosis']:>9.3f} {sa['sparsity_001']:>9.3f} "
              f"{sa['effective_rank']:>9.2f} {sa['l1_norm']:>10.2f} {sa['l2_norm']:>10.4f}")
        print()

    # --------------------------------------------------------
    # 3. Plot histograms
    # --------------------------------------------------------
    print("Generating plots...")
    plot_histograms(snn_weights, ann_weights, fold, save_dir)

    # --------------------------------------------------------
    # 4. Save summary JSON
    # --------------------------------------------------------
    summary = {
        "fold": fold,
        "snn_model_path": str(snn_path),
        "ann_model_path": str(ann_path),
        "layers": {},
    }
    for layer_name in LAYER_NAMES:
        summary["layers"][layer_name] = {
            "snn": snn_stats[layer_name],
            "ann": ann_stats[layer_name],
            "comparison": {
                "mean_diff": snn_stats[layer_name]["mean"] - ann_stats[layer_name]["mean"],
                "std_ratio": (snn_stats[layer_name]["std"] / ann_stats[layer_name]["std"]
                              if ann_stats[layer_name]["std"] > 0 else float("inf")),
                "kurtosis_diff": (snn_stats[layer_name]["kurtosis"]
                                  - ann_stats[layer_name]["kurtosis"]),
                "sparsity_diff": (snn_stats[layer_name]["sparsity_001"]
                                  - ann_stats[layer_name]["sparsity_001"]),
                "effective_rank_diff": (snn_stats[layer_name]["effective_rank"]
                                        - ann_stats[layer_name]["effective_rank"]),
            },
        }

    json_path = save_dir / f"weight_stats_fold{fold}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------
    print()
    print("=" * 60)
    print("Weight Distribution Analysis Summary")
    print("=" * 60)
    total_snn_params = sum(s["num_params"] for s in snn_stats.values())
    total_ann_params = sum(s["num_params"] for s in ann_stats.values())
    avg_snn_sparsity = np.mean([s["sparsity_001"] for s in snn_stats.values()])
    avg_ann_sparsity = np.mean([s["sparsity_001"] for s in ann_stats.values()])
    print(f"  Total weight params analysed: SNN={total_snn_params:,}, ANN={total_ann_params:,}")
    print(f"  Avg near-zero sparsity (|w|<0.01): SNN={avg_snn_sparsity:.3f}, ANN={avg_ann_sparsity:.3f}")
    print(f"  Figures saved to: {save_dir}")
    print(f"  JSON saved to:    {json_path}")


if __name__ == "__main__":
    main()
