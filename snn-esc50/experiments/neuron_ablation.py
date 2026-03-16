"""
neuron_ablation.py -- Neuron silencing fault tolerance comparison: SNN vs ANN.

Research question: How gracefully does each model degrade when random neurons
are permanently silenced at inference time? If SNNs distribute information
more broadly via temporal spike coding, they may tolerate neuron dropout
better than rate-coded ANNs.

Method:
  - Load trained SNN (direct encoding) and ANN models for each fold.
  - At inference, register forward hooks that zero out a random fraction
    of neurons (activations) at each layer:
      SNN: hook after lif1, lif2, lif3 (zeroes spikes)
      ANN: hook after ReLU in features[2], features[6], classifier[1]
  - Sweep ablation rates: 0%, 10%, 20%, 30%, 40%, 50%.
  - For each ablation rate, repeat with 5 random mask seeds and average.
  - Run on all 5 folds (or a single fold via --fold), report mean +/- std.
  - Save results JSON and degradation curve plot.

Key insight:
  SNN neurons communicate via binary spikes across T=25 timesteps, so the
  same information is temporally redundant. ANN neurons communicate via
  single scalar activations -- losing a neuron permanently destroys that
  channel's information.

Usage:
  source .venv/bin/activate
  cd snn-esc50/
  python experiments/neuron_ablation.py
  python experiments/neuron_ablation.py --fold 4
  python experiments/neuron_ablation.py --fold 1 --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn

from src.config import RESULTS_DIR, NUM_FOLDS, BATCH_SIZE, get_device, NUM_STEPS
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct
from src.models.snn_model import SpikingCNN
from src.models.ann_model import ConvANN


# ============================================================
# Ablation hook machinery
# ============================================================

class NeuronAblationHook:
    """Forward hook that zeros out a random fraction of neurons (channels).

    For convolutional layers the mask is applied per-channel (same spatial
    mask across H, W).  For fully-connected layers the mask is applied
    per-neuron.

    The mask is generated once at construction time for a given seed and
    ablation rate, then applied identically on every forward call.  This
    simulates permanent neuron death, not stochastic dropout.

    Args:
        num_neurons: Number of neurons (channels or hidden units) in the layer.
        ablation_rate: Fraction of neurons to silence, in [0, 1].
        seed: Random seed for reproducible mask generation.
        device: Torch device for the mask tensor.
    """

    def __init__(self, num_neurons: int, ablation_rate: float,
                 seed: int, device: torch.device):
        self.ablation_rate = ablation_rate
        rng = np.random.RandomState(seed)
        num_to_kill = int(round(num_neurons * ablation_rate))
        kill_indices = rng.choice(num_neurons, size=num_to_kill, replace=False) \
            if num_to_kill > 0 else np.array([], dtype=int)

        # Build a binary mask: 1 = keep, 0 = silenced
        mask = torch.ones(num_neurons, device=device)
        if len(kill_indices) > 0:
            mask[kill_indices] = 0.0
        self.mask = mask

    def __call__(self, module, input, output):
        """Apply the permanent ablation mask to the layer output.

        LIF layers return (spike, membrane) tuples. We mask the spikes
        and return the modified tuple. Conv/Linear layers return tensors.
        """
        if isinstance(output, tuple):
            # LIF layer: output = (spikes, membrane_potential)
            spk, mem = output
            if spk.dim() == 4:
                spk = spk * self.mask[None, :, None, None]
            elif spk.dim() == 2:
                spk = spk * self.mask[None, :]
            else:
                spk = spk * self.mask
            return (spk, mem)
        elif output.dim() == 4:
            # Conv output: (B, C, H, W) -- mask along channel dim
            return output * self.mask[None, :, None, None]
        elif output.dim() == 2:
            # FC output: (B, N) -- mask along neuron dim
            return output * self.mask[None, :]
        else:
            return output * self.mask


def register_snn_ablation_hooks(model: SpikingCNN, ablation_rate: float,
                                seed: int, device: torch.device) -> list:
    """Register ablation hooks on each LIF layer of the SNN.

    Hooks are placed on lif1 (32 channels), lif2 (64 channels), and
    lif3 (256 neurons).  We skip lif4 (output layer, 50 neurons) because
    silencing output neurons would trivially destroy those classes.

    The hook zeros spikes AFTER the LIF layer fires, so the ablated neuron
    produces zero spikes for all T timesteps -- equivalent to a dead neuron.

    Args:
        model: Trained SpikingCNN model.
        ablation_rate: Fraction of neurons to silence per layer.
        seed: Random seed (each layer gets a different sub-seed).
        device: Torch device.

    Returns:
        List of hook handles (call .remove() to detach).
    """
    handles = []
    layer_specs = [
        (model.lif1, 32,  seed),
        (model.lif2, 64,  seed + 1000),
        (model.lif3, 256, seed + 2000),
    ]
    for layer, n_neurons, layer_seed in layer_specs:
        hook = NeuronAblationHook(n_neurons, ablation_rate, layer_seed, device)
        h = layer.register_forward_hook(hook)
        handles.append(h)
    return handles


def register_ann_ablation_hooks(model: ConvANN, ablation_rate: float,
                                seed: int, device: torch.device) -> list:
    """Register ablation hooks on each ReLU layer of the ANN.

    Hooks placed on:
      features[2]  (ReLU after conv1+bn1, 32 channels)
      features[6]  (ReLU after conv2+bn2, 64 channels)
      classifier[1] (ReLU after fc1, 256 neurons)

    We skip the output layer (classifier[3]) for the same reason as the SNN.

    Args:
        model: Trained ConvANN model.
        ablation_rate: Fraction of neurons to silence per layer.
        seed: Random seed (each layer gets a different sub-seed).
        device: Torch device.

    Returns:
        List of hook handles.
    """
    handles = []
    layer_specs = [
        (model.features[2],    32,  seed),       # ReLU after conv1
        (model.features[6],    64,  seed + 1000), # ReLU after conv2
        (model.classifier[1],  256, seed + 2000), # ReLU after fc1
    ]
    for layer, n_neurons, layer_seed in layer_specs:
        hook = NeuronAblationHook(n_neurons, ablation_rate, layer_seed, device)
        h = layer.register_forward_hook(hook)
        handles.append(h)
    return handles


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_snn(model: SpikingCNN, test_loader, device: torch.device) -> float:
    """Evaluate SNN accuracy using direct encoding and rate decoding."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            spk_input = encode_direct(data, num_steps=NUM_STEPS)
            spk_out, mem_out = model(spk_input)
            # Rate decoding: sum spikes over time, argmax over classes
            preds = spk_out.sum(dim=0).argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_ann(model: ConvANN, test_loader, device: torch.device) -> float:
    """Evaluate ANN accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0.0


# ============================================================
# Single-fold ablation sweep
# ============================================================

def run_fold(fold: int, ablation_rates: list[float], num_mask_seeds: int,
             device: torch.device) -> dict:
    """Run the full ablation sweep for one fold.

    Args:
        fold: Test fold (1-5).
        ablation_rates: List of ablation fractions to sweep.
        num_mask_seeds: Number of random mask seeds per ablation rate.
        device: Torch device.

    Returns:
        Dict with per-rate accuracy lists for SNN and ANN.
    """
    # Load models
    snn_path = RESULTS_DIR / "snn" / "direct" / f"best_fold{fold}.pt"
    ann_path = RESULTS_DIR / "ann" / "none" / f"best_fold{fold}.pt"

    for p in [snn_path, ann_path]:
        if not p.exists():
            print(f"  FATAL: Model not found: {p}")
            sys.exit(1)

    snn_model = SpikingCNN().to(device)
    snn_model.load_state_dict(
        torch.load(snn_path, map_location=device, weights_only=True)
    )
    snn_model.eval()

    ann_model = ConvANN().to(device)
    ann_model.load_state_dict(
        torch.load(ann_path, map_location=device, weights_only=True)
    )
    ann_model.eval()

    # Data loader (test fold only, no augmentation)
    _, test_loader = get_fold_dataloaders(fold, batch_size=BATCH_SIZE, augment=False)

    fold_results = {
        "fold": fold,
        "ablation_rates": ablation_rates,
        "num_mask_seeds": num_mask_seeds,
        "snn": {},  # rate -> list of accuracies across seeds
        "ann": {},
    }

    for rate in ablation_rates:
        snn_accs = []
        ann_accs = []

        for seed_idx in range(num_mask_seeds):
            mask_seed = fold * 10000 + seed_idx * 100 + int(rate * 100)

            # --- SNN with ablation ---
            handles = register_snn_ablation_hooks(
                snn_model, rate, mask_seed, device
            )
            acc_snn = evaluate_snn(snn_model, test_loader, device)
            for h in handles:
                h.remove()
            snn_accs.append(acc_snn)

            # --- ANN with ablation ---
            handles = register_ann_ablation_hooks(
                ann_model, rate, mask_seed, device
            )
            acc_ann = evaluate_ann(ann_model, test_loader, device)
            for h in handles:
                h.remove()
            ann_accs.append(acc_ann)

        rate_key = f"{rate:.2f}"
        fold_results["snn"][rate_key] = snn_accs
        fold_results["ann"][rate_key] = ann_accs

        snn_mean = np.mean(snn_accs)
        ann_mean = np.mean(ann_accs)
        print(f"    rate={rate:.0%}  SNN={snn_mean:.2%} ± {np.std(snn_accs):.2%}"
              f"  ANN={ann_mean:.2%} ± {np.std(ann_accs):.2%}")

    return fold_results


# ============================================================
# Plotting
# ============================================================

def plot_degradation_curves(summary: dict, out_path: Path):
    """Plot SNN vs ANN accuracy degradation as a function of ablation rate.

    Shows mean across folds with std error bars.

    Args:
        summary: Dict with 'ablation_rates', 'snn_mean', 'snn_std',
                 'ann_mean', 'ann_std'.
        out_path: Path to save the PNG.
    """
    rates = [r * 100 for r in summary["ablation_rates"]]  # percent
    snn_mean = [v * 100 for v in summary["snn_mean"]]
    snn_std  = [v * 100 for v in summary["snn_std"]]
    ann_mean = [v * 100 for v in summary["ann_mean"]]
    ann_std  = [v * 100 for v in summary["ann_std"]]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.errorbar(rates, snn_mean, yerr=snn_std,
                marker="o", linewidth=2, capsize=4, label="SNN (direct)")
    ax.errorbar(rates, ann_mean, yerr=ann_std,
                marker="s", linewidth=2, capsize=4, label="ANN (ReLU)")

    ax.set_xlabel("Neuron Ablation Rate (%)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Fault Tolerance: Accuracy vs Neuron Silencing Rate", fontsize=13)
    ax.legend(fontsize=11, loc="best")
    ax.set_xlim(-2, 52)
    ax.set_ylim(0, max(max(snn_mean), max(ann_mean)) * 1.15)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def plot_relative_degradation(summary: dict, out_path: Path):
    """Plot accuracy retention (% of clean accuracy) to normalise the
    different baseline accuracies of SNN vs ANN.

    Args:
        summary: Dict with ablation results.
        out_path: Path to save the PNG.
    """
    rates = [r * 100 for r in summary["ablation_rates"]]

    # Compute retention relative to clean (rate=0%) accuracy
    snn_clean = summary["snn_mean"][0]
    ann_clean = summary["ann_mean"][0]

    if snn_clean == 0 or ann_clean == 0:
        print("  Skipping relative plot: clean accuracy is zero.")
        return

    snn_retention = [(v / snn_clean) * 100 for v in summary["snn_mean"]]
    ann_retention = [(v / ann_clean) * 100 for v in summary["ann_mean"]]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(rates, snn_retention, marker="o", linewidth=2,
            label=f"SNN (clean={snn_clean:.1%})")
    ax.plot(rates, ann_retention, marker="s", linewidth=2,
            label=f"ANN (clean={ann_clean:.1%})")

    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Neuron Ablation Rate (%)", fontsize=12)
    ax.set_ylabel("Accuracy Retention (% of clean)", fontsize=12)
    ax.set_title("Relative Degradation: Accuracy Retention vs Ablation Rate",
                 fontsize=13)
    ax.legend(fontsize=11, loc="best")
    ax.set_xlim(-2, 52)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Relative plot saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Neuron ablation fault tolerance: SNN vs ANN"
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Single fold to run (1-5). Default: all 5 folds."
    )
    parser.add_argument(
        "--ablation-rates", type=float, nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        help="Ablation rates to sweep (default: 0.0 0.1 0.2 0.3 0.4 0.5)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=5,
        help="Number of random mask seeds per ablation rate (default: 5)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (default: auto-detect)"
    )
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    # Determine which folds to run
    if args.fold is not None:
        if args.fold < 1 or args.fold > NUM_FOLDS:
            print(f"FATAL: fold must be 1-{NUM_FOLDS}, got {args.fold}")
            sys.exit(1)
        folds = [args.fold]
    else:
        folds = list(range(1, NUM_FOLDS + 1))

    ablation_rates = sorted(args.ablation_rates)

    print("=" * 60)
    print("Neuron Ablation Fault Tolerance: SNN vs ANN")
    print("=" * 60)
    print(f"  Folds          : {folds}")
    print(f"  Ablation rates : {[f'{r:.0%}' for r in ablation_rates]}")
    print(f"  Mask seeds     : {args.num_seeds}")
    print(f"  Device         : {device}")
    print()

    # Ensure dataset is available
    download_esc50()

    # Output directory
    out_dir = RESULTS_DIR / "snn" / "neuron_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Run each fold
    # --------------------------------------------------------
    all_fold_results = []

    for fold in folds:
        print(f"--- Fold {fold} ---")
        fold_result = run_fold(fold, ablation_rates, args.num_seeds, device)
        all_fold_results.append(fold_result)

        # Save per-fold result
        fold_path = out_dir / f"ablation_fold{fold}.json"
        with open(fold_path, "w") as f:
            json.dump(fold_result, f, indent=2)
        print(f"  Fold {fold} saved: {fold_path}")
        print()

    # --------------------------------------------------------
    # Aggregate across folds
    # --------------------------------------------------------
    # For each ablation rate, compute the mean accuracy (averaged over
    # seeds first, then across folds) and std across folds.
    summary = {
        "folds": folds,
        "ablation_rates": ablation_rates,
        "num_mask_seeds": args.num_seeds,
        "snn_mean": [],
        "snn_std": [],
        "ann_mean": [],
        "ann_std": [],
        "snn_per_fold": {},
        "ann_per_fold": {},
    }

    for rate in ablation_rates:
        rate_key = f"{rate:.2f}"
        snn_fold_means = []
        ann_fold_means = []

        for fr in all_fold_results:
            snn_fold_means.append(np.mean(fr["snn"][rate_key]))
            ann_fold_means.append(np.mean(fr["ann"][rate_key]))

        summary["snn_mean"].append(float(np.mean(snn_fold_means)))
        summary["snn_std"].append(float(np.std(snn_fold_means)))
        summary["ann_mean"].append(float(np.mean(ann_fold_means)))
        summary["ann_std"].append(float(np.std(ann_fold_means)))
        summary["snn_per_fold"][rate_key] = snn_fold_means
        summary["ann_per_fold"][rate_key] = ann_fold_means

    # Save summary
    summary_path = out_dir / "ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")

    # --------------------------------------------------------
    # Print results table
    # --------------------------------------------------------
    print()
    print("=" * 60)
    print("Results (mean +/- std across folds)")
    print("=" * 60)
    print(f"{'Rate':>6}  {'SNN Accuracy':>18}  {'ANN Accuracy':>18}  {'Delta':>8}")
    print("-" * 60)

    for i, rate in enumerate(ablation_rates):
        sm = summary["snn_mean"][i]
        ss = summary["snn_std"][i]
        am = summary["ann_mean"][i]
        as_ = summary["ann_std"][i]
        delta = sm - am
        print(f"{rate:>5.0%}   "
              f"{sm:>7.2%} +/- {ss:>5.2%}   "
              f"{am:>7.2%} +/- {as_:>5.2%}   "
              f"{delta:>+7.2%}")

    # --------------------------------------------------------
    # Degradation analysis
    # --------------------------------------------------------
    if len(ablation_rates) >= 2 and ablation_rates[0] == 0.0:
        snn_clean = summary["snn_mean"][0]
        ann_clean = summary["ann_mean"][0]
        snn_50 = summary["snn_mean"][-1]
        ann_50 = summary["ann_mean"][-1]

        if snn_clean > 0 and ann_clean > 0:
            snn_retention = snn_50 / snn_clean
            ann_retention = ann_50 / ann_clean

            print()
            print("Degradation at max ablation rate "
                  f"({ablation_rates[-1]:.0%}):")
            print(f"  SNN: {snn_clean:.2%} -> {snn_50:.2%} "
                  f"(retains {snn_retention:.1%} of clean accuracy)")
            print(f"  ANN: {ann_clean:.2%} -> {ann_50:.2%} "
                  f"(retains {ann_retention:.1%} of clean accuracy)")

            if snn_retention > ann_retention:
                print("  FINDING: SNN is MORE fault-tolerant "
                      f"(retains {snn_retention - ann_retention:+.1%} more).")
            elif ann_retention > snn_retention:
                print("  FINDING: ANN is MORE fault-tolerant "
                      f"(retains {ann_retention - snn_retention:+.1%} more).")
            else:
                print("  FINDING: Both models degrade equally.")

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    print()
    plot_degradation_curves(summary, out_dir / "ablation_curves.png")
    plot_relative_degradation(summary, out_dir / "ablation_relative.png")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
