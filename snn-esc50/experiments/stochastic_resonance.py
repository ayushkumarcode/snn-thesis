"""
stochastic_resonance.py -- Test whether adding noise IMPROVES SNN classification.

Research question: Does stochastic resonance occur in SNNs? Adding Gaussian
noise to membrane potential currents could help sub-threshold signals cross the
spike threshold, potentially improving classification accuracy. This is a
well-known phenomenon in biological neural systems.

Method:
  - Load trained SNN (direct encoding) and ANN for each fold
  - At inference, add Gaussian noise N(0, sigma^2) to pre-LIF currents (SNN)
    or pre-ReLU activations (ANN)
  - Sweep sigma: 0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0
  - For each sigma, run 3 evaluations with different noise seeds and average
  - Plot accuracy vs sigma for both SNN and ANN
  - Identify if any noise level improves over baseline (stochastic resonance!)

Key insight: If stochastic resonance is present, the accuracy-vs-sigma curve
should show a peak at some sigma > 0 before declining. This would demonstrate
that SNNs, like biological neurons, can exploit noise constructively.

Usage:
  source .venv/bin/activate
  cd snn-esc50/
  python experiments/stochastic_resonance.py
  python experiments/stochastic_resonance.py --fold 4
  python experiments/stochastic_resonance.py --fold 1 --device cpu
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

from src.config import RESULTS_DIR, NUM_FOLDS, BATCH_SIZE, NUM_STEPS, get_device
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct
from src.models.snn_model import SpikingCNN
from src.models.ann_model import ConvANN


# ============================================================
# Noise injection forward passes
# ============================================================

def snn_forward_with_noise(
    model: SpikingCNN,
    x: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SNN forward pass with Gaussian noise added to pre-LIF currents.

    Adds N(0, sigma^2) noise to cur1, cur2, cur3, cur4 before each
    LIF layer processes them. This tests whether sub-threshold signals
    can be pushed over the spike threshold by noise (stochastic resonance).

    Args:
        model: Trained SpikingCNN model (in eval mode).
        x: Spike input of shape (num_steps, batch, 1, n_mels, time_frames).
        sigma: Standard deviation of Gaussian noise. 0.0 = no noise.
        generator: Optional torch.Generator for reproducible noise.

    Returns:
        spk_out: Output spikes, shape (num_steps, batch, num_classes).
        mem_out: Output membrane potentials, shape (num_steps, batch, num_classes).
    """
    # Initialise hidden states
    mem1 = model.lif1.init_leaky()
    mem2 = model.lif2.init_leaky()
    mem3 = model.lif3.init_leaky()
    mem4 = model.lif4.init_leaky()

    spk_out_rec = []
    mem_out_rec = []

    for step in range(model.num_steps):
        x_t = x[step]  # (batch, 1, n_mels, time)

        # Conv block 1 + noise
        cur1 = model.pool1(model.bn1(model.conv1(x_t)))
        if sigma > 0.0:
            cur1 = cur1 + torch.randn_like(cur1) * sigma
        spk1, mem1 = model.lif1(cur1, mem1)

        # Conv block 2 + noise
        cur2 = model.pool2(model.bn2(model.conv2(spk1)))
        if sigma > 0.0:
            cur2 = cur2 + torch.randn_like(cur2) * sigma
        spk2, mem2 = model.lif2(cur2, mem2)

        # Pool + flatten
        pooled = model.avg_pool(spk2)
        flat = pooled.view(pooled.size(0), -1)

        # FC block 1 + noise
        cur3 = model.fc1(flat)
        if sigma > 0.0:
            cur3 = cur3 + torch.randn_like(cur3) * sigma
        spk3, mem3 = model.lif3(cur3, mem3)

        # FC block 2 (output) + noise
        cur4 = model.fc2(spk3)
        if sigma > 0.0:
            cur4 = cur4 + torch.randn_like(cur4) * sigma
        spk4, mem4 = model.lif4(cur4, mem4)

        spk_out_rec.append(spk4)
        mem_out_rec.append(mem4)

    return torch.stack(spk_out_rec), torch.stack(mem_out_rec)


def ann_forward_with_noise(
    model: ConvANN,
    x: torch.Tensor,
    sigma: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """ANN forward pass with Gaussian noise added to pre-ReLU activations.

    Mirrors the SNN experiment: noise is injected at the same 4 locations
    (after conv1+bn1+pool1, conv2+bn2+pool2, fc1, fc2) but before
    ReLU / output respectively.

    Args:
        model: Trained ConvANN model (in eval mode).
        x: Input tensor of shape (batch, 1, n_mels, time_frames).
        sigma: Standard deviation of Gaussian noise. 0.0 = no noise.
        generator: Optional torch.Generator for reproducible noise.

    Returns:
        logits: Output logits, shape (batch, num_classes).
    """
    # Unpack model.features layers manually to inject noise
    # model.features = Sequential(
    #   0: Conv2d(1,32), 1: BN(32), 2: ReLU, 3: MaxPool(2),
    #   4: Conv2d(32,64), 5: BN(64), 6: ReLU, 7: MaxPool(2),
    #   8: AvgPool2d(4,6)
    # )
    # model.classifier = Sequential(
    #   0: Linear(2304,256), 1: ReLU, 2: Dropout, 3: Linear(256,50)
    # )

    features = model.features
    classifier = model.classifier

    # Conv block 1: Conv2d -> BN -> (noise) -> ReLU -> MaxPool
    h = features[1](features[0](x))  # conv1 + bn1
    if sigma > 0.0:
        h = h + torch.randn_like(h) * sigma
    h = features[3](features[2](h))  # ReLU + MaxPool

    # Conv block 2: Conv2d -> BN -> (noise) -> ReLU -> MaxPool
    h = features[5](features[4](h))  # conv2 + bn2
    if sigma > 0.0:
        h = h + torch.randn_like(h) * sigma
    h = features[7](features[6](h))  # ReLU + MaxPool

    # AvgPool
    h = features[8](h)

    # Flatten
    flat = h.view(h.size(0), -1)

    # FC1 -> (noise) -> ReLU -> Dropout
    h = classifier[0](flat)  # Linear(2304, 256)
    if sigma > 0.0:
        h = h + torch.randn_like(h) * sigma
    h = classifier[2](classifier[1](h))  # ReLU + Dropout

    # FC2 (output) -> (noise)
    logits = classifier[3](h)  # Linear(256, 50)
    if sigma > 0.0:
        logits = logits + torch.randn_like(logits) * sigma

    return logits


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate_snn_noisy(
    model: SpikingCNN,
    loader,
    device,
    sigma: float,
    seed: int,
) -> float:
    """Evaluate SNN with noise injection at a given sigma and seed."""
    model.eval()
    # Generate noise on CPU to avoid MPS generator device mismatch
    torch.manual_seed(seed)

    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spk_input = encode_direct(data, num_steps=NUM_STEPS)
            spk_out, mem_out = snn_forward_with_noise(
                model, spk_input, sigma, None
            )
            # Rate decoding: class with most output spikes
            preds = spk_out.sum(dim=0).argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return correct / total if total > 0 else 0.0


def evaluate_ann_noisy(
    model: ConvANN,
    loader,
    device,
    sigma: float,
    seed: int,
) -> float:
    """Evaluate ANN with noise injection at a given sigma and seed."""
    model.eval()
    # Use global seed to avoid MPS generator device mismatch
    torch.manual_seed(seed)

    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            logits = ann_forward_with_noise(model, data, sigma, None)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return correct / total if total > 0 else 0.0


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Stochastic resonance: does noise improve SNN accuracy?"
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Specific fold to evaluate (default: all folds)"
    )
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    else:
        device = str(get_device())

    # Determine folds
    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = list(range(1, NUM_FOLDS + 1))

    # Noise sigma values to sweep
    sigmas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    num_noise_seeds = 3
    noise_seeds = [42, 123, 777]

    print("=" * 70)
    print("Stochastic Resonance Experiment")
    print("=" * 70)
    print(f"  Folds          : {folds}")
    print(f"  Noise sigmas   : {sigmas}")
    print(f"  Noise seeds    : {noise_seeds}")
    print(f"  Device         : {device}")
    print()

    # Ensure dataset is available
    download_esc50()

    # Output directory
    out_dir = RESULTS_DIR / "snn" / "stochastic_resonance"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect results across all folds
    all_fold_results = {}

    for fold in folds:
        print(f"\n{'='*70}")
        print(f"Fold {fold}")
        print(f"{'='*70}")

        # Check model paths
        snn_path = RESULTS_DIR / "snn" / "direct" / f"best_fold{fold}.pt"
        ann_path = RESULTS_DIR / "ann" / "none" / f"best_fold{fold}.pt"

        if not snn_path.exists():
            print(f"  WARNING: SNN model not found: {snn_path} — skipping fold")
            continue
        if not ann_path.exists():
            print(f"  WARNING: ANN model not found: {ann_path} — skipping fold")
            continue

        # Load models
        snn_model = SpikingCNN().to(device)
        snn_model.load_state_dict(
            torch.load(snn_path, map_location=device, weights_only=True)
        )
        snn_model.eval()
        print(f"  Loaded SNN: {snn_path}")

        ann_model = ConvANN().to(device)
        ann_model.load_state_dict(
            torch.load(ann_path, map_location=device, weights_only=True)
        )
        ann_model.eval()
        print(f"  Loaded ANN: {ann_path}")

        # Data loader
        _, test_loader = get_fold_dataloaders(fold, batch_size=BATCH_SIZE, augment=False)

        # Results for this fold
        fold_results = {
            "fold": fold,
            "sigmas": sigmas,
            "noise_seeds": noise_seeds,
            "snn": {"per_seed": {}, "mean": [], "std": []},
            "ann": {"per_seed": {}, "mean": [], "std": []},
        }

        # Header
        print()
        print(f"  {'sigma':>8}  {'SNN mean':>10}  {'SNN std':>9}  "
              f"{'ANN mean':>10}  {'ANN std':>9}")
        print(f"  {'-'*52}")

        for sigma in sigmas:
            snn_accs = []
            ann_accs = []

            for seed in noise_seeds:
                snn_acc = evaluate_snn_noisy(
                    snn_model, test_loader, device, sigma, seed
                )
                ann_acc = evaluate_ann_noisy(
                    ann_model, test_loader, device, sigma, seed
                )
                snn_accs.append(snn_acc)
                ann_accs.append(ann_acc)

                # Store per-seed results
                seed_key = str(seed)
                if seed_key not in fold_results["snn"]["per_seed"]:
                    fold_results["snn"]["per_seed"][seed_key] = []
                    fold_results["ann"]["per_seed"][seed_key] = []
                fold_results["snn"]["per_seed"][seed_key].append(float(snn_acc))
                fold_results["ann"]["per_seed"][seed_key].append(float(ann_acc))

            snn_mean = float(np.mean(snn_accs))
            snn_std = float(np.std(snn_accs))
            ann_mean = float(np.mean(ann_accs))
            ann_std = float(np.std(ann_accs))

            fold_results["snn"]["mean"].append(snn_mean)
            fold_results["snn"]["std"].append(snn_std)
            fold_results["ann"]["mean"].append(ann_mean)
            fold_results["ann"]["std"].append(ann_std)

            print(f"  {sigma:>8.3f}  {snn_mean:>10.4f}  {snn_std:>9.4f}  "
                  f"{ann_mean:>10.4f}  {ann_std:>9.4f}")

        # Detect stochastic resonance
        snn_baseline = fold_results["snn"]["mean"][0]  # sigma=0.0
        ann_baseline = fold_results["ann"]["mean"][0]
        snn_best_idx = int(np.argmax(fold_results["snn"]["mean"]))
        ann_best_idx = int(np.argmax(fold_results["ann"]["mean"]))
        snn_best_sigma = sigmas[snn_best_idx]
        ann_best_sigma = sigmas[ann_best_idx]
        snn_best_acc = fold_results["snn"]["mean"][snn_best_idx]
        ann_best_acc = fold_results["ann"]["mean"][ann_best_idx]

        snn_resonance = snn_best_sigma > 0.0 and snn_best_acc > snn_baseline
        ann_resonance = ann_best_sigma > 0.0 and ann_best_acc > ann_baseline

        fold_results["stochastic_resonance"] = {
            "snn": {
                "detected": snn_resonance,
                "baseline_acc": snn_baseline,
                "best_acc": snn_best_acc,
                "best_sigma": snn_best_sigma,
                "improvement": snn_best_acc - snn_baseline,
            },
            "ann": {
                "detected": ann_resonance,
                "baseline_acc": ann_baseline,
                "best_acc": ann_best_acc,
                "best_sigma": ann_best_sigma,
                "improvement": ann_best_acc - ann_baseline,
            },
        }

        print()
        print(f"  SNN baseline (sigma=0): {snn_baseline:.4f}")
        print(f"  SNN best (sigma={snn_best_sigma}): {snn_best_acc:.4f} "
              f"(delta={snn_best_acc - snn_baseline:+.4f})")
        if snn_resonance:
            print(f"  >>> STOCHASTIC RESONANCE DETECTED IN SNN! <<<")
            print(f"  >>> Noise sigma={snn_best_sigma} improves accuracy by "
                  f"{(snn_best_acc - snn_baseline)*100:+.2f} pp <<<")
        else:
            print(f"  No stochastic resonance in SNN (noise only hurts).")

        print(f"  ANN baseline (sigma=0): {ann_baseline:.4f}")
        print(f"  ANN best (sigma={ann_best_sigma}): {ann_best_acc:.4f} "
              f"(delta={ann_best_acc - ann_baseline:+.4f})")
        if ann_resonance:
            print(f"  >>> STOCHASTIC RESONANCE DETECTED IN ANN! <<<")
        else:
            print(f"  No stochastic resonance in ANN (expected: continuous activations).")

        all_fold_results[str(fold)] = fold_results

        # Clean up GPU memory
        del snn_model, ann_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Aggregate across folds
    # --------------------------------------------------------
    if len(all_fold_results) > 1:
        print(f"\n{'='*70}")
        print("Aggregate Results (mean across folds)")
        print(f"{'='*70}")

        agg_snn_means = []
        agg_ann_means = []
        for fold_key, fr in all_fold_results.items():
            agg_snn_means.append(fr["snn"]["mean"])
            agg_ann_means.append(fr["ann"]["mean"])

        agg_snn_means = np.array(agg_snn_means)  # (num_folds, num_sigmas)
        agg_ann_means = np.array(agg_ann_means)

        print(f"\n  {'sigma':>8}  {'SNN mean':>10}  {'SNN std':>9}  "
              f"{'ANN mean':>10}  {'ANN std':>9}")
        print(f"  {'-'*52}")

        for i, sigma in enumerate(sigmas):
            snn_m = float(agg_snn_means[:, i].mean())
            snn_s = float(agg_snn_means[:, i].std())
            ann_m = float(agg_ann_means[:, i].mean())
            ann_s = float(agg_ann_means[:, i].std())
            print(f"  {sigma:>8.3f}  {snn_m:>10.4f}  {snn_s:>9.4f}  "
                  f"{ann_m:>10.4f}  {ann_s:>9.4f}")

    # --------------------------------------------------------
    # Save results JSON
    # --------------------------------------------------------
    summary = {
        "experiment": "stochastic_resonance",
        "sigmas": sigmas,
        "noise_seeds": noise_seeds,
        "num_noise_seeds": num_noise_seeds,
        "folds": all_fold_results,
    }

    # Add aggregate if multiple folds
    if len(all_fold_results) > 1:
        summary["aggregate"] = {
            "snn_mean_per_sigma": agg_snn_means.mean(axis=0).tolist(),
            "snn_std_per_sigma": agg_snn_means.std(axis=0).tolist(),
            "ann_mean_per_sigma": agg_ann_means.mean(axis=0).tolist(),
            "ann_std_per_sigma": agg_ann_means.std(axis=0).tolist(),
        }

    out_path = out_dir / "stochastic_resonance.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved: {out_path}")

    # --------------------------------------------------------
    # Plot accuracy vs sigma
    # --------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        if len(all_fold_results) > 1:
            # Plot aggregate
            snn_means = agg_snn_means.mean(axis=0)
            snn_stds = agg_snn_means.std(axis=0)
            ann_means = agg_ann_means.mean(axis=0)
            ann_stds = agg_ann_means.std(axis=0)
            title_suffix = f"(mean over {len(all_fold_results)} folds)"
        else:
            # Plot single fold
            fold_key = list(all_fold_results.keys())[0]
            fr = all_fold_results[fold_key]
            snn_means = np.array(fr["snn"]["mean"])
            snn_stds = np.array(fr["snn"]["std"])
            ann_means = np.array(fr["ann"]["mean"])
            ann_stds = np.array(fr["ann"]["std"])
            title_suffix = f"(fold {fold_key})"

        sigmas_arr = np.array(sigmas)

        # SNN
        ax.plot(sigmas_arr, snn_means * 100, "o-", color="tab:blue",
                linewidth=2, markersize=6, label="SNN (direct)", zorder=3)
        ax.fill_between(sigmas_arr,
                        (snn_means - snn_stds) * 100,
                        (snn_means + snn_stds) * 100,
                        alpha=0.2, color="tab:blue")

        # ANN
        ax.plot(sigmas_arr, ann_means * 100, "s-", color="tab:red",
                linewidth=2, markersize=6, label="ANN", zorder=3)
        ax.fill_between(sigmas_arr,
                        (ann_means - ann_stds) * 100,
                        (ann_means + ann_stds) * 100,
                        alpha=0.2, color="tab:red")

        # Baseline reference lines
        ax.axhline(y=snn_means[0] * 100, color="tab:blue", linestyle="--",
                   alpha=0.4, linewidth=1)
        ax.axhline(y=ann_means[0] * 100, color="tab:red", linestyle="--",
                   alpha=0.4, linewidth=1)

        # Highlight stochastic resonance peaks if above baseline
        snn_best_i = int(np.argmax(snn_means))
        if snn_best_i > 0 and snn_means[snn_best_i] > snn_means[0]:
            ax.annotate(
                f"SR peak: {snn_means[snn_best_i]*100:.1f}%\n"
                f"(+{(snn_means[snn_best_i]-snn_means[0])*100:.2f} pp)",
                xy=(sigmas_arr[snn_best_i], snn_means[snn_best_i] * 100),
                xytext=(sigmas_arr[snn_best_i] + 0.05,
                        snn_means[snn_best_i] * 100 + 3),
                arrowprops=dict(arrowstyle="->", color="tab:blue"),
                fontsize=9, color="tab:blue", fontweight="bold",
            )

        ax.set_xlabel("Noise sigma", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"Stochastic Resonance: Accuracy vs Noise Level {title_suffix}",
                     fontsize=13)
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, max(sigmas) + 0.05)

        plot_path = out_dir / "stochastic_resonance.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved: {plot_path}")

        # Also make a log-scale x-axis version for better visibility of small sigmas
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

        # Skip sigma=0 for log scale; plot it separately
        nonzero_mask = sigmas_arr > 0
        sigmas_nz = sigmas_arr[nonzero_mask]

        ax2.plot(sigmas_nz, snn_means[nonzero_mask] * 100, "o-", color="tab:blue",
                 linewidth=2, markersize=6, label="SNN (direct)", zorder=3)
        ax2.fill_between(sigmas_nz,
                         (snn_means[nonzero_mask] - snn_stds[nonzero_mask]) * 100,
                         (snn_means[nonzero_mask] + snn_stds[nonzero_mask]) * 100,
                         alpha=0.2, color="tab:blue")

        ax2.plot(sigmas_nz, ann_means[nonzero_mask] * 100, "s-", color="tab:red",
                 linewidth=2, markersize=6, label="ANN", zorder=3)
        ax2.fill_between(sigmas_nz,
                         (ann_means[nonzero_mask] - ann_stds[nonzero_mask]) * 100,
                         (ann_means[nonzero_mask] + ann_stds[nonzero_mask]) * 100,
                         alpha=0.2, color="tab:red")

        # Baselines (sigma=0)
        ax2.axhline(y=snn_means[0] * 100, color="tab:blue", linestyle="--",
                    alpha=0.5, linewidth=1, label=f"SNN baseline ({snn_means[0]*100:.1f}%)")
        ax2.axhline(y=ann_means[0] * 100, color="tab:red", linestyle="--",
                    alpha=0.5, linewidth=1, label=f"ANN baseline ({ann_means[0]*100:.1f}%)")

        ax2.set_xscale("log")
        ax2.set_xlabel("Noise sigma (log scale)", fontsize=12)
        ax2.set_ylabel("Accuracy (%)", fontsize=12)
        ax2.set_title(f"Stochastic Resonance (log scale) {title_suffix}", fontsize=13)
        ax2.legend(fontsize=10, loc="best")
        ax2.grid(True, alpha=0.3, which="both")

        plot_log_path = out_dir / "stochastic_resonance_logscale.png"
        fig2.savefig(plot_log_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"Log-scale plot saved: {plot_log_path}")

    except ImportError:
        print("matplotlib not available — skipping plots.")
    except Exception as e:
        print(f"Plotting failed ({e}) — results JSON still saved.")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
