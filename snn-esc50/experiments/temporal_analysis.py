"""
temporal_analysis.py -- First-spike latency readout + raster plot analysis.

Research question: Does first-spike timing carry additional information
beyond the standard rate (spike-count) decoding used during training?

Experiments:
  1. First-spike latency accuracy vs rate-count accuracy on fold 4.
  2. Raster plots: T=25 timestep output spike patterns for 10 examples.
  3. Per-class first-spike latency: which classes fire earliest?
  4. Spike count distribution across 50 output classes.

Reference:
  "Beyond Rate Coding: Surrogate Gradients Enable Spike Timing Learning"
  arXiv:2507.16043 (2025).

Usage:
  source .venv/bin/activate
  cd snn-esc50/
  python experiments/temporal_analysis.py
  python experiments/temporal_analysis.py --fold 4 --raster-samples 10
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from src.config import RESULTS_DIR, NUM_STEPS, NUM_CLASSES
from src.models.snn_model import SpikingCNN
from src.encoding import encode_direct
from src.dataset import get_fold_dataloaders, get_class_labels


# ============================================================
# Decoding functions
# ============================================================

def rate_decode(spk_out: torch.Tensor) -> torch.Tensor:
    """Standard rate decoding: predict class with most spikes.

    Args:
        spk_out: (T, B, num_classes) — output spikes over all timesteps.

    Returns:
        (B,) class predictions.
    """
    return spk_out.sum(dim=0).argmax(dim=1)


def first_spike_decode(spk_out: torch.Tensor) -> torch.Tensor:
    """First-spike latency decoding: predict class that fires first.

    Ties broken by highest total spike count.
    Silent neurons (no spike ever) assigned latency = T.

    Args:
        spk_out: (T, B, num_classes) — output spikes.

    Returns:
        (B,) class predictions.
    """
    T, B, C = spk_out.shape

    # First-spike latency for each (batch, class) pair
    latency = torch.full((B, C), float(T), device=spk_out.device)
    for t in range(T):
        fired_now = (spk_out[t] > 0.5) & (latency == float(T))
        batch_idx, class_idx = fired_now.nonzero(as_tuple=True)
        latency[batch_idx, class_idx] = float(t)

    # Class with smallest latency (earliest spike) wins
    # Among ties, prefer higher spike count (use small negative offset)
    total_spikes = spk_out.sum(dim=0).float()  # (B, C)
    eps = 1e-4
    score = latency - eps * total_spikes  # lower = better

    return score.argmin(dim=1)


def mean_first_spike_latency(spk_out: torch.Tensor,
                              preds: torch.Tensor) -> torch.Tensor:
    """Mean first-spike latency of the predicted class per sample.

    Args:
        spk_out: (T, B, num_classes)
        preds: (B,) predicted class indices

    Returns:
        (B,) mean first-spike latency of the winner class.
    """
    T, B, C = spk_out.shape
    winner_spikes = spk_out[:, torch.arange(B), preds]  # (T, B)

    latency = torch.full((B,), float(T), device=spk_out.device)
    for t in range(T):
        fired_now = (winner_spikes[t] > 0.5) & (latency == float(T))
        latency[fired_now] = float(t)

    return latency


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Temporal spike analysis")
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--raster-samples", type=int, default=10,
                        help="Number of examples for raster plots (default: 10)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("Temporal Spike Analysis")
    print("=" * 60)
    print(f"  Fold   : {args.fold}")
    print(f"  Device : {device}")
    print()

    # Load SNN model
    snn_path = RESULTS_DIR / "snn" / "direct" / f"best_fold{args.fold}.pt"
    if not snn_path.exists():
        print(f"FATAL: SNN model not found: {snn_path}")
        sys.exit(1)

    model = SpikingCNN().to(device)
    model.load_state_dict(
        torch.load(snn_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"Loaded SNN: {snn_path}")

    _, test_loader = get_fold_dataloaders(args.fold, batch_size=32, augment=False)
    class_labels = get_class_labels()

    # --------------------------------------------------------
    # Collect spike outputs for full test set
    # --------------------------------------------------------
    all_spk_out    = []   # list of (T, B, 50) tensors
    all_targets    = []
    raster_data    = []   # (spk_out, target, pred_rate, pred_latency) for plotting

    print("Running forward pass on test set...")

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            spk_input = encode_direct(data, num_steps=NUM_STEPS)
            spk_out, mem_out = model(spk_input)  # (T, B, 50), (T, B, 50)

            all_spk_out.append(spk_out.cpu())
            all_targets.append(targets.cpu())

            # Collect raster examples (first batch only)
            if batch_idx == 0 and len(raster_data) < args.raster_samples:
                pred_rate    = rate_decode(spk_out).cpu()
                pred_latency = first_spike_decode(spk_out).cpu()
                for i in range(min(args.raster_samples, data.size(0))):
                    raster_data.append({
                        "spk_out":      spk_out[:, i, :].cpu().numpy().tolist(),
                        "target":       int(targets[i].item()),
                        "target_label": class_labels[targets[i].item()],
                        "pred_rate":    int(pred_rate[i].item()),
                        "pred_latency": int(pred_latency[i].item()),
                        "correct_rate":    bool(pred_rate[i].item() == targets[i].item()),
                        "correct_latency": bool(pred_latency[i].item() == targets[i].item()),
                    })

    # Concatenate across batches
    all_spk_cat = torch.cat([s.permute(1, 0, 2) for s in all_spk_out], dim=0)
    # all_spk_cat: (N_total, T, 50)
    all_spk_t = all_spk_cat.permute(1, 0, 2)  # (T, N_total, 50)
    all_targets_cat = torch.cat(all_targets)   # (N_total,)

    N = all_targets_cat.shape[0]
    print(f"Total samples: {N}")

    # --------------------------------------------------------
    # Accuracy comparison: rate vs first-spike
    # --------------------------------------------------------
    pred_rate    = rate_decode(all_spk_t)
    pred_latency = first_spike_decode(all_spk_t)

    acc_rate    = (pred_rate == all_targets_cat).float().mean().item()
    acc_latency = (pred_latency == all_targets_cat).float().mean().item()

    print()
    print(f"  Rate decoding accuracy    : {acc_rate:.4f} ({acc_rate:.2%})")
    print(f"  First-spike accuracy      : {acc_latency:.4f} ({acc_latency:.2%})")
    print(f"  Δ (first-spike - rate)    : {acc_latency - acc_rate:+.4f}")

    # --------------------------------------------------------
    # Per-class first-spike latency (mean latency of winner neuron
    # on correctly classified samples)
    # --------------------------------------------------------
    winner_latency = mean_first_spike_latency(all_spk_t, pred_rate)

    # Group by true class
    per_class_latency = {}
    for c in range(NUM_CLASSES):
        mask = (all_targets_cat == c)
        if mask.sum() > 0:
            lats = winner_latency[mask].tolist()
            per_class_latency[class_labels[c]] = {
                "mean": float(np.mean(lats)),
                "std":  float(np.std(lats)),
                "all":  [round(l, 2) for l in lats],
            }

    # Sort by mean latency (earliest-firing classes first)
    sorted_classes = sorted(
        per_class_latency.items(), key=lambda x: x[1]["mean"]
    )
    print()
    print("  Earliest-firing classes (mean winner first-spike latency):")
    for label, stats in sorted_classes[:5]:
        print(f"    {label:<25} {stats['mean']:.2f} ± {stats['std']:.2f} steps")
    print("  Latest-firing classes:")
    for label, stats in sorted_classes[-5:]:
        print(f"    {label:<25} {stats['mean']:.2f} ± {stats['std']:.2f} steps")

    # --------------------------------------------------------
    # Spike count distribution
    # --------------------------------------------------------
    # Total spikes per output neuron averaged across test set
    total_spikes_per_class = all_spk_t.sum(dim=0).mean(dim=0)  # (50,)
    spike_count_stats = {
        "mean_per_class": total_spikes_per_class.tolist(),
        "overall_mean": float(total_spikes_per_class.mean().item()),
        "overall_std": float(total_spikes_per_class.std().item()),
        "max_spikes_possible_per_sample": NUM_STEPS,
        "mean_firing_rate": float(
            (all_spk_t.sum(dim=0).float().mean(dim=0) / NUM_STEPS).mean().item()
        ),
    }

    print()
    print(f"  Output spike stats:")
    print(f"    Mean spikes/class/sample : {spike_count_stats['overall_mean']:.2f}"
          f" ± {spike_count_stats['overall_std']:.2f}")
    print(f"    Mean firing rate          : {spike_count_stats['mean_firing_rate']:.4f}"
          f" (fraction of T={NUM_STEPS} timesteps)")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    out_dir = RESULTS_DIR / "temporal_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "fold": args.fold,
        "num_steps": NUM_STEPS,
        "num_samples": N,
        "accuracy": {
            "rate_decoding":         acc_rate,
            "first_spike_decoding":  acc_latency,
            "delta":                 acc_latency - acc_rate,
        },
        "per_class_latency": dict(sorted_classes),
        "spike_count_stats": spike_count_stats,
        "raster_examples": raster_data,
    }

    out_path = out_dir / f"temporal_analysis_fold{args.fold}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved: {out_path}")
    print()

    # --------------------------------------------------------
    # Try to save raster plot (matplotlib optional)
    # --------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_examples = min(args.raster_samples, len(raster_data))
        fig, axes = plt.subplots(n_examples, 1, figsize=(14, n_examples * 1.5),
                                 tight_layout=True)
        if n_examples == 1:
            axes = [axes]

        for i, ex in enumerate(raster_data[:n_examples]):
            ax = axes[i]
            spk_arr = np.array(ex["spk_out"])  # (T, 50)
            spike_times, neuron_ids = np.where(spk_arr > 0.5)
            ax.scatter(spike_times, neuron_ids, s=4, color="black", alpha=0.6)
            ax.axvline(x=0, color="red", alpha=0.3, lw=0.5)

            # Highlight correct class
            tgt = ex["target"]
            ax.axhline(y=tgt, color="blue", alpha=0.4, lw=1.0, linestyle="--")

            correct_sym = "✓" if ex["correct_rate"] else "✗"
            ax.set_ylabel(f"Neuron", fontsize=7)
            ax.set_title(
                f"[{i+1}] True: {ex['target_label']} | "
                f"Rate {correct_sym} → class {ex['pred_rate']} | "
                f"Lat {'✓' if ex['correct_latency'] else '✗'} → class {ex['pred_latency']}",
                fontsize=8,
            )
            ax.set_xlim(-0.5, NUM_STEPS - 0.5)
            ax.set_ylim(-0.5, NUM_CLASSES - 0.5)

        axes[-1].set_xlabel("Timestep", fontsize=9)

        raster_path = out_dir / f"raster_fold{args.fold}.png"
        fig.savefig(raster_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Raster plot saved: {raster_path}")

    except ImportError:
        print("matplotlib not available — skipping raster plot.")
    except Exception as e:
        print(f"Raster plot failed ({e}) — results JSON still saved.")


if __name__ == "__main__":
    main()
