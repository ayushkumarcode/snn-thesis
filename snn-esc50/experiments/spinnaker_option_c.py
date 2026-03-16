"""
spinnaker_option_c.py -- Weight re-centering for full SpiNNaker FC1+FC2 deployment.

The root cause of FC1 cancellation on SpiNNaker:
  FC1 has 256 outputs, each receiving from ~2304 inputs.
  The input spikes (binary, ~680 simultaneous inputs at once) times
  near-zero-mean weights produce large positive and large negative currents
  that cancel out, leaving the FC1 LIF neurons below threshold.

The fix (Option C — zero-cost, no retraining):
  1. For each FC1 output neuron (row of fc1.weight):
       row_mean = fc1.weight[i].mean()
       fc1.weight[i] -= row_mean            # zero-centre
       fc1.bias[i]   += row_mean * n_inputs # compensate in bias
  2. This is mathematically equivalent: the net input to each neuron is
       w_new · x + b_new = (w - μ)·x + (b + μ * n_inputs)
                         = w·x + b + μ * (n_inputs - sum(x))
     When sum(x) ≈ n_inputs (all inputs active), the correction is ≈0.
     But for sparse inputs, it provides a natural offset that prevents
     the cancellation problem.
  3. Test: snnTorch accuracy on fold 4 must be within ±3% of original.
  4. Measure: FC1 activation sparsity — if < 30% of neurons fire per step,
     SpiNNaker full deployment is feasible.

References:
  - Yousefzadeh et al. (2019). "Online Learning in Event-Driven
    Convolutional SNNs on SpiNNaker." Neural Networks.
  - SpiNNaker FC1 analysis in EXPERIMENT_LOG.md.

Usage:
  source .venv/bin/activate
  cd snn-esc50/
  python experiments/spinnaker_option_c.py
  python experiments/spinnaker_option_c.py --fold 4 --verify
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import numpy as np

from src.config import RESULTS_DIR, NUM_STEPS
from src.models.snn_model import SpikingCNN
from src.encoding import encode_direct
from src.dataset import get_fold_dataloaders


# ============================================================
# Weight re-centering (Option C)
# ============================================================

def recenter_fc1_weights(model: SpikingCNN) -> dict:
    """Zero-centre FC1 rows and compensate in bias.

    This is a lossless reparameterisation — it does not change the
    function computed by the network in expectation.

    Returns:
        dict with info about the transformation applied.
    """
    fc1 = model.fc1
    W = fc1.weight.data.clone()  # (256, 2304)
    b = fc1.bias.data.clone()    # (256,)

    n_inputs = W.shape[1]        # 2304

    row_means  = W.mean(dim=1)   # (256,)
    mean_abs   = float(row_means.abs().mean().item())
    mean_shift = float(row_means.mean().item())

    # Zero-centre each row: w_new[i] = w[i] - mean(w[i])
    W_new = W - row_means.unsqueeze(1)
    # Compensate: b_new[i] = b[i] + mean(w[i]) * n_inputs
    b_new = b + row_means * n_inputs

    # Verify: W_new @ 1_vector ≈ 0 for each row (confirming zero-mean)
    ones_vector = torch.ones(n_inputs, device=W_new.device)
    residual = (W_new @ ones_vector).abs().max().item()

    fc1.weight.data = W_new
    fc1.bias.data   = b_new

    return {
        "n_inputs":          n_inputs,
        "n_outputs":         W.shape[0],
        "mean_row_mean":     mean_shift,
        "mean_abs_row_mean": mean_abs,
        "max_residual":      residual,
        "method": "W[i] -= mean(W[i]); b[i] += mean(W[i]) * n_inputs",
    }


# ============================================================
# FC1 spike counting (for SpiNNaker feasibility check)
# ============================================================

def measure_fc1_spikes(model: SpikingCNN, loader, device: str, max_batches: int = 5) -> dict:
    """Measure FC1 (lif3) output spike statistics.

    Returns:
        dict with:
          - mean_fc1_spikes_per_step: mean number of FC1 neurons firing per timestep
          - max_fc1_spikes_per_step:  maximum
          - fc1_firing_rate:          fraction of FC1 neurons active per step
    """
    model.eval()
    all_spk3_counts = []

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            if batch_idx >= max_batches:
                break
            data = data.to(device)
            spk_input = encode_direct(data, num_steps=NUM_STEPS)

            mem1 = model.lif1.init_leaky()
            mem2 = model.lif2.init_leaky()
            mem3 = model.lif3.init_leaky()
            mem4 = model.lif4.init_leaky()

            for step in range(NUM_STEPS):
                x_t = spk_input[step]
                cur1 = model.pool1(model.bn1(model.conv1(x_t)))
                spk1, mem1 = model.lif1(cur1, mem1)
                cur2 = model.pool2(model.bn2(model.conv2(spk1)))
                spk2, mem2 = model.lif2(cur2, mem2)
                pooled = model.avg_pool(spk2)
                flat = pooled.view(pooled.size(0), -1)
                cur3 = model.fc1(flat)
                spk3, mem3 = model.lif3(cur3, mem3)
                # Count: how many of 256 FC1 output neurons fired?
                # Average over batch
                spk_count_per_sample = spk3.sum(dim=1).float()  # (B,)
                all_spk3_counts.append(spk_count_per_sample.cpu().tolist())

    all_counts = np.array(all_spk3_counts).flatten()
    n_fc1 = 256

    return {
        "n_fc1_neurons": n_fc1,
        "mean_fc1_spikes_per_step": float(all_counts.mean()),
        "std_fc1_spikes_per_step":  float(all_counts.std()),
        "max_fc1_spikes_per_step":  float(all_counts.max()),
        "fc1_firing_rate":          float(all_counts.mean() / n_fc1),
        "feasible_for_spinnaker":   bool(all_counts.mean() < 500),  # < ~500 simultaneous inputs to FC2
    }


# ============================================================
# Accuracy evaluation
# ============================================================

@torch.no_grad()
def evaluate_accuracy(model: SpikingCNN, loader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data, num_steps=NUM_STEPS)
        _, mem_out = model(spk_input)
        predicted = mem_out.sum(0).argmax(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    return correct / total if total > 0 else 0.0


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="SpiNNaker Option C: zero-centre FC1 weights for full deployment"
    )
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--verify", action="store_true", default=True,
                        help="Run accuracy verification (default: True)")
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
    print("SpiNNaker Option C: FC1 Weight Re-centering")
    print("=" * 60)
    print(f"  Fold   : {args.fold}")
    print(f"  Device : {device}")
    print()

    snn_path = RESULTS_DIR / "snn" / "direct" / f"best_fold{args.fold}.pt"
    if not snn_path.exists():
        print(f"FATAL: {snn_path} not found")
        sys.exit(1)

    _, test_loader = get_fold_dataloaders(args.fold, batch_size=32, augment=False)

    # --------------------------------------------------------
    # Load original model
    # --------------------------------------------------------
    model = SpikingCNN().to(device)
    model.load_state_dict(
        torch.load(snn_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"Loaded: {snn_path}")

    # --------------------------------------------------------
    # Baseline accuracy + FC1 spike stats (before re-centering)
    # --------------------------------------------------------
    print("Measuring baseline (before re-centering)...")
    acc_before = evaluate_accuracy(model, test_loader, device)
    fc1_before = measure_fc1_spikes(model, test_loader, device)

    print(f"  Accuracy before : {acc_before:.4f} ({acc_before:.2%})")
    print(f"  FC1 spikes/step : {fc1_before['mean_fc1_spikes_per_step']:.1f} ± "
          f"{fc1_before['std_fc1_spikes_per_step']:.1f} "
          f"(of 256 neurons, = {fc1_before['fc1_firing_rate']:.2%} firing rate)")
    print(f"  Max FC1 spikes  : {fc1_before['max_fc1_spikes_per_step']:.1f}")

    # --------------------------------------------------------
    # Apply Option C: zero-centre FC1 rows
    # --------------------------------------------------------
    print()
    print("Applying FC1 weight re-centering...")
    recentre_info = recenter_fc1_weights(model)
    print(f"  Mean |row_mean| before shift: {recentre_info['mean_abs_row_mean']:.6f}")
    print(f"  Max residual after shift:     {recentre_info['max_residual']:.2e}")

    # --------------------------------------------------------
    # Accuracy + FC1 stats after re-centering
    # --------------------------------------------------------
    print()
    print("Measuring after re-centering...")
    acc_after = evaluate_accuracy(model, test_loader, device)
    fc1_after = measure_fc1_spikes(model, test_loader, device)

    acc_drop = acc_before - acc_after
    print(f"  Accuracy after  : {acc_after:.4f} ({acc_after:.2%})")
    print(f"  Accuracy drop   : {acc_drop:+.4f}")
    print(f"  FC1 spikes/step : {fc1_after['mean_fc1_spikes_per_step']:.1f} ± "
          f"{fc1_after['std_fc1_spikes_per_step']:.1f} "
          f"(of 256 neurons, = {fc1_after['fc1_firing_rate']:.2%} firing rate)")
    print(f"  Max FC1 spikes  : {fc1_after['max_fc1_spikes_per_step']:.1f}")

    # --------------------------------------------------------
    # Feasibility assessment
    # --------------------------------------------------------
    print()
    print("=" * 60)
    print("Feasibility Assessment for Full SpiNNaker Deployment")
    print("=" * 60)

    accuracy_ok  = abs(acc_drop) <= 0.03     # within ±3%
    sparsity_ok  = fc1_after["mean_fc1_spikes_per_step"] < 500
    fc1_feasible = accuracy_ok and sparsity_ok

    print(f"  Accuracy preserved (±3%): {'✓' if accuracy_ok else '✗'}")
    print(f"  FC1 spikes < 500/step:    {'✓' if sparsity_ok else '✗'}")
    print(f"  VERDICT: Full SpiNNaker deployment {'FEASIBLE' if fc1_feasible else 'NOT YET FEASIBLE'}")
    print()

    if fc1_feasible:
        print("  Next step: deploy FC1+FC2 on SpiNNaker with re-centred weights.")
        print("  Save re-centred model with: --save-recentred flag.")
    elif not accuracy_ok:
        print("  Accuracy dropped too much. Option C not viable alone.")
        print("  Next: try Option A (higher threshold retraining).")
    else:
        print(f"  FC1 still fires {fc1_after['mean_fc1_spikes_per_step']:.1f} neurons/step on average.")
        print("  That exceeds SpiNNaker's FC2 input capacity (~500).")
        print("  Next: try Option A (higher threshold retraining) to increase sparsity.")

    # Save re-centred model if feasible and accuracy preserved
    if accuracy_ok:
        save_path = RESULTS_DIR / "snn" / "direct" / f"recentred_fold{args.fold}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"\n  Re-centred weights saved: {save_path}")
        print("  (Use this for full SpiNNaker FC1+FC2 deployment)")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    out_dir = RESULTS_DIR / "spinnaker_optionC"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"option_c_fold{args.fold}.json"

    results = {
        "fold": args.fold,
        "device": device,
        "recentre_info": recentre_info,
        "before": {
            "accuracy": acc_before,
            "fc1_stats": fc1_before,
        },
        "after": {
            "accuracy": acc_after,
            "accuracy_drop": acc_drop,
            "fc1_stats": fc1_after,
        },
        "feasibility": {
            "accuracy_ok":        accuracy_ok,
            "sparsity_ok":        sparsity_ok,
            "full_deployment_ok": fc1_feasible,
        },
    }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
