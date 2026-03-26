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
        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # FC block 1
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        thresh3 = self.thresholds.get("fc1", 1.0)
        self.if3 = snn.Leaky(beta=1.0, threshold=thresh3, spike_grad=spike_grad,
                              learn_beta=False, learn_threshold=False)

        # FC block 2 (output) -- no spiking, accumulate membrane potential
        self.fc2 = nn.Linear(256, num_classes)

    def forward(
        self, x: torch.Tensor, num_steps: int = 25,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over num_steps timesteps.

        The input spectrogram is presented identically at each timestep
        (direct/repeat encoding). The IF neurons integrate without leak.

        Args:
            x: Spectrogram of shape (batch, 1, n_mels, time_frames).
            num_steps: Number of simulation timesteps.

        Returns:
            spk_out: Output spikes from IF3, shape (num_steps, batch, 256).
            mem_out: Accumulated membrane potential at output layer,
                     shape (num_steps, batch, num_classes).
        """
        # Initialize membrane potentials
        mem1 = self.if1.init_leaky()
        mem2 = self.if2.init_leaky()
        mem3 = self.if3.init_leaky()

        # Output accumulator (no spiking at output layer)
        batch_size = x.shape[0]
        device = x.device
        mem_out_acc = torch.zeros(batch_size, self.fc2.out_features, device=device)

        spk_rec = []
        mem_rec = []

        for step in range(num_steps):
            # Conv block 1
            cur1 = self.pool1(self.bn1(self.conv1(x)))
            spk1, mem1 = self.if1(cur1, mem1)

            # Conv block 2
            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.if2(cur2, mem2)

            # Pool + flatten
            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            # FC block 1
            cur3 = self.fc1(flat)
            spk3, mem3 = self.if3(cur3, mem3)

            # FC block 2 (accumulate, no spike)
            cur4 = self.fc2(spk3)
            mem_out_acc = mem_out_acc + cur4

            spk_rec.append(spk3)
            mem_rec.append(mem_out_acc.clone())

        return torch.stack(spk_rec), torch.stack(mem_rec)


def transfer_weights(ann_model: ConvANN, snn_model: ConvertedSNN):
    """Transfer weights from trained ANN to converted SNN.

    The ConvANN and ConvertedSNN have identical weight structures
    for conv1, bn1, conv2, bn2, fc1, fc2.
    """
    # Conv block 1
    snn_model.conv1.load_state_dict(ann_model.features[0].state_dict())
    snn_model.bn1.load_state_dict(ann_model.features[1].state_dict())

    # Conv block 2
    snn_model.conv2.load_state_dict(ann_model.features[4].state_dict())
    snn_model.bn2.load_state_dict(ann_model.features[5].state_dict())

    # FC layers
    snn_model.fc1.load_state_dict(ann_model.classifier[0].state_dict())
    snn_model.fc2.load_state_dict(ann_model.classifier[3].state_dict())


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_converted_snn(model, loader, device, num_steps):
    """Evaluate converted SNN at a given number of timesteps."""
    model.eval()
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_out, mem_out = model(data, num_steps=num_steps)

        # Predict from accumulated membrane potential at last timestep
        predicted = mem_out[-1].argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return correct / total if total > 0 else 0.0


@torch.no_grad()
def evaluate_ann(model, loader, device):
    """Evaluate original ANN for reference."""
    model.eval()
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        logits = model(data)
        predicted = logits.argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return correct / total if total > 0 else 0.0


# ============================================================
# Main Conversion Pipeline
# ============================================================

def convert_and_evaluate_fold(
    fold: int,
    device,
    percentile: float,
    max_timesteps: int,
):
    """Full ANN-to-SNN conversion pipeline for one fold.

    Steps:
      1. Load trained ANN
      2. Record activations over training set
      3. Compute per-layer thresholds
      4. Create converted SNN with calibrated thresholds
      5. Evaluate at multiple timestep budgets

    Returns:
        dict with fold results.
    """
    print(f"\n{'='*60}")
    print(f"  ANN-to-SNN Conversion | Fold {fold}/5")
    print(f"  Percentile: {percentile} | Max timesteps: {max_timesteps}")
    print(f"{'='*60}")

    ann_path = RESULTS_DIR / "ann" / "none" / f"best_fold{fold}.pt"
    if not ann_path.exists():
        print(f"  ERROR: ANN model not found at {ann_path}")
        return None

    # Step 1: Load trained ANN
    ann_model = ConvANN().to(device)
    ann_model.load_state_dict(
        torch.load(ann_path, map_location=device, weights_only=True)
    )
    ann_model.eval()
    print(f"  Loaded ANN: {ann_path}")

    # Step 2: Record activations over training set
    train_loader, test_loader = get_fold_dataloaders(fold, batch_size=BATCH_SIZE)

    recorder = ActivationRecorder()
    recorder.register_hooks(ann_model)

    print(f"  Recording activations over {len(train_loader.dataset)} training samples...")
    t0 = time.time()
    with torch.no_grad():
        for data, _ in train_loader:
            data = data.to(device)
            _ = ann_model(data)
    recorder.remove_hooks()
    print(f"  Activation recording done ({time.time()-t0:.1f}s)")

    # Step 3: Compute per-layer thresholds
    thresholds = recorder.compute_thresholds(percentile=percentile)
    print(f"  Calibrated thresholds (percentile={percentile}):")
    for name, thresh in thresholds.items():
        print(f"    {name}: {thresh:.4f}")

    # Step 4: Create converted SNN and transfer weights
    snn_model = ConvertedSNN(thresholds=thresholds).to(device)
    transfer_weights(ann_model, snn_model)
    snn_model.eval()
    print(f"  Created converted SNN with calibrated thresholds")

    # Step 5: Evaluate ANN baseline
    ann_acc = evaluate_ann(ann_model, test_loader, device)
    print(f"  ANN accuracy (reference): {ann_acc*100:.2f}%")

    # Step 6: Evaluate converted SNN at multiple timestep budgets
    timestep_values = sorted(set([1, 4, 8, 16, 25, 50, 100, max_timesteps]))
    timestep_values = [t for t in timestep_values if t <= max_timesteps]

    snn_results = {}
    print(f"\n  {'T':>6}  {'SNN Acc':>10}  {'% of ANN':>10}  {'Gap':>8}")
    print(f"  {'-'*40}")

    for T in timestep_values:
        t0 = time.time()
        snn_acc = evaluate_converted_snn(snn_model, test_loader, device, T)
        elapsed = time.time() - t0
        pct_of_ann = (snn_acc / ann_acc * 100) if ann_acc > 0 else 0
        gap = ann_acc - snn_acc

        snn_results[T] = {
            "accuracy": snn_acc,
            "pct_of_ann": pct_of_ann,
            "gap_pp": gap * 100,
            "eval_time_s": round(elapsed, 2),
        }
        print(f"  {T:>6}  {snn_acc*100:>9.2f}%  {pct_of_ann:>9.1f}%  {gap*100:>+7.2f}pp")

    # Find convergence point (first T where SNN >= 95% of ANN)
    convergence_T = None
    for T in timestep_values:
        if snn_results[T]["pct_of_ann"] >= 95.0:
            convergence_T = T
            break

    result = {
        "fold": fold,
        "percentile": percentile,
        "max_timesteps": max_timesteps,
        "thresholds": thresholds,
        "ann_accuracy": ann_acc,
        "timestep_values": timestep_values,
        "snn_results": {str(k): v for k, v in snn_results.items()},
        "convergence_T_95pct": convergence_T,
    }

    if convergence_T:
        print(f"\n  Convergence: SNN reaches 95% of ANN at T={convergence_T}")
    else:
        print(f"\n  SNN did not reach 95% of ANN within T={max_timesteps}")

    # Clean up
    del ann_model, snn_model, recorder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="ANN-to-SNN conversion with threshold calibration"
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Specific fold (1-5). If omitted, runs all 5 folds.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument(
        "--percentile", type=float, default=99.9,
        help="Activation percentile for threshold calibration (default: 99.9)",
    )
    parser.add_argument(
        "--max-timesteps", type=int, default=100,
        help="Maximum number of timesteps to evaluate (default: 100)",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Device: {device}")

    download_esc50()

    folds = [args.fold] if args.fold else list(range(1, NUM_FOLDS + 1))

    out_dir = RESULTS_DIR / "experiments" / "ann_to_snn_conversion"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for fold in folds:
        result = convert_and_evaluate_fold(fold, device, args.percentile, args.max_timesteps)
        if result is not None:
            all_results.append(result)
            # Save per-fold result
            with open(out_dir / f"result_fold{fold}.json", "w") as f:
                json.dump(result, f, indent=2)

    if not all_results:
        print("No results produced. Check that ANN models exist in results/ann/none/")
        return

    # Aggregate across folds
    print(f"\n{'='*60}")
    print(f"  ANN-to-SNN Conversion Summary ({len(all_results)} folds)")
    print(f"{'='*60}")

    # All timestep values (union across folds)
    all_Ts = sorted(set(
        T for r in all_results for T in r["timestep_values"]
    ))

    ann_accs = [r["ann_accuracy"] for r in all_results]
    print(f"\n  ANN reference: {np.mean(ann_accs)*100:.2f}% +/- {np.std(ann_accs)*100:.2f}%")

    print(f"\n  {'T':>6}  {'SNN mean':>10}  {'SNN std':>9}  {'% of ANN':>10}  {'Gap':>8}")
    print(f"  {'-'*50}")

    aggregate = {}
    for T in all_Ts:
        accs = []
        for r in all_results:
            key = str(T)
            if key in r["snn_results"]:
                accs.append(r["snn_results"][key]["accuracy"])
        if accs:
            mean_acc = float(np.mean(accs))
            std_acc = float(np.std(accs))
            mean_ann = float(np.mean(ann_accs))
            pct = (mean_acc / mean_ann * 100) if mean_ann > 0 else 0
            gap = mean_ann - mean_acc
            aggregate[T] = {
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "pct_of_ann": pct,
                "gap_pp": gap * 100,
            }
            print(f"  {T:>6}  {mean_acc*100:>9.2f}%  {std_acc*100:>8.2f}%  "
                  f"{pct:>9.1f}%  {gap*100:>+7.2f}pp")

    # Convergence analysis
    conv_Ts = [r["convergence_T_95pct"] for r in all_results if r["convergence_T_95pct"] is not None]
    if conv_Ts:
        print(f"\n  Convergence to 95% of ANN: T={np.mean(conv_Ts):.1f} (mean), "
              f"T={max(conv_Ts)} (worst-case)")
    else:
        print(f"\n  No fold reached 95% of ANN within the timestep budget.")

    # Threshold analysis
    print(f"\n  Threshold statistics across folds (percentile={args.percentile}):")
    for layer_name in ["conv1_bn", "conv2_bn", "fc1", "fc2"]:
        threshs = [r["thresholds"].get(layer_name, 0) for r in all_results]
        print(f"    {layer_name}: {np.mean(threshs):.4f} +/- {np.std(threshs):.4f}")

    # Save summary
    summary = {
        "experiment": "ann_to_snn_conversion",
        "percentile": args.percentile,
        "max_timesteps": args.max_timesteps,
        "ann_mean_accuracy": float(np.mean(ann_accs)),
        "ann_std_accuracy": float(np.std(ann_accs)),
        "timestep_values": all_Ts,
        "aggregate": {str(k): v for k, v in aggregate.items()},
        "convergence_Ts": conv_Ts,
        "per_fold": [
            {
                "fold": r["fold"],
                "ann_accuracy": r["ann_accuracy"],
                "thresholds": r["thresholds"],
                "convergence_T_95pct": r["convergence_T_95pct"],
                "snn_results": r["snn_results"],
            }
            for r in all_results
        ],
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_dir / 'summary.json'}")

    # Plot accuracy vs timesteps curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Per-fold curves
        for r in all_results:
            Ts = r["timestep_values"]
            accs = [r["snn_results"][str(T)]["accuracy"] * 100 for T in Ts]
            ax.plot(Ts, accs, "o--", alpha=0.3, markersize=4, color="tab:blue")

        # Mean curve
        mean_Ts = sorted(aggregate.keys())
        mean_accs_plot = [aggregate[T]["mean_accuracy"] * 100 for T in mean_Ts]
        mean_stds_plot = [aggregate[T]["std_accuracy"] * 100 for T in mean_Ts]
        ax.errorbar(
            mean_Ts, mean_accs_plot, yerr=mean_stds_plot,
            marker="o", capsize=4, linewidth=2.5, color="tab:blue",
            label="Converted SNN (mean)", zorder=5,
        )

        # ANN reference line
        ann_mean = float(np.mean(ann_accs)) * 100
        ann_std = float(np.std(ann_accs)) * 100
        ax.axhline(y=ann_mean, color="tab:red", linestyle="--", linewidth=2,
                    label=f"ANN reference ({ann_mean:.1f}%)")
        ax.fill_between(
            [min(mean_Ts), max(mean_Ts)],
            ann_mean - ann_std, ann_mean + ann_std,
            alpha=0.1, color="tab:red",
        )

        # 95% threshold
        ax.axhline(y=ann_mean * 0.95, color="tab:green", linestyle=":",
                    linewidth=1.5, alpha=0.7,
                    label=f"95% of ANN ({ann_mean * 0.95:.1f}%)")

        ax.set_xlabel("Number of Timesteps (T)", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("ANN-to-SNN Conversion: Accuracy vs Timesteps", fontsize=13)
        ax.legend(fontsize=10, loc="lower right")
