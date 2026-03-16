"""
spinnaker_option_a.py -- SpiNNaker-aware SNN retraining with MaxPool.

Root cause of FC1 cancellation on SpiNNaker (documented in DECISIONS.md #15):
  - AvgPool applied to binary spikes produces fractional outputs in [0,1]
  - FC1 weights (near-zero-mean) cannot distinguish fractional inputs
  - Result: net near-zero current -> zero hidden spikes

Fix: replace AvgPool2d(4,6) with MaxPool2d(4,6).
  - MaxPool on binary spikes: output is 0 or 1 (max of binary = binary)
  - FC1 inputs are now truly binary -> SpiNNaker-compatible
  - Same spatial output dimensions (4,9) -> same FC layer sizes

Optional: higher LIF threshold to reduce FC1 input density.

Usage:
    cd snn-esc50
    source .venv/bin/activate

    # Train fold 4 with MaxPool, threshold=1.0 (default)
    python experiments/spinnaker_option_a.py --fold 4

    # Train fold 4 with higher LIF threshold
    python experiments/spinnaker_option_a.py --fold 4 --threshold 2.0

    # Threshold sweep on fold 4
    python experiments/spinnaker_option_a.py --fold 4 --threshold-sweep

    # All 5 folds with best threshold
    python experiments/spinnaker_option_a.py --all-folds --threshold 1.5
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Make sure src imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import get_encoder


# ============================================================
# MaxPool SNN model (SpiNNaker-aware architecture)
# ============================================================

class SpikingCNN_MaxPool(nn.Module):
    """SpikingCNN with MaxPool instead of AvgPool for SpiNNaker compatibility.

    Key change: avg_pool -> max_pool (MaxPool2d(4,6) on binary spikes = binary output)
    FC dimensions unchanged: 64*4*9 = 2304 -> 256 -> 50

    Optional: higher LIF threshold to reduce FC1 input spike density.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        beta: float = BETA,
        num_steps: int = NUM_STEPS,
        n_mels: int = N_MELS,
        threshold: float = 1.0,
        spike_grad=None,
    ):
        super().__init__()
        self.num_steps = num_steps

        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, threshold=threshold)

        # KEY CHANGE: MaxPool instead of AvgPool
        # MaxPool2d(4,6) on (16,54) -> (4, 9) -- same dimensions as AvgPool
        # Binary in (spikes) -> binary out (max of 0/1 is 0/1)
        self.max_pool = nn.MaxPool2d(kernel_size=(4, 6))

        # FC: 64 * 4 * 9 = 2304 -> 256 (unchanged)
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_out_rec = []
        mem_out_rec = []
        spk3_rec = []  # record FC1 output spikes for sparsity analysis

        for step in range(self.num_steps):
            x_t = x[step]

            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            # MaxPool instead of AvgPool: binary -> binary
            pooled = self.max_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            cur3 = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)
            spk3_rec.append(flat)  # record FC1 inputs (should be binary)

        return (
            torch.stack(spk_out_rec),
            torch.stack(mem_out_rec),
            torch.stack(spk3_rec),  # (T, B, 2304) FC1 inputs
        )


# ============================================================
# Training helpers
# ============================================================

def train_epoch(model, loader, optimizer, encoder, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encoder(data).to(device)

        optimizer.zero_grad()
        spk_out, mem_out, _ = model(spk_input)

        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss += criterion(mem_out[step], targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_model(model, loader, encoder, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    all_fc1_inputs = []  # collect FC1 inputs for sparsity analysis

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encoder(data).to(device)

        spk_out, mem_out, fc1_inputs = model(spk_input)

        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss += criterion(mem_out[step], targets)
        total_loss += loss.item()

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        all_fc1_inputs.append(fc1_inputs.cpu())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    # Analyse FC1 input sparsity and binary-ness
    fc1_tensor = torch.cat(all_fc1_inputs, dim=1)  # (T, N_samples, 2304)
    sparsity = (fc1_tensor == 0).float().mean().item()
    # Check how binary the inputs are (should be exactly 0 or 1 with MaxPool)
    binary_fraction = ((fc1_tensor == 0) | (fc1_tensor == 1)).float().mean().item()
    mean_active = (fc1_tensor > 0).float().sum(dim=2).mean().item()

    return avg_loss, accuracy, {
        "fc1_sparsity": sparsity,
        "fc1_binary_fraction": binary_fraction,
        "fc1_mean_active_per_step": mean_active,
    }


def train_fold_option_a(fold, threshold, epochs, patience, device, encoder):
    """Train one fold with MaxPool architecture."""
    train_loader, test_loader = get_fold_dataloaders(fold, batch_size=BATCH_SIZE)

    model = SpikingCNN_MaxPool(threshold=threshold).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, encoder, device)
        te_loss, te_acc, fc1_stats = eval_model(model, test_loader, encoder, device)

        scheduler.step(te_loss)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), out_dir / f"best_fold{fold}.pt")
            # Save final FC1 stats from best epoch
            best_fc1_stats = fc1_stats
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Fold {fold} Ep {epoch:3d}/{epochs} | "
                f"tr={tr_acc:.3f} te={te_acc:.3f} | "
                f"FC1_active={fc1_stats['fc1_mean_active_per_step']:.1f}/2304 "
                f"binary={fc1_stats['fc1_binary_fraction']:.4f}"
            )

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    return best_acc, best_epoch, epoch, history, best_fc1_stats


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SpiNNaker Option A: MaxPool SNN")
    parser.add_argument("--fold", type=int, default=4, help="Fold to train (default: 4)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="LIF threshold (default: 1.0 = same as original)")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="Sweep thresholds {1.0, 1.5, 2.0, 3.0} on --fold")
    parser.add_argument("--all-folds", action="store_true",
                        help="Run all 5 folds (uses --threshold)")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs (default: 50)")
    parser.add_argument("--patience", type=int, default=10, help="Early stop patience")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    download_esc50()

    global out_dir
    out_dir = RESULTS_DIR / "snn" / "maxpool"
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = get_encoder("direct")  # direct encoding (best SNN result)

    if args.threshold_sweep:
        thresholds = [1.0, 1.5, 2.0, 3.0]
        print(f"\n=== Threshold Sweep on Fold {args.fold} ===")
        sweep_results = []
        for thresh in thresholds:
            print(f"\n--- Threshold = {thresh} ---")
            best_acc, best_ep, total_ep, hist, fc1_stats = train_fold_option_a(
                args.fold, thresh, args.epochs, args.patience, device, encoder
            )
            result = {
                "threshold": thresh,
                "best_acc": best_acc,
                "best_epoch": best_ep,
                "total_epochs": total_ep,
                "fc1_stats": fc1_stats,
            }
            sweep_results.append(result)
            print(f"  RESULT: threshold={thresh} acc={best_acc:.4f} "
                  f"FC1_active={fc1_stats['fc1_mean_active_per_step']:.1f}/2304")

        sweep_path = out_dir / f"threshold_sweep_fold{args.fold}.json"
        with open(sweep_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\nSweep saved: {sweep_path}")

        # Print summary table
        print("\n=== Sweep Summary ===")
        print(f"{'Threshold':>10} {'Acc':>8} {'FC1 Active':>12} {'Binary':>8}")
        for r in sweep_results:
            print(f"{r['threshold']:>10.1f} {r['best_acc']:>8.4f} "
                  f"{r['fc1_stats']['fc1_mean_active_per_step']:>12.1f} "
                  f"{r['fc1_stats']['fc1_binary_fraction']:>8.4f}")

    elif args.all_folds:
        print(f"\n=== Option A: All 5 Folds (threshold={args.threshold}) ===")
        fold_results = []
        for fold in range(1, 6):
            print(f"\n--- Fold {fold} ---")
            best_acc, best_ep, total_ep, hist, fc1_stats = train_fold_option_a(
                fold, args.threshold, args.epochs, args.patience, device, encoder
            )
            result = {
                "fold": fold,
                "threshold": args.threshold,
                "best_acc": best_acc,
                "best_epoch": best_ep,
                "total_epochs": total_ep,
                "fc1_stats": fc1_stats,
                "history": hist,
            }
            fold_results.append(result)
            out_path = out_dir / f"result_fold{fold}_thresh{args.threshold}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Fold {fold}: acc={best_acc:.4f} FC1_active="
                  f"{fc1_stats['fc1_mean_active_per_step']:.1f}/2304 "
                  f"binary={fc1_stats['fc1_binary_fraction']:.4f}")

        accs = [r["best_acc"] for r in fold_results]
        import numpy as np
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        summary = {
            "model": "snn_maxpool",
            "encoding": "direct",
            "threshold": args.threshold,
            "fold_accuracies": accs,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
        }
        summary_path = out_dir / f"summary_thresh{args.threshold}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n=== Final: {mean_acc:.4f} ± {std_acc:.4f} ===")

    else:
        # Single fold
        print(f"\n=== Option A: Fold {args.fold}, threshold={args.threshold} ===")
        best_acc, best_ep, total_ep, hist, fc1_stats = train_fold_option_a(
            args.fold, args.threshold, args.epochs, args.patience, device, encoder
        )

        result = {
            "fold": args.fold,
            "model": "snn_maxpool",
            "encoding": "direct",
            "threshold": args.threshold,
            "best_acc": best_acc,
            "best_epoch": best_ep,
            "total_epochs": total_ep,
            "fc1_stats": fc1_stats,
            "history": hist,
        }
        out_path = out_dir / f"result_fold{args.fold}_thresh{args.threshold}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n=== RESULT ===")
        print(f"  Best acc:         {best_acc:.4f} ({best_acc*100:.2f}%)")
        print(f"  Best epoch:       {best_ep}")
        print(f"  FC1 active/step:  {fc1_stats['fc1_mean_active_per_step']:.1f} / 2304")
        print(f"  FC1 binary frac:  {fc1_stats['fc1_binary_fraction']:.6f}")
        print(f"  FC1 sparsity:     {fc1_stats['fc1_sparsity']:.4f}")
        print(f"\nSpiNNaker suitability:")
        if fc1_stats["fc1_binary_fraction"] > 0.999:
            print("  ✅ FC1 inputs are binary (MaxPool working correctly)")
        else:
            print("  ❌ FC1 inputs are NOT fully binary — check pooling")
        if fc1_stats["fc1_mean_active_per_step"] < 500:
            print(f"  ✅ FC1 active {fc1_stats['fc1_mean_active_per_step']:.0f}/2304 "
                  f"< 500 threshold — SpiNNaker FC1 deployment feasible")
        else:
            print(f"  ⚠️  FC1 active {fc1_stats['fc1_mean_active_per_step']:.0f}/2304 "
                  f"— try higher threshold")
        print(f"\nResult saved: {out_path}")


if __name__ == "__main__":
    main()
