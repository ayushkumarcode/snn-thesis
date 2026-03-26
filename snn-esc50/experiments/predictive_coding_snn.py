"""
predictive_coding_snn.py -- Predictive Coding SNN experiment.

Suppress predictable spikes and transmit only prediction errors.
Based on "Predictive Coding Light" (Nature Communications, 2025).

Architecture: Same conv backbone, but after the hidden FC layer (lif3),
a prediction module tries to predict the *next* timestep's hidden spikes
from the *current* timestep's spikes. Only the prediction ERROR (residual)
is transmitted to FC2. This naturally reduces redundant spikes and acts
as a temporal regulariser on the small ESC-50 dataset.

Implementation details:
  - predict_fc = Linear(256, 256) predicts spk3[t] from spk3[t-1]
  - error[t] = spk3[t] - predict_fc(spk3[t-1])  (for t >= 1)
  - error[0] = spk3[0]  (no prediction available at first step)
  - FC2 receives error instead of spk3
  - Loss = CE_loss + lambda_pred * MSE(predict_fc(spk3[t-1]), spk3[t])
  - The MSE term encourages the network to produce predictable hidden
    representations, compressing redundancy across timesteps.

Also uses: learn_beta=True, Dropout(0.3), spike_rate_escape surrogate.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/predictive_coding_snn.py --fold 1
    python experiments/predictive_coding_snn.py              # all 5 folds
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_CLASSES, NUM_STEPS, BETA,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct


# ============================================================
# Predictive Coding SNN Model
# ============================================================

class PredictiveSNN(nn.Module):
    """SpikingCNN with predictive coding on the hidden FC layer.

    After lif3 produces hidden spikes (256-dim), a prediction module
    estimates spk3[t] from spk3[t-1]. Only the prediction error is
    sent to FC2. The prediction loss encourages temporal redundancy
    reduction -- fewer unique spikes need to be transmitted.
    """

    def __init__(self, num_classes=NUM_CLASSES, beta=BETA, num_steps=NUM_STEPS):
        super().__init__()
        self.num_steps = num_steps

        spike_grad = surrogate.spike_rate_escape(beta=1, slope=25)

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # FC block 1
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        # Predictive coding: predict next hidden spike from current
        self.predict_fc = nn.Linear(256, 256)

        # Dropout on hidden spikes / errors
        self.dropout = nn.Dropout(0.3)

        # FC block 2 (output) -- receives prediction errors
        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

    def forward(self, x):
        """Forward pass returning outputs + prediction errors for loss.

        Returns:
            spk_out: (num_steps, batch, num_classes)
            mem_out: (num_steps, batch, num_classes)
            pred_errors_mse: scalar -- mean squared prediction error (for loss)
            spike_stats: dict with spike counts for analysis
        """
        device = x.device

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_out_rec = []
        mem_out_rec = []

        # Track spike counts for energy comparison
        total_original_spikes = 0.0
        total_error_spikes = 0.0
        pred_error_sum = torch.zeros(1, device=device)
        pred_count = 0

        spk3_prev = None  # previous timestep hidden spikes

        for step in range(self.num_steps):
            x_t = x[step]

            # Conv block 1
            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, mem1 = self.lif1(cur1, mem1)

            # Conv block 2
            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            # Pool + flatten
            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            # FC block 1 -- produce hidden spikes
            cur3 = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3)

            # Track original spike count
            total_original_spikes += spk3.sum().item()

            # Predictive coding: compute prediction error
            if spk3_prev is not None:
                # Predict current spikes from previous
                predicted = self.predict_fc(spk3_prev)
                error = spk3 - predicted

                # Accumulate MSE for prediction loss
                pred_error_sum = pred_error_sum + ((error) ** 2).mean()
                pred_count += 1

                # Feed error to FC2 instead of raw spikes
                fc2_input = self.dropout(error)
            else:
                # First timestep: no prediction, use raw spikes
                fc2_input = self.dropout(spk3)

            # Track error "spikes" (non-zero elements)
            total_error_spikes += (fc2_input.abs() > 0.01).float().sum().item()

            # Save for next step prediction
            spk3_prev = spk3.detach()  # detach to avoid backprop through time for prediction target

            # FC block 2 (output)
            cur4 = self.fc2(fc2_input)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)

        # Average MSE prediction error
        if pred_count > 0:
            pred_errors_mse = pred_error_sum / pred_count
        else:
            pred_errors_mse = torch.zeros(1, device=device)

        spike_stats = {
            "original_spikes": total_original_spikes,
            "error_spikes": total_error_spikes,
        }

        return (torch.stack(spk_out_rec), torch.stack(mem_out_rec),
                pred_errors_mse, spike_stats)


# ============================================================
# Training / Evaluation
# ============================================================

def train_epoch(model, loader, optimizer, device, lambda_pred):
    """Train for one epoch with predictive coding loss."""
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_pred = 0.0
    correct = 0
    total = 0
    total_orig_spikes = 0.0
    total_err_spikes = 0.0

    ce_criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spike_input = encode_direct(data).to(device)

        optimizer.zero_grad()
        spk_out, mem_out, pred_mse, spike_stats = model(spike_input)

        # Per-timestep CE loss (standard snnTorch approach)
        ce_loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            ce_loss = ce_loss + ce_criterion(mem_out[step], targets)

        # Combined loss
        pred_loss = lambda_pred * pred_mse
        loss = ce_loss + pred_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_pred += pred_loss.item()

        # Predict using summed membrane potentials
        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        total_orig_spikes += spike_stats["original_spikes"]
        total_err_spikes += spike_stats["error_spikes"]

    n = len(loader)
    stats = {
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "pred_loss": total_pred / n,
        "accuracy": correct / total,
        "avg_original_spikes": total_orig_spikes / total,
        "avg_error_spikes": total_err_spikes / total,
    }
    return stats


@torch.no_grad()
def eval_model(model, loader, device, lambda_pred):
    """Evaluate the predictive coding SNN."""
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_pred = 0.0
    correct = 0
    total = 0
    total_orig_spikes = 0.0
    total_err_spikes = 0.0

    ce_criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spike_input = encode_direct(data).to(device)

        spk_out, mem_out, pred_mse, spike_stats = model(spike_input)

        ce_loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            ce_loss = ce_loss + ce_criterion(mem_out[step], targets)

        pred_loss = lambda_pred * pred_mse
        loss = ce_loss + pred_loss

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_pred += pred_loss.item()

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        total_orig_spikes += spike_stats["original_spikes"]
        total_err_spikes += spike_stats["error_spikes"]

    n = len(loader)
    stats = {
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "pred_loss": total_pred / n,
        "accuracy": correct / total,
        "avg_original_spikes": total_orig_spikes / total,
        "avg_error_spikes": total_err_spikes / total,
    }
    return stats


def train_fold(fold, device, epochs, patience, lambda_pred):
    """Train predictive coding SNN on one fold."""
    torch.manual_seed(42 + fold)

    train_loader, test_loader = get_fold_dataloaders(fold, batch_size=BATCH_SIZE)
    model = PredictiveSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
               "train_ce": [], "train_pred": [], "test_ce": [], "test_pred": [],
               "train_orig_spikes": [], "train_err_spikes": [],
               "test_orig_spikes": [], "test_err_spikes": []}

    best_model_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device, lambda_pred)
        te = eval_model(model, test_loader, device, lambda_pred)
        scheduler.step(te["loss"])

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["accuracy"])
        history["train_ce"].append(tr["ce_loss"])
        history["train_pred"].append(tr["pred_loss"])
        history["train_orig_spikes"].append(tr["avg_original_spikes"])
        history["train_err_spikes"].append(tr["avg_error_spikes"])
        history["test_loss"].append(te["loss"])
        history["test_acc"].append(te["accuracy"])
        history["test_ce"].append(te["ce_loss"])
        history["test_pred"].append(te["pred_loss"])
        history["test_orig_spikes"].append(te["avg_original_spikes"])
        history["test_err_spikes"].append(te["avg_error_spikes"])

        if te["accuracy"] > best_acc:
            best_acc = te["accuracy"]
            best_epoch = epoch
            no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            spike_reduction = (1 - te["avg_error_spikes"] / max(te["avg_original_spikes"], 1)) * 100
            print(f"  Ep {epoch:3d}/{epochs} | "
                  f"tr={tr['accuracy']:.3f} te={te['accuracy']:.3f} "
                  f"best={best_acc:.3f} | "
                  f"CE={te['ce_loss']:.2f} pred={te['pred_loss']:.4f} | "
                  f"spike_red={spike_reduction:.1f}% | "
                  f"{elapsed:.0f}s")

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}, best={best_acc:.4f}")
            break

    elapsed = time.time() - t0

    # Final eval with best model for spike stats
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
    final_te = eval_model(model, test_loader, device, lambda_pred)

    result = {
        "fold": fold,
        "lambda_pred": lambda_pred,
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "time_seconds": elapsed,
        "final_original_spikes_per_sample": final_te["avg_original_spikes"],
        "final_error_spikes_per_sample": final_te["avg_error_spikes"],
        "spike_reduction_pct": (1 - final_te["avg_error_spikes"] / max(final_te["avg_original_spikes"], 1)) * 100,
        "history": history,
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Predictive Coding SNN -- transmit only prediction errors")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold (1-5). If omitted, runs all 5.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/mps/cpu). Auto-detect if omitted.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Max epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--lambda-pred", type=float, default=0.1,
                        help="Weight for prediction MSE loss (default: 0.1)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Device: {device}")

    download_esc50()

    out_dir = RESULTS_DIR / "experiments" / "predictive_coding_snn"
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = [args.fold] if args.fold else list(range(1, 6))

    print(f"\n{'='*70}")
    print(f"  PREDICTIVE CODING SNN EXPERIMENT")
    print(f"  Folds: {folds} | lambda_pred: {args.lambda_pred} | epochs: {args.epochs}")
    print(f"{'='*70}")

    all_results = []

    for fold in folds:
        print(f"\n--- Fold {fold}/5 ---")
        result = train_fold(fold, device, args.epochs, PATIENCE, args.lambda_pred)
        all_results.append(result)

        # Save per-fold result
        fold_file = out_dir / f"fold{fold}_lambda{args.lambda_pred}.json"
        with open(fold_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Fold {fold}: {result['best_accuracy']*100:.2f}% "
              f"(spike reduction: {result['spike_reduction_pct']:.1f}%)")

    # Summary
    accs = [r["best_accuracy"] for r in all_results]
    spike_reds = [r["spike_reduction_pct"] for r in all_results]
    mean_acc = sum(accs) / len(accs)
    std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5
    mean_red = sum(spike_reds) / len(spike_reds)

    summary = {
        "experiment": "predictive_coding_snn",
        "lambda_pred": args.lambda_pred,
        "folds": folds,
        "fold_accuracies": accs,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "fold_spike_reductions_pct": spike_reds,
        "mean_spike_reduction_pct": mean_red,
        "baseline_comparison": {
            "baseline_direct_5fold": 0.4715,
            "note": "Baseline SNN (direct encoding, 5-fold): 47.15% +/- 4.50%"
        }
    }

    summary_file = out_dir / f"summary_lambda{args.lambda_pred}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  PREDICTIVE CODING SNN -- RESULTS")
    print(f"{'='*70}")
    print(f"  Per-fold accuracy: {[f'{a*100:.2f}%' for a in accs]}")
    print(f"  Mean accuracy:     {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
    print(f"  Mean spike reduction: {mean_red:.1f}%")
    print(f"  Baseline (direct): 47.15% +/- 4.50%")
    print(f"  Saved to: {summary_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
