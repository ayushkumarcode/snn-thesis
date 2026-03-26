"""
hybrid_ann_snn.py -- ANN-to-SNN hybrid initialization with fine-tuning.

Research question: Can transferring learned ANN weights to an enhanced SNN
(with learnable beta, threshold, dropout, and spike_rate_escape surrogate)
close the ANN-SNN accuracy gap?

Method:
  - Load trained ANN weights from results/ann/none/best_fold{fold}.pt
  - Map ANN conv/bn/fc layers to corresponding SNN layers
  - Create EnhancedSpikingCNN with learn_beta=True, learn_threshold=True,
    Dropout(0.3), spike_rate_escape surrogate gradient
  - Fine-tune for 20 epochs at lower learning rate (1e-4)
  - 5-fold CV, save results to results/experiments/hybrid_ann_snn/

Weight mapping (ConvANN -> SpikingCNN):
  features.0  (Conv2d)     -> conv1
  features.1  (BatchNorm)  -> bn1
  features.4  (Conv2d)     -> conv2
  features.5  (BatchNorm)  -> bn2
  classifier.0 (Linear)    -> fc1
  classifier.3 (Linear)    -> fc2

Usage:
  cd snn-esc50
  source .venv/bin/activate
  python experiments/hybrid_ann_snn.py
  python experiments/hybrid_ann_snn.py --fold 1 --epochs 20
  python experiments/hybrid_ann_snn.py --fold 0  # run all 5 folds
"""

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from src.config import (
    NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    BATCH_SIZE, WEIGHT_DECAY, PATIENCE, NUM_FOLDS,
    RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct
from src.models.ann_model import ConvANN


# ============================================================
# Enhanced SNN with learnable beta, threshold, and dropout
# ============================================================

class EnhancedSpikingCNN(nn.Module):
    """SpikingCNN with learnable membrane decay (beta), learnable threshold,
    dropout regularization, and spike_rate_escape surrogate gradient.

    Designed for ANN->SNN weight transfer: same conv/bn/fc dimensions as
    ConvANN, but with LIF neurons and temporal dynamics.

    Args:
        num_classes: Number of output classes.
        beta: Initial membrane potential decay rate.
        num_steps: Number of simulation timesteps.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        beta: float = BETA,
        num_steps: int = NUM_STEPS,
    ):
        super().__init__()
        self.num_steps = num_steps

        spike_grad = surrogate.spike_rate_escape(beta=1.0, slope=25)

        # -- Convolutional feature extraction --
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        # After two MaxPool2d(2) on input (64, 216): (16, 54)
        # AvgPool2d(4,6) on (16,54) -> (4, 9)
        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # -- Fully connected classifier with dropout --
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.dropout = nn.Dropout(0.3)
        self.lif3 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over all timesteps.

        Args:
            x: Input of shape (num_steps, batch, 1, n_mels, time_frames).

        Returns:
            spk_out: Output spikes, shape (num_steps, batch, num_classes).
            mem_out: Output membrane potentials, shape (num_steps, batch, num_classes).
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_out_rec = []
        mem_out_rec = []

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

            # FC block 1 with dropout
            cur3 = self.dropout(self.fc1(flat))
            spk3, mem3 = self.lif3(cur3, mem3)

            # FC block 2 (output)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)

        return torch.stack(spk_out_rec), torch.stack(mem_out_rec)


# ============================================================
# Weight transfer: ANN -> SNN
# ============================================================

def transfer_ann_weights(ann_model: ConvANN, snn_model: EnhancedSpikingCNN) -> None:
    """Transfer trained ANN weights to the enhanced SNN model (in-place).

    Mapping:
      ConvANN.features.0  (Conv2d)     -> EnhancedSpikingCNN.conv1
      ConvANN.features.1  (BatchNorm)  -> EnhancedSpikingCNN.bn1
      ConvANN.features.4  (Conv2d)     -> EnhancedSpikingCNN.conv2
      ConvANN.features.5  (BatchNorm)  -> EnhancedSpikingCNN.bn2
      ConvANN.classifier.0 (Linear)    -> EnhancedSpikingCNN.fc1
      ConvANN.classifier.3 (Linear)    -> EnhancedSpikingCNN.fc2
    """
    mapping = {
        "features.0": "conv1",
        "features.1": "bn1",
        "features.4": "conv2",
        "features.5": "bn2",
        "classifier.0": "fc1",
        "classifier.3": "fc2",
    }

    ann_sd = ann_model.state_dict()
    snn_sd = snn_model.state_dict()

    transferred = 0
    for ann_prefix, snn_prefix in mapping.items():
        for ann_key, ann_val in ann_sd.items():
            if ann_key.startswith(ann_prefix + "."):
                suffix = ann_key[len(ann_prefix):]
                snn_key = snn_prefix + suffix
                if snn_key in snn_sd:
                    snn_sd[snn_key] = ann_val.clone()
                    transferred += 1

    snn_model.load_state_dict(snn_sd)
    print(f"  Transferred {transferred} parameter tensors from ANN to SNN")


# ============================================================
# Training / evaluation
# ============================================================

def train_epoch(model, loader, optimizer, device):
    """Train SNN for one epoch with per-timestep CE loss (standard snnTorch)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data, num_steps=model.num_steps)

        optimizer.zero_grad()
        spk_out, mem_out = model(spk_input)

        # Accumulate loss across all timesteps
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
def eval_model(model, loader, device):
    """Evaluate SNN on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data, num_steps=model.num_steps)

        spk_out, mem_out = model(spk_input)

        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss += criterion(mem_out[step], targets)
        total_loss += loss.item()

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


# ============================================================
# Single fold training
# ============================================================

def train_fold(fold: int, device, epochs: int, patience: int):
    """Train hybrid ANN->SNN on a single fold.

    Returns:
        Dict with fold results.
    """
    print(f"\n{'='*60}")
    print(f"  Hybrid ANN->SNN | Fold {fold}/5")
    print(f"  Device: {device} | Epochs: {epochs} | Patience: {patience}")
    print(f"{'='*60}")

    # Check ANN weights exist
    ann_path = RESULTS_DIR / "ann" / "none" / f"best_fold{fold}.pt"
    if not ann_path.exists():
        print(f"FATAL: ANN weights not found: {ann_path}")
        sys.exit(1)

    # Load ANN and transfer weights
    ann_model = ConvANN()
    ann_model.load_state_dict(
        torch.load(ann_path, map_location="cpu", weights_only=True)
    )
    print(f"  Loaded ANN: {ann_path}")

    snn_model = EnhancedSpikingCNN().to(device)
    transfer_ann_weights(ann_model, snn_model)
    snn_model = snn_model.to(device)

    # Data
    train_loader, test_loader = get_fold_dataloaders(fold, batch_size=BATCH_SIZE)

    # Optimizer with lower learning rate for fine-tuning
    optimizer = torch.optim.Adam(
        snn_model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # Evaluate before fine-tuning (transfer-only accuracy)
    _, transfer_acc = eval_model(snn_model, test_loader, device)
    print(f"  Transfer-only accuracy (before fine-tuning): {transfer_acc:.4f}")

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
    }

    out_dir = RESULTS_DIR / "experiments" / "hybrid_ann_snn"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(snn_model, train_loader, optimizer, device)
        te_loss, te_acc = eval_model(snn_model, test_loader, device)
        scheduler.step(te_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(snn_model.state_dict(), out_dir / f"best_fold{fold}.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"Train: {tr_acc:.4f} | Test: {te_acc:.4f} | "
                f"Best: {best_acc:.4f} (ep {best_epoch}) | {elapsed:.0f}s"
            )

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    print(f"  Fold {fold} done in {elapsed:.1f}s | Best: {best_acc:.4f}")

    result = {
        "fold": fold,
        "method": "hybrid_ann_snn",
        "transfer_only_acc": transfer_acc,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "time_seconds": round(elapsed, 1),
        "lr": 1e-4,
        "epochs_max": epochs,
        "patience": patience,
        "features": ["learn_beta", "learn_threshold", "dropout_0.3", "spike_rate_escape"],
        "history": history,
    }

    with open(out_dir / f"result_fold{fold}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid ANN->SNN weight transfer + fine-tuning"
    )
    parser.add_argument("--fold", type=int, default=0,
                        help="Fold to train (1-5). 0 = all folds (default: 0)")
    parser.add_argument("--device", default=None, help="Device (default: auto)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Max fine-tuning epochs (default: 20)")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help=f"Early stopping patience (default: {PATIENCE})")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Device: {device}")

    download_esc50()

    out_dir = RESULTS_DIR / "experiments" / "hybrid_ann_snn"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.fold >= 1:
        # Single fold
        train_fold(args.fold, device, args.epochs, args.patience)
    else:
        # All 5 folds
        results = []
        for fold in range(1, NUM_FOLDS + 1):
            result = train_fold(fold, device, args.epochs, args.patience)
            results.append(result)

        # Summary
        accs = [r["best_acc"] for r in results]
        transfer_accs = [r["transfer_only_acc"] for r in results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_transfer = np.mean(transfer_accs)
        std_transfer = np.std(transfer_accs)

        print(f"\n{'='*60}")
        print(f"  HYBRID ANN->SNN 5-FOLD SUMMARY")
        print(f"{'='*60}")
        print(f"  Transfer-only (no fine-tune): {mean_transfer:.4f} +/- {std_transfer:.4f}")
        print(f"    Per-fold: {[f'{a:.4f}' for a in transfer_accs]}")
        print(f"  After fine-tuning:            {mean_acc:.4f} +/- {std_acc:.4f}")
        print(f"    Per-fold: {[f'{a:.4f}' for a in accs]}")
        print(f"  Baseline SNN (direct):        0.4715 +/- 0.0450")
        print(f"  Baseline ANN:                 0.6385 +/- 0.0307")
        improvement = mean_acc - 0.4715
        print(f"  Improvement over baseline SNN: {improvement:+.4f}")
        print(f"{'='*60}")

        summary = {
            "method": "hybrid_ann_snn",
            "fold_accuracies": accs,
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc),
