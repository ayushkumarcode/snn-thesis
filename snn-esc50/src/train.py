"""
Training loop for both SNN and ANN models on ESC-50.

Usage:
    # Train SNN with rate encoding, all 5 folds
    python -m src.train --model snn --encoding rate

    # Train ANN baseline, all 5 folds
    python -m src.train --model ann

    # Train SNN on specific fold
    python -m src.train --model snn --encoding rate --fold 1
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from snntorch import surrogate

from src.config import (
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    NUM_STEPS, NUM_FOLDS, RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import get_encoder
from src.models.snn_model import SpikingCNN
from src.models.ann_model import ConvANN

# Surrogate gradient registry (for ablation study)
SURROGATE_REGISTRY = {
    "fast_sigmoid": lambda: surrogate.fast_sigmoid(slope=25),
    "atan":         lambda: surrogate.atan(alpha=2.0),
    "sigmoid":      lambda: surrogate.sigmoid(slope=25),
    "ste":          lambda: surrogate.straight_through_estimator(),
    "triangular":   lambda: surrogate.triangular(),
    "sre":          lambda: surrogate.spike_rate_escape(beta=1, slope=25),
    "lso":          lambda: surrogate.LSO(slope=0.1),
    "sfs":          lambda: surrogate.SFS(slope=25),
}


def train_snn_epoch(model, loader, optimizer, encoder, device):
    """Train SNN for one epoch.

    Uses per-timestep cross-entropy on membrane potentials, summed
    across all timesteps. This is the standard snnTorch approach
    (Tutorial 5) and gives better gradient flow than rate-based loss.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)

        spk_input = encoder(data).to(device)

        optimizer.zero_grad()
        spk_out, mem_out = model(spk_input)

        # Accumulate loss across all timesteps (standard snnTorch approach)
        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss += criterion(mem_out[step], targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Predict using summed membrane potentials
        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def eval_snn(model, loader, encoder, device):
    """Evaluate SNN on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encoder(data).to(device)

        spk_out, mem_out = model(spk_input)

        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss += criterion(mem_out[step], targets)
        total_loss += loss.item()

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        all_preds.extend(predicted.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_targets


def train_ann_epoch(model, loader, optimizer, criterion, device):
    """Train ANN for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = logits.argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def eval_ann(model, loader, criterion, device):
    """Evaluate ANN on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)

        logits = model(data)
        loss = criterion(logits, targets)
        total_loss += loss.item()

        predicted = logits.argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        all_preds.extend(predicted.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy, all_preds, all_targets


def train_fold(model_type: str, fold: int, encoding: str = "rate",
               device: torch.device = None, augment: bool = False,
               num_epochs: int = None, run_suffix: str = "",
               spike_grad_name: str = "fast_sigmoid"):
    """Train and evaluate on a single fold.

    Args:
        model_type: "snn" or "ann".
        fold: Test fold number (1-5).
        encoding: Encoding method (only used for SNN).
        device: torch device.
        augment: If True, apply SpecAugment + TimeShift to training data.
        num_epochs: Override NUM_EPOCHS from config if provided.
        run_suffix: Appended to encoding name in output directory (e.g. "_aug").

    Returns:
        Dict with fold results.
    """
    if device is None:
        device = get_device()

    epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    out_encoding = f"{encoding}{run_suffix}" if run_suffix else encoding

    print(f"\n{'='*60}")
    print(f"  Fold {fold}/5 | Model: {model_type.upper()} | Encoding: {out_encoding}")
    print(f"  Device: {device} | Augment: {augment} | Epochs: {epochs}")
    print(f"{'='*60}")

    train_loader, test_loader = get_fold_dataloaders(fold, BATCH_SIZE, augment=augment)

    if model_type == "snn":
        sg = SURROGATE_REGISTRY.get(spike_grad_name, SURROGATE_REGISTRY["fast_sigmoid"])()
        model = SpikingCNN(spike_grad=sg).to(device)
        encoder = get_encoder(encoding)
    else:
        model = ConvANN().to(device)
        encoder = None

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    criterion = nn.CrossEntropyLoss() if model_type == "ann" else None

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        if model_type == "snn":
            train_loss, train_acc = train_snn_epoch(
                model, train_loader, optimizer, encoder, device,
            )
            test_loss, test_acc, preds, targets = eval_snn(
                model, test_loader, encoder, device,
            )
        else:
            train_loss, train_acc = train_ann_epoch(
                model, train_loader, optimizer, criterion, device,
            )
            test_loss, test_acc, preds, targets = eval_ann(
                model, test_loader, criterion, device,
            )

        scheduler.step(test_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            save_dir = RESULTS_DIR / model_type / encoding
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / f"best_fold{fold}.pt")
            # Save predictions for later analysis
            torch.save(
                {"preds": preds, "targets": targets},
                save_dir / f"preds_fold{fold}.pt",
            )
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} | "
                f"Best: {best_acc:.4f} (ep {best_epoch})"
            )

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    print(f"  Fold {fold} done in {elapsed:.1f}s | Best accuracy: {best_acc:.4f}")

    result = {
        "fold": fold,
        "model": model_type,
        "encoding": encoding,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "time_seconds": elapsed,
        "history": history,
    }

    # Save fold results
    save_dir = RESULTS_DIR / model_type / encoding
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"result_fold{fold}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def run_all_folds(model_type: str, encoding: str = "rate",
                  augment: bool = False, num_epochs: int = None,
                  run_suffix: str = "", spike_grad_name: str = "fast_sigmoid"):
    """Run 5-fold cross-validation."""
    device = get_device()
    results = []

    for fold in range(1, NUM_FOLDS + 1):
        result = train_fold(model_type, fold, encoding, device,
                            augment=augment, num_epochs=num_epochs,
                            run_suffix=run_suffix,
                            spike_grad_name=spike_grad_name)
        results.append(result)

    accs = [r["best_acc"] for r in results]
    mean_acc = sum(accs) / len(accs)
    std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5

    print(f"\n{'='*60}")
    print(f"  5-Fold CV Results | {model_type.upper()} | {encoding}")
    print(f"  Per-fold: {[f'{a:.4f}' for a in accs]}")
    print(f"  Mean: {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"{'='*60}")

    summary = {
        "model": model_type,
        "encoding": encoding,
        "fold_accuracies": accs,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
    }

    save_dir = RESULTS_DIR / model_type / encoding
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train SNN/ANN on ESC-50")
    parser.add_argument("--model", choices=["snn", "ann"], required=True)
    parser.add_argument("--encoding", default="rate",
                        choices=["rate", "delta", "latency", "direct", "burst", "phase"])
    parser.add_argument("--fold", type=int, default=None,
                        help="Specific fold (1-5). If omitted, runs all 5.")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Apply SpecAugment + TimeShift to training data.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs (default: config NUM_EPOCHS).")
    parser.add_argument("--run-suffix", default="",
                        help="Suffix appended to encoding name in output dir (e.g. '_aug').")
    parser.add_argument("--spike-grad", default="fast_sigmoid",
                        choices=list(SURROGATE_REGISTRY.keys()),
                        help="Surrogate gradient function for SNN training (default: fast_sigmoid).")
    args = parser.parse_args()

    download_esc50()

    if args.model == "ann":
        args.encoding = "none"

    if args.fold:
        train_fold(args.model, args.fold, args.encoding,
                   augment=args.augment, num_epochs=args.epochs,
                   run_suffix=args.run_suffix,
                   spike_grad_name=args.spike_grad)
    else:
        run_all_folds(args.model, args.encoding,
                      augment=args.augment, num_epochs=args.epochs,
                      run_suffix=args.run_suffix,
                      spike_grad_name=args.spike_grad)


if __name__ == "__main__":
    main()
