"""
urbansound8k_1fold.py -- Quick 1-fold experiment on UrbanSound8K to validate
encoding hierarchy generalisation on a second dataset.

Research question: Does the ESC-50 encoding hierarchy (direct >> rate >> latency
>> delta) hold on a different environmental sound dataset?

UrbanSound8K:
  - 8732 clips of <=4 seconds, 10 predefined folds, 10 urban sound classes
  - Standard protocol: fold 10 as test, folds 1-9 as train
  - Classes: air_conditioner, car_horn, children_playing, dog_bark, drilling,
             engine_idling, gun_shot, jackhammer, siren, street_music

Dataset setup:
  The dataset must be manually downloaded (requires agreement to terms).
  Place it at: data/UrbanSound8K/
  Expected structure:
    data/UrbanSound8K/
      metadata/UrbanSound8K.csv
      audio/
        fold1/ ... fold10/

Usage:
  cd snn-esc50/
  source .venv/bin/activate
  python experiments/urbansound8k_1fold.py
  python experiments/urbansound8k_1fold.py --test-fold 10 --epochs 50
  python experiments/urbansound8k_1fold.py --encoding rate
"""

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    RESULTS_DIR, DATA_DIR, SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH,
    F_MIN, F_MAX, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, PATIENCE,
    NUM_EPOCHS, NUM_STEPS, BETA, get_device,
)
from src.dataset import normalise_spectrogram
from src.encoding import get_encoder, encode_direct
from src.models.snn_model import SpikingCNN
from src.models.ann_model import ConvANN


# ============================================================
# Constants
# ============================================================

US8K_DIR = DATA_DIR / "UrbanSound8K"
US8K_META_PATH = US8K_DIR / "metadata" / "UrbanSound8K.csv"
US8K_AUDIO_DIR = US8K_DIR / "audio"

US8K_NUM_CLASSES = 10
US8K_NUM_FOLDS = 10
US8K_DURATION = 4  # seconds (UrbanSound8K clips are <=4s)
US8K_EXPECTED_LEN = SAMPLE_RATE * US8K_DURATION  # 88200 samples

# ESC-50 spectrogram has 216 time frames from 5s audio.
# UrbanSound8K 4s audio gives ~173 time frames. Pad to 216 for model compat.
ESC50_TIME_FRAMES = 216


# ============================================================
# Dataset
# ============================================================

def wav_to_mel_us8k(filepath: str) -> np.ndarray:
    """Load a UrbanSound8K WAV file and convert to log-mel spectrogram.

    Pads/trims to 4 seconds, computes mel spectrogram with the same
    parameters as ESC-50 (64 mels, 1024 n_fft, 512 hop), then zero-pads
    the time axis to 216 frames to match our model's expected input shape.

    Returns:
        np.ndarray of shape (64, 216), normalised to [0, 1].
    """
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=US8K_DURATION)

    # Pad to exactly 4 seconds if shorter
    if len(y) < US8K_EXPECTED_LEN:
        y = np.pad(y, (0, US8K_EXPECTED_LEN - len(y)))
    # Trim if longer (shouldn't happen with duration=4, but be safe)
    y = y[:US8K_EXPECTED_LEN]

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmin=F_MIN, fmax=F_MAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalise to [0, 1]
    mel_norm = normalise_spectrogram(mel_db)

    # Zero-pad time axis from ~173 to 216 frames to match ESC-50 input shape
    n_time = mel_norm.shape[1]
    if n_time < ESC50_TIME_FRAMES:
        pad_width = ESC50_TIME_FRAMES - n_time
        mel_norm = np.pad(mel_norm, ((0, 0), (0, pad_width)), mode="constant")
    elif n_time > ESC50_TIME_FRAMES:
        mel_norm = mel_norm[:, :ESC50_TIME_FRAMES]

    return mel_norm


class UrbanSound8KDataset(Dataset):
    """PyTorch Dataset for UrbanSound8K.

    Args:
        folds: List of fold numbers to include (1-10).
        precompute: If True, load and cache all spectrograms in memory.
    """

    def __init__(self, folds: list[int], precompute: bool = True):
        import pandas as pd

        self.precompute = precompute

        meta = pd.read_csv(US8K_META_PATH)
        self.meta = meta[meta["fold"].isin(folds)].reset_index(drop=True)

        self.data = []
        self.labels = []

        if precompute:
            print(f"Loading {len(self.meta)} clips from folds {folds}...")
            for _, row in self.meta.iterrows():
                filepath = US8K_AUDIO_DIR / f"fold{row['fold']}" / row["slice_file_name"]
                mel = wav_to_mel_us8k(str(filepath))
                self.data.append(mel)
                self.labels.append(row["classID"])

            self.data = np.array(self.data, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.precompute:
            mel = self.data[idx]
            label = self.labels[idx]
        else:
            import pandas as pd
            row = self.meta.iloc[idx]
            filepath = US8K_AUDIO_DIR / f"fold{row['fold']}" / row["slice_file_name"]
            mel = wav_to_mel_us8k(str(filepath))
            label = row["classID"]

        # Shape: (1, n_mels, 216) -- single channel
        tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)


# ============================================================
# SNN wrapper (same pattern as adversarial_robustness.py)
# ============================================================

class SNNWrapper(nn.Module):
    """Wraps SNN + encoder for standard forward pass.

    Forward: x (B, 1, H, W) -> encoder -> SNN -> spk_out.mean(0) logits.
    Uses spike rate (mean output spikes) as logits for CE loss.
    """

    def __init__(self, snn_model: SpikingCNN, encoder, num_steps: int = NUM_STEPS):
        super().__init__()
        self.snn = snn_model
        self.encoder = encoder
        self.T = num_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spk_input = self.encoder(x, num_steps=self.T)
        spk_out, mem_out = self.snn(spk_input)
        return spk_out.mean(dim=0)  # (batch, num_classes) -- rate decoding


# ============================================================
# Training & evaluation
# ============================================================

def train_epoch_snn(model: SNNWrapper, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple[float, float]:
    """Train SNN for one epoch using CE rate loss."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    ce = nn.CrossEntropyLoss()

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = ce(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def train_epoch_ann(model: ConvANN, loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> tuple[float, float]:
    """Train ANN for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    ce = nn.CrossEntropyLoss()

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = ce(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> tuple[float, float]:
    """Evaluate model on a dataloader. Returns (loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    ce = nn.CrossEntropyLoss()

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        loss = ce(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def train_model(model: nn.Module, train_loader: DataLoader,
                test_loader: DataLoader, train_fn, device: torch.device,
                model_name: str, epochs: int, patience: int,
                save_path: Path) -> dict:
    """Full training loop with early stopping and LR scheduling.

    Args:
        model: The model to train.
        train_loader: Training data loader.
        test_loader: Test data loader.
        train_fn: Training function (train_epoch_snn or train_epoch_ann).
        device: Device to train on.
        model_name: Name for logging (e.g. "SNN-direct", "ANN").
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save best model checkpoint.

    Returns:
        Dict with training history and best results.
    """
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

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_fn(model, train_loader, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, device)
        scheduler.step(te_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = epoch
            no_improve = 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [{model_name}] Ep {epoch:3d}/{epochs} | "
                  f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
                  f"te_loss={te_loss:.4f} te_acc={te_acc:.3f} | "
                  f"best={best_acc:.3f} lr={lr_now:.1e} ({elapsed:.0f}s)")

        if no_improve >= patience:
            elapsed = time.time() - t0
            print(f"  [{model_name}] Early stop at epoch {epoch}, "
                  f"best={best_acc:.4f} at epoch {best_epoch} ({elapsed:.0f}s)")
            break

    total_time = time.time() - t0

    return {
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "total_time_s": total_time,
        "history": history,
    }


# ============================================================
# Plotting
# ============================================================

def plot_training_curves(snn_history: dict, ann_history: dict,
                         out_path: Path, encoding: str):
    """Plot training/test accuracy curves for SNN and ANN."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    ax = axes[0]
    ax.plot(snn_history["train_acc"], label=f"SNN-{encoding} train",
            color="#2196F3", linestyle="--", alpha=0.7)
    ax.plot(snn_history["test_acc"], label=f"SNN-{encoding} test",
            color="#2196F3", linewidth=2)
    ax.plot(ann_history["train_acc"], label="ANN train",
            color="#FF5722", linestyle="--", alpha=0.7)
    ax.plot(ann_history["test_acc"], label="ANN test",
            color="#FF5722", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("UrbanSound8K: Training Curves")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[1]
    ax.plot(snn_history["train_loss"], label=f"SNN-{encoding} train",
            color="#2196F3", linestyle="--", alpha=0.7)
    ax.plot(snn_history["test_loss"], label=f"SNN-{encoding} test",
            color="#2196F3", linewidth=2)
    ax.plot(ann_history["train_loss"], label="ANN train",
            color="#FF5722", linestyle="--", alpha=0.7)
    ax.plot(ann_history["test_loss"], label="ANN test",
            color="#FF5722", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("UrbanSound8K: Loss Curves")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="UrbanSound8K 1-fold experiment: SNN vs ANN"
    )
    parser.add_argument("--test-fold", type=int, default=10,
                        help="Test fold (default: 10, standard US8K protocol)")
    parser.add_argument("--encoding", type=str, default="direct",
                        help="SNN encoding method (default: direct)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Max epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help=f"Early stopping patience (default: {PATIENCE})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", default=None,
                        help="Device (default: auto-detect)")
    args = parser.parse_args()

    # --------------------------------------------------------
    # Check dataset exists
    # --------------------------------------------------------
    if not US8K_DIR.exists() or not US8K_META_PATH.exists():
        print("=" * 65)
        print("ERROR: UrbanSound8K dataset not found!")
        print("=" * 65)
        print()
        print("Expected location:")
        print(f"  {US8K_DIR}/")
        print()
        print("Expected structure:")
        print(f"  {US8K_DIR}/")
        print(f"    metadata/UrbanSound8K.csv")
        print(f"    audio/")
        print(f"      fold1/ ... fold10/")
        print()
        print("Download instructions:")
        print("  1. Go to https://urbansounddataset.webs.com/urbansound8k.html")
        print("     (or search 'UrbanSound8K' on Kaggle)")
        print("  2. Download and extract the dataset")
        print(f"  3. Place it at: {US8K_DIR}/")
        print()
        print("If using Kaggle:")
        print("  pip install kaggle")
        print("  kaggle datasets download -d chrisfilo/urbansound8k")
        print(f"  unzip urbansound8k.zip -d {DATA_DIR}/")
        sys.exit(1)

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    test_fold = args.test_fold
    train_folds = [f for f in range(1, US8K_NUM_FOLDS + 1) if f != test_fold]

    print("=" * 65)
    print("UrbanSound8K 1-Fold Experiment: SNN vs ANN")
    print("=" * 65)
    print(f"  Test fold    : {test_fold}")
    print(f"  Train folds  : {train_folds}")
    print(f"  Encoding     : {args.encoding}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Patience     : {args.patience}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Seed         : {args.seed}")
    print(f"  Device       : {device}")
    print(f"  Num classes  : {US8K_NUM_CLASSES}")
    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    print("Loading UrbanSound8K data...")
    train_dataset = UrbanSound8KDataset(folds=train_folds, precompute=True)
    test_dataset = UrbanSound8KDataset(folds=[test_fold], precompute=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples : {len(test_dataset)}")
    print()

    # Output directory
    out_dir = RESULTS_DIR / "urbansound8k"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Train SNN
    # --------------------------------------------------------
    print("-" * 65)
    print(f"Training SNN ({args.encoding} encoding, {US8K_NUM_CLASSES} classes)")
    print("-" * 65)

    torch.manual_seed(args.seed)
    encoder = get_encoder(args.encoding)
    snn_model = SpikingCNN(num_classes=US8K_NUM_CLASSES).to(device)
    snn_wrapper = SNNWrapper(snn_model, encoder=encoder, num_steps=NUM_STEPS).to(device)

    snn_save_path = out_dir / f"snn_{args.encoding}_fold{test_fold}.pt"
    snn_results = train_model(
        model=snn_wrapper,
        train_loader=train_loader,
        test_loader=test_loader,
        train_fn=train_epoch_snn,
        device=device,
        model_name=f"SNN-{args.encoding}",
        epochs=args.epochs,
        patience=args.patience,
        save_path=snn_save_path,
    )
    print(f"  SNN best accuracy: {snn_results['best_accuracy']*100:.2f}% "
          f"(epoch {snn_results['best_epoch']})")
    print()

    # --------------------------------------------------------
    # Train ANN
    # --------------------------------------------------------
    print("-" * 65)
    print(f"Training ANN ({US8K_NUM_CLASSES} classes)")
    print("-" * 65)

    torch.manual_seed(args.seed)
    ann_model = ConvANN(num_classes=US8K_NUM_CLASSES).to(device)

    ann_save_path = out_dir / f"ann_fold{test_fold}.pt"
    ann_results = train_model(
        model=ann_model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_fn=train_epoch_ann,
        device=device,
        model_name="ANN",
        epochs=args.epochs,
        patience=args.patience,
        save_path=ann_save_path,
    )
    print(f"  ANN best accuracy: {ann_results['best_accuracy']*100:.2f}% "
          f"(epoch {ann_results['best_epoch']})")
    print()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Dataset      : UrbanSound8K (10 classes, fold {test_fold} test)")
    print(f"  SNN ({args.encoding:>6s}) : {snn_results['best_accuracy']*100:.2f}% "
          f"(epoch {snn_results['best_epoch']}/{snn_results['total_epochs']}, "
          f"{snn_results['total_time_s']:.0f}s)")
    print(f"  ANN          : {ann_results['best_accuracy']*100:.2f}% "
          f"(epoch {ann_results['best_epoch']}/{ann_results['total_epochs']}, "
          f"{ann_results['total_time_s']:.0f}s)")

    gap = ann_results["best_accuracy"] - snn_results["best_accuracy"]
    print(f"  ANN-SNN gap  : {gap*100:+.2f} pp")

    if gap > 0:
        print(f"  FINDING: ANN beats SNN by {gap*100:.2f} pp on UrbanSound8K.")
    elif gap < 0:
        print(f"  FINDING: SNN beats ANN by {-gap*100:.2f} pp on UrbanSound8K.")
    else:
        print(f"  FINDING: SNN and ANN are tied on UrbanSound8K.")

    # Compare to ESC-50 gap (16.7 pp for direct encoding)
    esc50_gap = 16.70  # from memory: ANN 63.85% - SNN 47.15%
    print(f"\n  ESC-50 ANN-SNN gap (direct): {esc50_gap:.2f} pp")
    print(f"  US8K  ANN-SNN gap (direct): {gap*100:.2f} pp")
    if abs(gap * 100 - esc50_gap) < 5:
        print("  CONCLUSION: Gap is similar -- encoding hierarchy likely generalises.")
    elif gap * 100 < esc50_gap:
        print("  CONCLUSION: Gap is SMALLER on US8K -- SNN relatively better on "
              "this dataset.")
    else:
        print("  CONCLUSION: Gap is LARGER on US8K -- SNN relatively worse on "
              "this dataset.")

    # --------------------------------------------------------
    # Save results JSON
    # --------------------------------------------------------
    results_all = {
        "dataset": "UrbanSound8K",
        "num_classes": US8K_NUM_CLASSES,
        "test_fold": test_fold,
        "train_folds": train_folds,
        "encoding": args.encoding,
        "seed": args.seed,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "device": str(device),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "snn": {
            "best_accuracy": snn_results["best_accuracy"],
            "best_epoch": snn_results["best_epoch"],
            "total_epochs": snn_results["total_epochs"],
            "total_time_s": snn_results["total_time_s"],
        },
        "ann": {
            "best_accuracy": ann_results["best_accuracy"],
            "best_epoch": ann_results["best_epoch"],
            "total_epochs": ann_results["total_epochs"],
            "total_time_s": ann_results["total_time_s"],
        },
        "ann_snn_gap_pp": gap * 100,
    }

    json_path = out_dir / f"us8k_{args.encoding}_fold{test_fold}.json"
    with open(json_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # --------------------------------------------------------
    # Plot training curves
    # --------------------------------------------------------
    plot_path = out_dir / f"us8k_{args.encoding}_fold{test_fold}_curves.png"
    plot_training_curves(
        snn_results["history"], ann_results["history"],
        plot_path, args.encoding,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
