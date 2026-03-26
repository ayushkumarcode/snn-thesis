"""
cochleagram_experiment.py -- Replace mel spectrogram with gammatone filterbank cochleagram.

Computes a "gammatonegram": STFT magnitude filtered through a 64-channel gammatone
filterbank (ERB-spaced from 50 Hz to Nyquist). This is a biologically-motivated
alternative to the mel spectrogram, modelling the basilar membrane frequency response
more accurately (asymmetric filters, level-dependent bandwidth).

Hypothesis: SNNs may benefit more from cochleagram input than ANNs, since cochleagram
better matches the auditory periphery that spiking neurons evolved to process.

Both SNN (EnhancedSpikingCNN with learn_beta, learn_threshold, SRE, dropout) and
ANN (ConvANN) are trained on cochleagrams for comparison against mel baselines.

Output shape is (1, 64, 216) -- same as mel spectrogram -- so existing model
architectures work without modification.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/cochleagram_experiment.py                    # both models, all 5 folds
    python experiments/cochleagram_experiment.py --fold 1           # single fold
    python experiments/cochleagram_experiment.py --model snn        # SNN only
    python experiments/cochleagram_experiment.py --model ann        # ANN only
    python experiments/cochleagram_experiment.py --device cuda       # specify device
"""

import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import Dataset, DataLoader
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_CLASSES, NUM_STEPS, N_MELS, BETA,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    SAMPLE_RATE, DURATION, N_FFT, HOP_LENGTH,
    RESULTS_DIR, get_device,
    ESC50_AUDIO_DIR, ESC50_META_PATH, NUM_FOLDS,
)
from src.dataset import download_esc50
from src.encoding import encode_direct


# ============================================================
# Gammatone filterbank implementation
# ============================================================

def erb(fc):
    """Equivalent Rectangular Bandwidth (Glasberg & Moore, 1990).

    ERB(fc) = 24.7 * (4.37 * fc/1000 + 1)

    Args:
        fc: Centre frequency in Hz (scalar or array).

    Returns:
        ERB bandwidth in Hz.
    """
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def gammatone_filterbank(sr, n_fft, n_filters=64, f_min=50.0, f_max=None):
    """Create a gammatone filterbank matrix for applying to STFT magnitude.

    Each row is one gammatone filter's frequency response, evaluated at the
    STFT frequency bins. The filters are 4th-order gammatone with ERB-spaced
    centre frequencies.

    The gammatone frequency response magnitude for a 4th-order filter centred
    at fc with bandwidth b is:
        |H(f)| = 1 / (1 + ((f - fc) / b)^2)^2

    This is the magnitude-squared of the transfer function of the cascade of
    4 first-order bandpass filters, which is a good approximation of the
    cochlear filter shape (Patterson et al., 1992).

    Args:
        sr: Sample rate in Hz.
        n_fft: FFT size.
        n_filters: Number of filters (channels). Default 64.
        f_min: Lowest centre frequency in Hz. Default 50.
        f_max: Highest centre frequency in Hz. Default sr/2.

    Returns:
        filterbank: np.ndarray of shape (n_filters, 1 + n_fft // 2).
    """
    if f_max is None:
        f_max = sr / 2.0

    # ERB-rate scale: convert Hz to ERB-rate, space linearly, convert back
    # ERB-rate(f) = 21.4 * log10(4.37 * f/1000 + 1)  (Glasberg & Moore)
    erb_lo = 21.4 * np.log10(4.37 * f_min / 1000.0 + 1.0)
    erb_hi = 21.4 * np.log10(4.37 * f_max / 1000.0 + 1.0)
    erb_points = np.linspace(erb_lo, erb_hi, n_filters)

    # Convert back to Hz
    center_freqs = (10.0 ** (erb_points / 21.4) - 1.0) * 1000.0 / 4.37

    # STFT frequency bins
    n_freqs = 1 + n_fft // 2
    freqs = np.linspace(0, sr / 2.0, n_freqs)

    filterbank = np.zeros((n_filters, n_freqs), dtype=np.float32)

    for i, fc in enumerate(center_freqs):
        # Bandwidth parameter: 1.019 * ERB(fc) -- standard gammatone
        b = 1.019 * erb(fc)

        # 4th-order gammatone magnitude response
        # |H(f)|^2 = 1 / (1 + ((f - fc) / b)^2)^4
        # We use the amplitude (not power) so take sqrt:
        # |H(f)| = 1 / (1 + ((f - fc) / b)^2)^2
        filterbank[i] = 1.0 / (1.0 + ((freqs - fc) / b) ** 2) ** 2

    # Normalise each filter to have unit peak (preserves relative energy)
    for i in range(n_filters):
        peak = filterbank[i].max()
        if peak > 0:
            filterbank[i] /= peak

    return filterbank, center_freqs


def wav_to_cochleagram(filepath, sr=SAMPLE_RATE, duration=DURATION,
                       n_fft=N_FFT, hop_length=HOP_LENGTH,
                       n_filters=64, f_min=50.0, f_max=None):
    """Load audio and compute gammatone cochleagram.

    Pipeline:
      1. Load and pad audio to exactly `duration` seconds
      2. Compute STFT magnitude
      3. Apply gammatone filterbank (matrix multiply)
      4. Convert to log scale (dB)
      5. Output shape: (n_filters, time_frames)

    The time_frames dimension matches mel spectrogram exactly because
    we use the same n_fft and hop_length.

    Args:
        filepath: Path to WAV file.
        sr: Sample rate.
        duration: Audio duration in seconds.
        n_fft: FFT window size.
        hop_length: STFT hop length.
        n_filters: Number of gammatone channels.
        f_min: Lowest centre frequency.
        f_max: Highest centre frequency (default: Nyquist).

    Returns:
        cochleagram: np.ndarray of shape (n_filters, time_frames).
    """
    # Load audio
    y, _ = librosa.load(filepath, sr=sr, duration=duration)

    # Pad to exactly duration seconds if shorter
    expected_len = sr * duration
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))

    # STFT magnitude (same windowing as mel spectrogram)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # S shape: (1 + n_fft//2, time_frames) = (513, 216)

    # Build gammatone filterbank
    fb, center_freqs = gammatone_filterbank(
        sr=sr, n_fft=n_fft, n_filters=n_filters,
        f_min=f_min, f_max=f_max,
    )
    # fb shape: (n_filters, 513)

    # Apply filterbank: cochleagram = fb @ S
    # Result shape: (n_filters, time_frames) = (64, 216)
    cochleagram = fb @ S

    # Convert to log scale (power to dB, same as mel pipeline)
    cochleagram_db = librosa.power_to_db(cochleagram ** 2, ref=np.max)

    return cochleagram_db


def normalise_spectrogram(spec):
    """Min-max normalise to [0, 1]."""
    min_val = spec.min()
    max_val = spec.max()
    if max_val - min_val == 0:
        return np.zeros_like(spec)
    return (spec - min_val) / (max_val - min_val)


# ============================================================
# Dataset
# ============================================================

class CochleagramDataset(Dataset):
    """ESC-50 dataset with gammatone cochleagram features.

    Drop-in replacement for ESC50Dataset but uses cochleagram instead of
    mel spectrogram.

    Args:
        folds: List of fold numbers to include (1-5).
        transform: Optional transform applied to the tensor.
        precompute: If True, load and cache all cochleagrams in memory.
    """

    def __init__(self, folds, transform=None, precompute=True):
        self.transform = transform
        self.precompute = precompute

        meta = pd.read_csv(ESC50_META_PATH)
        self.meta = meta[meta["fold"].isin(folds)].reset_index(drop=True)

        self.data = []
        self.labels = []

        if precompute:
            print(f"Computing cochleagrams for {len(self.meta)} clips from folds {folds}...")
            for _, row in self.meta.iterrows():
                filepath = ESC50_AUDIO_DIR / row["filename"]
                coch = wav_to_cochleagram(str(filepath))
                coch = normalise_spectrogram(coch)
                self.data.append(coch)
                self.labels.append(row["target"])

            self.data = np.array(self.data, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.precompute:
            coch = self.data[idx]
            label = self.labels[idx]
        else:
            row = self.meta.iloc[idx]
            filepath = ESC50_AUDIO_DIR / row["filename"]
            coch = wav_to_cochleagram(str(filepath))
            coch = normalise_spectrogram(coch)
            label = row["target"]

        # Shape: (1, n_filters, time_frames) -- single channel
        tensor = torch.tensor(coch, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(label, dtype=torch.long)


def get_cochleagram_dataloaders(test_fold, batch_size=BATCH_SIZE):
    """Create train/test DataLoaders using cochleagram features.

    Args:
        test_fold: Fold number (1-5) to use as test set.
        batch_size: Batch size for DataLoaders.

    Returns:
        (train_loader, test_loader)
    """
    train_folds = [f for f in range(1, NUM_FOLDS + 1) if f != test_fold]

    train_dataset = CochleagramDataset(folds=train_folds)
    test_dataset = CochleagramDataset(folds=[test_fold])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    return train_loader, test_loader


# ============================================================
# SNN Model (Enhanced: learn_beta, learn_threshold, SRE, dropout)
# ============================================================

class CochleagramSNN(nn.Module):
    """SpikingCNN for cochleagram input.

    Same architecture as EnhancedSpikingCNN:
      Conv(1,32) -> BN -> MaxPool(2) -> LIF ->
      Conv(32,64) -> BN -> MaxPool(2) -> LIF ->
      AvgPool(4,6) -> FC(2304,256) -> LIF -> Dropout(0.3) -> FC(256,50) -> LIF

    Uses spike_rate_escape surrogate, learnable beta and threshold.
    Input shape: (num_steps, batch, 1, 64, 216) -- identical to mel.
    """

    def __init__(self, num_classes=NUM_CLASSES, beta=BETA, num_steps=NUM_STEPS):
        super().__init__()
        self.num_steps = num_steps

        spike_grad = surrogate.spike_rate_escape(beta=1, slope=25)

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                               learn_beta=True, learn_threshold=True)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                               learn_beta=True, learn_threshold=True)

        # After two MaxPool2d(2) on (64, 216): -> (32, 108) -> (16, 54)
        # AvgPool2d(4,6) on (16, 54) -> (4, 9)
        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # FC block 1: 64 * 4 * 9 = 2304
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                               learn_beta=True, learn_threshold=True)

        self.dropout = nn.Dropout(0.3)

        # FC block 2 (output)
        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad,
                               learn_beta=True, learn_threshold=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input of shape (num_steps, batch, 1, 64, time_frames).

        Returns:
            spk_out: (num_steps, batch, num_classes)
            mem_out: (num_steps, batch, num_classes)
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_out_rec = []
        mem_out_rec = []

        for step in range(self.num_steps):
            x_t = x[step]

            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            cur3 = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3 = self.dropout(spk3)

            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)

        return torch.stack(spk_out_rec), torch.stack(mem_out_rec)


# ============================================================
# ANN Model (mirrors ConvANN exactly)
# ============================================================

class CochleagramANN(nn.Module):
    """ConvANN for cochleagram input. Same architecture as ConvANN.

    Input shape: (batch, 1, 64, 216).
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # AvgPool2d(4,6) on (16,54) -> (4,9)
            nn.AvgPool2d(kernel_size=(4, 6)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        flat = features.view(features.size(0), -1)
        return self.classifier(flat)


# ============================================================
# Training / Evaluation
# ============================================================

def train_snn_epoch(model, loader, optimizer, device):
    """Train SNN for one epoch using CE on membrane potentials."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)

        # Direct encoding: repeat input across timesteps
        spk_input = encode_direct(data).to(device)

        optimizer.zero_grad()
        spk_out, mem_out = model(spk_input)

        # Sum CE loss across timesteps (same as enhanced_snn.py)
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
def eval_snn(model, loader, device):
    """Evaluate SNN on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spk_input = encode_direct(data).to(device)

        spk_out, mem_out = model(spk_input)

        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss += criterion(mem_out[step], targets)
        total_loss += loss.item()

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


def train_ann_epoch(model, loader, optimizer, device):
    """Train ANN for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

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

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_ann(model, loader, device):
    """Evaluate ANN on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)

        logits = model(data)
        loss = criterion(logits, targets)
        total_loss += loss.item()

        predicted = logits.argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    return total_loss / len(loader), correct / total


# ============================================================
# Run fold
# ============================================================

def run_fold(fold, model_type, device, num_epochs=NUM_EPOCHS, patience=PATIENCE):
    """Train and evaluate one fold for a given model type.

    Args:
        fold: Test fold (1-5).
        model_type: "snn" or "ann".
        device: Torch device.
        num_epochs: Maximum training epochs.
        patience: Early stopping patience.

    Returns:
        dict with results.
    """
    print(f"\n{'='*60}")
    print(f"  Cochleagram {model_type.upper()} | Fold {fold}/5 | Device: {device}")
    print(f"{'='*60}")

    # Load cochleagram data
    train_loader, test_loader = get_cochleagram_dataloaders(fold, BATCH_SIZE)

    # Verify shapes
    sample_data, sample_label = next(iter(train_loader))
    print(f"  Input shape: {sample_data.shape}")
    assert sample_data.shape[1:] == (1, 64, 216), (
        f"Expected (1, 64, 216) but got {sample_data.shape[1:]}. "
        f"Cochleagram time frames mismatch -- check n_fft and hop_length."
    )

    if model_type == "snn":
        model = CochleagramSNN().to(device)
        train_fn = train_snn_epoch
        eval_fn = eval_snn
    else:
        model = CochleagramANN().to(device)
        train_fn = train_ann_epoch
        eval_fn = eval_ann

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
    }

    start = time.time()
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_fn(model, train_loader, optimizer, device)
        test_loss, test_acc = eval_fn(model, test_loader, device)
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
            save_dir = RESULTS_DIR / "experiments" / "cochleagram"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(),
                       save_dir / f"best_{model_type}_fold{fold}.pt")
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start
            print(f"  Ep {epoch:3d}/{num_epochs} | "
                  f"Train: {train_acc:.4f} | Test: {test_acc:.4f} | "
                  f"Best: {best_acc:.4f} (ep{best_epoch}) | {elapsed:.0f}s")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start
    print(f"  Fold {fold} {model_type.upper()} done in {elapsed:.1f}s | "
          f"Best: {best_acc*100:.2f}% (ep{best_epoch})")

    result = {
        "fold": fold,
        "model_type": model_type,
        "experiment": "cochleagram",
        "feature_type": "gammatone_cochleagram",
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "time_seconds": round(elapsed, 1),
        "n_params": n_params,
        "history": history,
        "config": {
            "n_filters": 64,
            "f_min": 50.0,
            "f_max": SAMPLE_RATE / 2.0,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "sr": SAMPLE_RATE,
            "duration": DURATION,
            "num_steps": NUM_STEPS,
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "num_epochs": NUM_EPOCHS,
        },
    }

    save_dir = RESULTS_DIR / "experiments" / "cochleagram"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / f"{model_type}_fold{fold}.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cochleagram experiment: gammatone filterbank vs mel spectrogram"
    )
    parser.add_argument("--fold", type=int, default=None,
                        help="Run a single fold (1-5). Default: all 5 folds.")
    parser.add_argument("--model", type=str, default="both",
                        choices=["snn", "ann", "both"],
                        help="Which model(s) to train. Default: both.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda, mps, cpu). Default: auto-detect.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Max epochs. Default: {NUM_EPOCHS}.")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help=f"Early stopping patience. Default: {PATIENCE}.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    download_esc50()

    # Verify cochleagram shape on a single file before training
    print("\nVerifying cochleagram pipeline...")
    sample_file = list(ESC50_AUDIO_DIR.glob("*.wav"))[0]
    coch = wav_to_cochleagram(str(sample_file))
    coch_norm = normalise_spectrogram(coch)
    print(f"  Raw cochleagram shape: {coch.shape}")
    print(f"  Range after normalisation: [{coch_norm.min():.4f}, {coch_norm.max():.4f}]")
    print(f"  Expected: (64, 216)")
    assert coch.shape == (64, 216), (
        f"Cochleagram shape mismatch: got {coch.shape}, expected (64, 216). "
        f"Check N_FFT={N_FFT} and HOP_LENGTH={HOP_LENGTH}."
    )
    print("  Cochleagram pipeline verified.\n")

    folds = [args.fold] if args.fold else list(range(1, 6))
    model_types = ["snn", "ann"] if args.model == "both" else [args.model]

    all_results = {}

    for model_type in model_types:
        fold_results = []
        for fold in folds:
            result = run_fold(fold, model_type, device, args.epochs, args.patience)
            fold_results.append(result)

        # Summary for this model type
        if len(fold_results) >= 2:
            accs = [r["best_acc"] for r in fold_results]
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f"\n{'='*60}")
            print(f"  Cochleagram {model_type.upper()} {len(folds)}-Fold Summary")
            print(f"  Mean: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
            print(f"  Per-fold: {[f'{a*100:.2f}%' for a in accs]}")
            print(f"{'='*60}")

            all_results[model_type] = {
                "fold_accuracies": accs,
                "mean_accuracy": float(mean_acc),
                "std_accuracy": float(std_acc),
                "per_fold": {f"fold{r['fold']}": r["best_acc"] for r in fold_results},
            }
        elif len(fold_results) == 1:
            all_results[model_type] = {
                "fold_accuracies": [fold_results[0]["best_acc"]],
                "mean_accuracy": fold_results[0]["best_acc"],
                "std_accuracy": 0.0,
                "per_fold": {f"fold{fold_results[0]['fold']}": fold_results[0]["best_acc"]},
            }

    # Save overall summary
    save_dir = RESULTS_DIR / "experiments" / "cochleagram"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Comparison with mel baselines
    mel_baselines = {
        "snn_mel_direct": {"mean": 0.4715, "std": 0.0450},
        "ann_mel": {"mean": 0.6385, "std": 0.0307},
    }

    summary = {
