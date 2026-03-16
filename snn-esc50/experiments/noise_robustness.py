"""
noise_robustness.py -- Evaluate SNN vs ANN accuracy under additive Gaussian noise
at various signal-to-noise ratio (SNR) levels.

Research question: Does the binary spike thresholding in SNNs provide natural
denoising that makes them more robust to input noise than ANNs?

Method:
  - Load raw audio waveforms from ESC-50
  - For each SNR level: clean (inf), 20dB, 10dB, 5dB, 0dB, -5dB
  - Add Gaussian white noise to the raw waveform at the specified SNR:
      noise_power = signal_power / 10^(SNR_dB / 10)
      noise = N(0, sqrt(noise_power))
  - Recompute mel spectrogram from noisy waveform, normalise to [0, 1]
  - Evaluate both SNN (direct encoding) and ANN on the noisy spectrograms
  - Report accuracy at each SNR level and plot degradation curves

Key difference from adversarial_robustness.py:
  Adversarial adds *optimised* perturbations to spectrograms.
  Noise robustness adds *random* perturbations to raw waveforms --
  a more realistic corruption model for real-world deployment.

Usage:
  source .venv/bin/activate
  cd snn-esc50/
  python experiments/noise_robustness.py
  python experiments/noise_robustness.py --fold 4
  python experiments/noise_robustness.py --folds 1 2 3 4 5
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    RESULTS_DIR, NUM_FOLDS, BATCH_SIZE, SAMPLE_RATE, DURATION,
    N_MELS, N_FFT, HOP_LENGTH, get_device,
)
from src.config import ESC50_AUDIO_DIR, ESC50_META_PATH, F_MIN, F_MAX
from src.config import NUM_STEPS, NUM_CLASSES, BETA
from src.dataset import download_esc50, normalise_spectrogram
from src.encoding import encode_direct
from src.models.snn_model import SpikingCNN
from src.models.ann_model import ConvANN


# ============================================================
# Noisy audio dataset
# ============================================================

def add_noise_at_snr(waveform: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian white noise to a waveform at a specified SNR (dB).

    Args:
        waveform: Clean audio signal, shape (num_samples,).
        snr_db: Target signal-to-noise ratio in decibels.
                float('inf') returns the clean waveform unchanged.

    Returns:
        Noisy waveform with the same shape.
    """
    if np.isinf(snr_db):
        return waveform.copy()

    signal_power = np.mean(waveform ** 2)

    # Avoid division by zero for silent clips
    if signal_power < 1e-10:
        return waveform.copy()

    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.randn(*waveform.shape) * np.sqrt(noise_power)
    return waveform + noise


def wav_to_mel_noisy(filepath: str, snr_db: float) -> np.ndarray:
    """Load audio, add noise at given SNR, compute normalised mel spectrogram.

    Args:
        filepath: Path to a .wav file.
        snr_db: Target SNR in dB. Use float('inf') for clean.

    Returns:
        Normalised mel spectrogram, shape (n_mels, time_frames), in [0, 1].
    """
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)

    # Pad to exactly DURATION seconds if shorter
    expected_len = SAMPLE_RATE * DURATION
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))

    # Add noise at the waveform level
    y_noisy = add_noise_at_snr(y, snr_db)

    # Compute mel spectrogram from noisy waveform
    mel = librosa.feature.melspectrogram(
        y=y_noisy, sr=sr,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmin=F_MIN, fmax=F_MAX,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return normalise_spectrogram(mel_db)


class NoisyESC50Dataset(Dataset):
    """ESC-50 dataset that adds noise at a specified SNR to raw audio
    before computing mel spectrograms.

    Args:
        folds: List of fold numbers to include (1-5).
        snr_db: Signal-to-noise ratio in dB. Use float('inf') for clean.
    """

    def __init__(self, folds: list[int], snr_db: float):
        self.snr_db = snr_db

        meta = pd.read_csv(ESC50_META_PATH)
        self.meta = meta[meta["fold"].isin(folds)].reset_index(drop=True)

        # Precompute all noisy spectrograms
        self.data = []
        self.labels = []

        for _, row in self.meta.iterrows():
            filepath = ESC50_AUDIO_DIR / row["filename"]
            mel = wav_to_mel_noisy(str(filepath), snr_db)
            self.data.append(mel)
            self.labels.append(row["target"])

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        mel = self.data[idx]
        label = self.labels[idx]
        tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)


# ============================================================
# SNN wrapper (same pattern as adversarial_robustness.py)
# ============================================================

class SNNWrapper(nn.Module):
    """Wraps SNN + direct encoder for standard forward pass.

    Forward: x (B, 1, H, W) -> encode_direct -> SNN -> mem_out.sum(0) logits.
    """

    def __init__(self, snn_model: SpikingCNN, num_steps: int = NUM_STEPS):
        super().__init__()
        self.snn = snn_model
        self.T = num_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spk_input = encode_direct(x, num_steps=self.T)
        spk_out, mem_out = self.snn(spk_input)
        return mem_out.sum(dim=0)


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate model accuracy on a dataloader."""
    model.eval()
    correct = 0
    total = 0

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        logits = model(data)
        correct += (logits.argmax(1) == targets).sum().item()
        total += targets.size(0)

    return correct / total if total > 0 else 0.0


# ============================================================
# Plotting
# ============================================================

def plot_degradation_curves(snr_levels, snn_means, snn_stds,
                            ann_means, ann_stds, out_path: Path):
    """Plot accuracy vs SNR degradation curves for SNN and ANN."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # x positions: use SNR values (inf plotted as max + step)
    x_labels = []
    x_positions = []
    for i, snr in enumerate(snr_levels):
        if np.isinf(snr):
            x_labels.append("Clean")
        else:
            x_labels.append(f"{snr:.0f}")
        x_positions.append(i)

    ax.errorbar(x_positions, [m * 100 for m in snn_means],
                yerr=[s * 100 for s in snn_stds],
                marker="o", linewidth=2, capsize=4,
                label="SNN (direct)", color="#2196F3")
    ax.errorbar(x_positions, [m * 100 for m in ann_means],
                yerr=[s * 100 for s in ann_stds],
                marker="s", linewidth=2, capsize=4,
                label="ANN (baseline)", color="#FF5722")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("SNR (dB)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Noise Robustness: SNN vs ANN on ESC-50", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Noise robustness evaluation: SNN vs ANN at various SNR levels"
    )
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Folds to evaluate (default: all 5)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Single fold shorthand (overridden by --folds)")
    parser.add_argument("--encoding", type=str, default="direct",
                        help="SNN encoding method (default: direct)")
    parser.add_argument("--device", default=None,
                        help="Device (default: auto-detect)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    args = parser.parse_args()

    # Resolve folds
    if args.folds is not None:
        folds = args.folds
    elif args.fold is not None:
        folds = [args.fold]
    else:
        folds = list(range(1, NUM_FOLDS + 1))

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    # SNR levels to evaluate (in dB). float('inf') = clean.
    snr_levels = [float("inf"), 20.0, 10.0, 5.0, 0.0, -5.0]
    snr_labels = ["clean", "20dB", "10dB", "5dB", "0dB", "-5dB"]

    # Determine encoding subdir for SNN weights
    encoding = args.encoding
    # ANN encoding subdir is always "none"
    ann_encoding = "none"

    print("=" * 65)
    print("Noise Robustness Evaluation: SNN vs ANN")
    print("=" * 65)
    print(f"  Folds    : {folds}")
    print(f"  Encoding : {encoding}")
    print(f"  SNR (dB) : {snr_labels}")
    print(f"  Device   : {device}")
    print()

    # Ensure dataset is available
    download_esc50()

    out_dir = RESULTS_DIR / "noise_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Storage: per-fold results at each SNR
    # snn_accs[snr_idx][fold_idx], ann_accs[snr_idx][fold_idx]
    snn_accs = {i: [] for i in range(len(snr_levels))}
    ann_accs = {i: [] for i in range(len(snr_levels))}

    for fold in folds:
        print(f"\n--- Fold {fold} ---")

        # Load models
        snn_path = RESULTS_DIR / "snn" / encoding / f"best_fold{fold}.pt"
        ann_path = RESULTS_DIR / "ann" / ann_encoding / f"best_fold{fold}.pt"

        for p in [snn_path, ann_path]:
            if not p.exists():
                print(f"FATAL: Model not found: {p}")
                sys.exit(1)

        snn_model = SpikingCNN().to(device)
        snn_model.load_state_dict(
            torch.load(snn_path, map_location=device, weights_only=True)
        )
        snn_model.eval()
        snn_wrapper = SNNWrapper(snn_model, num_steps=NUM_STEPS).to(device)
        print(f"  Loaded SNN: {snn_path.name}")

        ann_model = ConvANN().to(device)
        ann_model.load_state_dict(
            torch.load(ann_path, map_location=device, weights_only=True)
        )
        ann_model.eval()
        print(f"  Loaded ANN: {ann_path.name}")

        # Header
        print(f"\n  {'SNR':>8}  {'SNN Acc':>10}  {'ANN Acc':>10}")
        print(f"  {'-'*8}  {'-'*10}  {'-'*10}")

        for snr_idx, (snr_db, label) in enumerate(zip(snr_levels, snr_labels)):
            # Build noisy dataset for this SNR
            noisy_dataset = NoisyESC50Dataset(
                folds=[fold], snr_db=snr_db
            )
            noisy_loader = DataLoader(
                noisy_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=0,
            )

            # Evaluate both models
            acc_snn = evaluate(snn_wrapper, noisy_loader, device)
            acc_ann = evaluate(ann_model, noisy_loader, device)

            snn_accs[snr_idx].append(acc_snn)
            ann_accs[snr_idx].append(acc_ann)

            print(f"  {label:>8}  {acc_snn:>10.4f}  {acc_ann:>10.4f}")

    # --------------------------------------------------------
    # Aggregate across folds
    # --------------------------------------------------------
    print("\n" + "=" * 65)
    print("Aggregated Results (mean +/- std across folds)")
    print("=" * 65)

    snn_means = []
    snn_stds = []
    ann_means = []
    ann_stds = []

    print(f"\n{'SNR':>8}  {'SNN (mean +/- std)':>22}  {'ANN (mean +/- std)':>22}")
    print(f"{'-'*8}  {'-'*22}  {'-'*22}")

    results_summary = {
        "folds": folds,
        "encoding": encoding,
        "snr_levels_db": [str(s) if np.isinf(s) else s for s in snr_levels],
        "snr_labels": snr_labels,
        "snn": {"per_fold": {}, "mean": [], "std": []},
        "ann": {"per_fold": {}, "mean": [], "std": []},
    }

    for snr_idx, label in enumerate(snr_labels):
        s_arr = np.array(snn_accs[snr_idx])
        a_arr = np.array(ann_accs[snr_idx])

        s_mean, s_std = s_arr.mean(), s_arr.std()
        a_mean, a_std = a_arr.mean(), a_arr.std()

        snn_means.append(s_mean)
        snn_stds.append(s_std)
        ann_means.append(a_mean)
        ann_stds.append(a_std)

        results_summary["snn"]["per_fold"][label] = [float(v) for v in s_arr]
        results_summary["ann"]["per_fold"][label] = [float(v) for v in a_arr]
        results_summary["snn"]["mean"].append(float(s_mean))
        results_summary["snn"]["std"].append(float(s_std))
        results_summary["ann"]["mean"].append(float(a_mean))
        results_summary["ann"]["std"].append(float(a_std))

        print(f"{label:>8}  {s_mean:>8.4f} +/- {s_std:<8.4f}  "
              f"{a_mean:>8.4f} +/- {a_std:<8.4f}")

    # --------------------------------------------------------
    # Analysis: relative degradation
    # --------------------------------------------------------
    print()
    clean_snn = snn_means[0]
    clean_ann = ann_means[0]
    print(f"Clean accuracy: SNN={clean_snn:.2%}, ANN={clean_ann:.2%}")

    if len(snr_levels) > 1:
        worst_snn = snn_means[-1]
        worst_ann = ann_means[-1]
        drop_snn = clean_snn - worst_snn
        drop_ann = clean_ann - worst_ann

        print(f"At {snr_labels[-1]}: SNN={worst_snn:.2%}, ANN={worst_ann:.2%}")
        print(f"Accuracy drop (clean -> {snr_labels[-1]}): "
              f"SNN drops {drop_snn:.2%}, ANN drops {drop_ann:.2%}")

        if drop_snn < drop_ann:
            print("FINDING: SNN degrades LESS under noise -- more robust to "
                  "naturalistic noise corruption.")
        elif drop_snn > drop_ann:
            print("FINDING: ANN degrades LESS under noise -- SNN is more "
                  "sensitive to noise corruption.")
        else:
            print("FINDING: SNN and ANN degrade equally under noise.")

        # Relative degradation (fraction of clean accuracy lost)
        rel_drop_snn = drop_snn / clean_snn if clean_snn > 0 else float("inf")
        rel_drop_ann = drop_ann / clean_ann if clean_ann > 0 else float("inf")
        print(f"Relative degradation: SNN loses {rel_drop_snn:.1%} of clean acc, "
              f"ANN loses {rel_drop_ann:.1%}")

        results_summary["analysis"] = {
            "clean_snn": float(clean_snn),
            "clean_ann": float(clean_ann),
            "worst_snr_label": snr_labels[-1],
            "worst_snn": float(worst_snn),
            "worst_ann": float(worst_ann),
            "absolute_drop_snn": float(drop_snn),
            "absolute_drop_ann": float(drop_ann),
            "relative_drop_snn": float(rel_drop_snn),
            "relative_drop_ann": float(rel_drop_ann),
            "snn_more_robust": bool(drop_snn < drop_ann),
        }

    # --------------------------------------------------------
    # Save JSON results
    # --------------------------------------------------------
    fold_str = "_".join(str(f) for f in folds)
    json_path = out_dir / f"noise_robustness_folds{fold_str}.json"
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    plot_path = out_dir / f"noise_robustness_folds{fold_str}.png"
    plot_degradation_curves(
        snr_levels, snn_means, snn_stds,
        ann_means, ann_stds, plot_path,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
