"""
ESC-50 dataset loader with mel-spectrogram preprocessing.

The ESC-50 dataset contains 2000 environmental audio recordings
(5 seconds each), organised into 50 classes with 40 clips per class.
It comes with 5 predefined folds for cross-validation.
"""

import os
import random
import zipfile
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    DATA_DIR, ESC50_DIR, ESC50_AUDIO_DIR, ESC50_META_PATH, ESC50_URL,
    SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH, F_MIN, F_MAX, DURATION,
    BATCH_SIZE, NUM_FOLDS,
)


def download_esc50():
    """Download and extract ESC-50 dataset if not already present."""
    if ESC50_AUDIO_DIR.exists() and len(list(ESC50_AUDIO_DIR.glob("*.wav"))) > 0:
        print(f"ESC-50 already downloaded at {ESC50_DIR}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "ESC-50-master.zip"

    print("Downloading ESC-50 dataset...")
    urllib.request.urlretrieve(ESC50_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    zip_path.unlink()
    print(f"ESC-50 ready at {ESC50_DIR}")


def wav_to_mel_spectrogram(filepath: str) -> np.ndarray:
    """Load a WAV file and convert to log-mel spectrogram.

    Returns:
        np.ndarray of shape (n_mels, time_frames)
    """
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)

    # Pad to exactly DURATION seconds if shorter
    expected_len = SAMPLE_RATE * DURATION
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmin=F_MIN, fmax=F_MAX,
    )
    # Convert to log scale (dB)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db


def normalise_spectrogram(mel_db: np.ndarray) -> np.ndarray:
    """Min-max normalise a spectrogram to [0, 1]."""
    min_val = mel_db.min()
    max_val = mel_db.max()
    if max_val - min_val == 0:
        return np.zeros_like(mel_db)
    return (mel_db - min_val) / (max_val - min_val)


class SpecAugment:
    """SpecAugment: frequency masking + time masking on a mel spectrogram tensor.

    Applied only during training to reduce overfitting on the 1,600-sample
    ESC-50 training set. Parameters chosen to be mild — we mask at most
    12.5% of frequency bins and ~9% of time frames per mask.

    Args:
        freq_mask_param: Max number of frequency bins to mask per mask (F).
        time_mask_param: Max number of time frames to mask per mask (T).
        num_freq_masks: Number of independent frequency masks to apply.
        num_time_masks: Number of independent time masks to apply.
    """

    def __init__(self, freq_mask_param: int = 8, time_mask_param: int = 20,
                 num_freq_masks: int = 2, num_time_masks: int = 2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment to a (1, n_mels, time) spectrogram tensor."""
        _, n_mels, n_time = spec.shape
        spec = spec.clone()

        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            if f > 0:
                f0 = random.randint(0, max(0, n_mels - f))
                spec[:, f0:f0 + f, :] = 0.0

        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            if t > 0:
                t0 = random.randint(0, max(0, n_time - t))
                spec[:, :, t0:t0 + t] = 0.0

        return spec


class TimeShift:
    """Randomly shift the spectrogram along the time axis (circular roll).

    Simulates a different temporal offset for the same sound event,
    which is cheap and effective for audio.

    Args:
        max_shift_frac: Maximum shift as a fraction of total time frames.
                        Default 0.1 = ±10% = ±21 frames of 216.
    """

    def __init__(self, max_shift_frac: float = 0.1):
        self.max_shift_frac = max_shift_frac

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        _, _, n_time = spec.shape
        max_shift = int(self.max_shift_frac * n_time)
        if max_shift == 0:
            return spec
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(spec, shift, dims=2)


class ComposeAugment:
    """Apply a list of augmentation transforms sequentially."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            spec = t(spec)
        return spec


def get_train_augmentation() -> ComposeAugment:
    """Return the standard training augmentation pipeline.

    SpecAugment (freq + time masking) followed by random time shift.
    These are the two augmentations most commonly used for audio
    mel-spectrogram classification and require no additional data.
    """
    return ComposeAugment([
        SpecAugment(freq_mask_param=8, time_mask_param=20,
                    num_freq_masks=2, num_time_masks=2),
        TimeShift(max_shift_frac=0.1),
    ])


class ESC50Dataset(Dataset):
    """PyTorch Dataset for ESC-50.

    Args:
        folds: List of fold numbers to include (1-5).
        transform: Optional transform applied to the spectrogram tensor.
        precompute: If True, load and cache all spectrograms in memory.
    """

    def __init__(self, folds: list[int], transform=None, precompute=True):
        self.transform = transform
        self.precompute = precompute

        meta = pd.read_csv(ESC50_META_PATH)
        self.meta = meta[meta["fold"].isin(folds)].reset_index(drop=True)

        self.data = []
        self.labels = []

        if precompute:
            print(f"Loading {len(self.meta)} clips from folds {folds}...")
            for _, row in self.meta.iterrows():
                filepath = ESC50_AUDIO_DIR / row["filename"]
                mel = wav_to_mel_spectrogram(str(filepath))
                mel = normalise_spectrogram(mel)
                self.data.append(mel)
                self.labels.append(row["target"])

            self.data = np.array(self.data, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.precompute:
            mel = self.data[idx]
            label = self.labels[idx]
        else:
            row = self.meta.iloc[idx]
            filepath = ESC50_AUDIO_DIR / row["filename"]
            mel = wav_to_mel_spectrogram(str(filepath))
            mel = normalise_spectrogram(mel)
            label = row["target"]

        # Shape: (1, n_mels, time_frames) -- single channel
        tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, torch.tensor(label, dtype=torch.long)


def get_fold_dataloaders(test_fold: int, batch_size: int = BATCH_SIZE,
                         augment: bool = False):
    """Create train/test DataLoaders for a given test fold.

    ESC-50 uses 5-fold CV: 4 folds train, 1 fold test.

    Args:
        test_fold: Fold number (1-5) to use as test set.
        batch_size: Batch size for DataLoaders.
        augment: If True, apply SpecAugment + TimeShift to training data only.
                 Test data is never augmented.

    Returns:
        (train_loader, test_loader)
    """
    train_folds = [f for f in range(1, NUM_FOLDS + 1) if f != test_fold]

    train_transform = get_train_augmentation() if augment else None
    train_dataset = ESC50Dataset(folds=train_folds, transform=train_transform)
    test_dataset = ESC50Dataset(folds=[test_fold], transform=None)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    return train_loader, test_loader


def get_class_labels() -> list[str]:
    """Return the 50 ESC-50 class labels in order."""
    meta = pd.read_csv(ESC50_META_PATH)
    labels = meta.drop_duplicates("target").sort_values("target")["category"].tolist()
    return labels
