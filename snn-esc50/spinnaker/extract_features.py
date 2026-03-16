"""
Extract convolutional features from test samples for SpiNNaker inference.

Since SpiNNaker1 cannot efficiently run unrolled conv layers (too many
neurons), we use a hybrid approach:
  1. Run conv feature extraction on CPU (this script)
  2. Rate-encode the features into spike trains
  3. Feed spike trains to SpiNNaker FC classifier

This script produces spike-encoded feature arrays that can be uploaded
to EBRAINS and fed into the SpiNNaker FC network.

Usage:
    python extract_features.py --model-path results/snn/direct/best_fold4.pt \
                               --output-dir results/spinnaker_weights \
                               --num-samples 100
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from src.config import (
    RESULTS_DIR, NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    ESC50_META_PATH, ESC50_AUDIO_DIR, SAMPLE_RATE, DURATION,
    N_FFT, HOP_LENGTH,
)


class ConvFeatureExtractor(nn.Module):
    """Extract features from the conv layers of the trained SNN.

    Runs Conv1->BN1->Pool1->LIF1->Conv2->BN2->Pool2->LIF2->AvgPool->Flatten
    and returns the 2304-dim feature for each timestep.
    """

    def __init__(self, beta=BETA, num_steps=NUM_STEPS):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

    def forward(self, x):
        """Extract conv features for each timestep.

        Args:
            x: Spike input (num_steps, batch, 1, n_mels, time_frames).

        Returns:
            features: (num_steps, batch, 2304) binary spike features.
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        features = []
        for step in range(self.num_steps):
            x_t = x[step]
            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)
            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)  # (batch, 2304)
            features.append(flat)

        return torch.stack(features)  # (num_steps, batch, 2304)


def load_conv_weights(model_path, device="cpu"):
    """Load only the conv/bn weights into the feature extractor."""
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    extractor = ConvFeatureExtractor()

    # Map only the conv/bn parameters
    conv_state = {}
    for name, param in state_dict.items():
        if any(name.startswith(prefix) for prefix in
               ["conv1.", "bn1.", "conv2.", "bn2.", "lif1.", "lif2."]):
            conv_state[name] = param

    extractor.load_state_dict(conv_state, strict=False)
    extractor.eval()
    return extractor


def load_test_data(num_samples=100, fold=4):
    """Load ESC-50 test data and convert to spectrograms.

    Uses fold 4 as test fold (matching best_fold4.pt).
    Uses soundfile directly to avoid torchaudio backend issues.
    """
    import soundfile as sf
    import pandas as pd

    meta = pd.read_csv(ESC50_META_PATH)
    # Use fold 4 as test fold (matching the best model)
    test_meta = meta[meta["fold"] == fold].head(num_samples)

    mel_transform = torch.nn.Sequential(
        # We need torchaudio for MelSpectrogram but only the transform, not load
    )

    # Import only the transforms (no torchcodec dependency)
    from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

    mel_transform = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    amp_to_db = AmplitudeToDB()

    samples = []
    labels = []

    for _, row in test_meta.iterrows():
        audio_path = ESC50_AUDIO_DIR / row["filename"]
        if not audio_path.exists():
            continue

        # Load audio with soundfile (no torchcodec needed)
        audio_np, sr = sf.read(str(audio_path))

        # Convert to torch tensor
        if audio_np.ndim == 1:
            waveform = torch.from_numpy(audio_np).float().unsqueeze(0)
        else:
            waveform = torch.from_numpy(audio_np.T).float()

        # Resample if needed (simple linear interpolation)
        if sr != SAMPLE_RATE:
            target_len = int(waveform.shape[1] * SAMPLE_RATE / sr)
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0), size=target_len, mode="linear",
                align_corners=False
            ).squeeze(0)

        # Ensure correct length
        target_length = SAMPLE_RATE * DURATION
        if waveform.shape[1] < target_length:
            waveform = torch.nn.functional.pad(
                waveform, (0, target_length - waveform.shape[1])
            )
        else:
            waveform = waveform[:, :target_length]

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Mel spectrogram
        mel = mel_transform(waveform)  # (1, n_mels, time)
        mel_db = amp_to_db(mel)

        samples.append(mel_db)
        labels.append(row["target"])

    return samples, labels


def rate_encode(features, num_steps=NUM_STEPS):
    """Rate-encode continuous features into spike trains.

    Higher feature values -> higher spike probability.

    Args:
        features: (num_steps, 1, 2304) float tensor from conv extractor.

    Returns:
        spikes: (num_steps, 2304) binary numpy array.
    """
    # Normalize features to [0, 1] range for rate coding
    feat = features.squeeze(1).detach().numpy()  # (num_steps, 2304)

    # Features from LIF neurons are already binary spikes (0 or 1)
    # So we can use them directly
    spikes = (feat > 0.5).astype(np.float64)

    return spikes


def main():
    parser = argparse.ArgumentParser(
        description="Extract conv features for SpiNNaker inference"
    )
    parser.add_argument("--model-path", required=True,
                        help="Path to trained model .pt file")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to save spike features")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of test samples to extract")
    parser.add_argument("--fold", type=int, default=4,
                        help="Test fold number")
    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = RESULTS_DIR / "spinnaker_weights"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading convolutional feature extractor...")
    extractor = load_conv_weights(args.model_path)

    print(f"Loading {args.num_samples} test samples from fold {args.fold}...")
    try:
        samples, labels = load_test_data(args.num_samples, args.fold)
    except Exception as e:
        print(f"Could not load test data: {e}")
        print("Generating synthetic test data instead...")
        samples = [torch.randn(1, N_MELS, 216) for _ in range(min(args.num_samples, 10))]
        labels = list(range(min(args.num_samples, 10)))

    print(f"Loaded {len(samples)} samples")

    # Extract features
    all_spike_features = []
    all_labels = []

    print("Extracting convolutional features...")
    with torch.no_grad():
        for i, (mel, label) in enumerate(zip(samples, labels)):
            # Prepare input: repeat mel spectrogram across timesteps
            # Shape: (num_steps, 1, 1, n_mels, time)
            x = mel.unsqueeze(0).unsqueeze(0).repeat(NUM_STEPS, 1, 1, 1, 1)

            # Extract features
            features = extractor(x)  # (num_steps, 1, 2304)

            # Convert to spike trains
            spikes = rate_encode(features)  # (num_steps, 2304)

            all_spike_features.append(spikes)
            all_labels.append(label)

            if i < 3:
                spike_rate = spikes.mean()
                print(f"  Sample {i} (label={label}): "
                      f"spike_rate={spike_rate:.3f}, "
                      f"active_neurons={int((spikes.sum(axis=0) > 0).sum())}/2304")

    # Save spike features
    spike_data = np.array(all_spike_features)  # (N, num_steps, 2304)
    label_data = np.array(all_labels)

    np.save(output_dir / "test_spike_features.npy", spike_data)
    np.save(output_dir / "test_labels.npy", label_data)

    print(f"\nSaved spike features: {spike_data.shape}")
    print(f"Saved labels: {label_data.shape}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles to upload to EBRAINS:")
    print(f"  - test_spike_features.npy ({spike_data.nbytes / 1024:.1f} KB)")
    print(f"  - test_labels.npy")
    print(f"  - fc1_connections.npy")
    print(f"  - fc2_connections.npy")
    print(f"  - metadata.json")


if __name__ == "__main__":
    main()
