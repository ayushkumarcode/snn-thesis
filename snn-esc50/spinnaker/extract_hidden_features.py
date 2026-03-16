"""
extract_hidden_features.py -- Extract FC1/lif3 hidden spike trains from the trained SNN.

Problem with previous SpiNNaker approach:
    FC1 (2304 -> 256) receives ~1398 simultaneously active inputs per timestep.
    FC1 weights are near-zero-mean (mean=-0.0034, range=[-0.301, +0.282]).
    With 1398 simultaneous spikes: excitatory/inhibitory currents cancel.
    Net current to hidden neurons ≈ 0 regardless of weight_scale.
    No hidden neurons ever fire.

Solution: FC2-only SpiNNaker deployment
    1. Run conv+bn+pool+lif (x2) + avgpool + flatten + fc1 + lif3 on CPU (snnTorch).
    2. Record binary hidden spike trains: (N_samples, 25, 256).
    3. Deploy ONLY FC2 (256 -> 50) on SpiNNaker.

Why this works:
    - Hidden spikes (lif3 output) are SPARSE: ~10-30% of 256 neurons per timestep.
    - They are DATA-DEPENDENT: pattern reflects the actual sound content.
    - Max 256 simultaneous spikes (vs 1398) -- well within SpiNNaker router capacity.
    - FC2 weights can distinguish classes via the sparse hidden patterns.

Usage (regular snnTorch env, NOT venv-spinnaker):
    source .venv/bin/activate
    cd snn-esc50/
    python spinnaker/extract_hidden_features.py \\
        --model-path results/snn/direct/best_fold4.pt \\
        --num-samples 20 --fold 4

    # Or for a quick 5-sample test:
    python spinnaker/extract_hidden_features.py \\
        --model-path results/snn/direct/best_fold4.pt \\
        --num-samples 5 --fold 4

Outputs (saved to results/spinnaker_weights/):
    hidden_spike_features.npy   -- (N, 25, 256) binary float64 hidden spikes
    hidden_mem_final.npy        -- (N, 256) float32 membrane at last timestep
    snn_predictions.npy         -- (N,) int32 what the full snnTorch model predicts
    hidden_labels.npy           -- (N,) int32 true labels
    hidden_metadata.json        -- statistics, config, and analysis notes
"""

import argparse
import json
import sys
from pathlib import Path

# Add repo root (snn-esc50/) to sys.path so 'src' is importable
# when the script is invoked as 'python spinnaker/extract_hidden_features.py'
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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

# ============================================================
# Model classes
# ============================================================

class HiddenFeatureExtractor(nn.Module):
    """Run conv+bn+pool+lif (x2) + avgpool + fc1 + lif3 from the trained SNN.

    Produces hidden spike trains (spk3) and membrane potentials (mem3).
    These are the output of the hidden layer (lif3), which feeds into FC2
    for final classification.

    Key properties of spk3:
        - Binary (0 or 1): genuine spikes from the trained LIF neuron
        - Sparse: typically 10-30% of 256 neurons fire per timestep
        - Data-dependent: different sounds -> different firing patterns
        - Already trained: model weights encode sound-class relationships
    """

    def __init__(self, beta: float = BETA, num_steps: int = NUM_STEPS):
        super().__init__()
        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Conv block 1 (same architecture as SpikingCNN)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # AvgPool: (64, 16, 54) -> (64, 4, 9) = 2304 features
        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # FC block 1: 2304 -> 256
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x: torch.Tensor):
        """Extract hidden spike trains from conv+FC1+lif3.

        Args:
            x: (num_steps, batch, 1, n_mels, time_frames) spectrogram input.
               For direct encoding: same spectrogram repeated across timesteps.

        Returns:
            spk3_rec: (num_steps, batch, 256) binary float tensor.
            mem3_rec: (num_steps, batch, 256) float membrane potentials.
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk3_rec = []
        mem3_rec = []

        for step in range(self.num_steps):
            x_t = x[step]  # (batch, 1, n_mels, time_frames)

            # Conv block 1
            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, mem1 = self.lif1(cur1, mem1)

            # Conv block 2
            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            # Pool + flatten: (batch, 2304)
            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            # FC1 + hidden LIF
            cur3 = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        return torch.stack(spk3_rec), torch.stack(mem3_rec)


class FullSNN(nn.Module):
    """Full SNN (conv + FC1 + FC2) for reference predictions.

    Used to compute what the snnTorch model predicts on the same samples,
    so we can compare SpiNNaker FC2-only results against the correct baseline.
    """

    def __init__(self, beta: float = BETA, num_steps: int = NUM_STEPS):
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

        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(256, NUM_CLASSES)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x: torch.Tensor):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk4_rec = []
        mem4_rec = []

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

            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk4_rec.append(spk4)
            mem4_rec.append(mem4)

        return torch.stack(spk4_rec), torch.stack(mem4_rec)


# ============================================================
# Weight loading
# ============================================================

def load_hidden_extractor(model_path: str, device: str = "cpu") -> HiddenFeatureExtractor:
    """Load trained weights into HiddenFeatureExtractor.

    Copies conv1/bn1/lif1/conv2/bn2/lif2/fc1/lif3 weights from the
    full SpikingCNN state dict.
    """
    state_dict = torch.load(model_path, map_location=device, weights_only=True)

    extractor = HiddenFeatureExtractor()

    # All the layer names match exactly (conv1, bn1, lif1, conv2, bn2, lif2, fc1, lif3)
    conv_fc1_state = {}
    wanted_prefixes = ["conv1.", "bn1.", "lif1.", "conv2.", "bn2.", "lif2.", "fc1.", "lif3."]
    for name, param in state_dict.items():
        if any(name.startswith(p) for p in wanted_prefixes):
            conv_fc1_state[name] = param

    extractor.load_state_dict(conv_fc1_state, strict=True)
    extractor.eval()
    return extractor


def load_full_snn(model_path: str, device: str = "cpu") -> FullSNN:
    """Load trained weights into FullSNN for comparison."""
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model = FullSNN()
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ============================================================
# Data loading (reused from extract_features.py)
# ============================================================

def load_test_data(num_samples: int = 100, fold: int = 4):
    """Load ESC-50 test samples from the given fold.

    Uses EXACTLY the same pipeline as src/dataset.py:
        librosa.load → melspectrogram → power_to_db → min-max normalise to [0,1]

    This is critical: the model was trained on this pipeline.
    Previous bug: used torchaudio MelSpectrogram + AmplitudeToDB with NO normalisation,
    causing 5.8% inference accuracy vs 54% expected (3 March 2026 diagnosis).

    Args:
        num_samples: Maximum number of samples to load.
        fold: Which fold to use as test set.

    Returns:
        (samples, labels): list of mel spectrogram tensors (1, n_mels, time),
                           list of integer class labels.
    """
    import librosa
    import pandas as pd

    meta = pd.read_csv(ESC50_META_PATH)
    test_meta = meta[meta["fold"] == fold].head(num_samples)

    samples = []
    labels = []

    for _, row in test_meta.iterrows():
        audio_path = ESC50_AUDIO_DIR / row["filename"]
        if not audio_path.exists():
            print(f"  WARNING: {audio_path} not found, skipping.")
            continue

        # Step 1: load with librosa (identical to dataset.py wav_to_mel_spectrogram)
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, duration=DURATION)

        # Step 2: pad to exactly DURATION seconds if shorter
        expected_len = SAMPLE_RATE * DURATION
        if len(y) < expected_len:
            y = np.pad(y, (0, expected_len - len(y)))

        # Step 3: mel spectrogram with same params as training
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH,
            fmin=0, fmax=None,
        )

        # Step 4: log scale (dB) with same ref as training
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Step 5: min-max normalise to [0, 1] (identical to dataset.py normalise_spectrogram)
        min_val = mel_db.min()
        max_val = mel_db.max()
        if max_val - min_val == 0:
            mel_norm = np.zeros_like(mel_db)
        else:
            mel_norm = (mel_db - min_val) / (max_val - min_val)

        # Step 6: convert to tensor (1, n_mels, time_frames)
        tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)

        samples.append(tensor)
        labels.append(int(row["target"]))

    return samples, labels


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract FC1/lif3 hidden spike trains from trained SNN"
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to trained snnTorch model .pt file (e.g. results/snn/direct/best_fold4.pt)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to save outputs (default: results/spinnaker_weights/)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=20,
        help="Number of test samples to extract (default: 20)"
    )
    parser.add_argument(
        "--fold", type=int, default=4,
        help="Test fold number matching the model (default: 4)"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device (default: cpu; use mps or cuda if available)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "spinnaker_weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("extract_hidden_features.py")
    print("=" * 60)
    print(f"  Model      : {args.model_path}")
    print(f"  Fold       : {args.fold}")
    print(f"  Samples    : {args.num_samples}")
    print(f"  Device     : {args.device}")
    print(f"  Output dir : {output_dir}")
    print()

    # ---- Load models ----
    print("Loading HiddenFeatureExtractor (conv + FC1 + lif3)...")
    extractor = load_hidden_extractor(args.model_path, args.device)
    print("  OK")

    print("Loading FullSNN (for reference predictions)...")
    full_snn = load_full_snn(args.model_path, args.device)
    print("  OK")
    print()

    # ---- Load test data ----
    print(f"Loading {args.num_samples} samples from fold {args.fold}...")
    try:
        samples, labels = load_test_data(args.num_samples, args.fold)
        print(f"  Loaded {len(samples)} samples")
    except Exception as e:
        print(f"  ERROR loading test data: {e}")
        raise

    print()

    # ---- Extract hidden spike trains ----
    all_hidden_spikes = []    # (N, 25, 256) binary
    all_hidden_mem_final = [] # (N, 256) final membrane
    all_snn_predictions = []  # (N,) what the full model predicts
    all_labels = []           # (N,) true labels

    hidden_stats_per_sample = []

    print("Extracting hidden spike trains...")
    print(f"  {'Sample':>6}  {'Label':>5}  "
          f"{'SNNpred':>7}  {'Correct':>7}  "
          f"{'AvgActive':>9}  {'TotalSpk':>8}  {'Sparsity':>8}")
    print("  " + "-" * 68)

    with torch.no_grad():
        for i, (mel, label) in enumerate(zip(samples, labels)):
            # Prepare input: repeat mel spectrogram across timesteps (direct encoding)
            # Shape: (num_steps, 1, 1, n_mels, time)
            x = mel.unsqueeze(0).unsqueeze(0).repeat(NUM_STEPS, 1, 1, 1, 1)
            # Note: x is (25, 1, 1, 64, 216)

            # Extract hidden spikes
            spk3, mem3 = extractor(x)
            # spk3: (25, 1, 256), mem3: (25, 1, 256)

            spk3_np = spk3.squeeze(1).numpy()       # (25, 256) float32
            mem3_np = mem3[-1].squeeze(0).numpy()   # (256,) final membrane

            # Get full model prediction
            spk4, mem4 = full_snn(x)
            # spk4: (25, 1, 50) -- output spikes
            # mem4: (25, 1, 50) -- output membranes
            spk_counts = spk4.squeeze(1).sum(dim=0).numpy()  # (50,)
            if spk_counts.sum() > 0:
                pred = int(np.argmax(spk_counts))
                pred_basis = "spike count"
            else:
                # Fallback to final membrane voltage
                final_mem = mem4[-1].squeeze(0).numpy()  # (50,)
                pred = int(np.argmax(final_mem))
                pred_basis = "final membrane"

            # Statistics
            active_per_step = (spk3_np > 0.5).sum(axis=1)  # (25,) spikes/step
            total_hidden_spikes = int((spk3_np > 0.5).sum())
            avg_active = float(active_per_step.mean())
            max_active = int(active_per_step.max())
            sparsity = 1.0 - (total_hidden_spikes / (25 * 256))
            neurons_that_fired = int(((spk3_np > 0.5).sum(axis=0) > 0).sum())

            correct = (pred == label)

            print(f"  {i:>6}  {label:>5}  {pred:>7}  {str(correct):>7}  "
                  f"{avg_active:>9.1f}  {total_hidden_spikes:>8}  {sparsity:>8.3f}")

            # Collect
            all_hidden_spikes.append(spk3_np.astype(np.float64))
            all_hidden_mem_final.append(mem3_np.astype(np.float32))
            all_snn_predictions.append(pred)
            all_labels.append(label)

            hidden_stats_per_sample.append({
                "sample": i,
                "true_label": label,
                "snn_predicted": pred,
                "snn_correct": bool(correct),
                "snn_pred_basis": pred_basis,
                "total_hidden_spikes": total_hidden_spikes,
                "avg_active_per_step": float(avg_active),
                "max_active_per_step": int(max_active),
                "sparsity": float(sparsity),
                "neurons_that_fired": int(neurons_that_fired),
                "spikes_per_step": active_per_step.tolist(),
            })

    # ---- Summary statistics ----
    print()
    all_hidden_array = np.array(all_hidden_spikes)  # (N, 25, 256)
    total_samples = len(all_hidden_spikes)
    snn_correct = sum(1 for s in hidden_stats_per_sample if s["snn_correct"])
    snn_accuracy = snn_correct / total_samples if total_samples > 0 else 0.0

    mean_total_spikes = np.mean([s["total_hidden_spikes"] for s in hidden_stats_per_sample])
    mean_avg_active = np.mean([s["avg_active_per_step"] for s in hidden_stats_per_sample])
    mean_sparsity = np.mean([s["sparsity"] for s in hidden_stats_per_sample])
    mean_neurons_fired = np.mean([s["neurons_that_fired"] for s in hidden_stats_per_sample])

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Samples extracted       : {total_samples}")
    print(f"  snnTorch accuracy       : {snn_correct}/{total_samples} = {snn_accuracy:.1%}")
    print(f"  Mean hidden spikes/sample: {mean_total_spikes:.1f}")
    print(f"  Mean active/timestep    : {mean_avg_active:.1f} / 256 neurons")
    print(f"  Mean sparsity           : {mean_sparsity:.3f} ({mean_sparsity*100:.1f}% silent)")
    print(f"  Mean neurons that fired : {mean_neurons_fired:.1f} / 256 across all timesteps")
    print()
    print("SpiNNaker FC2-only deployment will use:")
    print(f"  Input: {total_samples} samples × 25 timesteps × ~{mean_avg_active:.0f} spikes/step")
    print(f"  Network: 256-neuron SpikeSourceArray -> 50-neuron IF_curr_exp (FC2 only)")
    print(f"  Max simultaneous spikes: {int(np.max([s['max_active_per_step'] for s in hidden_stats_per_sample]))}")
    print()

    # ---- Save outputs ----
    hidden_path = output_dir / "hidden_spike_features.npy"
    mem_path = output_dir / "hidden_mem_final.npy"
    pred_path = output_dir / "snn_predictions.npy"
    labels_path = output_dir / "hidden_labels.npy"
    meta_path = output_dir / "hidden_metadata.json"

    np.save(hidden_path, all_hidden_array)
    np.save(mem_path, np.array(all_hidden_mem_final))
    np.save(pred_path, np.array(all_snn_predictions, dtype=np.int32))
    np.save(labels_path, np.array(all_labels, dtype=np.int32))

    metadata = {
        "model_path": str(args.model_path),
        "fold": args.fold,
        "num_samples": total_samples,
        "timestamp": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "snn_accuracy": float(snn_accuracy),
        "snn_correct": int(snn_correct),
        "architecture": {
            "input_shape": [1, 64, 216],
            "hidden_size": 256,
            "output_size": 50,
            "num_steps": NUM_STEPS,
        },
        "snn_params": {
            "beta": BETA,
            "threshold": 1.0,
            "num_steps": NUM_STEPS,
        },
        "hidden_activity": {
            "mean_total_spikes_per_sample": float(mean_total_spikes),
            "mean_active_per_step": float(mean_avg_active),
            "mean_sparsity": float(mean_sparsity),
            "mean_neurons_that_fired": float(mean_neurons_fired),
            "max_simultaneous_spikes": int(np.max(
                [s["max_active_per_step"] for s in hidden_stats_per_sample]
            )),
        },
        "deployment_notes": [
            "hidden_spike_features.npy: (N, 25, 256) binary float64",
            "Feed to SpiNNaker as SpikeSourceArray with 256 inputs",
            "Connect to FC2 (256->50) IF_curr_exp population",
            "Use fc2_connections.npy from results/spinnaker_weights/",
            "Weight scale: start with 5-10x (lower than FC1 due to sparser input)",
            "Expected: output neurons fire selectively for correct class",
            "Root cause of previous failure: FC1 had 1398 simultaneous inputs with near-zero-mean weights",
            "This FC2-only approach bypasses that cancellation entirely",
        ],
        "per_sample": hidden_stats_per_sample,
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved:")
    print(f"  {hidden_path}  ({all_hidden_array.nbytes // 1024} KB)")
    print(f"  {mem_path}")
    print(f"  {pred_path}")
    print(f"  {labels_path}")
    print(f"  {meta_path}")
    print()
    print("Next step: Run SpiNNaker FC2-only inference with:")
    print("  source .venv-spinnaker/bin/activate")
    print("  python spinnaker/run_fc2_spinnaker.py")


if __name__ == "__main__":
    main()
