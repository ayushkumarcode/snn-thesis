"""
SNN Saliency Maps: Spike-aware Grad-CAM.

Adapts Grad-CAM to work with surrogate gradients in the SNN, producing
saliency maps showing which spectrogram regions drive classification.
Compares SNN vs ANN saliency on the same inputs.

Usage:
    python -m experiments.snn_saliency_maps [--fold 1] [--num-samples 20]
"""

import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, BATCH_SIZE, get_device
from src.dataset import download_esc50, get_fold_dataloaders, get_class_labels
from src.encoding import get_encoder
from src.models.snn_model import SpikingCNN
from src.models.ann_model import ConvANN


class GradCAMSNN:
    """Grad-CAM adapted for SNN with surrogate gradients.

    Hooks into conv2 layer (last conv before FC) and computes
    gradient-weighted activation maps across all timesteps.
    """

    def __init__(self, model, encoder, device):
        self.model = model
        self.encoder = encoder
        self.device = device
        self.activations = []
        self.gradients = []

        # Hook conv2 output (before pool2 and lif2)
        self.model.conv2.register_forward_hook(self._save_activation)
        self.model.conv2.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations.append(output)

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def generate(self, data, target_class=None):
        """Generate Grad-CAM saliency map for one sample."""
        self.activations = []
        self.gradients = []
        self.model.eval()

        # Enable grad for Grad-CAM
        data = data.unsqueeze(0).to(self.device)
        data.requires_grad_(True)

        spk_input = self.encoder(data).to(self.device)
        spk_out, mem_out = self.model(spk_input)

        # Sum membrane potentials across timesteps for classification
        logits = mem_out.sum(dim=0)  # (1, num_classes)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backprop through the target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if not self.gradients:
            return None, target_class

        # Average gradients and activations across timesteps
        # Each timestep produces one activation/gradient pair
        all_grads = torch.stack(self.gradients)  # (T, batch, C, H, W)
        all_acts = torch.stack(self.activations)  # (T, batch, C, H, W)

        # Global average pooling of gradients -> channel weights
        weights = all_grads.mean(dim=(0, 3, 4))  # (batch, C)

        # Weighted combination of activation maps
        cam = torch.zeros_like(all_acts[0, 0, 0])  # (H, W)
        for t in range(all_acts.shape[0]):
            for c in range(all_acts.shape[2]):
                cam += weights[0, c] * all_acts[t, 0, c]

        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.detach().cpu().numpy(), target_class


class GradCAMANN:
    """Standard Grad-CAM for the ANN baseline."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activation = None
        self.gradient = None

        # Hook the last conv layer (features[4] = Conv2d(32,64))
        self.model.features[4].register_forward_hook(self._save_activation)
        self.model.features[4].register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activation = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def generate(self, data, target_class=None):
        """Generate Grad-CAM saliency map for one sample."""
        self.model.eval()
        data = data.unsqueeze(0).to(self.device)
        data.requires_grad_(True)

        logits = self.model(data)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        if self.gradient is None:
            return None, target_class

        # Channel weights from global average pooling of gradients
        weights = self.gradient.mean(dim=(2, 3))  # (1, C)

        # Weighted combination
        cam = torch.zeros_like(self.activation[0, 0])  # (H, W)
        for c in range(self.activation.shape[1]):
            cam += weights[0, c] * self.activation[0, c]

        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.detach().cpu().numpy(), target_class


def run_saliency(fold=1, encoding="direct", num_samples=20, device=None):
    """Generate and compare SNN vs ANN saliency maps."""
    if device is None:
        device = get_device()

    _, test_loader = get_fold_dataloaders(fold, batch_size=1)
    class_labels = get_class_labels()

    # Load SNN
    snn_model = SpikingCNN().to(device)
    snn_path = RESULTS_DIR / "snn" / encoding / f"best_fold{fold}.pt"
    snn_model.load_state_dict(torch.load(snn_path, map_location=device, weights_only=True))
    encoder = get_encoder(encoding)
    snn_cam = GradCAMSNN(snn_model, encoder, device)

    # Load ANN
    ann_model = ConvANN().to(device)
    ann_path = RESULTS_DIR / "ann" / "none" / f"best_fold{fold}.pt"
    ann_model.load_state_dict(torch.load(ann_path, map_location=device, weights_only=True))
    ann_cam = GradCAMANN(ann_model, device)

    save_dir = RESULTS_DIR / "snn" / "saliency"
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    sample_idx = 0

    for data, targets in test_loader:
        if sample_idx >= num_samples:
            break

        target = targets[0].item()
        true_label = class_labels[target]

        # SNN saliency
        snn_map, snn_pred = snn_cam.generate(data[0], target_class=target)
        # ANN saliency
        ann_map, ann_pred = ann_cam.generate(data[0], target_class=target)

        if snn_map is None or ann_map is None:
            continue

        # Compute overlap (IoU of top-50% regions)
        snn_binary = snn_map > 0.5
        ann_binary = ann_map > 0.5
        intersection = (snn_binary & ann_binary).sum()
        union = (snn_binary | ann_binary).sum()
        iou = float(intersection / union) if union > 0 else 0.0

        results.append({
            "sample": sample_idx,
            "true_class": true_label,
            "snn_pred": class_labels[snn_pred],
            "ann_pred": class_labels[ann_pred],
            "iou": iou,
        })

        # Plot comparison (every 5th sample or first 4)
        if sample_idx < 4 or sample_idx % 5 == 0:
            plot_saliency_comparison(
                data[0, 0].cpu().numpy(),
                snn_map, ann_map,
                true_label, snn_pred, ann_pred,
                class_labels, iou,
                save_dir / f"saliency_sample{sample_idx}.png",
            )

        sample_idx += 1
        print(f"  Sample {sample_idx}: {true_label} | "
              f"SNN→{class_labels[snn_pred]} ANN→{class_labels[ann_pred]} | IoU={iou:.3f}")

    # Summary
    mean_iou = np.mean([r["iou"] for r in results]) if results else 0
    snn_correct = sum(1 for r in results if r["snn_pred"] == r["true_class"])
    ann_correct = sum(1 for r in results if r["ann_pred"] == r["true_class"])

    summary = {
        "fold": fold,
        "encoding": encoding,
        "num_samples": len(results),
        "mean_iou": float(mean_iou),
        "snn_correct": snn_correct,
        "ann_correct": ann_correct,
        "samples": results,
    }

    with open(save_dir / f"saliency_fold{fold}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Saliency Summary (Fold {fold})")
    print(f"{'='*50}")
    print(f"  Mean IoU (SNN vs ANN saliency): {mean_iou:.3f}")
    print(f"  SNN correct: {snn_correct}/{len(results)}")
    print(f"  ANN correct: {ann_correct}/{len(results)}")
    if mean_iou < 0.3:
        print(f"  → LOW overlap: SNN and ANN attend to different regions!")
    elif mean_iou > 0.7:
        print(f"  → HIGH overlap: SNN and ANN attend to similar regions")
    else:
        print(f"  → MODERATE overlap")

    return summary


def plot_saliency_comparison(spectrogram, snn_map, ann_map,
                             true_label, snn_pred, ann_pred,
                             class_labels, iou, save_path):
    """Plot spectrogram with SNN and ANN saliency overlays."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Original spectrogram
    axes[0].imshow(spectrogram, aspect="auto", origin="lower", cmap="magma")
    axes[0].set_title(f"Input: {true_label}", fontsize=10)
    axes[0].set_ylabel("Mel bin")

    # SNN saliency overlay
    axes[1].imshow(spectrogram, aspect="auto", origin="lower", cmap="gray", alpha=0.5)
    # Resize saliency map to match spectrogram
    import scipy.ndimage
    snn_resized = scipy.ndimage.zoom(snn_map,
        (spectrogram.shape[0] / snn_map.shape[0],
         spectrogram.shape[1] / snn_map.shape[1]),
        order=1)
    axes[1].imshow(snn_resized, aspect="auto", origin="lower",
                   cmap="jet", alpha=0.6, vmin=0, vmax=1)
    snn_label = class_labels[snn_pred] if isinstance(snn_pred, int) else snn_pred
    axes[1].set_title(f"SNN Grad-CAM → {snn_label}", fontsize=10)

    # ANN saliency overlay
    axes[2].imshow(spectrogram, aspect="auto", origin="lower", cmap="gray", alpha=0.5)
    ann_resized = scipy.ndimage.zoom(ann_map,
        (spectrogram.shape[0] / ann_map.shape[0],
         spectrogram.shape[1] / ann_map.shape[1]),
        order=1)
    axes[2].imshow(ann_resized, aspect="auto", origin="lower",
                   cmap="jet", alpha=0.6, vmin=0, vmax=1)
    ann_label = class_labels[ann_pred] if isinstance(ann_pred, int) else ann_pred
    axes[2].set_title(f"ANN Grad-CAM → {ann_label}", fontsize=10)

    for ax in axes:
        ax.set_xlabel("Time frame")

    fig.suptitle(f"IoU={iou:.3f}", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SNN saliency maps (Grad-CAM)")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--encoding", default="direct")
    args = parser.parse_args()

    download_esc50()
    run_saliency(fold=args.fold, encoding=args.encoding, num_samples=args.num_samples)
