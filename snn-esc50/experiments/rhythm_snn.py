"""
rhythm_snn.py -- Oscillatory Modulation (Rhythm-SNN) for ESC-50.

Implements learnable oscillatory modulation of membrane potentials, inspired by
Zhao et al. (Nature Communications 2025) "Rhythm-SNN":
    v[t] = beta * v[t-1] + I[t] + A * sin(2*pi*f*t/T + phi)

where A (amplitude), f (frequency), phi (phase) are LEARNABLE per-neuron
parameters. This adds a biologically-motivated oscillatory current that
can help temporal feature extraction in audio classification.

Model also uses learn_beta=True, spike_rate_escape surrogate, Dropout(0.3).

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/rhythm_snn.py --fold 1 --device mps
    python experiments/rhythm_snn.py                      # all 5 folds
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    NUM_FOLDS, RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct


# ============================================================
# RhythmLIF Neuron
# ============================================================

class RhythmLIF(nn.Module):
    """Leaky Integrate-and-Fire neuron with learnable oscillatory modulation.

    Membrane dynamics:
        v[t] = beta * v[t-1] * (1 - spk[t-1]) + I[t] + A * sin(2*pi*f*t/T + phi)

    When v[t] >= threshold, a spike is emitted and v is reset (soft reset via
    multiplication by (1 - spk), same as snnTorch Leaky default).

    Parameters:
        neuron_shape: Shape of the neuron population (e.g., (32,) for conv channels,
                      (256,) for FC hidden layer). Oscillation params have this shape.
        beta: Initial membrane decay rate (learnable).
        threshold: Spike threshold (fixed at 1.0).
        num_steps: Total simulation timesteps T (for oscillation period).
        spike_grad: Surrogate gradient function for backward pass.
    """

    def __init__(
        self,
        neuron_shape: tuple,
        beta: float = BETA,
        threshold: float = 1.0,
        num_steps: int = NUM_STEPS,
        spike_grad=None,
    ):
        super().__init__()
        self.threshold = threshold
        self.num_steps = num_steps

        if spike_grad is None:
            spike_grad = surrogate.spike_rate_escape(beta=1.0, slope=25)
        self.spike_grad = spike_grad

        # Learnable membrane decay
        # Store as raw parameter, sigmoid it in forward for (0, 1) constraint
        self.beta_raw = nn.Parameter(torch.full(neuron_shape, math.log(beta / (1 - beta))))

        # Learnable oscillation parameters (per-neuron)
        self.amplitude = nn.Parameter(torch.full(neuron_shape, 0.1))
        self.frequency = nn.Parameter(torch.empty(neuron_shape).uniform_(0.5, 5.0))
        self.phase = nn.Parameter(torch.zeros(neuron_shape))

    @property
    def beta(self):
        """Membrane decay constrained to (0, 1) via sigmoid."""
        return torch.sigmoid(self.beta_raw)

    def _oscillation(self, t: int) -> torch.Tensor:
        """Compute oscillatory modulation at timestep t.

        Returns tensor with shape matching neuron_shape, broadcastable
        to the membrane potential tensor.
        """
        angle = 2.0 * math.pi * self.frequency * t / self.num_steps + self.phase
        return self.amplitude * torch.sin(angle)

    def init_rhythm(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize membrane potential to zeros.

        Returns:
            mem: Zeros with shape (batch_size, *neuron_shape).
        """
        return torch.zeros(batch_size, *self.amplitude.shape, device=device)

    def forward(
        self, input_current: torch.Tensor, mem: torch.Tensor, t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single timestep forward.

        Args:
            input_current: Synaptic input, shape (batch, *neuron_shape, ...).
            mem: Previous membrane potential, same shape as input_current.
            t: Current timestep index (0-indexed).

        Returns:
            spk: Output spikes (binary), same shape as mem.
            mem: Updated membrane potential.
        """
        beta = self.beta  # (neuron_shape)
        osc = self._oscillation(t)  # (neuron_shape)

        # Broadcast beta and osc to match input_current dimensions
        # For conv layers: input_current is (batch, channels, H, W), neuron_shape is (channels,)
        # For FC layers: input_current is (batch, hidden), neuron_shape is (hidden,)
        ndim_extra = input_current.dim() - 1 - len(self.amplitude.shape)
        beta_bc = beta
        osc_bc = osc
        for _ in range(ndim_extra):
            beta_bc = beta_bc.unsqueeze(-1)
            osc_bc = osc_bc.unsqueeze(-1)

        # Leaky integration with oscillatory modulation
        mem = beta_bc * mem + input_current + osc_bc

        # Spike generation with surrogate gradient
        spk = self.spike_grad(mem - self.threshold)

        # Soft reset: mem *= (1 - spk)
        mem = mem * (1.0 - spk.detach())

        return spk, mem


# ============================================================
# Rhythm-SNN Model
# ============================================================

class RhythmSpikingCNN(nn.Module):
    """Convolutional SNN with RhythmLIF neurons for ESC-50.

    Same architecture as baseline SpikingCNN but all 4 LIF neurons
    are replaced with RhythmLIF neurons that add learnable oscillatory
    modulation to membrane dynamics. Includes Dropout(0.3) for
    regularisation.

    Architecture:
        Conv2d(1,32) -> BN -> MaxPool(2) -> RhythmLIF
        Conv2d(32,64) -> BN -> MaxPool(2) -> RhythmLIF
        AvgPool(4,6) -> Flatten
        FC(2304,256) -> RhythmLIF -> Dropout(0.3)
        FC(256,50) -> RhythmLIF
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

        # Conv block 1: output is (batch, 32, 32, 108) after MaxPool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = RhythmLIF(
            neuron_shape=(32,), beta=beta, num_steps=num_steps, spike_grad=spike_grad,
        )

        # Conv block 2: output is (batch, 64, 16, 54) after MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = RhythmLIF(
            neuron_shape=(64,), beta=beta, num_steps=num_steps, spike_grad=spike_grad,
        )

        # Pooling: (64, 16, 54) -> (64, 4, 9) -> flatten -> 2304
        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # FC block 1
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = RhythmLIF(
            neuron_shape=(256,), beta=beta, num_steps=num_steps, spike_grad=spike_grad,
        )
        self.dropout = nn.Dropout(0.3)

        # FC block 2 (output)
        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = RhythmLIF(
            neuron_shape=(num_classes,), beta=beta, num_steps=num_steps, spike_grad=spike_grad,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over all timesteps.

        Args:
            x: Input of shape (num_steps, batch, 1, n_mels, time_frames).

        Returns:
            spk_out: Output spikes, shape (num_steps, batch, num_classes).
            mem_out: Output membrane potentials, shape (num_steps, batch, num_classes).
        """
        batch_size = x.shape[1]
        device = x.device

        # Initialize membrane potentials
        mem1 = self.lif1.init_rhythm(batch_size, device)
        mem2 = self.lif2.init_rhythm(batch_size, device)
        mem3 = self.lif3.init_rhythm(batch_size, device)
        mem4 = self.lif4.init_rhythm(batch_size, device)

        # Expand mem for conv layers: (batch, C) -> (batch, C, H, W)
        # Will be set correctly on first iteration
        mem1_initialized = False
        mem2_initialized = False

        spk_out_rec = []
        mem_out_rec = []

        for step in range(self.num_steps):
            x_t = x[step]  # (batch, 1, n_mels, time)

            # Conv block 1
            cur1 = self.pool1(self.bn1(self.conv1(x_t)))  # (batch, 32, 32, 108)
            if not mem1_initialized:
                mem1 = torch.zeros_like(cur1)
                mem1_initialized = True
            spk1, mem1 = self.lif1(cur1, mem1, step)

            # Conv block 2
            cur2 = self.pool2(self.bn2(self.conv2(spk1)))  # (batch, 64, 16, 54)
            if not mem2_initialized:
                mem2 = torch.zeros_like(cur2)
                mem2_initialized = True
            spk2, mem2 = self.lif2(cur2, mem2, step)

            # Pool + flatten
            pooled = self.avg_pool(spk2)  # (batch, 64, 4, 9)
            flat = pooled.view(pooled.size(0), -1)  # (batch, 2304)

            # FC block 1
            cur3 = self.fc1(flat)  # (batch, 256)
            spk3, mem3 = self.lif3(cur3, mem3, step)
            spk3 = self.dropout(spk3)

            # FC block 2 (output)
            cur4 = self.fc2(spk3)  # (batch, 50)
            spk4, mem4 = self.lif4(cur4, mem4, step)

            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)

        return torch.stack(spk_out_rec), torch.stack(mem_out_rec)


# ============================================================
# Training and evaluation
# ============================================================

def train_epoch(model, loader, optimizer, device):
    """Train one epoch with standard per-timestep CE loss on membrane potentials."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        spike_inputs = encode_direct(inputs).to(device)

        optimizer.zero_grad()
        spk_out, mem_out = model(spike_inputs)

        # Per-timestep CE loss on membrane potentials (standard approach)
        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss += criterion(mem_out[step], labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_model(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        spike_inputs = encode_direct(inputs).to(device)

        spk_out, mem_out = model(spike_inputs)

        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss += criterion(mem_out[step], labels)
        total_loss += loss.item()

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def train_fold(fold, device, epochs, patience, seed):
    """Train a single fold and return results."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"  Rhythm-SNN | Fold {fold}/5 | seed={seed}")
    print(f"  Device: {device} | Epochs: {epochs} | Patience: {patience}")
    print(f"{'='*60}")

    train_loader, test_loader = get_fold_dataloaders(fold, batch_size=BATCH_SIZE)
    model = RhythmSpikingCNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    osc_params = 0
    for name, p in model.named_parameters():
        if any(k in name for k in ["amplitude", "frequency", "phase", "beta_raw"]):
            osc_params += p.numel()
    print(f"  Total params: {total_params:,} (oscillation/beta: {osc_params:,})")

    out_dir = RESULTS_DIR / "experiments" / "rhythm_snn"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device)
        te_loss, te_acc = eval_model(model, test_loader, device)
        scheduler.step(te_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), out_dir / f"best_fold{fold}.pt")
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"  Ep {epoch:3d}/{epochs} | "
                f"tr_acc={tr_acc:.4f} te_acc={te_acc:.4f} best={best_acc:.4f} | "
                f"loss={te_loss:.2f} | {elapsed:.0f}s"
            )

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}, best={best_acc:.4f}")
            break

    elapsed = time.time() - start_time

    # Extract learned oscillation parameters
    osc_summary = {}
    for name, module in model.named_modules():
        if isinstance(module, RhythmLIF):
            osc_summary[name] = {
                "beta_mean": float(module.beta.mean().item()),
                "beta_std": float(module.beta.std().item()),
                "amplitude_mean": float(module.amplitude.mean().item()),
                "amplitude_std": float(module.amplitude.std().item()),
                "amplitude_abs_mean": float(module.amplitude.abs().mean().item()),
                "frequency_mean": float(module.frequency.mean().item()),
                "frequency_std": float(module.frequency.std().item()),
                "phase_mean": float(module.phase.mean().item()),
                "phase_std": float(module.phase.std().item()),
            }

    result = {
        "fold": fold,
        "seed": seed,
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "time_seconds": round(elapsed, 1),
        "total_params": total_params,
        "oscillation_params": osc_params,
        "oscillation_summary": osc_summary,
        "history": history,
    }

    with open(out_dir / f"result_fold{fold}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Fold {fold} done in {elapsed:.1f}s | Best: {best_acc:.4f} at epoch {best_epoch}")

    # Print oscillation analysis
    print(f"\n  Learned oscillation parameters:")
    for name, vals in osc_summary.items():
        print(
            f"    {name}: beta={vals['beta_mean']:.3f}+-{vals['beta_std']:.3f}, "
            f"A={vals['amplitude_abs_mean']:.4f}, "
            f"f={vals['frequency_mean']:.2f}+-{vals['frequency_std']:.2f}, "
            f"phi={vals['phase_mean']:.3f}"
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Rhythm-SNN: Oscillatory modulation for SNN on ESC-50"
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Specific fold (1-5). If omitted, runs all 5 folds.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"Max epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--patience", type=int, default=PATIENCE, help=f"Early stop patience (default: {PATIENCE})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Device: {device}")

    download_esc50()

    folds = [args.fold] if args.fold else list(range(1, NUM_FOLDS + 1))

    all_results = []
    for fold in folds:
        result = train_fold(fold, device, args.epochs, args.patience, args.seed)
        all_results.append(result)

    # Summary
    out_dir = RESULTS_DIR / "experiments" / "rhythm_snn"
    out_dir.mkdir(parents=True, exist_ok=True)

    accs = [r["best_accuracy"] for r in all_results]
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))

    print(f"\n{'='*60}")
    print(f"  Rhythm-SNN Summary")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  Fold {r['fold']}: {r['best_accuracy']*100:.2f}% (epoch {r['best_epoch']})")
    print(f"  Mean: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
    print(f"  Baseline (direct, standard LIF): 47.15% +/- 4.50%")
    diff = mean_acc * 100 - 47.15
    print(f"  Delta vs baseline: {diff:+.2f} pp")
    print(f"  Param overhead: {all_results[0]['oscillation_params']} oscillation params "
          f"({all_results[0]['oscillation_params']/all_results[0]['total_params']*100:.1f}% of total)")

    summary = {
        "experiment": "rhythm_snn",
        "seed": args.seed,
        "epochs": args.epochs,
        "patience": args.patience,
        "fold_accuracies": accs,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "baseline_mean": 0.4715,
        "baseline_std": 0.0450,
        "delta_vs_baseline_pp": round(diff, 2),
        "total_params": all_results[0]["total_params"],
        "oscillation_params": all_results[0]["oscillation_params"],
        "per_fold": [
            {
                "fold": r["fold"],
                "accuracy": r["best_accuracy"],
                "best_epoch": r["best_epoch"],
                "total_epochs": r["total_epochs"],
                "time_seconds": r["time_seconds"],
                "oscillation_summary": r["oscillation_summary"],
            }
            for r in all_results
        ],
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
