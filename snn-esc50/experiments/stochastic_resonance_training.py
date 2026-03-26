"""
stochastic_resonance_training.py -- Trainable Stochastic Resonance SNN.

We already showed SR exists at sigma=0.02 (+0.25pp in inference). This script
makes sigma a LEARNABLE per-neuron parameter, optimized jointly with the network.

Custom SRLIF neuron:
    v[t] = beta * v[t-1] * (1 - spk[t-1]) + I[t] + sigma * randn()
    sigma is nn.Parameter per neuron, initialized at 0.02 (proven SR sweet spot)

Also includes SR+Rhythm variant: noise + oscillatory modulation combined.

Based on Entropy 2025: "Trainable Stochastic Resonance in Neural Networks".

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/stochastic_resonance_training.py --fold 1
    python experiments/stochastic_resonance_training.py --with-rhythm --fold 1
    python experiments/stochastic_resonance_training.py               # all 5 folds, SR only
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
# SRLIF Neuron: LIF with Trainable Stochastic Resonance
# ============================================================

class SRLIF(nn.Module):
    """Leaky Integrate-and-Fire neuron with learnable noise amplitude.

    Membrane dynamics:
        v[t] = beta * v[t-1] * (1 - spk[t-1]) + I[t] + sigma * N(0,1)

    sigma is an nn.Parameter per neuron (per channel for conv, per unit for FC).
    Initialized at init_sigma (default 0.02, our proven SR sweet spot).

    During training: noise is sampled fresh each forward pass.
    During eval: sigma is set to 0 (noise acts as regularizer during training).

    Args:
        neuron_shape: Shape of the neuron population.
            For conv layers: (channels,) -- broadcasts spatially.
            For FC layers: (hidden_size,).
        beta: Initial membrane decay rate (learnable if learn_beta=True).
        threshold: Spike threshold.
        init_sigma: Initial noise amplitude per neuron.
        spike_grad: Surrogate gradient function.
        learn_beta: Whether beta is learnable.
    """

    def __init__(
        self,
        neuron_shape: tuple,
        beta: float = BETA,
        threshold: float = 1.0,
        init_sigma: float = 0.02,
        spike_grad=None,
        learn_beta: bool = True,
    ):
        super().__init__()
        self.threshold = threshold

        if spike_grad is None:
            spike_grad = surrogate.spike_rate_escape(beta=1.0, slope=25)
        self.spike_grad = spike_grad

        # Learnable membrane decay
        if learn_beta:
            self.beta_raw = nn.Parameter(
                torch.full(neuron_shape, math.log(beta / (1.0 - beta)))
            )
        else:
            self.register_buffer(
                "beta_raw",
                torch.full(neuron_shape, math.log(beta / (1.0 - beta)))
            )

        # Learnable noise amplitude (stored as log for positivity constraint)
        # Initialize so that exp(log_sigma) = init_sigma
        self.log_sigma = nn.Parameter(
            torch.full(neuron_shape, math.log(max(init_sigma, 1e-8)))
        )

        self.neuron_shape = neuron_shape

    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw)

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def init_sr(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize membrane potential to zeros."""
        return torch.zeros(batch_size, *self.neuron_shape, device=device)

    def forward(
        self, input_current: torch.Tensor, mem: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single timestep forward.

        Args:
            input_current: (batch, *neuron_shape, ...) synaptic input.
            mem: Previous membrane potential, same shape as input_current.

        Returns:
            spk: Output spikes (binary).
            mem: Updated membrane potential.
        """
        beta = self.beta
        sigma = self.sigma

        # Broadcast beta and sigma to match input_current dimensions
        ndim_extra = input_current.dim() - 1 - len(self.neuron_shape)
        beta_bc = beta
        sigma_bc = sigma
        for _ in range(ndim_extra):
            beta_bc = beta_bc.unsqueeze(-1)
            sigma_bc = sigma_bc.unsqueeze(-1)

        # Leaky integration
        mem = beta_bc * mem + input_current

        # Add noise (training only)
        if self.training:
            noise = torch.randn_like(mem) * sigma_bc
            mem = mem + noise

        # Spike generation with surrogate gradient
        spk = self.spike_grad(mem - self.threshold)

        # Soft reset
        mem = mem * (1.0 - spk.detach())

        return spk, mem


# ============================================================
# SR+Rhythm LIF Neuron
# ============================================================

class SRRhythmLIF(nn.Module):
    """LIF neuron combining trainable stochastic resonance AND oscillatory modulation.

    Membrane dynamics:
        v[t] = beta * v[t-1] * (1-spk) + I[t] + sigma * N(0,1) + A * sin(2*pi*f*t/T + phi)

    Combines both noise-based SR and rhythm oscillation for maximum
    biologically-inspired temporal processing.
    """

    def __init__(
        self,
        neuron_shape: tuple,
        beta: float = BETA,
        threshold: float = 1.0,
        init_sigma: float = 0.02,
        num_steps: int = NUM_STEPS,
        spike_grad=None,
    ):
        super().__init__()
        self.threshold = threshold
        self.num_steps = num_steps

        if spike_grad is None:
            spike_grad = surrogate.spike_rate_escape(beta=1.0, slope=25)
        self.spike_grad = spike_grad

        # Learnable beta
        self.beta_raw = nn.Parameter(
            torch.full(neuron_shape, math.log(beta / (1.0 - beta)))
        )

        # SR: learnable noise amplitude
        self.log_sigma = nn.Parameter(
            torch.full(neuron_shape, math.log(max(init_sigma, 1e-8)))
        )

        # Rhythm: learnable oscillation parameters
        self.amplitude = nn.Parameter(torch.full(neuron_shape, 0.1))
        self.frequency = nn.Parameter(torch.empty(neuron_shape).uniform_(0.5, 5.0))
        self.phase = nn.Parameter(torch.zeros(neuron_shape))

        self.neuron_shape = neuron_shape

    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw)

    @property
    def sigma(self):
        return torch.exp(self.log_sigma)

    def _oscillation(self, t: int) -> torch.Tensor:
        angle = 2.0 * math.pi * self.frequency * t / self.num_steps + self.phase
        return self.amplitude * torch.sin(angle)

    def init_srrhythm(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, *self.neuron_shape, device=device)

    def forward(
        self, input_current: torch.Tensor, mem: torch.Tensor, t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta = self.beta
        sigma = self.sigma
        osc = self._oscillation(t)

        ndim_extra = input_current.dim() - 1 - len(self.neuron_shape)
        beta_bc = beta
        sigma_bc = sigma
        osc_bc = osc
        for _ in range(ndim_extra):
            beta_bc = beta_bc.unsqueeze(-1)
            sigma_bc = sigma_bc.unsqueeze(-1)
            osc_bc = osc_bc.unsqueeze(-1)

        # Leaky integration + oscillation
        mem = beta_bc * mem + input_current + osc_bc

        # Add noise (training only)
        if self.training:
            noise = torch.randn_like(mem) * sigma_bc
            mem = mem + noise

        # Spike
        spk = self.spike_grad(mem - self.threshold)
        mem = mem * (1.0 - spk.detach())

        return spk, mem


# ============================================================
# SR-SNN Model (noise only)
# ============================================================

class SRSNN(nn.Module):
    """SpikingCNN with trainable stochastic resonance per neuron.

    Same architecture as baseline SpikingCNN but with SRLIF neurons.
    Uses learn_beta=True, spike_rate_escape, Dropout(0.3).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        beta: float = BETA,
        num_steps: int = NUM_STEPS,
        init_sigma: float = 0.02,
    ):
        super().__init__()
        self.num_steps = num_steps

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = SRLIF(neuron_shape=(32,), beta=beta, init_sigma=init_sigma)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = SRLIF(neuron_shape=(64,), beta=beta, init_sigma=init_sigma)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = SRLIF(neuron_shape=(256,), beta=beta, init_sigma=init_sigma)
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = SRLIF(neuron_shape=(num_classes,), beta=beta, init_sigma=init_sigma)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[1]
        device = x.device

        mem1 = self.lif1.init_sr(batch_size, device)
        mem2 = self.lif2.init_sr(batch_size, device)
        mem3 = self.lif3.init_sr(batch_size, device)
        mem4 = self.lif4.init_sr(batch_size, device)

        mem1_init = False
        mem2_init = False

        spk_out_rec = []
        mem_out_rec = []

        for step in range(self.num_steps):
            x_t = x[step]

            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            if not mem1_init:
                mem1 = torch.zeros_like(cur1)
                mem1_init = True
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            if not mem2_init:
                mem2 = torch.zeros_like(cur2)
                mem2_init = True
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
# SR+Rhythm SNN Model (noise + oscillation)
# ============================================================

class SRRhythmSNN(nn.Module):
    """SpikingCNN with both trainable SR noise and oscillatory modulation."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        beta: float = BETA,
        num_steps: int = NUM_STEPS,
        init_sigma: float = 0.02,
    ):
        super().__init__()
        self.num_steps = num_steps

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = SRRhythmLIF(
            neuron_shape=(32,), beta=beta, init_sigma=init_sigma, num_steps=num_steps,
        )

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = SRRhythmLIF(
            neuron_shape=(64,), beta=beta, init_sigma=init_sigma, num_steps=num_steps,
        )

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = SRRhythmLIF(
            neuron_shape=(256,), beta=beta, init_sigma=init_sigma, num_steps=num_steps,
        )
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = SRRhythmLIF(
            neuron_shape=(num_classes,), beta=beta, init_sigma=init_sigma, num_steps=num_steps,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[1]
        device = x.device

        mem1 = self.lif1.init_srrhythm(batch_size, device)
        mem2 = self.lif2.init_srrhythm(batch_size, device)
        mem3 = self.lif3.init_srrhythm(batch_size, device)
        mem4 = self.lif4.init_srrhythm(batch_size, device)

        mem1_init = False
        mem2_init = False

        spk_out_rec = []
        mem_out_rec = []

        for step in range(self.num_steps):
            x_t = x[step]

            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            if not mem1_init:
                mem1 = torch.zeros_like(cur1)
                mem1_init = True
            spk1, mem1 = self.lif1(cur1, mem1, step)

            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            if not mem2_init:
                mem2 = torch.zeros_like(cur2)
                mem2_init = True
            spk2, mem2 = self.lif2(cur2, mem2, step)

            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            cur3 = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3, step)
            spk3 = self.dropout(spk3)

            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4, step)

            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)

        return torch.stack(spk_out_rec), torch.stack(mem_out_rec)


# ============================================================
# Training and Evaluation
# ============================================================

def train_epoch(model, loader, optimizer, device):
    """Train one epoch with per-timestep CE on membrane potentials."""
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
    """Evaluate model (noise disabled in eval mode)."""
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


def extract_sr_params(model) -> dict:
    """Extract learned sigma/beta values from all SRLIF/SRRhythmLIF layers."""
    summary = {}
    for name, module in model.named_modules():
        if isinstance(module, (SRLIF, SRRhythmLIF)):
            info = {
                "beta_mean": float(module.beta.mean().item()),
                "beta_std": float(module.beta.std().item()),
                "sigma_mean": float(module.sigma.mean().item()),
                "sigma_std": float(module.sigma.std().item()),
                "sigma_min": float(module.sigma.min().item()),
                "sigma_max": float(module.sigma.max().item()),
            }
            if isinstance(module, SRRhythmLIF):
                info["amplitude_abs_mean"] = float(module.amplitude.abs().mean().item())
                info["frequency_mean"] = float(module.frequency.mean().item())
                info["phase_mean"] = float(module.phase.mean().item())
            summary[name] = info
    return summary


def train_fold(fold, with_rhythm, device, epochs, patience, seed, init_sigma):
    """Train a single fold."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    variant = "SR+Rhythm" if with_rhythm else "SR"

    print(f"\n{'='*60}")
    print(f"  Trainable {variant} SNN | Fold {fold}/5 | seed={seed}")
    print(f"  init_sigma={init_sigma} | Device: {device}")
    print(f"{'='*60}")

    train_loader, test_loader = get_fold_dataloaders(fold, batch_size=BATCH_SIZE)

    if with_rhythm:
        model = SRRhythmSNN(init_sigma=init_sigma).to(device)
    else:
        model = SRSNN(init_sigma=init_sigma).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    sr_params = 0
    for pname, p in model.named_parameters():
        if "log_sigma" in pname or "beta_raw" in pname:
            sr_params += p.numel()
        if with_rhythm and any(k in pname for k in ["amplitude", "frequency", "phase"]):
            sr_params += p.numel()
    print(f"  Total params: {total_params:,} (SR/rhythm params: {sr_params:,})")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    out_dir = RESULTS_DIR / "experiments" / "stochastic_resonance_training"
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
            torch.save(model.state_dict(), out_dir / f"best_{variant.lower().replace('+','_')}_fold{fold}.pt")
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            # Show current sigma statistics
            sigmas = []
            for m in model.modules():
                if isinstance(m, (SRLIF, SRRhythmLIF)):
                    sigmas.append(float(m.sigma.mean().item()))
            mean_sigma = np.mean(sigmas)
            print(
                f"  Ep {epoch:3d}/{epochs} | "
                f"tr={tr_acc:.4f} te={te_acc:.4f} best={best_acc:.4f} | "
                f"sigma={mean_sigma:.5f} | {elapsed:.0f}s"
            )

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}, best={best_acc:.4f}")
            break

    elapsed = time.time() - start_time

    # Extract learned SR parameters
    sr_summary = extract_sr_params(model)

    result = {
        "variant": variant,
        "fold": fold,
        "seed": seed,
        "init_sigma": init_sigma,
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "time_seconds": round(elapsed, 1),
        "total_params": total_params,
        "sr_params": sr_params,
        "learned_sr_params": sr_summary,
        "history": history,
    }

    suffix = "rhythm" if with_rhythm else "sr"
    with open(out_dir / f"result_{suffix}_fold{fold}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Fold {fold} done in {elapsed:.1f}s | Best: {best_acc:.4f} at epoch {best_epoch}")
    print(f"  Learned SR parameters:")
    for layer_name, info in sr_summary.items():
        line = f"    {layer_name}: beta={info['beta_mean']:.3f}, sigma={info['sigma_mean']:.5f}"
        if "amplitude_abs_mean" in info:
            line += f", A={info['amplitude_abs_mean']:.4f}, f={info['frequency_mean']:.2f}"
        print(line)

    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Trainable Stochastic Resonance SNN for ESC-50"
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Specific fold (1-5). If omitted, runs all 5 folds.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"Max epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--patience", type=int, default=PATIENCE, help=f"Early stop patience (default: {PATIENCE})")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--init-sigma", type=float, default=0.02,
        help="Initial noise sigma per neuron (default: 0.02, our SR sweet spot)",
    )
    parser.add_argument(
        "--with-rhythm", action="store_true",
        help="Add oscillatory modulation (SR + Rhythm combined)",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Device: {device}")

    download_esc50()

    folds = [args.fold] if args.fold else list(range(1, NUM_FOLDS + 1))
    variant = "SR+Rhythm" if args.with_rhythm else "SR"

    out_dir = RESULTS_DIR / "experiments" / "stochastic_resonance_training"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for fold in folds:
        result = train_fold(
            fold, args.with_rhythm, device,
            args.epochs, args.patience, args.seed, args.init_sigma,
        )
        all_results.append(result)

    # Summary
    accs = [r["best_accuracy"] for r in all_results]
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))

    print(f"\n{'='*60}")
    print(f"  Trainable {variant} SNN Summary")
    print(f"{'='*60}")
    for r in all_results:
        print(f"  Fold {r['fold']}: {r['best_accuracy']*100:.2f}% (epoch {r['best_epoch']})")
    print(f"  Mean: {mean_acc*100:.2f}% +/- {std_acc*100:.2f}%")
    print(f"  Baseline (direct, standard LIF): 47.15% +/- 4.50%")
    diff = mean_acc * 100 - 47.15
    print(f"  Delta vs baseline: {diff:+.2f} pp")
    print(f"  Init sigma: {args.init_sigma}")

    # Sigma evolution analysis
    print(f"\n  Sigma evolution (init -> final):")
    for r in all_results:
        for layer, info in r["learned_sr_params"].items():
            change = info["sigma_mean"] - args.init_sigma
            direction = "increased" if change > 0 else "decreased"
            print(f"    Fold {r['fold']} {layer}: {info['sigma_mean']:.5f} "
                  f"({direction} by {abs(change):.5f})")

    summary = {
        "experiment": "stochastic_resonance_training",
        "variant": variant,
        "seed": args.seed,
        "init_sigma": args.init_sigma,
        "epochs": args.epochs,
        "patience": args.patience,
        "with_rhythm": args.with_rhythm,
        "fold_accuracies": accs,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "baseline_mean": 0.4715,
        "baseline_std": 0.0450,
        "delta_vs_baseline_pp": round(diff, 2),
        "total_params": all_results[0]["total_params"],
        "sr_params": all_results[0]["sr_params"],
        "per_fold": [
            {
                "fold": r["fold"],
                "accuracy": r["best_accuracy"],
