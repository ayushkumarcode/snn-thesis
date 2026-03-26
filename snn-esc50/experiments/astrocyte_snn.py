"""
astrocyte_snn.py -- Astrocyte-Augmented SNN experiment.

Slow glial modulation of neural excitability for homeostatic regulation.
Based on "Astrocyte-gated multi-timescale plasticity" (PMC, 2025).

Each LIF layer is wrapped with an "astrocyte" module that:
  1. Monitors average spike rate via exponential moving average
     (tau_astro ~ 100 timesteps, much slower than neural dynamics)
  2. Modulates the effective threshold based on activity level:
     - High activity -> raise threshold (inhibit runaway firing)
     - Low activity  -> lower threshold (rescue dead neurons)
  3. Creates homeostatic regulation that prevents both over-firing
     and under-firing -- particularly useful for small datasets
     where training signal is sparse.

Implementation:
  - AstrocyteLIF wraps snn.Leaky and adds threshold modulation
  - astro_state: EMA of population spike rate (per-channel/neuron)
  - Effective threshold = base_threshold * (1 + gain * (astro_state - target_rate))
  - tau_astro and gain are learnable; target_rate is learnable (init 0.1)

Also uses: learn_beta=True, Dropout(0.3), spike_rate_escape surrogate.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/astrocyte_snn.py --fold 1
    python experiments/astrocyte_snn.py              # all 5 folds
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_CLASSES, NUM_STEPS, BETA,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct


# ============================================================
# Astrocyte LIF Module
# ============================================================

class AstrocyteLIF(nn.Module):
    """LIF neuron with astrocyte-mediated threshold modulation.

    The astrocyte maintains a slow exponential moving average of the
    population spike rate and modulates the firing threshold to maintain
    homeostasis. When neurons fire too much, the threshold increases
    (suppression). When too few fire, the threshold decreases (excitation).

    Args:
        size: Number of features (channels for conv, neurons for FC).
        beta: Membrane decay rate for the underlying LIF neuron.
        spike_grad: Surrogate gradient function.
        learn_beta: Whether beta is learnable.
        tau_astro_init: Initial astrocyte time constant (0.99 = very slow).
        gain_init: Initial modulation gain (0.1 = gentle modulation).
        target_rate_init: Desired average firing rate (0.1 = 10%).
    """

    def __init__(self, size, beta=BETA, spike_grad=None, learn_beta=True,
                 tau_astro_init=0.99, gain_init=0.1, target_rate_init=0.1):
        super().__init__()
        self.size = size

        # Underlying LIF neuron
        self.lif = snn.Leaky(beta=beta, spike_grad=spike_grad,
                             learn_beta=learn_beta, threshold=1.0)

        # Learnable astrocyte parameters
        # Use raw parameters and apply sigmoid/softplus for constraints
        self._tau_astro_logit = nn.Parameter(
            torch.tensor(self._inv_sigmoid(tau_astro_init)))
        self._gain_raw = nn.Parameter(torch.tensor(gain_init))
        self._target_rate_logit = nn.Parameter(
            torch.tensor(self._inv_sigmoid(target_rate_init)))

    @staticmethod
    def _inv_sigmoid(x):
        """Inverse sigmoid for parameter initialisation."""
        x = max(min(x, 0.999), 0.001)
        return torch.log(torch.tensor(x / (1 - x))).item()

    @property
    def tau_astro(self):
        """Astrocyte time constant in (0, 1), constrained by sigmoid."""
        return torch.sigmoid(self._tau_astro_logit)

    @property
    def gain(self):
        """Modulation gain, unconstrained (can be positive or negative)."""
        return self._gain_raw

    @property
    def target_rate(self):
        """Target firing rate in (0, 1), constrained by sigmoid."""
        return torch.sigmoid(self._target_rate_logit)

    def init_astrocyte(self, device):
        """Initialise astrocyte state (EMA of spike rate)."""
        return torch.zeros(self.size, device=device)

    def forward(self, cur, mem, astro_state):
        """Forward pass with astrocyte modulation.

        Args:
            cur: Input current, shape (batch, size, ...).
            mem: Previous membrane potential (from lif.init_leaky()).
            astro_state: Previous astrocyte state, shape (size,).

        Returns:
            spk: Output spikes.
            mem: Updated membrane potential.
            astro_state: Updated astrocyte state.
            effective_thresh: The modulated threshold used this step.
        """
        device = cur.device

        # Compute spikes with current (base) threshold
        spk, mem = self.lif(cur, mem)

        # Compute population spike rate for this timestep
        # Average over batch and spatial dims, keep channel/neuron dim
        if spk.dim() == 4:
            # Conv layer: (batch, channels, H, W) -> mean over batch, H, W
            spike_rate = spk.mean(dim=(0, 2, 3))  # (channels,)
        elif spk.dim() == 2:
            # FC layer: (batch, neurons) -> mean over batch
            spike_rate = spk.mean(dim=0)  # (neurons,)
        else:
            spike_rate = spk.mean(dim=0).view(-1)

        # Update astrocyte EMA
        tau = self.tau_astro.to(device)
        astro_state = tau * astro_state + (1 - tau) * spike_rate.detach()

        # Compute threshold modulation for NEXT timestep
        # (modifying this step's output retroactively would break causality)
        target = self.target_rate.to(device)
        gain = self.gain.to(device)
        # threshold_mod > 1 when activity > target (suppress)
        # threshold_mod < 1 when activity < target (excite)
        threshold_mod = 1.0 + gain * (astro_state - target)
        # Clamp to prevent negative or extreme thresholds
        threshold_mod = threshold_mod.clamp(0.5, 2.0)

        # Apply modulation to membrane potential: effectively scale the
        # distance-to-threshold. We do this by dividing mem by threshold_mod,
        # which is equivalent to multiplying the threshold.
        # Reshape threshold_mod for broadcasting
        if mem.dim() == 4:
            mod = threshold_mod.view(1, -1, 1, 1)
        elif mem.dim() == 2:
            mod = threshold_mod.view(1, -1)
        else:
            mod = threshold_mod.view(1, -1)

        # Scale membrane potential (lower mod = easier to fire next step)
        mem = mem / mod

        effective_thresh = threshold_mod.mean().item()

        return spk, mem, astro_state, effective_thresh


# ============================================================
# Astrocyte SNN Model
# ============================================================

class AstrocyteSNN(nn.Module):
    """SpikingCNN with astrocyte-mediated homeostatic threshold regulation."""

    def __init__(self, num_classes=NUM_CLASSES, beta=BETA, num_steps=NUM_STEPS):
        super().__init__()
        self.num_steps = num_steps

        spike_grad = surrogate.spike_rate_escape(beta=1, slope=25)

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.astro_lif1 = AstrocyteLIF(32, beta=beta, spike_grad=spike_grad)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.astro_lif2 = AstrocyteLIF(64, beta=beta, spike_grad=spike_grad)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # FC block 1
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.astro_lif3 = AstrocyteLIF(256, beta=beta, spike_grad=spike_grad)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # FC block 2 (output)
        self.fc2 = nn.Linear(256, num_classes)
        self.astro_lif4 = AstrocyteLIF(num_classes, beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        """Forward pass.

        Returns:
            spk_out: (num_steps, batch, num_classes)
            mem_out: (num_steps, batch, num_classes)
            layer_stats: dict with per-layer spike rates and thresholds
        """
        device = x.device

        # Init LIF states
        mem1 = self.astro_lif1.lif.init_leaky()
        mem2 = self.astro_lif2.lif.init_leaky()
        mem3 = self.astro_lif3.lif.init_leaky()
        mem4 = self.astro_lif4.lif.init_leaky()

        # Init astrocyte states
        astro1 = self.astro_lif1.init_astrocyte(device)
        astro2 = self.astro_lif2.init_astrocyte(device)
        astro3 = self.astro_lif3.init_astrocyte(device)
        astro4 = self.astro_lif4.init_astrocyte(device)

        spk_out_rec = []
        mem_out_rec = []

        # Tracking stats per layer per timestep
        spike_rates = {"lif1": [], "lif2": [], "lif3": [], "lif4": []}
        thresholds = {"lif1": [], "lif2": [], "lif3": [], "lif4": []}

        for step in range(self.num_steps):
            x_t = x[step]

            # Conv block 1
            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, mem1, astro1, thresh1 = self.astro_lif1(cur1, mem1, astro1)

            # Conv block 2
            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2, astro2, thresh2 = self.astro_lif2(cur2, mem2, astro2)

            # Pool + flatten
            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            # FC block 1
            cur3 = self.fc1(flat)
            spk3, mem3, astro3, thresh3 = self.astro_lif3(cur3, mem3, astro3)

            # Dropout
            spk3 = self.dropout(spk3)

            # FC block 2
            cur4 = self.fc2(spk3)
            spk4, mem4, astro4, thresh4 = self.astro_lif4(cur4, mem4, astro4)

            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)

            # Record stats
            spike_rates["lif1"].append(spk1.mean().item())
            spike_rates["lif2"].append(spk2.mean().item())
            spike_rates["lif3"].append(spk3.mean().item())
            spike_rates["lif4"].append(spk4.mean().item())
            thresholds["lif1"].append(thresh1)
            thresholds["lif2"].append(thresh2)
            thresholds["lif3"].append(thresh3)
            thresholds["lif4"].append(thresh4)

        layer_stats = {
            "spike_rates": spike_rates,
            "thresholds": thresholds,
        }

        return torch.stack(spk_out_rec), torch.stack(mem_out_rec), layer_stats


# ============================================================
# Training / Evaluation
# ============================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_spike_rates = {"lif1": 0.0, "lif2": 0.0, "lif3": 0.0, "lif4": 0.0}

    ce_criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spike_input = encode_direct(data).to(device)

        optimizer.zero_grad()
        spk_out, mem_out, layer_stats = model(spike_input)

        # Per-timestep CE loss
        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss = loss + ce_criterion(mem_out[step], targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        # Accumulate spike rates
        for layer in all_spike_rates:
            all_spike_rates[layer] += sum(layer_stats["spike_rates"][layer]) / len(layer_stats["spike_rates"][layer])

    n = len(loader)
    stats = {
        "loss": total_loss / n,
        "accuracy": correct / total,
        "spike_rates": {k: v / n for k, v in all_spike_rates.items()},
    }
    return stats


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_spike_rates = {"lif1": 0.0, "lif2": 0.0, "lif3": 0.0, "lif4": 0.0}
    all_thresholds = {"lif1": 0.0, "lif2": 0.0, "lif3": 0.0, "lif4": 0.0}

    ce_criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spike_input = encode_direct(data).to(device)

        spk_out, mem_out, layer_stats = model(spike_input)

        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss = loss + ce_criterion(mem_out[step], targets)

        total_loss += loss.item()

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        for layer in all_spike_rates:
            sr = layer_stats["spike_rates"][layer]
            th = layer_stats["thresholds"][layer]
            all_spike_rates[layer] += sum(sr) / len(sr)
            all_thresholds[layer] += sum(th) / len(th)

    n = len(loader)
    stats = {
        "loss": total_loss / n,
        "accuracy": correct / total,
        "spike_rates": {k: v / n for k, v in all_spike_rates.items()},
        "thresholds": {k: v / n for k, v in all_thresholds.items()},
    }
    return stats


def get_learned_astrocyte_params(model):
    """Extract learned astrocyte parameters from model."""
    params = {}
    for name in ["astro_lif1", "astro_lif2", "astro_lif3", "astro_lif4"]:
        module = getattr(model, name)
        params[name] = {
            "tau_astro": module.tau_astro.item(),
            "gain": module.gain.item(),
            "target_rate": module.target_rate.item(),
            "beta": module.lif.beta.item() if hasattr(module.lif.beta, 'item') else module.lif.beta,
        }
    return params


def train_fold(fold, device, epochs, patience):
    """Train astrocyte SNN on one fold."""
    torch.manual_seed(42 + fold)

    train_loader, test_loader = get_fold_dataloaders(fold, batch_size=BATCH_SIZE)
    model = AstrocyteSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
               "spike_rates_per_epoch": [], "thresholds_per_epoch": []}

    best_model_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, optimizer, device)
        te = eval_model(model, test_loader, device)
        scheduler.step(te["loss"])

        history["train_loss"].append(tr["loss"])
        history["train_acc"].append(tr["accuracy"])
        history["test_loss"].append(te["loss"])
        history["test_acc"].append(te["accuracy"])
        history["spike_rates_per_epoch"].append(te["spike_rates"])
        history["thresholds_per_epoch"].append(te["thresholds"])

        if te["accuracy"] > best_acc:
            best_acc = te["accuracy"]
            best_epoch = epoch
            no_improve = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            sr = te["spike_rates"]
            th = te["thresholds"]
            print(f"  Ep {epoch:3d}/{epochs} | "
                  f"tr={tr['accuracy']:.3f} te={te['accuracy']:.3f} "
                  f"best={best_acc:.3f} | "
                  f"SR=[{sr['lif1']:.3f},{sr['lif2']:.3f},{sr['lif3']:.3f},{sr['lif4']:.3f}] | "
                  f"TH=[{th['lif1']:.3f},{th['lif2']:.3f},{th['lif3']:.3f},{th['lif4']:.3f}] | "
