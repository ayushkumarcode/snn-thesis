"""
tet_training.py -- Temporal Efficient Training (TET) loss for SNNs.

Implements the TET loss from Deng et al. (ICLR 2022):
    L_TET = (1/T) * sum_t CE(mem_t, y) + lambda * (1/T) * sum_t (CE(mem_t, y) - L_mean)^2

The first term is the standard per-timestep CE loss (same as baseline).
The second term penalises variance across timesteps, encouraging the SNN
to make consistent predictions at EVERY timestep, not just at the end.
This flattens the temporal loss landscape and improves gradient flow.

Model uses learn_beta=True, learn_threshold=True, spike_rate_escape surrogate.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/tet_training.py --fold 1 --device mps
    python experiments/tet_training.py                      # all 5 folds
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
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
# TET-SNN Model (learnable beta + threshold, spike_rate_escape)
# ============================================================

class TETSpikingCNN(nn.Module):
    """SpikingCNN with learnable beta/threshold and spike_rate_escape surrogate.

    Architecture matches the baseline SpikingCNN exactly:
    Conv2d(1,32) -> BN -> MaxPool(2) -> LIF -> Conv2d(32,64) -> BN -> MaxPool(2)
    -> LIF -> AvgPool(4,6) -> FC(2304,256) -> LIF -> FC(256,50) -> LIF

    Differences from baseline:
        - learn_beta=True: beta (membrane decay) is a learnable parameter
        - learn_threshold=True: threshold is a learnable parameter
        - spike_rate_escape surrogate gradient (best from ablation study)
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

        # Convolutional feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        # After two MaxPool2d(2) on input (64, 216): (16, 54)
        # AvgPool2d(4,6) on (16,54) -> (4, 9)
        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # Fully connected classifier: 64 * 4 * 9 = 2304
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = snn.Leaky(
            beta=beta, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over all timesteps.

        Args:
            x: Input of shape (num_steps, batch, 1, n_mels, time_frames).

        Returns:
            spk_out: Output spikes, shape (num_steps, batch, num_classes).
            mem_out: Output membrane potentials, shape (num_steps, batch, num_classes).
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

            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)

        return torch.stack(spk_out_rec), torch.stack(mem_out_rec)


# ============================================================
# TET Loss
# ============================================================

class TETLoss(nn.Module):
    """Temporal Efficient Training loss (Deng et al., ICLR 2022).

    L_TET = L_CE + lambda_tet * L_var

    where:
        L_CE  = (1/T) * sum_t CE(mem_t, y)          -- mean per-timestep CE
        L_var = (1/T) * sum_t (CE(mem_t, y) - L_CE)^2  -- temporal variance penalty

    The variance term encourages the network to produce correct predictions
    at every timestep, not just accumulate information over time.
    """

    def __init__(self, lambda_tet: float = 1.0):
        super().__init__()
        self.lambda_tet = lambda_tet
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self, mem_out: torch.Tensor, targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute TET loss.

        Args:
            mem_out: Membrane potentials, shape (T, batch, num_classes).
            targets: Class labels, shape (batch,).

        Returns:
            total_loss: L_CE + lambda * L_var (scalar).
            l_ce: Mean per-timestep CE loss (scalar).
            l_var: Temporal variance penalty (scalar).
        """
        T = mem_out.shape[0]

        # Per-timestep CE losses: shape (T, batch)
        per_step_losses = torch.stack([
            self.ce(mem_out[t], targets) for t in range(T)
        ])  # (T, batch)

        # Mean across timesteps for each sample, then mean across batch
        l_ce = per_step_losses.mean()  # scalar

        # Per-timestep mean loss across batch: shape (T,)
        per_step_mean = per_step_losses.mean(dim=1)  # (T,)

        # L_mean = (1/T) * sum_t mean_batch_CE_t
        l_mean = per_step_mean.mean()  # scalar

        # Variance term: (1/T) * sum_t (mean_batch_CE_t - L_mean)^2
        l_var = ((per_step_mean - l_mean) ** 2).mean()  # scalar

        total_loss = l_ce + self.lambda_tet * l_var

        return total_loss, l_ce, l_var


# ============================================================
# Training and evaluation
# ============================================================

def train_epoch(model, loader, optimizer, device, tet_loss_fn):
    """Train one epoch with TET loss."""
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_var = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        spike_inputs = encode_direct(inputs).to(device)

        optimizer.zero_grad()
        spk_out, mem_out = model(spike_inputs)

        loss, l_ce, l_var = tet_loss_fn(mem_out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += l_ce.item()
        total_var += l_var.item()

        # Predict using summed membrane potentials (same as baseline)
        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    n = len(loader)
    return total_loss / n, total_ce / n, total_var / n, correct / total


@torch.no_grad()
def eval_model(model, loader, device, tet_loss_fn):
    """Evaluate model with TET loss."""
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_var = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        spike_inputs = encode_direct(inputs).to(device)

        spk_out, mem_out = model(spike_inputs)
        loss, l_ce, l_var = tet_loss_fn(mem_out, labels)

        total_loss += loss.item()
        total_ce += l_ce.item()
        total_var += l_var.item()

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    n = len(loader)
    return total_loss / n, total_ce / n, total_var / n, correct / total


def train_fold(fold, device, epochs, patience, lambda_tet, seed):
    """Train a single fold with TET loss and return results."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"  TET Training | Fold {fold}/5 | lambda={lambda_tet} | seed={seed}")
    print(f"  Device: {device} | Epochs: {epochs} | Patience: {patience}")
    print(f"{'='*60}")

    train_loader, test_loader = get_fold_dataloaders(fold, batch_size=BATCH_SIZE)
    model = TETSpikingCNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )
    tet_loss_fn = TETLoss(lambda_tet=lambda_tet)

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    history = {
        "train_loss": [], "train_ce": [], "train_var": [], "train_acc": [],
        "test_loss": [], "test_ce": [], "test_var": [], "test_acc": [],
    }

    # Results directory
    out_dir = RESULTS_DIR / "experiments" / "tet_training"
    out_dir.mkdir(parents=True, exist_ok=True)

