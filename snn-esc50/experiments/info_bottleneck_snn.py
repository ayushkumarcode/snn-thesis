"""
info_bottleneck_snn.py -- Information Bottleneck SNN experiment.

Compress spike representations via variational information bottleneck.
Based on "Learning to Time-Decode via Information Bottleneck" (NeurIPS, 2024).

Adds a variational information bottleneck (VIB) after the hidden layer (lif3):
  - mu = Linear(256, 256)
  - logvar = Linear(256, 256)
  - z = mu + exp(0.5 * logvar) * N(0,1)  (reparameterisation trick)
  - KL_loss = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
  - Total loss = CE_loss + beta_ib * KL_loss

During training, z replaces spk3 as input to FC2. The KL term encourages
the 256-dim hidden spike representation to be maximally compressed while
retaining task-relevant information. This acts as a powerful regulariser
on the small 1600-sample ESC-50 dataset.

During evaluation: use mu directly (no sampling, no stochasticity).

Also uses: learn_beta=True, Dropout(0.3), spike_rate_escape surrogate.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/info_bottleneck_snn.py --fold 1
    python experiments/info_bottleneck_snn.py --beta-ib 1e-3    # all 5 folds
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
# Information Bottleneck SNN Model
# ============================================================

class InfoBottleneckSNN(nn.Module):
    """SpikingCNN with variational information bottleneck on hidden layer.

    After lif3 produces 256-dim hidden spikes, a VIB layer compresses
    the representation into a stochastic latent z. The KL divergence
    regulariser encourages z to be as close to N(0,1) as possible
    while still allowing accurate classification -- this forces the
    network to learn maximally compressed features.
    """

    def __init__(self, num_classes=NUM_CLASSES, beta=BETA, num_steps=NUM_STEPS):
        super().__init__()
        self.num_steps = num_steps

        spike_grad = surrogate.spike_rate_escape(beta=1, slope=25)

        # Conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # FC block 1
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

        # Information bottleneck: project hidden spikes to mu and logvar
        self.ib_mu = nn.Linear(256, 256)
        self.ib_logvar = nn.Linear(256, 256)

        # Dropout on bottleneck output
        self.dropout = nn.Dropout(0.3)

        # FC block 2 (output) -- receives bottleneck z
        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True)

    def reparameterise(self, mu, logvar):
        """Reparameterisation trick: z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, sample=True):
        """Forward pass with information bottleneck.

        Args:
            x: Spike input (num_steps, batch, 1, n_mels, time).
            sample: If True, use reparameterisation (training).
                    If False, use mu only (evaluation).

        Returns:
            spk_out: (num_steps, batch, num_classes)
            mem_out: (num_steps, batch, num_classes)
            kl_loss: Scalar KL divergence (averaged over timesteps)
            bottleneck_stats: dict with mu/logvar norms for monitoring
        """
        device = x.device

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk_out_rec = []
        mem_out_rec = []

        total_kl = torch.zeros(1, device=device)
        total_mu_norm = 0.0
        total_logvar_mean = 0.0

        for step in range(self.num_steps):
            x_t = x[step]

            # Conv block 1
            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, mem1 = self.lif1(cur1, mem1)

            # Conv block 2
            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            # Pool + flatten
            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            # FC block 1 -- produce hidden spikes
            cur3 = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3)

            # Information bottleneck
            mu = self.ib_mu(spk3)
            logvar = self.ib_logvar(spk3)

            if sample:
                z = self.reparameterise(mu, logvar)
            else:
                z = mu  # deterministic at eval time

            # KL divergence: KL(q(z|x) || N(0,1))
            # = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            total_kl = total_kl + kl.mean()

            # Track stats
            total_mu_norm += mu.norm(dim=1).mean().item()
            total_logvar_mean += logvar.mean().item()

            # Apply dropout to bottleneck output
            z = self.dropout(z)

            # FC block 2 (output)
            cur4 = self.fc2(z)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)

        # Average KL over timesteps
        kl_loss = total_kl / self.num_steps

        bottleneck_stats = {
            "avg_mu_norm": total_mu_norm / self.num_steps,
            "avg_logvar_mean": total_logvar_mean / self.num_steps,
        }

        return (torch.stack(spk_out_rec), torch.stack(mem_out_rec),
                kl_loss, bottleneck_stats)


# ============================================================
# Training / Evaluation
# ============================================================

def train_epoch(model, loader, optimizer, device, beta_ib):
    """Train for one epoch with information bottleneck loss."""
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    correct = 0
    total = 0
    total_mu_norm = 0.0
    total_logvar_mean = 0.0

    ce_criterion = nn.CrossEntropyLoss()

    for data, targets in loader:
        data, targets = data.to(device), targets.to(device)
        spike_input = encode_direct(data).to(device)

        optimizer.zero_grad()
        spk_out, mem_out, kl_loss, bn_stats = model(spike_input, sample=True)

        # Per-timestep CE loss
        ce_loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            ce_loss = ce_loss + ce_criterion(mem_out[step], targets)

        # Combined loss: CE + beta_ib * KL
        loss = ce_loss + beta_ib * kl_loss
