"""
learnable_delays.py -- Learnable Synaptic Delays for temporal pattern matching.

Adds per-output-neuron learnable delays to FC layers. Each output neuron
reads from a past timestep in its input history, enabling the network to
learn temporal alignment patterns in audio spectrograms.

Implementation:
    - DelayedLinear wraps nn.Linear with a delay buffer
    - Each output neuron j has a learnable delay d_j in [0, max_delay]
    - During training: continuous delays, soft interpolation between timesteps
    - During inference: delays rounded to nearest integer
    - Applied to FC1 (2304->256) and FC2 (256->50) only

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/learnable_delays.py --fold 1 --device mps
    python experiments/learnable_delays.py  # all 5 folds
    python experiments/learnable_delays.py --max-delay 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NUM_CLASSES, BETA, NUM_STEPS, N_MELS,
    NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE, BATCH_SIZE,
    RESULTS_DIR, get_device,
)
from src.dataset import download_esc50, get_fold_dataloaders
from src.encoding import encode_direct


# ============================================================
# Delayed Linear Layer
# ============================================================

class DelayedLinear(nn.Module):
    """Linear layer with per-output-neuron learnable synaptic delays.

    Maintains a circular buffer of past inputs. Each output neuron j reads
    from buffer[delay_j] rather than the current input, allowing the network
    to learn temporal offsets for pattern matching.

    During training, delays are continuous and soft interpolation is used
    between adjacent integer timesteps for gradient flow. During eval,
    delays are rounded to the nearest integer.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        max_delay: Maximum allowed delay in timesteps.
        bias: Whether to include bias.
    """

    def __init__(self, in_features: int, out_features: int,
                 max_delay: int = 5, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_delay = max_delay

        # Standard linear weights
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Learnable delay per output neuron
        # Initialize to small random values near 0.5 so frac is nonzero from
        # the start, ensuring gradient flow through the interpolation weights.
        # Stored as raw parameter, clamped to [0, max_delay] during forward.
        self.delay_raw = nn.Parameter(
            torch.empty(out_features).uniform_(0.1, 0.9)
        )

        # Buffer for past inputs — not a persistent buffer, managed manually
        # Shape will be (max_delay + 1, batch, in_features) during forward
        self._buffer = None
        self._buffer_idx = 0  # circular index

    def init_buffer(self, batch_size: int, device: torch.device):
        """Reset the input history buffer for a new sequence."""
        self._buffer = torch.zeros(
            self.max_delay + 1, batch_size, self.in_features, device=device
        )
        self._buffer_idx = 0

    @property
    def delays(self):
        """Get clamped delays in [0, max_delay]."""
        return self.delay_raw.clamp(0.0, float(self.max_delay))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single timestep.

        Args:
            x: Input tensor of shape (batch, in_features).

        Returns:
            Output tensor of shape (batch, out_features).
        """
        batch_size = x.shape[0]

        # Initialize buffer on first call or if batch size changed
        if self._buffer is None or self._buffer.shape[1] != batch_size:
            self.init_buffer(batch_size, x.device)

        # Store current input in circular buffer (detached to avoid graph explosion
        # over many timesteps — gradient flows through the live tensor replacement below)
        write_idx = self._buffer_idx % (self.max_delay + 1)
        self._buffer[write_idx] = x.detach()

        delays = self.delays  # (out_features,)

        if self.training:
            # Soft interpolation for gradient flow through delay parameters.
            # For each output neuron j with delay d_j:
            #   delayed_input_j = (1 - frac) * buffer[floor_idx] + frac * buffer[ceil_idx]

            d_floor = delays.floor().long()  # (out_features,)
            d_ceil = (d_floor + 1).clamp(max=self.max_delay)
            frac = delays - delays.floor()  # fractional part, (out_features,)

            # Buffer read indices (going back in time from current write position)
            buf_size = self.max_delay + 1
            floor_read = (write_idx - d_floor) % buf_size  # (out_features,)
            ceil_read = (write_idx - d_ceil) % buf_size     # (out_features,)

            # Gather delayed inputs: (out_features, batch, in_features)
            floor_inputs = self._buffer[floor_read]
            ceil_inputs = self._buffer[ceil_read]

            # Substitute the live (gradient-attached) current input x wherever
            # a read index points to the current write position. This allows
            # gradient flow through x for neurons whose delay reads "now",
            # enabling the linear weights to learn even at delay=0.
            x_expanded = x.unsqueeze(0).expand(self.out_features, -1, -1)
            floor_is_current = (floor_read == write_idx).view(-1, 1, 1)
            ceil_is_current = (ceil_read == write_idx).view(-1, 1, 1)
            floor_inputs = torch.where(floor_is_current, x_expanded, floor_inputs)
            ceil_inputs = torch.where(ceil_is_current, x_expanded, ceil_inputs)

            # Interpolate
            frac_expanded = frac.view(-1, 1, 1)  # (out_features, 1, 1)
            delayed_inputs = (1.0 - frac_expanded) * floor_inputs + frac_expanded * ceil_inputs

            # Apply linear transform: W[j,i] * delayed_inputs[j, b, i], sum over i
            W = self.linear.weight  # (out_features, in_features)
            output = (W.unsqueeze(1) * delayed_inputs).sum(dim=2)  # (out_features, batch)
            output = output.t()  # (batch, out_features)

            if self.linear.bias is not None:
                output = output + self.linear.bias.unsqueeze(0)

        else:
            # Inference: round delays to integers
            d_int = delays.round().long()
            buf_size = self.max_delay + 1
            read_idx = (write_idx - d_int) % buf_size

            delayed_inputs = self._buffer[read_idx]

            W = self.linear.weight
            output = (W.unsqueeze(1) * delayed_inputs).sum(dim=2).t()

            if self.linear.bias is not None:
                output = output + self.linear.bias.unsqueeze(0)

        self._buffer_idx += 1
        return output


# ============================================================
# SNN Model with Learnable Delays
# ============================================================

class DelayedSpikingCNN(nn.Module):
    """Convolutional SNN with learnable synaptic delays on FC layers.

    Architecture mirrors SpikingCNN:
    Conv2d(1,32) -> BN -> MaxPool(2) -> LIF
    Conv2d(32,64) -> BN -> MaxPool(2) -> LIF
    AvgPool(4,6) -> DelayedFC(2304,256) -> LIF -> Dropout -> DelayedFC(256,50) -> LIF

    LIF neurons use learnable beta and threshold for maximum flexibility.

    Args:
        num_classes: Number of output classes.
        num_steps: Number of simulation timesteps.
        max_delay: Maximum synaptic delay in timesteps.
        spike_grad: Surrogate gradient function.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        num_steps: int = NUM_STEPS,
        max_delay: int = 5,
        spike_grad=None,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.max_delay = max_delay

        if spike_grad is None:
            spike_grad = surrogate.spike_rate_escape(beta=1.0, slope=25)

        # Conv block 1 (no delays — spatial, not temporal)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(
            beta=BETA, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        # Conv block 2 (no delays)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(
            beta=BETA, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        # Pooling
        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # FC block 1 with delays
        self.fc1 = DelayedLinear(64 * 4 * 9, 256, max_delay=max_delay)
        self.lif3 = snn.Leaky(
            beta=BETA, spike_grad=spike_grad,
            learn_beta=True, learn_threshold=True,
        )

        self.dropout = nn.Dropout(0.3)

        # FC block 2 with delays
        self.fc2 = DelayedLinear(256, num_classes, max_delay=max_delay)
        self.lif4 = snn.Leaky(
            beta=BETA, spike_grad=spike_grad,
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
        batch_size = x.shape[1]

        # Initialize LIF hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        # Initialize delay buffers
        self.fc1.init_buffer(batch_size, x.device)
        self.fc2.init_buffer(batch_size, x.device)

        spk_out_rec = []
        mem_out_rec = []

        for step in range(self.num_steps):
            x_t = x[step]  # (batch, 1, n_mels, time)

            # Conv block 1
            cur1 = self.pool1(self.bn1(self.conv1(x_t)))
            spk1, mem1 = self.lif1(cur1, mem1)

            # Conv block 2
            cur2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            # Pool + flatten
            pooled = self.avg_pool(spk2)
            flat = pooled.view(pooled.size(0), -1)

            # FC block 1 with delay
            cur3 = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3)

            # Dropout
            spk3_dropped = self.dropout(spk3)

            # FC block 2 with delay
            cur4 = self.fc2(spk3_dropped)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)

        return torch.stack(spk_out_rec), torch.stack(mem_out_rec)


# ============================================================
# Training and Evaluation
# ============================================================

def train_epoch(model, loader, optimizer, device, num_steps):
    """Train for one epoch using per-timestep CE on membrane potentials."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        spike_inputs = encode_direct(inputs, num_steps=num_steps)

        optimizer.zero_grad()
        spk_out, mem_out = model(spike_inputs)

        # Per-timestep CE loss on membrane potentials
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
def eval_model(model, loader, device, num_steps):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        spike_inputs = encode_direct(inputs, num_steps=num_steps)
        spk_out, mem_out = model(spike_inputs)

        loss = torch.zeros(1, device=device)
        for step in range(mem_out.shape[0]):
            loss += criterion(mem_out[step], labels)
        total_loss += loss.item()

        predicted = mem_out.sum(dim=0).argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def get_delay_stats(model):
    """Extract learned delay values for analysis."""
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, DelayedLinear):
            delays = module.delays.detach().cpu()
            stats[name] = {
                "delays_mean": float(delays.mean()),
                "delays_std": float(delays.std()),
                "delays_min": float(delays.min()),
                "delays_max": float(delays.max()),
                "delays_median": float(delays.median()),
                "delays_histogram": {
                    f"d={d}": int((delays.round().long() == d).sum())
                    for d in range(module.max_delay + 1)
                },
                "delays_all": delays.tolist(),
            }
    return stats


def get_lif_stats(model):
    """Extract learned LIF parameters (beta, threshold)."""
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, snn.Leaky):
            beta = module.beta.detach().cpu()
            threshold = module.threshold.detach().cpu()
            stats[name] = {
                "beta": float(beta) if beta.dim() == 0 else float(beta.mean()),
                "threshold": float(threshold) if threshold.dim() == 0 else float(threshold.mean()),
            }
    return stats


def train_fold(fold, args, device):
    """Train and evaluate a single fold."""
    print(f"\n{'='*60}")
    print(f"Fold {fold} | max_delay={args.max_delay} | Device={device}")
    print(f"{'='*60}")

    torch.manual_seed(42 + fold)

    train_loader, test_loader = get_fold_dataloaders(fold, batch_size=BATCH_SIZE)

    model = DelayedSpikingCNN(
        num_classes=NUM_CLASSES,
        num_steps=NUM_STEPS,
        max_delay=args.max_delay,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    delay_params = sum(
        m.delay_raw.numel() for m in model.modules() if isinstance(m, DelayedLinear)
    )
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Delay parameters: {delay_params:,} (FC1: 256 + FC2: 50 = 306)")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_acc = 0.0
    best_epoch = 0
    no_improve = 0
    epoch_log = []

    out_dir = RESULTS_DIR / "experiments" / "learnable_delays"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, device, NUM_STEPS)
        te_loss, te_acc = eval_model(model, test_loader, device, NUM_STEPS)
        scheduler.step(te_loss)

        epoch_log.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "test_loss": round(te_loss, 4),
            "test_acc": round(te_acc, 4),
        })

        if te_acc > best_acc:
            best_acc = te_acc
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), out_dir / f"best_fold{fold}.pt")
        else:
            no_improve += 1

        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1 or no_improve == 0:
            # Show current delay stats inline
            fc1_delays = model.fc1.delays.detach()
            fc2_delays = model.fc2.delays.detach()
            print(f"  Ep {epoch:3d}/{args.epochs} | "
                  f"tr={tr_acc:.3f} te={te_acc:.3f} best={best_acc:.3f}@{best_epoch} | "
                  f"fc1_d={fc1_delays.mean():.1f}+/-{fc1_delays.std():.1f} "
                  f"fc2_d={fc2_delays.mean():.1f}+/-{fc2_delays.std():.1f} "
                  f"({elapsed:.0f}s)")

        if no_improve >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}, best={best_acc:.4f}@{best_epoch}")
            break

    # Load best model for analysis
    model.load_state_dict(
        torch.load(out_dir / f"best_fold{fold}.pt", map_location=device, weights_only=True)
    )
    delay_stats = get_delay_stats(model)
    lif_stats = get_lif_stats(model)

    return {
        "fold": fold,
        "best_accuracy": best_acc,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "total_params": total_params,
        "delay_params": delay_params,
        "max_delay": args.max_delay,
        "delay_stats": delay_stats,
