"""
Convolutional Spiking Neural Network for ESC-50 classification.

Architecture:
  Conv2d → LIF → Conv2d → LIF → Flatten → Linear → LIF → Linear → LIF

Uses surrogate gradient descent for training (handled by snnTorch).
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from src.config import NUM_CLASSES, BETA, NUM_STEPS, N_MELS


class SpikingCNN(nn.Module):
    """Convolutional SNN for environmental sound classification.

    Args:
        num_classes: Number of output classes.
        beta: Membrane potential decay rate.
        num_steps: Number of simulation timesteps.
        n_mels: Number of mel frequency bins (height of spectrogram).
        spike_grad: Surrogate gradient function from snntorch.surrogate.
                    Defaults to fast_sigmoid(slope=25).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        beta: float = BETA,
        num_steps: int = NUM_STEPS,
        n_mels: int = N_MELS,
        spike_grad=None,
    ):
        super().__init__()
        self.num_steps = num_steps

        if spike_grad is None:
            spike_grad = surrogate.fast_sigmoid(slope=25)

        # -- Convolutional feature extraction --
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # After two MaxPool2d(2) on input (64, 216): (16, 54)
        # Use standard AvgPool2d instead of Adaptive (MPS compatibility)
        # AvgPool2d(4,6) on (16,54) → (4, 9)
        self.avg_pool = nn.AvgPool2d(kernel_size=(4, 6))

        # -- Fully connected classifier --
        # 64 channels * 4 * 9 = 2304
        self.fc1 = nn.Linear(64 * 4 * 9, 256)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over all timesteps.

        Args:
            x: Spike input of shape (num_steps, batch, 1, n_mels, time_frames).

        Returns:
            spk_out: Output spikes, shape (num_steps, batch, num_classes).
            mem_out: Output membrane potentials, shape (num_steps, batch, num_classes).
        """
        # Initialise hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

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

            # FC block 1
            cur3 = self.fc1(flat)
            spk3, mem3 = self.lif3(cur3, mem3)

            # FC block 2 (output)
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk_out_rec.append(spk4)
            mem_out_rec.append(mem4)

        return torch.stack(spk_out_rec), torch.stack(mem_out_rec)
