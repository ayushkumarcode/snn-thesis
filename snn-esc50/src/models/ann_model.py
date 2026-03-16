"""
Convolutional ANN baseline for ESC-50 classification.

Same architecture as the SNN (SpikingCNN) but with ReLU activations
instead of LIF spiking neurons, and no temporal dimension.
This serves as the non-spiking baseline for comparison.
"""

import torch
import torch.nn as nn

from src.config import NUM_CLASSES, N_MELS


class ConvANN(nn.Module):
    """Convolutional ANN baseline, mirroring SpikingCNN architecture.

    Args:
        num_classes: Number of output classes.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # AvgPool2d(4,6) on (16,54) → (4, 9), MPS-compatible
            nn.AvgPool2d(kernel_size=(4, 6)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 9, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_frames).

        Returns:
            Logits of shape (batch, num_classes).
        """
        features = self.features(x)
        flat = features.view(features.size(0), -1)
        return self.classifier(flat)
