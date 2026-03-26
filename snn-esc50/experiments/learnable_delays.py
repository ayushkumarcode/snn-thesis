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

