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
