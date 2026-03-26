"""
spiking_leaf.py -- Spiking-LEAF: Learnable auditory frontend on raw waveforms.

Based on Song et al. ICASSP 2024: learnable Gabor filterbank + PCEN frontend
that processes raw audio waveforms instead of precomputed mel spectrograms.
The filterbank is jointly trained end-to-end with the SNN/ANN classifier.

Architecture:
  1. Raw waveform (batch, 1, 110250)
  2. Gabor filterbank: 64 1D convolutions, log-spaced 50-11025 Hz, kernel=401
  3. Per-Channel Energy Normalization (PCEN): learnable alpha, delta, r, s
  4. Output: (batch, 64, ~216) -- same shape as mel spectrograms
  5. SpikingCNN / ConvANN backbone on top

For SNN: encode_direct on (batch, 1, 64, ~216) then feed to SNN backbone.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/spiking_leaf.py --fold 1
    python experiments/spiking_leaf.py --model snn --fold 1 --device mps
    python experiments/spiking_leaf.py --model both       # all 5 folds, SNN+ANN
"""

import argparse
import json
import math
import sys
