"""
spinnaker_energy_deploy.py -- Deploy energy-optimized SNN on SpiNNaker.

Extracts hidden features from trained model, then deploys FC2 on SpiNNaker.

Two phases:
  Phase 1 (--extract-only): Run in .venv to extract features
  Phase 2 (default): Run in .venv-spinnaker for SpiNNaker inference

Usage:
    source .venv/bin/activate
    python experiments/spinnaker_energy_deploy.py --extract-only --fold 4

    source .venv-spinnaker/bin/activate
    python experiments/spinnaker_energy_deploy.py --fold 4 --num-samples 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import RESULTS_DIR, NUM_STEPS, NUM_CLASSES

