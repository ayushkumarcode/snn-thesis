"""
always_on_energy_model.py -- System-level energy analysis for 24h monitoring.

Computes energy budgets for continuous environmental sound monitoring:
- ANN on microcontroller (always computing)
- SNN on neuromorphic hardware (event-driven, zero during silence)
- Different duty cycles (2%, 5%, 10%, 20%, 50%)

This is a CALCULATION script -- no training, just produces tables.

Usage:
    python -m experiments.always_on_energy_model
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import RESULTS_DIR


def compute_energy_model():
    """Compute 24-hour energy budgets for different scenarios."""

    # ============================================================
    # Per-inference energy (from NeuroBench measurements)
    # ============================================================
