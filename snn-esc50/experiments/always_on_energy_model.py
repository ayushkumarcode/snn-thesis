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
    per_inference = {
        "SNN_baseline": {
            "energy_nj": 968,
            "accuracy": 0.4715,
            "description": "Baseline SNN (T=25, 26.4% spike rate)",
        },
        "SNN_rhythm": {
            "energy_nj": 968,  # Similar architecture
            "accuracy": 0.6110,
            "description": "Rhythm SNN (T=25, our best accuracy)",
        },
        "SNN_optimized_10x": {
            "energy_nj": 97,   # Target: 10x reduction
            "accuracy": 0.55,  # Estimated
            "description": "Optimized SNN (spike reg + early exit)",
        },
        "SNN_optimized_20x": {
            "energy_nj": 48,   # Target: 20x reduction
            "accuracy": 0.50,  # Estimated
            "description": "Aggressively optimized SNN",
        },
        "ANN_baseline": {
            "energy_nj": 454,
            "accuracy": 0.6385,
            "description": "ANN baseline (single forward pass)",
        },
        "ANN_int8": {
            "energy_nj": 227,   # ~2x reduction from quantization
