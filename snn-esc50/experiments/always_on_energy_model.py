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
            "accuracy": 0.63,
            "description": "INT8 quantized ANN",
        },
    }

    # ============================================================
    # Hardware platforms
    # ============================================================
    platforms = {
        "SpiNNaker_1": {
            "idle_power_mw": 1000,  # ~1W per chip
            "active_overhead_mw": 0,  # Included in per-synop cost
            "per_synop_pj": 630,  # nJ per synop on SpiNNaker 1
            "note": "SpiNNaker 1 (48-core chip, ~1W total)",
        },
        "SpiNNaker_2": {
            "idle_power_mw": 50,    # Much lower with DVFS
            "active_overhead_mw": 0,
            "per_synop_pj": 10,     # ~10 pJ/synop
            "note": "SpiNNaker 2 (DVFS to 0.5V)",
        },
        "Xylo_Audio_2": {
            "idle_power_mw": 0.217, # 217 µW idle
            "active_power_mw": 0.298,  # 298 µW active
            "energy_per_inf_uj": 6.6,
            "note": "SynSense Xylo Audio 2",
        },
        "ARM_Cortex_M4": {
            "idle_power_mw": 0.1,   # ~100 µW sleep
            "active_power_mw": 50,  # ~50 mW active
            "note": "Typical MCU for edge audio",
        },
        "Jetson_Nano": {
            "idle_power_mw": 1000,
            "active_power_mw": 5000,
            "note": "NVIDIA Jetson Nano (5W mode)",
        },
    }

    # ============================================================
    # Deployment scenarios
    # ============================================================
    duty_cycles = [0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
    inference_rate_hz = 1.0  # 1 inference per second during events
    seconds_per_day = 86400

    results = {"scenarios": [], "per_inference": per_inference,
               "platforms": {k: v["note"] for k, v in platforms.items()}}

    for duty in duty_cycles:
        active_seconds = seconds_per_day * duty
        idle_seconds = seconds_per_day * (1 - duty)
        n_inferences = active_seconds * inference_rate_hz

        scenario = {
            "duty_cycle": duty,
            "active_hours": duty * 24,
            "n_inferences_per_day": n_inferences,
            "comparisons": {},
        }

        # SNN on neuromorphic hardware (event-driven)
        # During silence: ZERO computation (truly event-driven)
        for model_name, model_info in per_inference.items():
            if not model_name.startswith("SNN"):
                continue

            # SpiNNaker 2 (best neuromorphic option)
            snn_active_j = (model_info["energy_nj"] * 1e-9 *
                            n_inferences)
            snn_idle_j = 0  # True event-driven: zero during silence
            snn_total_j = snn_active_j + snn_idle_j

            scenario["comparisons"][f"{model_name}_SpiNNaker2"] = {
                "active_energy_j": snn_active_j,
                "idle_energy_j": snn_idle_j,
                "total_energy_j": snn_total_j,
                "total_energy_mj": snn_total_j * 1000,
            }

            # Xylo Audio 2
            xylo_active_j = 6.6e-6 * n_inferences  # 6.6 µJ/inf
            xylo_idle_j = 0.217e-3 * idle_seconds   # 217 µW idle
            xylo_total_j = xylo_active_j + xylo_idle_j
