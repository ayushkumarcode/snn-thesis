"""
ann_to_snn_conversion.py -- ANN-to-SNN conversion with threshold calibration.

Based on Bojkovic AISTATS 2024 and Rathi et al. ICLR 2020:
  1. Load trained ANN from results/ann/none/best_fold{fold}.pt
  2. Run all training data through ANN, record max activation per layer
  3. Set SNN threshold per layer = percentile of max activations
  4. Convert: replace ReLU with IF neurons (beta=1.0, no leak)
  5. Evaluate converted SNN at different timestep budgets: T=1,4,8,16,25,50,100
  6. Report accuracy-vs-T curve

Key insight: IF neurons (no leak, beta=1.0) are used because in the conversion
framework, ReLU activations map to spike rates, and leakage would distort
the mapping. snn.Leaky(beta=1.0) gives a perfect integrator.

Usage:
    cd snn-esc50
    source .venv/bin/activate
    python experiments/ann_to_snn_conversion.py --fold 1
    python experiments/ann_to_snn_conversion.py --percentile 99.5 --max-timesteps 200
    python experiments/ann_to_snn_conversion.py                    # all 5 folds
"""

import argparse
import json
import sys
import time
from pathlib import Path
