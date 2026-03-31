#!/usr/bin/env python3
"""Generate all 5 thesis figures as PDF."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUTDIR = '/Users/kumar/Documents/University/Year3/thesisproject/snn-esc50/paper/figures'

# Colorblind-friendly palette (Wong 2011)
BLUE = '#0072B2'
ORANGE = '#E69F00'
RED = '#D55E00'
GREEN = '#009E73'
PURPLE = '#CC79A7'
GREY = '#999999'


# ============================================================
# FIGURE 1: Pruning-Energy Pareto Curve
# ============================================================
def fig1_pruning_pareto():
    prune_levels = [0, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    acc =   [57.35, 60.45, 59.15, 60.15, 59.60, 59.85, 59.65, 59.40, 59.65, 58.05, 54.20]
    energy = [4706,  3059,  2854,  2602,  2372,  2122,  1857,  1624,  1353,  1072,   747]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Pareto-optimal front shading (50-85% region where acc stays high, energy drops)
    pareto_idx = [1, 3, 5, 8]  # 50%, 60%, 70%, 85% are Pareto-optimal
    pareto_e = [energy[i] for i in pareto_idx]
    pareto_a = [acc[i] for i in pareto_idx]
    ax.fill_between(
        sorted(pareto_e),
        [min(pareto_a) - 1] * len(pareto_e),
        [max(pareto_a) + 1] * len(pareto_e),
        alpha=0.08, color=GREEN, label='Pareto-optimal region'
    )

    # Main scatter + line
    ax.plot(energy, acc, 'o-', color=BLUE, markersize=7, linewidth=1.5, zorder=5)

    # ANN baselines
    ax.axhline(y=63.85, color=RED, linestyle='--', linewidth=1, alpha=0.7, label='ANN accuracy (63.85%)')
