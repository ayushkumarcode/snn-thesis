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
    ax.axvline(x=454, color=RED, linestyle=':', linewidth=1, alpha=0.7, label='ANN energy (454 nJ)')

    # Label key points
    ax.annotate('0% (unpruned)', (energy[0], acc[0]),
                textcoords='offset points', xytext=(10, -12), fontsize=9, color=GREY)
    ax.annotate('85% (sweet spot)', (energy[8], acc[8]),
                textcoords='offset points', xytext=(-30, 10), fontsize=9, fontweight='bold', color=GREEN)
    ax.annotate('95%', (energy[10], acc[10]),
                textcoords='offset points', xytext=(-25, -12), fontsize=9, color=GREY)

    ax.set_xlabel('Energy per Inference (nJ)')
    ax.set_ylabel('SpiNNaker Accuracy (%)')
    ax.legend(fontsize=9, loc='lower right')
    ax.set_xlim(400, 5100)
    ax.set_ylim(52, 66)
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/pruning_pareto.pdf', bbox_inches='tight')
    plt.close(fig)
