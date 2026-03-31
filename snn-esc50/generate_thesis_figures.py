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

