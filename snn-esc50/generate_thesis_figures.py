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
