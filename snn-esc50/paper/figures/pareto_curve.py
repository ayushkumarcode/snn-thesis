#!/usr/bin/env python3
"""Generate pruning-energy Pareto curve for ICONS paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from neurobench_pruned_sweep.json (5-fold means)
prune_pcts = [0, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
spinnaker_acc = [57.35, 60.45, 59.15, 60.15, 59.60, 59.85, 59.65, 59.40, 59.65, 58.05, 54.20]
snntorch_acc = [59.50, 61.85, 60.35, 59.70, 60.40, 60.05, 60.60, 59.10, 60.25, 58.40, 55.85]
gaps = [2.15, 1.40, 1.20, -0.45, 0.80, 0.20, 0.95, -0.30, 0.60, 0.35, 1.65]
energy_nj = [4706, 3059, 2854, 2602, 2372, 2122, 1857, 1624, 1353, 1072, 747]

# ANN baseline energy
ann_energy = 470  # approximate from energy_vs_ann ratios

fig, ax1 = plt.subplots(1, 1, figsize=(3.5, 2.8))

# Plot accuracy vs energy
color_sp = '#2166AC'
color_snn = '#B2182B'
color_gap = '#4DAF4A'

ax1.plot(energy_nj, spinnaker_acc, 'o-', color=color_sp, markersize=5, linewidth=1.5,
         label='SpiNNaker', zorder=3)
ax1.plot(energy_nj, snntorch_acc, 's--', color=color_snn, markersize=4, linewidth=1.0,
         label='snnTorch', alpha=0.7, zorder=2)

# Annotate key points
ax1.annotate('85%\n0.6pp gap', xy=(1353, 59.65), xytext=(1600, 56.5),
             fontsize=6, ha='center', arrowprops=dict(arrowstyle='->', lw=0.8),
             bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.3))
ax1.annotate('50%\n+3.1pp', xy=(3059, 60.45), xytext=(3400, 62.5),
             fontsize=6, ha='center', arrowprops=dict(arrowstyle='->', lw=0.8),
             bbox=dict(boxstyle='round,pad=0.2', fc='lightgreen', alpha=0.3))
ax1.annotate('0%\n(unpruned)', xy=(4706, 57.35), xytext=(4400, 54.5),
             fontsize=6, ha='center', arrowprops=dict(arrowstyle='->', lw=0.8))

# ANN reference line
ax1.axhline(y=63.85, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax1.text(800, 64.2, 'ANN baseline', fontsize=6, color='gray')

# Highlight negative gap region
for i, (e, g) in enumerate(zip(energy_nj, gaps)):
    if g < 0:
        ax1.plot(e, spinnaker_acc[i], 'D', color='gold', markersize=8,
                 zorder=4, markeredgecolor='black', markeredgewidth=0.5)

ax1.set_xlabel('Energy per inference (nJ)', fontsize=8)
ax1.set_ylabel('Accuracy (%)', fontsize=8)
ax1.legend(fontsize=7, loc='lower left')
ax1.tick_params(labelsize=7)
ax1.set_xlim(500, 5200)
ax1.set_ylim(52, 65)
ax1.invert_xaxis()  # Lower energy = better, on the right
ax1.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('pareto_curve.pdf', bbox_inches='tight', dpi=300)
plt.savefig('pareto_curve.png', bbox_inches='tight', dpi=300)
print("Saved pareto_curve.pdf and .png")
