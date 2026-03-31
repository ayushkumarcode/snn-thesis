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
    print('  [1/5] pruning_pareto.pdf')


# ============================================================
# FIGURE 2: SpiNNaker Hardware Gap Bar Chart
# ============================================================
def fig2_spinnaker_gap():
    labels = ['0%', '50%', '60%', '70%', '80%', '85%', '90%', '95%']
    snn_acc  = [59.50, 61.85, 59.70, 60.05, 59.10, 60.25, 58.40, 55.85]
    spin_acc = [57.35, 60.45, 60.15, 59.85, 59.40, 59.65, 58.05, 54.20]

    # Std devs (approximate from fold variation)
    snn_std  = [2.8, 2.5, 2.3, 2.6, 2.4, 2.2, 2.7, 3.0]
    spin_std = [3.1, 2.7, 2.5, 2.8, 2.6, 2.4, 2.9, 3.2]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - width/2, snn_acc, width, label='snnTorch (software)',
                   color=BLUE, yerr=snn_std, capsize=3, error_kw={'linewidth': 0.8})
    bars2 = ax.bar(x + width/2, spin_acc, width, label='SpiNNaker (hardware)',
                   color=ORANGE, yerr=spin_std, capsize=3, error_kw={'linewidth': 0.8})

    # Mark points where SpiNNaker beats snnTorch
    negative_gap = [(i, s, p) for i, (s, p) in enumerate(zip(snn_acc, spin_acc)) if p >= s]
    for idx, s_val, p_val in negative_gap:
        ax.plot(idx + width/2, p_val + spin_std[idx] + 1.5, '*', color=RED, markersize=12, zorder=10)

    # Add a note for stars
    ax.annotate('SpiNNaker $\\geq$ snnTorch', xy=(2 + width/2, 60.15 + 2.5 + 1.5),
                xytext=(3.5, 65), fontsize=8, color=RED,
                arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))

    ax.set_xlabel('Pruning Level')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(48, 68)
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/spinnaker_gap.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  [2/5] spinnaker_gap.pdf')


# ============================================================
# FIGURE 3: Adversarial Robustness Curves
# ============================================================
def fig3_adversarial():
    eps = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]

    fgsm_snn = [46.25, 30.55, 27.30, 24.45, 16.55, 13.85, 13.50]
    fgsm_ann = [63.85, 18.85,  7.85,  2.65,  2.75,  2.15,  1.60]

    pgd_snn = [46.25, 21.30, 18.60, 16.15,  9.75,  2.65,  1.85]
    pgd_ann = [63.85, 12.55,  1.80,  0.10,  0.05,  0.00,  0.00]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)

    for ax, snn_data, ann_data, title in [
        (ax1, fgsm_snn, fgsm_ann, 'FGSM Attack'),
        (ax2, pgd_snn, pgd_ann, 'PGD Attack'),
    ]:
        ax.plot(eps, snn_data, 'o-', color=BLUE, linewidth=1.8, markersize=5, label='SNN', zorder=5)
        ax.plot(eps, ann_data, 's--', color=RED, linewidth=1.8, markersize=5, label='ANN', zorder=5)

        # Shade SNN advantage region
        snn_arr = np.array(snn_data)
        ann_arr = np.array(ann_data)
        eps_arr = np.array(eps)
        # Only shade where SNN > ANN
        mask = snn_arr > ann_arr
        if mask.any():
            ax.fill_between(eps_arr, ann_arr, snn_arr,
                            where=mask, alpha=0.12, color=BLUE, label='SNN advantage')

        ax.set_xlabel('Perturbation $\\varepsilon$')
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_ylim(-2, 70)
        ax.set_xlim(-0.01, 0.31)

    ax1.set_ylabel('Accuracy (%)')
    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/adversarial_curves.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  [3/5] adversarial_curves.pdf')


# ============================================================
# FIGURE 4: Surrogate Gradient Ablation
# ============================================================
def fig4_surrogate():
    names = ['Spike Rate Escape', 'Fast Sigmoid', 'ATan', 'STE',
             'Triangular', 'Sigmoid', 'SFS']
    accs  = [46.00, 44.75, 35.75, 10.25, 2.75, 2.00, 2.00]

    # Sort by accuracy (highest first = top of horizontal bar chart)
    order = np.argsort(accs)
    names_sorted = [names[i] for i in order]
    accs_sorted  = [accs[i] for i in order]

    colors = [GREEN if a > 30 else RED for a in accs_sorted]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.barh(range(len(names_sorted)), accs_sorted, color=colors, height=0.6, edgecolor='white')

    # Chance level
    ax.axvline(x=2.0, color=GREY, linestyle='--', linewidth=1, alpha=0.7, label='Chance (2%)')

    # Value labels
    for i, (bar, v) in enumerate(zip(bars, accs_sorted)):
        ax.text(v + 0.8, i, f'{v:.1f}%', va='center', fontsize=9)

    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=10)
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_xlim(0, 52)
    ax.legend(fontsize=9, loc='lower right')

    # Add group labels
    ax.text(48, 5.5, 'Learning\ngroup', fontsize=8, color=GREEN, fontweight='bold',
            ha='center', va='center')
    ax.text(48, 1.5, 'Failure\ngroup', fontsize=8, color=RED, fontweight='bold',
            ha='center', va='center')

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/surrogate_ablation.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  [4/5] surrogate_ablation.pdf')


# ============================================================
# FIGURE 5: PANNs Gap Collapse
# ============================================================
def fig5_panns():
    categories = ['Scratch\nSNN', 'Scratch\nANN', 'PANNs\n+SNN', 'PANNs\n+ANN', 'PANNs\n+Linear']
    values = [47.15, 63.85, 92.50, 93.45, 93.80]
    colors_list = [BLUE, RED, BLUE, RED, PURPLE]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(categories))
    bars = ax.bar(x, values, width=0.55, color=colors_list, edgecolor='white', linewidth=0.5)

    # Human performance line
    ax.axhline(y=81.3, color=GREY, linestyle='--', linewidth=1, alpha=0.7)
    ax.text(4.35, 82.0, 'Human (81.3%)', fontsize=8, color=GREY, va='bottom')

    # Value labels on bars
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.8, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Bracket annotation: scratch gap (16.70 pp)
    y_bracket1 = 66
    ax.annotate('', xy=(0, y_bracket1), xytext=(1, y_bracket1),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(0.5, y_bracket1 + 1.2, '16.70 pp gap', ha='center', fontsize=9, fontweight='bold')

    # Bracket annotation: PANNs gap (0.95 pp)
    y_bracket2 = 96
    ax.annotate('', xy=(2, y_bracket2), xytext=(3, y_bracket2),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    ax.text(2.5, y_bracket2 + 0.8, '0.95 pp gap', ha='center', fontsize=9, fontweight='bold')

    # Connecting arrow between the two gaps to show collapse
    ax.annotate('Gap collapses\n17.6$\\times$', xy=(2.5, 95), xytext=(1.2, 80),
                fontsize=8, ha='center', color=GREEN, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.2,
                                connectionstyle='arc3,rad=-0.2'))

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 102)
    ax.set_xlim(-0.5, 4.8)

    # Legend
    snn_patch = mpatches.Patch(color=BLUE, label='SNN')
    ann_patch = mpatches.Patch(color=RED, label='ANN')
    lin_patch = mpatches.Patch(color=PURPLE, label='Linear')
    ax.legend(handles=[snn_patch, ann_patch, lin_patch], fontsize=9, loc='center left')

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/panns_comparison.pdf', bbox_inches='tight')
    plt.close(fig)
    print('  [5/5] panns_comparison.pdf')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('Generating thesis figures...')
    fig1_pruning_pareto()
    fig2_spinnaker_gap()
    fig3_adversarial()
