"""Generate Figure 2: SpiNNaker Hybrid Deployment Pipeline for ICONS paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 4))

def draw_box(ax, x, y, w, h, text, color, fontcolor='white', fontsize=8, alpha=0.9):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor='black', linewidth=1.5,
        alpha=alpha,
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=fontcolor)

def draw_arrow(ax, x1, y1, x2, y2, color='black', style='->', lw=2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# ====== SOFTWARE SIDE (snnTorch on CPU/GPU) ======
# Background box
soft_bg = FancyBboxPatch(
    (0.2, 0.3), 6.5, 3.2,
    boxstyle="round,pad=0.15",
    facecolor='#E8EAF6', edgecolor='#3F51B5', linewidth=2,
    alpha=0.4, linestyle='--',
)
ax.add_patch(soft_bg)
ax.text(3.45, 3.3, 'Software (snnTorch on CPU/GPU)', ha='center',
        fontsize=10, fontweight='bold', color='#283593', style='italic')

# Input
draw_box(ax, 1.0, 2.0, 1.2, 0.8, 'Audio\n5s clip', '#E3F2FD', fontcolor='black')

# Mel spectrogram
draw_box(ax, 2.8, 2.0, 1.2, 0.8, 'Mel\nSpectrogram\n64×216', '#90CAF9', fontcolor='black')

# Conv layers
draw_box(ax, 4.6, 2.0, 1.2, 0.8, 'Conv1→LIF₁\nConv2→LIF₂\nAvgPool', '#1976D2')

# FC1 + LIF3
draw_box(ax, 6.2, 2.0, 1.0, 0.8, 'FC₁→LIF₃\n(256-d)', '#1565C0')

# Arrows in software
draw_arrow(ax, 1.6, 2.0, 2.2, 2.0)
draw_arrow(ax, 3.4, 2.0, 4.0, 2.0)
draw_arrow(ax, 5.2, 2.0, 5.7, 2.0)

# Binary spikes label
draw_box(ax, 7.5, 2.0, 0.9, 0.6, 'Binary\nSpikes\n(256-d)', '#FFF9C4', fontcolor='black', fontsize=7)
draw_arrow(ax, 6.7, 2.0, 7.05, 2.0)

# ====== HARDWARE SIDE (SpiNNaker) ======
hw_bg = FancyBboxPatch(
    (8.2, 0.3), 3.5, 3.2,
    boxstyle="round,pad=0.15",
    facecolor='#FBE9E7', edgecolor='#BF360C', linewidth=2,
    alpha=0.4, linestyle='--',
)
ax.add_patch(hw_bg)
ax.text(9.95, 3.3, 'SpiNNaker Hardware', ha='center',
        fontsize=10, fontweight='bold', color='#BF360C', style='italic')

# FC2 on SpiNNaker
draw_box(ax, 9.2, 2.0, 1.0, 0.8, 'FC₂→LIF₄\n(256→50)', '#D32F2F')

# Classification output
draw_box(ax, 10.8, 2.0, 1.0, 0.8, 'Class\nPrediction\n(1 of 50)', '#4CAF50')

# Arrows hardware
draw_arrow(ax, 7.95, 2.0, 8.7, 2.0, color='#BF360C', lw=2.5)
draw_arrow(ax, 9.7, 2.0, 10.3, 2.0)

# ====== CROSSED OUT: Full FC1 deployment ======
# Show the failed path
ax.text(4.6, 0.7, 'FC₁ on SpiNNaker', ha='center', fontsize=8, color='#D32F2F',
        style='italic', alpha=0.7)
ax.plot([3.8, 5.4], [0.5, 0.9], color='red', lw=3, alpha=0.6)
ax.plot([3.8, 5.4], [0.9, 0.5], color='red', lw=3, alpha=0.6)
ax.text(4.6, 0.3, '(Failed: E/I cancellation\nfrom AvgPool outputs)', ha='center',
        fontsize=6, color='#D32F2F', alpha=0.8)

# Stats annotation
ax.text(7.5, 0.7, '21.7% active\nper timestep\n(sparse binary)', ha='center',
        fontsize=6, color='#6A1B9A', style='italic',
        bbox=dict(boxstyle='round', facecolor='#F3E5F5', alpha=0.7))

ax.set_xlim(-0.2, 12.2)
ax.set_ylim(-0.1, 3.8)
ax.axis('off')
ax.set_title('Hybrid SpiNNaker Deployment Pipeline (FC2-only)', fontsize=12, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('paper/figures/spinnaker_pipeline.png', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/spinnaker_pipeline.pdf', bbox_inches='tight')
plt.close()
print("Saved: paper/figures/spinnaker_pipeline.png + .pdf")
