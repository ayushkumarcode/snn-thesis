"""Generate Figure 1: SpikingCNN Architecture Diagram for ICONS paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Figure setup ──────────────────────────────────────────────────────────
# Wider figure to give boxes room; shorter height for two-column paper
fig, ax = plt.subplots(figsize=(14, 3.8))

# ── Layer definitions ─────────────────────────────────────────────────────
# (label_below, box_width, box_height, color, display_text)
layers = [
    ("Input\n1x64x216",    0.65, 1.8, "#E3F2FD", "Mel\nSpectrogram"),
    ("Conv1+BN\n32x64x216", 0.85, 2.2, "#1976D2", "Conv2d(1->32)\nBN, k=3"),
    ("Pool1\n32x32x108",   0.55, 1.6, "#42A5F5", "MaxPool(2)"),
    ("LIF1\n32x32x108",    0.55, 1.6, "#FF9800", "Spike\nbeta=0.95"),
    ("Conv2+BN\n64x32x108", 0.85, 2.2, "#1565C0", "Conv2d(32->64)\nBN, k=3"),
    ("Pool2\n64x16x54",    0.55, 1.4, "#42A5F5", "MaxPool(2)"),
    ("LIF2\n64x16x54",     0.55, 1.4, "#FF9800", "Spike\nbeta=0.95"),
    ("AvgPool\n64x4x9",    0.55, 1.1, "#90CAF9", "AvgPool\n(4x6)"),
    ("FC1\n256",            0.75, 1.8, "#1976D2", "Linear\n2304->256"),
    ("LIF3\n256",           0.55, 1.4, "#FF9800", "Spike\nbeta=0.95"),
    ("FC2\n50",             0.75, 1.2, "#1565C0", "Linear\n256->50"),
    ("LIF4\n50",            0.55, 1.1, "#FF9800", "Output\nSpikes"),
]

# ── Layout constants ──────────────────────────────────────────────────────
gap = 0.25          # horizontal gap between box edges (was 0.1, too tight)
box_pad = 0.04      # FancyBboxPatch rounding pad
y_center = 1.7      # vertical center for all boxes
arrow_props = dict(arrowstyle='->', color='black', lw=1.5,
                   mutation_scale=12)

# ── Draw layers ───────────────────────────────────────────────────────────
x_pos = 0.5  # left edge of first box (center will be x_pos + w/2)

box_centers = []  # store (cx, w, h) for arrow drawing

for i, (label, w, h, color, text) in enumerate(layers):
    cx = x_pos + w / 2.0
    cy = y_center

    # Draw rounded box
    rect = FancyBboxPatch(
        (x_pos, cy - h / 2.0), w, h,
        boxstyle=f"round,pad={box_pad}",
        facecolor=color, edgecolor='black', linewidth=1.2,
        alpha=0.88, zorder=2,
    )
    ax.add_patch(rect)

    # Inner text (layer type)
    fontcolor = 'white' if color in ('#1976D2', '#1565C0') else 'black'
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=6.5, fontweight='bold', color=fontcolor,
            zorder=3, linespacing=1.15)

    # Dimension label below the box
    dim_label = label.split('\n')[-1]
    ax.text(cx, cy - h / 2.0 - box_pad - 0.12, dim_label,
            ha='center', va='top', fontsize=5, color='#555555',
            style='italic', zorder=3)

    box_centers.append((cx, w, h))
    x_pos += w + gap

# ── Draw arrows between consecutive boxes ─────────────────────────────────
for i in range(len(box_centers) - 1):
    cx1, w1, h1 = box_centers[i]
    cx2, w2, h2 = box_centers[i + 1]
    # Arrow from right edge of box i to left edge of box i+1
    x_start = cx1 + w1 / 2.0 + box_pad + 0.02
    x_end   = cx2 - w2 / 2.0 - box_pad - 0.02
    ax.annotate('', xy=(x_end, y_center), xytext=(x_start, y_center),
                arrowprops=arrow_props, zorder=1)

# ── Legend (placed top-left to avoid overlap with boxes) ──────────────────
legend_elements = [
    mpatches.Patch(facecolor='#1976D2', edgecolor='black',
                   label='Conv / FC Layer'),
    mpatches.Patch(facecolor='#FF9800', edgecolor='black',
                   label='LIF Spiking Neuron'),
    mpatches.Patch(facecolor='#42A5F5', edgecolor='black',
                   label='Pooling Layer'),
    mpatches.Patch(facecolor='#E3F2FD', edgecolor='black',
                   label='Input'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=7,
          framealpha=0.95, edgecolor='#cccccc', borderpad=0.6,
          handlelength=1.2, handletextpad=0.5)

# ── Title ─────────────────────────────────────────────────────────────────
ax.set_title('SpikingCNN Architecture (~622K parameters, T=25 timesteps)',
             fontsize=11, fontweight='bold', pad=8)

# ── Axis limits ───────────────────────────────────────────────────────────
ax.set_xlim(0.0, x_pos + 0.2)
ax.set_ylim(-0.1, 3.4)
ax.axis('off')

plt.tight_layout(pad=0.3)
plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight')
plt.savefig('architecture_diagram.pdf', bbox_inches='tight')
plt.close()
print("Saved: architecture_diagram.png + .pdf")
