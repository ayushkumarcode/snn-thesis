"""Generate Figure 1: SpikingCNN Architecture Diagram for ICONS paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(12, 3.5))

# Layer definitions: (name, width, height, color, text)
layers = [
    ("Input\n1×64×216", 0.6, 2.0, "#E3F2FD", "Mel\nSpectrogram"),
    ("Conv1+BN\n32×64×216", 0.8, 2.4, "#1976D2", "Conv2d(1→32)\nBN, k=3"),
    ("Pool1\n32×32×108", 0.5, 1.8, "#42A5F5", "MaxPool(2)"),
    ("LIF₁\n32×32×108", 0.5, 1.8, "#FF9800", "Spike\nβ=0.95"),
    ("Conv2+BN\n64×32×108", 0.8, 2.4, "#1565C0", "Conv2d(32→64)\nBN, k=3"),
    ("Pool2\n64×16×54", 0.5, 1.6, "#42A5F5", "MaxPool(2)"),
    ("LIF₂\n64×16×54", 0.5, 1.6, "#FF9800", "Spike\nβ=0.95"),
    ("AvgPool\n64×4×9", 0.5, 1.2, "#90CAF9", "AvgPool\n(4×6)"),
    ("FC₁\n256", 0.7, 2.0, "#1976D2", "Linear\n2304→256"),
    ("LIF₃\n256", 0.5, 1.6, "#FF9800", "Spike\nβ=0.95"),
    ("FC₂\n50", 0.7, 1.4, "#1565C0", "Linear\n256→50"),
    ("LIF₄\n50", 0.5, 1.2, "#FF9800", "Output\nSpikes"),
]

x_pos = 0.3
y_center = 1.5
arrow_props = dict(arrowstyle='->', color='black', lw=1.5)

for i, (label, w, h, color, text) in enumerate(layers):
    # Draw box
    rect = FancyBboxPatch(
        (x_pos - w/2, y_center - h/2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor='black', linewidth=1.2,
        alpha=0.85,
    )
    ax.add_patch(rect)

    # Add text
    fontcolor = 'white' if color in ['#1976D2', '#1565C0'] else 'black'
    ax.text(x_pos, y_center + 0.05, text, ha='center', va='center',
            fontsize=6, fontweight='bold', color=fontcolor)

    # Dimension label below
    ax.text(x_pos, y_center - h/2 - 0.15, label.split('\n')[-1],
            ha='center', va='top', fontsize=5, color='gray', style='italic')

    # Arrow to next
    if i < len(layers) - 1:
        next_w = layers[i+1][1]
        ax.annotate('', xy=(x_pos + w/2 + 0.08 + next_w/2, y_center),
                    xytext=(x_pos + w/2 + 0.02, y_center),
                    arrowprops=arrow_props)

    x_pos += w + 0.1

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#1976D2', edgecolor='black', label='Conv/FC Layer'),
    mpatches.Patch(facecolor='#FF9800', edgecolor='black', label='LIF Spiking Neuron'),
    mpatches.Patch(facecolor='#42A5F5', edgecolor='black', label='Pooling'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=7, framealpha=0.9)

# Title
ax.set_title('SpikingCNN Architecture (~622K parameters, T=25 timesteps)', fontsize=11, fontweight='bold')

ax.set_xlim(-0.1, x_pos + 0.3)
ax.set_ylim(-0.3, 3.2)
ax.axis('off')

plt.tight_layout()
plt.savefig('paper/figures/architecture_diagram.png', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/architecture_diagram.pdf', bbox_inches='tight')
plt.close()
print("Saved: paper/figures/architecture_diagram.png + .pdf")
