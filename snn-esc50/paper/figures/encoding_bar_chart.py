"""Generate Figure 3: Encoding comparison bar chart for ICONS paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data from 5-fold cross-validation results
encodings = ["Direct", "Rate", "Phase", "Population", "Latency", "Delta", "Burst", "ANN"]
means = [47.15, 24.00, 24.15, 19.15, 16.30, 7.25, 6.50, 63.85]
stds = [4.50, 1.90, 1.66, 2.79, 1.62, 0.94, 1.54, 3.07]

# Colors: SNN encodings in blue shades, ANN in red
colors = ['#1565C0', '#1976D2', '#1E88E5', '#2196F3', '#42A5F5', '#90CAF9', '#BBDEFB', '#D32F2F']

fig, ax = plt.subplots(figsize=(8, 4.5))

x = np.arange(len(encodings))
bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor='black',
              linewidth=0.5, error_kw={'linewidth': 1.2})

# Add value labels on bars
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 1,
            f'{mean:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(encodings, rotation=30, ha='right', fontsize=9)
ax.set_ylabel('Test Accuracy (%)', fontsize=11)
ax.set_title('ESC-50 Classification: 7 SNN Encodings vs ANN Baseline', fontsize=11)
ax.set_ylim(0, 78)
ax.grid(axis='y', alpha=0.3)

# Add horizontal line for chance level
ax.axhline(y=2, color='gray', linestyle=':', alpha=0.5, label='Chance (2%)')

# Separator between SNN and ANN
ax.axvline(x=6.5, color='black', linestyle='--', alpha=0.3)
ax.text(3, 72, 'SNN Encodings', ha='center', fontsize=9, style='italic', color='#1565C0')
ax.text(7, 72, 'ANN', ha='center', fontsize=9, style='italic', color='#D32F2F')

plt.tight_layout()
plt.savefig('paper/figures/encoding_bar_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('paper/figures/encoding_bar_chart.pdf', bbox_inches='tight')
plt.close()
print("Saved: paper/figures/encoding_bar_chart.png + .pdf")
