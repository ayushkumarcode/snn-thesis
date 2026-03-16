"""
Generate all paper-quality figures and tables for thesis/ICONS 2026.

Usage:
    python experiments/generate_paper_figures.py

Outputs to results/paper_figures/:
  - encoding_comparison.pdf    -- bar chart: all SNN encodings + ANN
  - adversarial_robustness.pdf -- accuracy vs epsilon, 2 panels (FGSM, PGD)
  - energy_table.txt           -- NeuroBench energy comparison
  - continual_forgetting.pdf   -- BWT/forgetting curves (if available)
  - temporal_analysis.pdf      -- rate vs first-spike accuracy
  - panns_comparison.pdf       -- PANNs comparison bar chart
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.config import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (colourblind-friendly) ──────────────────────────────────
ANN_COLOUR = "#2196F3"   # blue
SNN_COLOUR = "#F44336"   # red
PANNS_COLOUR = "#4CAF50" # green
GREY = "#9E9E9E"


def load_summary(model, encoding):
    p = RESULTS_DIR / model / encoding / "summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ── 1. Encoding comparison bar chart ──────────────────────────────────────
def plot_encoding_comparison():
    encodings = [
        ("SNN – Direct",     "snn", "direct",     SNN_COLOUR),
        ("SNN – Rate",       "snn", "rate",        "#EF9A9A"),
        ("SNN – Burst",      "snn", "burst",       "#FFCC80"),
        ("SNN – Phase",      "snn", "phase",       "#CE93D8"),
        ("SNN – Latency",    "snn", "latency",     "#A5D6A7"),
        ("SNN – Delta",      "snn", "delta",       "#80DEEA"),
        ("SNN – Population", "snn", "population",  "#FFF59D"),
        ("ANN baseline",     "ann", "none",         ANN_COLOUR),
    ]

    labels, means, stds = [], [], []
    colours = []
    for label, model, enc, colour in encodings:
        s = load_summary(model, enc)
        if s is not None:
            labels.append(label)
            means.append(s["mean_accuracy"] * 100)
            stds.append(s["std_accuracy"] * 100)
            colours.append(colour)

    if not labels:
        print("No encoding results found.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colours,
                  edgecolor="black", linewidth=0.6, error_kw={"linewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("5-Fold CV Accuracy (%)", fontsize=11)
    ax.set_title("SNN Encoding Comparison on ESC-50 (50 classes)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.axhline(81.3, color="gray", linestyle=":", linewidth=1,
               label="Human performance (81.3%)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars with accuracy values
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1.0,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = OUT_DIR / "encoding_comparison.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 2. Adversarial robustness plot ─────────────────────────────────────────
def plot_adversarial():
    adv_path = RESULTS_DIR / "adversarial" / "robustness_fold4.json"
    if not adv_path.exists():
        print("No adversarial results found.")
        return

    with open(adv_path) as f:
        d = json.load(f)

    epsilons = d["epsilons"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, attack in zip(axes, ["fgsm", "pgd"]):
        snn_accs = [a * 100 for a in d[attack]["snn"]]
        ann_accs = [a * 100 for a in d[attack]["ann"]]
        ax.plot(epsilons, snn_accs, "o-", color=SNN_COLOUR, label="SNN (direct)",
                linewidth=2, markersize=6)
        ax.plot(epsilons, ann_accs, "s--", color=ANN_COLOUR, label="ANN",
                linewidth=2, markersize=6)
        ax.set_xlabel("Perturbation ε", fontsize=11)
        ax.set_title(f"{attack.upper()} Attack", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 75)
        ax.grid(alpha=0.3)
        ax.fill_between(epsilons, snn_accs, ann_accs,
                        where=[s > a for s, a in zip(snn_accs, ann_accs)],
                        alpha=0.15, color=SNN_COLOUR, label="SNN advantage")

    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    fig.suptitle("Adversarial Robustness: SNN vs ANN on ESC-50 (fold 4, 400 samples)",
                 fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "adversarial_robustness.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 3. PANNs comparison bar chart ──────────────────────────────────────────
def plot_panns():
    panns_path = RESULTS_DIR / "panns" / "panns_snn_head_all_folds_50ep.json"
    if not panns_path.exists():
        print("No PANNs 5-fold results found.")
        return

    with open(panns_path) as f:
        d = json.load(f)

    models = ["snn", "ann", "linear"]
    labels = ["PANNs + SNN head", "PANNs + ANN head", "PANNs + Linear"]
    colours = [SNN_COLOUR, ANN_COLOUR, GREY]

    means, stds = [], []
    for m in models:
        r = d["results"].get(m)
        if r:
            means.append(r["mean_accuracy"] * 100)
            stds.append(r["std_accuracy"] * 100)
        else:
            means.append(0)
            stds.append(0)

    # Also add scratch SNN and ANN for comparison
    scratch_snn = load_summary("snn", "direct")
    scratch_ann = load_summary("ann", "none")
    if scratch_snn:
        labels.insert(0, "SNN (scratch)")
        means.insert(0, scratch_snn["mean_accuracy"] * 100)
        stds.insert(0, scratch_snn["std_accuracy"] * 100)
        colours.insert(0, "#EF9A9A")
    if scratch_ann:
        labels.insert(1, "ANN (scratch)")
        means.insert(1, scratch_ann["mean_accuracy"] * 100)
        stds.insert(1, scratch_ann["std_accuracy"] * 100)
        colours.insert(1, "#90CAF9")

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colours,
                  edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("5-Fold CV Accuracy (%)", fontsize=11)
    ax.set_title("Impact of AudioSet Pre-training (PANNs) on SNN vs ANN", fontsize=12)
    ax.set_ylim(0, 100)
    ax.axhline(81.3, color="gray", linestyle=":", linewidth=1,
               label="Human (81.3%)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 1.0,
                f"{mean:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = OUT_DIR / "panns_comparison.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── 4. Energy comparison table ─────────────────────────────────────────────
def print_energy_table():
    nb_path = RESULTS_DIR / "neurobench" / "analysis_fold4.json"
    if not nb_path.exists():
        print("No NeuroBench results found.")
        return

    with open(nb_path) as f:
        d = json.load(f)

    print("\n=== NeuroBench Energy Table ===")
    print(f"{'Metric':<35} {'SNN':>15} {'ANN':>15}")
    print("-" * 67)
    en = d.get("energy", {})
    snn_m = d.get("snn", {})
    ann_m = d.get("ann", {})

    rows = [
        ("Classification Accuracy", f"{snn_m.get('ClassificationAccuracy',0)*100:.1f}%",
         f"{ann_m.get('ClassificationAccuracy',0)*100:.1f}%"),
        ("Activation Sparsity", f"{snn_m.get('ActivationSparsity',0)*100:.1f}%",
         f"{ann_m.get('ActivationSparsity',0)*100:.1f}%"),
        ("Eff. ACs / sample", f"{en.get('snn_eff_acs_per_sample',0)/1e6:.2f}M", "0"),
        ("Eff. MACs / sample", "0", f"{en.get('ann_eff_macs_per_sample',0)/1e3:.1f}K"),
        ("Energy / sample (nJ)", f"{en.get('snn_energy_pj_per_sample',0)/1e3:.3f}",
         f"{en.get('ann_energy_pj_per_sample',0)/1e3:.3f}"),
        ("Parameters", f"{snn_m.get('ParameterCount',0)/1e3:.0f}K",
         f"{ann_m.get('ParameterCount',0)/1e3:.0f}K"),
    ]
    for label, sv, av in rows:
        print(f"  {label:<33} {sv:>15} {av:>15}")
    print()

    # Save as text
    out = OUT_DIR / "energy_table.txt"
    with open(out, "w") as f:
        f.write("NeuroBench Energy Analysis (fold 4)\n")
        f.write("="*67 + "\n")
        f.write(f"{'Metric':<35} {'SNN':>15} {'ANN':>15}\n")
        f.write("-"*67 + "\n")
        for label, sv, av in rows:
            f.write(f"  {label:<33} {sv:>15} {av:>15}\n")
    print(f"Saved: {out}")


# ── 5. Continual learning forgetting curves ────────────────────────────────
def plot_continual():
    cl_dir = RESULTS_DIR / "continual_learning"
    if not cl_dir.exists():
        print("No continual learning results yet.")
        return

    files = list(cl_dir.glob("*.json"))
    if not files:
        print("No continual learning result files.")
        return

    with open(files[0]) as f:
        d = json.load(f)

    tasks = [t["name"] for t in d["tasks"]]
    n_tasks = len(tasks)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model_key, label, colour in [
        (axes[0], "snn", "SNN", SNN_COLOUR),
        (axes[1], "ann", "ANN", ANN_COLOUR),
    ]:
        res = d.get(model_key, {})
        mat = res.get("task_accuracy_matrix", [])
        if not mat:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        mat = np.array(mat)
        for i in range(mat.shape[1]):
            valid = mat[i:, i]  # task i evaluated after tasks i, i+1, ...
            xs = list(range(i, i + len(valid)))
            ax.plot(xs, valid * 100, "o-", label=f"Task {i+1}: {tasks[i][:12]}",
                    linewidth=1.5, markersize=4)

        ax.set_title(f"{label} — Task Accuracy Over Time", fontsize=11)
        ax.set_xlabel("Task index (sequential training)", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

        forgetting = res.get("mean_forgetting", None)
        bwt = res.get("mean_bwt", None)
        if forgetting is not None:
            ax.set_title(
                f"{label} — Forgetting: {forgetting*100:.1f}%  BWT: {bwt*100:+.1f}%",
                fontsize=10)

    fig.suptitle("Continual Learning: SNN vs ANN Catastrophic Forgetting (ESC-50)",
                 fontsize=12)
    plt.tight_layout()
    out = OUT_DIR / "continual_forgetting.pdf"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}\n")
    plot_encoding_comparison()
    plot_adversarial()
    plot_panns()
    print_energy_table()
    plot_continual()
    print("\nDone. All figures saved to results/paper_figures/")
