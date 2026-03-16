"""
analysis_suite.py -- Full analysis: confusion matrices, per-class, t-SNE, statistics.

Produces publication-quality figures and a comprehensive JSON results file.

Analyses:
  1. 5-fold accuracy summary with 95% CIs
  2. Paired t-test + Wilcoxon signed-rank test (SNN vs ANN)
  3. Per-class accuracy (best/worst 10 for each model)
  4. 50×50 confusion matrices (best fold = fold 4)
  5. t-SNE of FC1 activations (256-d → 2-d) with class colouring
  6. Comparison of what each model fails on differently

Usage:
  source .venv/bin/activate
  cd snn-esc50/
  python experiments/analysis_suite.py
"""

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch

from src.config import RESULTS_DIR, NUM_CLASSES, NUM_STEPS
from src.models.snn_model import SpikingCNN
from src.models.ann_model import ConvANN
from src.encoding import encode_direct
from src.dataset import get_fold_dataloaders, get_class_labels


# ============================================================
# 1. Load all fold predictions
# ============================================================

def load_all_preds(model_type: str, encoding: str) -> list:
    """Load per-fold (preds, targets) from saved .pt files."""
    results = []
    for fold in range(1, 6):
        p = RESULTS_DIR / model_type / encoding / f"preds_fold{fold}.pt"
        if not p.exists():
            print(f"WARNING: missing {p}")
            results.append(None)
            continue
        d = torch.load(p, map_location="cpu", weights_only=True)
        results.append({
            "fold": fold,
            "preds":   np.array(d["preds"]),
            "targets": np.array(d["targets"]),
            "acc": float(np.mean(np.array(d["preds"]) == np.array(d["targets"]))),
        })
    return results


# ============================================================
# 2. Statistical tests
# ============================================================

def stats_tests(accs_a: list, accs_b: list, label_a: str, label_b: str) -> dict:
    """Paired t-test and Wilcoxon signed-rank test."""
    from scipy.stats import ttest_rel, wilcoxon

    accs_a = np.array(accs_a)
    accs_b = np.array(accs_b)
    diff = accs_b - accs_a  # positive = B is better

    _, p_ttest   = ttest_rel(accs_b, accs_a, alternative="two-sided")
    _, p_wilcoxon = wilcoxon(accs_b, accs_a, alternative="two-sided")

    mean_a = float(accs_a.mean())
    mean_b = float(accs_b.mean())
    std_a  = float(accs_a.std())
    std_b  = float(accs_b.std())

    # 95% CI on the mean difference (paired)
    n = len(diff)
    mean_diff = float(diff.mean())
    se_diff   = float(diff.std() / np.sqrt(n))
    ci95      = 1.96 * se_diff

    return {
        f"mean_{label_a}": mean_a,
        f"std_{label_a}":  std_a,
        f"mean_{label_b}": mean_b,
        f"std_{label_b}":  std_b,
        "mean_difference_b_minus_a": mean_diff,
        "ci95_difference": ci95,
        "p_ttest_paired":   float(p_ttest),
        "p_wilcoxon":       float(p_wilcoxon),
        "significant_0.05": bool(p_ttest < 0.05),
        "significant_0.01": bool(p_ttest < 0.01),
    }


# ============================================================
# 3. Per-class accuracy
# ============================================================

def per_class_accuracy(all_fold_data: list) -> dict:
    """Compute per-class accuracy aggregated across all 5 folds."""
    preds   = np.concatenate([d["preds"]   for d in all_fold_data if d])
    targets = np.concatenate([d["targets"] for d in all_fold_data if d])

    class_labels = get_class_labels()
    per_class = {}
    for c in range(NUM_CLASSES):
        mask = targets == c
        if mask.sum() > 0:
            acc = float((preds[mask] == targets[mask]).mean())
            per_class[class_labels[c]] = {
                "class_id": c,
                "accuracy": acc,
                "n_samples": int(mask.sum()),
            }
    return per_class


# ============================================================
# 4. Confusion matrix
# ============================================================

def confusion_matrix(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return (NUM_CLASSES, NUM_CLASSES) confusion matrix."""
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm


# ============================================================
# 5. t-SNE on FC1 activations
# ============================================================

def extract_fc1_activations(model: SpikingCNN, loader,
                              device: str) -> tuple:
    """Extract post-FC1 (256-d) spike activations from SNN."""
    model.eval()
    all_activations = []
    all_targets     = []

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            spk_input = encode_direct(data, num_steps=NUM_STEPS)

            # Run SNN and collect spk3 (FC1 output spikes)
            mem1 = model.lif1.init_leaky()
            mem2 = model.lif2.init_leaky()
            mem3 = model.lif3.init_leaky()
            mem4 = model.lif4.init_leaky()

            spk3_acc = torch.zeros(data.size(0), 256, device=device)

            for step in range(NUM_STEPS):
                x_t = spk_input[step]
                cur1 = model.pool1(model.bn1(model.conv1(x_t)))
                spk1, mem1 = model.lif1(cur1, mem1)
                cur2 = model.pool2(model.bn2(model.conv2(spk1)))
                spk2, mem2 = model.lif2(cur2, mem2)
                pooled = model.avg_pool(spk2)
                flat = pooled.view(pooled.size(0), -1)
                cur3 = model.fc1(flat)
                spk3, mem3 = model.lif3(cur3, mem3)
                spk3_acc += spk3  # accumulate spikes over time

            # Normalise by num_steps to get firing rate
            spk3_acc = spk3_acc / NUM_STEPS
            all_activations.append(spk3_acc.cpu().numpy())
            all_targets.extend(targets.cpu().numpy().tolist())

    return np.concatenate(all_activations, axis=0), np.array(all_targets)


def extract_ann_fc1_activations(model: ConvANN, loader, device: str) -> tuple:
    """Extract post-first-FC (256-d) ReLU activations from ANN.

    ConvANN uses self.features (Sequential) + self.classifier (Sequential):
      classifier[0] = Linear(2304, 256)
      classifier[1] = ReLU
      classifier[2] = Dropout
      classifier[3] = Linear(256, 50)
    """
    model.eval()
    all_activations = []
    all_targets     = []

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            h = model.features(data)               # (B, 64, 4, 9)
            h = h.view(h.size(0), -1)              # (B, 2304)
            h = model.classifier[0](h)             # Linear(2304→256)
            h = model.classifier[1](h)             # ReLU  → (B, 256)

            all_activations.append(h.cpu().numpy())
            all_targets.extend(targets.cpu().numpy().tolist())

    return np.concatenate(all_activations, axis=0), np.array(all_targets)


def run_tsne(activations: np.ndarray,
             targets: np.ndarray,
             title: str,
             save_path: Path):
    """Compute t-SNE and save plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        print(f"  Running t-SNE on {activations.shape}...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        embedded = tsne.fit_transform(activations)

        # Assign colours: 5 super-categories × 10 classes each
        super_cat_colours = [
            "red", "blue", "green", "orange", "purple"
        ]  # Animals, Nature, Human, Domestic, Urban
        colors = [super_cat_colours[t // 10] for t in targets]

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            embedded[:, 0], embedded[:, 1],
            c=colors, alpha=0.4, s=8
        )

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="red",    label="Animals (0–9)"),
            Patch(facecolor="blue",   label="Nature (10–19)"),
            Patch(facecolor="green",  label="Human (20–29)"),
            Patch(facecolor="orange", label="Domestic (30–39)"),
            Patch(facecolor="purple", label="Urban (40–49)"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")

        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  t-SNE saved: {save_path}")
        return embedded.tolist()

    except ImportError as e:
        print(f"  t-SNE skipped (missing library: {e})")
        return None
    except Exception as e:
        print(f"  t-SNE failed: {e}")
        return None


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full analysis suite: SNN vs ANN")
    parser.add_argument("--fold", type=int, default=4,
                        help="Fold for confusion matrix + t-SNE (default: 4)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-tsne", action="store_true", default=False,
                        help="Skip t-SNE (slow on CPU)")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print("Full Analysis Suite: SNN vs ANN")
    print("=" * 60)
    print(f"  Confusion/t-SNE fold : {args.fold}")
    print(f"  Device               : {device}")
    print()

    out_dir = RESULTS_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    class_labels = get_class_labels()

    # --------------------------------------------------------
    # 1. Load all predictions
    # --------------------------------------------------------
    print("Loading predictions...")
    snn_folds = load_all_preds("snn", "direct")
    ann_folds = load_all_preds("ann", "none")

    snn_accs = [d["acc"] for d in snn_folds if d]
    ann_accs = [d["acc"] for d in ann_folds if d]

    print(f"  SNN 5-fold: {[f'{a:.4f}' for a in snn_accs]}")
    print(f"  SNN mean ± std: {np.mean(snn_accs):.4f} ± {np.std(snn_accs):.4f}")
    print(f"  ANN 5-fold: {[f'{a:.4f}' for a in ann_accs]}")
    print(f"  ANN mean ± std: {np.mean(ann_accs):.4f} ± {np.std(ann_accs):.4f}")

    # --------------------------------------------------------
    # 2. Statistical tests
    # --------------------------------------------------------
    print()
    print("Running statistical tests...")
    stats = stats_tests(snn_accs, ann_accs, "snn", "ann")
    print(f"  Mean difference (ANN - SNN): {stats['mean_difference_b_minus_a']:+.4f}")
    print(f"  95% CI on difference: ±{stats['ci95_difference']:.4f}")
    print(f"  Paired t-test p-value: {stats['p_ttest_paired']:.4f}")
    print(f"  Wilcoxon p-value: {stats['p_wilcoxon']:.4f}")
    if stats["significant_0.05"]:
        print("  FINDING: Gap is statistically significant at p<0.05.")
    else:
        print("  NOTE: Gap is NOT statistically significant at p<0.05 (n=5 is small).")

    # --------------------------------------------------------
    # 3. Per-class accuracy
    # --------------------------------------------------------
    print()
    print("Per-class accuracy (aggregated over all 5 folds)...")
    snn_per_class = per_class_accuracy(snn_folds)
    ann_per_class = per_class_accuracy(ann_folds)

    snn_sorted = sorted(snn_per_class.items(), key=lambda x: x[1]["accuracy"])
    ann_sorted = sorted(ann_per_class.items(), key=lambda x: x[1]["accuracy"])

    print(f"  {'Class':<25} {'SNN':>6} {'ANN':>6} {'Diff':>6}")
    print(f"  {'─'*50}")
    print("  SNN hardest 5:")
    for lbl, stats_c in snn_sorted[:5]:
        ann_acc = ann_per_class.get(lbl, {}).get("accuracy", 0)
        print(f"    {lbl:<25} {stats_c['accuracy']:>6.2%} {ann_acc:>6.2%} "
              f"{ann_acc - stats_c['accuracy']:>+6.2%}")

    print("  SNN easiest 5:")
    for lbl, stats_c in snn_sorted[-5:]:
        ann_acc = ann_per_class.get(lbl, {}).get("accuracy", 0)
        print(f"    {lbl:<25} {stats_c['accuracy']:>6.2%} {ann_acc:>6.2%} "
              f"{ann_acc - stats_c['accuracy']:>+6.2%}")

    # Classes where SNN is BETTER than ANN
    snn_better = {
        lbl: (snn_per_class[lbl]["accuracy"] - ann_per_class[lbl]["accuracy"])
        for lbl in snn_per_class
        if snn_per_class[lbl]["accuracy"] > ann_per_class.get(lbl, {}).get("accuracy", 0)
    }
    snn_better_sorted = sorted(snn_better.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Classes where SNN > ANN (top 5):")
    for lbl, diff in snn_better_sorted[:5]:
        print(f"    {lbl:<25} SNN better by {diff:+.2%}")

    # --------------------------------------------------------
    # 4. Confusion matrices
    # --------------------------------------------------------
    print()
    print(f"Computing confusion matrices (fold {args.fold})...")
    snn_fold_data = next((d for d in snn_folds if d and d["fold"] == args.fold), None)
    ann_fold_data = next((d for d in ann_folds if d and d["fold"] == args.fold), None)

    snn_cm = confusion_matrix(snn_fold_data["preds"], snn_fold_data["targets"])
    ann_cm = confusion_matrix(ann_fold_data["preds"], ann_fold_data["targets"])

    # Save confusion matrix figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for cm_data, model_name in [(snn_cm, "SNN"), (ann_cm, "ANN")]:
            fig, ax = plt.subplots(figsize=(16, 14))
            im = ax.imshow(cm_data, cmap="Blues", aspect="auto")
            ax.set_xlabel("Predicted class", fontsize=10)
            ax.set_ylabel("True class", fontsize=10)
            ax.set_title(f"{model_name} Confusion Matrix — Fold {args.fold}", fontsize=12)
            plt.colorbar(im, ax=ax)

            # Label axes with class names (abbreviated)
            short_labels = [lbl[:10] for lbl in class_labels]
            ax.set_xticks(range(NUM_CLASSES))
            ax.set_xticklabels(short_labels, rotation=90, fontsize=5)
            ax.set_yticks(range(NUM_CLASSES))
            ax.set_yticklabels(short_labels, fontsize=5)

            cm_path = out_dir / f"confusion_{model_name.lower()}_fold{args.fold}.png"
            fig.savefig(cm_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {cm_path}")

    except ImportError:
        print("  matplotlib not available — skipping confusion matrix plots")
    except Exception as e:
        print(f"  Confusion matrix plot failed: {e}")

    # --------------------------------------------------------
    # 5. t-SNE
    # --------------------------------------------------------
    if not args.skip_tsne:
        print()
        print(f"Extracting activations for t-SNE (fold {args.fold})...")

        _, test_loader = get_fold_dataloaders(args.fold, batch_size=64, augment=False)

        # SNN t-SNE
        snn_model = SpikingCNN().to(device)
        snn_model.load_state_dict(
            torch.load(RESULTS_DIR / "snn" / "direct" / f"best_fold{args.fold}.pt",
                       map_location=device, weights_only=True)
        )
        snn_act, snn_targets_tsne = extract_fc1_activations(snn_model, test_loader, device)
        del snn_model

        snn_tsne = run_tsne(
            snn_act, snn_targets_tsne,
            f"SNN Hidden Representations (FC1, 256-d) — Fold {args.fold}",
            out_dir / f"tsne_snn_fold{args.fold}.png",
        )

        # ANN t-SNE
        ann_model = ConvANN().to(device)
        ann_model.load_state_dict(
            torch.load(RESULTS_DIR / "ann" / "none" / f"best_fold{args.fold}.pt",
                       map_location=device, weights_only=True)
        )
        ann_act, ann_targets_tsne = extract_ann_fc1_activations(ann_model, test_loader, device)
        del ann_model

        ann_tsne = run_tsne(
            ann_act, ann_targets_tsne,
            f"ANN Hidden Representations (FC1, 256-d) — Fold {args.fold}",
            out_dir / f"tsne_ann_fold{args.fold}.png",
        )
    else:
        snn_tsne = None
        ann_tsne = None
        print("  t-SNE skipped (--skip-tsne).")

    # --------------------------------------------------------
    # Save full results
    # --------------------------------------------------------
    print()
    print("Saving results...")

    results = {
        "fold_for_cm_tsne": args.fold,
        "accuracy_summary": {
            "snn": {
                "fold_accuracies": snn_accs,
                "mean":  float(np.mean(snn_accs)),
                "std":   float(np.std(snn_accs)),
                "ci95":  1.96 * float(np.std(snn_accs) / np.sqrt(len(snn_accs))),
            },
            "ann": {
                "fold_accuracies": ann_accs,
                "mean":  float(np.mean(ann_accs)),
                "std":   float(np.std(ann_accs)),
                "ci95":  1.96 * float(np.std(ann_accs) / np.sqrt(len(ann_accs))),
            },
        },
        "statistical_tests": stats,
        "per_class_accuracy": {
            "snn": {lbl: v for lbl, v in snn_per_class.items()},
            "ann": {lbl: v for lbl, v in ann_per_class.items()},
            "snn_better_than_ann": dict(snn_better_sorted),
        },
        "confusion_matrices": {
            "snn_fold4": snn_cm.tolist(),
            "ann_fold4": ann_cm.tolist(),
        },
    }

    out_path = out_dir / "analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {out_path}")
    print()
    print("=" * 60)
    print("Analysis Summary")
    print("=" * 60)
    print(f"  ANN: {np.mean(ann_accs):.4f} ± {np.std(ann_accs):.4f} (95% CI ±{1.96 * np.std(ann_accs) / np.sqrt(5):.4f})")
    print(f"  SNN: {np.mean(snn_accs):.4f} ± {np.std(snn_accs):.4f} (95% CI ±{1.96 * np.std(snn_accs) / np.sqrt(5):.4f})")
    print(f"  Gap: {np.mean(ann_accs) - np.mean(snn_accs):.4f} (ANN > SNN)")
    print(f"  p-value (paired t-test): {stats['p_ttest_paired']:.4f}")
    print(f"  Classes where SNN wins: {len(snn_better)}/{NUM_CLASSES}")


if __name__ == "__main__":
    main()
