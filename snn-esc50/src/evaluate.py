"""
Evaluation metrics and analysis for ESC-50 experiments.

Generates:
  - Accuracy per fold
  - Confusion matrices
  - Per-class accuracy analysis
  - F1 scores
  - Cross-fold statistical summary
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)

from src.config import RESULTS_DIR, NUM_FOLDS, NUM_CLASSES
from src.dataset import get_class_labels


def load_predictions(model_type: str, encoding: str, fold: int) -> tuple:
    """Load saved predictions for a given fold."""
    path = RESULTS_DIR / model_type / encoding / f"preds_fold{fold}.pt"
    data = torch.load(path, weights_only=True)
    return data["preds"], data["targets"]


def compute_fold_metrics(preds: list, targets: list, class_labels: list[str]) -> dict:
    """Compute all metrics for a single fold."""
    acc = accuracy_score(targets, preds)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(targets, preds, average="weighted", zero_division=0)
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))

    # Per-class accuracy
    per_class_acc = {}
    for i, label in enumerate(class_labels):
        mask = np.array(targets) == i
        if mask.sum() > 0:
            per_class_acc[label] = accuracy_score(
                np.array(targets)[mask], np.array(preds)[mask],
            )
        else:
            per_class_acc[label] = 0.0

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_acc,
    }


def plot_confusion_matrix(cm: np.ndarray, class_labels: list[str],
                          title: str, save_path: Path):
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=class_labels, yticklabels=class_labels, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix: {save_path}")


def plot_per_class_accuracy(per_class_acc: dict, title: str, save_path: Path):
    """Plot per-class accuracy as a horizontal bar chart."""
    sorted_items = sorted(per_class_acc.items(), key=lambda x: x[1])
    labels = [item[0] for item in sorted_items]
    accs = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 14))
    colors = ["#d32f2f" if a < 0.5 else "#1976d2" if a < 0.75 else "#388e3c" for a in accs]
    ax.barh(labels, accs, color=colors)
    ax.set_xlabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved per-class accuracy: {save_path}")


def plot_training_curves(model_type: str, encoding: str, save_path: Path):
    """Plot training/test loss and accuracy across epochs for all folds."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    for fold in range(1, NUM_FOLDS + 1):
        result_path = RESULTS_DIR / model_type / encoding / f"result_fold{fold}.json"
        if not result_path.exists():
            continue
        with open(result_path) as f:
            result = json.load(f)
        history = result["history"]

        axes[0].plot(history["train_loss"], alpha=0.5, label=f"Train F{fold}")
        axes[0].plot(history["test_loss"], alpha=0.7, linestyle="--", label=f"Test F{fold}")
        axes[1].plot(history["train_acc"], alpha=0.5, label=f"Train F{fold}")
        axes[1].plot(history["test_acc"], alpha=0.7, linestyle="--", label=f"Test F{fold}")

    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Training Curves: {model_type.upper()} ({encoding})")
    axes[0].legend(fontsize=7)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved training curves: {save_path}")


def evaluate_experiment(model_type: str, encoding: str):
    """Run full evaluation for an experiment (all 5 folds)."""
    print(f"\nEvaluating {model_type.upper()} / {encoding}")
    class_labels = get_class_labels()

    all_preds = []
    all_targets = []
    fold_accs = []

    save_dir = RESULTS_DIR / model_type / encoding
    save_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(1, NUM_FOLDS + 1):
        preds, targets = load_predictions(model_type, encoding, fold)
        metrics = compute_fold_metrics(preds, targets, class_labels)
        fold_accs.append(metrics["accuracy"])

        print(f"  Fold {fold}: Acc={metrics['accuracy']:.4f} F1={metrics['f1_macro']:.4f}")

        all_preds.extend(preds)
        all_targets.extend(targets)

    # Aggregate confusion matrix across all folds
    agg_cm = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))

    # Aggregate per-class accuracy
    agg_metrics = compute_fold_metrics(all_preds, all_targets, class_labels)

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"  Overall: {mean_acc:.4f} +/- {std_acc:.4f}")

    # Save plots
    plot_confusion_matrix(
        np.array(agg_cm), class_labels,
        f"Confusion Matrix: {model_type.upper()} ({encoding})\n"
        f"Accuracy: {mean_acc:.2%} +/- {std_acc:.2%}",
        save_dir / "confusion_matrix.png",
    )

    plot_per_class_accuracy(
        agg_metrics["per_class_accuracy"],
        f"Per-Class Accuracy: {model_type.upper()} ({encoding})",
        save_dir / "per_class_accuracy.png",
    )

    plot_training_curves(model_type, encoding, save_dir / "training_curves.png")

    # Save aggregate metrics
    summary = {
        "model": model_type,
        "encoding": encoding,
        "fold_accuracies": fold_accs,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "aggregate_f1_macro": agg_metrics["f1_macro"],
        "aggregate_f1_weighted": agg_metrics["f1_weighted"],
        "per_class_accuracy": agg_metrics["per_class_accuracy"],
    }
    with open(save_dir / "evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_comparison_table(experiments: list[tuple[str, str]]):
    """Generate a comparison table across experiments.

    Args:
        experiments: List of (model_type, encoding) tuples.
    """
    print(f"\n{'='*70}")
    print(f"{'Model':<8} {'Encoding':<10} {'Mean Acc':>10} {'Std':>8} {'F1 Macro':>10}")
    print(f"{'-'*70}")

    rows = []
    for model_type, encoding in experiments:
        eval_path = RESULTS_DIR / model_type / encoding / "evaluation.json"
        if not eval_path.exists():
            print(f"  {model_type:<8} {encoding:<10} -- not evaluated yet --")
            continue
        with open(eval_path) as f:
            data = json.load(f)
        print(
            f"  {model_type:<8} {encoding:<10} "
            f"{data['mean_accuracy']:>10.4f} {data['std_accuracy']:>8.4f} "
            f"{data['aggregate_f1_macro']:>10.4f}"
        )
        rows.append(data)

    print(f"{'='*70}")
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["snn", "ann"])
    parser.add_argument("--encoding", default="rate")
    args = parser.parse_args()

    evaluate_experiment(args.model, args.encoding)
