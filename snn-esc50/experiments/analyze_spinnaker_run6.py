#!/usr/bin/env python3
"""
Analyze SpiNNaker 400-sample fold-4 inference results (Run 6).
Reads results/spinnaker_results/fc2_all_iterations.jsonl and generates
the thesis §5.3 table, class-level breakdown, and agreement analysis.

Usage:
    python experiments/analyze_spinnaker_run6.py [--session-id SESSION_ID]
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

# ESC-50 class names (50 classes, 0-indexed)
ESC50_CLASSES = [
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow",  # 0-9: Animals
    "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
    "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm",  # 10-19: Nature
    "crying_baby", "sneezing", "clapping", "breathing", "coughing",
    "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping",  # 20-29: Human
    "door_knock", "mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening",
    "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking",  # 30-39: Domestic
    "helicopter", "chainsaw", "siren", "car_horn", "engine",
    "train", "church_bells", "airplane", "fireworks", "hand_saw",  # 40-49: Urban
]

RESULTS_DIR = Path(__file__).parent.parent / "results" / "spinnaker_results"


def load_run6_data(session_id=None):
    """Load Run 6 (400-sample) data from fc2_all_iterations.jsonl."""
    jsonl_path = RESULTS_DIR / "fc2_all_iterations.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    entries = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                e = json.loads(line.strip())
                if e.get("phase") != "inference":
                    continue
                if session_id and e.get("session_id") != session_id:
                    continue
                elif not session_id and e.get("timestamp", "").startswith("2026-03-04"):
                    entries.append(e)
                elif session_id:
                    entries.append(e)
            except json.JSONDecodeError:
                continue

    return entries


def analyze(entries):
    """Compute all metrics from the entries."""
    n = len(entries)
    if n == 0:
        return {"error": "No entries found"}

    # Overall accuracy
    spinnaker_correct = sum(1 for e in entries if e.get("correct", False))
    snn_correct = sum(1 for e in entries if e.get("snn_predicted") == e.get("true_label"))
    agreement = sum(1 for e in entries
                    if e.get("predicted") == e.get("snn_predicted"))

    # SpiNNaker correct but snnTorch wrong
    spinnaker_wins = sum(1 for e in entries
                         if e.get("correct") and e.get("snn_predicted") != e.get("true_label"))
    # snnTorch correct but SpiNNaker wrong
    snn_wins = sum(1 for e in entries
                   if not e.get("correct") and e.get("snn_predicted") == e.get("true_label"))
    # Both wrong
    both_wrong = sum(1 for e in entries
                     if not e.get("correct") and e.get("snn_predicted") != e.get("true_label"))
    # Both correct
    both_correct = sum(1 for e in entries
                       if e.get("correct") and e.get("snn_predicted") == e.get("true_label"))

    # Per-class analysis
    per_class_spinnaker = defaultdict(lambda: {"correct": 0, "total": 0})
    per_class_snn = defaultdict(lambda: {"correct": 0, "total": 0})

    for e in entries:
        cls = e.get("true_label")
        if cls is None:
            continue
        per_class_spinnaker[cls]["total"] += 1
        per_class_snn[cls]["total"] += 1
        if e.get("correct"):
            per_class_spinnaker[cls]["correct"] += 1
        if e.get("snn_predicted") == cls:
            per_class_snn[cls]["correct"] += 1

    # Class accuracies
    class_results = []
    for cls in sorted(per_class_spinnaker.keys()):
        name = ESC50_CLASSES[cls] if cls < len(ESC50_CLASSES) else f"class_{cls}"
        sp_acc = per_class_spinnaker[cls]["correct"] / max(per_class_spinnaker[cls]["total"], 1)
        sn_acc = per_class_snn[cls]["correct"] / max(per_class_snn[cls]["total"], 1)
        class_results.append({
            "class_idx": cls,
            "class_name": name,
            "spinnaker_acc": sp_acc,
            "snn_acc": sn_acc,
            "n": per_class_spinnaker[cls]["total"],
        })

    # Super-category breakdown (groups of 10 classes)
    categories = ["Animals (0-9)", "Nature (10-19)", "Human (20-29)", "Domestic (30-39)", "Urban (40-49)"]
    cat_results = []
    for i, cat in enumerate(categories):
        cat_classes = [r for r in class_results if 10 * i <= r["class_idx"] < 10 * (i + 1)]
        if cat_classes:
            sp_accs = [r["spinnaker_acc"] for r in cat_classes]
            sn_accs = [r["snn_acc"] for r in cat_classes]
            cat_results.append({
                "category": cat,
                "spinnaker_acc": np.mean(sp_accs),
                "snn_acc": np.mean(sn_accs),
            })

    return {
        "n_samples": n,
        "spinnaker_accuracy": spinnaker_correct / n,
        "snn_accuracy": snn_correct / n,
        "agreement_rate": agreement / n,
        "error_analysis": {
            "both_correct": both_correct,
            "spinnaker_only": spinnaker_wins,
            "snn_only": snn_wins,
            "both_wrong": both_wrong,
        },
        "per_class": class_results,
        "per_category": cat_results,
    }


def print_report(results):
    """Print formatted thesis §5.3 tables."""
    n = results["n_samples"]
    sp_acc = results["spinnaker_accuracy"]
    sn_acc = results["snn_accuracy"]
    agree = results["agreement_rate"]
    err = results["error_analysis"]

    print("\n" + "=" * 60)
    print("SpiNNaker Run 6: 400-Sample Fold-4 Inference Results")
    print("=" * 60)

    print(f"\n### §5.3.2 Full Inference Results (n={n})")
    print(f"| Metric                          | Value |")
    print(f"|----------------------------------|-------|")
    print(f"| SpiNNaker Accuracy              | {100*sp_acc:.1f}% |")
    print(f"| snnTorch Reference Accuracy     | {100*sn_acc:.1f}% |")
    print(f"| Hardware gap (snnTorch−SpiNN)   | {100*(sn_acc-sp_acc):.1f} pp |")
    print(f"| Agreement rate                  | {100*agree:.1f}% |")

    print(f"\n### §5.3.3 Error Analysis (n={n})")
    print(f"| Category                              | Count | % |")
    print(f"|---------------------------------------|-------|---|")
    print(f"| Both correct                          | {err['both_correct']} | {100*err['both_correct']/n:.1f}% |")
    print(f"| SpiNNaker correct, snnTorch wrong     | {err['spinnaker_only']} | {100*err['spinnaker_only']/n:.1f}% |")
    print(f"| snnTorch correct, SpiNNaker wrong     | {err['snn_only']} | {100*err['snn_only']/n:.1f}% |")
    print(f"| Both wrong                            | {err['both_wrong']} | {100*err['both_wrong']/n:.1f}% |")

    print(f"\n### Super-Category Breakdown")
    print(f"| Category       | SpiNNaker Acc | snnTorch Acc | Gap |")
    print(f"|----------------|--------------|-------------|-----|")
    for cat in results["per_category"]:
        gap = cat["snn_acc"] - cat["spinnaker_acc"]
        print(f"| {cat['category']:<16}| {100*cat['spinnaker_acc']:>12.1f}% | {100*cat['snn_acc']:>11.1f}% | {100*gap:>4.1f} pp |")

    print(f"\n### Top 10 Hardest Classes for SpiNNaker")
    sorted_classes = sorted(results["per_class"], key=lambda x: x["spinnaker_acc"])
    print(f"| Class | SpiNNaker Acc | snnTorch Acc | Gap |")
    print(f"|-------|-------------|-------------|-----|")
    for r in sorted_classes[:10]:
        gap = r["snn_acc"] - r["spinnaker_acc"]
        print(f"| {r['class_name']:<25} | {100*r['spinnaker_acc']:>12.1f}% | {100*r['snn_acc']:>11.1f}% | {100*gap:>4.1f} pp |")

    print(f"\n### Top 10 Easiest Classes for SpiNNaker")
    for r in reversed(sorted_classes[-10:]):
        gap = r["snn_acc"] - r["spinnaker_acc"]
        print(f"| {r['class_name']:<25} | {100*r['spinnaker_acc']:>12.1f}% | {100*r['snn_acc']:>11.1f}% | {100*gap:>4.1f} pp |")


def main():
    parser = argparse.ArgumentParser(description="Analyze SpiNNaker Run 6 results")
    parser.add_argument("--session-id", type=str, default=None,
                        help="Specific session ID to analyze (default: today's entries)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/spinnaker_results/run6_analysis.json)")
    args = parser.parse_args()

    print("Loading SpiNNaker Run 6 data...")
    entries = load_run6_data(args.session_id)
    print(f"Found {len(entries)} inference entries")

    if len(entries) == 0:
        print("No data found. Check that the run has started and session_id is correct.")
        return

    results = analyze(entries)
    print_report(results)

    # Save JSON
    out_path = args.output or str(RESULTS_DIR / "run6_analysis.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAnalysis saved to: {out_path}")

    # Quick thesis §5.3.2 summary for copy-paste
    print("\n### Thesis §5.3.2 Table Entry (for copy-paste):")
    print(f"| {len(entries)}/400 | {100*results['spinnaker_accuracy']:.1f}% | "
          f"snnTorch ref: {100*results['snn_accuracy']:.1f}%; "
          f"agreement {100*results['agreement_rate']:.0f}%; "
          f"hardware gap {100*(results['snn_accuracy']-results['spinnaker_accuracy']):.1f} pp |")


if __name__ == "__main__":
    main()
