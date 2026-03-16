"""
Comprehensive statistical tests for ALL thesis claims.

Computes paired t-tests, Wilcoxon signed-rank tests, and Cohen's d
effect sizes for every comparison in the thesis. Saves to JSON.

Usage:
    python -m experiments.compute_all_statistics
"""

import sys
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import RESULTS_DIR


def cohens_d(x, y):
    """Compute Cohen's d for paired samples."""
    diff = np.array(x) - np.array(y)
    return float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0


def safe_wilcoxon(x, y):
    """Wilcoxon with fallback for tied/zero differences."""
    try:
        stat, p = wilcoxon(x, y)
        return float(stat), float(p)
    except ValueError:
        return float('nan'), float('nan')


def paired_test(x, y, label=""):
    """Run paired t-test + Wilcoxon + Cohen's d."""
    x, y = np.array(x), np.array(y)
    t_stat, t_p = ttest_rel(x, y)
    w_stat, w_p = safe_wilcoxon(x, y)
    d = cohens_d(x, y)
    result = {
        "label": label,
        "x_mean": float(x.mean()), "x_std": float(x.std()),
        "y_mean": float(y.mean()), "y_std": float(y.std()),
        "difference_mean": float((x - y).mean()),
        "t_statistic": float(t_stat), "t_p_value": float(t_p),
        "wilcoxon_statistic": float(w_stat), "wilcoxon_p_value": float(w_p),
        "cohens_d": d,
        "significant_005": bool(t_p < 0.05),
        "n": len(x),
    }
    sig = "YES" if t_p < 0.05 else "no"
    print(f"  {label}: diff={float((x-y).mean()):.4f}, t_p={t_p:.4f}, "
          f"w_p={w_p:.4f}, d={d:.3f}, sig={sig}")
    return result


def main():
    results = {}

    # ================================================================
    # 1. CORE: SNN direct vs ANN (5-fold)
    # ================================================================
    print("\n=== 1. Core: SNN direct vs ANN ===")
    snn_accs = []
    ann_accs = []
    for fold in range(1, 6):
        snn_path = RESULTS_DIR / "snn" / "direct" / f"result_fold{fold}.json"
        ann_path = RESULTS_DIR / "ann" / "none" / f"result_fold{fold}.json"
        if snn_path.exists() and ann_path.exists():
            with open(snn_path) as f:
                snn_accs.append(json.load(f)["best_acc"])
            with open(ann_path) as f:
                ann_accs.append(json.load(f)["best_acc"])
    if len(snn_accs) == 5:
        results["snn_vs_ann"] = paired_test(snn_accs, ann_accs, "SNN direct vs ANN")
    else:
        print(f"  Only {len(snn_accs)} folds found, skipping")

    # ================================================================
    # 2. Encoding pairwise comparisons
    # ================================================================
    print("\n=== 2. Encoding pairwise comparisons ===")
    encodings = ["direct", "rate", "phase", "population", "latency", "delta", "burst"]
    encoding_accs = {}
    for enc in encodings:
        accs = []
        for fold in range(1, 6):
            path = RESULTS_DIR / "snn" / enc / f"result_fold{fold}.json"
            if path.exists():
                with open(path) as f:
                    accs.append(json.load(f)["best_acc"])
        if len(accs) == 5:
            encoding_accs[enc] = accs

    results["encoding_pairwise"] = {}
    # Direct vs each other encoding
    if "direct" in encoding_accs:
        for enc in encodings[1:]:
            if enc in encoding_accs:
                key = f"direct_vs_{enc}"
                results["encoding_pairwise"][key] = paired_test(
                    encoding_accs["direct"], encoding_accs[enc],
                    f"direct vs {enc}"
                )

    # Rate vs phase (the near-tie)
    if "rate" in encoding_accs and "phase" in encoding_accs:
        results["rate_vs_phase"] = paired_test(
            encoding_accs["rate"], encoding_accs["phase"], "rate vs phase"
        )

    # ================================================================
    # 3. PANNs comparison
    # ================================================================
    print("\n=== 3. PANNs transfer learning ===")
    panns_path = RESULTS_DIR / "panns"
    panns_files = list(panns_path.glob("*all_folds*50ep*.json")) if panns_path.exists() else []
    if panns_files:
        with open(panns_files[0]) as f:
            panns = json.load(f)
        # Try different key patterns
        snn_p = panns.get("snn_fold_accuracies", panns.get("snn", {}).get("fold_accuracies", []))
        ann_p = panns.get("ann_fold_accuracies", panns.get("ann", {}).get("fold_accuracies", []))
        lin_p = panns.get("linear_fold_accuracies", panns.get("linear", {}).get("fold_accuracies", []))

        if not snn_p:
            # Try loading from per-fold results
            for key in panns:
                if "fold" in key.lower() and isinstance(panns[key], dict):
                    print(f"  Found key: {key}")

        if snn_p and ann_p:
            results["panns_snn_vs_ann"] = paired_test(
                [x/100 if max(snn_p) > 1 else x for x in snn_p],
                [x/100 if max(ann_p) > 1 else x for x in ann_p],
                "PANNs+SNN vs PANNs+ANN"
            )
        if snn_p and lin_p:
            results["panns_snn_vs_linear"] = paired_test(
                [x/100 if max(snn_p) > 1 else x for x in snn_p],
                [x/100 if max(lin_p) > 1 else x for x in lin_p],
                "PANNs+SNN vs PANNs+Linear"
            )

    # ================================================================
    # 4. Adversarial robustness (5-fold)
    # ================================================================
    print("\n=== 4. Adversarial robustness (5-fold) ===")
    adv_dir = RESULTS_DIR / "adversarial"
    if adv_dir.exists():
        epsilons = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
        adv_results = {}
        for ei, eps in enumerate(epsilons):
            snn_at_eps = []
            ann_at_eps = []
            for fold in range(1, 6):
                path = adv_dir / f"robustness_fold{fold}.json"
                if path.exists():
                    with open(path) as f:
                        data = json.load(f)
                    snn_at_eps.append(data["fgsm"]["snn"][ei])
                    ann_at_eps.append(data["fgsm"]["ann"][ei])
            if len(snn_at_eps) == 5:
                key = f"fgsm_eps_{eps}"
                adv_results[key] = paired_test(
                    snn_at_eps, ann_at_eps,
                    f"FGSM eps={eps}: SNN vs ANN"
                )
        results["adversarial_fgsm"] = adv_results

    # ================================================================
    # 5. Noise robustness (5-fold)
    # ================================================================
    print("\n=== 5. Noise robustness (5-fold) ===")
    noise_path = RESULTS_DIR / "noise_robustness"
    noise_files = list(noise_path.glob("*folds*.json")) if noise_path.exists() else []
    if noise_files:
        with open(noise_files[0]) as f:
            noise = json.load(f)
        snr_labels = noise.get("snr_labels", [])
        noise_results = {}
        for si, snr in enumerate(snr_labels):
            snn_folds = noise["snn"]["per_fold"].get(snr, [])
            ann_folds = noise["ann"]["per_fold"].get(snr, [])
            if len(snn_folds) == 5 and len(ann_folds) == 5:
                noise_results[snr] = paired_test(
                    snn_folds, ann_folds,
                    f"Noise {snr}: SNN vs ANN"
                )
        results["noise_robustness"] = noise_results

    # ================================================================
    # 6. SpiNNaker vs snnTorch (5-fold)
    # ================================================================
    print("\n=== 6. SpiNNaker vs snnTorch ===")
    spinn_path = RESULTS_DIR / "spinnaker_results" / "5fold_summary.json"
    if spinn_path.exists():
        with open(spinn_path) as f:
            spinn = json.load(f)
        spinn_accs = spinn.get("spinnaker_accuracies", spinn.get("per_fold_spinnaker", []))
        snntorch_accs = spinn.get("snntorch_accuracies", spinn.get("per_fold_snntorch", []))
        if spinn_accs and snntorch_accs:
            results["spinnaker_vs_snntorch"] = paired_test(
                [x/100 if max(spinn_accs) > 1 else x for x in spinn_accs],
                [x/100 if max(snntorch_accs) > 1 else x for x in snntorch_accs],
                "SpiNNaker vs snnTorch"
            )

    # ================================================================
    # 7. Temporal ablation T=20 vs T=25
    # ================================================================
    print("\n=== 7. Temporal ablation T=20 vs T=25 ===")
    ta_path = RESULTS_DIR / "snn" / "temporal_ablation" / "ablation_direct_5fold.json"
    if ta_path.exists():
        with open(ta_path) as f:
            ta = json.load(f)
        # Get per-fold values at T=20 and T=25
        t20_accs = []
        t25_accs = []
        for fold_key in [f"fold_{i}" for i in range(1, 6)]:
            if fold_key in ta:
                fold_data = ta[fold_key]
                if isinstance(fold_data, dict):
                    t20_accs.append(fold_data.get("20", fold_data.get(20, 0)))
                    t25_accs.append(fold_data.get("25", fold_data.get(25, 0)))
        if len(t20_accs) == 5:
            results["temporal_t20_vs_t25"] = paired_test(
                t20_accs, t25_accs, "T=20 vs T=25"
            )

    # ================================================================
    # Save
    # ================================================================
    save_dir = RESULTS_DIR / "statistical_tests"
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "comprehensive_tests.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"STATISTICAL TESTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Test':<40} {'p-value':>10} {'Cohen d':>10} {'Sig?':>6}")
    print(f"{'-'*70}")
    for key, val in results.items():
        if isinstance(val, dict) and "t_p_value" in val:
            sig = "YES" if val["significant_005"] else "no"
            print(f"{val['label']:<40} {val['t_p_value']:>10.4f} "
                  f"{val['cohens_d']:>10.3f} {sig:>6}")
        elif isinstance(val, dict):
            for subkey, subval in val.items():
                if isinstance(subval, dict) and "t_p_value" in subval:
                    sig = "YES" if subval["significant_005"] else "no"
                    print(f"  {subval['label']:<38} {subval['t_p_value']:>10.4f} "
                          f"{subval['cohens_d']:>10.3f} {sig:>6}")


if __name__ == "__main__":
    main()
