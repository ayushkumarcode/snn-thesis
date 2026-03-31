#!/usr/bin/env python3
"""
Compute statistical tests and hardware energy estimates from SpiNNaker results.

TASK 1: Hardware energy for all 50 pruned SpiNNaker deployments
TASK 2: Paired t-tests for SpiNNaker vs snnTorch at each pruning level
TASK 3: 95% CIs on key 5-fold means
TASK 4: Cohen's d for key comparisons

Uses only standard library (json, math, os, glob).
"""

import json
import math
import os
import glob


# ============================================================
# Helper functions (no numpy/scipy needed)
# ============================================================

def mean(xs):
    return sum(xs) / len(xs)

def std_sample(xs):
    """Sample standard deviation (ddof=1)."""
    m = mean(xs)
    return math.sqrt(sum((x - m)**2 for x in xs) / (len(xs) - 1))

def pooled_sd(xs, ys):
    """Pooled standard deviation for two equal-size groups."""
    n = len(xs)
    sx = std_sample(xs)
    sy = std_sample(ys)
    return math.sqrt(((n - 1) * sx**2 + (n - 1) * sy**2) / (2 * n - 2))

def paired_t_test(xs, ys):
    """
    Paired t-test for two equal-length lists.
    Returns t-statistic, two-tailed p-value (approximate), and 95% CI on difference.
    """
    n = len(xs)
    diffs = [x - y for x, y in zip(xs, ys)]
    d_bar = mean(diffs)
    if n < 2:
        return 0.0, 1.0, (0.0, 0.0)
    sd_d = std_sample(diffs)
    se_d = sd_d / math.sqrt(n)
    if se_d == 0:
        return float('inf'), 0.0, (d_bar, d_bar)
    t_stat = d_bar / se_d
    df = n - 1
    # Approximate two-tailed p-value using Student's t CDF approximation
    p_value = two_tailed_p_from_t(abs(t_stat), df)
    # 95% CI
    t_crit = t_critical_95(df)
    ci_lo = d_bar - t_crit * se_d
    ci_hi = d_bar + t_crit * se_d
    return t_stat, p_value, (ci_lo, ci_hi)

def cohens_d_paired(xs, ys):
    """Cohen's d for paired samples: mean(diff) / sd(diff)."""
    diffs = [x - y for x, y in zip(xs, ys)]
    d_bar = mean(diffs)
    sd_d = std_sample(diffs)
    if sd_d == 0:
        return float('inf')
    return d_bar / sd_d

def cohens_d_independent(m1, s1, m2, s2, n=5):
    """Cohen's d for independent groups using pooled SD from means and SDs."""
    sp = math.sqrt(((n-1)*s1**2 + (n-1)*s2**2) / (2*n - 2))
    if sp == 0:
        return float('inf')
    return (m1 - m2) / sp

def t_critical_95(df):
    """
    Critical t-value for 95% CI (two-tailed, alpha=0.05).
    Lookup table for small df values.
    """
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021,
        50: 2.009, 100: 1.984, 1000: 1.962
    }
    if df in table:
        return table[df]
    # Linear interpolation for unlisted df
    keys = sorted(table.keys())
    for i in range(len(keys) - 1):
        if keys[i] <= df <= keys[i+1]:
            frac = (df - keys[i]) / (keys[i+1] - keys[i])
            return table[keys[i]] + frac * (table[keys[i+1]] - table[keys[i]])
    return 1.96  # fallback for very large df

def two_tailed_p_from_t(t_abs, df):
    """
    Approximate two-tailed p-value from |t| and df using the
    regularized incomplete beta function approximation.
    For n=5 (df=4), use lookup + interpolation.
    """
    # For small df, use a numerical approximation via the relationship:
    # p = I_{df/(df+t^2)}(df/2, 1/2)
    # We approximate using the Abramowitz and Stegun formula (26.7.4)
    # Simpler: use the relation p = 2 * (1 - t_cdf(|t|, df))
    # We implement t_cdf using the regularized incomplete beta function
    x = df / (df + t_abs**2)
    p = regularized_incomplete_beta(x, df / 2.0, 0.5)
    return p

def regularized_incomplete_beta(x, a, b, n_terms=200):
    """
    Regularized incomplete beta function I_x(a, b) using continued fraction.
    Uses Lentz's algorithm.
    """
    if x < 0 or x > 1:
        return 0.0
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    # Use the continued fraction representation
    # First compute ln(B(a,b)) using lgamma
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a

    # Use the continued fraction for I_x(a, b)
    # If x < (a+1)/(a+b+2), use direct; else use 1 - I_{1-x}(b, a)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_cf(x, a, b, n_terms)
    else:
        lbeta2 = math.lgamma(b) + math.lgamma(a) - math.lgamma(a + b)
        front2 = math.exp(b * math.log(1.0 - x) + a * math.log(x) - lbeta2) / b
        return 1.0 - front2 * _beta_cf(1.0 - x, b, a, n_terms)

def _beta_cf(x, a, b, n_terms):
    """Continued fraction for incomplete beta function."""
    TINY = 1e-30
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < TINY:
        d = TINY
    d = 1.0 / d
    f = d

    for m in range(1, n_terms + 1):
        # Even step
        numerator = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
        d = 1.0 + numerator * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + numerator / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        f *= c * d

        # Odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
        d = 1.0 + numerator * d
        if abs(d) < TINY:
            d = TINY
        c = 1.0 + numerator / c
        if abs(c) < TINY:
            c = TINY
        d = 1.0 / d
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < 1e-10:
            break

    return f


# ============================================================
# Paths
# ============================================================
BASE = "/Users/kumar/Documents/University/Year3/thesisproject/snn-esc50"
DEPLOY_DIR = os.path.join(BASE, "results/spinnaker_results/full_deploy_t3")
MASTER_PATH = os.path.join(BASE, "results/MASTER_RESULTS.json")
ENERGY_OUT = os.path.join(BASE, "results/energy/hardware_energy_pruned.json")
STATS_OUT = os.path.join(BASE, "results/energy/statistical_tests.json")

# Energy constants
NJ_PER_SYNAPTIC_EVENT_CONSERVATIVE = 8   # nJ
NJ_PER_SYNAPTIC_EVENT_LIBERAL = 20       # nJ
FC2_FAN_OUT = 50  # 256 hidden -> 50 output neurons


# ============================================================
# TASK 1: Hardware energy from per-sample spike counts
# ============================================================
def compute_hardware_energy():
    print("=" * 60)
    print("TASK 1: SpiNNaker Hardware Energy for Pruned Deployments")
    print("=" * 60)

    results = {}
    prune_levels = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

    for pct in prune_levels:
        fold_results = []
        for fold in range(1, 6):
            fname = f"fast_pruned{pct}_fold{fold}_400_N256.json"
            fpath = os.path.join(DEPLOY_DIR, fname)

            with open(fpath) as f:
                data = json.load(f)

            per_sample = data["per_sample"]
            n = len(per_sample)

            h_spikes = [s["h_total_spk"] for s in per_sample]
            o_spikes = [s["o_total_spk"] for s in per_sample]

            mean_h = mean(h_spikes)
            mean_o = mean(o_spikes)
            std_h = std_sample(h_spikes) if n > 1 else 0.0
            std_o = std_sample(o_spikes) if n > 1 else 0.0

            # Energy = (h_spikes * fan_out + o_spikes * 1) * nJ_per_event
            # Each hidden spike triggers 50 synaptic events (FC2: 256->50)
            # Each output spike triggers 1 event (readout)
            syn_events = mean_h * FC2_FAN_OUT + mean_o * 1
            energy_conservative = syn_events * NJ_PER_SYNAPTIC_EVENT_CONSERVATIVE
            energy_liberal = syn_events * NJ_PER_SYNAPTIC_EVENT_LIBERAL

            fold_result = {
                "prune_pct": pct,
                "fold": fold,
                "n_samples": n,
                "mean_h_total_spk": round(mean_h, 2),
                "std_h_total_spk": round(std_h, 2),
                "mean_o_total_spk": round(mean_o, 2),
                "std_o_total_spk": round(std_o, 2),
                "mean_synaptic_events": round(syn_events, 2),
                "energy_nj_8nJ": round(energy_conservative, 2),
                "energy_nj_20nJ": round(energy_liberal, 2),
                "spinnaker_accuracy": data["summary"]["spinnaker_accuracy"],
                "snntorch_accuracy": data["summary"]["snntorch_accuracy"]
            }
            fold_results.append(fold_result)

        # Aggregate across folds
        mean_energy_8 = mean([fr["energy_nj_8nJ"] for fr in fold_results])
        mean_energy_20 = mean([fr["energy_nj_20nJ"] for fr in fold_results])
        mean_syn = mean([fr["mean_synaptic_events"] for fr in fold_results])
        mean_h_all = mean([fr["mean_h_total_spk"] for fr in fold_results])
        mean_o_all = mean([fr["mean_o_total_spk"] for fr in fold_results])

        results[str(pct)] = {
            "per_fold": fold_results,
            "summary": {
                "prune_pct": pct,
                "mean_h_spk_across_folds": round(mean_h_all, 2),
                "mean_o_spk_across_folds": round(mean_o_all, 2),
                "mean_synaptic_events": round(mean_syn, 2),
                "mean_energy_nj_8nJ": round(mean_energy_8, 2),
                "std_energy_nj_8nJ": round(std_sample([fr["energy_nj_8nJ"] for fr in fold_results]), 2),
                "mean_energy_nj_20nJ": round(mean_energy_20, 2),
                "std_energy_nj_20nJ": round(std_sample([fr["energy_nj_20nJ"] for fr in fold_results]), 2),
                "mean_spinnaker_acc": round(mean([fr["spinnaker_accuracy"] for fr in fold_results]), 2),
                "mean_snntorch_acc": round(mean([fr["snntorch_accuracy"] for fr in fold_results]), 2)
            }
        }

        print(f"\nPruned {pct}%:")
        print(f"  Mean hidden spikes/sample: {mean_h_all:.1f}")
        print(f"  Mean output spikes/sample: {mean_o_all:.1f}")
        print(f"  Mean synaptic events: {mean_syn:.1f}")
        print(f"  Energy (8 nJ): {mean_energy_8:.1f} nJ")
        print(f"  Energy (20 nJ): {mean_energy_20:.1f} nJ")

    # Also compute for unpruned T=3 and T=1
    for tag, pattern in [("unpruned_t3", "fast_t3_fold{}_400_N256.json"),
                         ("unpruned_t1", "fast_t1_fold{}_400_N256.json")]:
        fold_results = []
        for fold in range(1, 6):
            fname = pattern.format(fold)
            fpath = os.path.join(DEPLOY_DIR, fname)

            with open(fpath) as f:
                data = json.load(f)

            per_sample = data["per_sample"]
            n = len(per_sample)

            h_spikes = [s["h_total_spk"] for s in per_sample]
            o_spikes = [s["o_total_spk"] for s in per_sample]

            mean_h = mean(h_spikes)
            mean_o = mean(o_spikes)
            std_h = std_sample(h_spikes) if n > 1 else 0.0
            std_o = std_sample(o_spikes) if n > 1 else 0.0

            syn_events = mean_h * FC2_FAN_OUT + mean_o * 1
            energy_conservative = syn_events * NJ_PER_SYNAPTIC_EVENT_CONSERVATIVE
            energy_liberal = syn_events * NJ_PER_SYNAPTIC_EVENT_LIBERAL

            fold_result = {
                "fold": fold,
                "n_samples": n,
                "mean_h_total_spk": round(mean_h, 2),
                "std_h_total_spk": round(std_h, 2),
                "mean_o_total_spk": round(mean_o, 2),
                "std_o_total_spk": round(std_o, 2),
                "mean_synaptic_events": round(syn_events, 2),
                "energy_nj_8nJ": round(energy_conservative, 2),
                "energy_nj_20nJ": round(energy_liberal, 2),
                "spinnaker_accuracy": data["summary"]["spinnaker_accuracy"],
                "snntorch_accuracy": data["summary"]["snntorch_accuracy"]
            }
            fold_results.append(fold_result)

        mean_energy_8 = mean([fr["energy_nj_8nJ"] for fr in fold_results])
        mean_energy_20 = mean([fr["energy_nj_20nJ"] for fr in fold_results])
        mean_syn = mean([fr["mean_synaptic_events"] for fr in fold_results])
        mean_h_all = mean([fr["mean_h_total_spk"] for fr in fold_results])
        mean_o_all = mean([fr["mean_o_total_spk"] for fr in fold_results])

        results[tag] = {
            "per_fold": fold_results,
            "summary": {
                "tag": tag,
                "mean_h_spk_across_folds": round(mean_h_all, 2),
                "mean_o_spk_across_folds": round(mean_o_all, 2),
                "mean_synaptic_events": round(mean_syn, 2),
                "mean_energy_nj_8nJ": round(mean_energy_8, 2),
                "std_energy_nj_8nJ": round(std_sample([fr["energy_nj_8nJ"] for fr in fold_results]), 2),
                "mean_energy_nj_20nJ": round(mean_energy_20, 2),
                "std_energy_nj_20nJ": round(std_sample([fr["energy_nj_20nJ"] for fr in fold_results]), 2),
                "mean_spinnaker_acc": round(mean([fr["spinnaker_accuracy"] for fr in fold_results]), 2),
                "mean_snntorch_acc": round(mean([fr["snntorch_accuracy"] for fr in fold_results]), 2)
            }
        }

        print(f"\n{tag}:")
        print(f"  Mean hidden spikes/sample: {mean_h_all:.1f}")
        print(f"  Mean output spikes/sample: {mean_o_all:.1f}")
        print(f"  Mean synaptic events: {mean_syn:.1f}")
        print(f"  Energy (8 nJ): {mean_energy_8:.1f} nJ")
        print(f"  Energy (20 nJ): {mean_energy_20:.1f} nJ")

    # Save
    os.makedirs(os.path.dirname(ENERGY_OUT), exist_ok=True)
    with open(ENERGY_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {ENERGY_OUT}")

    return results


# ============================================================
# TASK 2: Paired t-tests SpiNNaker vs snnTorch
# ============================================================
def compute_statistical_tests():
    print("\n" + "=" * 60)
    print("TASK 2: Paired t-tests (SpiNNaker vs snnTorch)")
    print("=" * 60)

    # Load MASTER_RESULTS for summary data
    with open(MASTER_PATH) as f:
        master = json.load(f)

    results = {}

    # --- Pruned levels: extract per-fold from individual JSONs ---
    prune_levels = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

    for pct in prune_levels:
        spinn_accs = []
        snnt_accs = []
