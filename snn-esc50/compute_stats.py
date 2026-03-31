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
