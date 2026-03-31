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
