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
