"""
reduce_inputs.py -- Reduce input dimensionality to avoid SpiNNaker spike flooding.

Problem:
    The full 2304-neuron SpikeSourceArray sends up to 1398 simultaneous spikes
    per timestep, exceeding SpiNNaker's multicast router capacity and causing
    dropped packets. Hidden neurons never receive enough current to fire.

Solution:
    Keep only the top-K most influential input neurons (by total weight
    contribution to the hidden layer). Re-index the FC1 connections and
    spike feature array to match.

Usage:
    source .venv-spinnaker/bin/activate
    python spinnaker/reduce_inputs.py               # top_k=256 (default)
    python spinnaker/reduce_inputs.py --top-k 128
    python spinnaker/reduce_inputs.py --top-k 512 --criterion activity

Outputs (saved to results/spinnaker_weights/):
    reduced_spike_features_K{k}.npy    -- (N_samples, 25, K)
    reduced_fc1_connections_K{k}.npy   -- (M, 4) with re-indexed pre-synaptic
    reduced_metadata_K{k}.json         -- documents what was done
"""

import argparse
import json
from pathlib import Path

import numpy as np

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
WEIGHTS_DIR = REPO_ROOT / "results" / "spinnaker_weights"

# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser(
    description="Reduce SpiNNaker input dimensionality to avoid spike flooding"
)
parser.add_argument(
    "--top-k", type=int, default=256,
    help="Keep the top K input neurons (default: 256)"
)
parser.add_argument(
    "--criterion", choices=["weight", "activity", "combined"], default="weight",
    help=(
        "How to rank input neurons for selection:\n"
        "  weight   -- sum of |FC1 weights| to hidden layer (most important)\n"
        "  activity -- total spikes in first test sample\n"
        "  combined -- product of both scores (default: weight)"
    )
)
parser.add_argument(
    "--sample-for-activity", type=int, default=0,
    help="Which test sample to use for activity-based selection (default: 0)"
)
args = parser.parse_args()

TOP_K = args.top_k
CRITERION = args.criterion
SAMPLE_IDX = args.sample_for_activity

print(f"reduce_inputs.py")
print(f"  top_k     : {TOP_K}")
print(f"  criterion : {CRITERION}")
print(f"  sample    : {SAMPLE_IDX}")
print()

# ============================================================
# Load data
# ============================================================
fc1_path = WEIGHTS_DIR / "fc1_connections.npy"
spike_path = WEIGHTS_DIR / "test_spike_features.npy"
metadata_path = WEIGHTS_DIR / "metadata.json"

for p in [fc1_path, spike_path, metadata_path]:
    if not p.exists():
        print(f"FATAL: {p} not found. Run convert_weights.py and extract_features.py first.")
        raise SystemExit(1)

fc1_all = np.load(fc1_path)               # (N_conn, 4): [pre, post, weight, delay]
spike_features = np.load(spike_path)      # (N_samples, 25, 2304)
with open(metadata_path) as f:
    metadata = json.load(f)

N_SAMPLES, N_STEPS, N_INPUTS = spike_features.shape
print(f"Loaded:")
print(f"  FC1 connections : {fc1_all.shape[0]:,} connections")
print(f"  Spike features  : {N_SAMPLES} samples × {N_STEPS} steps × {N_INPUTS} inputs")
print(f"  FC1 weight range: [{fc1_all[:, 2].min():.5f}, {fc1_all[:, 2].max():.5f}]")
print(f"  FC1 weight mean : {fc1_all[:, 2].mean():.6f}")
print()

# ============================================================
# Compute per-input-neuron importance scores
# ============================================================

# Weight criterion: for each input neuron i, sum |w_{i,j}| over all hidden j
weight_scores = np.zeros(N_INPUTS)
for i in range(N_INPUTS):
    mask = fc1_all[:, 0] == i
    if mask.any():
        weight_scores[i] = np.abs(fc1_all[mask, 2]).sum()

# Activity criterion: total spikes across all timesteps for first test sample
activity_scores = spike_features[SAMPLE_IDX].sum(axis=0).astype(float)  # (N_INPUTS,)

print(f"Weight score stats  : min={weight_scores.min():.5f}  "
      f"max={weight_scores.max():.5f}  "
      f"mean={weight_scores.mean():.5f}  "
      f"nonzero={( weight_scores > 0).sum()}")
print(f"Activity score stats: min={activity_scores.min():.0f}  "
      f"max={activity_scores.max():.0f}  "
      f"mean={activity_scores.mean():.2f}  "
      f"nonzero={(activity_scores > 0).sum()}")

# Combined criterion: normalise both, take product
if CRITERION == "weight":
    scores = weight_scores
elif CRITERION == "activity":
    scores = activity_scores
else:  # combined
    norm_w = weight_scores / (weight_scores.max() + 1e-8)
    norm_a = activity_scores / (activity_scores.max() + 1e-8)
    scores = norm_w * norm_a

print(f"\nUsing criterion   : {CRITERION}")
print(f"Score range       : [{scores.min():.5f}, {scores.max():.5f}]")

# Select top K
top_k_indices = np.argsort(scores)[-TOP_K:]
top_k_indices = np.sort(top_k_indices)  # keep sorted for reproducibility

n_nonzero_selected = (scores[top_k_indices] > 0).sum()
print(f"\nSelected top-{TOP_K} neurons:")
print(f"  Indices range  : [{top_k_indices.min()}, {top_k_indices.max()}]")
print(f"  Non-zero score : {n_nonzero_selected}/{TOP_K}")
print(f"  Score range    : [{scores[top_k_indices].min():.5f}, "
      f"{scores[top_k_indices].max():.5f}]")

# Estimate new spike counts
reduced_activity = spike_features[SAMPLE_IDX][:, top_k_indices]   # (25, K)
max_spikes_per_step = int((reduced_activity > 0.5).sum(axis=1).max())
total_spikes = int((reduced_activity > 0.5).sum())
print(f"\nEstimated spike stats after reduction (sample {SAMPLE_IDX}):")
print(f"  Max spikes/timestep : {max_spikes_per_step}  "
      f"(was up to 1398 with full 2304)")
print(f"  Total spikes        : {total_spikes}")

# ============================================================
# Build index remapping: old_idx -> new_idx in [0, K-1]
# ============================================================
old_to_new = {int(old): new for new, old in enumerate(top_k_indices)}

# ============================================================
# Reduce FC1 connections
# ============================================================
mask_pre = np.isin(fc1_all[:, 0].astype(int), top_k_indices)
fc1_reduced_raw = fc1_all[mask_pre].copy()

# Remap pre-synaptic indices to new [0, K-1] range
new_pre_indices = np.array([old_to_new[int(i)] for i in fc1_reduced_raw[:, 0]])
fc1_reduced_raw[:, 0] = new_pre_indices

print(f"\nFC1 connection reduction:")
print(f"  Original   : {fc1_all.shape[0]:,} connections ({N_INPUTS} inputs)")
print(f"  Reduced    : {fc1_reduced_raw.shape[0]:,} connections ({TOP_K} inputs)")
print(f"  Retention  : {fc1_reduced_raw.shape[0] / fc1_all.shape[0] * 100:.1f}%")
print(f"  Weight range: [{fc1_reduced_raw[:, 2].min():.5f}, "
      f"{fc1_reduced_raw[:, 2].max():.5f}]")

# ============================================================
# Reduce spike features
# ============================================================
reduced_features = spike_features[:, :, top_k_indices]   # (N_samples, 25, K)

print(f"\nSpike feature reduction:")
print(f"  Original shape : {spike_features.shape}")
print(f"  Reduced shape  : {reduced_features.shape}")

# Verify per-step spike distribution
all_per_step = []
for s in range(N_SAMPLES):
    per_step = (reduced_features[s] > 0.5).sum(axis=1)
    all_per_step.append(per_step.max())
print(f"  Max spikes/step (across all samples): {max(all_per_step)}")
print(f"  Avg max spikes/step                 : {np.mean(all_per_step):.1f}")

# ============================================================
# Save outputs
# ============================================================
out_feats = WEIGHTS_DIR / f"reduced_spike_features_K{TOP_K}.npy"
out_fc1 = WEIGHTS_DIR / f"reduced_fc1_connections_K{TOP_K}.npy"
out_meta = WEIGHTS_DIR / f"reduced_metadata_K{TOP_K}.json"

np.save(out_feats, reduced_features)
np.save(out_fc1, fc1_reduced_raw)

meta_out = {
    "reduction": {
        "top_k": TOP_K,
        "criterion": CRITERION,
        "sample_for_activity": SAMPLE_IDX,
    },
    "original": {
        "n_inputs": N_INPUTS,
        "fc1_connections": int(fc1_all.shape[0]),
        "spike_features_shape": list(spike_features.shape),
    },
    "reduced": {
        "n_inputs": TOP_K,
        "fc1_connections": int(fc1_reduced_raw.shape[0]),
        "spike_features_shape": list(reduced_features.shape),
    },
    "top_k_original_indices": top_k_indices.tolist(),
    "max_spikes_per_step_sample0": int(max_spikes_per_step),
    "architecture": metadata.get("architecture", {}),
    "snn_params": metadata.get("snn_params", {}),
}
with open(out_meta, "w") as f:
    json.dump(meta_out, f, indent=2)

print(f"\nSaved:")
print(f"  {out_feats}")
print(f"  {out_fc1}")
print(f"  {out_meta}")
print()
print(f"To use with auto_calibrate.py, add a --reduced-k {TOP_K} flag, OR")
print(f"temporarily rename files for testing:")
print(f"  cp {WEIGHTS_DIR}/fc1_connections.npy {WEIGHTS_DIR}/fc1_connections_full.npy")
print(f"  cp {out_fc1} {WEIGHTS_DIR}/fc1_connections.npy")
print(f"  cp {WEIGHTS_DIR}/test_spike_features.npy {WEIGHTS_DIR}/test_spike_features_full.npy")
print(f"  cp {out_feats} {WEIGHTS_DIR}/test_spike_features.npy")
print(f"  python spinnaker/auto_calibrate.py --skip-to-phase 3")
print(f"  # After done, restore originals:")
print(f"  cp {WEIGHTS_DIR}/fc1_connections_full.npy {WEIGHTS_DIR}/fc1_connections.npy")
print(f"  cp {WEIGHTS_DIR}/test_spike_features_full.npy {WEIGHTS_DIR}/test_spike_features.npy")
