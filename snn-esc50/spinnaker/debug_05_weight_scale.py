"""
debug_05_weight_scale.py -- Calibrate weight scaling factor.

WHAT THIS TESTS:
    Our trained snnTorch weights are typically small (e.g., mean |w| ~ 0.05-0.2).
    sPyNNaker's IF_curr_exp integrates current with synaptic decay, which means
    more charge is needed to reach threshold than in snnTorch's discrete-time
    Leaky neuron. This script sweeps a weight multiplier to find the minimum
    scaling that causes at least one hidden neuron to fire.

WHAT IT PROVES/DISPROVES:
    - FIRST_WORKING_SCALE = 1: No scaling needed. Weights are fine as-is.
      The problem is structural, not weight magnitude. Check debug_01-03.
    - FIRST_WORKING_SCALE = 2-10: Moderate scaling needed. Use
      run_on_spinnaker.py --weight-scale N with the found value.
    - FIRST_WORKING_SCALE = 50+: Very large scaling needed. Something is
      fundamentally mismatched. Consider using IF_curr_delta instead
      (run_on_spinnaker.py --neuron-model IF_curr_delta).
    - NO WORKING SCALE (all fail): Even 50x scaling doesn't fire. The
      problem is not weight magnitude. Run debug_02_tau_syn.py and also
      try IF_curr_delta neuron model.

SETUP:
    Same as debug_04: first N_HIDDEN_SUBSET (20) hidden neurons, sample 0.
    Connection count stays the same; only the weight values are multiplied.

OUTPUT:
    For each scale factor: spike count and which neurons fired.
    FIRST WORKING SCALE printed prominently.

USAGE:
    source .venv-spinnaker/bin/activate
    python spinnaker/debug_05_weight_scale.py

    # Custom scale sweep:
    python spinnaker/debug_05_weight_scale.py --scales 1 5 10 20 50 100
"""

import argparse
import sys
import time
from pathlib import Path

SCRIPT_START = time.time()
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] debug_05_weight_scale.py -- starting")
print("=" * 70)
print("TEST: What weight multiplier makes our trained weights work on SpiNNaker?")
print("=" * 70)

try:
    import numpy as np
except ImportError:
    print("FATAL: numpy not available")
    sys.exit(1)

try:
    import pyNN.spiNNaker as sim
except ImportError as exc:
    print(f"\nFATAL: Cannot import pyNN.spiNNaker: {exc}")
    print("Make sure you activated the spinnaker venv:")
    print("  source .venv-spinnaker/bin/activate")
    sys.exit(1)

# ============================================================
# Arguments
# ============================================================
parser = argparse.ArgumentParser(description="debug_05: weight scaling calibration")
parser.add_argument("--weights-dir", type=str,
                    default=str(Path(__file__).parent.parent / "results" / "spinnaker_weights"),
                    help="Directory containing fc1_connections.npy and test_spike_features.npy")
parser.add_argument("--n-hidden", type=int, default=20,
                    help="Number of hidden neurons (default: 20)")
parser.add_argument("--sample-idx", type=int, default=0,
                    help="Test sample index (default: 0)")
parser.add_argument("--scales", type=float, nargs="+",
                    default=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
                    help="Weight scale factors to try (default: 1 2 5 10 20 50)")
args = parser.parse_args()

WEIGHTS_DIR = Path(args.weights_dir)
N_HIDDEN_SUBSET = min(args.n_hidden, 256)
SAMPLE_IDX = args.sample_idx
SCALE_VALUES = sorted(args.scales)

# ============================================================
# Parameters
# ============================================================
NUM_STEPS = 25
DT = 1.0

LIF_PARAMS = {
    "cm": 1.0,
    "tau_m": 20.0,
    "tau_syn_E": 1.0,
    "tau_syn_I": 1.0,
    "tau_refrac": 0.1,
    "v_reset": 0.0,
    "v_rest": 0.0,
    "v_thresh": 1.0,
}

print(f"\nConfiguration:")
print(f"  SpiNNaker server   : spinnaker.cs.man.ac.uk")
print(f"  Simulation time    : {NUM_STEPS} ms (timestep={DT} ms)")
print(f"  Hidden neurons     : first {N_HIDDEN_SUBSET} of 256")
print(f"  Test sample index  : {SAMPLE_IDX}")
print(f"  Weight scales      : {SCALE_VALUES}")
print(f"  LIF params         : {LIF_PARAMS}")
print(f"  Weights directory  : {WEIGHTS_DIR}")

# ============================================================
# Load weights and test data (once, shared across all scale tests)
# ============================================================
print(f"\nLoading data...")

fc1_conn_path = WEIGHTS_DIR / "fc1_connections.npy"
spike_feat_path = WEIGHTS_DIR / "test_spike_features.npy"

if not fc1_conn_path.exists():
    print(f"FATAL: fc1_connections.npy not found at {fc1_conn_path}")
    sys.exit(1)
if not spike_feat_path.exists():
    print(f"FATAL: test_spike_features.npy not found at {spike_feat_path}")
    sys.exit(1)

fc1_all = np.load(fc1_conn_path)
spike_features = np.load(spike_feat_path)

if SAMPLE_IDX >= spike_features.shape[0]:
    print(f"FATAL: sample_idx={SAMPLE_IDX} out of range (max={spike_features.shape[0]-1})")
    sys.exit(1)

# Filter to first N_HIDDEN_SUBSET hidden neurons
mask = fc1_all[:, 1] < N_HIDDEN_SUBSET
fc1_subset = fc1_all[mask].copy()   # (N_conns, 4): [pre, post, weight, delay]

print(f"  fc1_connections: {len(fc1_all)} total, {len(fc1_subset)} retained (post < {N_HIDDEN_SUBSET})")

if len(fc1_subset) == 0:
    print(f"FATAL: No connections for neurons 0..{N_HIDDEN_SUBSET-1}")
    sys.exit(1)

# Baseline weight statistics (before scaling)
raw_weights = fc1_subset[:, 2]
print(f"\nBaseline weight statistics (scale=1.0):")
print(f"  min={raw_weights.min():.5f}, max={raw_weights.max():.5f}, "
      f"mean={raw_weights.mean():.5f}, std={raw_weights.std():.5f}")
print(f"  exc count: {(raw_weights > 0).sum()}, inh count: {(raw_weights < 0).sum()}")

# Spike input setup
spike_input = spike_features[SAMPLE_IDX]   # (25, 2304)
INPUT_SIZE = spike_input.shape[1]

spike_times_list = []
for neuron_idx in range(INPUT_SIZE):
    times = np.where(spike_input[:, neuron_idx] > 0.5)[0].astype(float).tolist()
    spike_times_list.append(times)

total_input_spikes = sum(len(t) for t in spike_times_list)
active_inputs = sum(1 for t in spike_times_list if len(t) > 0)

print(f"\nInput statistics (sample {SAMPLE_IDX}):")
print(f"  Active inputs: {active_inputs}/{INPUT_SIZE}")
print(f"  Total spikes : {total_input_spikes}")


def _handle_sim_exception(exc: Exception) -> None:
    exc_type = type(exc).__name__
    exc_str = str(exc)
    print(f"  EXCEPTION: {exc_type}: {exc_str}")
    if "No buffer space" in exc_str or "SpinnmanIOException" in exc_type:
        print("  MEANING: UDP buffer overflow. Try --n-hidden 5 or --n-hidden 10.")
    elif "timeout" in exc_str.lower() or "SpinnmanTimeoutException" in exc_type:
        print("  MEANING: Board timeout. Check connectivity.")
    else:
        print("  MEANING: Unknown error.")


def run_with_scale(scale_factor: float) -> dict:
    """Run the simulation with all FC1 weights multiplied by scale_factor.

    Args:
        scale_factor: Weight multiplier.

    Returns:
        Dict with: scale, n_hidden_spikes, firing_neurons, max_v, error
    """
    print(f"\n{'─' * 70}")
    print(f"  scale_factor = {scale_factor:.1f}x  "
          f"(effective weight range: [{raw_weights.min()*scale_factor:.4f}, "
          f"{raw_weights.max()*scale_factor:.4f}])")
    print(f"{'─' * 70}")

    # Apply scale
    scaled_conns = fc1_subset.copy()
    scaled_conns[:, 2] = scaled_conns[:, 2] * scale_factor

    # Separate exc/inh
    fc1_exc = scaled_conns[scaled_conns[:, 2] > 0].tolist()
    fc1_inh_data = scaled_conns[scaled_conns[:, 2] < 0].copy()
    fc1_inh_data[:, 2] = np.abs(fc1_inh_data[:, 2])
    fc1_inh = fc1_inh_data.tolist()

    print(f"  Scaled exc conns: {len(fc1_exc)}, inh conns: {len(fc1_inh)}")

    sim.setup(timestep=DT)

    input_pop = sim.Population(
        INPUT_SIZE, sim.SpikeSourceArray,
        {"spike_times": spike_times_list},
        label="input"
    )

    hidden_pop = sim.Population(
        N_HIDDEN_SUBSET, sim.IF_curr_exp(**LIF_PARAMS),
        label="hidden"
    )
    hidden_pop.initialize(v=0.0)
    hidden_pop.record(["spikes", "v"])

    if fc1_exc:
        sim.Projection(input_pop, hidden_pop,
                       sim.FromListConnector(fc1_exc),
                       receptor_type="excitatory")
    if fc1_inh:
        sim.Projection(input_pop, hidden_pop,
                       sim.FromListConnector(fc1_inh),
                       receptor_type="inhibitory")

    print(f"  Running simulation...")
    t0 = time.time()
    try:
        sim.run(NUM_STEPS)
    except Exception as exc:
        _handle_sim_exception(exc)
        sim.end()
        return {
            "scale": scale_factor, "n_hidden_spikes": -1,
            "firing_neurons": [], "max_v": float("nan"), "error": str(exc)
        }
    elapsed = time.time() - t0

    # Extract data
    spike_data = hidden_pop.get_data("spikes")
    v_data = hidden_pop.get_data("v")

    n_spikes_per_neuron = np.zeros(N_HIDDEN_SUBSET, dtype=int)
    for st in spike_data.segments[-1].spiketrains:
        nid = int(st.annotations.get("source_index", 0))
        if nid < N_HIDDEN_SUBSET:
            n_spikes_per_neuron[nid] = len(st)

    max_v_per_neuron = np.zeros(N_HIDDEN_SUBSET)
    for sig in v_data.segments[-1].analogsignals:
        for n in range(min(sig.shape[1], N_HIDDEN_SUBSET)):
            trace = [float(sig[t, n]) for t in range(sig.shape[0])]
            if trace:
                max_v_per_neuron[n] = max(trace)

    sim.end()

    total_spikes = int(n_spikes_per_neuron.sum())
    firing = list(np.where(n_spikes_per_neuron > 0)[0])
    global_max_v = float(max_v_per_neuron.max())

    print(f"  Wall clock: {elapsed*1000:.0f} ms")
    print(f"  Total hidden spikes: {total_spikes}")
    print(f"  Firing neurons: {firing}")
    print(f"  Max voltage across all neurons: {global_max_v:.6f}")

    return {
        "scale": scale_factor,
        "n_hidden_spikes": total_spikes,
        "firing_neurons": firing,
        "max_v": global_max_v,
    }


# ============================================================
# Run scale sweep
# ============================================================
results = []
first_working_scale = None

for scale in SCALE_VALUES:
    r = run_with_scale(scale)
    results.append(r)
    if r["n_hidden_spikes"] > 0 and first_working_scale is None:
        first_working_scale = scale
        print(f"\n  *** FIRST WORKING SCALE: {scale} ***")
        print(f"  You can stop the sweep here if desired.")

# ============================================================
# Summary
# ============================================================
print("\n")
print("=" * 70)
print("SUMMARY -- debug_05_weight_scale.py")
print("=" * 70)
print(f"  {'Scale':>8}  {'Hidden Spikes':>14}  {'Firing Neurons':>20}  {'Max V':>10}  Status")
print(f"  {'─'*8}  {'─'*14}  {'─'*20}  {'─'*10}  {'─'*11}")
for r in results:
    if "error" in r:
        status = "ERROR"
        label = f"[{r['error'][:40]}]"
    elif r["n_hidden_spikes"] > 0:
        status = "=== PASS ==="
        label = str(r["firing_neurons"])
    else:
        status = "=== FAIL ==="
        label = "[]"
    print(f"  {r['scale']:>8.1f}x  {r['n_hidden_spikes']:>14}  {label:>20}  "
          f"{r['max_v']:>10.5f}  {status}")

print()
if first_working_scale is not None:
    print(f"  FIRST WORKING SCALE: {first_working_scale}x")
    print(f"\n  RECOMMENDED ACTION:")
    print(f"    python spinnaker/run_on_spinnaker.py --weight-scale {first_working_scale}")
    print(f"    python spinnaker/run_on_spinnaker.py --weight-scale {first_working_scale} --max-hidden 20")
    if first_working_scale >= 20:
        print(f"\n  NOTE: Scale of {first_working_scale}x is large.")
        print(f"  Consider also trying IF_curr_delta (no synaptic decay):")
        print(f"    python spinnaker/run_on_spinnaker.py --neuron-model IF_curr_delta")
else:
    print("  CONCLUSION: NO scale factor in the tested range caused any firing.")
    print("  Weight magnitude alone cannot explain the silence.")
    print("  Recommended next steps (in order):")
    print("    1. Verify debug_01_can_fire.py passes at weight=5.0")
    print("    2. Run debug_02_tau_syn.py with tau_syn_E=0.1")
    print("    3. Try: run_on_spinnaker.py --neuron-model IF_curr_delta")
    print("    4. Try: run_on_spinnaker.py --weight-scale 50 --tau-syn 0.1 --max-hidden 20")

SCRIPT_END = time.time()
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] debug_05_weight_scale.py -- finished")
print(f"Total wall clock time: {SCRIPT_END - SCRIPT_START:.1f} s")
