"""
debug_04_real_weights.py -- Real trained weights on tiny subnetwork.

WHAT THIS TESTS:
    Whether our actual trained snnTorch FC1 weights can drive any activity
    in sPyNNaker IF_curr_exp neurons. To avoid the UDP buffer overflow from
    Run 3 (451K connections), we use only the first 20 hidden neurons instead
    of all 256. All 2304 input neurons are used, so the weight magnitudes are
    unchanged.

WHAT IT PROVES/DISPROVES:
    - PASS (any hidden neuron fires): Our trained weights ARE strong enough,
      at least for some neurons. The problem with run_on_spinnaker must be:
      (a) combined effect of all 256 neurons flooding the board, or (b) only
      some neurons fire and the accuracy was low for other reasons.
    - FAIL (all 20 neurons silent with max_v < 0.01): Trained weights are
      definitively too small. Need weight scaling. Proceed to debug_05.
    - FAIL (all 20 neurons silent, max_v > 0.01 but < 1.0): Current IS
      accumulating but not reaching threshold. Weight scale of 1/max_v_seen
      would be required to fire. Proceed to debug_05 for calibration.

WHY 20 NEURONS:
    20 hidden neurons x 2304 inputs = 46,080 connections.
    256 hidden neurons x 2304 inputs = 589,824 connections (caused buffer overflow).
    46,080 connections fits comfortably within UDP transfer limits.

WEIGHT STATISTICS:
    The script prints detailed statistics for the first 20 hidden neurons'
    incoming connections before running, so you can see weight magnitudes
    without even running the simulation.

USAGE:
    source .venv-spinnaker/bin/activate
    python spinnaker/debug_04_real_weights.py

    # Override weights directory if needed:
    python spinnaker/debug_04_real_weights.py --weights-dir /path/to/weights
"""

import argparse
import sys
import time
from pathlib import Path

SCRIPT_START = time.time()
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] debug_04_real_weights.py -- starting")
print("=" * 70)
print("TEST: With real trained weights, does any voltage accumulate in IF_curr_exp?")
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
parser = argparse.ArgumentParser(description="debug_04: real weights on 20 hidden neurons")
parser.add_argument("--weights-dir", type=str,
                    default=str(Path(__file__).parent.parent / "results" / "spinnaker_weights"),
                    help="Directory containing fc1_connections.npy and test_spike_features.npy")
parser.add_argument("--n-hidden", type=int, default=20,
                    help="Number of hidden neurons to use (default: 20, max: 256)")
parser.add_argument("--sample-idx", type=int, default=0,
                    help="Which test sample to use (default: 0)")
args = parser.parse_args()

WEIGHTS_DIR = Path(args.weights_dir)
N_HIDDEN_SUBSET = min(args.n_hidden, 256)
SAMPLE_IDX = args.sample_idx

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
print(f"  Hidden neurons     : first {N_HIDDEN_SUBSET} of 256 (to avoid buffer overflow)")
print(f"  Input neurons      : all 2304 (full FC1 input size)")
print(f"  Test sample index  : {SAMPLE_IDX}")
print(f"  LIF params         : {LIF_PARAMS}")
print(f"  Weights directory  : {WEIGHTS_DIR}")

# ============================================================
# Load weights and data
# ============================================================
print(f"\nLoading weights and test data...")

fc1_conn_path = WEIGHTS_DIR / "fc1_connections.npy"
spike_feat_path = WEIGHTS_DIR / "test_spike_features.npy"

if not fc1_conn_path.exists():
    print(f"FATAL: fc1_connections.npy not found at {fc1_conn_path}")
    print("Run convert_weights.py first.")
    sys.exit(1)

if not spike_feat_path.exists():
    print(f"FATAL: test_spike_features.npy not found at {spike_feat_path}")
    print("Run extract_features.py first.")
    sys.exit(1)

fc1_all = np.load(fc1_conn_path)       # (N_conns, 4): [pre, post, weight, delay]
spike_features = np.load(spike_feat_path)  # (N_samples, num_steps, 2304)

print(f"  fc1_connections shape: {fc1_all.shape}")
print(f"  test_spike_features shape: {spike_features.shape}")

if SAMPLE_IDX >= spike_features.shape[0]:
    print(f"FATAL: sample_idx={SAMPLE_IDX} >= {spike_features.shape[0]} available samples")
    sys.exit(1)

# --- Filter: only connections to first N_HIDDEN_SUBSET hidden neurons ---
# Column 1 is the post-synaptic (hidden) neuron index
mask = fc1_all[:, 1] < N_HIDDEN_SUBSET
fc1_subset = fc1_all[mask]

print(f"\nConnection filtering:")
print(f"  Total FC1 connections   : {len(fc1_all)}")
print(f"  Connections to neurons  : 0..{N_HIDDEN_SUBSET-1} only")
print(f"  Retained connections    : {len(fc1_subset)}")
print(f"  Dropped connections     : {len(fc1_all) - len(fc1_subset)}")

if len(fc1_subset) == 0:
    print("FATAL: No connections found for the selected hidden neurons.")
    sys.exit(1)

# --- Weight statistics for selected subset ---
weights_subset = fc1_subset[:, 2]
exc_weights = weights_subset[weights_subset > 0]
inh_weights = weights_subset[weights_subset < 0]

print(f"\nWeight statistics for first {N_HIDDEN_SUBSET} hidden neurons:")
print(f"  All weights  : min={weights_subset.min():.5f}, max={weights_subset.max():.5f}, "
      f"mean={weights_subset.mean():.5f}, std={weights_subset.std():.5f}")
print(f"  Excitatory   : n={len(exc_weights)}, min={exc_weights.min() if len(exc_weights)>0 else 0:.5f}, "
      f"max={exc_weights.max() if len(exc_weights)>0 else 0:.5f}, "
      f"mean={exc_weights.mean() if len(exc_weights)>0 else 0:.5f}")
print(f"  Inhibitory   : n={len(inh_weights)}, min={inh_weights.min() if len(inh_weights)>0 else 0:.5f}, "
      f"max={inh_weights.max() if len(inh_weights)>0 else 0:.5f}, "
      f"mean={inh_weights.mean() if len(inh_weights)>0 else 0:.5f}")

# Percentiles
percentiles = [5, 25, 50, 75, 95]
pct_vals = np.percentile(np.abs(weights_subset), percentiles)
print(f"  |weight| percentiles:")
for p, v in zip(percentiles, pct_vals):
    print(f"    p{p:2d}: {v:.5f}")

# Per-hidden-neuron connection stats
print(f"\nPer-hidden-neuron connection counts (first {N_HIDDEN_SUBSET}):")
for hidden_idx in range(N_HIDDEN_SUBSET):
    neuron_conns = fc1_subset[fc1_subset[:, 1] == hidden_idx]
    n_exc = (neuron_conns[:, 2] > 0).sum()
    n_inh = (neuron_conns[:, 2] < 0).sum()
    w_mean = neuron_conns[:, 2].mean() if len(neuron_conns) > 0 else 0.0
    print(f"  hidden[{hidden_idx:3d}]: {len(neuron_conns):5d} conns "
          f"({n_exc} exc, {n_inh} inh), mean_w={w_mean:.5f}")

# --- Prepare spike input for sample 0 ---
spike_input = spike_features[SAMPLE_IDX]   # (25, 2304)
INPUT_SIZE = spike_input.shape[1]

spike_times_list = []
for neuron_idx in range(INPUT_SIZE):
    times = np.where(spike_input[:, neuron_idx] > 0.5)[0].astype(float).tolist()
    spike_times_list.append(times)

active_inputs = sum(1 for t in spike_times_list if len(t) > 0)
total_input_spikes = sum(len(t) for t in spike_times_list)

spikes_per_step = np.sum(spike_input > 0.5, axis=1)

print(f"\nInput spike statistics (sample {SAMPLE_IDX}):")
print(f"  Active input neurons    : {active_inputs} / {INPUT_SIZE}")
print(f"  Total input spikes      : {total_input_spikes}")
print(f"  Spikes per timestep     : min={spikes_per_step.min()}, "
      f"max={spikes_per_step.max()}, mean={spikes_per_step.mean():.1f}")

# Rough expected current per neuron per active timestep
# Each input spike delivers weight * exp(-dt/tau_syn) * (1/(cm*dt)) charge
import math
current_per_spike = exc_weights.mean() if len(exc_weights) > 0 else 0.0
fraction_delivered = 1.0 - math.exp(-DT / LIF_PARAMS["tau_syn_E"])
expected_current = current_per_spike * active_inputs / N_HIDDEN_SUBSET * fraction_delivered
print(f"\n  Rough expected current per hidden neuron per step:")
print(f"    mean_exc_weight={current_per_spike:.5f}, "
      f"tau_syn_E decay={fraction_delivered:.3f},")
print(f"    ~{active_inputs/N_HIDDEN_SUBSET:.0f} active inputs per neuron")
print(f"    -> expected I ~ {expected_current:.5f} (threshold = {LIF_PARAMS['v_thresh']})")

# --- Separate exc/inh ---
fc1_exc = fc1_subset[fc1_subset[:, 2] > 0].tolist()
fc1_inh_data = fc1_subset[fc1_subset[:, 2] < 0].copy()
fc1_inh_data[:, 2] = np.abs(fc1_inh_data[:, 2])
fc1_inh = fc1_inh_data.tolist()

print(f"\nConnection lists:")
print(f"  FC1 excitatory: {len(fc1_exc)}")
print(f"  FC1 inhibitory: {len(fc1_inh)}")


def _handle_sim_exception(exc: Exception) -> None:
    exc_type = type(exc).__name__
    exc_str = str(exc)
    print(f"\nEXCEPTION: {exc_type}: {exc_str}")
    if "SpinnmanIOException" in exc_type or "No buffer space" in exc_str:
        print("MEANING: UDP buffer overflow despite using only 20 neurons.")
        print("WHAT TO TRY: Reduce --n-hidden further (e.g., --n-hidden 5).")
        print("             Or apply weight pruning: only keep |w| > 0.1.")
    elif "SpinnmanTimeoutException" in exc_type or "timeout" in exc_str.lower():
        print("MEANING: SpiNNaker board timed out.")
        print("WHAT TO TRY: Check board status and network connectivity.")
    else:
        print(f"MEANING: Unexpected error: {exc_str}")


# ============================================================
# Run simulation
# ============================================================
print(f"\nSetting up SpiNNaker simulation...")
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
    sim.Projection(
        input_pop, hidden_pop,
        sim.FromListConnector(fc1_exc),
        receptor_type="excitatory"
    )
if fc1_inh:
    sim.Projection(
        input_pop, hidden_pop,
        sim.FromListConnector(fc1_inh),
        receptor_type="inhibitory"
    )

print(f"Running {NUM_STEPS} ms simulation...")
t0 = time.time()
try:
    sim.run(NUM_STEPS)
except Exception as exc:
    _handle_sim_exception(exc)
    sim.end()
    SCRIPT_END = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] debug_04_real_weights.py -- ABORTED")
    print(f"Total wall clock time: {SCRIPT_END - SCRIPT_START:.1f} s")
    sys.exit(1)
elapsed = time.time() - t0
print(f"Simulation complete in {elapsed*1000:.0f} ms wall clock.")

# ============================================================
# Extract and analyse results
# ============================================================
spike_data = hidden_pop.get_data("spikes")
v_data = hidden_pop.get_data("v")

# Per-neuron spikes
n_spikes_per_neuron = np.zeros(N_HIDDEN_SUBSET, dtype=int)
spike_times_per_neuron = [[] for _ in range(N_HIDDEN_SUBSET)]
for st in spike_data.segments[-1].spiketrains:
    nid = int(st.annotations.get("source_index", 0))
    if nid < N_HIDDEN_SUBSET:
        n_spikes_per_neuron[nid] = len(st)
        spike_times_per_neuron[nid] = [float(t) for t in st]

# Per-neuron max voltage
max_v_per_neuron = np.zeros(N_HIDDEN_SUBSET)
v_traces = {}
for sig in v_data.segments[-1].analogsignals:
    for n in range(min(sig.shape[1], N_HIDDEN_SUBSET)):
        trace = [float(sig[t, n]) for t in range(sig.shape[0])]
        v_traces[n] = trace
        if trace:
            max_v_per_neuron[n] = max(trace)

sim.end()

total_hidden_spikes = n_spikes_per_neuron.sum()

# ============================================================
# Print per-neuron report
# ============================================================
print(f"\nPer-neuron results (first {N_HIDDEN_SUBSET} hidden neurons):")
print(f"  {'NeuronID':>9}  {'Spikes':>7}  {'MaxV':>10}  {'SpikeTimes'}")
for n in range(N_HIDDEN_SUBSET):
    fired_marker = "  <-- FIRED" if n_spikes_per_neuron[n] > 0 else ""
    print(f"  hidden[{n:3d}]: {n_spikes_per_neuron[n]:7d}  "
          f"{max_v_per_neuron[n]:10.6f}  "
          f"{spike_times_per_neuron[n][:8]}{fired_marker}")

print(f"\nVoltage statistics across {N_HIDDEN_SUBSET} neurons:")
print(f"  Global max voltage : {max_v_per_neuron.max():.6f}")
print(f"  Global mean max v  : {max_v_per_neuron.mean():.6f}")
print(f"  Global min max v   : {max_v_per_neuron.min():.6f}")
print(f"  Neurons that fired : {(n_spikes_per_neuron > 0).sum()} / {N_HIDDEN_SUBSET}")
print(f"  Total hidden spikes: {total_hidden_spikes}")

# ============================================================
# Summary
# ============================================================
print("\n")
print("=" * 70)
print("SUMMARY -- debug_04_real_weights.py")
print("=" * 70)

any_fired = total_hidden_spikes > 0

if any_fired:
    print(f"  ==> === PASS ===: {total_hidden_spikes} spikes fired "
          f"from {(n_spikes_per_neuron > 0).sum()} hidden neurons")
    firing_neurons = np.where(n_spikes_per_neuron > 0)[0]
    print(f"      Firing neurons: {list(firing_neurons)}")
    print()
    print("  CONCLUSION: Trained weights ARE sufficient to drive some hidden neurons.")
    print("  -> The problem in run_on_spinnaker.py was likely the buffer overflow,")
    print("     not the weights themselves. Try run_on_spinnaker.py --max-hidden 20")
    print("     or increase pruning threshold.")
else:
    print(f"  ==> === FAIL ===: 0 hidden spikes. All {N_HIDDEN_SUBSET} neurons silent.")
    gmax = max_v_per_neuron.max()
    print(f"  Max voltage seen : {gmax:.6f} (threshold = {LIF_PARAMS['v_thresh']})")
    if gmax > 0.001:
        scale_needed = LIF_PARAMS["v_thresh"] / gmax
        print(f"  Voltage DID accumulate but not enough.")
        print(f"  Estimated weight scale needed: ~{scale_needed:.1f}x")
        print(f"  -> Run debug_05_weight_scale.py for calibration.")
    else:
        print(f"  Voltage near zero. Current not reaching membrane.")
        print(f"  -> tau_syn_E=1.0 may be killing the current.")
        print(f"     Run debug_02_tau_syn.py and/or debug_05_weight_scale.py.")

SCRIPT_END = time.time()
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] debug_04_real_weights.py -- finished")
print(f"Total wall clock time: {SCRIPT_END - SCRIPT_START:.1f} s")
