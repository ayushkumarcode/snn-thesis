"""
run_on_spinnaker.py -- ESC-50 SNN inference on SpiNNaker hardware (v2, debug edition).

Deployment strategy (FC-only / hybrid):
    - Conv features extracted offline on CPU (test_spike_features.npy)
    - FC classifier (2304 -> 256 -> 50 LIF neurons) runs on SpiNNaker
    - Excitatory and inhibitory weights separated per sPyNNaker requirement

Critical parameter mapping notes:
    - Population.initialize(v=0.0) is required to match snnTorch's init
    - tau_syn_E/I = 1.0 ms by default; use --tau-syn to override
    - Weight pruning at --prune-threshold (default 0.05) limits connections
    - --weight-scale multiplies all weights to compensate for tau_syn mismatch
    - --neuron-model selects between IF_curr_exp (with synaptic decay) and
      IF_curr_delta (no decay, closer to snnTorch snn.Leaky behaviour)
    - --max-hidden limits to first N hidden neurons (reduces UDP load)

Usage:
    source .venv-spinnaker/bin/activate
    python spinnaker/run_on_spinnaker.py [options]

    # Example: compensate for tau_syn mismatch with weight scaling
    python spinnaker/run_on_spinnaker.py --weight-scale 10 --tau-syn 0.1

    # Example: use IF_curr_delta (no synaptic decay)
    python spinnaker/run_on_spinnaker.py --neuron-model IF_curr_delta

    # Example: debug mode with small network
    python spinnaker/run_on_spinnaker.py --max-hidden 20 --weight-scale 5 --num-samples 3

Debugging workflow (if 0 hidden spikes):
    1. python spinnaker/debug_01_can_fire.py   -- can the neuron model fire?
    2. python spinnaker/debug_02_tau_syn.py    -- is tau_syn killing signal?
    3. python spinnaker/debug_03_two_layer.py  -- does signal propagate?
    4. python spinnaker/debug_04_real_weights.py -- do real weights fire?
    5. python spinnaker/debug_05_weight_scale.py -- calibrate scaling factor
    See spinnaker/README_DEBUGGING.md for full decision tree.
"""

import os
import argparse
import json
import time
from pathlib import Path

# Enable provenance data collection
os.environ["SPYNNAKER_DISABLE_PROVENANCE"] = "0"

import numpy as np

# ============================================================
# Parse arguments
# ============================================================
parser = argparse.ArgumentParser(
    description="Run ESC-50 SNN inference on SpiNNaker hardware"
)
parser.add_argument("--num-samples", type=int, default=10,
                    help="Number of test samples to run (default: 10)")
parser.add_argument("--prune-threshold", type=float, default=0.05,
                    help="Prune weights smaller than this absolute value (default: 0.05)")
parser.add_argument("--weight-scale", type=float, default=1.0,
                    help="Multiply all weights by this factor before simulation. "
                         "Use to compensate for tau_syn mismatch between snnTorch "
                         "and sPyNNaker. Find calibration value with debug_05_weight_scale.py.")
parser.add_argument("--tau-syn", type=float, default=1.0,
                    help="Synaptic time constant tau_syn_E/I for IF_curr_exp in ms. "
                         "Lower = faster current injection = closer to snnTorch behaviour. "
                         "Use 0.1 for near-instantaneous, 1.0 for default. "
                         "(Only applies to IF_curr_exp model.)")
parser.add_argument("--neuron-model", type=str, default="IF_curr_exp",
                    choices=["IF_curr_exp", "IF_curr_delta"],
                    help="Neuron model. IF_curr_delta has no synaptic decay -- "
                         "closest to snnTorch snn.Leaky behaviour. (default: IF_curr_exp)")
parser.add_argument("--max-hidden", type=int, default=256,
                    help="Only use first N hidden neurons out of 256. "
                         "Reduces connections for debugging or to avoid UDP overflow. "
                         "Use 20 to start debugging. (default: 256)")
args = parser.parse_args()

# ============================================================
# Import sPyNNaker after argument parsing
# ============================================================
import pyNN.spiNNaker as sim

# ============================================================
# Script start timestamp
# ============================================================
SCRIPT_START = time.time()
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] run_on_spinnaker.py -- starting")
print("=" * 70)
print("SpiNNaker ESC-50 SNN Inference (v2, debug edition)")
print("=" * 70)

# ============================================================
# Configuration
# ============================================================
WEIGHTS_DIR = Path(__file__).parent.parent / "results" / "spinnaker_weights"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "spinnaker_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHT_PRUNE_THRESHOLD = args.prune_threshold
NUM_TEST_SAMPLES = args.num_samples
WEIGHT_SCALE = args.weight_scale
TAU_SYN = args.tau_syn
NEURON_MODEL = args.neuron_model
MAX_HIDDEN = args.max_hidden

print(f"\nRun configuration:")
print(f"  SpiNNaker server   : spinnaker.cs.man.ac.uk (from ~/.spynnaker.cfg)")
print(f"  Num samples        : {NUM_TEST_SAMPLES}")
print(f"  Prune threshold    : {WEIGHT_PRUNE_THRESHOLD}")
print(f"  Weight scale       : {WEIGHT_SCALE}x")
print(f"  Tau_syn            : {TAU_SYN} ms  (IF_curr_exp only)")
print(f"  Neuron model       : {NEURON_MODEL}")
print(f"  Max hidden neurons : {MAX_HIDDEN} (of 256)")
print(f"  Provenance env     : SPYNNAKER_DISABLE_PROVENANCE={os.environ.get('SPYNNAKER_DISABLE_PROVENANCE', 'not set')}")

# ============================================================
# Load weights and data
# ============================================================
print(f"\nLoading weights and test data...")
fc1_connections_raw = np.load(WEIGHTS_DIR / "fc1_connections.npy")
fc2_connections_raw = np.load(WEIGHTS_DIR / "fc2_connections.npy")
test_features = np.load(WEIGHTS_DIR / "test_spike_features.npy")
test_labels = np.load(WEIGHTS_DIR / "test_labels.npy")

# Limit samples
test_features = test_features[:NUM_TEST_SAMPLES]
test_labels = test_labels[:NUM_TEST_SAMPLES]

with open(WEIGHTS_DIR / "metadata.json") as f:
    metadata = json.load(f)

INPUT_SIZE = metadata["architecture"]["flatten_size"]    # 2304
HIDDEN_SIZE = min(MAX_HIDDEN, metadata["architecture"]["fc1"]["out"])   # ≤256
OUTPUT_SIZE = metadata["architecture"]["fc2"]["out"]     # 50
NUM_STEPS = metadata["snn_params"]["num_steps"]          # 25
NUM_SAMPLES = len(test_labels)

print(f"\nNetwork architecture:")
print(f"  Input neurons    : {INPUT_SIZE}")
print(f"  Hidden neurons   : {HIDDEN_SIZE} (of {metadata['architecture']['fc1']['out']})")
print(f"  Output neurons   : {OUTPUT_SIZE}")
print(f"  Simulation steps : {NUM_STEPS} ms")
print(f"  Test samples     : {NUM_SAMPLES}")

# ============================================================
# Prune weights
# ============================================================
fc1_pruned = fc1_connections_raw[np.abs(fc1_connections_raw[:, 2]) > WEIGHT_PRUNE_THRESHOLD]
fc2_pruned = fc2_connections_raw[np.abs(fc2_connections_raw[:, 2]) > WEIGHT_PRUNE_THRESHOLD]

# If MAX_HIDDEN < 256, filter FC1 to only connections targeting first MAX_HIDDEN hidden neurons
if HIDDEN_SIZE < 256:
    fc1_pruned = fc1_pruned[fc1_pruned[:, 1] < HIDDEN_SIZE]
    # FC2: only connections from first MAX_HIDDEN hidden neurons
    fc2_pruned = fc2_pruned[fc2_pruned[:, 0] < HIDDEN_SIZE]

print(f"\nWeight pruning (threshold={WEIGHT_PRUNE_THRESHOLD}):")
print(f"  FC1: {len(fc1_connections_raw)} -> {len(fc1_pruned)} "
      f"({100*len(fc1_pruned)/max(1,len(fc1_connections_raw)):.1f}% retained)")
print(f"  FC2: {len(fc2_connections_raw)} -> {len(fc2_pruned)} "
      f"({100*len(fc2_pruned)/max(1,len(fc2_connections_raw)):.1f}% retained)")

# ============================================================
# Apply weight scaling
# ============================================================
fc1_scaled = fc1_pruned.copy()
fc2_scaled = fc2_pruned.copy()
if WEIGHT_SCALE != 1.0:
    fc1_scaled[:, 2] = fc1_scaled[:, 2] * WEIGHT_SCALE
    fc2_scaled[:, 2] = fc2_scaled[:, 2] * WEIGHT_SCALE
    print(f"\nWeight scaling applied: {WEIGHT_SCALE}x")

# ============================================================
# Detailed weight statistics
# ============================================================
def print_weight_stats(connections: np.ndarray, label: str) -> None:
    """Print comprehensive weight statistics for a connection array."""
    weights = connections[:, 2]
    exc = weights[weights > 0]
    inh = weights[weights < 0]
    pcts = np.percentile(weights, [5, 25, 50, 75, 95]) if len(weights) > 0 else [0]*5

    print(f"\n  {label} weight statistics ({len(connections)} connections):")
    print(f"    All   : min={weights.min():.5f}, max={weights.max():.5f}, "
          f"mean={weights.mean():.5f}, std={weights.std():.5f}")
    print(f"    Exc({len(exc):6d}): min={exc.min() if len(exc)>0 else 0:.5f}, "
          f"max={exc.max() if len(exc)>0 else 0:.5f}, "
          f"mean={exc.mean() if len(exc)>0 else 0:.5f}")
    print(f"    Inh({len(inh):6d}): min={inh.min() if len(inh)>0 else 0:.5f}, "
          f"max={inh.max() if len(inh)>0 else 0:.5f}, "
          f"mean={inh.mean() if len(inh)>0 else 0:.5f}")
    print(f"    Percentiles (p5/p25/p50/p75/p95):")
    print(f"      {pcts[0]:.5f} / {pcts[1]:.5f} / {pcts[2]:.5f} / "
          f"{pcts[3]:.5f} / {pcts[4]:.5f}")


print("\nPre-simulation weight analysis:")
print_weight_stats(fc1_scaled, "FC1 (scaled)")
print_weight_stats(fc2_scaled, "FC2 (scaled)")

# Per-hidden-neuron connection count analysis
fc1_post_indices = fc1_scaled[:, 1].astype(int)
conns_per_hidden = np.bincount(fc1_post_indices, minlength=HIDDEN_SIZE)
zero_conn_count = (conns_per_hidden == 0).sum()

print(f"\n  FC1 connections per hidden neuron:")
print(f"    Mean: {conns_per_hidden.mean():.1f}")
print(f"    Min : {conns_per_hidden.min()}")
print(f"    Max : {conns_per_hidden.max()}")
print(f"    Hidden neurons with ZERO incoming connections: {zero_conn_count}/{HIDDEN_SIZE}")
if zero_conn_count > 0:
    dead_neurons = np.where(conns_per_hidden == 0)[0]
    print(f"    Dead neurons: {list(dead_neurons[:10])}{'...' if len(dead_neurons)>10 else ''}")

# ============================================================
# Separate excitatory and inhibitory connections
# ============================================================
print("\nPreparing excitatory/inhibitory split...")

fc1_exc = fc1_scaled[fc1_scaled[:, 2] > 0].tolist()
fc1_inh_data = fc1_scaled[fc1_scaled[:, 2] < 0].copy()
fc1_inh_data[:, 2] = np.abs(fc1_inh_data[:, 2])
fc1_inh = fc1_inh_data.tolist()

fc2_exc = fc2_scaled[fc2_scaled[:, 2] > 0].tolist()
fc2_inh_data = fc2_scaled[fc2_scaled[:, 2] < 0].copy()
fc2_inh_data[:, 2] = np.abs(fc2_inh_data[:, 2])
fc2_inh = fc2_inh_data.tolist()

total_conns = len(fc1_exc) + len(fc1_inh) + len(fc2_exc) + len(fc2_inh)
print(f"  FC1: {len(fc1_exc)} exc + {len(fc1_inh)} inh = {len(fc1_exc)+len(fc1_inh)}")
print(f"  FC2: {len(fc2_exc)} exc + {len(fc2_inh)} inh = {len(fc2_exc)+len(fc2_inh)}")
print(f"  Total connections: {total_conns}")

# ============================================================
# LIF parameters
# ============================================================
# snnTorch snn.Leaky: mem[t+1] = beta * mem[t] + I, threshold=1.0, reset=0
# PyNN IF_curr_exp:   V[t+1] = V[t]*(1-dt/tau_m) + I_syn*dt/cm
#   with tau_m=20, cm=1, dt=1: V[t+1] = 0.95*V + I_syn
#   matches beta=0.95 when I_syn = instantaneous current
# PyNN IF_curr_delta: V[t+1] = V[t]*(1-dt/tau_m) + weight*spike (no decay)
#   even closer to snnTorch: current delivered instantaneously

lif_params = {
    "cm": 1.0,
    "tau_m": 20.0,
    "tau_refrac": 0.1,    # minimal refractory (snnTorch has none)
    "v_reset": 0.0,
    "v_rest": 0.0,
    "v_thresh": 1.0,
}
if NEURON_MODEL == "IF_curr_exp":
    lif_params["tau_syn_E"] = TAU_SYN
    lif_params["tau_syn_I"] = TAU_SYN

print(f"\nNeuron model: {NEURON_MODEL}")
print(f"LIF params  : {lif_params}")
if NEURON_MODEL == "IF_curr_exp":
    import math
    frac = 1.0 - math.exp(-1.0 / TAU_SYN)
    print(f"  tau_syn_E={TAU_SYN} ms -> {frac*100:.1f}% of current delivered per timestep")
    print(f"  (snnTorch delivers 100% instantaneously)")
else:
    print(f"  IF_curr_delta: no synaptic decay, current delivered instantaneously")
    print(f"  This is the closest approximation to snnTorch snn.Leaky")

# ============================================================
# Run inference
# ============================================================
print(f"\n{'─' * 70}")
print(f"Running inference on {NUM_SAMPLES} samples")
print(f"{'─' * 70}")

results = []
correct = 0
total_sim_time = 0.0


def _handle_sim_exception(exc: Exception, sample_idx: int) -> None:
    """Print plain-English explanation for common SpiNNaker exceptions."""
    exc_type = type(exc).__name__
    exc_str = str(exc)
    print(f"\n  [EXCEPTION] Sample {sample_idx}: {exc_type}")
    print(f"  Message: {exc_str}")
    if "No buffer space" in exc_str or "SpinnmanIOException" in exc_type:
        print("  MEANING: UDP send buffer overflow. Too many connections to transfer at once.")
        print("  WHAT TO TRY:")
        print("    1. Increase --prune-threshold (e.g., 0.1 or 0.2) to keep fewer connections")
        print("    2. Reduce --max-hidden (e.g., --max-hidden 64 or --max-hidden 20)")
        print("    3. Reduce --num-samples to 1 to isolate whether it's per-sample or setup")
        print("       The current run has {total_conns} connections total.")
    elif "SpinnmanTimeoutException" in exc_type or "timeout" in exc_str.lower():
        print("  MEANING: SpiNNaker board did not respond within the timeout window.")
        print("  WHAT TO TRY:")
        print("    1. Check VPN connection to Manchester network")
        print("    2. ping spinnaker.cs.man.ac.uk")
        print("    3. Check spalloc queue for existing jobs: use spalloc client")
        print("    4. Retry -- timeouts can be transient")
    elif "ConnectionRefused" in exc_str or "ECONNREFUSED" in exc_str:
        print("  MEANING: Cannot connect to SpiNNaker spalloc server.")
        print("  WHAT TO TRY: Ensure Manchester VPN is active.")
    else:
        print("  MEANING: Unexpected error. Check SpiNNaker board status.")


for sample_idx in range(NUM_SAMPLES):
    spike_input = test_features[sample_idx]  # (25, 2304)
    true_label = int(test_labels[sample_idx])

    print(f"\n{'─' * 50}")
    print(f"  Sample {sample_idx+1}/{NUM_SAMPLES}  (true label={true_label})")
    print(f"{'─' * 50}")

    # Prepare spike times
    spike_times_list = []
    for neuron_idx in range(INPUT_SIZE):
        times = np.where(spike_input[:, neuron_idx] > 0.5)[0].astype(float).tolist()
        spike_times_list.append(times)

    active_inputs = sum(1 for t in spike_times_list if len(t) > 0)
    total_input_spikes = sum(len(t) for t in spike_times_list)
    spikes_per_step = np.sum(spike_input > 0.5, axis=1)

    print(f"  Input spike stats:")
    print(f"    Active input neurons: {active_inputs}/{INPUT_SIZE}")
    print(f"    Total input spikes  : {total_input_spikes}")
    print(f"    Spikes per timestep : min={spikes_per_step.min()} "
          f"max={spikes_per_step.max()} mean={spikes_per_step.mean():.1f}")
    print(f"    Max simultaneous    : {spikes_per_step.max()} spikes at step "
          f"{int(np.argmax(spikes_per_step))}")
    if spikes_per_step.max() > 800:
        print(f"    WARNING: {spikes_per_step.max()} simultaneous spikes may stress UDP buffer")

    # --- Setup ---
    sim.setup(timestep=1.0)

    # --- Populations ---
    input_pop = sim.Population(
        INPUT_SIZE, sim.SpikeSourceArray,
        {"spike_times": spike_times_list}, label="input")
    input_pop.record("spikes")   # Record to verify SpikeSourceArray fired correctly

    hidden_pop = sim.Population(
        HIDDEN_SIZE,
        getattr(sim, NEURON_MODEL)(**lif_params),
        label="hidden")
    hidden_pop.initialize(v=0.0)
    hidden_pop.record(["spikes", "v"])   # Record both spikes and voltage

    output_pop = sim.Population(
        OUTPUT_SIZE,
        getattr(sim, NEURON_MODEL)(**lif_params),
        label="output")
    output_pop.initialize(v=0.0)
    output_pop.record(["spikes", "v"])

    # --- Projections ---
    if fc1_exc:
        sim.Projection(input_pop, hidden_pop,
                       sim.FromListConnector(fc1_exc),
                       receptor_type="excitatory")
    if fc1_inh:
        sim.Projection(input_pop, hidden_pop,
                       sim.FromListConnector(fc1_inh),
                       receptor_type="inhibitory")
    if fc2_exc:
        sim.Projection(hidden_pop, output_pop,
                       sim.FromListConnector(fc2_exc),
                       receptor_type="excitatory")
    if fc2_inh:
        sim.Projection(hidden_pop, output_pop,
                       sim.FromListConnector(fc2_inh),
                       receptor_type="inhibitory")

    # --- Run ---
    print(f"  Running {NUM_STEPS} ms simulation...")
    start = time.time()
    try:
        sim.run(NUM_STEPS)
    except Exception as exc:
        _handle_sim_exception(exc, sample_idx)
        sim.end()
        results.append({
            "sample": sample_idx,
            "true_label": true_label,
            "predicted": -1,
            "correct": False,
            "error": str(exc),
            "output_spikes": 0,
            "hidden_spikes": 0,
            "input_spikes": total_input_spikes,
        })
        continue
    elapsed = time.time() - start
    total_sim_time += elapsed
    print(f"  Simulation complete in {elapsed*1000:.0f} ms wall clock.")

    # ============================================================
    # Extract all recorded data
    # ============================================================

    # --- Input spike verification ---
    in_spike_data = input_pop.get_data("spikes")
    in_spikes_per_step = np.zeros(NUM_STEPS, dtype=int)
    in_total_actual = 0
    for st in in_spike_data.segments[-1].spiketrains:
        for t in st:
            step = int(round(float(t)))
            if 0 <= step < NUM_STEPS:
                in_spikes_per_step[step] += 1
        in_total_actual += len(st)

    print(f"\n  Input spike verification:")
    print(f"    Expected input spikes: {total_input_spikes}")
    print(f"    Actual input spikes  : {in_total_actual}")
    if in_total_actual == 0:
        print(f"    WARNING: SpikeSourceArray sent 0 spikes! Check spike_times_list construction.")
    elif abs(in_total_actual - total_input_spikes) > total_input_spikes * 0.05:
        print(f"    WARNING: Input spike count differs from expected by "
              f"{abs(in_total_actual - total_input_spikes)}")
    print(f"    Per-timestep (input): {list(in_spikes_per_step)}")

    # --- Hidden layer spikes ---
    hidden_spike_data = hidden_pop.get_data("spikes")
    hidden_spikes_per_step = np.zeros(NUM_STEPS, dtype=int)
    hidden_spikes_per_neuron = np.zeros(HIDDEN_SIZE, dtype=int)
    hidden_spike_times_per_neuron = [[] for _ in range(HIDDEN_SIZE)]

    for st in hidden_spike_data.segments[-1].spiketrains:
        nid = int(st.annotations.get("source_index", 0))
        for t in st:
            step = int(round(float(t)))
            if 0 <= step < NUM_STEPS:
                hidden_spikes_per_step[step] += 1
        if nid < HIDDEN_SIZE:
            hidden_spikes_per_neuron[nid] = len(st)
            hidden_spike_times_per_neuron[nid] = [float(t) for t in st]

    hidden_spike_count = int(hidden_spikes_per_neuron.sum())

    # --- Hidden layer voltage ---
    hidden_v_data = hidden_pop.get_data("v")
    hidden_v_traces = {}   # neuron_id -> list of voltage values over time
    for sig in hidden_v_data.segments[-1].analogsignals:
        for n in range(min(sig.shape[1], HIDDEN_SIZE)):
            hidden_v_traces[n] = sig.magnitude[:, n].tolist()

    # Find top 5 most-active hidden neurons (by spike count, break ties by max_v)
    max_v_per_hidden = np.array([
        max(hidden_v_traces.get(n, [0.0])) if hidden_v_traces.get(n) else 0.0
        for n in range(HIDDEN_SIZE)
    ])
    top5_hidden = np.argsort(
        hidden_spikes_per_neuron * 1000 + max_v_per_hidden
    )[-5:][::-1]

    # --- Output layer spikes ---
    output_data = output_pop.get_data("spikes")
    spike_counts = np.zeros(OUTPUT_SIZE)
    output_spikes_per_step = np.zeros(NUM_STEPS, dtype=int)
    output_spike_times_per_neuron = [[] for _ in range(OUTPUT_SIZE)]

    for spiketrain in output_data.segments[-1].spiketrains:
        neuron_id = int(spiketrain.annotations.get("source_index", 0))
        spike_counts[neuron_id] = len(spiketrain)
        output_spike_times_per_neuron[neuron_id] = [float(t) for t in spiketrain]
        for t in spiketrain:
            step = int(round(float(t)))
            if 0 <= step < NUM_STEPS:
                output_spikes_per_step[step] += 1

    # --- Output layer voltage ---
    output_v_data = output_pop.get_data("v")
    final_v = np.zeros(OUTPUT_SIZE)
    output_v_all_steps = {}  # neuron_id -> list
    for sig in output_v_data.segments[-1].analogsignals:
        if len(sig) > 0:
            for n in range(min(sig.shape[1], OUTPUT_SIZE)):
                output_v_all_steps[n] = sig.magnitude[:, n].tolist()
            # final_v = last timestep
            last_row = sig[-1].magnitude.flatten()
            final_v[:min(OUTPUT_SIZE, len(last_row))] = last_row[:OUTPUT_SIZE]

    sim.end()

    # ============================================================
    # Print detailed per-sample diagnostics
    # ============================================================

    print(f"\n  Hidden layer activity:")
    print(f"    Total hidden spikes    : {hidden_spike_count}")
    print(f"    Neurons that fired     : {(hidden_spikes_per_neuron > 0).sum()}/{HIDDEN_SIZE}")
    print(f"    Max spikes/neuron      : {hidden_spikes_per_neuron.max()}")
    print(f"    Max voltage (all hid)  : {max_v_per_hidden.max():.6f}")
    print(f"    Mean max voltage       : {max_v_per_hidden.mean():.6f}")
    print(f"    Per-timestep (hidden)  : {list(hidden_spikes_per_step)}")

    print(f"\n  Top 5 most-active hidden neurons (by spike count then max_v):")
    for nid in top5_hidden:
        v_trace = hidden_v_traces.get(int(nid), [])
        print(f"    hidden[{nid:3d}]: {hidden_spikes_per_neuron[nid]:3d} spikes, "
              f"max_v={max_v_per_hidden[nid]:.6f}, "
              f"spike_times={hidden_spike_times_per_neuron[nid][:6]}")
        if v_trace:
            print(f"      V trace: {[f'{v:.4f}' for v in v_trace[:NUM_STEPS]]}")

    print(f"\n  Output layer activity:")
    print(f"    Total output spikes    : {int(spike_counts.sum())}")
    print(f"    Per-timestep (output)  : {list(output_spikes_per_step)}")

    # Output neurons that fired: print timesteps
    firing_output_neurons = [(i, output_spike_times_per_neuron[i])
                             for i in range(OUTPUT_SIZE)
                             if spike_counts[i] > 0]
    if firing_output_neurons:
        print(f"    Firing output neurons ({len(firing_output_neurons)}):")
        for neuron_id, times in sorted(firing_output_neurons, key=lambda x: -len(x[1]))[:10]:
            print(f"      output[{neuron_id:2d}]: {len(times):3d} spikes at t={times[:8]}")
    else:
        print(f"    No output neurons fired.")

    # Top 10 output neurons by final voltage
    top10_output_v = sorted(
        [(int(i), float(final_v[i])) for i in range(OUTPUT_SIZE)],
        key=lambda x: -x[1]
    )[:10]
    print(f"    Top 10 output neurons by final membrane voltage:")
    for nid, v in top10_output_v:
        spike_marker = f"  ({int(spike_counts[nid])} spikes)" if spike_counts[nid] > 0 else ""
        print(f"      output[{nid:2d}]: v={v:.6f}{spike_marker}")

    # ============================================================
    # Classify
    # ============================================================
    if spike_counts.sum() > 0:
        predicted = int(np.argmax(spike_counts))
        classification_basis = "spike count"
    else:
        predicted = int(np.argmax(final_v))
        classification_basis = "final membrane voltage (no spikes)"

    is_correct = (predicted == true_label)
    if is_correct:
        correct += 1

    status = "CORRECT" if is_correct else "wrong"
    print(f"\n  Classification: pred={predicted} true={true_label} ({status})")
    print(f"  Basis: {classification_basis}")

    result = {
        "sample": sample_idx,
        "true_label": true_label,
        "predicted": predicted,
        "correct": is_correct,
        "classification_basis": classification_basis,
        "output_spikes": int(spike_counts.sum()),
        "hidden_spikes": hidden_spike_count,
        "input_spikes": total_input_spikes,
        "input_spikes_actual": in_total_actual,
        "active_inputs": active_inputs,
        "hidden_neurons_fired": int((hidden_spikes_per_neuron > 0).sum()),
        "max_hidden_v": float(max_v_per_hidden.max()),
        "inference_ms": elapsed * 1000,
        "top5_output_v": top10_output_v[:5],
    }
    results.append(result)

# ============================================================
# Final Summary
# ============================================================
valid_results = [r for r in results if "error" not in r]
accuracy = correct / NUM_SAMPLES if NUM_SAMPLES > 0 else 0.0
avg_time = total_sim_time / max(1, len(valid_results))

print(f"\n")
print(f"{'=' * 70}")
print(f"FINAL SUMMARY -- run_on_spinnaker.py")
print(f"{'=' * 70}")
print(f"  Platform         : SpiNNaker (spinnaker.cs.man.ac.uk)")
print(f"  Neuron model     : {NEURON_MODEL}")
print(f"  Tau_syn          : {TAU_SYN} ms")
print(f"  Weight scale     : {WEIGHT_SCALE}x")
print(f"  Prune threshold  : {WEIGHT_PRUNE_THRESHOLD}")
print(f"  Max hidden       : {MAX_HIDDEN}")
print(f"  Total connections: {total_conns}")
print(f"{'─' * 70}")
print(f"  Samples          : {NUM_SAMPLES}")
print(f"  Errored samples  : {NUM_SAMPLES - len(valid_results)}")
print(f"  Correct          : {correct}")
print(f"  Accuracy         : {accuracy:.1%}")
print(f"  Avg wall clock   : {avg_time*1000:.1f} ms per sample")
print(f"  Total wall clock : {total_sim_time:.1f} s")
print(f"  Sim time         : {NUM_STEPS} ms per sample (on-chip)")
print(f"{'─' * 70}")

# Per-sample results table
print(f"  Per-sample results:")
print(f"  {'Sample':>7}  {'True':>5}  {'Pred':>5}  {'Result':>9}  "
      f"{'OutSpk':>7}  {'HidSpk':>7}  {'MaxHidV':>8}  {'HidFired':>9}")
for r in results:
    if "error" in r:
        print(f"  {r['sample']:>7}  {r['true_label']:>5}  {'ERR':>5}  {'ERROR':>9}  "
              f"  [{r['error'][:30]}]")
    else:
        result_str = "CORRECT" if r["correct"] else "wrong"
        print(f"  {r['sample']:>7}  {r['true_label']:>5}  {r['predicted']:>5}  "
              f"{result_str:>9}  {r['output_spikes']:>7}  {r['hidden_spikes']:>7}  "
              f"{r['max_hidden_v']:>8.4f}  "
              f"{r['hidden_neurons_fired']:>9}/{HIDDEN_SIZE}")

# ============================================================
# Provenance data instructions
# ============================================================
print(f"\n{'─' * 70}")
print("Provenance data (SpiNNaker internal diagnostics):")
try:
    import glob as glob_mod
    # sPyNNaker saves provenance to a timestamped subdirectory
    prov_candidates = sorted(
        glob_mod.glob(str(Path.home() / "spynnaker_output" / "*" / "provenance_data" / "*.sqlite3")) +
        glob_mod.glob("/tmp/spynnaker_output/*/provenance_data/*.sqlite3") +
        glob_mod.glob(str(Path.cwd() / "spynnaker_output" / "*" / "provenance_data" / "*.sqlite3"))
    )
    if prov_candidates:
        latest_prov = prov_candidates[-1]
        print(f"  Provenance database found:")
        print(f"    {latest_prov}")
        print(f"  Query it with:")
        print(f'    sqlite3 "{latest_prov}" "SELECT * FROM provenance_data LIMIT 50;"')
        print(f'    sqlite3 "{latest_prov}" "SELECT description, the_value FROM provenance_data '
              f'WHERE description LIKE \'%spike%\';"')
    else:
        print("  No provenance .sqlite3 file found in standard locations.")
        print("  Check ~/spynnaker_output/ or current working directory.")
        print("  Set SPYNNAKER_DISABLE_PROVENANCE=0 (already set in this script).")
except Exception as prov_exc:
    print(f"  Could not locate provenance file: {prov_exc}")

# ============================================================
# Save results
# ============================================================
output = {
    "platform": "SpiNNaker",
    "spalloc_server": "spinnaker.cs.man.ac.uk",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model": "ESC-50 Convolutional SNN (best_fold4.pt)",
    "deployment": "FC-only hybrid (conv on CPU, FC on SpiNNaker)",
    "config": {
        "neuron_model": NEURON_MODEL,
        "tau_syn": TAU_SYN,
        "weight_scale": WEIGHT_SCALE,
        "prune_threshold": WEIGHT_PRUNE_THRESHOLD,
        "max_hidden": MAX_HIDDEN,
        "total_connections": total_conns,
    },
    "num_samples": NUM_SAMPLES,
    "correct": correct,
    "accuracy": accuracy,
    "avg_wall_clock_ms": avg_time * 1000,
    "sim_time_ms": NUM_STEPS,
    "total_wall_clock_s": total_sim_time,
    "network": {
        "input_neurons": INPUT_SIZE,
        "hidden_neurons": HIDDEN_SIZE,
        "output_neurons": OUTPUT_SIZE,
        "total_neurons": INPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE,
        "total_connections": total_conns,
    },
    "lif_params": lif_params,
    "per_sample": results,
}

results_file = RESULTS_DIR / "spinnaker_inference.json"
with open(results_file, "w") as f:
    json.dump(output, f, indent=2)

SCRIPT_END = time.time()
print(f"\n  Results saved to: {results_file}")
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] run_on_spinnaker.py -- finished")
print(f"Total wall clock time: {SCRIPT_END - SCRIPT_START:.1f} s")
