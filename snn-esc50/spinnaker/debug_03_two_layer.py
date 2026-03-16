"""
debug_03_two_layer.py -- Two-layer signal propagation test.

WHAT THIS TESTS:
    Whether a spiking signal can propagate through two layers of IF_curr_exp
    neurons: input -> hidden -> output. This mirrors our actual FC1 -> FC2
    structure and tests whether layer-to-layer transmission works even when
    individual neurons CAN fire (as confirmed by debug_01).

WHAT IT PROVES/DISPROVES:
    - PASS (layer 1 fires AND layer 2 fires): Both layers propagate signal.
      The architecture is sound. Problems with run_on_spinnaker must be
      weight-related, not architectural.
    - PASS layer 1, FAIL layer 2: Layer-to-layer transmission fails.
      Hidden neurons fire but cannot drive output neurons. Likely:
      (a) weights between layers too small, or (b) output layer needs
      different params. Check spike rate from hidden layer.
    - FAIL layer 1: Even with weight=2.0, hidden neurons don't fire.
      Go back to debug_01 and debug_02 -- fundamental firing problem.

TOPOLOGY:
    10 SpikeSourceArray (fire every step)
    -> 5 IF_curr_exp hidden neurons  (all-to-all, weight=2.0)
    -> 3 IF_curr_exp output neurons  (all-to-all, weight=2.0)

    All weights excitatory. All-to-all connectivity. Generous weight.

PER-LAYER MONITORING:
    - Input: record spikes from SpikeSourceArray
    - Hidden: record spikes AND voltage
    - Output: record spikes AND voltage
    - Print per-timestep spike counts for all three layers

USAGE:
    source .venv-spinnaker/bin/activate
    python spinnaker/debug_03_two_layer.py
"""

import sys
import time

SCRIPT_START = time.time()
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] debug_03_two_layer.py -- starting")
print("=" * 70)
print("TEST: Two-layer signal propagation. Does layer 1 drive layer 2?")
print("=" * 70)

try:
    import pyNN.spiNNaker as sim
except ImportError as exc:
    print(f"\nFATAL: Cannot import pyNN.spiNNaker: {exc}")
    print("Make sure you activated the spinnaker venv:")
    print("  source .venv-spinnaker/bin/activate")
    sys.exit(1)

import numpy as np

# ============================================================
# Parameters
# ============================================================
NUM_STEPS = 25
DT = 1.0

N_INPUT = 10
N_HIDDEN = 5
N_OUTPUT = 3

WEIGHT = 2.0   # Generous weight: enough to fire any reasonable neuron
DELAY = 1.0    # Minimum delay

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
print(f"  SpiNNaker server : spinnaker.cs.man.ac.uk")
print(f"  Simulation time  : {NUM_STEPS} ms (timestep={DT} ms)")
print(f"  Topology         : {N_INPUT} input -> {N_HIDDEN} hidden -> {N_OUTPUT} output")
print(f"  Connectivity     : all-to-all excitatory")
print(f"  Weight           : {WEIGHT} (all connections)")
print(f"  Delay            : {DELAY} ms")
print(f"  Input pattern    : all {N_INPUT} inputs fire every timestep")
print(f"  LIF params       : {LIF_PARAMS}")
print(f"  Total connections: {N_INPUT * N_HIDDEN} (input->hidden) + "
      f"{N_HIDDEN * N_OUTPUT} (hidden->output) = {N_INPUT*N_HIDDEN + N_HIDDEN*N_OUTPUT}")
print()


def _handle_sim_exception(exc: Exception) -> None:
    exc_type = type(exc).__name__
    exc_str = str(exc)
    print(f"\nEXCEPTION: {exc_type}: {exc_str}")
    if "SpinnmanIOException" in exc_type or "No buffer space" in exc_str:
        print("MEANING: UDP buffer overflow. This should NOT happen with this tiny network.")
        print("WHAT TO TRY: Restart the script. If persistent, the board may be overloaded.")
    elif "SpinnmanTimeoutException" in exc_type or "timeout" in exc_str.lower():
        print("MEANING: SpiNNaker board timed out.")
        print("WHAT TO TRY: Check board is powered and accessible.")
    else:
        print("MEANING: Unknown error.")


def build_all_to_all_connections(pre_size: int, post_size: int,
                                  weight: float, delay: float) -> list:
    """Build all-to-all connection list in FromListConnector format."""
    conns = []
    for pre in range(pre_size):
        for post in range(post_size):
            conns.append([pre, post, weight, delay])
    return conns


# ============================================================
# Build connection lists
# ============================================================
input_to_hidden_conns = build_all_to_all_connections(N_INPUT, N_HIDDEN, WEIGHT, DELAY)
hidden_to_output_conns = build_all_to_all_connections(N_HIDDEN, N_OUTPUT, WEIGHT, DELAY)

print(f"Connection list sizes:")
print(f"  input->hidden : {len(input_to_hidden_conns)} connections")
print(f"  hidden->output: {len(hidden_to_output_conns)} connections")

# All 10 inputs fire at every timestep
spike_times_per_neuron = [list(range(NUM_STEPS)) for _ in range(N_INPUT)]

# ============================================================
# Run simulation
# ============================================================
print(f"\nSetting up SpiNNaker simulation...")
sim.setup(timestep=DT)

input_pop = sim.Population(
    N_INPUT, sim.SpikeSourceArray,
    {"spike_times": spike_times_per_neuron},
    label="input"
)
input_pop.record("spikes")

hidden_pop = sim.Population(
    N_HIDDEN, sim.IF_curr_exp(**LIF_PARAMS),
    label="hidden"
)
hidden_pop.initialize(v=0.0)
hidden_pop.record(["spikes", "v"])

output_pop = sim.Population(
    N_OUTPUT, sim.IF_curr_exp(**LIF_PARAMS),
    label="output"
)
output_pop.initialize(v=0.0)
output_pop.record(["spikes", "v"])

sim.Projection(
    input_pop, hidden_pop,
    sim.FromListConnector(input_to_hidden_conns),
    receptor_type="excitatory"
)

sim.Projection(
    hidden_pop, output_pop,
    sim.FromListConnector(hidden_to_output_conns),
    receptor_type="excitatory"
)

print(f"Running {NUM_STEPS} ms simulation...")
t0 = time.time()
try:
    sim.run(NUM_STEPS)
except Exception as exc:
    _handle_sim_exception(exc)
    sim.end()
    SCRIPT_END = time.time()
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] debug_03_two_layer.py -- ABORTED")
    print(f"Total wall clock time: {SCRIPT_END - SCRIPT_START:.1f} s")
    sys.exit(1)
elapsed = time.time() - t0
print(f"Simulation complete in {elapsed*1000:.0f} ms wall clock.")

# ============================================================
# Extract data
# ============================================================

# --- Input layer ---
in_spike_data = input_pop.get_data("spikes")
in_spikes_per_t = np.zeros(NUM_STEPS, dtype=int)
in_total = 0
for st in in_spike_data.segments[-1].spiketrains:
    for t in st:
        step = int(round(float(t)))
        if 0 <= step < NUM_STEPS:
            in_spikes_per_t[step] += 1
    in_total += len(st)

# --- Hidden layer ---
hid_spike_data = hidden_pop.get_data("spikes")
hid_spikes_per_t = np.zeros(NUM_STEPS, dtype=int)
hid_total = 0
hid_spike_times_per_neuron = [[] for _ in range(N_HIDDEN)]
for st in hid_spike_data.segments[-1].spiketrains:
    nid = int(st.annotations.get("source_index", 0))
    for t in st:
        step = int(round(float(t)))
        if 0 <= step < NUM_STEPS:
            hid_spikes_per_t[step] += 1
        if nid < N_HIDDEN:
            hid_spike_times_per_neuron[nid].append(float(t))
    hid_total += len(st)

hid_v_data = hidden_pop.get_data("v")
hid_v_traces = {}  # neuron_id -> list of v values
for sig in hid_v_data.segments[-1].analogsignals:
    for n in range(min(sig.shape[1], N_HIDDEN)):
        hid_v_traces[n] = [float(sig[t, n]) for t in range(sig.shape[0])]

# --- Output layer ---
out_spike_data = output_pop.get_data("spikes")
out_spikes_per_t = np.zeros(NUM_STEPS, dtype=int)
out_total = 0
out_spike_times_per_neuron = [[] for _ in range(N_OUTPUT)]
for st in out_spike_data.segments[-1].spiketrains:
    nid = int(st.annotations.get("source_index", 0))
    for t in st:
        step = int(round(float(t)))
        if 0 <= step < NUM_STEPS:
            out_spikes_per_t[step] += 1
        if nid < N_OUTPUT:
            out_spike_times_per_neuron[nid].append(float(t))
    out_total += len(st)

out_v_data = output_pop.get_data("v")
out_v_traces = {}
for sig in out_v_data.segments[-1].analogsignals:
    for n in range(min(sig.shape[1], N_OUTPUT)):
        out_v_traces[n] = [float(sig[t, n]) for t in range(sig.shape[0])]

sim.end()

# ============================================================
# Print detailed per-timestep breakdown
# ============================================================
print(f"\n{'─' * 70}")
print("Per-timestep spike counts across all layers:")
print(f"{'─' * 70}")
print(f"  {'t':>4}  {'Input':>8}  {'Hidden':>8}  {'Output':>8}")
print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}")
for t in range(NUM_STEPS):
    in_bar = "#" * in_spikes_per_t[t]
    hid_bar = "#" * hid_spikes_per_t[t]
    out_bar = "#" * out_spikes_per_t[t]
    print(f"  {t:>4}  {in_spikes_per_t[t]:>8}  {hid_spikes_per_t[t]:>8}  {out_spikes_per_t[t]:>8}  "
          f"|{in_bar:<12}|{hid_bar:<8}|{out_bar}")

print(f"\n  TOTAL: input={in_total}  hidden={hid_total}  output={out_total}")

# Per-neuron firing report
print(f"\nHidden neuron firing summary (N={N_HIDDEN}):")
for n in range(N_HIDDEN):
    times = hid_spike_times_per_neuron[n]
    max_v = max(hid_v_traces.get(n, [0.0])) if hid_v_traces.get(n) else 0.0
    print(f"  hidden[{n}]: {len(times):3d} spikes  max_v={max_v:.5f}  times={times[:10]}")

print(f"\nOutput neuron firing summary (N={N_OUTPUT}):")
for n in range(N_OUTPUT):
    times = out_spike_times_per_neuron[n]
    max_v = max(out_v_traces.get(n, [0.0])) if out_v_traces.get(n) else 0.0
    print(f"  output[{n}]: {len(times):3d} spikes  max_v={max_v:.5f}  times={times[:10]}")

# ============================================================
# Summary
# ============================================================
print("\n")
print("=" * 70)
print("SUMMARY -- debug_03_two_layer.py")
print("=" * 70)

layer1_pass = hid_total > 0
layer2_pass = out_total > 0

print(f"  Input  -> Hidden: {'=== PASS ===' if layer1_pass else '=== FAIL ==='}"
      f"  ({in_total} in -> {hid_total} hidden spikes)")
print(f"  Hidden -> Output: {'=== PASS ===' if layer2_pass else '=== FAIL ==='}"
      f"  ({hid_total} hidden -> {out_total} output spikes)")
print()

if layer1_pass and layer2_pass:
    print("  CONCLUSION: Signal propagated through BOTH layers.")
    print("  -> Architecture is working. Problems in run_on_spinnaker are weight-related.")
    print("     Proceed to debug_04_real_weights.py or debug_05_weight_scale.py.")
elif layer1_pass and not layer2_pass:
    print("  CONCLUSION: Layer 1 fires but Layer 2 does NOT.")
    print(f"  -> {hid_total} hidden spikes exist but cannot drive output neurons.")
    print("     With weight=2.0 all-to-all, output should easily fire.")
    print("     Check: is output_pop being initialized correctly?")
    print("     Try increasing WEIGHT to 5.0 in this script.")
    print("     Or check if output layer needs different params.")
else:
    print("  CONCLUSION: Even layer 1 FAILED to fire.")
    print("  -> With weight=2.0 and 10 simultaneous inputs, no firing is a serious problem.")
    print("     Go back to debug_01_can_fire.py and verify weight=5.0 passes.")
    print("     Then run debug_02_tau_syn.py.")

SCRIPT_END = time.time()
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] debug_03_two_layer.py -- finished")
print(f"Total wall clock time: {SCRIPT_END - SCRIPT_START:.1f} s")
