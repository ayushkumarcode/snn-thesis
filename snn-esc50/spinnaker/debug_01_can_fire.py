"""
debug_01_can_fire.py -- Minimal single-neuron firing test.

WHAT THIS TESTS:
    The most fundamental question: can IF_curr_exp neurons fire at all using our
    exact LIF parameters (cm=1.0, tau_m=20.0, tau_syn_E=1.0, v_thresh=1.0)?

WHAT IT PROVES/DISPROVES:
    - PASS: IF_curr_exp with our params fires when given a generous weight (5.0).
      The hardware link and neuron model are working.
    - FAIL at weight=5.0: The neuron model parameters are completely wrong or the
      SpiNNaker connection is broken. Nothing else will work until this passes.
    - FAIL at weight=1.0 but PASS at weight=5.0: Our trained weights (which are
      typically in the range 0.01-0.5) are too small. We need weight scaling.
    - FAIL at weight=0.1 but PASS at weight=1.0: Confirms trained weights need 10x+
      scaling. See debug_05_weight_scale.py for calibration.

INTERPRETATION:
    Read the full voltage trace carefully:
    - If voltage stays at 0.0 for all timesteps -> current is not reaching the
      membrane. Either tau_syn is killing the current (run debug_02_tau_syn.py)
      or the projection was not created (connection list bug).
    - If voltage accumulates but never reaches 1.0 -> current is too weak.
      Need larger weights or lower threshold.
    - If voltage spikes once then resets -> neuron fired! Check refractory effects.

NEXT STEPS:
    - If all 3 weights fail: check SpiNNaker connection, run with --help to verify
      the board is reachable.
    - If only weight=5.0 passes: go to debug_02_tau_syn.py.
    - If weight=1.0 passes: go to debug_04_real_weights.py.

USAGE:
    source .venv-spinnaker/bin/activate
    python spinnaker/debug_01_can_fire.py
"""

import sys
import time

# Print timestamp at start
SCRIPT_START = time.time()
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] debug_01_can_fire.py -- starting")
print("=" * 70)
print("TEST: Can IF_curr_exp fire at all with our LIF parameters?")
print("=" * 70)

try:
    import pyNN.spiNNaker as sim
except ImportError as exc:
    print(f"\nFATAL: Cannot import pyNN.spiNNaker: {exc}")
    print("Make sure you activated the spinnaker venv:")
    print("  source .venv-spinnaker/bin/activate")
    sys.exit(1)

# ============================================================
# Parameters -- printed before simulation so any run failure
# leaves a complete audit trail in the log
# ============================================================
NUM_STEPS = 25          # 25 ms simulation (matches our trained model)
DT = 1.0                # timestep in ms

# Exact LIF params from run_on_spinnaker.py
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

# Three test weights: very generous, actual scale, small
TEST_WEIGHTS = [5.0, 1.0, 0.1]

print(f"\nConfiguration:")
print(f"  SpiNNaker server : spinnaker.cs.man.ac.uk (from ~/.spynnaker.cfg)")
print(f"  Simulation time  : {NUM_STEPS} ms (timestep={DT} ms)")
print(f"  Input pattern    : fires every timestep 0-{NUM_STEPS-1} (all {NUM_STEPS} steps)")
print(f"  Neuron model     : IF_curr_exp")
print(f"  LIF params       : {LIF_PARAMS}")
print(f"  Test weights     : {TEST_WEIGHTS}")
print(f"  Expected outcome : weight=5.0 MUST fire, weight=0.1 likely won't")
print()


def run_single_neuron_test(weight: float, test_label: str) -> dict:
    """Run a 1-input -> 1-neuron simulation and return diagnostics.

    Args:
        weight: Synaptic weight for the single connection.
        test_label: Human-readable label for this test.

    Returns:
        Dict with keys: weight, n_spikes, v_trace, max_v, fired
    """
    print(f"\n{'─' * 70}")
    print(f"  Subtest: weight = {weight:.2f}  [{test_label}]")
    print(f"{'─' * 70}")

    # Input: single neuron fires at every timestep 0..24
    spike_times = list(range(NUM_STEPS))  # [0, 1, 2, ..., 24]
    print(f"  Input spike times: {spike_times}")
    print(f"  Weight: {weight}")
    print(f"  Delay:  1.0 ms (minimum)")

    sim.setup(timestep=DT)

    # 1 input neuron, fires every step
    input_pop = sim.Population(
        1, sim.SpikeSourceArray,
        {"spike_times": [spike_times]},
        label="input_1"
    )
    input_pop.record("spikes")

    # 1 IF_curr_exp neuron with our exact params
    neuron = sim.Population(
        1, sim.IF_curr_exp(**LIF_PARAMS),
        label="neuron_1"
    )
    neuron.initialize(v=0.0)  # Critical: must start at 0, not sPyNNaker default -65mV
    neuron.record(["spikes", "v"])

    # Single excitatory connection
    sim.Projection(
        input_pop, neuron,
        sim.FromListConnector([[0, 0, weight, 1.0]]),
        receptor_type="excitatory"
    )

    print(f"  Running {NUM_STEPS} ms simulation...")
    t0 = time.time()
    try:
        sim.run(NUM_STEPS)
    except Exception as exc:
        _handle_sim_exception(exc)
        sim.end()
        return {"weight": weight, "n_spikes": -1, "v_trace": [], "max_v": float("nan"), "fired": False, "error": str(exc)}
    elapsed = time.time() - t0

    # --- Extract spikes ---
    spike_data = neuron.get_data("spikes")
    spikes = spike_data.segments[-1].spiketrains
    n_spikes = sum(len(st) for st in spikes)
    spike_times_out = [float(t) for st in spikes for t in st]

    # --- Extract voltage trace ---
    v_data = neuron.get_data("v")
    v_trace = []
    for sig in v_data.segments[-1].analogsignals:
        if sig.shape[1] >= 1:
            v_trace = [float(v) for v in sig[:, 0]]

    max_v = max(v_trace) if v_trace else float("nan")
    fired = n_spikes > 0

    # --- Extract input spike verification ---
    in_spike_data = input_pop.get_data("spikes")
    in_spikes = sum(len(st) for st in in_spike_data.segments[-1].spiketrains)

    sim.end()

    # --- Print detailed results ---
    print(f"\n  Results (wall clock: {elapsed*1000:.0f} ms):")
    print(f"    Input spikes sent       : {in_spikes} (expected {NUM_STEPS})")
    print(f"    Output spikes fired     : {n_spikes}")
    print(f"    Spike times             : {spike_times_out}")
    print(f"    Max voltage reached     : {max_v:.6f}  (threshold = {LIF_PARAMS['v_thresh']})")
    print(f"\n  Full voltage trace (all {len(v_trace)} timesteps):")
    if v_trace:
        for i, v in enumerate(v_trace):
            marker = " <-- FIRED (reset after)" if any(abs(t - i) < 1.0 for t in spike_times_out) else ""
            print(f"    t={i:3d}: v = {v:8.5f}{marker}")
    else:
        print("    (no voltage data recorded -- recording may have failed)")

    return {
        "weight": weight,
        "n_spikes": n_spikes,
        "v_trace": v_trace,
        "max_v": max_v,
        "fired": fired,
        "input_spikes_verified": in_spikes,
    }


def _handle_sim_exception(exc: Exception) -> None:
    """Print plain-English explanation for common SpiNNaker exceptions."""
    exc_type = type(exc).__name__
    exc_str = str(exc)
    print(f"\n  EXCEPTION: {exc_type}: {exc_str}")

    if "SpinnmanIOException" in exc_type or "No buffer space" in exc_str:
        print("  MEANING: UDP send buffer overflow. Too many packets sent at once.")
        print("  WHAT TO TRY: Reduce NUM_STEPS, or use fewer input spikes.")
    elif "SpinnmanTimeoutException" in exc_type or "timeout" in exc_str.lower():
        print("  MEANING: SpiNNaker board did not respond in time.")
        print("  WHAT TO TRY: Check board is powered on and network is reachable.")
        print("    ping spinnaker.cs.man.ac.uk")
        print("    Check spalloc server is accepting jobs.")
    elif "ConnectionRefused" in exc_str or "ECONNREFUSED" in exc_str:
        print("  MEANING: Cannot connect to SpiNNaker spalloc server.")
        print("  WHAT TO TRY: Check VPN is connected to Manchester network.")
    else:
        print("  MEANING: Unknown error. Check SpiNNaker board status.")
        print("  WHAT TO TRY: Rerun with verbose logging enabled.")


# ============================================================
# Run all three tests
# ============================================================
labels = ["very generous (should definitely fire)", "actual trained weight scale", "small weight (probably won't fire)"]
results = []

for weight, label in zip(TEST_WEIGHTS, labels):
    result = run_single_neuron_test(weight, label)
    results.append(result)

# ============================================================
# Summary section
# ============================================================
print("\n")
print("=" * 70)
print("SUMMARY -- debug_01_can_fire.py")
print("=" * 70)

min_working_weight = None
for r in results:
    status = "=== PASS ===" if r.get("fired") else "=== FAIL ==="
    err = f"  [ERROR: {r.get('error', '')}]" if "error" in r else ""
    print(f"  weight={r['weight']:.2f}: {status}  "
          f"spikes={r['n_spikes']:3d}  max_v={r['max_v']:.4f}{err}")
    if r.get("fired") and min_working_weight is None:
        min_working_weight = r["weight"]

print()
if min_working_weight is not None:
    print(f"  CONCLUSION: Minimum weight to fire appears to be ~{min_working_weight}")
    if min_working_weight <= 1.0:
        print("  -> Weight=1.0 fires. Real weights may be close to working.")
        print("     Proceed to debug_04_real_weights.py.")
    else:
        print(f"  -> Weight={min_working_weight} required. Trained weights need scaling.")
        print("     Proceed to debug_02_tau_syn.py to check if tau_syn is the cause,")
        print("     then debug_05_weight_scale.py for calibration.")
else:
    print("  CONCLUSION: NO test passed. IF_curr_exp cannot fire with our params.")
    print("  -> Proceed to debug_02_tau_syn.py immediately.")
    print("     The synaptic current decay (tau_syn_E=1.0) may be killing all signal.")

SCRIPT_END = time.time()
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] debug_01_can_fire.py -- finished")
print(f"Total wall clock time: {SCRIPT_END - SCRIPT_START:.1f} s")
