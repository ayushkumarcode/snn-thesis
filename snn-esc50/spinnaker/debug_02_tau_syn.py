"""
debug_02_tau_syn.py -- Isolate whether tau_syn_E is killing the signal.

WHAT THIS TESTS:
    snnTorch's snn.Leaky neuron has NO synaptic current decay. Current from a
    presynaptic spike is injected instantaneously into the postsynaptic membrane.
    sPyNNaker's IF_curr_exp has an EXPONENTIAL synaptic current decay with time
    constant tau_syn_E (ms). If tau_syn_E is too large relative to the timestep,
    much of the current decays before it can be integrated, and the membrane
    voltage never builds up enough to reach threshold.

    The governing equation for IF_curr_exp:
        dI_syn/dt = -I_syn / tau_syn_E   (current decays exponentially)
        dV/dt = -(V - V_rest) / tau_m + I_syn / cm

    With tau_syn_E = 1.0 ms and dt = 1.0 ms, by the next timestep the current
    has decayed by a factor of exp(-1/1) = 0.368. So 63.2% of the current is
    lost per step. With tau_syn_E = 0.1 ms (near-instantaneous), the decay is
    exp(-1/0.1) = 4.5e-5, essentially all current is delivered in one step --
    closest to snnTorch behaviour.

WHAT IT PROVES/DISPROVES:
    - If tau_syn_E = 0.1 fires but tau_syn_E = 1.0 does not:
      Synaptic current decay IS the problem. Use --tau-syn 0.1 in
      run_on_spinnaker.py, or switch to IF_curr_delta (no synaptic decay).
    - If tau_syn_E = 5.0 never fires (worse than 1.0):
      Confirms decay kills signal. Smaller tau_syn is better.
    - If none fire regardless of tau_syn:
      The problem is NOT tau_syn. Return to debug_01 and check weight scale.

NEXT STEPS:
    - If tau_syn_E = 0.1 is the fix: use run_on_spinnaker.py --tau-syn 0.1
    - If this test shows no improvement: try debug_03_two_layer.py and
      debug_04_real_weights.py with large weights to isolate weight magnitude.

USAGE:
    source .venv-spinnaker/bin/activate
    python spinnaker/debug_02_tau_syn.py
"""

import sys
import time

SCRIPT_START = time.time()
print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] debug_02_tau_syn.py -- starting")
print("=" * 70)
print("TEST: Does reducing tau_syn_E allow IF_curr_exp neurons to fire?")
print("=" * 70)

try:
    import pyNN.spiNNaker as sim
except ImportError as exc:
    print(f"\nFATAL: Cannot import pyNN.spiNNaker: {exc}")
    print("Make sure you activated the spinnaker venv:")
    print("  source .venv-spinnaker/bin/activate")
    sys.exit(1)

# ============================================================
# Parameters
# ============================================================
NUM_STEPS = 25
DT = 1.0
WEIGHT = 1.0       # Fixed weight: our 'actual scale' from debug_01
INPUT_FIRES_EVERY_STEP = True   # Fire at every timestep 0-24

# tau_syn values to test (sorted best-to-worst expected)
TAU_SYN_VALUES = [0.1, 0.5, 1.0, 5.0]

# Base LIF params -- tau_syn_E will be varied per test
BASE_LIF_PARAMS = {
    "cm": 1.0,
    "tau_m": 20.0,
    "tau_syn_I": 1.0,
    "tau_refrac": 0.1,
    "v_reset": 0.0,
    "v_rest": 0.0,
    "v_thresh": 1.0,
}

print(f"\nConfiguration:")
print(f"  SpiNNaker server   : spinnaker.cs.man.ac.uk")
print(f"  Simulation time    : {NUM_STEPS} ms (timestep={DT} ms)")
print(f"  Fixed weight       : {WEIGHT} (excitatory, delay=1.0)")
print(f"  Input pattern      : fires every timestep 0-{NUM_STEPS-1}")
print(f"  Base LIF params    : {BASE_LIF_PARAMS}")
print(f"  tau_syn_E values   : {TAU_SYN_VALUES}")
print(f"\n  Theory (for weight={WEIGHT}, dt={DT} ms):")
for tau in TAU_SYN_VALUES:
    import math
    decay_per_step = math.exp(-DT / tau)
    frac_delivered = 1.0 - decay_per_step
    print(f"    tau_syn_E={tau:4.1f}: exp(-dt/tau)={decay_per_step:.4f}, "
          f"~{frac_delivered*100:.1f}% of current delivered per spike")
print()


def _handle_sim_exception(exc: Exception) -> None:
    exc_type = type(exc).__name__
    exc_str = str(exc)
    print(f"\n  EXCEPTION: {exc_type}: {exc_str}")
    if "SpinnmanIOException" in exc_type or "No buffer space" in exc_str:
        print("  MEANING: UDP send buffer overflow.")
        print("  WHAT TO TRY: Reduce NUM_STEPS.")
    elif "SpinnmanTimeoutException" in exc_type or "timeout" in exc_str.lower():
        print("  MEANING: SpiNNaker board timed out.")
        print("  WHAT TO TRY: Check board power and network connectivity.")
    elif "ConnectionRefused" in exc_str:
        print("  MEANING: Cannot connect to spalloc server.")
        print("  WHAT TO TRY: Ensure you are on Manchester VPN.")
    else:
        print("  MEANING: Unknown error.")


def run_tau_syn_test(tau_syn_e: float) -> dict:
    """Run 1-input -> 1-neuron test with a specific tau_syn_E value.

    Args:
        tau_syn_e: Synaptic time constant for excitatory inputs (ms).

    Returns:
        Dict with diagnostics.
    """
    print(f"\n{'─' * 70}")
    print(f"  Subtest: tau_syn_E = {tau_syn_e} ms")
    print(f"{'─' * 70}")

    lif_params = dict(BASE_LIF_PARAMS)
    lif_params["tau_syn_E"] = tau_syn_e

    spike_times = list(range(NUM_STEPS))

    sim.setup(timestep=DT)

    input_pop = sim.Population(
        1, sim.SpikeSourceArray,
        {"spike_times": [spike_times]},
        label="input"
    )
    input_pop.record("spikes")

    neuron = sim.Population(
        1, sim.IF_curr_exp(**lif_params),
        label="neuron"
    )
    neuron.initialize(v=0.0)
    neuron.record(["spikes", "v"])

    sim.Projection(
        input_pop, neuron,
        sim.FromListConnector([[0, 0, WEIGHT, 1.0]]),
        receptor_type="excitatory"
    )

    print(f"  Running simulation...")
    t0 = time.time()
    try:
        sim.run(NUM_STEPS)
    except Exception as exc:
        _handle_sim_exception(exc)
        sim.end()
        return {
            "tau_syn_e": tau_syn_e, "n_spikes": -1,
            "v_trace": [], "max_v": float("nan"), "fired": False, "error": str(exc)
        }
    elapsed = time.time() - t0

    # Extract spikes
    spike_data = neuron.get_data("spikes")
    n_spikes = sum(len(st) for st in spike_data.segments[-1].spiketrains)
    spike_times_out = [float(t) for st in spike_data.segments[-1].spiketrains for t in st]

    # Extract voltage trace
    v_data = neuron.get_data("v")
    v_trace = []
    for sig in v_data.segments[-1].analogsignals:
        if sig.shape[1] >= 1:
            v_trace = [float(v) for v in sig[:, 0]]

    # Extract input spike verification
    in_data = input_pop.get_data("spikes")
    in_spikes = sum(len(st) for st in in_data.segments[-1].spiketrains)

    max_v = max(v_trace) if v_trace else float("nan")
    fired = n_spikes > 0

    sim.end()

    print(f"  Wall clock: {elapsed*1000:.0f} ms")
    print(f"  Input spikes sent: {in_spikes} (expected {NUM_STEPS})")
    print(f"  Output spikes: {n_spikes}  |  Spike times: {spike_times_out}")
    print(f"  Max voltage: {max_v:.6f}  (threshold = {lif_params['v_thresh']})")
    print(f"\n  Full voltage trace (all {len(v_trace)} timesteps):")
    if v_trace:
        for i, v in enumerate(v_trace):
            bar_width = max(0, int(v * 40))
            bar = "#" * bar_width
            fired_marker = " FIRE" if any(abs(t - i) < 1.0 for t in spike_times_out) else ""
            print(f"    t={i:3d}: v = {v:8.5f}  |{bar}{fired_marker}")
    else:
        print("    (no voltage data)")

    return {
        "tau_syn_e": tau_syn_e,
        "n_spikes": n_spikes,
        "v_trace": v_trace,
        "max_v": max_v,
        "fired": fired,
        "input_spikes_verified": in_spikes,
    }


# ============================================================
# Run all tau_syn tests
# ============================================================
results = []
for tau_val in TAU_SYN_VALUES:
    result = run_tau_syn_test(tau_val)
    results.append(result)

# ============================================================
# Summary
# ============================================================
print("\n")
print("=" * 70)
print("SUMMARY -- debug_02_tau_syn.py")
print("=" * 70)
print(f"  Fixed weight = {WEIGHT}, input fires every step for {NUM_STEPS} steps\n")

first_passing_tau = None
for r in results:
    status = "=== PASS ===" if r.get("fired") else "=== FAIL ==="
    err = f"  [ERROR: {r.get('error', '')}]" if "error" in r else ""
    print(f"  tau_syn_E={r['tau_syn_e']:4.1f}: {status}  "
          f"spikes={r['n_spikes']:3d}  max_v={r['max_v']:.5f}{err}")
    if r.get("fired") and first_passing_tau is None:
        first_passing_tau = r["tau_syn_e"]

print()
if first_passing_tau is not None:
    print(f"  CONCLUSION: tau_syn_E = {first_passing_tau} ms is sufficient to fire.")
    if first_passing_tau < 1.0:
        print("  -> tau_syn IS the problem. Current setting (1.0 ms) kills signal.")
        print(f"     Use: python run_on_spinnaker.py --tau-syn {first_passing_tau}")
        print("     Or switch to IF_curr_delta: --neuron-model IF_curr_delta")
    else:
        print("  -> tau_syn=1.0 already works. The problem is not tau_syn.")
        print("     Proceed to debug_04_real_weights.py.")
else:
    print("  CONCLUSION: NO tau_syn value allowed firing with weight=1.0.")
    print("  -> tau_syn alone is not the fix. The weight is too small.")
    print("     Proceed to debug_05_weight_scale.py.")
    print("     Also try debug_01_can_fire.py with weight=5.0 to confirm hardware works.")

SCRIPT_END = time.time()
print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] debug_02_tau_syn.py -- finished")
print(f"Total wall clock time: {SCRIPT_END - SCRIPT_START:.1f} s")
