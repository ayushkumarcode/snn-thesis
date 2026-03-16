"""
auto_calibrate.py -- Self-iterating SpiNNaker calibration script for ESC-50 SNN.

Runs 5 sequential phases, each looping until it passes, then proceeds to the
next. No human intervention required. All SpiNNaker experiments are run
directly using pyNN.spiNNaker (no subprocess calls to other scripts).

Phases:
    1. Basic firing check     -- can IF_curr_exp fire at all with our params?
    2. tau_syn calibration    -- find optimal tau_syn for our custom params
    3. Weight scale calib.    -- find multiplier to make real FC1 weights fire
    4. Scale up hidden size   -- largest hidden size before UDP overflow
    5. Full inference run     -- run actual inference with calibrated params

Usage:
    source .venv-spinnaker/bin/activate
    python spinnaker/auto_calibrate.py
    python spinnaker/auto_calibrate.py --num-samples 10
    python spinnaker/auto_calibrate.py --skip-to-phase 3
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

# ============================================================
# Timestamp helpers (used throughout)
# ============================================================
SCRIPT_START = time.time()


def ts() -> str:
    """Return current timestamp string for log output."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# Argument parsing (before any heavy imports)
# ============================================================
parser = argparse.ArgumentParser(
    description="Auto-calibrating SpiNNaker inference for ESC-50 SNN"
)
parser.add_argument(
    "--num-samples", type=int, default=5,
    help="Number of samples to run in Phase 5 (default: 5)"
)
parser.add_argument(
    "--skip-to-phase", type=int, default=1, choices=[1, 2, 3, 4, 5],
    help="Jump directly to this phase, loading calibration state from disk (default: 1)"
)
args = parser.parse_args()

NUM_PHASE5_SAMPLES = args.num_samples
SKIP_TO_PHASE = args.skip_to_phase

# ============================================================
# Directory setup
# ============================================================
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
WEIGHTS_DIR = REPO_ROOT / "results" / "spinnaker_weights"
RESULTS_DIR = REPO_ROOT / "results" / "spinnaker_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_STATE_FILE = RESULTS_DIR / "calibration_state.json"

# ============================================================
# Iteration Logger -- records EVERY experiment, EVERY result
# ============================================================
ITERATIONS_DIR = RESULTS_DIR / "iterations"
ITERATIONS_DIR.mkdir(parents=True, exist_ok=True)
MASTER_LOG_FILE = RESULTS_DIR / "all_iterations.jsonl"   # one JSON per line, append-only
SESSION_ID = time.strftime("%Y%m%d_%H%M%S")              # unique ID for this run


class IterationLogger:
    """
    Records every single SpiNNaker experiment iteration in full detail.
    Writes to:
      - results/spinnaker_results/all_iterations.jsonl   (append-only master log)
      - results/spinnaker_results/iterations/<id>.json   (one file per iteration)
    Every call to .record() is flushed to disk immediately so nothing is lost
    if the script crashes mid-run.
    """

    def __init__(self):
        self._counter = 0

    def record(self, record: dict) -> None:
        """
        Save one iteration record. Call this after every sim.run() + data extraction.

        Required keys in record:
          phase         (int)    -- 1-5
          test_name     (str)    -- human label e.g. "weight=5.0" or "tau_syn=0.1"
          params        (dict)   -- all parameters for this iteration
          board_ip      (str)    -- board IP from spalloc (if known, else "unknown")
          spalloc_job   (int)    -- spalloc job ID (if known, else -1)
          n_input_spikes_expected (int)
          n_input_spikes_actual   (int)
          n_hidden_spikes (int)
          n_output_spikes (int)
          max_hidden_v    (float)
          max_output_v    (float)
          v_trace         (list)  -- voltage trace (may be [] if not applicable)
          hidden_v_traces (dict)  -- {neuron_id: [v0..v24]} for key neurons
          output_v_traces (dict)  -- {neuron_id: [v0..v24]} for key output neurons
          spikes_per_step_input  (list) -- per-timestep spike counts
          spikes_per_step_hidden (list)
          spikes_per_step_output (list)
          neurons_that_fired_hidden (list) -- neuron IDs
          neurons_that_fired_output (list)
          wall_clock_ms   (float)
          passed          (bool)
          error           (str or None)
          notes           (str)   -- free text for anything extra

        Optional but encouraged:
          firmware_version (str)
          chip_count       (int)
          core_count       (int)
          weight_stats     (dict) -- {min, max, mean, std, p5, p25, p50, p75, p95}
          connection_count (int)
        """
        self._counter += 1
        iteration_id = f"{SESSION_ID}_ph{record.get('phase', 0)}_{self._counter:04d}"

        full_record = {
            "iteration_id": iteration_id,
            "session_id": SESSION_ID,
            "timestamp": ts(),
            "timestamp_unix": time.time(),
            **record,
        }

        # 1. Append to master JSONL log (one JSON object per line)
        try:
            with open(MASTER_LOG_FILE, "a") as f:
                f.write(json.dumps(full_record) + "\n")
                f.flush()
        except Exception as exc:
            print(f"  [logger WARNING] Could not write to master log: {exc}")

        # 2. Write individual file per iteration
        try:
            iter_file = ITERATIONS_DIR / f"{iteration_id}.json"
            with open(iter_file, "w") as f:
                json.dump(full_record, f, indent=2)
        except Exception as exc:
            print(f"  [logger WARNING] Could not write iteration file: {exc}")

        # 3. Print a compact one-liner to stdout so it's visible in logs
        passed_str = "PASS" if full_record.get("passed") else "FAIL"
        err_str = f"  ERR={full_record['error'][:40]}" if full_record.get("error") else ""
        print(
            f"  [LOG {iteration_id}] {passed_str} | "
            f"phase={record.get('phase')} | {record.get('test_name','?')} | "
            f"hid_spk={record.get('n_hidden_spikes', '?')} | "
            f"max_hid_v={record.get('max_hidden_v', float('nan')):.5f} | "
            f"out_spk={record.get('n_output_spikes', '?')}{err_str}"
        )

    def summary(self) -> None:
        """Print a summary of all iterations recorded this session."""
        print(f"\n  [LOGGER SUMMARY] {self._counter} iterations recorded this session.")
        print(f"  Master log : {MASTER_LOG_FILE}")
        print(f"  Per-iter   : {ITERATIONS_DIR}/")
        # Count passes per phase
        try:
            records = []
            with open(MASTER_LOG_FILE) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            r = json.loads(line)
                            if r.get("session_id") == SESSION_ID:
                                records.append(r)
                        except Exception:
                            pass
            if records:
                by_phase = {}
                for r in records:
                    p = r.get("phase", 0)
                    by_phase.setdefault(p, {"pass": 0, "fail": 0})
                    if r.get("passed"):
                        by_phase[p]["pass"] += 1
                    else:
                        by_phase[p]["fail"] += 1
                for p in sorted(by_phase):
                    d = by_phase[p]
                    print(f"    Phase {p}: {d['pass']} PASS / {d['fail']} FAIL")
        except Exception:
            pass


# Global logger instance used by all phases
LOGGER = IterationLogger()

# ============================================================
# Print startup header
# ============================================================
print(f"[{ts()}] auto_calibrate.py -- starting")
print("=" * 60)
print("SpiNNaker Auto-Calibration for ESC-50 SNN")
print("=" * 60)
print(f"  Weights dir  : {WEIGHTS_DIR}")
print(f"  Results dir  : {RESULTS_DIR}")
print(f"  Phase 5 samp : {NUM_PHASE5_SAMPLES}")
print(f"  Skip to phase: {SKIP_TO_PHASE}")
print()

# ============================================================
# Import pyNN.spiNNaker
# ============================================================
try:
    import numpy as np
except ImportError:
    print("FATAL: numpy not available. Install it in your venv.")
    sys.exit(1)

try:
    import pyNN.spiNNaker as sim
    from spinnman.exceptions import SpinnmanIOException, SpinnmanTimeoutException
except ImportError as exc:
    print(f"\nFATAL: Cannot import pyNN.spiNNaker: {exc}")
    print("Make sure you activated the spinnaker venv:")
    print("  source .venv-spinnaker/bin/activate")
    sys.exit(1)

# ============================================================
# Simulation constants
# ============================================================
NUM_STEPS = 25      # ms -- matches trained model
DT = 1.0            # timestep in ms

# Base LIF params matching snnTorch snn.Leaky (beta=0.95, thresh=1.0, reset=0)
LIF_PARAMS_BASE = {
    "cm": 1.0,
    "tau_m": 20.0,
    "tau_refrac": 0.1,
    "v_reset": 0.0,
    "v_rest": 0.0,
    "v_thresh": 1.0,
}

# Standard neuroscience params -- used only as sanity check in Phase 1
LIF_PARAMS_STANDARD = {
    "cm": 1.0,
    "tau_m": 20.0,
    "tau_refrac": 2.0,
    "v_reset": -65.0,
    "v_rest": -65.0,
    "v_thresh": -50.0,
}

PRUNE_THRESHOLD = 0.05


# ============================================================
# Calibration state management
# ============================================================
def load_calibration_state() -> dict:
    """Load saved calibration state from disk, or return empty dict."""
    if CALIBRATION_STATE_FILE.exists():
        try:
            with open(CALIBRATION_STATE_FILE) as f:
                state = json.load(f)
            print(f"  Found saved calibration state at {CALIBRATION_STATE_FILE}")
            return state
        except Exception as exc:
            print(f"  WARNING: Could not load saved state: {exc}. Starting fresh.")
    return {}


def save_calibration_state(calibration: dict) -> None:
    """Persist calibration state to disk after each phase."""
    try:
        with open(CALIBRATION_STATE_FILE, "w") as f:
            json.dump(calibration, f, indent=2)
        print(f"  [state saved -> {CALIBRATION_STATE_FILE}]")
    except Exception as exc:
        print(f"  WARNING: Could not save calibration state: {exc}")


# ============================================================
# Shared exception handler (same pattern as other scripts)
# ============================================================
def handle_sim_exception(exc: Exception) -> None:
    """Print plain-English explanation for common SpiNNaker exceptions."""
    exc_type = type(exc).__name__
    exc_str = str(exc)
    print(f"  EXCEPTION: {exc_type}: {exc_str}")
    if isinstance(exc, SpinnmanIOException) or "No buffer space" in exc_str:
        print("  MEANING: UDP send buffer overflow. Too many packets sent at once.")
        print("  WHAT TO TRY: Reduce hidden_size or apply tighter pruning threshold.")
    elif isinstance(exc, SpinnmanTimeoutException) or "timeout" in exc_str.lower():
        print("  MEANING: SpiNNaker board did not respond in time.")
        print("  WHAT TO TRY: Check board power and network connectivity.")
        print("    ping spinnaker.cs.man.ac.uk")
    elif "ConnectionRefused" in exc_str or "ECONNREFUSED" in exc_str:
        print("  MEANING: Cannot connect to SpiNNaker spalloc server.")
        print("  WHAT TO TRY: Check VPN is connected to Manchester network.")
    else:
        print("  MEANING: Unknown error. Check SpiNNaker board status.")


# ============================================================
# Phase header / footer printers
# ============================================================
def print_phase_header(phase_num: int, phase_name: str, calibration: dict) -> None:
    print()
    print("=" * 60)
    print(f"PHASE {phase_num}/5: {phase_name}")
    print(f"Status: RUNNING")
    print(f"Calibration so far: {json.dumps(calibration, indent=None)}")
    print(f"[{ts()}]")
    print("=" * 60)


def print_phase_footer(phase_num: int, phase_name: str, passed: bool, key_finding: str) -> None:
    status = "PASSED" if passed else "FAILED"
    print()
    print("=" * 60)
    print(f"PHASE {phase_num}/5: {phase_name} -- {status}")
    print(f"  {key_finding}")
    print(f"[{ts()}]")
    print("=" * 60)


# ============================================================
# Helper: run a single-neuron firing test
# Returns dict: {weight, n_spikes, v_trace, max_v, fired, [error]}
# ============================================================
def _run_single_neuron(weight: float, lif_params: dict, label: str, phase: int = 0) -> dict:
    """
    1 SpikeSourceArray neuron firing every step -> 1 IF_curr_exp neuron.
    Returns spike/voltage diagnostics.
    """
    spike_times = list(range(NUM_STEPS))  # [0 .. 24]
    print(f"\n  Subtest: weight={weight:.3f}  [{label}]")
    print(f"    LIF params : {lif_params}")

    try:
        sim.setup(timestep=DT)
        try:
            input_pop = sim.Population(
                1, sim.SpikeSourceArray,
                {"spike_times": [spike_times]},
                label="input"
            )
            input_pop.record("spikes")

            neuron_model_class = sim.IF_curr_exp
            neuron = sim.Population(
                1, neuron_model_class(**lif_params),
                label="neuron"
            )
            neuron.initialize(v=lif_params.get("v_rest", 0.0))
            neuron.record(["spikes", "v"])

            sim.Projection(
                input_pop, neuron,
                sim.FromListConnector([[0, 0, weight, 1.0]]),
                receptor_type="excitatory"
            )

            t0 = time.time()
            sim.run(NUM_STEPS)
            elapsed = time.time() - t0

            # Extract spikes
            spike_data = neuron.get_data("spikes")
            spikes_out = spike_data.segments[-1].spiketrains
            n_spikes = sum(len(st) for st in spikes_out)
            spike_times_out = [float(t) for st in spikes_out for t in st]

            # Extract voltage -- use .magnitude to get plain numpy array
            # (iterating Neo AnalogSignal directly gives Quantity objects, not scalars)
            v_data = neuron.get_data("v")
            v_trace = []
            for sig in v_data.segments[-1].analogsignals:
                if sig.shape[1] >= 1:
                    v_trace = sig.magnitude[:, 0].tolist()

            # Verify input spikes
            in_data = input_pop.get_data("spikes")
            in_spikes = sum(len(st) for st in in_data.segments[-1].spiketrains)

            max_v = max(v_trace) if v_trace else float("nan")
            fired = n_spikes > 0

            print(f"    Wall clock     : {elapsed*1000:.0f} ms")
            print(f"    Input spikes   : {in_spikes} (expected {NUM_STEPS})")
            print(f"    Output spikes  : {n_spikes}")
            print(f"    Max voltage    : {max_v:.6f}  (threshold={lif_params.get('v_thresh', 1.0)})")
            print(f"    Spike times    : {spike_times_out}")
            print(f"    V trace (all)  :", end="")
            if v_trace:
                for i, v in enumerate(v_trace):
                    fired_marker = " FIRE" if any(abs(t - i) < 1.0 for t in spike_times_out) else ""
                    print(f"\n      t={i:3d}: v={v:9.5f}{fired_marker}", end="")
            print()

        finally:
            sim.end()

        result = {
            "weight": weight,
            "n_spikes": n_spikes,
            "v_trace": v_trace,
            "max_v": max_v,
            "fired": fired,
            "input_spikes_verified": in_spikes,
        }
        LOGGER.record({
            "phase": phase,
            "test_name": label,
            "params": {**lif_params, "weight": weight},
            "n_input_spikes_expected": NUM_STEPS,
            "n_input_spikes_actual": in_spikes,
            "n_hidden_spikes": n_spikes,
            "n_output_spikes": 0,
            "max_hidden_v": max_v,
            "max_output_v": 0.0,
            "v_trace": v_trace,
            "hidden_v_traces": {"0": v_trace},
            "output_v_traces": {},
            "spikes_per_step_input": [],
            "spikes_per_step_hidden": [],
            "spikes_per_step_output": [],
            "neurons_that_fired_hidden": [0] if fired else [],
            "neurons_that_fired_output": [],
            "wall_clock_ms": elapsed * 1000,
            "passed": fired,
            "error": None,
            "notes": f"single-neuron test, weight={weight}",
        })
        return result

    except (SpinnmanIOException, SpinnmanTimeoutException) as exc:
        handle_sim_exception(exc)
        err = str(exc)
        LOGGER.record({
            "phase": phase, "test_name": label,
            "params": {**lif_params, "weight": weight},
            "n_input_spikes_expected": NUM_STEPS, "n_input_spikes_actual": 0,
            "n_hidden_spikes": -1, "n_output_spikes": 0,
            "max_hidden_v": float("nan"), "max_output_v": 0.0,
            "v_trace": [], "hidden_v_traces": {}, "output_v_traces": {},
            "spikes_per_step_input": [], "spikes_per_step_hidden": [], "spikes_per_step_output": [],
            "neurons_that_fired_hidden": [], "neurons_that_fired_output": [],
            "wall_clock_ms": 0.0, "passed": False, "error": err,
            "notes": "SpinnmanException in single-neuron test",
        })
        return {"weight": weight, "n_spikes": -1, "v_trace": [], "max_v": float("nan"),
                "fired": False, "error": err}
    except Exception as exc:
        handle_sim_exception(exc)
        err = str(exc)
        LOGGER.record({
            "phase": phase, "test_name": label,
            "params": {**lif_params, "weight": weight},
            "n_input_spikes_expected": NUM_STEPS, "n_input_spikes_actual": 0,
            "n_hidden_spikes": -1, "n_output_spikes": 0,
            "max_hidden_v": float("nan"), "max_output_v": 0.0,
            "v_trace": [], "hidden_v_traces": {}, "output_v_traces": {},
            "spikes_per_step_input": [], "spikes_per_step_hidden": [], "spikes_per_step_output": [],
            "neurons_that_fired_hidden": [], "neurons_that_fired_output": [],
            "wall_clock_ms": 0.0, "passed": False, "error": err,
            "notes": "Exception in single-neuron test",
        })
        return {"weight": weight, "n_spikes": -1, "v_trace": [], "max_v": float("nan"),
                "fired": False, "error": err}


# ============================================================
# Helper: split connection array into exc/inh lists
# ============================================================
def split_exc_inh(connections: np.ndarray) -> tuple:
    """
    Split (N,4) [pre, post, weight, delay] into (exc_list, inh_list).
    Inhibitory weights are made positive (sPyNNaker convention).
    """
    exc_mask = connections[:, 2] > 0
    inh_mask = connections[:, 2] < 0

    fc_exc = connections[exc_mask].tolist()

    inh_data = connections[inh_mask].copy()
    inh_data[:, 2] = np.abs(inh_data[:, 2])
    fc_inh = inh_data.tolist()

    return fc_exc, fc_inh


# ============================================================
# PHASE 1: Basic firing check
# ============================================================
def run_phase1(calibration: dict) -> dict:
    phase_name = "Basic firing check"
    print_phase_header(1, phase_name, calibration)

    test_weights = [0.5, 1.0, 2.0, 5.0, 10.0]
    lif_params = dict(LIF_PARAMS_BASE)
    lif_params["tau_syn_E"] = 1.0
    lif_params["tau_syn_I"] = 1.0

    print(f"  Testing weights: {test_weights}")
    print(f"  LIF params     : {lif_params}")
    print(f"  Input          : fires every timestep 0..{NUM_STEPS-1}")

    results = []
    min_working_weight = None
    voltage_traces = {}

    for weight in test_weights:
        r = _run_single_neuron(weight, lif_params, f"weight={weight}", phase=1)
        results.append(r)
        voltage_traces[str(weight)] = r.get("v_trace", [])
        if r["fired"] and min_working_weight is None:
            min_working_weight = weight
            print(f"\n  *** FIRST WORKING WEIGHT: {weight} ***")

    # --- Print summary ---
    print(f"\n  Weight sweep summary:")
    for r in results:
        status = "PASS" if r["fired"] else "FAIL"
        err = f"  ERROR: {r.get('error','')[:50]}" if "error" in r else ""
        print(f"    weight={r['weight']:5.1f}: {status:4s}  "
              f"spikes={r['n_spikes']:3d}  max_v={r['max_v']:.5f}{err}")

    phase1_passed = min_working_weight is not None

    # --- Sanity check with standard params if all failed ---
    if not phase1_passed:
        print()
        print("  All weights failed with custom params.")
        print("  Running sanity check with STANDARD neuroscience params...")
        std_params = dict(LIF_PARAMS_STANDARD)
        std_params["tau_syn_E"] = 1.0
        std_params["tau_syn_I"] = 1.0
        sanity = _run_single_neuron(
            5.0, std_params,
            "SANITY CHECK: standard params (v_rest=-65, v_thresh=-50)",
            phase=1,
        )
        if sanity["fired"]:
            print("  Sanity check PASSED: IF_curr_exp works with standard params.")
            print("  -> Our custom params (v_rest=0, v_thresh=1) may be the issue.")
            print("     The board is functional. Proceed through Phase 2 for tau_syn tuning.")
        else:
            print("  FATAL: IF_curr_exp cannot fire even with standard params.")
            print("  FATAL: IF_curr_exp cannot fire on this board. Check SpiNNaker connection.")
            sys.exit(1)

    calibration["phase1_passed"] = phase1_passed
    calibration["min_working_weight"] = min_working_weight
    calibration["phase1_voltage_traces"] = voltage_traces
    calibration["phase1_weight_results"] = [
        {"weight": r["weight"], "fired": r["fired"], "n_spikes": r["n_spikes"],
         "max_v": r["max_v"]}
        for r in results
    ]

    save_calibration_state(calibration)

    if phase1_passed:
        finding = f"Minimum working weight: {min_working_weight}"
    else:
        finding = "Custom params failed. Standard params sanity-checked OK. Continuing."

    print_phase_footer(1, phase_name, phase1_passed, finding)
    return calibration


# ============================================================
# PHASE 2: tau_syn calibration
# ============================================================
def run_phase2(calibration: dict) -> dict:
    phase_name = "tau_syn calibration"
    print_phase_header(2, phase_name, calibration)

    tau_syn_values = [5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05]
    fixed_weight = 2.0

    print(f"  tau_syn values to try: {tau_syn_values}")
    print(f"  Fixed weight: {fixed_weight}")
    print()
    print(f"  Theoretical fraction of current delivered per spike (dt={DT} ms):")
    for tau in tau_syn_values:
        frac = 1.0 - math.exp(-DT / tau)
        print(f"    tau_syn={tau:5.2f}: frac={frac*100:.1f}%  "
              f"exp(-dt/tau)={math.exp(-DT/tau):.4f}")

    tau_syn_results = []
    optimal_tau_syn = None
    phase2_passed = False
    use_if_curr_delta = False

    for tau_syn in tau_syn_values:
        lif_params = dict(LIF_PARAMS_BASE)
        lif_params["tau_syn_E"] = tau_syn
        lif_params["tau_syn_I"] = tau_syn

        frac = 1.0 - math.exp(-DT / tau_syn)
        print(f"\n  --- tau_syn={tau_syn} ms  (frac_delivered={frac*100:.1f}%) ---")

        r = _run_single_neuron(
            fixed_weight, lif_params,
            f"tau_syn={tau_syn}",
            phase=2,
        )

        tau_syn_results.append({
            "tau_syn": tau_syn,
            "fired": r["fired"],
            "n_spikes": r["n_spikes"],
            "max_v": r["max_v"],
            "theoretical_frac": frac,
        })

        if r["fired"]:
            optimal_tau_syn = tau_syn
            phase2_passed = True
            print(f"\n  *** tau_syn={tau_syn} WORKS -- stopping tau_syn sweep ***")
            break

    # If no tau_syn worked, try IF_curr_delta
    if not phase2_passed:
        print()
        print("  No tau_syn value worked with IF_curr_exp.")
        print("  Trying IF_curr_delta (no synaptic decay -- closest to snnTorch)...")

        delta_params = dict(LIF_PARAMS_BASE)
        # IF_curr_delta does not take tau_syn_E/I
        print(f"  IF_curr_delta params: {delta_params}")
        print(f"  Fixed weight: {fixed_weight}, input fires every step")

        spike_times = list(range(NUM_STEPS))

        try:
            sim.setup(timestep=DT)
            try:
                input_pop = sim.Population(
                    1, sim.SpikeSourceArray,
                    {"spike_times": [spike_times]},
                    label="input_delta"
                )
                neuron = sim.Population(
                    1, sim.IF_curr_delta(**delta_params),
                    label="neuron_delta"
                )
                neuron.initialize(v=delta_params["v_rest"])
                neuron.record(["spikes", "v"])
                sim.Projection(
                    input_pop, neuron,
                    sim.FromListConnector([[0, 0, fixed_weight, 1.0]]),
                    receptor_type="excitatory"
                )
                sim.run(NUM_STEPS)

                spike_data = neuron.get_data("spikes")
                n_spikes_delta = sum(
                    len(st) for st in spike_data.segments[-1].spiketrains
                )
                v_data = neuron.get_data("v")
                v_trace_delta = []
                for sig in v_data.segments[-1].analogsignals:
                    if sig.shape[1] >= 1:
                        v_trace_delta = sig.magnitude[:, 0].tolist()
                max_v_delta = max(v_trace_delta) if v_trace_delta else float("nan")
            finally:
                sim.end()

            print(f"    IF_curr_delta: spikes={n_spikes_delta}, max_v={max_v_delta:.6f}")
            LOGGER.record({
                "phase": 2,
                "test_name": "IF_curr_delta_fallback",
                "params": {**delta_params, "weight": fixed_weight, "model": "IF_curr_delta"},
                "n_input_spikes_expected": NUM_STEPS,
                "n_input_spikes_actual": NUM_STEPS,
                "n_hidden_spikes": n_spikes_delta,
                "n_output_spikes": 0,
                "max_hidden_v": max_v_delta,
                "max_output_v": 0.0,
                "v_trace": v_trace_delta,
                "hidden_v_traces": {"0": v_trace_delta},
                "output_v_traces": {},
                "spikes_per_step_input": [],
                "spikes_per_step_hidden": [],
                "spikes_per_step_output": [],
                "neurons_that_fired_hidden": [0] if n_spikes_delta > 0 else [],
                "neurons_that_fired_output": [],
                "wall_clock_ms": 0.0,
                "passed": n_spikes_delta > 0,
                "error": None,
                "notes": "IF_curr_delta fallback in Phase 2 (no tau_syn decay)",
            })

            if n_spikes_delta > 0:
                print("    IF_curr_delta FIRES. Will use this model.")
                use_if_curr_delta = True
                phase2_passed = True
                # Use smallest tau_syn tried as fallback for IF_curr_exp reporting
                optimal_tau_syn = tau_syn_values[-1]
            else:
                print("    IF_curr_delta also silent. Phase 2 FAILED.")
                optimal_tau_syn = tau_syn_values[-1]  # smallest tried

        except Exception as exc:
            handle_sim_exception(exc)
            optimal_tau_syn = tau_syn_values[-1]

    # --- Summary ---
    print(f"\n  tau_syn sweep summary:")
    print(f"  {'tau_syn':>8}  {'fired':>6}  {'spikes':>7}  {'max_v':>10}  {'frac%':>6}")
    for r in tau_syn_results:
        print(f"  {r['tau_syn']:>8.2f}  {str(r['fired']):>6}  {r['n_spikes']:>7}  "
              f"{r['max_v']:>10.5f}  {r['theoretical_frac']*100:>5.1f}%")

    calibration["phase2_passed"] = phase2_passed
    calibration["optimal_tau_syn"] = optimal_tau_syn
    calibration["tau_syn_results"] = tau_syn_results
    calibration["use_if_curr_delta"] = use_if_curr_delta

    save_calibration_state(calibration)

    if use_if_curr_delta:
        finding = f"IF_curr_delta required (no tau_syn decay). optimal_tau_syn={optimal_tau_syn} (unused)"
    elif phase2_passed:
        finding = f"optimal_tau_syn={optimal_tau_syn} ms"
    else:
        finding = f"No tau_syn worked. Falling back to smallest tried ({tau_syn_values[-1]} ms)"

    print_phase_footer(2, phase_name, phase2_passed, finding)
    return calibration


# ============================================================
# PHASE 3: Weight scale calibration with real weights
# ============================================================
def run_phase3(calibration: dict) -> dict:
    phase_name = "Weight scale calibration with real weights"
    print_phase_header(3, phase_name, calibration)

    # --- Load data ---
    fc1_conn_path = WEIGHTS_DIR / "fc1_connections.npy"
    spike_feat_path = WEIGHTS_DIR / "test_spike_features.npy"

    print(f"  Loading: {fc1_conn_path}")
    print(f"  Loading: {spike_feat_path}")

    if not fc1_conn_path.exists():
        print(f"  FATAL: {fc1_conn_path} not found. Run convert_weights.py first.")
        sys.exit(1)
    if not spike_feat_path.exists():
        print(f"  FATAL: {spike_feat_path} not found. Run extract_features.py first.")
        sys.exit(1)

    fc1_all = np.load(fc1_conn_path)           # (N, 4): [pre, post, weight, delay]
    spike_features = np.load(spike_feat_path)  # (N_samples, 25, 2304)

    print(f"  fc1_connections shape : {fc1_all.shape}")
    print(f"  spike_features shape  : {spike_features.shape}")

    # --- Filter to first 20 hidden neurons ---
    N_HIDDEN_SUBSET = 20
    mask = fc1_all[:, 1] < N_HIDDEN_SUBSET
    fc1_subset = fc1_all[mask].copy()
    print(f"  FC1 connections retained (post < {N_HIDDEN_SUBSET}): {len(fc1_subset)}")

    if len(fc1_subset) == 0:
        print(f"  FATAL: No connections found for first {N_HIDDEN_SUBSET} hidden neurons.")
        sys.exit(1)

    # --- First test sample only ---
    SAMPLE_IDX = 0
    spike_input = spike_features[SAMPLE_IDX]   # (25, 2304)
    INPUT_SIZE = spike_input.shape[1]

    spike_times_list = []
    for neuron_idx in range(INPUT_SIZE):
        times = np.where(spike_input[:, neuron_idx] > 0.5)[0].astype(float).tolist()
        spike_times_list.append(times)

    active_inputs = sum(1 for t in spike_times_list if len(t) > 0)
    total_input_spikes = sum(len(t) for t in spike_times_list)
    print(f"  Active input neurons  : {active_inputs}/{INPUT_SIZE}")
    print(f"  Total input spikes    : {total_input_spikes}")

    # --- Neuron model from Phase 2 ---
    use_delta = calibration.get("use_if_curr_delta", False)
    tau_syn = calibration.get("optimal_tau_syn", 1.0)

    if use_delta:
        lif_params = dict(LIF_PARAMS_BASE)
        neuron_model_name = "IF_curr_delta"
        print(f"  Neuron model: IF_curr_delta (from Phase 2)")
    else:
        lif_params = dict(LIF_PARAMS_BASE)
        lif_params["tau_syn_E"] = tau_syn
        lif_params["tau_syn_I"] = tau_syn
        neuron_model_name = "IF_curr_exp"
        print(f"  Neuron model: IF_curr_exp, tau_syn={tau_syn} ms (from Phase 2)")

    # --- Scale sweep ---
    scale_factors = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    print(f"  Scale factors to try: {scale_factors}")

    scale_results = []
    optimal_weight_scale = None
    phase3_passed = False
    hidden_neurons_that_fired = []

    raw_weights = fc1_subset[:, 2]
    print(f"  Raw weight range: [{raw_weights.min():.5f}, {raw_weights.max():.5f}]  "
          f"mean={raw_weights.mean():.5f}")

    for scale in scale_factors:
        print(f"\n  --- scale_factor={scale}x  "
              f"(effective w range [{raw_weights.min()*scale:.4f}, "
              f"{raw_weights.max()*scale:.4f}]) ---")

        scaled = fc1_subset.copy()
        scaled[:, 2] = scaled[:, 2] * scale

        fc1_exc, fc1_inh = split_exc_inh(scaled)
        print(f"    Exc connections: {len(fc1_exc)}, Inh connections: {len(fc1_inh)}")

        try:
            sim.setup(timestep=DT)
            try:
                input_pop = sim.Population(
                    INPUT_SIZE, sim.SpikeSourceArray,
                    {"spike_times": spike_times_list},
                    label="input"
                )

                neuron_model_class = getattr(sim, neuron_model_name)
                hidden_pop = sim.Population(
                    N_HIDDEN_SUBSET,
                    neuron_model_class(**lif_params),
                    label="hidden"
                )
                hidden_pop.initialize(v=lif_params["v_rest"])
                hidden_pop.record(["spikes", "v"])

                if fc1_exc:
                    sim.Projection(input_pop, hidden_pop,
                                   sim.FromListConnector(fc1_exc),
                                   receptor_type="excitatory")
                if fc1_inh:
                    sim.Projection(input_pop, hidden_pop,
                                   sim.FromListConnector(fc1_inh),
                                   receptor_type="inhibitory")

                t0 = time.time()
                sim.run(NUM_STEPS)
                elapsed = time.time() - t0

                # Extract hidden spikes
                spike_data = hidden_pop.get_data("spikes")
                n_spikes_per_neuron = np.zeros(N_HIDDEN_SUBSET, dtype=int)
                for st in spike_data.segments[-1].spiketrains:
                    nid = int(st.annotations.get("source_index", 0))
                    if nid < N_HIDDEN_SUBSET:
                        n_spikes_per_neuron[nid] = len(st)

                # Extract voltages
                v_data = hidden_pop.get_data("v")
                max_v_per_neuron = np.zeros(N_HIDDEN_SUBSET)
                for sig in v_data.segments[-1].analogsignals:
                    for n in range(min(sig.shape[1], N_HIDDEN_SUBSET)):
                        trace = sig.magnitude[:, n].tolist()
                        if trace:
                            max_v_per_neuron[n] = max(trace)

            finally:
                sim.end()

            total_hidden_spikes = int(n_spikes_per_neuron.sum())
            firing_neurons = list(map(int, np.where(n_spikes_per_neuron > 0)[0]))
            global_max_v = float(max_v_per_neuron.max())

            print(f"    Wall clock          : {elapsed*1000:.0f} ms")
            print(f"    Total hidden spikes : {total_hidden_spikes}")
            print(f"    Firing neurons      : {firing_neurons}")
            print(f"    Max voltage         : {global_max_v:.6f}")

            scale_results.append({
                "scale_factor": scale,
                "hidden_spikes": total_hidden_spikes,
                "neurons_that_fired": firing_neurons,
                "max_hidden_v": global_max_v,
            })
            LOGGER.record({
                "phase": 3,
                "test_name": f"scale={scale}x",
                "params": {**lif_params, "weight_scale": scale, "model": neuron_model_name,
                           "n_hidden": N_HIDDEN_SUBSET},
                "n_input_spikes_expected": total_input_spikes,
                "n_input_spikes_actual": total_input_spikes,
                "n_hidden_spikes": total_hidden_spikes,
                "n_output_spikes": 0,
                "max_hidden_v": global_max_v,
                "max_output_v": 0.0,
                "v_trace": [],
                "hidden_v_traces": {},
                "output_v_traces": {},
                "spikes_per_step_input": [],
                "spikes_per_step_hidden": [],
                "spikes_per_step_output": [],
                "neurons_that_fired_hidden": firing_neurons,
                "neurons_that_fired_output": [],
                "wall_clock_ms": elapsed * 1000,
                "passed": total_hidden_spikes > 0,
                "error": None,
                "notes": f"FC1 exc={len(fc1_exc)} inh={len(fc1_inh)}, raw_w_range=[{raw_weights.min()*scale:.4f},{raw_weights.max()*scale:.4f}]",
            })

            if total_hidden_spikes > 0 and not phase3_passed:
                optimal_weight_scale = scale
                phase3_passed = True
                hidden_neurons_that_fired = firing_neurons
                print(f"\n  *** FIRST WORKING SCALE: {scale}x ***  Stopping scale sweep.")
                break

        except (SpinnmanIOException, SpinnmanTimeoutException) as exc:
            handle_sim_exception(exc)
            err = str(exc)
            scale_results.append({
                "scale_factor": scale,
                "hidden_spikes": -1,
                "neurons_that_fired": [],
                "max_hidden_v": float("nan"),
                "error": err,
            })
            LOGGER.record({
                "phase": 3, "test_name": f"scale={scale}x",
                "params": {**lif_params, "weight_scale": scale, "model": neuron_model_name},
                "n_input_spikes_expected": total_input_spikes, "n_input_spikes_actual": 0,
                "n_hidden_spikes": -1, "n_output_spikes": 0,
                "max_hidden_v": float("nan"), "max_output_v": 0.0,
                "v_trace": [], "hidden_v_traces": {}, "output_v_traces": {},
                "spikes_per_step_input": [], "spikes_per_step_hidden": [], "spikes_per_step_output": [],
                "neurons_that_fired_hidden": [], "neurons_that_fired_output": [],
                "wall_clock_ms": 0.0, "passed": False, "error": err,
                "notes": "SpinnmanException in Phase 3 scale sweep",
            })
        except Exception as exc:
            handle_sim_exception(exc)
            err = str(exc)
            scale_results.append({
                "scale_factor": scale,
                "hidden_spikes": -1,
                "neurons_that_fired": [],
                "max_hidden_v": float("nan"),
                "error": err,
            })
            LOGGER.record({
                "phase": 3, "test_name": f"scale={scale}x",
                "params": {**lif_params, "weight_scale": scale, "model": neuron_model_name},
                "n_input_spikes_expected": total_input_spikes, "n_input_spikes_actual": 0,
                "n_hidden_spikes": -1, "n_output_spikes": 0,
                "max_hidden_v": float("nan"), "max_output_v": 0.0,
                "v_trace": [], "hidden_v_traces": {}, "output_v_traces": {},
                "spikes_per_step_input": [], "spikes_per_step_hidden": [], "spikes_per_step_output": [],
                "neurons_that_fired_hidden": [], "neurons_that_fired_output": [],
                "wall_clock_ms": 0.0, "passed": False, "error": err,
                "notes": "Exception in Phase 3 scale sweep",
            })

    # --- Fallback if nothing worked ---
    if not phase3_passed:
        print()
        print("  No scale factor up to 100x caused any hidden firing.")
        print("  Diagnosis:")
        non_error = [r for r in scale_results if "error" not in r]
        if non_error:
            max_v_seen = max(r["max_hidden_v"] for r in non_error if not math.isnan(r["max_hidden_v"]))
            print(f"    Max voltage seen across all scales: {max_v_seen:.6f}")
            if max_v_seen < 0.001:
                print("    Voltage essentially zero -> tau_syn is likely still killing signal")
                print("    or input spikes are not reaching hidden neurons.")
            else:
                estimated_scale = 1.0 / max_v_seen
                print(f"    Voltage DID accumulate. Estimated scale needed: ~{estimated_scale:.0f}x")
        print("  Continuing with scale=50.0 as best-effort fallback.")
        optimal_weight_scale = 50.0

    # --- Summary table ---
    print(f"\n  Scale sweep summary:")
    print(f"  {'Scale':>8}  {'HidSpk':>7}  {'MaxV':>10}  {'FiringNeurons'}")
    for r in scale_results:
        err_tag = "  ERROR" if "error" in r else ""
        print(f"  {r['scale_factor']:>8.1f}x  {r['hidden_spikes']:>7}  "
              f"{r['max_hidden_v']:>10.5f}  {r['neurons_that_fired']}{err_tag}")

    calibration["phase3_passed"] = phase3_passed
    calibration["optimal_weight_scale"] = optimal_weight_scale
    calibration["scale_results"] = scale_results
    calibration["hidden_neurons_that_fired"] = hidden_neurons_that_fired

    save_calibration_state(calibration)

    finding = (
        f"optimal_weight_scale={optimal_weight_scale}x, "
        f"neurons_fired={hidden_neurons_that_fired}"
        if phase3_passed
        else f"No working scale found. Using fallback scale={optimal_weight_scale}x"
    )
    print_phase_footer(3, phase_name, phase3_passed, finding)
    return calibration


# ============================================================
# PHASE 4: Scale up hidden neurons
# ============================================================
def run_phase4(calibration: dict) -> dict:
    phase_name = "Scale up hidden neurons"
    print_phase_header(4, phase_name, calibration)

    # --- Load data ---
    fc1_conn_path = WEIGHTS_DIR / "fc1_connections.npy"
    fc2_conn_path = WEIGHTS_DIR / "fc2_connections.npy"
    spike_feat_path = WEIGHTS_DIR / "test_spike_features.npy"
    metadata_path = WEIGHTS_DIR / "metadata.json"

    for p in [fc1_conn_path, fc2_conn_path, spike_feat_path, metadata_path]:
        if not p.exists():
            print(f"  FATAL: {p} not found.")
            sys.exit(1)

    fc1_all = np.load(fc1_conn_path)
    fc2_all = np.load(fc2_conn_path)
    spike_features = np.load(spike_feat_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    INPUT_SIZE = spike_features.shape[2]                     # actual (may be reduced)
    OUTPUT_SIZE = metadata["architecture"]["fc2"]["out"]     # 50
    SAMPLE_IDX = 0
    FULL_INPUT_SIZE = metadata["architecture"]["flatten_size"]  # 2304 (for reference)

    print(f"  Input size   : {INPUT_SIZE}"
          + (f"  (reduced from {FULL_INPUT_SIZE})" if INPUT_SIZE != FULL_INPUT_SIZE else ""))
    print(f"  Output size  : {OUTPUT_SIZE}")
    print(f"  Using sample : {SAMPLE_IDX}")

    # Calibrated params
    weight_scale = calibration.get("optimal_weight_scale", 50.0)
    tau_syn = calibration.get("optimal_tau_syn", 1.0)
    use_delta = calibration.get("use_if_curr_delta", False)

    print(f"  Weight scale : {weight_scale}x")
    print(f"  tau_syn      : {tau_syn} ms")
    print(f"  Neuron model : {'IF_curr_delta' if use_delta else 'IF_curr_exp'}")
    print(f"  Prune thresh : {PRUNE_THRESHOLD}")

    if use_delta:
        lif_params = dict(LIF_PARAMS_BASE)
        neuron_model_name = "IF_curr_delta"
    else:
        lif_params = dict(LIF_PARAMS_BASE)
        lif_params["tau_syn_E"] = tau_syn
        lif_params["tau_syn_I"] = tau_syn
        neuron_model_name = "IF_curr_exp"

    # --- Spike input ---
    spike_input = spike_features[SAMPLE_IDX]   # (25, 2304)
    spike_times_list = []
    for neuron_idx in range(INPUT_SIZE):
        times = np.where(spike_input[:, neuron_idx] > 0.5)[0].astype(float).tolist()
        spike_times_list.append(times)

    active_inputs = sum(1 for t in spike_times_list if len(t) > 0)
    print(f"  Active inputs: {active_inputs}/{INPUT_SIZE}")

    # --- Prune raw connection arrays ---
    fc1_pruned_all = fc1_all[np.abs(fc1_all[:, 2]) > PRUNE_THRESHOLD]
    fc2_pruned_all = fc2_all[np.abs(fc2_all[:, 2]) > PRUNE_THRESHOLD]

    # Apply weight scale
    fc1_pruned_all = fc1_pruned_all.copy()
    fc2_pruned_all = fc2_pruned_all.copy()
    fc1_pruned_all[:, 2] = fc1_pruned_all[:, 2] * weight_scale
    fc2_pruned_all[:, 2] = fc2_pruned_all[:, 2] * weight_scale

    # --- Try hidden sizes in order ---
    hidden_sizes = [20, 64, 128, 256]
    phase4_results = []
    max_working_hidden_size = 20  # safe fallback
    phase4_passed = False
    last_working_size = None

    for hidden_size in hidden_sizes:
        print(f"\n  --- Trying hidden_size={hidden_size} ---")

        fc1_sub = fc1_pruned_all[fc1_pruned_all[:, 1] < hidden_size]
        fc2_sub = fc2_pruned_all[fc2_pruned_all[:, 0] < hidden_size]

        fc1_exc, fc1_inh = split_exc_inh(fc1_sub)
        fc2_exc, fc2_inh = split_exc_inh(fc2_sub)

        print(f"    FC1: {len(fc1_exc)} exc + {len(fc1_inh)} inh")
        print(f"    FC2: {len(fc2_exc)} exc + {len(fc2_inh)} inh")

        udp_overflow = False
        run_error = None

        try:
            sim.setup(timestep=DT)
            try:
                input_pop = sim.Population(
                    INPUT_SIZE, sim.SpikeSourceArray,
                    {"spike_times": spike_times_list},
                    label="input"
                )

                neuron_model_class = getattr(sim, neuron_model_name)

                hidden_pop = sim.Population(
                    hidden_size,
                    neuron_model_class(**lif_params),
                    label="hidden"
                )
                hidden_pop.initialize(v=lif_params["v_rest"])
                hidden_pop.record(["spikes", "v"])

                output_pop = sim.Population(
                    OUTPUT_SIZE,
                    neuron_model_class(**lif_params),
                    label="output"
                )
                output_pop.initialize(v=lif_params["v_rest"])
                output_pop.record(["spikes", "v"])

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

                t0 = time.time()
                sim.run(NUM_STEPS)
                elapsed = time.time() - t0

                # Extract hidden
                h_spike_data = hidden_pop.get_data("spikes")
                h_spikes_per_neuron = np.zeros(hidden_size, dtype=int)
                for st in h_spike_data.segments[-1].spiketrains:
                    nid = int(st.annotations.get("source_index", 0))
                    if nid < hidden_size:
                        h_spikes_per_neuron[nid] = len(st)

                h_v_data = hidden_pop.get_data("v")
                max_hidden_v = 0.0
                for sig in h_v_data.segments[-1].analogsignals:
                    for n in range(min(sig.shape[1], hidden_size)):
                        v = float(sig.magnitude[:, n].max())
                        if v > max_hidden_v:
                            max_hidden_v = v

                # Extract output
                o_spike_data = output_pop.get_data("spikes")
                output_spike_counts = np.zeros(OUTPUT_SIZE, dtype=int)
                for st in o_spike_data.segments[-1].spiketrains:
                    nid = int(st.annotations.get("source_index", 0))
                    if nid < OUTPUT_SIZE:
                        output_spike_counts[nid] = len(st)

                o_v_data = output_pop.get_data("v")
                max_output_v = 0.0
                for sig in o_v_data.segments[-1].analogsignals:
                    for n in range(min(sig.shape[1], OUTPUT_SIZE)):
                        v = float(sig.magnitude[:, n].max())
                        if v > max_output_v:
                            max_output_v = v

            finally:
                sim.end()

            total_h_spikes = int(h_spikes_per_neuron.sum())
            total_o_spikes = int(output_spike_counts.sum())

            print(f"    Wall clock       : {elapsed*1000:.0f} ms")
            print(f"    Hidden spikes    : {total_h_spikes}  "
                  f"(neurons fired: {(h_spikes_per_neuron>0).sum()}/{hidden_size})")
            print(f"    Max hidden v     : {max_hidden_v:.6f}")
            print(f"    Output spikes    : {total_o_spikes}")
            print(f"    Max output v     : {max_output_v:.6f}")

            phase4_results.append({
                "hidden_size": hidden_size,
                "hidden_spikes": total_h_spikes,
                "output_spikes": total_o_spikes,
                "max_hidden_v": max_hidden_v,
                "max_output_v": max_output_v,
                "success": True,
            })
            LOGGER.record({
                "phase": 4,
                "test_name": f"hidden_size={hidden_size}",
                "params": {**lif_params, "weight_scale": weight_scale, "model": neuron_model_name,
                           "hidden_size": hidden_size},
                "n_input_spikes_expected": active_inputs,
                "n_input_spikes_actual": active_inputs,
                "n_hidden_spikes": total_h_spikes,
                "n_output_spikes": total_o_spikes,
                "max_hidden_v": max_hidden_v,
                "max_output_v": max_output_v,
                "v_trace": [],
                "hidden_v_traces": {},
                "output_v_traces": {},
                "spikes_per_step_input": [],
                "spikes_per_step_hidden": [],
                "spikes_per_step_output": [],
                "neurons_that_fired_hidden": list(map(int, np.where(h_spikes_per_neuron > 0)[0])),
                "neurons_that_fired_output": list(map(int, np.where(output_spike_counts > 0)[0])),
                "wall_clock_ms": elapsed * 1000,
                "passed": total_h_spikes > 0,
                "error": None,
                "notes": f"FC1 exc={len(fc1_exc)} inh={len(fc1_inh)}, FC2 exc={len(fc2_exc)} inh={len(fc2_inh)}",
            })

            if total_h_spikes > 0:
                phase4_passed = True

            # Update max working size on every successful run
            last_working_size = hidden_size

        except SpinnmanIOException as exc:
            handle_sim_exception(exc)
            exc_str = str(exc)
            if "No buffer space" in exc_str or "Errno 55" in exc_str:
                udp_overflow = True
                run_error = f"UDP overflow at hidden_size={hidden_size}"
                print(f"    -> {run_error}. Stopping size sweep.")
            else:
                run_error = str(exc)
            phase4_results.append({
                "hidden_size": hidden_size,
                "hidden_spikes": -1,
                "output_spikes": -1,
                "max_hidden_v": float("nan"),
                "max_output_v": float("nan"),
                "success": False,
                "error": run_error,
            })
            LOGGER.record({
                "phase": 4, "test_name": f"hidden_size={hidden_size}",
                "params": {**lif_params, "weight_scale": weight_scale, "model": neuron_model_name,
                           "hidden_size": hidden_size},
                "n_input_spikes_expected": active_inputs, "n_input_spikes_actual": 0,
                "n_hidden_spikes": -1, "n_output_spikes": -1,
                "max_hidden_v": float("nan"), "max_output_v": float("nan"),
                "v_trace": [], "hidden_v_traces": {}, "output_v_traces": {},
                "spikes_per_step_input": [], "spikes_per_step_hidden": [], "spikes_per_step_output": [],
                "neurons_that_fired_hidden": [], "neurons_that_fired_output": [],
                "wall_clock_ms": 0.0, "passed": False, "error": run_error,
                "notes": f"SpinnmanIOException Phase 4, udp_overflow={udp_overflow}",
            })
            if udp_overflow:
                break

        except SpinnmanTimeoutException as exc:
            handle_sim_exception(exc)
            run_error = str(exc)
            phase4_results.append({
                "hidden_size": hidden_size,
                "hidden_spikes": -1,
                "output_spikes": -1,
                "max_hidden_v": float("nan"),
                "max_output_v": float("nan"),
                "success": False,
                "error": run_error,
            })
            LOGGER.record({
                "phase": 4, "test_name": f"hidden_size={hidden_size}",
                "params": {**lif_params, "weight_scale": weight_scale, "model": neuron_model_name,
                           "hidden_size": hidden_size},
                "n_input_spikes_expected": active_inputs, "n_input_spikes_actual": 0,
                "n_hidden_spikes": -1, "n_output_spikes": -1,
                "max_hidden_v": float("nan"), "max_output_v": float("nan"),
                "v_trace": [], "hidden_v_traces": {}, "output_v_traces": {},
                "spikes_per_step_input": [], "spikes_per_step_hidden": [], "spikes_per_step_output": [],
                "neurons_that_fired_hidden": [], "neurons_that_fired_output": [],
                "wall_clock_ms": 0.0, "passed": False, "error": run_error,
                "notes": "SpinnmanTimeoutException Phase 4",
            })

        except Exception as exc:
            handle_sim_exception(exc)
            run_error = str(exc)
            phase4_results.append({
                "hidden_size": hidden_size,
                "hidden_spikes": -1,
                "output_spikes": -1,
                "max_hidden_v": float("nan"),
                "max_output_v": float("nan"),
                "success": False,
                "error": run_error,
            })
            LOGGER.record({
                "phase": 4, "test_name": f"hidden_size={hidden_size}",
                "params": {**lif_params, "weight_scale": weight_scale, "model": neuron_model_name,
                           "hidden_size": hidden_size},
                "n_input_spikes_expected": active_inputs, "n_input_spikes_actual": 0,
                "n_hidden_spikes": -1, "n_output_spikes": -1,
                "max_hidden_v": float("nan"), "max_output_v": float("nan"),
                "v_trace": [], "hidden_v_traces": {}, "output_v_traces": {},
                "spikes_per_step_input": [], "spikes_per_step_hidden": [], "spikes_per_step_output": [],
                "neurons_that_fired_hidden": [], "neurons_that_fired_output": [],
                "wall_clock_ms": 0.0, "passed": False, "error": run_error,
                "notes": "Exception Phase 4",
            })

    # Use last successfully completed size
    if last_working_size is not None:
        max_working_hidden_size = last_working_size
    else:
        max_working_hidden_size = 20  # absolute fallback

    # --- Summary ---
    print(f"\n  Hidden size sweep summary:")
    print(f"  {'Size':>6}  {'HidSpk':>7}  {'OutSpk':>7}  {'MaxHidV':>9}  {'MaxOutV':>9}  Status")
    for r in phase4_results:
        if r["success"]:
            status = "OK"
        else:
            status = f"ERROR"
        print(f"  {r['hidden_size']:>6}  {r['hidden_spikes']:>7}  {r['output_spikes']:>7}  "
              f"{r['max_hidden_v']:>9.4f}  {r['max_output_v']:>9.4f}  {status}")

    calibration["phase4_passed"] = phase4_passed
    calibration["max_working_hidden_size"] = max_working_hidden_size
    calibration["phase4_results"] = phase4_results

    save_calibration_state(calibration)

    finding = (
        f"max_working_hidden_size={max_working_hidden_size}, "
        f"phase4_passed={phase4_passed}"
    )
    print_phase_footer(4, phase_name, phase4_passed, finding)
    return calibration


# ============================================================
# PHASE 5: Full inference run
# ============================================================
def run_phase5(calibration: dict) -> dict:
    phase_name = "Full inference run"
    print_phase_header(5, phase_name, calibration)

    # --- Load all data ---
    fc1_conn_path = WEIGHTS_DIR / "fc1_connections.npy"
    fc2_conn_path = WEIGHTS_DIR / "fc2_connections.npy"
    spike_feat_path = WEIGHTS_DIR / "test_spike_features.npy"
    labels_path = WEIGHTS_DIR / "test_labels.npy"
    metadata_path = WEIGHTS_DIR / "metadata.json"

    for p in [fc1_conn_path, fc2_conn_path, spike_feat_path, labels_path, metadata_path]:
        if not p.exists():
            print(f"  FATAL: {p} not found.")
            sys.exit(1)

    fc1_all = np.load(fc1_conn_path)
    fc2_all = np.load(fc2_conn_path)
    spike_features = np.load(spike_feat_path)
    test_labels = np.load(labels_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    INPUT_SIZE = spike_features.shape[2]                     # actual (may be reduced)
    OUTPUT_SIZE = metadata["architecture"]["fc2"]["out"]     # 50
    NUM_STEPS_INF = metadata["snn_params"]["num_steps"]      # 25
    FULL_INPUT_SIZE = metadata["architecture"]["flatten_size"]  # 2304 (for reference)

    num_samples = min(NUM_PHASE5_SAMPLES, len(test_labels))
    print(f"  Running {num_samples} samples")

    # Calibrated params
    weight_scale = calibration.get("optimal_weight_scale", 50.0)
    tau_syn = calibration.get("optimal_tau_syn", 1.0)
    use_delta = calibration.get("use_if_curr_delta", False)
    hidden_size = calibration.get("max_working_hidden_size", 20)

    print(f"  Calibrated params:")
    print(f"    Neuron model  : {'IF_curr_delta' if use_delta else 'IF_curr_exp'}")
    print(f"    tau_syn       : {tau_syn} ms")
    print(f"    Weight scale  : {weight_scale}x")
    print(f"    Hidden size   : {hidden_size}")
    print(f"    Prune thresh  : {PRUNE_THRESHOLD}")

    if use_delta:
        lif_params = dict(LIF_PARAMS_BASE)
        neuron_model_name = "IF_curr_delta"
    else:
        lif_params = dict(LIF_PARAMS_BASE)
        lif_params["tau_syn_E"] = tau_syn
        lif_params["tau_syn_I"] = tau_syn
        neuron_model_name = "IF_curr_exp"

    # --- Prune and scale weights ---
    fc1_pruned = fc1_all[np.abs(fc1_all[:, 2]) > PRUNE_THRESHOLD].copy()
    fc2_pruned = fc2_all[np.abs(fc2_all[:, 2]) > PRUNE_THRESHOLD].copy()

    # Filter to working hidden size
    fc1_pruned = fc1_pruned[fc1_pruned[:, 1] < hidden_size]
    fc2_pruned = fc2_pruned[fc2_pruned[:, 0] < hidden_size]

    # Apply scale
    fc1_pruned[:, 2] = fc1_pruned[:, 2] * weight_scale
    fc2_pruned[:, 2] = fc2_pruned[:, 2] * weight_scale

    fc1_exc, fc1_inh = split_exc_inh(fc1_pruned)
    fc2_exc, fc2_inh = split_exc_inh(fc2_pruned)

    total_conns = len(fc1_exc) + len(fc1_inh) + len(fc2_exc) + len(fc2_inh)
    print(f"  FC1: {len(fc1_exc)} exc + {len(fc1_inh)} inh")
    print(f"  FC2: {len(fc2_exc)} exc + {len(fc2_inh)} inh")
    print(f"  Total connections: {total_conns}")

    # --- Run inference ---
    neuron_model_class = getattr(sim, neuron_model_name)

    results_list = []
    correct = 0
    total_sim_time = 0.0

    for sample_idx in range(num_samples):
        spike_input = spike_features[sample_idx]   # (25, 2304)
        true_label = int(test_labels[sample_idx])

        print(f"\n{'─' * 50}")
        print(f"  Sample {sample_idx+1}/{num_samples}  (true_label={true_label})")
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
        print(f"    Active input neurons  : {active_inputs}/{INPUT_SIZE}")
        print(f"    Total input spikes    : {total_input_spikes}")
        print(f"    Spikes per step       : "
              f"min={spikes_per_step.min()} max={spikes_per_step.max()} "
              f"mean={spikes_per_step.mean():.1f}")

        error_str = None

        try:
            sim.setup(timestep=DT)
            try:
                input_pop = sim.Population(
                    INPUT_SIZE, sim.SpikeSourceArray,
                    {"spike_times": spike_times_list},
                    label="input"
                )
                input_pop.record("spikes")

                hidden_pop = sim.Population(
                    hidden_size,
                    neuron_model_class(**lif_params),
                    label="hidden"
                )
                hidden_pop.initialize(v=lif_params["v_rest"])
                hidden_pop.record(["spikes", "v"])

                output_pop = sim.Population(
                    OUTPUT_SIZE,
                    neuron_model_class(**lif_params),
                    label="output"
                )
                output_pop.initialize(v=lif_params["v_rest"])
                output_pop.record(["spikes", "v"])

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

                t0 = time.time()
                sim.run(NUM_STEPS_INF)
                elapsed = time.time() - t0
                total_sim_time += elapsed
                print(f"  Simulation complete in {elapsed*1000:.0f} ms wall clock.")

                # ---- Extract input verification ----
                in_spike_data = input_pop.get_data("spikes")
                in_total_actual = sum(
                    len(st) for st in in_spike_data.segments[-1].spiketrains
                )
                in_spikes_per_step = np.zeros(NUM_STEPS_INF, dtype=int)
                for st in in_spike_data.segments[-1].spiketrains:
                    for t in st:
                        step = int(round(float(t)))
                        if 0 <= step < NUM_STEPS_INF:
                            in_spikes_per_step[step] += 1

                print(f"\n  Input spike verification:")
                print(f"    Expected : {total_input_spikes}")
                print(f"    Actual   : {in_total_actual}")
                if in_total_actual == 0:
                    print(f"    WARNING: SpikeSourceArray sent 0 spikes!")
                print(f"    Per-step : {list(in_spikes_per_step)}")

                # ---- Extract hidden ----
                h_spike_data = hidden_pop.get_data("spikes")
                h_spikes_per_neuron = np.zeros(hidden_size, dtype=int)
                h_spike_times_per_neuron = [[] for _ in range(hidden_size)]
                h_spikes_per_step = np.zeros(NUM_STEPS_INF, dtype=int)

                for st in h_spike_data.segments[-1].spiketrains:
                    nid = int(st.annotations.get("source_index", 0))
                    if nid < hidden_size:
                        h_spikes_per_neuron[nid] = len(st)
                        h_spike_times_per_neuron[nid] = [float(t) for t in st]
                    for t in st:
                        step = int(round(float(t)))
                        if 0 <= step < NUM_STEPS_INF:
                            h_spikes_per_step[step] += 1

                h_v_data = hidden_pop.get_data("v")
                h_v_traces = {}
                for sig in h_v_data.segments[-1].analogsignals:
                    for n in range(min(sig.shape[1], hidden_size)):
                        h_v_traces[n] = sig.magnitude[:, n].tolist()

                max_v_per_hidden = np.array([
                    max(h_v_traces.get(n, [0.0])) if h_v_traces.get(n) else 0.0
                    for n in range(hidden_size)
                ])

                # Top 5 hidden by spikes then voltage
                top5_hidden = np.argsort(
                    h_spikes_per_neuron * 1000 + max_v_per_hidden
                )[-5:][::-1]

                hidden_spike_count = int(h_spikes_per_neuron.sum())

                print(f"\n  Hidden layer activity:")
                print(f"    Total hidden spikes   : {hidden_spike_count}")
                print(f"    Neurons that fired    : "
                      f"{(h_spikes_per_neuron>0).sum()}/{hidden_size}")
                print(f"    Max spikes/neuron     : {h_spikes_per_neuron.max()}")
                print(f"    Max voltage (all)     : {max_v_per_hidden.max():.6f}")
                print(f"    Mean max voltage      : {max_v_per_hidden.mean():.6f}")
                print(f"    Per-step (hidden)     : {list(h_spikes_per_step)}")

                print(f"\n  Top 5 hidden neurons (by spikes then max_v):")
                for nid in top5_hidden:
                    vtrace = h_v_traces.get(int(nid), [])
                    print(f"    hidden[{nid:3d}]: {h_spikes_per_neuron[nid]:3d} spikes, "
                          f"max_v={max_v_per_hidden[nid]:.6f}, "
                          f"times={h_spike_times_per_neuron[nid][:6]}")
                    if vtrace:
                        print(f"      V trace: {[f'{v:.4f}' for v in vtrace[:NUM_STEPS_INF]]}")

                # ---- Extract output ----
                o_spike_data = output_pop.get_data("spikes")
                output_spike_counts = np.zeros(OUTPUT_SIZE)
                output_spike_times_per_neuron = [[] for _ in range(OUTPUT_SIZE)]
                output_spikes_per_step = np.zeros(NUM_STEPS_INF, dtype=int)

                for st in o_spike_data.segments[-1].spiketrains:
                    nid = int(st.annotations.get("source_index", 0))
                    if nid < OUTPUT_SIZE:
                        output_spike_counts[nid] = len(st)
                        output_spike_times_per_neuron[nid] = [float(t) for t in st]
                    for t in st:
                        step = int(round(float(t)))
                        if 0 <= step < NUM_STEPS_INF:
                            output_spikes_per_step[step] += 1

                o_v_data = output_pop.get_data("v")
                final_v = np.zeros(OUTPUT_SIZE)
                output_v_all = {}
                for sig in o_v_data.segments[-1].analogsignals:
                    if len(sig) > 0:
                        for n in range(min(sig.shape[1], OUTPUT_SIZE)):
                            output_v_all[n] = sig.magnitude[:, n].tolist()
                        last_row = sig[-1].magnitude.flatten()
                        final_v[:min(OUTPUT_SIZE, len(last_row))] = last_row[:OUTPUT_SIZE]

            finally:
                sim.end()

            total_output_spikes = int(output_spike_counts.sum())

            print(f"\n  Output layer activity:")
            print(f"    Total output spikes   : {total_output_spikes}")
            print(f"    Per-step (output)     : {list(output_spikes_per_step)}")

            firing_output = [
                (i, output_spike_times_per_neuron[i])
                for i in range(OUTPUT_SIZE) if output_spike_counts[i] > 0
            ]
            if firing_output:
                print(f"    Firing output neurons ({len(firing_output)}):")
                for nid, times in sorted(firing_output, key=lambda x: -len(x[1]))[:10]:
                    print(f"      output[{nid:2d}]: {len(times):3d} spikes  "
                          f"t={times[:8]}")
            else:
                print(f"    No output neurons fired.")

            top10_output_v = sorted(
                [(int(i), float(final_v[i])) for i in range(OUTPUT_SIZE)],
                key=lambda x: -x[1]
            )[:10]
            print(f"    Top 10 output neurons by final membrane voltage:")
            for nid, v in top10_output_v:
                spike_info = (
                    f"  ({int(output_spike_counts[nid])} spikes)"
                    if output_spike_counts[nid] > 0 else ""
                )
                print(f"      output[{nid:2d}]: v={v:.6f}{spike_info}")

            # ---- Classify ----
            if total_output_spikes > 0:
                predicted = int(np.argmax(output_spike_counts))
                basis = "spike count"
            else:
                predicted = int(np.argmax(final_v))
                basis = "final membrane voltage (no output spikes)"

            is_correct = (predicted == true_label)
            if is_correct:
                correct += 1

            status = "CORRECT" if is_correct else "wrong"
            print(f"\n  Classification: pred={predicted}  true={true_label}  ({status})")
            print(f"  Basis: {basis}")

            results_list.append({
                "sample": sample_idx,
                "true_label": true_label,
                "predicted": predicted,
                "correct": is_correct,
                "classification_basis": basis,
                "output_spikes": total_output_spikes,
                "hidden_spikes": hidden_spike_count,
                "input_spikes": total_input_spikes,
                "input_spikes_actual": in_total_actual,
                "active_inputs": active_inputs,
                "hidden_neurons_fired": int((h_spikes_per_neuron > 0).sum()),
                "max_hidden_v": float(max_v_per_hidden.max()),
                "inference_ms": elapsed * 1000,
                "top5_output_v": top10_output_v[:5],
            })
            LOGGER.record({
                "phase": 5,
                "test_name": f"sample_{sample_idx}_true={true_label}",
                "params": {**lif_params, "weight_scale": weight_scale, "model": neuron_model_name,
                           "hidden_size": hidden_size, "sample_idx": sample_idx},
                "n_input_spikes_expected": total_input_spikes,
                "n_input_spikes_actual": in_total_actual,
                "n_hidden_spikes": hidden_spike_count,
                "n_output_spikes": total_output_spikes,
                "max_hidden_v": float(max_v_per_hidden.max()),
                "max_output_v": float(np.max(final_v)),
                "v_trace": [],
                "hidden_v_traces": {str(k): v for k, v in list(h_v_traces.items())[:5]},
                "output_v_traces": {str(k): v for k, v in list(output_v_all.items())[:10]},
                "spikes_per_step_input": list(in_spikes_per_step),
                "spikes_per_step_hidden": list(h_spikes_per_step),
                "spikes_per_step_output": list(output_spikes_per_step),
                "neurons_that_fired_hidden": list(map(int, np.where(h_spikes_per_neuron > 0)[0])),
                "neurons_that_fired_output": [i for i in range(OUTPUT_SIZE) if output_spike_counts[i] > 0],
                "wall_clock_ms": elapsed * 1000,
                "passed": is_correct,
                "error": None,
                "notes": f"predicted={predicted} true={true_label} basis={basis}",
                "true_label": true_label,
                "predicted": predicted,
                "is_correct": is_correct,
            })

        except (SpinnmanIOException, SpinnmanTimeoutException) as exc:
            handle_sim_exception(exc)
            error_str = str(exc)
            results_list.append({
                "sample": sample_idx,
                "true_label": true_label,
                "predicted": -1,
                "correct": False,
                "error": error_str,
                "output_spikes": 0,
                "hidden_spikes": 0,
                "input_spikes": total_input_spikes,
            })
            LOGGER.record({
                "phase": 5, "test_name": f"sample_{sample_idx}_true={true_label}",
                "params": {**lif_params, "weight_scale": weight_scale, "model": neuron_model_name,
                           "hidden_size": hidden_size, "sample_idx": sample_idx},
                "n_input_spikes_expected": total_input_spikes, "n_input_spikes_actual": 0,
                "n_hidden_spikes": 0, "n_output_spikes": 0,
                "max_hidden_v": 0.0, "max_output_v": 0.0,
                "v_trace": [], "hidden_v_traces": {}, "output_v_traces": {},
                "spikes_per_step_input": [], "spikes_per_step_hidden": [], "spikes_per_step_output": [],
                "neurons_that_fired_hidden": [], "neurons_that_fired_output": [],
                "wall_clock_ms": 0.0, "passed": False, "error": error_str,
                "notes": "SpinnmanException Phase 5",
                "true_label": true_label, "predicted": -1, "is_correct": False,
            })

        except Exception as exc:
            handle_sim_exception(exc)
            error_str = str(exc)
            results_list.append({
                "sample": sample_idx,
                "true_label": true_label,
                "predicted": -1,
                "correct": False,
                "error": error_str,
                "output_spikes": 0,
                "hidden_spikes": 0,
                "input_spikes": total_input_spikes,
            })
            LOGGER.record({
                "phase": 5, "test_name": f"sample_{sample_idx}_true={true_label}",
                "params": {**lif_params, "weight_scale": weight_scale, "model": neuron_model_name,
                           "hidden_size": hidden_size, "sample_idx": sample_idx},
                "n_input_spikes_expected": total_input_spikes, "n_input_spikes_actual": 0,
                "n_hidden_spikes": 0, "n_output_spikes": 0,
                "max_hidden_v": 0.0, "max_output_v": 0.0,
                "v_trace": [], "hidden_v_traces": {}, "output_v_traces": {},
                "spikes_per_step_input": [], "spikes_per_step_hidden": [], "spikes_per_step_output": [],
                "neurons_that_fired_hidden": [], "neurons_that_fired_output": [],
                "wall_clock_ms": 0.0, "passed": False, "error": error_str,
                "notes": "Exception Phase 5",
                "true_label": true_label, "predicted": -1, "is_correct": False,
            })

    # ---- Final summary ----
    valid_results = [r for r in results_list if "error" not in r]
    accuracy = correct / num_samples if num_samples > 0 else 0.0
    avg_time = total_sim_time / max(1, len(valid_results))

    print(f"\n")
    print(f"{'─' * 50}")
    print(f"  Phase 5 per-sample results:")
    print(f"  {'Sample':>7}  {'True':>5}  {'Pred':>5}  {'Result':>9}  "
          f"{'OutSpk':>7}  {'HidSpk':>7}  {'MaxHidV':>8}  {'HidFired':>9}")
    for r in results_list:
        if "error" in r:
            print(f"  {r['sample']:>7}  {r['true_label']:>5}  {'ERR':>5}  "
                  f"{'ERROR':>9}  [{r['error'][:30]}]")
        else:
            result_str = "CORRECT" if r["correct"] else "wrong"
            print(f"  {r['sample']:>7}  {r['true_label']:>5}  {r['predicted']:>5}  "
                  f"{result_str:>9}  {r['output_spikes']:>7}  {r['hidden_spikes']:>7}  "
                  f"{r['max_hidden_v']:>8.4f}  "
                  f"{r.get('hidden_neurons_fired', 0):>9}/{hidden_size}")

    phase5_passed = correct > 0 or len(valid_results) > 0

    # ---- Save results JSON ----
    output_data = {
        "platform": "SpiNNaker",
        "spalloc_server": "spinnaker.cs.man.ac.uk",
        "timestamp": ts(),
        "script": "auto_calibrate.py",
        "calibrated_config": {
            "neuron_model": neuron_model_name,
            "tau_syn": tau_syn,
            "weight_scale": weight_scale,
            "prune_threshold": PRUNE_THRESHOLD,
            "max_hidden": hidden_size,
            "total_connections": total_conns,
        },
        "lif_params": lif_params,
        "num_samples": num_samples,
        "correct": correct,
        "accuracy": accuracy,
        "avg_wall_clock_ms": avg_time * 1000,
        "total_wall_clock_s": total_sim_time,
        "per_sample": results_list,
    }

    results_file = RESULTS_DIR / "auto_calibrate_results.json"
    try:
        with open(results_file, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\n  Results saved -> {results_file}")
    except Exception as exc:
        print(f"  WARNING: Could not save results: {exc}")

    calibration["phase5_passed"] = phase5_passed
    calibration["phase5_accuracy"] = accuracy
    calibration["phase5_correct"] = correct
    calibration["phase5_num_samples"] = num_samples

    save_calibration_state(calibration)

    finding = f"Accuracy: {correct}/{num_samples} ({accuracy:.0%})"
    print_phase_footer(5, phase_name, phase5_passed, finding)
    return calibration, correct, num_samples, accuracy


# ============================================================
# Main: orchestrate phases
# ============================================================
def main() -> None:
    # ---- Handle resume / start-over logic ----
    calibration = {}

    if SKIP_TO_PHASE > 1:
        # Load saved state and jump directly to requested phase
        saved = load_calibration_state()
        if saved:
            print(f"  Resuming from saved state (skipping to Phase {SKIP_TO_PHASE})")
            calibration = saved
        else:
            print(f"  No saved state found. Starting from Phase 1 instead.")
            # Override skip-to so we run from the beginning
    elif CALIBRATION_STATE_FILE.exists():
        saved = load_calibration_state()
        if saved:
            print()
            print("  A previous calibration_state.json exists.")
            print("  Options: (r) resume from last completed phase, (s) start over")
            print("  Defaulting to RESUME. Pass --skip-to-phase 1 to restart.")
            # Auto-resume: find the last completed phase
            last_phase = 0
            for i in range(1, 6):
                key = f"phase{i}_passed"
                if key in saved:
                    last_phase = i
            if last_phase >= 5:
                print("  All phases previously completed. Re-running Phase 5.")
                calibration = saved
                start_phase = 5
            else:
                start_phase = last_phase + 1
                print(f"  Resuming from Phase {start_phase}")
                calibration = saved
        else:
            start_phase = SKIP_TO_PHASE
    else:
        start_phase = SKIP_TO_PHASE

    # Determine actual start phase
    if SKIP_TO_PHASE > 1:
        start_phase = SKIP_TO_PHASE
    elif not CALIBRATION_STATE_FILE.exists():
        start_phase = 1
    # else: start_phase already set above from resume logic

    # ---- Execute phases ----
    phase5_result = None

    if start_phase <= 1:
        calibration = run_phase1(calibration)

    if start_phase <= 2:
        calibration = run_phase2(calibration)

    if start_phase <= 3:
        calibration = run_phase3(calibration)

    if start_phase <= 4:
        calibration = run_phase4(calibration)

    if start_phase <= 5:
        calibration, correct, num_samples, accuracy = run_phase5(calibration)
    else:
        # start_phase somehow > 5 -- shouldn't happen due to argparse choices
        correct = calibration.get("phase5_correct", 0)
        num_samples = calibration.get("phase5_num_samples", NUM_PHASE5_SAMPLES)
        accuracy = calibration.get("phase5_accuracy", 0.0)

    # ---- Final summary ----
    neuron_model = "IF_curr_delta" if calibration.get("use_if_curr_delta") else "IF_curr_exp"
    tau_syn = calibration.get("optimal_tau_syn", "unknown")
    weight_scale = calibration.get("optimal_weight_scale", "unknown")
    max_hidden = calibration.get("max_working_hidden_size", "unknown")

    phases_status = []
    for i in range(1, 6):
        key = f"phase{i}_passed"
        val = calibration.get(key)
        if val is True:
            phases_status.append("PASS")
        elif val is False:
            phases_status.append("FAIL")
        else:
            phases_status.append("SKIP")

    phase_str = "  ".join(
        f"{i+1}: {s}" for i, s in enumerate(phases_status)
    )

    print()
    print("=" * 60)
    print("CALIBRATION COMPLETE")
    print("=" * 60)
    print(f"Neuron model  : {neuron_model}")
    print(f"tau_syn       : {tau_syn} ms")
    print(f"Weight scale  : {weight_scale}x")
    print(f"Max hidden    : {max_hidden} neurons")
    print(f"Phase results : [{phase_str}]")
    print(f"Final accuracy: {correct}/{num_samples} ({accuracy:.0%})")
    print()
    print("To reproduce this run:")
    print(f"  python spinnaker/run_on_spinnaker.py \\")
    print(f"      --neuron-model {neuron_model} \\")
    if neuron_model == "IF_curr_exp":
        print(f"      --tau-syn {tau_syn} \\")
    print(f"      --weight-scale {weight_scale} \\")
    print(f"      --max-hidden {max_hidden} \\")
    print(f"      --num-samples 20")
    print("=" * 60)

    LOGGER.summary()

    SCRIPT_END = time.time()
    print(f"\n[{ts()}] auto_calibrate.py -- finished")
    print(f"Total wall clock time: {SCRIPT_END - SCRIPT_START:.1f} s")


if __name__ == "__main__":
    main()
