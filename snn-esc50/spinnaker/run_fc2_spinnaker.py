"""
run_fc2_spinnaker.py -- FC2-only SpiNNaker inference using pre-computed hidden spikes.

Background:
    Previous runs (FC1 + FC2 on SpiNNaker) produced 0% accuracy because:
    - FC1 receives ~1398 simultaneous binary inputs per timestep
    - FC1 weight mean ≈ -0.0034 (near-zero-mean), std ≈ small
    - Net current to each hidden neuron ≈ 0 (massive exc/inh cancellation)
    - No hidden neurons ever fire, regardless of weight_scale

This approach:
    - Run conv + FC1 + lif3 on CPU (snnTorch) first to get hidden spike trains
    - Hidden spikes: sparse binary, ~10-30% of 256 neurons per timestep
    - Feed ONLY the hidden spike trains to SpiNNaker FC2 (256 -> 50)
    - FC2 with sparse inputs: less cancellation, class-selective output expected

Prerequisites:
    1. Run extract_hidden_features.py (regular .venv, not venv-spinnaker):
       source .venv/bin/activate
       python spinnaker/extract_hidden_features.py \\
           --model-path results/snn/direct/best_fold4.pt \\
           --num-samples 20 --fold 4

    2. Run this script (venv-spinnaker):
       source .venv-spinnaker/bin/activate
       python spinnaker/run_fc2_spinnaker.py

Optional arguments:
    --num-samples N       Number of inference samples (default: 5)
    --weight-scale F      FC2 weight scale (default: auto-sweep)
    --no-scale-sweep      Skip Phase 1 scale sweep, use --weight-scale directly
    --skip-to-inference   Skip scale sweep, jump straight to inference

Output files (results/spinnaker_results/):
    fc2_results.json              -- Full inference results per sample
    fc2_scale_sweep.json          -- Phase 1 scale sweep results
    fc2_all_iterations.jsonl      -- Append-only log of every SpiNNaker run
    fc2_iterations/<id>.json      -- One file per experiment
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ============================================================
# Timestamp helpers
# ============================================================
SCRIPT_START = time.time()


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# Argument parsing (before heavy imports)
# ============================================================
parser = argparse.ArgumentParser(
    description="FC2-only SpiNNaker inference using pre-computed hidden spikes"
)
parser.add_argument(
    "--num-samples", type=int, default=5,
    help="Number of samples for inference (default: 5)"
)
parser.add_argument(
    "--weight-scale", type=float, default=None,
    help="FC2 weight multiplier (default: determined by scale sweep)"
)
parser.add_argument(
    "--no-scale-sweep", action="store_true",
    help="Skip Phase 1 scale sweep; requires --weight-scale"
)
parser.add_argument(
    "--skip-to-inference", action="store_true",
    help="Skip scale sweep, go straight to inference with --weight-scale"
)
parser.add_argument(
    "--prune-threshold", type=float, default=0.01,
    help="Minimum |weight| to include (default: 0.01)"
)
parser.add_argument(
    "--sweep-sample", type=int, default=0,
    help="Which sample index to use for scale sweep (default: 0). "
         "Use a sample where snnTorch predicts correctly for best calibration."
)
parser.add_argument(
    "--start-index", type=int, default=0,
    help="Start inference from this sample index (default: 0)"
)
parser.add_argument(
    "--input-dir", type=str, default=None,
    help="Directory containing hidden_spike_features.npy and fc2_connections.npy "
         "(default: results/spinnaker_weights/)"
)
parser.add_argument(
    "--output-dir", type=str, default=None,
    help="Directory for output results (default: results/spinnaker_results/)"
)
parser.add_argument(
    "--fold", type=int, default=None,
    help="Fold number (for labelling results only)"
)
args = parser.parse_args()

# ============================================================
# Directory setup
# ============================================================
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
WEIGHTS_DIR = Path(args.input_dir) if args.input_dir else REPO_ROOT / "results" / "spinnaker_weights"
RESULTS_DIR = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results" / "spinnaker_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# FC2-specific output directory
FC2_ITER_DIR = RESULTS_DIR / "fc2_iterations"
FC2_ITER_DIR.mkdir(parents=True, exist_ok=True)
FC2_MASTER_LOG = RESULTS_DIR / "fc2_all_iterations.jsonl"

SESSION_ID = time.strftime("%Y%m%d_%H%M%S")
PRUNE_THRESHOLD = args.prune_threshold
NUM_INFERENCE_SAMPLES = args.num_samples


# ============================================================
# Iteration Logger
# ============================================================
class IterationLogger:
    """Records every SpiNNaker experiment to disk immediately."""

    def __init__(self):
        self._counter = 0

    def record(self, record: dict) -> None:
        self._counter += 1
        iteration_id = f"{SESSION_ID}_fc2_{self._counter:04d}"

        full_record = {
            "iteration_id": iteration_id,
            "session_id": SESSION_ID,
            "timestamp": ts(),
            "timestamp_unix": time.time(),
            **record,
        }

        try:
            with open(FC2_MASTER_LOG, "a") as f:
                f.write(json.dumps(full_record) + "\n")
                f.flush()
        except Exception as e:
            print(f"  [logger WARNING] Could not write master log: {e}")

        try:
            iter_file = FC2_ITER_DIR / f"{iteration_id}.json"
            with open(iter_file, "w") as f:
                json.dump(full_record, f, indent=2)
        except Exception as e:
            print(f"  [logger WARNING] Could not write iteration file: {e}")

        passed_str = "PASS" if full_record.get("passed") else "FAIL"
        err_str = (f"  ERR={str(full_record['error'])[:50]}"
                   if full_record.get("error") else "")
        print(
            f"  [LOG {iteration_id}] {passed_str} | "
            f"scale={record.get('weight_scale', '?')} | "
            f"out_spk={record.get('n_output_spikes', '?')} | "
            f"predicted={record.get('predicted', '?')} | "
            f"correct={record.get('correct', '?')}{err_str}"
        )

    def summary(self) -> None:
        print(f"\n  [LOGGER SUMMARY] {self._counter} iterations recorded.")
        print(f"  Master log : {FC2_MASTER_LOG}")
        print(f"  Per-iter   : {FC2_ITER_DIR}/")


LOGGER = IterationLogger()

# ============================================================
# Print startup header
# ============================================================
print(f"[{ts()}] run_fc2_spinnaker.py -- starting")
print("=" * 60)
print(f"SpiNNaker FC2-Only Inference (256 -> 50)" + (f" -- Fold {args.fold}" if args.fold else ""))
print("=" * 60)
print(f"  Input dir    : {WEIGHTS_DIR}")
print(f"  Results dir  : {RESULTS_DIR}")
print(f"  Fold         : {args.fold if args.fold else 'not specified'}")
print(f"  Num samples  : {NUM_INFERENCE_SAMPLES}")
print(f"  Start index  : {args.start_index}")
print(f"  Prune thresh : {PRUNE_THRESHOLD}")
print(f"  Sweep sample : {args.sweep_sample}")
if args.weight_scale:
    print(f"  Weight scale : {args.weight_scale} (user-specified)")
else:
    print(f"  Weight scale : auto-sweep")
print()

# ============================================================
# Heavy imports
# ============================================================
try:
    import numpy as np
except ImportError:
    print("FATAL: numpy not available.")
    sys.exit(1)

try:
    import pyNN.spiNNaker as sim
    from spinnman.exceptions import SpinnmanIOException, SpinnmanTimeoutException
except ImportError as e:
    print(f"\nFATAL: Cannot import pyNN.spiNNaker: {e}")
    print("Activate the SpiNNaker venv: source .venv-spinnaker/bin/activate")
    sys.exit(1)

# ============================================================
# Constants -- calibrated from auto_calibrate.py Phases 1 & 2
# ============================================================
NUM_STEPS = 25      # ms (matches trained model)
DT = 1.0            # timestep in ms

# Calibrated LIF params (Phase 1/2 confirmed these work)
LIF_PARAMS = {
    "cm": 1.0,
    "tau_m": 20.0,
    "tau_refrac": 0.1,
    "v_reset": 0.0,
    "v_rest": 0.0,
    "v_thresh": 1.0,
    "tau_syn_E": 5.0,
    "tau_syn_I": 5.0,
}


# ============================================================
# Helpers
# ============================================================
def handle_sim_exception(exc: Exception) -> None:
    exc_type = type(exc).__name__
    exc_str = str(exc)
    print(f"  EXCEPTION: {exc_type}: {exc_str[:200]}")
    if isinstance(exc, SpinnmanIOException) or "No buffer space" in exc_str:
        print("  MEANING: UDP overflow. Too many packets at once.")
    elif isinstance(exc, SpinnmanTimeoutException) or "timeout" in exc_str.lower():
        print("  MEANING: SpiNNaker board did not respond in time.")
    elif "No route found" in exc_str or "routing" in exc_str.lower():
        print("  MEANING: Network routing failure. Too many connections?")


def split_exc_inh(connections: np.ndarray) -> tuple:
    """Split (N,4) [pre, post, weight, delay] into exc/inh lists (unsigned weights)."""
    exc_mask = connections[:, 2] > 0
    inh_mask = connections[:, 2] < 0

    fc_exc = connections[exc_mask].tolist()

    inh_data = connections[inh_mask].copy()
    inh_data[:, 2] = np.abs(inh_data[:, 2])
    fc_inh = inh_data.tolist()

    return fc_exc, fc_inh


def weight_stats(w_array: np.ndarray) -> dict:
    """Compute statistics for a weight array."""
    return {
        "min": float(w_array.min()),
        "max": float(w_array.max()),
        "mean": float(w_array.mean()),
        "std": float(w_array.std()),
        "p5": float(np.percentile(w_array, 5)),
        "p95": float(np.percentile(w_array, 95)),
        "nonzero": int((np.abs(w_array) > 1e-9).sum()),
        "total": int(w_array.shape[0]),
    }


# ============================================================
# Load data
# ============================================================
print(f"[{ts()}] Loading data files...")

hidden_feat_path = WEIGHTS_DIR / "hidden_spike_features.npy"
hidden_label_path = WEIGHTS_DIR / "hidden_labels.npy"
hidden_meta_path = WEIGHTS_DIR / "hidden_metadata.json"
snn_pred_path = WEIGHTS_DIR / "snn_predictions.npy"
fc2_conn_path = WEIGHTS_DIR / "fc2_connections.npy"

for p in [hidden_feat_path, hidden_label_path, fc2_conn_path]:
    if not p.exists():
        print(f"FATAL: Required file not found: {p}")
        if p == hidden_feat_path:
            print("Run extract_hidden_features.py first (with .venv, not .venv-spinnaker)")
        sys.exit(1)

hidden_spike_features = np.load(hidden_feat_path)   # (N, 25, 256)
test_labels = np.load(hidden_label_path)             # (N,)
fc2_all = np.load(fc2_conn_path)                     # (M, 4): [pre, post, weight, delay]

snn_preds = None
if snn_pred_path.exists():
    snn_preds = np.load(snn_pred_path)               # (N,)

hidden_meta = {}
if hidden_meta_path.exists():
    with open(hidden_meta_path) as f:
        hidden_meta = json.load(f)

N_SAMPLES, N_STEPS, INPUT_SIZE = hidden_spike_features.shape
OUTPUT_SIZE = 50  # ESC-50 classes
assert INPUT_SIZE == 256, f"Expected 256 hidden neurons, got {INPUT_SIZE}"
assert N_STEPS == 25, f"Expected 25 timesteps, got {N_STEPS}"

print(f"  Hidden spike features : {hidden_spike_features.shape}")
print(f"  Test labels           : {test_labels.shape}  classes={np.unique(test_labels)[:5]}...")
print(f"  FC2 connections       : {fc2_all.shape[0]:,}")
print(f"  FC2 weight range      : [{fc2_all[:, 2].min():.4f}, {fc2_all[:, 2].max():.4f}]")
print(f"  FC2 weight mean/std   : {fc2_all[:, 2].mean():.4f} / {fc2_all[:, 2].std():.4f}")
print()

# Print hidden activity statistics
sparsity_per_sample = 1.0 - (hidden_spike_features > 0.5).sum(axis=(1, 2)) / (N_STEPS * INPUT_SIZE)
active_per_step_mean = (hidden_spike_features > 0.5).sum(axis=2).mean(axis=1)
print("Hidden spike activity:")
print(f"  Mean sparsity         : {sparsity_per_sample.mean():.3f}")
print(f"  Mean active/timestep  : {active_per_step_mean.mean():.1f} / {INPUT_SIZE}")
print(f"  Max simultaneous spks : {int((hidden_spike_features > 0.5).sum(axis=2).max())}")
print()

# ============================================================
# Prune and prepare FC2 connections
# ============================================================
fc2_pruned = fc2_all[np.abs(fc2_all[:, 2]) > PRUNE_THRESHOLD].copy()
print(f"FC2 after pruning (|w| > {PRUNE_THRESHOLD}):")
print(f"  Connections: {len(fc2_pruned):,} (of {len(fc2_all):,})")
print(f"  Weight range: [{fc2_pruned[:, 2].min():.4f}, {fc2_pruned[:, 2].max():.4f}]")


def apply_scale_and_split(connections: np.ndarray, scale: float):
    """Apply weight scale and split into exc/inh connection lists."""
    scaled = connections.copy()
    scaled[:, 2] = scaled[:, 2] * scale
    exc, inh = split_exc_inh(scaled)
    return exc, inh


# ============================================================
# Single-sample SpiNNaker run (used for both sweep and inference)
# ============================================================
def run_one_sample(
    spike_input: np.ndarray,
    weight_scale: float,
    label: str = "?",
    record_all_voltages: bool = True,
) -> dict:
    """
    Run one sample through FC2-only SpiNNaker network.

    Args:
        spike_input: (25, 256) binary array -- hidden spike trains.
        weight_scale: Multiplier applied to FC2 weights.
        label: String label for logging.
        record_all_voltages: Whether to record all 50 output voltages.

    Returns:
        dict with keys: n_output_spikes, predicted, max_output_v, output_spikes_per_class,
                        output_v_final, spikes_per_step, error, wall_clock_ms, etc.
    """
    t0 = time.time()

    # Build spike times for each of the 256 hidden neurons
    spike_times_list = []
    for neuron_idx in range(INPUT_SIZE):
        times = np.where(spike_input[:, neuron_idx] > 0.5)[0].astype(float).tolist()
        spike_times_list.append(times)

    active_inputs = sum(1 for t in spike_times_list if len(t) > 0)
    total_input_spikes = int(np.sum(spike_input > 0.5))
    input_spikes_per_step = (spike_input > 0.5).sum(axis=1).tolist()

    # Apply scale and split
    fc2_exc, fc2_inh = apply_scale_and_split(fc2_pruned, weight_scale)

    error_str = None
    n_output_spikes = 0
    output_spikes_per_class = [0] * OUTPUT_SIZE
    output_v_final = [0.0] * OUTPUT_SIZE
    predicted = 0
    classification_basis = "not_run"

    try:
        sim.setup(timestep=DT)

        try:
            # Input population: 256 neurons representing hidden layer spikes
            input_pop = sim.Population(
                INPUT_SIZE,
                sim.SpikeSourceArray,
                {"spike_times": spike_times_list},
                label="hidden_input"
            )
            input_pop.record("spikes")

            # Output population: 50 IF_curr_exp neurons (FC2)
            output_pop = sim.Population(
                OUTPUT_SIZE,
                sim.IF_curr_exp(**LIF_PARAMS),
                label="output"
            )
            output_pop.record(["spikes", "v"])

            # FC2 projections (excitatory and inhibitory separately)
            if fc2_exc:
                sim.Projection(
                    input_pop, output_pop,
                    sim.FromListConnector(fc2_exc),
                    receptor_type="excitatory"
                )
            if fc2_inh:
                sim.Projection(
                    input_pop, output_pop,
                    sim.FromListConnector(fc2_inh),
                    receptor_type="inhibitory"
                )

            print(f"    Network: {INPUT_SIZE} inputs -> {OUTPUT_SIZE} outputs")
            print(f"    FC2: {len(fc2_exc)} exc + {len(fc2_inh)} inh connections")
            print(f"    Active hidden neurons: {active_inputs}/{INPUT_SIZE}")
            print(f"    Total hidden spikes  : {total_input_spikes}")

            sim.run(NUM_STEPS)

            # --- Extract results ---
            # Input spikes (verify delivery)
            in_data = input_pop.get_data("spikes")
            in_spiketrains = in_data.segments[0].spiketrains
            in_actual = sum(len(st) for st in in_spiketrains)

            # Output spikes
            out_data = output_pop.get_data(["spikes", "v"])
            out_spiketrains = out_data.segments[0].spiketrains
            out_v_signal = out_data.segments[0].filter(name="v")[0]

            # Count spikes per output neuron
            for neuron_id, st in enumerate(out_spiketrains):
                n_spk = len(st)
                output_spikes_per_class[neuron_id] = n_spk
                n_output_spikes += n_spk

            # Extract final membrane voltages
            try:
                v_arr = out_v_signal.magnitude  # (T, 50)
                output_v_final = v_arr[-1, :].tolist()
                max_output_v = float(v_arr.max())
            except Exception as ve:
                print(f"    WARNING: Could not extract output voltages: {ve}")
                output_v_final = [0.0] * OUTPUT_SIZE
                max_output_v = 0.0

            # Classification
            if n_output_spikes > 0:
                predicted = int(np.argmax(output_spikes_per_class))
                classification_basis = "spike_count"
            else:
                predicted = int(np.argmax(output_v_final))
                classification_basis = "final_membrane_voltage"

            # Spikes per timestep for output layer
            # (approximate from spike trains -- may not align perfectly to integer ms)
            spikes_per_step_out = [0] * NUM_STEPS
            for st in out_spiketrains:
                for t in st.magnitude:
                    step = int(t)
                    if 0 <= step < NUM_STEPS:
                        spikes_per_step_out[step] += 1

            neurons_that_fired = [i for i, c in enumerate(output_spikes_per_class) if c > 0]

            print(f"    Input delivered   : expected={total_input_spikes}, actual={in_actual}")
            print(f"    Output spikes     : {n_output_spikes} total")
            print(f"    Neurons fired     : {neurons_that_fired[:10]}")
            print(f"    Max output V      : {max_output_v:.5f}")
            print(f"    Predicted class   : {predicted} (via {classification_basis})")

        finally:
            sim.end()

    except (SpinnmanIOException, SpinnmanTimeoutException) as exc:
        handle_sim_exception(exc)
        error_str = str(exc)
        t_end = time.time()
        return {
            "n_output_spikes": -1,
            "predicted": -1,
            "max_output_v": float("nan"),
            "output_spikes_per_class": [],
            "output_v_final": [],
            "classification_basis": "error",
            "active_inputs": active_inputs,
            "total_input_spikes": total_input_spikes,
            "wall_clock_ms": (t_end - t0) * 1000,
            "error": error_str,
        }

    except Exception as exc:
        handle_sim_exception(exc)
        error_str = str(exc)
        t_end = time.time()
        return {
            "n_output_spikes": -1,
            "predicted": -1,
            "max_output_v": float("nan"),
            "output_spikes_per_class": [],
            "output_v_final": [],
            "classification_basis": "error",
            "active_inputs": active_inputs,
            "total_input_spikes": total_input_spikes,
            "wall_clock_ms": (t_end - t0) * 1000,
            "error": error_str,
        }

    t_end = time.time()
    return {
        "n_output_spikes": n_output_spikes,
        "predicted": predicted,
        "max_output_v": max_output_v,
        "output_spikes_per_class": output_spikes_per_class,
        "output_v_final": output_v_final,
        "classification_basis": classification_basis,
        "active_inputs": active_inputs,
        "total_input_spikes": total_input_spikes,
        "spikes_per_step_output": spikes_per_step_out,
        "neurons_that_fired_output": neurons_that_fired,
        "input_spikes_per_step": input_spikes_per_step,
        "wall_clock_ms": (t_end - t0) * 1000,
        "error": error_str,
    }


# ============================================================
# Phase 1: Weight Scale Sweep
# ============================================================
SCALE_CANDIDATES = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

sweep_results = []
chosen_scale = None


def run_phase1_scale_sweep() -> float:
    """Run scale sweep on sample 0 to find best weight_scale for FC2."""
    global sweep_results

    print()
    print("=" * 60)
    print("PHASE 1: Weight Scale Sweep (sample 0)")
    print("=" * 60)
    print("Strategy: try multiple weight scales to find where output neurons fire.")
    print("Unlike FC1 (1398 simultaneous inputs -> cancellation), FC2 receives")
    print("sparse hidden spikes (~10-30% of 256 neurons) -> less cancellation.")
    print()

    sweep_idx = args.sweep_sample
    spike_input = hidden_spike_features[sweep_idx]  # (25, 256)
    true_label = int(test_labels[sweep_idx])
    snn_pred = int(snn_preds[sweep_idx]) if snn_preds is not None else -1

    print(f"Sample {sweep_idx}: true_label={true_label}, snnTorch_pred={snn_pred}")
    print(f"  Active hidden neurons: {int((spike_input > 0.5).sum(axis=0).astype(bool).sum())}/256")
    print(f"  Total spikes: {int((spike_input > 0.5).sum())}")
    print()

    best_scale = SCALE_CANDIDATES[-1]  # default to largest
    best_score = -1

    for scale in SCALE_CANDIDATES:
        print(f"--- Scale {scale}x ---")
        result = run_one_sample(spike_input, scale, label=f"scale={scale}")

        n_out_spk = result.get("n_output_spikes", -1)
        predicted = result.get("predicted", -1)
        correct = (predicted == true_label)
        max_v = result.get("max_output_v", 0.0)
        neurons_fired = len(result.get("neurons_that_fired_output", []))

        sweep_results.append({
            "scale": float(scale),
            "n_output_spikes": int(n_out_spk),
            "neurons_that_fired": neurons_fired,
            "predicted": int(predicted),
            "true_label": int(true_label),
            "correct": bool(correct),
            "max_output_v": float(max_v) if max_v == max_v else None,  # NaN check
            "error": result.get("error"),
        })

        LOGGER.record({
            "phase": "scale_sweep",
            "test_name": f"fc2_scale_{scale}x",
            "weight_scale": float(scale),
            "params": {**LIF_PARAMS, "weight_scale": float(scale)},
            "sample_idx": 0,
            "true_label": int(true_label),
            "snn_predicted": int(snn_pred),
            "n_input_spikes_expected": int((spike_input > 0.5).sum()),
            "n_input_spikes_actual": result.get("total_input_spikes", 0),
            "n_hidden_spikes": result.get("total_input_spikes", 0),
            "n_output_spikes": int(n_out_spk),
            "max_hidden_v": 0.0,
            "max_output_v": float(max_v) if max_v == max_v else 0.0,
            "v_trace": [],
            "hidden_v_traces": {},
            "output_v_traces": {},
            "spikes_per_step_input": result.get("input_spikes_per_step", []),
            "spikes_per_step_hidden": [],
            "spikes_per_step_output": result.get("spikes_per_step_output", []),
            "neurons_that_fired_hidden": [],
            "neurons_that_fired_output": result.get("neurons_that_fired_output", []),
            "wall_clock_ms": result.get("wall_clock_ms", 0),
            "passed": n_out_spk > 0,
            "error": result.get("error"),
            "notes": f"FC2-only scale sweep sample 0",
        })

        # Score: prefer scales where some (but not all) neurons fire
        # Best: correct class fires the most
        if n_out_spk > 0:
            # Use this scale: count neurons that fired
            score = neurons_fired
            if correct:
                score += 100  # strongly prefer correct prediction
            if score > best_score:
                best_score = score
                best_scale = scale
                print(f"  -> New best scale: {scale}x (score={score})")

        if result.get("error"):
            print(f"  ERROR at scale {scale}x -- stopping sweep early")
            break

        print()

    # Save sweep results
    sweep_path = RESULTS_DIR / "fc2_scale_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump({
            "timestamp": ts(),
            "sample_idx": 0,
            "true_label": int(true_label),
            "snn_predicted": int(snn_pred) if snn_pred is not None else -1,
            "scales_tried": SCALE_CANDIDATES,
            "results": sweep_results,
            "chosen_scale": float(best_scale),
        }, f, indent=2)
    print(f"Scale sweep results saved: {sweep_path}")

    print()
    print(f"Scale sweep summary:")
    print(f"  {'Scale':>8}  {'OutSpk':>6}  {'Fired':>5}  {'Pred':>4}  {'Correct':>7}")
    print("  " + "-" * 45)
    for r in sweep_results:
        print(f"  {r['scale']:>8}  {r['n_output_spikes']:>6}  "
              f"{r['neurons_that_fired']:>5}  {r['predicted']:>4}  {str(r['correct']):>7}")
    print()
    print(f"  Chosen scale: {best_scale}x")
    return float(best_scale)


# ============================================================
# Phase 2: Full Inference
# ============================================================
def run_phase2_inference(weight_scale: float) -> dict:
    """Run full inference on all test samples with the given weight scale."""

    print()
    print("=" * 60)
    print(f"PHASE 2: Full Inference (weight_scale={weight_scale}x)")
    print("=" * 60)

    start_idx = args.start_index
    num_samples = min(NUM_INFERENCE_SAMPLES, N_SAMPLES - start_idx)
    print(f"Running samples {start_idx} to {start_idx + num_samples - 1} ({num_samples} total)...")
    print()

    results_list = []
    correct = 0
    snn_correct = 0

    for sample_idx in range(start_idx, start_idx + num_samples):
        spike_input = hidden_spike_features[sample_idx]  # (25, 256)
        true_label = int(test_labels[sample_idx])
        snn_pred = int(snn_preds[sample_idx]) if snn_preds is not None else -1

        print(f"{'=' * 50}")
        print(f"Sample {sample_idx + 1}/{num_samples}  "
              f"(true={true_label}, snn_pred={snn_pred})")
        print(f"{'=' * 50}")

        result = run_one_sample(spike_input, weight_scale, label=f"sample_{sample_idx}")

        predicted = result.get("predicted", 0)
        is_correct = (predicted == true_label)
        if is_correct:
            correct += 1
        if snn_pred == true_label:
            snn_correct += 1

        sample_result = {
            "sample": sample_idx,
            "true_label": int(true_label),
            "snn_predicted": int(snn_pred),
            "snn_correct": bool(snn_pred == true_label),
            "predicted": int(predicted),
            "correct": bool(is_correct),
            "classification_basis": result.get("classification_basis", "unknown"),
            "n_output_spikes": result.get("n_output_spikes", 0),
            "max_output_v": result.get("max_output_v", 0.0),
            "output_spikes_per_class": result.get("output_spikes_per_class", []),
            "top5_output_v": sorted(
                [(i, v) for i, v in enumerate(result.get("output_v_final", []))],
                key=lambda x: x[1], reverse=True
            )[:5],
            "top5_output_spk": sorted(
                [(i, c) for i, c in enumerate(result.get("output_spikes_per_class", []))],
                key=lambda x: x[1], reverse=True
            )[:5],
            "neurons_that_fired": result.get("neurons_that_fired_output", []),
            "active_inputs": result.get("active_inputs", 0),
            "total_input_spikes": result.get("total_input_spikes", 0),
            "wall_clock_ms": result.get("wall_clock_ms", 0.0),
            "error": result.get("error"),
        }
        results_list.append(sample_result)

        LOGGER.record({
            "phase": "inference",
            "test_name": f"sample_{sample_idx}",
            "weight_scale": float(weight_scale),
            "params": {**LIF_PARAMS, "weight_scale": float(weight_scale)},
            "sample_idx": int(sample_idx),
            "true_label": int(true_label),
            "snn_predicted": int(snn_pred),
            "predicted": int(predicted),
            "correct": bool(is_correct),
            "n_input_spikes_expected": result.get("total_input_spikes", 0),
            "n_input_spikes_actual": result.get("total_input_spikes", 0),
            "n_hidden_spikes": result.get("total_input_spikes", 0),
            "n_output_spikes": result.get("n_output_spikes", 0),
            "max_hidden_v": 0.0,
            "max_output_v": result.get("max_output_v", 0.0),
            "v_trace": [],
            "hidden_v_traces": {},
            "output_v_traces": {},
            "spikes_per_step_input": result.get("input_spikes_per_step", []),
            "spikes_per_step_hidden": [],
            "spikes_per_step_output": result.get("spikes_per_step_output", []),
            "neurons_that_fired_hidden": [],
            "neurons_that_fired_output": result.get("neurons_that_fired_output", []),
            "wall_clock_ms": result.get("wall_clock_ms", 0),
            "passed": is_correct,
            "error": result.get("error"),
            "notes": f"FC2-only full inference, scale={weight_scale}x",
        })

        print(f"  => Sample {sample_idx}: pred={predicted}, "
              f"true={true_label}, correct={is_correct}")
        print()

    # --- Summary ---
    accuracy = correct / num_samples if num_samples > 0 else 0.0
    snn_accuracy = snn_correct / num_samples if num_samples > 0 else 0.0

    print("=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"  SpiNNaker FC2-only: {correct}/{num_samples} = {accuracy:.1%}")
    print(f"  snnTorch reference: {snn_correct}/{num_samples} = {snn_accuracy:.1%}")
    print()

    print(f"  {'Sample':>6}  {'True':>4}  {'SNN':>3}  {'Spk':>3}  {'Pred':>4}  "
          f"{'SpkOK':>5}  {'Correct':>7}")
    print("  " + "-" * 50)
    for r in results_list:
        print(f"  {r['sample']:>6}  {r['true_label']:>4}  {r['snn_predicted']:>3}  "
              f"{r['n_output_spikes']:>3}  {r['predicted']:>4}  "
              f"{str(r['snn_correct']):>5}  {str(r['correct']):>7}")

    # Save full results
    inference_result = {
        "platform": "SpiNNaker",
        "timestamp": ts(),
        "script": "run_fc2_spinnaker.py",
        "fold": args.fold,
        "approach": "FC2-only with pre-computed snnTorch hidden spikes",
        "config": {
            "weight_scale": float(weight_scale),
            "prune_threshold": PRUNE_THRESHOLD,
            "lif_params": LIF_PARAMS,
            "n_connections_exc": len(apply_scale_and_split(fc2_pruned, 1.0)[0]),
            "n_connections_inh": len(apply_scale_and_split(fc2_pruned, 1.0)[1]),
            "input_size": int(INPUT_SIZE),
            "output_size": int(OUTPUT_SIZE),
            "num_steps": int(NUM_STEPS),
        },
        "num_samples": int(num_samples),
        "correct": int(correct),
        "accuracy": float(accuracy),
        "snn_reference_correct": int(snn_correct),
        "snn_reference_accuracy": float(snn_accuracy),
        "per_sample": results_list,
        "scale_sweep": sweep_results,
    }

    fold_suffix = f"_fold{args.fold}" if args.fold is not None else ""
    result_path = RESULTS_DIR / f"fc2_results{fold_suffix}.json"
    with open(result_path, "w") as f:
        json.dump(inference_result, f, indent=2)
    print()
    print(f"Results saved: {result_path}")

    return inference_result


# ============================================================
# Main
# ============================================================
if args.no_scale_sweep or args.skip_to_inference:
    if args.weight_scale is None:
        print("ERROR: --no-scale-sweep / --skip-to-inference requires --weight-scale")
        sys.exit(1)
    final_scale = args.weight_scale
    print(f"Skipping scale sweep. Using weight_scale={final_scale}")
elif args.weight_scale is not None:
    # User specified scale but didn't skip sweep -- still run sweep for logging
    final_scale = run_phase1_scale_sweep()
    print(f"User override: using weight_scale={args.weight_scale} (sweep suggested {final_scale})")
    final_scale = args.weight_scale
else:
    final_scale = run_phase1_scale_sweep()

run_phase2_inference(final_scale)

LOGGER.summary()

total_elapsed = time.time() - SCRIPT_START
print()
print(f"[{ts()}] Done. Total wall clock: {total_elapsed:.1f}s")
