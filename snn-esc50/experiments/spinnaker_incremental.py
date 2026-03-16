"""
spinnaker_incremental.py -- Incremental SpiNNaker deployment debugging.

Build up from the SMALLEST working network to full FC1+FC2 deployment.
Each step proves one more thing works before going to the next.

Steps:
  1   Tiny test network (10 -> 5 -> 2). Synthetic inputs. Proves pipeline works.
  2   FC2 dimensions (256 -> 50) with KNOWN GOOD synthetic weights.
      Identity-like mapping: input i excites output i%50. Proves FC2-scale works.
  3a  FC1 EXCITATORY-ONLY (2304 -> 256). All negative weights set to 0.
      Eliminates cancellation entirely. Hidden neurons WILL fire.
  3b  FC1 TOP-K inputs (K most important connections per hidden neuron).
      Reduces input density to control net current.
  3c  FC1 with POSITIVE BIAS INJECTION. Keep all weights but add a fixed
      excitatory bias current to overcome inhibitory cancellation.
  3d  FC1 full weights with TEMPORAL SEPARATION. Excitatory inputs fire
      in early timesteps, inhibitory delayed. Lets excitation build up first.
  4   FC1 -> FC2 chain. Use the best FC1 strategy, feed hidden spikes to FC2.
      Measure end-to-end accuracy vs snnTorch.

Usage (always from .venv-spinnaker):
    source .venv-spinnaker/bin/activate
    cd snn-esc50

    python experiments/spinnaker_incremental.py --step 1
    python experiments/spinnaker_incremental.py --step 2
    python experiments/spinnaker_incremental.py --step 3a
    python experiments/spinnaker_incremental.py --step 3b --top-k 200
    python experiments/spinnaker_incremental.py --step 3c --bias-current 0.5
    python experiments/spinnaker_incremental.py --step 3d
    python experiments/spinnaker_incremental.py --step 4 --fc1-strategy exc_only

Prerequisites:
    - .venv-spinnaker with pyNN.spiNNaker, numpy
    - ~/.spynnaker.cfg pointing to spinnaker.cs.man.ac.uk
    - Weight/feature files in results/spinnaker_results/full_deploy_cond/
      and results/spinnaker_weights/fold4/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
WEIGHTS_DIR = REPO_ROOT / "results" / "spinnaker_weights" / "fold4"
COND_DIR = REPO_ROOT / "results" / "spinnaker_results" / "full_deploy_cond"
RESULTS_DIR = REPO_ROOT / "results" / "spinnaker_results" / "incremental"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# SpiNNaker import
# ============================================================
try:
    import pyNN.spiNNaker as sim
except ImportError as e:
    print(f"FATAL: Cannot import pyNN.spiNNaker: {e}")
    print("Activate: source .venv-spinnaker/bin/activate")
    sys.exit(1)

# ============================================================
# Calibrated neuron parameters (same as working FC2 script)
# ============================================================
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
NUM_STEPS = 25
DT = 1.0


# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser(
    description="Incremental SpiNNaker deployment debugging"
)
parser.add_argument(
    "--step", type=str, required=True,
    choices=["1", "2", "3a", "3b", "3c", "3d", "4"],
    help="Which step to run"
)
parser.add_argument(
    "--weight-scale", type=float, default=None,
    help="Weight multiplier (default: auto per step. FC2-only=5.0, FC1 exc-only=1.0)"
)
parser.add_argument(
    "--top-k", type=int, default=200,
    help="For step 3b: number of top connections per hidden neuron"
)
parser.add_argument(
    "--bias-current", type=float, default=0.04,
    help="For step 3c: constant excitatory bias current (nA). "
         "V_ss = bias * 20 (tau_m/cm). 0.04 -> V_ss=0.8, just below threshold."
)
parser.add_argument(
    "--sample-idx", type=int, default=0,
    help="Which sample to use from FC1 input features (0-19)"
)
parser.add_argument(
    "--fc1-strategy", type=str, default="exc_only",
    choices=["exc_only", "top_k", "bias", "temporal"],
    help="For step 4: which FC1 strategy to chain with FC2"
)
parser.add_argument(
    "--num-samples", type=int, default=5,
    help="For step 4: how many samples to run end-to-end"
)
parser.add_argument(
    "--max-connections", type=int, default=300000,
    help="Max total connections per layer. Prune smallest if exceeded. "
         "SpiNNaker can handle ~500K but routing errors start >300K."
)
parser.add_argument(
    "--prune-threshold", type=float, default=0.001,
    help="Minimum |weight| to include in connection list (default: 0.001)"
)

args = parser.parse_args()

# Auto-scale: different defaults per step
if args.weight_scale is None:
    STEP_DEFAULT_SCALES = {
        "1": 5.0,
        "2": 5.0,
        "3a": 1.0,   # Exc-only: current already ~12 per neuron at scale=1
        "3b": 5.0,
        "3c": 5.0,
        "3d": 5.0,
        "4": 1.0,    # FC1+FC2: start conservative
    }
    args.weight_scale = STEP_DEFAULT_SCALES.get(args.step, 5.0)
    print(f"  Auto weight-scale for step {args.step}: {args.weight_scale}")


# ============================================================
# Helper: build spike times from binary matrix
# ============================================================
def binary_to_spike_times(binary_matrix: np.ndarray) -> list:
    """Convert (T, N) binary matrix to list of spike time lists.

    Returns list of length N, where each element is a list of spike
    times (floats) for that neuron.
    """
    T, N = binary_matrix.shape
    spike_times_list = []
    for n in range(N):
        times = np.where(binary_matrix[:, n] > 0.5)[0].astype(float).tolist()
        spike_times_list.append(times)
    return spike_times_list


# ============================================================
# Helper: run a network and extract results
# ============================================================
def run_network(
    input_spike_times: list,
    n_input: int,
    layer_configs: list,
    run_time: float = NUM_STEPS,
    label: str = "test",
) -> dict:
    """
    Run an arbitrary feedforward SNN on SpiNNaker.

    Args:
        input_spike_times: list of length n_input, each a list of spike times.
        n_input: number of input neurons.
        layer_configs: list of dicts, each with:
            - 'n': number of neurons in this layer
            - 'exc_conns': list of [pre, post, weight, delay] for excitatory
            - 'inh_conns': list of [pre, post, weight, delay] for inhibitory
            - 'lif_params': dict of neuron params (optional, defaults to LIF_PARAMS)
            - 'i_offset': float, constant injection current (optional, default 0)
        run_time: simulation duration in ms.
        label: string label for logging.

    Returns:
        dict with per-layer spike counts, voltages, timing info.
    """
    t0 = time.time()
    result = {
        "label": label,
        "n_input": n_input,
        "n_layers": len(layer_configs),
        "run_time": run_time,
        "error": None,
    }

    sim_started = False
    try:
        sim.setup(timestep=DT)
        sim_started = True
        try:
            # Input population
            input_pop = sim.Population(
                n_input,
                sim.SpikeSourceArray,
                {"spike_times": input_spike_times},
                label="input"
            )
            input_pop.record("spikes")

            # Build layers
            populations = [input_pop]
            for i, cfg in enumerate(layer_configs):
                n_neurons = cfg["n"]
                params = cfg.get("lif_params", LIF_PARAMS).copy()
                i_offset = cfg.get("i_offset", 0.0)
                if i_offset != 0.0:
                    params["i_offset"] = i_offset

                pop = sim.Population(
                    n_neurons,
                    sim.IF_curr_exp(**params),
                    label=f"layer_{i}"
                )
                pop.record(["spikes", "v"])
                populations.append(pop)

                # Excitatory connections from previous layer
                exc_conns = cfg.get("exc_conns", [])
                if exc_conns:
                    sim.Projection(
                        populations[-2], pop,
                        sim.FromListConnector(exc_conns),
                        receptor_type="excitatory"
                    )

                # Inhibitory connections from previous layer
                inh_conns = cfg.get("inh_conns", [])
                if inh_conns:
                    sim.Projection(
                        populations[-2], pop,
                        sim.FromListConnector(inh_conns),
                        receptor_type="inhibitory"
                    )

            sim.run(run_time)

            # Extract results from input
            in_data = input_pop.get_data("spikes")
            in_trains = in_data.segments[0].spiketrains
            result["input_spikes_delivered"] = sum(len(st) for st in in_trains)
            result["input_spikes_expected"] = sum(len(t) for t in input_spike_times)

            # Extract results from each layer
            result["layers"] = []
            for i, pop in enumerate(populations[1:]):
                layer_result = {"name": f"layer_{i}", "n": layer_configs[i]["n"]}

                data = pop.get_data(["spikes", "v"])
                spiketrains = data.segments[0].spiketrains
                v_signal = data.segments[0].filter(name="v")[0]

                # Spike counts per neuron
                spike_counts = [len(st) for st in spiketrains]
                total_spikes = sum(spike_counts)
                neurons_fired = [j for j, c in enumerate(spike_counts) if c > 0]

                # Voltage
                try:
                    v_arr = v_signal.magnitude  # (T, N)
                    v_final = v_arr[-1, :].tolist()
                    v_max = float(v_arr.max())
                    v_mean_final = float(v_arr[-1, :].mean())
                except Exception:
                    v_final = []
                    v_max = 0.0
                    v_mean_final = 0.0

                layer_result["total_spikes"] = total_spikes
                layer_result["neurons_fired"] = len(neurons_fired)
                layer_result["neurons_fired_ids"] = neurons_fired[:20]  # cap for readability
                layer_result["spike_counts"] = spike_counts
                layer_result["v_max"] = v_max
                layer_result["v_mean_final"] = v_mean_final
                layer_result["v_final_top5"] = sorted(
                    enumerate(v_final), key=lambda x: x[1], reverse=True
                )[:5]

                # Predicted class (argmax of spike counts, fallback to voltage)
                if total_spikes > 0:
                    layer_result["predicted"] = int(np.argmax(spike_counts))
                    layer_result["classification_basis"] = "spike_count"
                else:
                    if v_final:
                        layer_result["predicted"] = int(np.argmax(v_final))
                        layer_result["classification_basis"] = "final_voltage"
                    else:
                        layer_result["predicted"] = -1
                        layer_result["classification_basis"] = "none"

                result["layers"].append(layer_result)

        finally:
            if sim_started:
                sim.end()

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        print(f"  ERROR: {result['error']}")

    result["wall_clock_ms"] = (time.time() - t0) * 1000
    return result


def print_layer_summary(result: dict) -> None:
    """Print a human-readable summary of network results."""
    if result.get("error"):
        print(f"  ERROR: {result['error']}")
        return

    print(f"  Input: delivered={result['input_spikes_delivered']} "
          f"(expected={result['input_spikes_expected']})")

    for lr in result.get("layers", []):
        n = lr["n"]
        fired = lr["neurons_fired"]
        total = lr["total_spikes"]
        v_max = lr["v_max"]
        pred = lr.get("predicted", -1)
        basis = lr.get("classification_basis", "?")
        print(f"  {lr['name']} ({n}n): {fired} fired, {total} spikes, "
              f"v_max={v_max:.4f}, pred={pred} ({basis})")
        if fired > 0 and fired <= 20:
            print(f"    Fired IDs: {lr['neurons_fired_ids']}")
            # Show top spike counts
            top5 = sorted(enumerate(lr["spike_counts"]),
                          key=lambda x: x[1], reverse=True)[:5]
            print(f"    Top-5 spike counts: {top5}")


def save_result(step: str, result: dict) -> None:
    """Save result to JSON file."""
    path = RESULTS_DIR / f"step_{step}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved: {path}")


# ============================================================
# STEP 1: Tiny test network (10 -> 5 -> 2)
# ============================================================
def step_1():
    """
    10 input neurons -> 5 hidden neurons -> 2 output neurons.
    Synthetic: inputs 0-4 excite hidden 0-4 with strong weights.
    Hidden 0-2 excite output 0. Hidden 3-4 excite output 1.
    Feed spikes to inputs 0-2 only -> expect output 0 to fire.
    """
    print("=" * 60)
    print("STEP 1: Tiny Test Network (10 -> 5 -> 2)")
    print("=" * 60)
    print("Goal: Prove SpiNNaker pipeline works with a trivial network.")
    print("Inputs 0-2 fire -> hidden 0-2 fire -> output 0 fires.")
    print()

    n_input = 10
    n_hidden = 5
    n_output = 2
    scale = args.weight_scale

    # Input: neurons 0-2 fire at regular intervals
    input_spike_times = []
    for i in range(n_input):
        if i < 3:
            # Fire every 2ms for strong drive
            input_spike_times.append([float(t) for t in range(0, NUM_STEPS, 2)])
        else:
            input_spike_times.append([])  # silent

    # Layer 1 (hidden): strong 1-to-1 excitatory from input i to hidden i
    # Weight = 0.5 * scale -> with 13 spikes from each input over 25 steps,
    # each hidden neuron gets plenty of drive
    exc_conns_l1 = []
    for i in range(min(n_input, n_hidden)):
        exc_conns_l1.append([i, i, 0.5 * scale, 1.0])

    # Layer 2 (output): hidden 0-2 -> output 0, hidden 3-4 -> output 1
    exc_conns_l2 = []
    for h in range(3):
        exc_conns_l2.append([h, 0, 0.3 * scale, 1.0])
    for h in range(3, 5):
        exc_conns_l2.append([h, 1, 0.3 * scale, 1.0])

    layer_configs = [
        {"n": n_hidden, "exc_conns": exc_conns_l1, "inh_conns": []},
        {"n": n_output, "exc_conns": exc_conns_l2, "inh_conns": []},
    ]

    print(f"  Scale: {scale}")
    print(f"  L1 exc connections: {len(exc_conns_l1)}")
    print(f"  L2 exc connections: {len(exc_conns_l2)}")
    print(f"  Input spike times sample: {input_spike_times[0][:5]}...")
    print()

    result = run_network(input_spike_times, n_input, layer_configs, label="step1_tiny")
    print_layer_summary(result)

    # Validate
    if result.get("error"):
        print("\n  STEP 1 FAILED -- SpiNNaker pipeline error.")
    else:
        hidden_layer = result["layers"][0]
        output_layer = result["layers"][1]

        hidden_ok = hidden_layer["neurons_fired"] >= 2
        output_ok = output_layer["neurons_fired"] >= 1

        if hidden_ok and output_ok:
            print(f"\n  STEP 1 PASSED -- Hidden fired: {hidden_layer['neurons_fired']}/5, "
                  f"Output fired: {output_layer['neurons_fired']}/2")
            if output_layer.get("predicted") == 0:
                print("  BONUS: Output 0 correctly predicted (inputs 0-2 drove it).")
            else:
                print(f"  NOTE: Output predicted {output_layer.get('predicted')}, "
                      f"expected 0. Not critical for pipeline test.")
        else:
            print(f"\n  STEP 1 PARTIAL -- Hidden fired: {hidden_layer['neurons_fired']}/5, "
                  f"Output fired: {output_layer['neurons_fired']}/2")
            print("  Some neurons not firing. Check weight scale.")

    save_result("1", result)
    return result


# ============================================================
# STEP 2: FC2-scale synthetic test (256 -> 50)
# ============================================================
def step_2():
    """
    256 input -> 50 output with known-good synthetic weights.
    Input neuron i has strong excitatory weight to output i%50.
    Fire inputs 0-9 (which map to outputs 0-9) -> expect output 0 to fire most.
    Also fire inputs 50-54 (also map to outputs 0-4) -> reinforces.
    """
    print("=" * 60)
    print("STEP 2: FC2-Scale Synthetic Test (256 -> 50)")
    print("=" * 60)
    print("Goal: Verify 256->50 network at FC2 scale with synthetic weights.")
    print("Input i -> output i%50 with strong weight. Only some inputs fire.")
    print()

    n_input = 256
    n_output = 50
    scale = args.weight_scale

    # Fire inputs 0-9 and 50-54 (sparsely, like real hidden spikes ~55/256)
    active_neurons = list(range(10)) + list(range(50, 55))
    input_spike_times = []
    for i in range(n_input):
        if i in active_neurons:
            # Irregular firing to mimic real hidden activity
            times = [float(t) for t in range(0, NUM_STEPS, 3)]  # every 3ms
            input_spike_times.append(times)
        else:
            input_spike_times.append([])

    # Synthetic weights: neuron i -> output i%50, weight = 0.2
    exc_conns = []
    for i in range(n_input):
        target = i % n_output
        exc_conns.append([i, target, 0.2 * scale, 1.0])

    layer_configs = [
        {"n": n_output, "exc_conns": exc_conns, "inh_conns": []},
    ]

    total_active = len(active_neurons)
    print(f"  Scale: {scale}")
    print(f"  Active inputs: {total_active}/{n_input} ({100*total_active/n_input:.1f}%)")
    print(f"  Connections: {len(exc_conns)} exc, 0 inh")
    print(f"  Expected: outputs 0-9 fire (driven by inputs 0-9 + 50-54)")
    print()

    result = run_network(input_spike_times, n_input, layer_configs, label="step2_fc2_synth")
    print_layer_summary(result)

    if not result.get("error"):
        layer = result["layers"][0]
        if layer["neurons_fired"] > 0:
            print(f"\n  STEP 2 PASSED -- {layer['neurons_fired']}/50 outputs fired, "
                  f"{layer['total_spikes']} total spikes.")
            # Check if outputs 0-4 are among the top firers (they get 2x input)
            top5 = sorted(enumerate(layer["spike_counts"]),
                          key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top-5 outputs: {top5}")
            reinforced = [0, 1, 2, 3, 4]  # these get input from both i and i+50
            reinforced_in_top5 = [t[0] for t in top5 if t[0] in reinforced]
            if len(reinforced_in_top5) >= 3:
                print(f"  BONUS: Reinforced outputs {reinforced_in_top5} correctly in top-5!")
        else:
            print(f"\n  STEP 2 FAILED -- 0 outputs fired. v_max={layer['v_max']:.4f}")
    else:
        print(f"\n  STEP 2 FAILED -- error.")

    save_result("2", result)
    return result


# ============================================================
# Load real FC1 data (shared by steps 3a-3d and 4)
# ============================================================
def load_fc1_data():
    """Load FC1 weights, inputs, labels for real data experiments."""
    fc1_weight = np.load(COND_DIR / "fc1_weight_fold4.npy")  # (256, 2304)
    fc1_inputs = np.load(COND_DIR / "fc1_input_features_fold4.npy")  # (20, 25, 2304)
    labels = np.load(COND_DIR / "labels_fold4.npy")  # (20,)
    preds = np.load(COND_DIR / "snntorch_preds_fold4.npy")  # (20,)

    # Also load FC1 bias if available
    bias_path = WEIGHTS_DIR / "fc1_bias.npy"
    fc1_bias = np.load(bias_path) if bias_path.exists() else None

    print(f"  FC1 weights: {fc1_weight.shape}")
    print(f"  FC1 inputs: {fc1_inputs.shape}")
    print(f"  FC1 bias: {fc1_bias.shape if fc1_bias is not None else 'not found'}")
    print(f"  Labels: {labels}")
    print(f"  snnTorch preds: {preds}")

    # Input stats
    sample = fc1_inputs[args.sample_idx]  # (25, 2304)
    active_per_step = (sample > 0.5).sum(axis=1)
    print(f"\n  Sample {args.sample_idx} input stats:")
    print(f"    Active/step: min={active_per_step.min()}, max={active_per_step.max()}, "
          f"mean={active_per_step.mean():.1f}")
    print(f"    Total spikes: {int((sample > 0.5).sum())}")
    print(f"    True label: {labels[args.sample_idx]}, "
          f"snnTorch pred: {preds[args.sample_idx]}")

    return fc1_weight, fc1_inputs, labels, preds, fc1_bias


# ============================================================
# Helper: build FC1 connection list from weight matrix
# ============================================================
def build_connection_list(
    weight_matrix: np.ndarray,
    scale: float = 1.0,
    prune_threshold: float = None,
    delay: float = 1.0,
    max_connections: int = None,
) -> tuple:
    """
    Convert (n_post, n_pre) weight matrix to exc/inh connection lists.

    Uses vectorized numpy for speed (critical for 256x2304 = 589K elements).

    Args:
        weight_matrix: (n_post, n_pre) array
        scale: multiply all weights by this
        prune_threshold: drop connections with |weight*scale| below this
        delay: synaptic delay in ms
        max_connections: if total connections exceed this, prune smallest

    Returns (exc_list, inh_list) where each is list of [pre, post, |weight|, delay].
    """
    if prune_threshold is None:
        prune_threshold = args.prune_threshold

    n_post, n_pre = weight_matrix.shape
    scaled = weight_matrix * scale

    # Vectorized: find all non-zero connections
    post_idx, pre_idx = np.nonzero(np.abs(scaled) > prune_threshold)
    weights = scaled[post_idx, pre_idx]

    # If too many connections, prune the smallest
    if max_connections is not None and len(weights) > max_connections:
        print(f"    WARNING: {len(weights):,} connections exceed max {max_connections:,}. "
              f"Pruning smallest.")
        # Keep the max_connections largest by absolute value
        top_k_idx = np.argsort(np.abs(weights))[-max_connections:]
        post_idx = post_idx[top_k_idx]
        pre_idx = pre_idx[top_k_idx]
        weights = weights[top_k_idx]
        print(f"    After pruning: {len(weights):,} connections. "
              f"Min |w|={np.abs(weights).min():.5f}")

    # Split exc/inh
    exc_mask = weights > 0
    inh_mask = weights < 0

    exc_list = np.column_stack([
        pre_idx[exc_mask],
        post_idx[exc_mask],
        weights[exc_mask],
        np.full(exc_mask.sum(), delay),
    ]).tolist() if exc_mask.any() else []

    inh_list = np.column_stack([
        pre_idx[inh_mask],
        post_idx[inh_mask],
        np.abs(weights[inh_mask]),
        np.full(inh_mask.sum(), delay),
    ]).tolist() if inh_mask.any() else []

    return exc_list, inh_list


# ============================================================
# STEP 3a: FC1 excitatory-only
# ============================================================
def step_3a():
    """
    FC1 (2304 -> 256) with ONLY excitatory weights.
    All negative weights set to 0. This eliminates cancellation entirely.
    Hidden neurons WILL fire (analysis shows 233/256 with scale=1.0).
    """
    print("=" * 60)
    print("STEP 3a: FC1 Excitatory-Only (2304 -> 256)")
    print("=" * 60)
    print("Goal: Get hidden neurons to FIRE by removing all inhibitory weights.")
    print("This is not functionally correct but proves the pipeline works at scale.")
    print()

    fc1_weight, fc1_inputs, labels, preds, fc1_bias = load_fc1_data()

    n_input = 2304
    n_hidden = 256
    scale = args.weight_scale

    # Zero out negative weights
    fc1_exc_only = fc1_weight.copy()
    fc1_exc_only[fc1_exc_only < 0] = 0.0

    total_w = fc1_weight.size
    exc_count = (fc1_weight > 0).sum()
    inh_count = (fc1_weight < 0).sum()
    print(f"  Original: {exc_count} exc ({100*exc_count/total_w:.1f}%), "
          f"{inh_count} inh ({100*inh_count/total_w:.1f}%)")
    print(f"  After zeroing inh: {exc_count} exc remaining")

    # Build connections (only excitatory)
    exc_conns, _ = build_connection_list(
        fc1_exc_only, scale=scale,
        max_connections=args.max_connections
    )
    print(f"  Excitatory connections after pruning: {len(exc_conns):,}")
    print(f"  Weight scale: {scale}")
    print()

    # Get sample input
    sample = fc1_inputs[args.sample_idx]  # (25, 2304)
    input_spike_times = binary_to_spike_times(sample)
    total_input_spikes = sum(len(t) for t in input_spike_times)
    active_neurons = sum(1 for t in input_spike_times if len(t) > 0)
    print(f"  Input sample {args.sample_idx}: {active_neurons} active neurons, "
          f"{total_input_spikes} total spikes")
    print(f"  True label: {labels[args.sample_idx]}, snnTorch: {preds[args.sample_idx]}")
    print()

    layer_configs = [
        {"n": n_hidden, "exc_conns": exc_conns, "inh_conns": []},
    ]

    print(f"  Running on SpiNNaker...")
    result = run_network(input_spike_times, n_input, layer_configs,
                         label="step3a_fc1_exc_only")
    print_layer_summary(result)

    if not result.get("error"):
        layer = result["layers"][0]
        if layer["neurons_fired"] > 0:
            print(f"\n  STEP 3a PASSED -- {layer['neurons_fired']}/256 hidden neurons fired!")
            print(f"  Total hidden spikes: {layer['total_spikes']}")
            print(f"  This proves FC1 can work on SpiNNaker when cancellation is avoided.")
        else:
            print(f"\n  STEP 3a FAILED -- 0 hidden neurons fired. v_max={layer['v_max']:.4f}")
            print(f"  Try increasing --weight-scale (current: {scale})")

    result["true_label"] = int(labels[args.sample_idx])
    result["snntorch_pred"] = int(preds[args.sample_idx])
    result["strategy"] = "excitatory_only"
    result["weight_scale"] = scale
    save_result("3a", result)
    return result


# ============================================================
# STEP 3b: FC1 top-K connections
# ============================================================
def step_3b():
    """
    FC1 (2304 -> 256) using only the TOP-K strongest connections per hidden neuron.
    Both excitatory and inhibitory are kept, but total fan-in is limited.
    """
    print("=" * 60)
    print(f"STEP 3b: FC1 Top-K Connections (K={args.top_k})")
    print("=" * 60)
    print(f"Goal: Reduce fan-in to {args.top_k} connections per hidden neuron.")
    print("Keeps both exc and inh but limits total input density.")
    print()

    fc1_weight, fc1_inputs, labels, preds, fc1_bias = load_fc1_data()

    n_input = 2304
    n_hidden = 256
    scale = args.weight_scale
    K = args.top_k

    # For each hidden neuron, keep only top-K by absolute weight
    fc1_topk = np.zeros_like(fc1_weight)
    for h in range(n_hidden):
        w_h = fc1_weight[h, :]
        top_k_idx = np.argsort(np.abs(w_h))[-K:]
        fc1_topk[h, top_k_idx] = w_h[top_k_idx]

    kept = (fc1_topk != 0).sum()
    exc_kept = (fc1_topk > 0).sum()
    inh_kept = (fc1_topk < 0).sum()
    print(f"  Top-K={K}: {kept} connections kept ({exc_kept} exc, {inh_kept} inh)")
    print(f"  Mean |weight| of kept: {np.abs(fc1_topk[fc1_topk != 0]).mean():.5f}")
    print(f"  Mean |weight| of all:  {np.abs(fc1_weight[fc1_weight != 0]).mean():.5f}")

    # Estimate: with K inputs per neuron, net current
    currents = []
    for h in range(n_hidden):
        w_h = fc1_topk[h, :]
        # Assume all active (K inputs, some subset will be active)
        net = w_h[w_h != 0].sum() * scale
        currents.append(net)
    currents = np.array(currents)
    print(f"  Net current (all K fire): mean={currents.mean():.3f}, "
          f">{LIF_PARAMS['v_thresh']}: {(currents > LIF_PARAMS['v_thresh']).sum()}/256")

    exc_conns, inh_conns = build_connection_list(
        fc1_topk, scale=scale,
        max_connections=args.max_connections
    )
    print(f"  Connections: {len(exc_conns):,} exc + {len(inh_conns):,} inh")
    print()

    sample = fc1_inputs[args.sample_idx]
    input_spike_times = binary_to_spike_times(sample)
    total_input_spikes = sum(len(t) for t in input_spike_times)
    print(f"  Sample {args.sample_idx}: {total_input_spikes} input spikes")
    print()

    layer_configs = [
        {"n": n_hidden, "exc_conns": exc_conns, "inh_conns": inh_conns},
    ]

    print("  Running on SpiNNaker...")
    result = run_network(input_spike_times, n_input, layer_configs,
                         label=f"step3b_fc1_topk{K}")
    print_layer_summary(result)

    if not result.get("error"):
        layer = result["layers"][0]
        if layer["neurons_fired"] > 0:
            print(f"\n  STEP 3b PASSED -- {layer['neurons_fired']}/256 hidden neurons fired!")
        else:
            print(f"\n  STEP 3b FAILED -- 0 hidden neurons fired. v_max={layer['v_max']:.4f}")

    result["true_label"] = int(labels[args.sample_idx])
    result["snntorch_pred"] = int(preds[args.sample_idx])
    result["strategy"] = f"top_k_{K}"
    result["weight_scale"] = scale
    result["top_k"] = K
    save_result("3b", result)
    return result


# ============================================================
# STEP 3c: FC1 with bias injection
# ============================================================
def step_3c():
    """
    FC1 (2304 -> 256) with ALL weights but a constant excitatory bias current
    injected into each hidden neuron to overcome inhibitory cancellation.
    """
    print("=" * 60)
    print(f"STEP 3c: FC1 with Bias Current Injection ({args.bias_current} nA)")
    print("=" * 60)
    print("Goal: Keep all weights but add constant excitatory current to hidden neurons.")
    print("IF_curr_exp supports i_offset parameter for this.")
    print()

    fc1_weight, fc1_inputs, labels, preds, fc1_bias = load_fc1_data()

    n_input = 2304
    n_hidden = 256
    scale = args.weight_scale
    bias = args.bias_current

    exc_conns, inh_conns = build_connection_list(
        fc1_weight, scale=scale,
        max_connections=args.max_connections
    )
    print(f"  Connections: {len(exc_conns):,} exc + {len(inh_conns):,} inh")
    print(f"  Bias current: {bias} nA (i_offset)")
    print(f"  Weight scale: {scale}")

    # Estimate effect of bias
    # With tau_m=20ms, the steady-state voltage from i_offset alone is:
    # V_ss = v_rest + i_offset * tau_m / cm = 0 + bias * 20 / 1 = bias * 20
    v_ss_bias = bias * LIF_PARAMS["tau_m"] / LIF_PARAMS["cm"]
    print(f"  Bias-only steady-state V: {v_ss_bias:.2f} (threshold={LIF_PARAMS['v_thresh']})")
    if v_ss_bias >= LIF_PARAMS["v_thresh"]:
        print(f"  WARNING: Bias alone exceeds threshold -- all neurons will fire regardless.")
        print(f"  This will produce spikes but no discrimination. Reduce --bias-current.")
    print()

    sample = fc1_inputs[args.sample_idx]
    input_spike_times = binary_to_spike_times(sample)
    total_input_spikes = sum(len(t) for t in input_spike_times)
    print(f"  Sample {args.sample_idx}: {total_input_spikes} input spikes")
    print()

    # Use i_offset in LIF params for hidden layer
    hidden_params = LIF_PARAMS.copy()
    # Note: i_offset is handled in run_network via the i_offset key

    layer_configs = [
        {"n": n_hidden, "exc_conns": exc_conns, "inh_conns": inh_conns,
         "i_offset": bias},
    ]

    print("  Running on SpiNNaker...")
    result = run_network(input_spike_times, n_input, layer_configs,
                         label=f"step3c_fc1_bias{bias}")
    print_layer_summary(result)

    if not result.get("error"):
        layer = result["layers"][0]
        if layer["neurons_fired"] > 0:
            print(f"\n  STEP 3c PASSED -- {layer['neurons_fired']}/256 hidden neurons fired!")
            if layer["neurons_fired"] == n_hidden:
                print("  NOTE: ALL neurons fired -- bias may be too strong for discrimination.")
        else:
            print(f"\n  STEP 3c FAILED -- 0 hidden neurons fired. v_max={layer['v_max']:.4f}")

    result["true_label"] = int(labels[args.sample_idx])
    result["snntorch_pred"] = int(preds[args.sample_idx])
    result["strategy"] = f"bias_injection_{bias}"
    result["weight_scale"] = scale
    result["bias_current"] = bias
    save_result("3c", result)
    return result


# ============================================================
# STEP 3d: FC1 with temporal separation
# ============================================================
def step_3d():
    """
    FC1 (2304 -> 256) with temporal separation of exc/inh.
    Excitatory connections use delay=1.0ms (arrive first).
    Inhibitory connections use delay=5.0ms (arrive later).
    This lets excitatory current build up membrane potential before
    inhibition kicks in. With tau_syn=5.0ms, the excitatory PSP peaks
    at ~2ms and starts decaying. Inhibition arrives at 5ms.
    """
    print("=" * 60)
    print("STEP 3d: FC1 with Temporal Separation (exc first, inh delayed)")
    print("=" * 60)
    print("Goal: Let excitatory current build membrane potential before inhibition arrives.")
    print("Exc delay=1.0ms, Inh delay=5.0ms. tau_syn=5.0ms -> PSP peaks ~2ms after spike.")
    print()

    fc1_weight, fc1_inputs, labels, preds, fc1_bias = load_fc1_data()

    n_input = 2304
    n_hidden = 256
    scale = args.weight_scale

    # Build exc/inh connections with DIFFERENT delays
    # Excitatory: delay=1.0ms (arrive first)
    # Inhibitory: delay=5.0ms (arrive later, letting exc build up)
    fc1_exc_part = fc1_weight.copy()
    fc1_exc_part[fc1_exc_part < 0] = 0.0
    fc1_inh_part = fc1_weight.copy()
    fc1_inh_part[fc1_inh_part > 0] = 0.0
    fc1_inh_part = np.abs(fc1_inh_part)  # make positive for inh connection list

    exc_conns, _ = build_connection_list(
        fc1_exc_part, scale=scale, delay=1.0,
        max_connections=args.max_connections
    )
    # For inh, we need to build as all-positive weights but mark as inhibitory
    # build_connection_list treats positive weights as exc, so we pass inh_part
    # (already positive) and take the "exc" output, then use it as inh
    inh_as_exc, _ = build_connection_list(
        fc1_inh_part, scale=scale, delay=5.0,
        max_connections=args.max_connections
    )
    inh_conns = inh_as_exc  # these go into receptor_type="inhibitory"

    print(f"  Connections: {len(exc_conns):,} exc (delay=1.0ms) + "
          f"{len(inh_conns):,} inh (delay=5.0ms)")
    print(f"  Weight scale: {scale}")
    print()

    sample = fc1_inputs[args.sample_idx]
    input_spike_times = binary_to_spike_times(sample)
    total_input_spikes = sum(len(t) for t in input_spike_times)
    print(f"  Sample {args.sample_idx}: {total_input_spikes} input spikes")
    print()

    layer_configs = [
        {"n": n_hidden, "exc_conns": exc_conns, "inh_conns": inh_conns},
    ]

    print("  Running on SpiNNaker...")
    result = run_network(input_spike_times, n_input, layer_configs,
                         label="step3d_fc1_temporal")
    print_layer_summary(result)

    if not result.get("error"):
        layer = result["layers"][0]
        if layer["neurons_fired"] > 0:
            print(f"\n  STEP 3d PASSED -- {layer['neurons_fired']}/256 hidden neurons fired!")
        else:
            print(f"\n  STEP 3d FAILED -- 0 hidden neurons fired. v_max={layer['v_max']:.4f}")

    result["true_label"] = int(labels[args.sample_idx])
    result["snntorch_pred"] = int(preds[args.sample_idx])
    result["strategy"] = "temporal_separation"
    result["weight_scale"] = scale
    save_result("3d", result)
    return result


# ============================================================
# STEP 4: FC1 -> FC2 chain (end-to-end)
# ============================================================
def step_4():
    """
    Full FC1 -> FC2 chain on SpiNNaker. Uses the best FC1 strategy
    from step 3 and chains it with real FC2 weights.
    Measures accuracy vs snnTorch on multiple samples.
    """
    print("=" * 60)
    print(f"STEP 4: FC1 -> FC2 Chain (strategy={args.fc1_strategy})")
    print("=" * 60)
    print("Goal: End-to-end FC1->FC2 inference on SpiNNaker. Measure accuracy.")
    print()

    fc1_weight, fc1_inputs, labels, preds, fc1_bias = load_fc1_data()

    # Load FC2 connections (already in [pre, post, weight, delay] format)
    fc2_conn_path = WEIGHTS_DIR / "fc2_connections.npy"
    fc2_all = np.load(fc2_conn_path)
    print(f"  FC2 connections: {fc2_all.shape[0]}")

    n_input = 2304
    n_hidden = 256
    n_output = 50
    scale = args.weight_scale

    # Build FC1 connections based on strategy
    print(f"\n  FC1 strategy: {args.fc1_strategy}")

    if args.fc1_strategy == "exc_only":
        fc1_mod = fc1_weight.copy()
        fc1_mod[fc1_mod < 0] = 0.0
        fc1_exc, fc1_inh = build_connection_list(
            fc1_mod, scale=scale, max_connections=args.max_connections
        )
        strategy_desc = "excitatory_only"

    elif args.fc1_strategy == "top_k":
        K = args.top_k
        fc1_mod = np.zeros_like(fc1_weight)
        for h in range(n_hidden):
            top_k_idx = np.argsort(np.abs(fc1_weight[h, :]))[-K:]
            fc1_mod[h, top_k_idx] = fc1_weight[h, top_k_idx]
        fc1_exc, fc1_inh = build_connection_list(
            fc1_mod, scale=scale, max_connections=args.max_connections
        )
        strategy_desc = f"top_k_{K}"

    elif args.fc1_strategy == "bias":
        fc1_exc, fc1_inh = build_connection_list(
            fc1_weight, scale=scale, max_connections=args.max_connections
        )
        strategy_desc = f"bias_{args.bias_current}"

    elif args.fc1_strategy == "temporal":
        # Exc with delay=1.0, Inh with delay=5.0
        fc1_exc_part = fc1_weight.copy()
        fc1_exc_part[fc1_exc_part < 0] = 0.0
        fc1_inh_part = np.abs(fc1_weight.copy())
        fc1_inh_part[fc1_weight > 0] = 0.0  # zero out where original was positive

        fc1_exc, _ = build_connection_list(
            fc1_exc_part, scale=scale, delay=1.0,
            max_connections=args.max_connections
        )
        fc1_inh_as_exc, _ = build_connection_list(
            fc1_inh_part, scale=scale, delay=5.0,
            max_connections=args.max_connections
        )
        fc1_inh = fc1_inh_as_exc
        strategy_desc = "temporal_separation"
    else:
        print(f"  ERROR: Unknown strategy: {args.fc1_strategy}")
        return

    print(f"  FC1: {len(fc1_exc)} exc + {len(fc1_inh)} inh connections")

    # Build FC2 connections (scale and split)
    fc2_pruned = fc2_all[np.abs(fc2_all[:, 2]) > 0.01].copy()
    fc2_scaled = fc2_pruned.copy()
    fc2_scaled[:, 2] *= scale

    fc2_exc = fc2_scaled[fc2_scaled[:, 2] > 0].tolist()
    fc2_inh_data = fc2_scaled[fc2_scaled[:, 2] < 0].copy()
    fc2_inh_data[:, 2] = np.abs(fc2_inh_data[:, 2])
    fc2_inh = fc2_inh_data.tolist()

    print(f"  FC2: {len(fc2_exc)} exc + {len(fc2_inh)} inh connections")
    print()

    # Layer configs
    fc1_layer = {
        "n": n_hidden,
        "exc_conns": fc1_exc,
        "inh_conns": fc1_inh,
    }
    if args.fc1_strategy == "bias":
        fc1_layer["i_offset"] = args.bias_current

    fc2_layer = {
        "n": n_output,
        "exc_conns": fc2_exc,
        "inh_conns": fc2_inh,
    }

    layer_configs = [fc1_layer, fc2_layer]

    # Run on multiple samples
    num_samples = min(args.num_samples, len(labels))
    results_list = []
    correct = 0
    snn_correct = 0

    for idx in range(num_samples):
        print(f"\n{'='*50}")
        print(f"Sample {idx}/{num_samples}  (true={labels[idx]}, snn_pred={preds[idx]})")
        print(f"{'='*50}")

        sample = fc1_inputs[idx]  # (25, 2304)
        input_spike_times = binary_to_spike_times(sample)

        result = run_network(input_spike_times, n_input, layer_configs,
                             label=f"step4_sample{idx}")
        print_layer_summary(result)

        if not result.get("error") and len(result.get("layers", [])) == 2:
            hidden_layer = result["layers"][0]
            output_layer = result["layers"][1]

            predicted = output_layer.get("predicted", -1)
            true_label = int(labels[idx])
            is_correct = (predicted == true_label)
            snn_is_correct = (int(preds[idx]) == true_label)

            if is_correct:
                correct += 1
            if snn_is_correct:
                snn_correct += 1

            sample_result = {
                "sample": idx,
                "true_label": true_label,
                "snntorch_pred": int(preds[idx]),
                "snntorch_correct": snn_is_correct,
                "spinnaker_pred": predicted,
                "spinnaker_correct": is_correct,
                "hidden_neurons_fired": hidden_layer["neurons_fired"],
                "hidden_total_spikes": hidden_layer["total_spikes"],
                "output_neurons_fired": output_layer["neurons_fired"],
                "output_total_spikes": output_layer["total_spikes"],
                "classification_basis": output_layer.get("classification_basis", "?"),
            }
            results_list.append(sample_result)

            print(f"  => pred={predicted}, true={true_label}, correct={is_correct}")
            print(f"     hidden: {hidden_layer['neurons_fired']} fired, "
                  f"{hidden_layer['total_spikes']} spikes")
            print(f"     output: {output_layer['neurons_fired']} fired, "
                  f"{output_layer['total_spikes']} spikes")
        else:
            results_list.append({
                "sample": idx,
                "error": result.get("error", "unknown"),
            })

    # Summary
    accuracy = correct / num_samples if num_samples > 0 else 0.0
    snn_accuracy = snn_correct / num_samples if num_samples > 0 else 0.0

    print("\n" + "=" * 60)
    print("STEP 4 RESULTS: FC1 -> FC2 End-to-End")
    print("=" * 60)
    print(f"  Strategy:     {strategy_desc}")
    print(f"  Weight scale: {scale}")
    print(f"  Samples:      {num_samples}")
    print(f"  SpiNNaker:    {correct}/{num_samples} = {accuracy:.1%}")
    print(f"  snnTorch:     {snn_correct}/{num_samples} = {snn_accuracy:.1%}")
    print()

    print(f"  {'Sample':>6}  {'True':>4}  {'SNN':>3}  {'Spk':>3}  "
          f"{'HidFired':>8}  {'OutFired':>8}  {'Correct':>7}")
    print("  " + "-" * 55)
    for r in results_list:
        if "error" in r and r.get("error"):
            print(f"  {r['sample']:>6}  ERROR: {r['error'][:40]}")
        else:
            print(f"  {r['sample']:>6}  {r['true_label']:>4}  "
                  f"{r['snntorch_pred']:>3}  {r['spinnaker_pred']:>3}  "
                  f"{r['hidden_neurons_fired']:>8}  {r['output_neurons_fired']:>8}  "
                  f"{str(r['spinnaker_correct']):>7}")

    full_result = {
        "step": 4,
        "strategy": strategy_desc,
        "weight_scale": scale,
        "lif_params": LIF_PARAMS,
        "num_samples": num_samples,
        "spinnaker_correct": correct,
        "spinnaker_accuracy": accuracy,
        "snntorch_correct": snn_correct,
        "snntorch_accuracy": snn_accuracy,
        "per_sample": results_list,
    }
    if args.fc1_strategy == "bias":
        full_result["bias_current"] = args.bias_current
    if args.fc1_strategy == "top_k":
        full_result["top_k"] = args.top_k

    save_result("4", full_result)
    return full_result


# ============================================================
# Main dispatch
# ============================================================
STEPS = {
    "1": step_1,
    "2": step_2,
    "3a": step_3a,
    "3b": step_3b,
    "3c": step_3c,
    "3d": step_3d,
    "4": step_4,
}

print(f"[{ts()}] spinnaker_incremental.py -- step {args.step}")
print(f"  Weight scale: {args.weight_scale}")
print(f"  Results dir: {RESULTS_DIR}")
print()

t_start = time.time()
result = STEPS[args.step]()
elapsed = time.time() - t_start

print(f"\n[{ts()}] Step {args.step} complete. Wall clock: {elapsed:.1f}s")
