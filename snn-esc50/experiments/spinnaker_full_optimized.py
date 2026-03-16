"""
spinnaker_full_optimized.py -- Optimized full FC1+FC2 SpiNNaker deployment.

Tests 4 strategies for adding inhibitory weights back to FC1 while staying
under the ~25K connection limit per layer:

  A. Mixed top-K:  top-20 exc + top-20 inh per neuron = 40 conns/neuron
                   x 256 = 10,240 total FC1 connections (under limit)
  B. Temporal separation:  exc delay=1ms, inh delay=3ms. Excitation builds
                           before inhibition arrives.
  C. Balanced pruning:  top-K by absolute magnitude regardless of sign.
                        K=80 per neuron = 20,480 total.
  D. Lower threshold:   v_thresh=0.5 with balanced top-K=60 = 15,360 total.
                         Lower threshold compensates for sparse connectivity.

Phase 1: Test all 4 strategies on sample 0 (find which gets hidden neurons
          firing with mixed exc/inh and correct discrimination).
Phase 2: Run best strategy on 20 samples.
Phase 3: Compute accuracy, compare to FC2-only baseline (43% on fold 4).

CRITICAL SpiNNaker lessons learned:
  - ALWAYS use hidden_pop.initialize(v=0.0) -- sPyNNaker defaults v to v_rest,
    and at larger population sizes defaults to -65.0 even when v_rest=0.0.
  - ALWAYS call sim.end() in try/finally.
  - Keep total connections under ~25K per layer to avoid routing congestion.
  - Use IF_curr_exp, NOT IF_cond_exp.
  - FC2 weight_scale=5.0 is the known-good value from FC2-only runs.

Usage (always from .venv-spinnaker, Python 3.11):
    cd snn-esc50
    source .venv-spinnaker/bin/activate

    # Phase 1: Compare all strategies on sample 0
    python experiments/spinnaker_full_optimized.py --phase 1

    # Phase 2+3: Run best strategy on 20 samples
    python experiments/spinnaker_full_optimized.py --phase 2

    # Both phases
    python experiments/spinnaker_full_optimized.py --phase all
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
COND_DIR = REPO_ROOT / "results" / "spinnaker_results" / "full_deploy_cond"
RESULTS_DIR = REPO_ROOT / "results" / "spinnaker_results" / "full_deploy_optimized"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
# Constants
# ============================================================
NUM_STEPS = 25
DT = 1.0
N_INPUT = 2304
N_HIDDEN = 256
N_OUTPUT = 50
MAX_CONNS_PER_LAYER = 25000

# Calibrated neuron params (working config from incremental debugging)
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


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# Data loading
# ============================================================
def load_data():
    """Load FC1/FC2 weights, input features, and labels."""
    fc1_weight = np.load(COND_DIR / "fc1_weight_fold4.npy")   # (256, 2304)
    fc2_weight = np.load(COND_DIR / "fc2_weight_fold4.npy")   # (50, 256)
    fc1_inputs = np.load(COND_DIR / "fc1_input_features_fold4.npy")  # (N, 25, 2304)
    labels = np.load(COND_DIR / "labels_fold4.npy")            # (N,)

    # Also try to load snnTorch predictions for comparison
    preds_path = COND_DIR / "snntorch_preds_fold4.npy"
    preds = np.load(preds_path) if preds_path.exists() else None

    print(f"  FC1 weights: {fc1_weight.shape} "
          f"(exc: {(fc1_weight > 0).sum()}, inh: {(fc1_weight < 0).sum()})")
    print(f"  FC2 weights: {fc2_weight.shape}")
    print(f"  Input features: {fc1_inputs.shape}")
    print(f"  Labels: {labels.shape} (unique: {np.unique(labels).shape[0]})")
    if preds is not None:
        print(f"  snnTorch preds available: {preds.shape}")

    return fc1_weight, fc2_weight, fc1_inputs, labels, preds


# ============================================================
# Helper: binary matrix -> spike times
# ============================================================
def binary_to_spike_times(binary_matrix):
    """Convert (T, N) matrix to list of spike time lists for SpikeSourceArray.

    Values > 0.5 are treated as spikes. Spike times are in ms, starting at 0.
    """
    T, N = binary_matrix.shape
    spike_times_list = []
    for n in range(N):
        times = np.where(binary_matrix[:, n] > 0.5)[0].astype(float).tolist()
        spike_times_list.append(times)
    return spike_times_list


# ============================================================
# Connection-building helpers
# ============================================================
def build_topk_mixed(weight_matrix, k_exc, k_inh, scale=1.0):
    """Strategy A: top-K excitatory + top-K inhibitory per post-synaptic neuron.

    Returns (exc_list, inh_list) each as list of [pre, post, |weight|, delay].
    """
    n_post, n_pre = weight_matrix.shape
    exc_list = []
    inh_list = []

    for post in range(n_post):
        w = weight_matrix[post, :]

        # Top-K excitatory (largest positive weights)
        exc_mask = w > 0
        exc_indices = np.where(exc_mask)[0]
        if len(exc_indices) > k_exc:
            top_exc = exc_indices[np.argsort(w[exc_indices])[-k_exc:]]
        else:
            top_exc = exc_indices

        for pre in top_exc:
            exc_list.append([int(pre), int(post), float(w[pre] * scale), 1.0])

        # Top-K inhibitory (largest magnitude negative weights)
        inh_mask = w < 0
        inh_indices = np.where(inh_mask)[0]
        if len(inh_indices) > k_inh:
            # Sort by magnitude (most negative first), take top k_inh
            top_inh = inh_indices[np.argsort(w[inh_indices])[:k_inh]]
        else:
            top_inh = inh_indices

        for pre in top_inh:
            inh_list.append([int(pre), int(post), float(abs(w[pre]) * scale), 1.0])

    return exc_list, inh_list


def build_topk_mixed_temporal(weight_matrix, k_exc, k_inh, scale=1.0,
                               exc_delay=1.0, inh_delay=3.0):
    """Strategy B: Same as A but exc at delay=1ms, inh at delay=3ms.

    Temporal separation lets excitation build membrane potential before
    inhibition kicks in.
    """
    n_post, n_pre = weight_matrix.shape
    exc_list = []
    inh_list = []

    for post in range(n_post):
        w = weight_matrix[post, :]

        exc_indices = np.where(w > 0)[0]
        if len(exc_indices) > k_exc:
            top_exc = exc_indices[np.argsort(w[exc_indices])[-k_exc:]]
        else:
            top_exc = exc_indices

        for pre in top_exc:
            exc_list.append([int(pre), int(post), float(w[pre] * scale), exc_delay])

        inh_indices = np.where(w < 0)[0]
        if len(inh_indices) > k_inh:
            top_inh = inh_indices[np.argsort(w[inh_indices])[:k_inh]]
        else:
            top_inh = inh_indices

        for pre in top_inh:
            inh_list.append([int(pre), int(post), float(abs(w[pre]) * scale), inh_delay])

    return exc_list, inh_list


def build_topk_balanced(weight_matrix, k_total, scale=1.0):
    """Strategy C: top-K by absolute magnitude per neuron, regardless of sign.

    This keeps the most important connections whether they are exc or inh.
    """
    n_post, n_pre = weight_matrix.shape
    exc_list = []
    inh_list = []

    for post in range(n_post):
        w = weight_matrix[post, :]
        abs_w = np.abs(w)

        # Non-zero connections sorted by magnitude
        nonzero = np.where(abs_w > 0)[0]
        if len(nonzero) > k_total:
            top_k = nonzero[np.argsort(abs_w[nonzero])[-k_total:]]
        else:
            top_k = nonzero

        for pre in top_k:
            if w[pre] > 0:
                exc_list.append([int(pre), int(post), float(w[pre] * scale), 1.0])
            else:
                inh_list.append([int(pre), int(post), float(abs(w[pre]) * scale), 1.0])

    return exc_list, inh_list


def build_fc2_connections(fc2_weight, scale=5.0, prune_threshold=0.01):
    """Build FC2 connection lists from weight matrix.

    FC2 uses the known-good weight_scale=5.0 from FC2-only runs.
    256 -> 50 = 12,800 max connections (well under 25K limit).
    """
    n_post, n_pre = fc2_weight.shape
    exc_list = []
    inh_list = []

    for post in range(n_post):
        for pre in range(n_pre):
            w = fc2_weight[post, pre] * scale
            if abs(w) < prune_threshold:
                continue
            if w > 0:
                exc_list.append([int(pre), int(post), float(w), 1.0])
            else:
                inh_list.append([int(pre), int(post), float(abs(w)), 1.0])

    return exc_list, inh_list


# ============================================================
# Core SpiNNaker inference: run one sample through FC1+FC2
# ============================================================
def run_fc1_fc2_sample(
    input_spike_times,
    fc1_exc, fc1_inh,
    fc2_exc, fc2_inh,
    lif_params=None,
    run_time=NUM_STEPS,
    label="sample",
):
    """Run one sample through FC1->FC2 on SpiNNaker.

    Returns dict with hidden/output spike counts, prediction, timing.
    """
    if lif_params is None:
        lif_params = LIF_PARAMS

    t0 = time.time()
    result = {
        "label": label,
        "error": None,
    }

    sim_started = False
    try:
        sim.setup(timestep=DT)
        sim_started = True

        # ----- Input population (2304 SpikeSourceArray neurons) -----
        input_pop = sim.Population(
            N_INPUT,
            sim.SpikeSourceArray,
            {"spike_times": input_spike_times},
            label="input"
        )
        input_pop.record("spikes")

        # ----- Hidden population (256 IF_curr_exp neurons) -----
        hidden_pop = sim.Population(
            N_HIDDEN,
            sim.IF_curr_exp(**lif_params),
            label="hidden"
        )
        hidden_pop.record(["spikes", "v"])
        # CRITICAL: force v=0.0 explicitly (sPyNNaker default may be -65)
        hidden_pop.initialize(v=0.0)

        # FC1 projections
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

        # ----- Output population (50 IF_curr_exp neurons) -----
        output_pop = sim.Population(
            N_OUTPUT,
            sim.IF_curr_exp(**lif_params),
            label="output"
        )
        output_pop.record(["spikes", "v"])
        # CRITICAL: force v=0.0 explicitly
        output_pop.initialize(v=0.0)

        # FC2 projections
        if fc2_exc:
            sim.Projection(
                hidden_pop, output_pop,
                sim.FromListConnector(fc2_exc),
                receptor_type="excitatory"
            )
        if fc2_inh:
            sim.Projection(
                hidden_pop, output_pop,
                sim.FromListConnector(fc2_inh),
                receptor_type="inhibitory"
            )

        # ----- Run simulation -----
        sim.run(run_time)

        # ----- Extract input spike delivery -----
        in_data = input_pop.get_data("spikes")
        in_trains = in_data.segments[0].spiketrains
        result["input_spikes_delivered"] = sum(len(st) for st in in_trains)
        result["input_spikes_expected"] = sum(len(t) for t in input_spike_times)

        # ----- Extract hidden layer results -----
        hid_data = hidden_pop.get_data(["spikes", "v"])
        hid_trains = hid_data.segments[0].spiketrains
        hid_v = hid_data.segments[0].filter(name="v")[0]

        hid_spike_counts = [len(st) for st in hid_trains]
        hid_fired_ids = [j for j, c in enumerate(hid_spike_counts) if c > 0]

        try:
            hid_v_arr = hid_v.magnitude  # (T, N_HIDDEN)
            hid_v_max = float(hid_v_arr.max())
        except Exception:
            hid_v_max = 0.0

        result["hidden"] = {
            "neurons_fired": len(hid_fired_ids),
            "total_spikes": sum(hid_spike_counts),
            "fired_ids": hid_fired_ids[:30],
            "v_max": hid_v_max,
            "spike_counts": hid_spike_counts,
        }

        # ----- Extract output layer results -----
        out_data = output_pop.get_data(["spikes", "v"])
        out_trains = out_data.segments[0].spiketrains
        out_v = out_data.segments[0].filter(name="v")[0]

        out_spike_counts = [len(st) for st in out_trains]
        out_fired_ids = [j for j, c in enumerate(out_spike_counts) if c > 0]

        try:
            out_v_arr = out_v.magnitude  # (T, N_OUTPUT)
            out_v_max = float(out_v_arr.max())
            out_v_sum = out_v_arr.sum(axis=0).tolist()
        except Exception:
            out_v_max = 0.0
            out_v_sum = [0.0] * N_OUTPUT

        result["output"] = {
            "neurons_fired": len(out_fired_ids),
            "total_spikes": sum(out_spike_counts),
            "fired_ids": out_fired_ids,
            "spike_counts": out_spike_counts,
            "v_max": out_v_max,
            "v_sum": out_v_sum,
        }

        # ----- Prediction -----
        total_out_spikes = sum(out_spike_counts)
        if total_out_spikes > 0:
            result["predicted"] = int(np.argmax(out_spike_counts))
            result["classification_basis"] = "spike_count"
        else:
            # Fallback to voltage sum (membrane potential accumulation)
            result["predicted"] = int(np.argmax(out_v_sum))
            result["classification_basis"] = "voltage_sum"

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        traceback.print_exc()
    finally:
        if sim_started:
            try:
                sim.end()
            except Exception:
                pass  # sim.end() failure should not mask the original error

    result["wall_clock_s"] = time.time() - t0
    return result


# ============================================================
# Strategy definitions
# ============================================================
STRATEGIES = {}


def register_strategy(name, description, builder_fn, fc1_scale, fc2_scale,
                       lif_override=None):
    """Register a connection-building strategy."""
    STRATEGIES[name] = {
        "name": name,
        "description": description,
        "builder_fn": builder_fn,
        "fc1_scale": fc1_scale,
        "fc2_scale": fc2_scale,
        "lif_override": lif_override,
    }


def build_strategy_A(fc1_weight, fc2_weight):
    """A: Mixed top-K (20 exc + 20 inh per neuron)."""
    fc1_exc, fc1_inh = build_topk_mixed(
        fc1_weight, k_exc=20, k_inh=20,
        scale=STRATEGIES["A"]["fc1_scale"]
    )
    fc2_exc, fc2_inh = build_fc2_connections(
        fc2_weight, scale=STRATEGIES["A"]["fc2_scale"]
    )
    return fc1_exc, fc1_inh, fc2_exc, fc2_inh


def build_strategy_B(fc1_weight, fc2_weight):
    """B: Temporal separation (exc delay=1ms, inh delay=3ms)."""
    fc1_exc, fc1_inh = build_topk_mixed_temporal(
        fc1_weight, k_exc=20, k_inh=20,
        scale=STRATEGIES["B"]["fc1_scale"],
        exc_delay=1.0, inh_delay=3.0,
    )
    fc2_exc, fc2_inh = build_fc2_connections(
        fc2_weight, scale=STRATEGIES["B"]["fc2_scale"]
    )
    return fc1_exc, fc1_inh, fc2_exc, fc2_inh


def build_strategy_C(fc1_weight, fc2_weight):
    """C: Balanced pruning, top-80 by absolute magnitude per neuron."""
    fc1_exc, fc1_inh = build_topk_balanced(
        fc1_weight, k_total=80,
        scale=STRATEGIES["C"]["fc1_scale"]
    )
    fc2_exc, fc2_inh = build_fc2_connections(
        fc2_weight, scale=STRATEGIES["C"]["fc2_scale"]
    )
    return fc1_exc, fc1_inh, fc2_exc, fc2_inh


def build_strategy_D(fc1_weight, fc2_weight):
    """D: Lower threshold (0.5) with balanced top-60 per neuron."""
    fc1_exc, fc1_inh = build_topk_balanced(
        fc1_weight, k_total=60,
        scale=STRATEGIES["D"]["fc1_scale"]
    )
    fc2_exc, fc2_inh = build_fc2_connections(
        fc2_weight, scale=STRATEGIES["D"]["fc2_scale"]
    )
    return fc1_exc, fc1_inh, fc2_exc, fc2_inh


# Register all strategies
# FC1 scale=1.0 worked for exc-only in incremental debugging.
# With inhibition added, we may need slightly higher scale since
# inhibition subtracts from membrane potential. Start at 1.0.
register_strategy(
    "A", "Mixed top-K (20 exc + 20 inh per neuron, 10240 FC1 conns)",
    build_strategy_A, fc1_scale=1.0, fc2_scale=5.0
)
register_strategy(
    "B", "Temporal separation (exc delay=1ms, inh delay=3ms, 10240 FC1 conns)",
    build_strategy_B, fc1_scale=1.0, fc2_scale=5.0
)
register_strategy(
    "C", "Balanced pruning (top-80 by |weight|, 20480 FC1 conns)",
    build_strategy_C, fc1_scale=1.0, fc2_scale=5.0
)
register_strategy(
    "D", "Lower threshold=0.5 + balanced top-60 (15360 FC1 conns)",
    build_strategy_D, fc1_scale=1.0, fc2_scale=5.0,
    lif_override={"v_thresh": 0.5}
)


# ============================================================
# Phase 1: Compare all strategies on sample 0
# ============================================================
def phase1_compare_strategies(fc1_weight, fc2_weight, fc1_inputs, labels, preds):
    """Test all 4 strategies on sample 0. Find best one."""
    print("=" * 70)
    print("PHASE 1: Compare Strategies A-D on Sample 0")
    print("=" * 70)

    sample_idx = 0
    sample = fc1_inputs[sample_idx]
    input_spike_times = binary_to_spike_times(sample)
    total_input = sum(len(t) for t in input_spike_times)
    active_inputs = sum(1 for t in input_spike_times if len(t) > 0)

    true_label = int(labels[sample_idx])
    snn_pred = int(preds[sample_idx]) if preds is not None else -1

    print(f"\n  Sample 0: true_label={true_label}, snnTorch_pred={snn_pred}")
    print(f"  Input: {active_inputs} active neurons, {total_input} total spikes")
    print()

    strategy_results = {}

    for name in ["A", "B", "C", "D"]:
        strat = STRATEGIES[name]
        print(f"\n{'='*60}")
        print(f"  Strategy {name}: {strat['description']}")
        print(f"{'='*60}")

        # Build connections
        t_build = time.time()
        fc1_exc, fc1_inh, fc2_exc, fc2_inh = strat["builder_fn"](fc1_weight, fc2_weight)
        build_time = time.time() - t_build

        fc1_total = len(fc1_exc) + len(fc1_inh)
        fc2_total = len(fc2_exc) + len(fc2_inh)
        print(f"  FC1: {len(fc1_exc)} exc + {len(fc1_inh)} inh = {fc1_total} total")
        print(f"  FC2: {len(fc2_exc)} exc + {len(fc2_inh)} inh = {fc2_total} total")
        print(f"  Build time: {build_time:.1f}s")

        if fc1_total > MAX_CONNS_PER_LAYER:
            print(f"  WARNING: FC1 connections {fc1_total} > limit {MAX_CONNS_PER_LAYER}")

        # Set up LIF params
        lif = LIF_PARAMS.copy()
        if strat["lif_override"]:
            lif.update(strat["lif_override"])
            print(f"  LIF overrides: {strat['lif_override']}")

        # Run on SpiNNaker
        print(f"  Running on SpiNNaker...")
        result = run_fc1_fc2_sample(
            input_spike_times,
            fc1_exc, fc1_inh,
            fc2_exc, fc2_inh,
            lif_params=lif,
            label=f"strategy_{name}_sample0",
        )

        # Print summary
        if result.get("error"):
            print(f"  ERROR: {result['error']}")
            strategy_results[name] = {
                "strategy": name,
                "error": result["error"],
                "hidden_fired": 0,
                "output_fired": 0,
                "predicted": -1,
                "correct": False,
            }
        else:
            hid = result["hidden"]
            out = result["output"]
            predicted = result["predicted"]
            correct = (predicted == true_label)

            print(f"  Hidden: {hid['neurons_fired']}/{N_HIDDEN} fired, "
                  f"{hid['total_spikes']} spikes, v_max={hid['v_max']:.4f}")
            print(f"  Output: {out['neurons_fired']}/{N_OUTPUT} fired, "
                  f"{out['total_spikes']} spikes, v_max={out['v_max']:.4f}")
            print(f"  Prediction: {predicted} (true={true_label}) "
                  f"{'CORRECT' if correct else 'WRONG'} "
                  f"[{result['classification_basis']}]")
            print(f"  Wall clock: {result['wall_clock_s']:.1f}s")

            # Show top-5 output spike counts
            top5_out = sorted(enumerate(out["spike_counts"]),
                              key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top-5 output: {top5_out}")

            # Show top-5 hidden spike counts
            top5_hid = sorted(enumerate(hid["spike_counts"]),
                              key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top-5 hidden: {top5_hid}")

            strategy_results[name] = {
                "strategy": name,
                "description": strat["description"],
                "fc1_connections": fc1_total,
                "fc2_connections": fc2_total,
                "fc1_scale": strat["fc1_scale"],
                "fc2_scale": strat["fc2_scale"],
                "lif_override": strat["lif_override"],
                "hidden_fired": hid["neurons_fired"],
                "hidden_total_spikes": hid["total_spikes"],
                "output_fired": out["neurons_fired"],
                "output_total_spikes": out["total_spikes"],
                "predicted": predicted,
                "true_label": true_label,
                "correct": correct,
                "classification_basis": result["classification_basis"],
                "wall_clock_s": result["wall_clock_s"],
                "hidden_spike_counts": hid["spike_counts"],
                "output_spike_counts": out["spike_counts"],
                "output_v_sum": out["v_sum"],
            }

    # Summary table
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)
    print(f"{'Strat':>5} {'HidFired':>9} {'HidSpks':>8} {'OutFired':>9} "
          f"{'OutSpks':>8} {'Pred':>5} {'True':>5} {'Correct':>8} {'Basis':>10}")
    print("-" * 75)

    for name in ["A", "B", "C", "D"]:
        r = strategy_results[name]
        if r.get("error"):
            print(f"  {name:>3}  ERROR: {r['error'][:50]}")
        else:
            print(f"  {name:>3} {r['hidden_fired']:>9} {r.get('hidden_total_spikes',0):>8} "
                  f"{r['output_fired']:>9} {r.get('output_total_spikes',0):>8} "
                  f"{r['predicted']:>5} {r['true_label']:>5} "
                  f"{'YES' if r['correct'] else 'NO':>8} "
                  f"{r.get('classification_basis','?'):>10}")

    # Rank strategies: prefer hidden neurons fired > output discrimination
    ranking = []
    for name in ["A", "B", "C", "D"]:
        r = strategy_results[name]
        if r.get("error"):
            score = -1
        else:
            # Score: hidden fired + 10*correct + output_spikes diversity
            out_diversity = r["output_fired"]
            score = r["hidden_fired"] + (10 if r["correct"] else 0) + out_diversity
        ranking.append((name, score))
    ranking.sort(key=lambda x: x[1], reverse=True)

    best_name = ranking[0][0]
    print(f"\n  Best strategy: {best_name} (score={ranking[0][1]})")
    print(f"  Ranking: {ranking}")

    # Save phase 1 results
    phase1_result = {
        "phase": 1,
        "sample_idx": sample_idx,
        "true_label": true_label,
        "snntorch_pred": snn_pred,
        "strategies": strategy_results,
        "ranking": ranking,
        "best_strategy": best_name,
        "timestamp": ts(),
    }
    p1_path = RESULTS_DIR / "phase1_strategy_comparison.json"
    with open(p1_path, "w") as f:
        json.dump(phase1_result, f, indent=2, default=str)
    print(f"\n  Saved: {p1_path}")

    return best_name, strategy_results


# ============================================================
# Phase 2: Run best strategy on 20 samples
# ============================================================
def phase2_multi_sample(fc1_weight, fc2_weight, fc1_inputs, labels, preds,
                         best_strategy, num_samples=20):
    """Run the best strategy on multiple samples and compute accuracy."""
    strat = STRATEGIES[best_strategy]

    print("\n" + "=" * 70)
    print(f"PHASE 2: Run Strategy {best_strategy} on {num_samples} Samples")
    print(f"  {strat['description']}")
    print("=" * 70)

    # Build connections once (shared across all samples)
    print("\n  Building connections...")
    t_build = time.time()
    fc1_exc, fc1_inh, fc2_exc, fc2_inh = strat["builder_fn"](fc1_weight, fc2_weight)
    print(f"  FC1: {len(fc1_exc)} exc + {len(fc1_inh)} inh = "
          f"{len(fc1_exc)+len(fc1_inh)} total")
    print(f"  FC2: {len(fc2_exc)} exc + {len(fc2_inh)} inh = "
          f"{len(fc2_exc)+len(fc2_inh)} total")
    print(f"  Build time: {time.time()-t_build:.1f}s")

    lif = LIF_PARAMS.copy()
    if strat["lif_override"]:
        lif.update(strat["lif_override"])
        print(f"  LIF overrides: {strat['lif_override']}")

    n = min(num_samples, len(labels))
    per_sample = []
    correct_count = 0
    snn_correct_count = 0
    total_wall = 0.0

    for idx in range(n):
        print(f"\n  --- Sample {idx}/{n} ---")
        sample = fc1_inputs[idx]
        input_spike_times = binary_to_spike_times(sample)

        true_label = int(labels[idx])
        snn_pred = int(preds[idx]) if preds is not None else -1

        result = run_fc1_fc2_sample(
            input_spike_times,
            fc1_exc, fc1_inh,
            fc2_exc, fc2_inh,
            lif_params=lif,
            label=f"strategy_{best_strategy}_sample{idx}",
        )

        if result.get("error"):
            print(f"    ERROR: {result['error'][:80]}")
            sample_result = {
                "sample": idx,
                "true_label": true_label,
                "snntorch_pred": snn_pred,
                "error": result["error"],
            }
        else:
            hid = result["hidden"]
            out = result["output"]
            predicted = result["predicted"]
            is_correct = (predicted == true_label)
            snn_is_correct = (snn_pred == true_label)

            if is_correct:
                correct_count += 1
            if snn_is_correct:
                snn_correct_count += 1

            total_wall += result["wall_clock_s"]

            print(f"    Hid: {hid['neurons_fired']} fired, {hid['total_spikes']} spks | "
                  f"Out: {out['neurons_fired']} fired, {out['total_spikes']} spks | "
                  f"Pred={predicted} True={true_label} "
                  f"{'OK' if is_correct else 'WRONG'} "
                  f"[{result['classification_basis']}]")

            sample_result = {
                "sample": idx,
                "true_label": true_label,
                "snntorch_pred": snn_pred,
                "snntorch_correct": snn_is_correct,
                "spinnaker_pred": predicted,
                "spinnaker_correct": is_correct,
                "classification_basis": result["classification_basis"],
                "hidden_fired": hid["neurons_fired"],
                "hidden_total_spikes": hid["total_spikes"],
                "output_fired": out["neurons_fired"],
                "output_total_spikes": out["total_spikes"],
                "output_spike_counts": out["spike_counts"],
                "output_v_sum": out["v_sum"],
                "wall_clock_s": result["wall_clock_s"],
            }

        per_sample.append(sample_result)

    # ---- Phase 3: Compute accuracy and compare ----
    spk_accuracy = correct_count / n if n > 0 else 0
    snn_accuracy = snn_correct_count / n if n > 0 else 0
    fc2_only_baseline = 0.43  # 43% from run6, fold 4

    print("\n" + "=" * 70)
    print("PHASE 3: RESULTS")
    print("=" * 70)
    print(f"\n  Strategy:              {best_strategy} - {strat['description']}")
    print(f"  Samples:               {n}")
    print(f"  SpiNNaker (full):      {correct_count}/{n} = {spk_accuracy:.1%}")
    print(f"  snnTorch reference:    {snn_correct_count}/{n} = {snn_accuracy:.1%}")
    print(f"  FC2-only baseline:     43.0% (fold 4, 400 samples)")
    print(f"  Gap vs snnTorch:       {(snn_accuracy - spk_accuracy)*100:.1f} pp")
    print(f"  Gap vs FC2-only:       {(spk_accuracy - fc2_only_baseline)*100:.1f} pp")
    print(f"  Avg wall clock:        {total_wall/n:.1f}s per sample")

    # Per-sample table
    print(f"\n  {'Idx':>4} {'True':>5} {'SNN':>5} {'Spk':>5} {'Correct':>8} "
          f"{'HidF':>5} {'OutF':>5} {'Basis':>10}")
    print("  " + "-" * 58)
    for r in per_sample:
        if r.get("error"):
            print(f"  {r['sample']:>4}  ERROR: {r['error'][:40]}")
        else:
            print(f"  {r['sample']:>4} {r['true_label']:>5} "
                  f"{r['snntorch_pred']:>5} {r['spinnaker_pred']:>5} "
                  f"{'YES' if r['spinnaker_correct'] else 'NO':>8} "
                  f"{r['hidden_fired']:>5} {r['output_fired']:>5} "
                  f"{r['classification_basis']:>10}")

    # Agreement analysis
    both_correct = sum(1 for r in per_sample
                       if not r.get("error")
                       and r.get("spinnaker_correct") and r.get("snntorch_correct"))
    spk_only = sum(1 for r in per_sample
                   if not r.get("error")
                   and r.get("spinnaker_correct") and not r.get("snntorch_correct"))
    snn_only = sum(1 for r in per_sample
                   if not r.get("error")
                   and not r.get("spinnaker_correct") and r.get("snntorch_correct"))
    both_wrong = sum(1 for r in per_sample
                     if not r.get("error")
                     and not r.get("spinnaker_correct") and not r.get("snntorch_correct"))
    n_valid = sum(1 for r in per_sample if not r.get("error"))
    agreement = (both_correct + both_wrong) / n_valid if n_valid > 0 else 0

    print(f"\n  Agreement analysis ({n_valid} valid samples):")
    print(f"    Both correct:    {both_correct}")
    print(f"    SpiNNaker only:  {spk_only}")
    print(f"    snnTorch only:   {snn_only}")
    print(f"    Both wrong:      {both_wrong}")
    print(f"    Agreement rate:  {agreement:.1%}")

    # Save full results
    full_result = {
        "phase": "2+3",
        "strategy": best_strategy,
        "strategy_description": strat["description"],
        "fc1_scale": strat["fc1_scale"],
        "fc2_scale": strat["fc2_scale"],
        "lif_params": lif,
        "lif_override": strat["lif_override"],
        "fc1_connections": len(fc1_exc) + len(fc1_inh),
        "fc2_connections": len(fc2_exc) + len(fc2_inh),
        "num_samples": n,
        "spinnaker_correct": correct_count,
        "spinnaker_accuracy": spk_accuracy,
        "snntorch_correct": snn_correct_count,
        "snntorch_accuracy": snn_accuracy,
        "fc2_only_baseline_accuracy": fc2_only_baseline,
        "gap_vs_snntorch_pp": round((snn_accuracy - spk_accuracy) * 100, 2),
        "gap_vs_fc2only_pp": round((spk_accuracy - fc2_only_baseline) * 100, 2),
        "agreement_rate": agreement,
        "error_analysis": {
            "both_correct": both_correct,
            "spinnaker_only": spk_only,
            "snntorch_only": snn_only,
            "both_wrong": both_wrong,
        },
        "per_sample": per_sample,
        "avg_wall_clock_s": total_wall / n if n > 0 else 0,
        "timestamp": ts(),
    }

    p2_path = RESULTS_DIR / f"phase2_strategy{best_strategy}_{n}samples.json"
    with open(p2_path, "w") as f:
        json.dump(full_result, f, indent=2, default=str)
    print(f"\n  Saved: {p2_path}")

    return full_result


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Optimized full FC1+FC2 SpiNNaker deployment"
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["1", "2", "all"],
        help="Phase to run: 1=compare strategies, 2=multi-sample, all=both"
    )
    parser.add_argument(
        "--num-samples", type=int, default=20,
        help="Number of samples for phase 2 (default: 20)"
    )
    parser.add_argument(
        "--force-strategy", type=str, default=None,
        choices=["A", "B", "C", "D"],
        help="Force a specific strategy for phase 2 (skips phase 1 ranking)"
    )
    args = parser.parse_args()

    print(f"[{ts()}] spinnaker_full_optimized.py")
    print(f"  Phase: {args.phase}")
    print(f"  Results dir: {RESULTS_DIR}")
    print()

    # Load data
    print("Loading data...")
    fc1_weight, fc2_weight, fc1_inputs, labels, preds = load_data()

    # Weight distribution analysis
    print(f"\n  FC1 weight stats: mean={fc1_weight.mean():.5f}, "
          f"std={fc1_weight.std():.5f}, "
          f"min={fc1_weight.min():.5f}, max={fc1_weight.max():.5f}")
    print(f"  FC2 weight stats: mean={fc2_weight.mean():.5f}, "
          f"std={fc2_weight.std():.5f}, "
          f"min={fc2_weight.min():.5f}, max={fc2_weight.max():.5f}")

    # Connection budget analysis
    print(f"\n  Connection budget analysis:")
    for name in ["A", "B", "C", "D"]:
        strat = STRATEGIES[name]
        fc1_exc, fc1_inh, fc2_exc, fc2_inh = strat["builder_fn"](fc1_weight, fc2_weight)
        fc1_t = len(fc1_exc) + len(fc1_inh)
        fc2_t = len(fc2_exc) + len(fc2_inh)
        exc_ratio = len(fc1_exc) / fc1_t * 100 if fc1_t > 0 else 0
        print(f"    {name}: FC1={fc1_t:>6} ({exc_ratio:.0f}% exc), "
              f"FC2={fc2_t:>6}, "
              f"Total={fc1_t+fc2_t:>6} "
              f"{'OK' if fc1_t <= MAX_CONNS_PER_LAYER else 'OVER LIMIT'}")
    print()

    t_total = time.time()

    if args.phase in ("1", "all"):
        best_name, strat_results = phase1_compare_strategies(
            fc1_weight, fc2_weight, fc1_inputs, labels, preds
        )
    else:
        best_name = None

    if args.phase in ("2", "all"):
        if args.force_strategy:
            best_name = args.force_strategy
            print(f"\n  Forcing strategy: {best_name}")
        elif best_name is None:
            # Try to load from phase 1 results
            p1_path = RESULTS_DIR / "phase1_strategy_comparison.json"
            if p1_path.exists():
                with open(p1_path) as f:
                    p1 = json.load(f)
                best_name = p1["best_strategy"]
                print(f"\n  Loaded best strategy from phase 1: {best_name}")
            else:
                print("  ERROR: No phase 1 results found and no --force-strategy set.")
                print("  Run --phase 1 first or use --force-strategy.")
                sys.exit(1)

        phase2_multi_sample(
            fc1_weight, fc2_weight, fc1_inputs, labels, preds,
            best_name, num_samples=args.num_samples
        )

    elapsed = time.time() - t_total
    print(f"\n[{ts()}] Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
