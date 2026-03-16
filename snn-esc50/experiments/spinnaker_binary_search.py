"""
spinnaker_binary_search.py -- Find the EXACT breaking point for SpiNNaker
hidden population size, and test strategies to fix it.

PROBLEM STATEMENT:
  2304 inputs -> 50 hidden (exc-only, initialize(v=0.0)):  32/50 fire  (WORKS)
  2304 inputs -> 256 hidden (exc-only, initialize(v=0.0)): 0/256 fire  (BROKEN)

Something breaks between 50 and 256 hidden neurons. This script does:

  PART 1: Binary search over hidden neuron count
          Tests: 50, 60, 70, 80, 100, 120, 150, 200, 256
          All with initialize(v=0.0), exc-only connections, scale=1.0
          This finds the exact breakpoint.

  PART 2: Test remediation strategies at 256 hidden neurons
          A. set_number_of_neurons_per_core(sim.IF_curr_exp, N) for N in [32, 64, 128]
             -> Forces the 256-neuron pop to be split across multiple cores
             -> Reduces per-core synaptic load
          B. set_number_of_neurons_per_core(sim.SpikeSourceArray, N) for N in [128, 256, 512]
             -> Splits input population across more cores
             -> Reduces per-core spike output, easing routing
          C. Both A + B together
          D. Spread spikes across sub-timesteps using jittered spike times
             -> Instead of time 5.0, use 5.0 + random(0, 0.5)
             -> Reduces peak simultaneous spikes
          E. Use smaller weight scale to reduce SDRAM pressure
          F. Reduce input population by grouping (use only top-K active inputs)

  PART 3: Full 256-hidden with the best-working strategy from Part 2

CRITICAL LESSONS FROM RESEARCH:
  1. POP_TABLE_MAX_ROW_LENGTH = 256: This is the max synaptic row length.
     When a pre-neuron connects to N post-neurons on one core, N <= 256.
     With 256 hidden neurons on 1 core, this is EXACTLY at the limit.
  2. DEFAULT_MAX_ATOMS_PER_CORE = 256 for IF_curr_exp: The 256 hidden
     neurons all go on ONE core. All synapses from 2304 inputs must be
     stored in that core's SDRAM. That's up to 590K synapses.
  3. The SpikeSourceArray warning "too many spikes" is a ROUTING issue.
     ~1800 simultaneous spikes overwhelm the router.
  4. DMA transfer bottleneck: when a spike arrives at a core, the entire
     synaptic row for that pre-neuron must be fetched from SDRAM via DMA.
     With 256 post-neurons, each row is 256 entries = 1024 bytes.
     1800 spikes/timestep * 1024 bytes = 1.8 MB DMA per timestep.

HYPOTHESIS: The breakpoint is at the SDRAM or DMA limit for the hidden
core. By splitting the hidden population across multiple cores
(set_number_of_neurons_per_core < 256), each core handles fewer synapses
and shorter DMA transfers.

Usage (from .venv-spinnaker):
    cd snn-esc50
    source .venv-spinnaker/bin/activate

    # Part 1: Binary search (find breakpoint)
    python experiments/spinnaker_binary_search.py --part 1

    # Part 2: Test strategies at 256 hidden
    python experiments/spinnaker_binary_search.py --part 2

    # Part 3: Full test with best strategy
    python experiments/spinnaker_binary_search.py --part 3

    # All parts
    python experiments/spinnaker_binary_search.py --part all

    # Quick test: just one hidden size
    python experiments/spinnaker_binary_search.py --part 1 --sizes 50,256
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
DATA_DIR = REPO_ROOT / "results" / "spinnaker_results" / "full_deploy_cond"
RESULTS_DIR = REPO_ROOT / "results" / "spinnaker_results" / "binary_search"
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

# Calibrated neuron params (known-good from FC2-only runs)
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
def load_sample_data(sample_idx=0):
    """Load FC1 weights and one sample's input features."""
    fc1_weight = np.load(DATA_DIR / "fc1_weight_fold4.npy")   # (256, 2304)
    fc1_inputs = np.load(DATA_DIR / "fc1_input_features_fold4.npy")  # (N, 25, 2304)
    labels = np.load(DATA_DIR / "labels_fold4.npy")

    sample = fc1_inputs[sample_idx]  # (25, 2304)
    label = int(labels[sample_idx])

    # Input stats
    active_per_step = (sample > 0.5).sum(axis=1)
    total_spikes = int((sample > 0.5).sum())

    print(f"  FC1 weights: {fc1_weight.shape}")
    print(f"  Sample {sample_idx}: {total_spikes} input spikes, "
          f"mean {active_per_step.mean():.0f}/step, "
          f"max {active_per_step.max()}/step, "
          f"label={label}")

    return fc1_weight, sample, label


def binary_to_spike_times(binary_matrix):
    """Convert (T, N) binary matrix to list of spike time lists."""
    T, N = binary_matrix.shape
    spike_times_list = []
    for n in range(N):
        times = np.where(binary_matrix[:, n] > 0.5)[0].astype(float).tolist()
        spike_times_list.append(times)
    return spike_times_list


def build_exc_only_connections(fc1_weight, n_hidden, scale=1.0,
                                prune_threshold=0.001):
    """Build excitatory-only connection list for a given hidden size.

    Uses the first n_hidden rows of fc1_weight (256, 2304).
    Zeroes out negative weights. Returns list of [pre, post, weight, delay].
    """
    # Use first n_hidden rows of the weight matrix
    w = fc1_weight[:n_hidden, :].copy()
    w[w < 0] = 0.0

    # Vectorized extraction
    post_idx, pre_idx = np.where(w > prune_threshold)
    weights = w[post_idx, pre_idx] * scale

    conn_list = []
    for i in range(len(weights)):
        conn_list.append([int(pre_idx[i]), int(post_idx[i]),
                          float(weights[i]), 1.0])

    return conn_list


def build_exc_only_topk(fc1_weight, n_hidden, k_per_neuron=20, scale=1.0):
    """Build top-K excitatory connections per hidden neuron.

    Only keeps the K strongest positive connections per hidden neuron.
    Total connections = n_hidden * K (much less than full connectivity).
    """
    w = fc1_weight[:n_hidden, :].copy()
    conn_list = []

    for post in range(n_hidden):
        row = w[post, :]
        # Only positive weights
        pos_mask = row > 0
        pos_indices = np.where(pos_mask)[0]
        pos_weights = row[pos_indices]

        if len(pos_indices) > k_per_neuron:
            top_k_idx = np.argsort(pos_weights)[-k_per_neuron:]
            selected_pre = pos_indices[top_k_idx]
            selected_w = pos_weights[top_k_idx]
        else:
            selected_pre = pos_indices
            selected_w = pos_weights

        for pre, wt in zip(selected_pre, selected_w):
            conn_list.append([int(pre), int(post), float(wt * scale), 1.0])

    return conn_list


# ============================================================
# Core SpiNNaker test function
# ============================================================
def run_hidden_test(
    input_spike_times,
    n_hidden,
    exc_conns,
    lif_params=None,
    neurons_per_core_hidden=None,
    neurons_per_core_input=None,
    label="test",
):
    """Run a single test: 2304 inputs -> n_hidden hidden neurons on SpiNNaker.

    Args:
        input_spike_times: list of 2304 spike time lists
        n_hidden: number of hidden neurons
        exc_conns: list of [pre, post, weight, delay]
        lif_params: neuron parameters (default: LIF_PARAMS)
        neurons_per_core_hidden: if set, call set_number_of_neurons_per_core
                                 for IF_curr_exp BEFORE creating populations
        neurons_per_core_input: if set, call set_number_of_neurons_per_core
                                for SpikeSourceArray
        label: descriptive label

    Returns:
        dict with results
    """
    if lif_params is None:
        lif_params = LIF_PARAMS

    t0 = time.time()
    result = {
        "label": label,
        "n_input": N_INPUT,
        "n_hidden": n_hidden,
        "n_connections": len(exc_conns),
        "neurons_per_core_hidden": neurons_per_core_hidden,
        "neurons_per_core_input": neurons_per_core_input,
        "error": None,
    }

    sim_started = False
    try:
        sim.setup(timestep=DT)
        sim_started = True

        # STRATEGY: control core splitting
        if neurons_per_core_hidden is not None:
            sim.set_number_of_neurons_per_core(
                sim.IF_curr_exp, neurons_per_core_hidden)
            n_hidden_cores = (n_hidden + neurons_per_core_hidden - 1) // neurons_per_core_hidden
            print(f"    Set IF_curr_exp neurons_per_core={neurons_per_core_hidden} "
                  f"-> {n_hidden_cores} cores for {n_hidden} neurons")

        if neurons_per_core_input is not None:
            sim.set_number_of_neurons_per_core(
                sim.SpikeSourceArray, neurons_per_core_input)
            n_input_cores = (N_INPUT + neurons_per_core_input - 1) // neurons_per_core_input
            print(f"    Set SpikeSourceArray neurons_per_core={neurons_per_core_input} "
                  f"-> {n_input_cores} cores for {N_INPUT} neurons")

        try:
            # Input population
            input_pop = sim.Population(
                N_INPUT,
                sim.SpikeSourceArray,
                {"spike_times": input_spike_times},
                label="input"
            )
            input_pop.record("spikes")

            # Hidden population
            hidden_pop = sim.Population(
                n_hidden,
                sim.IF_curr_exp(**lif_params),
                label="hidden"
            )
            hidden_pop.record(["spikes", "v"])

            # CRITICAL: Always initialize v=0.0 explicitly
            # sPyNNaker may default to -65.0 even when v_rest=0.0
            hidden_pop.initialize(v=0.0)

            # Excitatory projection
            if exc_conns:
                sim.Projection(
                    input_pop, hidden_pop,
                    sim.FromListConnector(exc_conns),
                    receptor_type="excitatory"
                )

            # Run
            sim.run(NUM_STEPS)

            # Extract input spikes
            in_data = input_pop.get_data("spikes")
            in_trains = in_data.segments[0].spiketrains
            result["input_spikes_delivered"] = sum(len(st) for st in in_trains)
            result["input_spikes_expected"] = sum(
                len(t) for t in input_spike_times)

            # Extract hidden layer results
            hid_data = hidden_pop.get_data(["spikes", "v"])
            hid_trains = hid_data.segments[0].spiketrains
            hid_v = hid_data.segments[0].filter(name="v")[0]

            spike_counts = [len(st) for st in hid_trains]
            fired_ids = [j for j, c in enumerate(spike_counts) if c > 0]

            try:
                v_arr = hid_v.magnitude  # (T, n_hidden)
                v_max = float(v_arr.max())
                v_mean_final = float(v_arr[-1, :].mean())
                # Sample some initial voltages to verify initialize worked
                v_t0 = v_arr[0, :5].tolist() if v_arr.shape[0] > 0 else []
            except Exception:
                v_max = 0.0
                v_mean_final = 0.0
                v_t0 = []

            result["neurons_fired"] = len(fired_ids)
            result["total_spikes"] = sum(spike_counts)
            result["fired_ids"] = fired_ids[:30]
            result["v_max"] = v_max
            result["v_mean_final"] = v_mean_final
            result["v_t0_sample"] = v_t0
            result["spike_counts_top10"] = sorted(
                enumerate(spike_counts), key=lambda x: x[1], reverse=True
            )[:10]

        finally:
            sim.end()

    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        traceback.print_exc()
        if sim_started:
            try:
                sim.end()
            except Exception:
                pass

    result["wall_clock_s"] = time.time() - t0
    return result


# ============================================================
# PART 1: Binary search over hidden neuron count
# ============================================================
def part1_binary_search(fc1_weight, sample, label, sizes=None):
    """Find the exact hidden neuron count where SpiNNaker breaks."""
    print("\n" + "=" * 70)
    print("PART 1: Binary Search Over Hidden Neuron Count")
    print("=" * 70)
    print("Testing: 2304 inputs -> N hidden, exc-only, initialize(v=0.0)")
    print("Goal: Find the exact N where neurons stop firing.")
    print()

    if sizes is None:
        sizes = [50, 60, 70, 80, 100, 120, 150, 200, 256]

    input_spike_times = binary_to_spike_times(sample)

    results = {}
    for n_hidden in sizes:
        print(f"\n--- Testing n_hidden={n_hidden} ---")

        # Build exc-only connections
        exc_conns = build_exc_only_connections(
            fc1_weight, n_hidden, scale=1.0)
        print(f"  Connections: {len(exc_conns):,} exc-only")

        # Also compute expected current for diagnostic
        w_sub = fc1_weight[:n_hidden, :].copy()
        w_sub[w_sub < 0] = 0.0
        active_per_step = (sample > 0.5).sum(axis=1).mean()
        expected_current = w_sub.sum(axis=1).mean() * (active_per_step / N_INPUT)
        print(f"  Expected mean exc current per neuron: {expected_current:.3f}")

        result = run_hidden_test(
            input_spike_times, n_hidden, exc_conns,
            label=f"binary_search_n{n_hidden}"
        )

        if result.get("error"):
            print(f"  ERROR: {result['error'][:100]}")
        else:
            print(f"  Neurons fired: {result['neurons_fired']}/{n_hidden}")
            print(f"  Total spikes: {result['total_spikes']}")
            print(f"  v_max: {result['v_max']:.4f}")
            print(f"  v_t0_sample: {result['v_t0_sample']}")
            print(f"  Input delivered: {result['input_spikes_delivered']}"
                  f"/{result['input_spikes_expected']}")
            print(f"  Wall clock: {result['wall_clock_s']:.1f}s")

        results[n_hidden] = result

    # Summary table
    print("\n" + "=" * 70)
    print("PART 1 SUMMARY: Binary Search Results")
    print("=" * 70)
    print(f"{'N_hidden':>10} {'Fired':>8} {'Total':>8} {'v_max':>10} "
          f"{'v_t0[0]':>10} {'Conns':>10} {'Error':>8}")
    print("-" * 68)

    breakpoint_found = None
    last_working = None
    for n_hidden in sizes:
        r = results[n_hidden]
        if r.get("error"):
            print(f"  {n_hidden:>8}    ERROR: {r['error'][:40]}")
        else:
            fired = r["neurons_fired"]
            v_t0 = r["v_t0_sample"][0] if r["v_t0_sample"] else "?"
            status = "OK" if fired > 0 else "BROKEN"
            print(f"  {n_hidden:>8} {fired:>8} {r['total_spikes']:>8} "
                  f"{r['v_max']:>10.4f} {str(v_t0):>10} "
                  f"{r['n_connections']:>10,} {status:>8}")

            if fired > 0:
                last_working = n_hidden
            elif breakpoint_found is None:
                breakpoint_found = n_hidden

    if breakpoint_found:
        print(f"\n  BREAKPOINT: Between {last_working} and {breakpoint_found} "
              f"hidden neurons")
        print(f"  Last working: {last_working}")
        print(f"  First broken: {breakpoint_found}")
    elif last_working:
        print(f"\n  All sizes worked! Last tested: {sizes[-1]}")
    else:
        print(f"\n  ALL sizes broken! Even {sizes[0]} failed.")

    # Save results
    save_path = RESULTS_DIR / "part1_binary_search.json"
    save_data = {
        "part": 1,
        "sizes_tested": sizes,
        "breakpoint": breakpoint_found,
        "last_working": last_working,
        "results": {str(k): v for k, v in results.items()},
        "timestamp": ts(),
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Saved: {save_path}")

    return results, breakpoint_found, last_working


# ============================================================
# PART 2: Test remediation strategies at 256 hidden
# ============================================================
def part2_test_strategies(fc1_weight, sample, label):
    """Test strategies to make 2304->256 work on SpiNNaker."""
    print("\n" + "=" * 70)
    print("PART 2: Test Remediation Strategies (2304 -> 256 hidden)")
    print("=" * 70)

    input_spike_times = binary_to_spike_times(sample)
    n_hidden = 256
    strategies = {}

    # ---- Strategy A: Split hidden population across cores ----
    print("\n--- Strategy A: set_number_of_neurons_per_core(IF_curr_exp, N) ---")
    print("Splits 256 hidden neurons across multiple cores.")
    print("Each core handles fewer synapses and shorter DMA transfers.\n")

    for npc in [32, 64, 128]:
        name = f"A_npc{npc}"
        print(f"  A: neurons_per_core_hidden={npc} "
              f"({256//npc} cores for hidden)")

        # Use top-K connections to keep total manageable
        exc_conns = build_exc_only_connections(
            fc1_weight, n_hidden, scale=1.0)
        print(f"    Connections: {len(exc_conns):,}")

        result = run_hidden_test(
            input_spike_times, n_hidden, exc_conns,
            neurons_per_core_hidden=npc,
            label=name,
        )
        strategies[name] = result

        if result.get("error"):
            print(f"    ERROR: {result['error'][:80]}")
        else:
            print(f"    Fired: {result['neurons_fired']}/{n_hidden}, "
                  f"spikes={result['total_spikes']}, "
                  f"v_max={result['v_max']:.4f}, "
                  f"v_t0={result['v_t0_sample']}")

    # ---- Strategy B: Split input population across more cores ----
    print("\n--- Strategy B: set_number_of_neurons_per_core(SpikeSourceArray, N) ---")
    print("Splits 2304 input neurons across more cores.")
    print("Reduces per-core spike output, easing routing pressure.\n")

    for npc_in in [128, 256, 512]:
        name = f"B_npc_in{npc_in}"
        n_cores = (N_INPUT + npc_in - 1) // npc_in
        print(f"  B: neurons_per_core_input={npc_in} "
              f"({n_cores} cores for input)")

        exc_conns = build_exc_only_connections(
            fc1_weight, n_hidden, scale=1.0)

        result = run_hidden_test(
            input_spike_times, n_hidden, exc_conns,
            neurons_per_core_input=npc_in,
            label=name,
        )
        strategies[name] = result

        if result.get("error"):
            print(f"    ERROR: {result['error'][:80]}")
        else:
            print(f"    Fired: {result['neurons_fired']}/{n_hidden}, "
                  f"spikes={result['total_spikes']}, "
                  f"v_max={result['v_max']:.4f}")

    # ---- Strategy C: Both A + B together ----
    print("\n--- Strategy C: Split BOTH hidden AND input ---")
    print("Combined: hidden split + input split.\n")

    for npc_h, npc_i in [(64, 256), (32, 128), (128, 512)]:
        name = f"C_h{npc_h}_i{npc_i}"
        print(f"  C: hidden npc={npc_h}, input npc={npc_i}")

        exc_conns = build_exc_only_connections(
            fc1_weight, n_hidden, scale=1.0)

        result = run_hidden_test(
            input_spike_times, n_hidden, exc_conns,
            neurons_per_core_hidden=npc_h,
            neurons_per_core_input=npc_i,
            label=name,
        )
        strategies[name] = result

        if result.get("error"):
            print(f"    ERROR: {result['error'][:80]}")
        else:
            print(f"    Fired: {result['neurons_fired']}/{n_hidden}, "
                  f"spikes={result['total_spikes']}, "
                  f"v_max={result['v_max']:.4f}")

    # ---- Strategy D: Jittered spike times ----
    print("\n--- Strategy D: Jittered spike times (reduce simultaneous spikes) ---")
    print("Add small random offset to spike times to spread them out.\n")

    np.random.seed(42)
    jittered_spike_times = []
    for n in range(N_INPUT):
        orig = np.where(sample[:, n] > 0.5)[0].astype(float)
        if len(orig) > 0:
            # Add jitter: uniform [0, 0.5) ms
            jittered = orig + np.random.uniform(0, 0.5, size=len(orig))
            jittered_spike_times.append(jittered.tolist())
        else:
            jittered_spike_times.append([])

    # Check peak spikes after jitter
    from collections import Counter
    all_times = []
    for st in jittered_spike_times:
        all_times.extend([round(t, 1) for t in st])
    counter = Counter(all_times)
    peak = counter.most_common(1)[0] if counter else (0, 0)
    print(f"  After jitter: peak = {peak[1]} spikes at t={peak[0]}")

    exc_conns = build_exc_only_connections(fc1_weight, n_hidden, scale=1.0)

    result = run_hidden_test(
        jittered_spike_times, n_hidden, exc_conns,
        label="D_jittered",
    )
    strategies["D_jittered"] = result

    if result.get("error"):
        print(f"    ERROR: {result['error'][:80]}")
    else:
        print(f"    Fired: {result['neurons_fired']}/{n_hidden}, "
              f"spikes={result['total_spikes']}, "
              f"v_max={result['v_max']:.4f}")

    # ---- Strategy E: Top-K connections (reduce total synapses) ----
    print("\n--- Strategy E: Top-K connections per neuron (reduce synapse count) ---")
    print("Keeps only K strongest positive weights per hidden neuron.\n")

    for k in [10, 20, 50]:
        name = f"E_topk{k}"
        exc_conns = build_exc_only_topk(
            fc1_weight, n_hidden, k_per_neuron=k, scale=1.0)
        print(f"  E: top-{k} per neuron -> {len(exc_conns):,} total connections")

        result = run_hidden_test(
            input_spike_times, n_hidden, exc_conns,
            label=name,
        )
        strategies[name] = result

        if result.get("error"):
            print(f"    ERROR: {result['error'][:80]}")
        else:
            print(f"    Fired: {result['neurons_fired']}/{n_hidden}, "
                  f"spikes={result['total_spikes']}, "
                  f"v_max={result['v_max']:.4f}")

    # ---- Strategy F: Top-K + split hidden cores ----
    print("\n--- Strategy F: Top-K connections + split hidden cores ---")
    print("Combines reduced connectivity with core splitting.\n")

    for k, npc_h in [(20, 64), (20, 32), (50, 64), (50, 128)]:
        name = f"F_topk{k}_npc{npc_h}"
        exc_conns = build_exc_only_topk(
            fc1_weight, n_hidden, k_per_neuron=k, scale=1.0)
        print(f"  F: top-{k}, hidden npc={npc_h} -> "
              f"{len(exc_conns):,} conns, "
              f"{256//npc_h} hidden cores")

        result = run_hidden_test(
            input_spike_times, n_hidden, exc_conns,
            neurons_per_core_hidden=npc_h,
            label=name,
        )
        strategies[name] = result

        if result.get("error"):
            print(f"    ERROR: {result['error'][:80]}")
        else:
            print(f"    Fired: {result['neurons_fired']}/{n_hidden}, "
                  f"spikes={result['total_spikes']}, "
                  f"v_max={result['v_max']:.4f}")

    # ---- Strategy G: Higher weight scale ----
    print("\n--- Strategy G: Higher weight scale (compensate for losses) ---")

    for scale in [2.0, 5.0, 10.0]:
        name = f"G_scale{scale}"
        exc_conns = build_exc_only_connections(
            fc1_weight, n_hidden, scale=scale)
        print(f"  G: scale={scale} -> {len(exc_conns):,} connections")

        result = run_hidden_test(
            input_spike_times, n_hidden, exc_conns,
            label=name,
        )
        strategies[name] = result

        if result.get("error"):
            print(f"    ERROR: {result['error'][:80]}")
        else:
            print(f"    Fired: {result['neurons_fired']}/{n_hidden}, "
                  f"spikes={result['total_spikes']}, "
                  f"v_max={result['v_max']:.4f}")

    # ---- Strategy H: Split hidden + higher scale ----
    print("\n--- Strategy H: Split hidden cores + higher scale ---")

    for npc_h, scale in [(64, 5.0), (32, 5.0), (64, 10.0), (32, 10.0)]:
        name = f"H_npc{npc_h}_scale{scale}"
        exc_conns = build_exc_only_connections(
            fc1_weight, n_hidden, scale=scale)
        print(f"  H: npc={npc_h}, scale={scale} -> {len(exc_conns):,} conns")

        result = run_hidden_test(
            input_spike_times, n_hidden, exc_conns,
            neurons_per_core_hidden=npc_h,
            label=name,
        )
        strategies[name] = result

        if result.get("error"):
            print(f"    ERROR: {result['error'][:80]}")
        else:
            print(f"    Fired: {result['neurons_fired']}/{n_hidden}, "
                  f"spikes={result['total_spikes']}, "
                  f"v_max={result['v_max']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("PART 2 SUMMARY: Strategy Comparison")
    print("=" * 70)
    print(f"{'Strategy':>25} {'Fired':>6} {'Spikes':>8} {'v_max':>8} "
          f"{'Conns':>8} {'Time':>6}")
    print("-" * 70)

    best_name = None
    best_fired = 0
    for name in sorted(strategies.keys()):
        r = strategies[name]
        if r.get("error"):
            print(f"  {name:>23}   ERROR")
        else:
            fired = r["neurons_fired"]
            print(f"  {name:>23} {fired:>6} {r['total_spikes']:>8} "
                  f"{r['v_max']:>8.4f} "
                  f"{r['n_connections']:>8,} "
                  f"{r['wall_clock_s']:>5.0f}s")
            if fired > best_fired:
                best_fired = fired
                best_name = name

    if best_name:
        print(f"\n  Best strategy: {best_name} ({best_fired} fired)")
    else:
        print(f"\n  NO strategy worked at 256 hidden neurons!")

    # Save
    save_path = RESULTS_DIR / "part2_strategies.json"
    save_data = {
        "part": 2,
        "n_hidden": n_hidden,
        "strategies": strategies,
        "best_strategy": best_name,
        "best_fired": best_fired,
        "timestamp": ts(),
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"  Saved: {save_path}")

    return strategies, best_name


# ============================================================
# PART 3: Full test with best strategy on multiple samples
# ============================================================
def part3_full_test(fc1_weight, best_strategy_name, num_samples=5):
    """Run the best strategy on multiple samples."""
    print("\n" + "=" * 70)
    print(f"PART 3: Full Test with Best Strategy: {best_strategy_name}")
    print("=" * 70)

    fc1_inputs = np.load(DATA_DIR / "fc1_input_features_fold4.npy")
    labels = np.load(DATA_DIR / "labels_fold4.npy")
    preds = np.load(DATA_DIR / "snntorch_preds_fold4.npy")

    n_hidden = 256
    n = min(num_samples, len(labels))

    # Parse strategy name to extract parameters
    npc_h = None
    npc_i = None
    scale = 1.0
    k_per_neuron = None

    if "npc" in best_strategy_name:
        # Extract neurons_per_core settings
        parts = best_strategy_name.split("_")
        for p in parts:
            if p.startswith("npc") and "in" not in best_strategy_name:
                npc_h = int(p[3:])
            elif p.startswith("h"):
                try:
                    npc_h = int(p[1:])
                except ValueError:
                    pass
            elif p.startswith("i"):
                try:
                    npc_i = int(p[1:])
                except ValueError:
                    pass

    if "scale" in best_strategy_name:
        parts = best_strategy_name.split("_")
        for p in parts:
            if p.startswith("scale"):
                scale = float(p[5:])

    if "topk" in best_strategy_name:
        parts = best_strategy_name.split("_")
        for p in parts:
            if p.startswith("topk"):
                k_per_neuron = int(p[4:])

    print(f"  Parsed: npc_h={npc_h}, npc_i={npc_i}, scale={scale}, "
          f"topk={k_per_neuron}")

    per_sample = []
    for idx in range(n):
        sample = fc1_inputs[idx]
        true_label = int(labels[idx])
        snn_pred = int(preds[idx])

        input_spike_times = binary_to_spike_times(sample)

        if k_per_neuron:
            exc_conns = build_exc_only_topk(
                fc1_weight, n_hidden, k_per_neuron=k_per_neuron, scale=scale)
        else:
            exc_conns = build_exc_only_connections(
                fc1_weight, n_hidden, scale=scale)

        print(f"\n  Sample {idx}: true={true_label}, snnTorch={snn_pred}, "
              f"conns={len(exc_conns):,}")

        result = run_hidden_test(
            input_spike_times, n_hidden, exc_conns,
            neurons_per_core_hidden=npc_h,
            neurons_per_core_input=npc_i,
            label=f"part3_sample{idx}",
        )

        result["true_label"] = true_label
        result["snntorch_pred"] = snn_pred

        if result.get("error"):
            print(f"    ERROR: {result['error'][:80]}")
        else:
            print(f"    Fired: {result['neurons_fired']}/{n_hidden}, "
                  f"spikes={result['total_spikes']}, "
                  f"v_max={result['v_max']:.4f}")

        per_sample.append(result)

    # Save
    save_path = RESULTS_DIR / f"part3_{best_strategy_name}_{n}samples.json"
    save_data = {
        "part": 3,
        "strategy": best_strategy_name,
        "npc_hidden": npc_h,
        "npc_input": npc_i,
        "scale": scale,
        "topk": k_per_neuron,
        "n_samples": n,
        "per_sample": per_sample,
        "timestamp": ts(),
    }
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n  Saved: {save_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SpiNNaker binary search: find the exact breakpoint "
                    "and test remediation strategies."
    )
    parser.add_argument(
        "--part", type=str, default="all",
        choices=["1", "2", "3", "all"],
        help="Which part to run: 1=binary search, 2=strategies, "
             "3=full test, all=all"
    )
    parser.add_argument(
        "--sizes", type=str, default=None,
        help="Comma-separated hidden sizes for part 1 "
             "(default: 50,60,70,80,100,120,150,200,256)"
    )
    parser.add_argument(
        "--sample-idx", type=int, default=0,
        help="Which sample to use (default: 0)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=5,
        help="Number of samples for part 3 (default: 5)"
    )
    parser.add_argument(
        "--force-strategy", type=str, default=None,
        help="Force a specific strategy name for part 3 "
             "(e.g., 'A_npc64')"
    )
    args = parser.parse_args()

    sizes = None
    if args.sizes:
        sizes = [int(x) for x in args.sizes.split(",")]

    print(f"[{ts()}] spinnaker_binary_search.py")
    print(f"  Part: {args.part}")
    print(f"  Results dir: {RESULTS_DIR}")
    print()

    # Load data
    print("Loading data...")
    fc1_weight, sample, label = load_sample_data(args.sample_idx)
    print()

    best_strategy = args.force_strategy

    if args.part in ("1", "all"):
        results, breakpoint, last_working = part1_binary_search(
            fc1_weight, sample, label, sizes=sizes)

    if args.part in ("2", "all"):
        strategies, best_name = part2_test_strategies(
            fc1_weight, sample, label)
        if best_name and not best_strategy:
            best_strategy = best_name

    if args.part in ("3", "all"):
        if best_strategy is None:
            # Try to load from part 2 results
            p2_path = RESULTS_DIR / "part2_strategies.json"
            if p2_path.exists():
                with open(p2_path) as f:
                    p2 = json.load(f)
                best_strategy = p2.get("best_strategy")
                print(f"  Loaded best strategy from part 2: {best_strategy}")

        if best_strategy:
            part3_full_test(fc1_weight, best_strategy,
                            num_samples=args.num_samples)
        else:
            print("  ERROR: No strategy found. Run part 2 first or "
                  "use --force-strategy.")

    print(f"\n[{ts()}] Done.")


if __name__ == "__main__":
    main()
