"""
SpiNNaker Latency & Energy Measurement.

Measures wall-clock inference time on SpiNNaker vs CPU/GPU for the FC2-only
pipeline. Also queries sPyNNaker provenance data for synaptic event counts
and estimates energy from published per-chip figures.

Usage (requires .venv-spinnaker):
    python -m experiments.spinnaker_latency_energy --fold 4

For CPU/GPU baseline (regular .venv):
    python -m experiments.spinnaker_latency_energy --cpu-only --fold 4
"""

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RESULTS_DIR, NUM_STEPS


def measure_cpu_gpu_latency(fold=4, encoding="direct", num_samples=100):
    """Measure snnTorch FC2 inference latency on CPU/GPU."""
    import torch
    from src.config import get_device
    from src.models.snn_model import SpikingCNN
    from src.encoding import get_encoder

    device = get_device()
    print(f"Measuring CPU/GPU latency on {device} ({num_samples} samples)")

    # Load model
    model = SpikingCNN().to(device)
    model_path = RESULTS_DIR / "snn" / encoding / f"best_fold{fold}.pt"
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Load hidden features (same as SpiNNaker uses)
    features_path = RESULTS_DIR / "spinnaker_weights" / "hidden_spike_features.npy"
    if features_path.exists():
        features = np.load(features_path)  # (N, 25, 256)
        print(f"  Loaded {features.shape[0]} pre-computed features")
    else:
        print(f"  Features not found at {features_path}")
        print(f"  Measuring full model inference instead")
        features = None

    if features is not None:
        # FC2-only latency (matching SpiNNaker setup)
        fc2_weight = model.fc2.weight.data.to(device)
        fc2_bias = model.fc2.bias.data.to(device) if model.fc2.bias is not None else None
        import snntorch as snn
        from snntorch import surrogate
        lif4 = snn.Leaky(beta=0.95, spike_grad=surrogate.fast_sigmoid(slope=25)).to(device)

        n = min(num_samples, features.shape[0])
        latencies = []

        # Warmup
        for _ in range(10):
            x = torch.tensor(features[0], dtype=torch.float32, device=device)
            mem4 = lif4.init_leaky()
            for t in range(NUM_STEPS):
                cur = torch.matmul(x[t], fc2_weight.T)
                if fc2_bias is not None:
                    cur = cur + fc2_bias
                _, mem4 = lif4(cur.unsqueeze(0), mem4)

        # Timed runs
        for i in range(n):
            x = torch.tensor(features[i], dtype=torch.float32, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            mem4 = lif4.init_leaky()
            mem_acc = torch.zeros(1, 50, device=device)
            for t in range(NUM_STEPS):
                cur = torch.matmul(x[t], fc2_weight.T)
                if fc2_bias is not None:
                    cur = cur + fc2_bias
                spk4, mem4 = lif4(cur.unsqueeze(0), mem4)
                mem_acc += mem4

            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        return {
            "device": str(device),
            "type": "fc2_only",
            "num_samples": n,
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "median_ms": float(np.median(latencies)),
        }
    return None


def measure_spinnaker_latency(fold=4, num_samples=100):
    """Measure SpiNNaker FC2 inference latency."""
    try:
        import pyNN.spiNNaker as sim
    except ImportError:
        print("sPyNNaker not available. Run with .venv-spinnaker.")
        return None

    features_path = RESULTS_DIR / "spinnaker_weights" / "hidden_spike_features.npy"
    if not features_path.exists():
        print(f"Features not found: {features_path}")
        return None

    features = np.load(features_path)

    # Load FC2 connections (combined format: pre, post, weight, delay)
    fold_dir = RESULTS_DIR / "spinnaker_weights" / f"fold{fold}"
    if fold_dir.exists():
        fc2_conns = np.load(fold_dir / "fc2_connections.npy", allow_pickle=True)
    else:
        fc2_conns = np.load(RESULTS_DIR / "spinnaker_weights" / "fc2_connections.npy", allow_pickle=True)

    # Split into excitatory and inhibitory
    fc2_exc = [c for c in fc2_conns if c[2] > 0]
    fc2_inh = [(c[0], c[1], abs(c[2]), c[3]) for c in fc2_conns if c[2] < 0]

    n = min(num_samples, features.shape[0])
    latencies = []

    for i in range(n):
        sample_spikes = features[i]  # (25, 256)

        start = time.perf_counter()

        sim.setup(timestep=1.0)

        # Input population
        spike_times = {j: [] for j in range(256)}
        for t in range(NUM_STEPS):
            for j in range(256):
                if sample_spikes[t, j] > 0:
                    spike_times[j].append(float(t + 1))

        input_pop = sim.Population(
            256,
            sim.SpikeSourceArray(spike_times=[spike_times[j] for j in range(256)]),
            label="input"
        )

        # Output population
        lif_params = {
            "cm": 1.0, "tau_m": 20.0, "tau_refrac": 0.1,
            "v_reset": 0.0, "v_rest": 0.0, "v_thresh": 1.0,
            "tau_syn_E": 5.0, "tau_syn_I": 5.0,
        }
        output_pop = sim.Population(50, sim.IF_curr_exp(**lif_params), label="output")
        output_pop.record(["v"])

        # Projections
        if len(fc2_exc) > 0:
            sim.Projection(input_pop, output_pop,
                          sim.FromListConnector(fc2_exc.tolist()),
                          receptor_type="excitatory")
        if len(fc2_inh) > 0:
            sim.Projection(input_pop, output_pop,
                          sim.FromListConnector(fc2_inh.tolist()),
                          receptor_type="inhibitory")

        sim.run(NUM_STEPS)

        # Read output
        v_data = output_pop.get_data("v")
        sim.end()

        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

        if (i + 1) % 10 == 0:
            print(f"  Sample {i+1}/{n}: {latencies[-1]:.1f} ms")

    return {
        "device": "SpiNNaker",
        "type": "fc2_only",
        "num_samples": n,
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "median_ms": float(np.median(latencies)),
        "note": "Includes sim.setup/sim.end overhead per sample",
    }


def estimate_energy(features, num_samples=100):
    """Estimate SpiNNaker energy from spike counts and published figures."""
    n = min(num_samples, features.shape[0])

    # Count synaptic events: each input spike to each connected output neuron
    # FC2: 256 -> 50, fully connected = 12,800 synapses
    # Each input spike generates 50 synaptic events
    total_input_spikes = 0
    for i in range(n):
        total_input_spikes += features[i].sum()

    avg_input_spikes = total_input_spikes / n
    avg_synaptic_events = avg_input_spikes * 50  # each spike hits 50 output neurons

    # Published energy figures
    # SpiNNaker 1: ~5.9 µJ per synaptic event (Gutzen et al. 2022, Ostrau et al. 2022)
    # SpiNNaker 1 idle power: ~255 mW per chip (Stromatias et al. 2013)
    # Simulation time: 25ms
    energy_per_event_uj = 5.9
    idle_power_mw = 255
    sim_time_ms = NUM_STEPS  # 25 ms

    e_idle_uj = idle_power_mw * sim_time_ms  # µJ
    e_synaptic_uj = avg_synaptic_events * energy_per_event_uj
    e_total_uj = e_idle_uj + e_synaptic_uj

    return {
        "avg_input_spikes_per_sample": float(avg_input_spikes),
        "avg_synaptic_events_per_sample": float(avg_synaptic_events),
        "energy_per_synaptic_event_uJ": energy_per_event_uj,
        "idle_power_mW": idle_power_mw,
        "sim_time_ms": sim_time_ms,
        "estimated_idle_energy_uJ": float(e_idle_uj),
        "estimated_synaptic_energy_uJ": float(e_synaptic_uj),
        "estimated_total_energy_uJ": float(e_total_uj),
        "estimated_total_energy_mJ": float(e_total_uj / 1000),
        "note": "Based on Gutzen et al. 2022 and Stromatias et al. 2013 figures",
    }


def main():
    parser = argparse.ArgumentParser(description="SpiNNaker latency & energy")
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--cpu-only", action="store_true",
                        help="Only measure CPU/GPU, skip SpiNNaker")
    args = parser.parse_args()

    save_dir = RESULTS_DIR / "spinnaker_results" / "latency_energy"
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {"fold": args.fold}

    # CPU/GPU baseline
    print("\n=== CPU/GPU Latency ===")
    cpu_result = measure_cpu_gpu_latency(args.fold, num_samples=args.num_samples)
    if cpu_result:
        results["cpu_gpu"] = cpu_result
        print(f"  Mean: {cpu_result['mean_ms']:.3f} ms ± {cpu_result['std_ms']:.3f}")

    # SpiNNaker
    if not args.cpu_only:
        print("\n=== SpiNNaker Latency ===")
        spk_result = measure_spinnaker_latency(args.fold, args.num_samples)
        if spk_result:
            results["spinnaker"] = spk_result
            print(f"  Mean: {spk_result['mean_ms']:.1f} ms ± {spk_result['std_ms']:.1f}")

    # Energy estimation
    print("\n=== Energy Estimation ===")
    features_path = RESULTS_DIR / "spinnaker_weights" / "hidden_spike_features.npy"
    if features_path.exists():
        features = np.load(features_path)
        energy = estimate_energy(features, args.num_samples)
        results["energy"] = energy
        print(f"  Avg input spikes: {energy['avg_input_spikes_per_sample']:.0f}")
        print(f"  Avg synaptic events: {energy['avg_synaptic_events_per_sample']:.0f}")
        print(f"  Est. energy: {energy['estimated_total_energy_mJ']:.2f} mJ/sample")

    # Comparison
    if "cpu_gpu" in results and "energy" in results:
        print(f"\n=== Comparison ===")
        print(f"  CPU/GPU FC2: {results['cpu_gpu']['mean_ms']:.3f} ms/sample")
        if "spinnaker" in results:
            speedup = results["spinnaker"]["mean_ms"] / results["cpu_gpu"]["mean_ms"]
            print(f"  SpiNNaker FC2: {results['spinnaker']['mean_ms']:.1f} ms/sample")
            print(f"  SpiNNaker/CPU ratio: {speedup:.1f}x")
        print(f"  SpiNNaker energy: {results['energy']['estimated_total_energy_mJ']:.2f} mJ")
        print(f"  NeuroBench SNN (software): 0.000976 mJ")
        print(f"  NeuroBench ANN (software): 0.000463 mJ")

    # Save
    with open(save_dir / f"latency_energy_fold{args.fold}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_dir / f'latency_energy_fold{args.fold}.json'}")


if __name__ == "__main__":
    main()
