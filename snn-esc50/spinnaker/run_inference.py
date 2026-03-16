"""
Run SNN inference on SpiNNaker hardware via sPyNNaker/PyNN.

This script is designed to run on EBRAINS or any machine with sPyNNaker
installed and configured for SpiNNaker access.

Deployment strategy (FC-only):
  - The convolutional feature extraction (Conv+BN+Pool) runs on CPU
    to produce a 2304-dimensional feature vector per timestep.
  - The FC classifier (2304->256->50 with LIF neurons) runs on SpiNNaker.
  - This is the most practical approach for SpiNNaker1 since unrolling
    the full conv network would require ~221K neurons (too many).

Usage (on EBRAINS Jupyter or SpiNNaker-connected machine):
    python run_inference.py --weights-dir ./spinnaker_weights --num-samples 10

Note on weight handling:
    SpiNNaker/PyNN separates excitatory (positive) and inhibitory (negative)
    connections. This script creates two Projections per layer: one for
    positive weights and one for negative weights (with abs values).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def build_spinnaker_network(weights_dir: str, num_classes: int = 50):
    """Build a sPyNNaker network from pre-trained weights.

    Uses PyNN with sPyNNaker backend. Creates LIF populations
    connected by the trained FC weight matrices.

    NOTE: This requires sPyNNaker to be installed and configured.
    It will NOT work on a standard laptop -- run on EBRAINS.

    Args:
        weights_dir: Path to directory with converted weight files.
        num_classes: Number of output classes.

    Returns:
        (sim, input_pop, hidden_pop, output_pop) or None if sPyNNaker
        is not available.
    """
    try:
        import pyNN.spiNNaker as sim
    except ImportError:
        print("=" * 60)
        print("ERROR: sPyNNaker not available.")
        print("This script must be run on EBRAINS or a SpiNNaker-")
        print("connected machine.")
        print()
        print("To get access:")
        print("  1. Register at https://ebrains.eu/register")
        print("  2. Email neuromorphic@humanbrainproject.eu with")
        print("     your EBRAINS username to request access")
        print("  3. Use the EBRAINS Collaboratory Job Manager or")
        print("     Jupyter notebooks to run this script")
        print("=" * 60)
        return None

    weights_dir = Path(weights_dir)

    # Load pre-computed connection lists (pre, post, weight, delay)
    fc1_connections = np.load(weights_dir / "fc1_connections.npy")
    fc2_connections = np.load(weights_dir / "fc2_connections.npy")

    # Load metadata for network dimensions
    with open(weights_dir / "metadata.json") as f:
        metadata = json.load(f)

    input_size = metadata["architecture"]["flatten_size"]  # 2304
    hidden_size = metadata["architecture"]["fc1"]["out"]   # 256
    output_size = metadata["architecture"]["fc2"]["out"]   # 50

    print(f"Network topology: {input_size} -> {hidden_size} -> {output_size}")
    print(f"FC1 connections: {len(fc1_connections)}")
    print(f"FC2 connections: {len(fc2_connections)}")

    # LIF neuron parameters matching snnTorch beta=0.95
    # beta = exp(-dt/tau_m) => tau_m = -dt/ln(beta) = -1/ln(0.95) = 19.5 ~ 20ms
    lif_params = metadata.get("spinnaker_lif_params", {})
    lif_cell_params = {
        "cm": lif_params.get("cm", 1.0),
        "tau_m": lif_params.get("tau_m", 20.0),
        "tau_refrac": lif_params.get("tau_refrac", 2.0),
        "v_reset": lif_params.get("v_reset", 0.0),
        "v_rest": lif_params.get("v_rest", 0.0),
        "v_thresh": lif_params.get("v_thresh", 1.0),
    }

    print(f"LIF parameters: {lif_cell_params}")

    # --- Set up simulation ---
    sim.setup(timestep=1.0)  # 1ms timestep

    # --- Input population (SpikeSourceArray) ---
    input_pop = sim.Population(
        input_size,
        sim.SpikeSourceArray(spike_times=[]),
        label="input",
    )

    # --- Hidden layer (LIF neurons) ---
    hidden_pop = sim.Population(
        hidden_size,
        sim.IF_curr_exp(**lif_cell_params),
        label="hidden",
    )
    hidden_pop.record("spikes")

    # --- Output layer (LIF neurons) ---
    output_pop = sim.Population(
        output_size,
        sim.IF_curr_exp(**lif_cell_params),
        label="output",
    )
    output_pop.record(["spikes", "v"])

    # --- Connect layers ---
    # SpiNNaker requires separate excitatory and inhibitory projections

    # FC1: Input -> Hidden
    fc1_exc = fc1_connections[fc1_connections[:, 2] > 0]
    fc1_inh = fc1_connections[fc1_connections[:, 2] < 0].copy()
    fc1_inh[:, 2] = np.abs(fc1_inh[:, 2])  # inhibitory weights must be positive

    print(f"FC1: {len(fc1_exc)} excitatory, {len(fc1_inh)} inhibitory connections")

    if len(fc1_exc) > 0:
        sim.Projection(
            input_pop, hidden_pop,
            sim.FromListConnector(fc1_exc.tolist()),
            receptor_type="excitatory",
            label="fc1_exc",
        )
    if len(fc1_inh) > 0:
        sim.Projection(
            input_pop, hidden_pop,
            sim.FromListConnector(fc1_inh.tolist()),
            receptor_type="inhibitory",
            label="fc1_inh",
        )

    # FC2: Hidden -> Output
    fc2_exc = fc2_connections[fc2_connections[:, 2] > 0]
    fc2_inh = fc2_connections[fc2_connections[:, 2] < 0].copy()
    fc2_inh[:, 2] = np.abs(fc2_inh[:, 2])

    print(f"FC2: {len(fc2_exc)} excitatory, {len(fc2_inh)} inhibitory connections")

    if len(fc2_exc) > 0:
        sim.Projection(
            hidden_pop, output_pop,
            sim.FromListConnector(fc2_exc.tolist()),
            receptor_type="excitatory",
            label="fc2_exc",
        )
    if len(fc2_inh) > 0:
        sim.Projection(
            hidden_pop, output_pop,
            sim.FromListConnector(fc2_inh.tolist()),
            receptor_type="inhibitory",
            label="fc2_inh",
        )

    return sim, input_pop, hidden_pop, output_pop


def run_inference_sample(sim, input_pop, output_pop,
                         spike_input, num_steps=25):
    """Run inference for a single sample on SpiNNaker.

    Args:
        sim: sPyNNaker simulator module.
        input_pop: Input SpikeSourceArray population.
        output_pop: Output LIF population.
        spike_input: Binary spike array (num_steps, input_size).
        num_steps: Simulation duration in ms.

    Returns:
        (predicted_class, spike_counts) tuple.
    """
    input_size = spike_input.shape[1]

    # Set spike times for each input neuron
    for i in range(input_size):
        spike_times = np.where(spike_input[:, i] > 0.5)[0].astype(float).tolist()
        input_pop[i].set(spike_times=spike_times)

    # Run simulation
    sim.run(num_steps)

    # Read output spikes
    output_data = output_pop.get_data("spikes")
    spike_counts = np.zeros(output_pop.size)

    for spiketrain in output_data.segments[-1].spiketrains:
        neuron_id = int(spiketrain.annotations.get("source_index", 0))
        spike_counts[neuron_id] = len(spiketrain)

    predicted = int(np.argmax(spike_counts))

    sim.reset()

    return predicted, spike_counts


def main():
    parser = argparse.ArgumentParser(
        description="SpiNNaker SNN inference for ESC-50"
    )
    parser.add_argument("--weights-dir", required=True,
                        help="Directory with converted sPyNNaker weights")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of test samples to run")
    parser.add_argument("--num-steps", type=int, default=25,
                        help="Simulation timesteps (ms)")
    args = parser.parse_args()

    print("=" * 60)
    print("SpiNNaker SNN Inference - ESC-50 Sound Classification")
    print("=" * 60)

    print("\nBuilding SpiNNaker network...")
    result = build_spinnaker_network(args.weights_dir)

    if result is None:
        print("\nCould not build network. Exiting.")
        return

    sim, input_pop, hidden_pop, output_pop = result

    print(f"\nRunning inference on {args.num_samples} samples...")
    print("NOTE: Using random spike input for connectivity test.")
    print("For real inference, integrate with the feature extraction pipeline.")

    correct = 0
    total = args.num_samples
    times = []

    for i in range(total):
        # Random spike input (replace with real rate-coded features)
        fake_input = (np.random.rand(args.num_steps, input_pop.size) > 0.9).astype(float)
        fake_label = np.random.randint(0, 50)

        start = time.time()
        pred, counts = run_inference_sample(
            sim, input_pop, output_pop, fake_input, args.num_steps,
        )
        elapsed = time.time() - start
        times.append(elapsed)

        if pred == fake_label:
            correct += 1

        if i < 5:
            print(f"  Sample {i}: pred={pred}, label={fake_label}, "
                  f"total_spikes={counts.sum():.0f}, time={elapsed*1000:.1f}ms")

    sim.end()

    avg_time = np.mean(times)
    print(f"\nResults:")
    print(f"  Average inference time: {avg_time*1000:.1f}ms per sample")
    print(f"  Random accuracy: {correct/total:.1%} (expected ~2% for 50 classes)")

    # Save results
    output_dir = Path(args.weights_dir).parent / "spinnaker_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "platform": "SpiNNaker",
        "num_samples": total,
        "avg_inference_ms": avg_time * 1000,
        "random_accuracy": correct / total,
        "network": {
            "input_neurons": input_pop.size,
            "hidden_neurons": hidden_pop.size,
            "output_neurons": output_pop.size,
            "total_neurons": input_pop.size + hidden_pop.size + output_pop.size,
        },
        "note": "Random input test -- replace with real encoded spectrograms",
    }
    with open(output_dir / "spinnaker_inference.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
