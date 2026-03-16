"""
PyNN script for SpiNNaker inference on EBRAINS.

This script is designed to be submitted to the EBRAINS Neuromorphic
Computing Platform (either via the Job Manager or the nmpi Python client).

It loads pre-extracted spike features and FC connection weights, builds
an SNN on SpiNNaker, and runs inference on the test samples.

NOTE: This script runs ON SpiNNaker via EBRAINS -- it is NOT meant to
be run locally. Use submit_ebrains.py to submit this script.

The script expects the following files in the working directory:
  - fc1_connections.npy: (N, 4) array of (pre, post, weight, delay)
  - fc2_connections.npy: (N, 4) array of (pre, post, weight, delay)
  - test_spike_features.npy: (num_samples, 25, 2304) binary spike arrays
  - test_labels.npy: (num_samples,) ground truth labels
  - metadata.json: network architecture and parameter info
"""

import json
import time
import numpy as np

# PyNN with SpiNNaker backend (available on EBRAINS)
from pyNN.utility import get_simulator
sim, options = get_simulator()

# ============================================================
# Load data
# ============================================================
print("Loading weights and test data...")

fc1_connections = np.load("fc1_connections.npy")
fc2_connections = np.load("fc2_connections.npy")
test_features = np.load("test_spike_features.npy")
test_labels = np.load("test_labels.npy")

with open("metadata.json") as f:
    metadata = json.load(f)

INPUT_SIZE = metadata["architecture"]["flatten_size"]    # 2304
HIDDEN_SIZE = metadata["architecture"]["fc1"]["out"]     # 256
OUTPUT_SIZE = metadata["architecture"]["fc2"]["out"]     # 50
NUM_STEPS = metadata["snn_params"]["num_steps"]          # 25
NUM_SAMPLES = len(test_labels)

print(f"Network: {INPUT_SIZE} -> {HIDDEN_SIZE} -> {OUTPUT_SIZE}")
print(f"Test samples: {NUM_SAMPLES}")
print(f"Simulation timesteps: {NUM_STEPS}")

# ============================================================
# LIF neuron parameters (matching snnTorch beta=0.95)
# ============================================================
lif_params = {
    "cm": 1.0,
    "tau_m": 20.0,
    "tau_refrac": 2.0,
    "v_reset": 0.0,
    "v_rest": 0.0,
    "v_thresh": 1.0,
}

# ============================================================
# Split connections into excitatory and inhibitory
# ============================================================
fc1_exc = fc1_connections[fc1_connections[:, 2] > 0].tolist()
fc1_inh_raw = fc1_connections[fc1_connections[:, 2] < 0].copy()
fc1_inh_raw[:, 2] = np.abs(fc1_inh_raw[:, 2])
fc1_inh = fc1_inh_raw.tolist()

fc2_exc = fc2_connections[fc2_connections[:, 2] > 0].tolist()
fc2_inh_raw = fc2_connections[fc2_connections[:, 2] < 0].copy()
fc2_inh_raw[:, 2] = np.abs(fc2_inh_raw[:, 2])
fc2_inh = fc2_inh_raw.tolist()

print(f"FC1: {len(fc1_exc)} exc + {len(fc1_inh)} inh connections")
print(f"FC2: {len(fc2_exc)} exc + {len(fc2_inh)} inh connections")

# ============================================================
# Run inference on each test sample
# ============================================================
results = []
correct = 0
total_time = 0.0

for sample_idx in range(NUM_SAMPLES):
    spike_input = test_features[sample_idx]  # (25, 2304)
    true_label = int(test_labels[sample_idx])

    # --- Setup simulation ---
    sim.setup(timestep=1.0)

    # --- Create populations ---
    # Input: SpikeSourceArray with pre-computed spike times
    spike_times_list = []
    for neuron_idx in range(INPUT_SIZE):
        times = np.where(spike_input[:, neuron_idx] > 0.5)[0].astype(float).tolist()
        spike_times_list.append(times)

    input_pop = sim.Population(
        INPUT_SIZE,
        sim.SpikeSourceArray,
        {"spike_times": spike_times_list},
        label="input",
    )

    hidden_pop = sim.Population(
        HIDDEN_SIZE,
        sim.IF_curr_exp(**lif_params),
        label="hidden",
    )
    hidden_pop.record("spikes")

    output_pop = sim.Population(
        OUTPUT_SIZE,
        sim.IF_curr_exp(**lif_params),
        label="output",
    )
    output_pop.record(["spikes", "v"])

    # --- Create projections ---
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

    # --- Run simulation ---
    start = time.time()
    sim.run(NUM_STEPS)
    elapsed = time.time() - start
    total_time += elapsed

    # --- Read output spikes ---
    output_data = output_pop.get_data("spikes")
    spike_counts = np.zeros(OUTPUT_SIZE)
    for spiketrain in output_data.segments[-1].spiketrains:
        neuron_id = int(spiketrain.annotations.get("source_index", 0))
        spike_counts[neuron_id] = len(spiketrain)

    predicted = int(np.argmax(spike_counts))
    is_correct = (predicted == true_label)
    if is_correct:
        correct += 1

    results.append({
        "sample": sample_idx,
        "true_label": true_label,
        "predicted": predicted,
        "correct": is_correct,
        "total_spikes": int(spike_counts.sum()),
        "inference_ms": elapsed * 1000,
    })

    print(f"  Sample {sample_idx}: pred={predicted}, true={true_label}, "
          f"{'OK' if is_correct else 'WRONG'}, "
          f"spikes={spike_counts.sum():.0f}, time={elapsed*1000:.1f}ms")

    sim.end()

# ============================================================
# Summary
# ============================================================
accuracy = correct / NUM_SAMPLES if NUM_SAMPLES > 0 else 0
avg_time = total_time / NUM_SAMPLES if NUM_SAMPLES > 0 else 0

print(f"\n{'='*60}")
print(f"SpiNNaker Inference Results - ESC-50")
print(f"{'='*60}")
print(f"Samples:  {NUM_SAMPLES}")
print(f"Correct:  {correct}")
print(f"Accuracy: {accuracy:.1%}")
print(f"Avg time: {avg_time*1000:.1f}ms per sample")
print(f"Total:    {total_time:.1f}s")

# Save results
output = {
    "platform": "SpiNNaker (EBRAINS)",
    "num_samples": NUM_SAMPLES,
    "correct": correct,
    "accuracy": accuracy,
    "avg_inference_ms": avg_time * 1000,
    "total_time_s": total_time,
    "network": {
        "input_neurons": INPUT_SIZE,
        "hidden_neurons": HIDDEN_SIZE,
        "output_neurons": OUTPUT_SIZE,
        "total_neurons": INPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE,
    },
    "per_sample": results,
}

with open("spinnaker_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to spinnaker_results.json")
