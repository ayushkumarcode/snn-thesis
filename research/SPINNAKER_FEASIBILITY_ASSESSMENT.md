# SpiNNaker Deployment Feasibility Assessment for SNN Thesis Project

**Date:** 2026-02-28
**Scope:** Comprehensive assessment of whether deploying trained SNN models on SpiNNaker neuromorphic hardware is feasible as an optional addition to the thesis project.
**Verdict:** FEASIBLE AS OPTIONAL ADD-ON, but with significant caveats. SpiNNaker1 is accessible via EBRAINS. SpiNNaker2 access is restricted. The NIR conversion pathway from snnTorch exists but is early-stage.

---

## Executive Summary

SpiNNaker deployment is a viable optional enhancement to the thesis, NOT a core requirement. The realistic workflow is: (1) train your SNN in snnTorch on a regular computer, (2) export to NIR format using `snntorch.export_nir`, (3) deploy on SpiNNaker for inference, (4) compare real hardware energy/latency against simulated NeuroBench estimates. This has been demonstrated in recent papers (2024-2025) for DVS gesture recognition and RL tasks on SpiNNaker2. However, SpiNNaker1 (accessible via EBRAINS) uses sPyNNaker/PyNN which is a fundamentally different programming paradigm from snnTorch, and the NIR pathway to SpiNNaker2 requires restricted `py-spinnaker2` software. The safest approach is to treat SpiNNaker deployment as a "bonus chapter" that could add 2-4 weeks of work if access and conversion go smoothly.

---

## Question 1: sPyNNaker Programming Model vs snnTorch

### Fundamental Paradigm Difference

These are **completely different programming models**. They share the concept of spiking neurons but differ in almost every other way.

| Aspect | snnTorch | sPyNNaker (PyNN) |
|--------|----------|------------------|
| **Paradigm** | Deep learning framework extension | Neuroscience simulator interface |
| **Base framework** | PyTorch | PyNN (simulator-independent) |
| **Training** | Surrogate gradient backpropagation | STDP (on-chip), or pre-trained weights |
| **GPU support** | Yes (via PyTorch CUDA) | No (runs on SpiNNaker ARM cores) |
| **Network definition** | Sequential layers (like PyTorch) | Populations + Projections + Connectors |
| **Neuron access** | Per-timestep forward pass | Biological simulation (real-time) |
| **Convolutions** | `nn.Conv2d` (standard PyTorch) | `KernelConnector` / `ConvolutionConnector` |
| **Weight format** | 32-bit float tensors | 16-bit fixed-point integers |
| **Batching** | Mini-batch training | Single-instance simulation |
| **Output** | Loss, accuracy, spike tensors | Spike trains, membrane voltage traces |

### Side-by-Side Code Comparison

**snnTorch: Simple 2-layer SNN**

```python
import torch
import torch.nn as nn
import snntorch as snn

# Define network as sequential PyTorch layers
class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 500)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(500, 10)
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)

        return torch.stack(spk_rec)

# Training: standard PyTorch loop with surrogate gradients
net = SimpleSNN()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = snn.functional.mse_count_loss()
```

**sPyNNaker (PyNN): Equivalent network on SpiNNaker**

```python
import pyNN.spiNNaker as sim

sim.setup(timestep=1.0)

# Define neuron parameters (LIF)
cell_params = {
    'cm': 0.25,         # membrane capacitance (nF)
    'tau_m': 20.0,       # membrane time constant (ms)
    'tau_refrac': 2.0,   # refractory period (ms)
    'tau_syn_E': 5.0,    # excitatory synapse time constant
    'tau_syn_I': 5.0,    # inhibitory synapse time constant
    'v_reset': -70.0,    # reset voltage (mV)
    'v_rest': -65.0,     # resting voltage (mV)
    'v_thresh': -50.0    # threshold voltage (mV)
}

# Create populations (NOT layers -- groups of neurons)
input_pop = sim.Population(784, sim.SpikeSourceArray(spike_times=[...]))
hidden_pop = sim.Population(500, sim.IF_curr_exp(**cell_params))
output_pop = sim.Population(10, sim.IF_curr_exp(**cell_params))

# Create projections (connections between populations)
proj1 = sim.Projection(
    input_pop, hidden_pop,
    sim.AllToAllConnector(),
    sim.StaticSynapse(weight=0.5, delay=1.0)
)
proj2 = sim.Projection(
    hidden_pop, output_pop,
    sim.AllToAllConnector(),
    sim.StaticSynapse(weight=0.5, delay=1.0)
)

# Record spikes
output_pop.record(['spikes', 'v'])

# Run simulation (in biological real-time on SpiNNaker)
sim.run(1000)  # 1000ms

# Retrieve data
spikes = output_pop.get_data('spikes')
voltages = output_pop.get_data('v')

sim.end()
```

