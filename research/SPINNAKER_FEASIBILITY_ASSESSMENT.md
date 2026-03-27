# SpiNNaker deployment feasibility -- can i actually get my SNN running on it?

Looking into whether deploying trained SNN models on SpiNNaker neuromorphic hardware is feasible as an optional addition to the thesis project. Short answer: **feasible as an optional add-on**, but with significant caveats. SpiNNaker1 is accessible via EBRAINS. SpiNNaker2 access is restricted. The NIR conversion pathway from snnTorch exists but is early-stage.

The realistic workflow would be: (1) train SNN in snnTorch on a regular computer, (2) export to NIR format using `snntorch.export_nir`, (3) deploy on SpiNNaker for inference, (4) compare real hardware energy/latency against simulated NeuroBench estimates. This has been demonstrated in recent papers (2024-2025) for DVS gesture recognition and RL tasks on SpiNNaker2. However, SpiNNaker1 (accessible via EBRAINS) uses sPyNNaker/PyNN which is a fundamentally different programming model from snnTorch, and the NIR pathway to SpiNNaker2 requires restricted `py-spinnaker2` software. Safest approach: treat SpiNNaker deployment as a "bonus chapter" that could add 2-4 weeks if access and conversion go smoothly.

---

## 1. sPyNNaker vs snnTorch -- completely different worlds

These are **completely different programming models**. They share the concept of spiking neurons but differ in almost everything else.

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

### Side-by-side code comparison

**snnTorch: Simple 2-layer SNN**

```python
import torch
import torch.nn as nn
import snntorch as snn

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

net = SimpleSNN()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = snn.functional.mse_count_loss()
```

**sPyNNaker (PyNN): Same network on SpiNNaker**

```python
import pyNN.spiNNaker as sim

sim.setup(timestep=1.0)

cell_params = {
    'cm': 0.25, 'tau_m': 20.0, 'tau_refrac': 2.0,
    'tau_syn_E': 5.0, 'tau_syn_I': 5.0,
    'v_reset': -70.0, 'v_rest': -65.0, 'v_thresh': -50.0
}

# Populations (NOT layers -- groups of neurons)
input_pop = sim.Population(784, sim.SpikeSourceArray(spike_times=[...]))
hidden_pop = sim.Population(500, sim.IF_curr_exp(**cell_params))
output_pop = sim.Population(10, sim.IF_curr_exp(**cell_params))

# Projections (connections between populations)
proj1 = sim.Projection(input_pop, hidden_pop,
    sim.AllToAllConnector(), sim.StaticSynapse(weight=0.5, delay=1.0))
proj2 = sim.Projection(hidden_pop, output_pop,
    sim.AllToAllConnector(), sim.StaticSynapse(weight=0.5, delay=1.0))

output_pop.record(['spikes', 'v'])
sim.run(1000)  # 1000ms

spikes = output_pop.get_data('spikes')
voltages = output_pop.get_data('v')
sim.end()
```

### Key differences summed up

1. snnTorch thinks in **layers** (Linear, Conv2d, Leaky). PyNN thinks in **populations** and **projections**.
2. snnTorch trains via **backpropagation** through surrogate gradients. sPyNNaker typically does NOT train -- it runs inference or STDP.
3. snnTorch uses PyTorch tensors on GPU. sPyNNaker compiles to ARM machine code on SpiNNaker chips.
4. You cannot directly run snnTorch code on SpiNNaker. They are fundamentally different systems.

---

## 2. snnTorch to SpiNNaker conversion

### The NIR pathway

Yes, a conversion pathway exists -- via NIR (Neuromorphic Intermediate Representation). This is the critical finding.

NIR is a standardized graph-based format bridging multiple SNN frameworks and hardware platforms. It currently connects:
- **Software**: snnTorch, Norse, Lava, Nengo, Rockpool, Sinabs, Spyx
- **Hardware**: Loihi 2 (via Lava), Speck (via Sinabs), **SpiNNaker2** (via py-spinnaker2), Xylo (via Rockpool)

Source: [Nature Communications paper on NIR](https://www.nature.com/articles/s41467-024-52259-9)

### How it works

```python
# Step 1: Train in snnTorch (standard workflow)
# ... train with surrogate gradients ...

# Step 2: Export to NIR
from snntorch.export_nir import export_to_nir
sample_data = torch.randn(1, 784)
nir_graph = export_to_nir(net, sample_data)

# Step 3: Save NIR file
import nir
nir.write("my_model.nir", nir_graph)

# Step 4: Import into SpiNNaker2 via py-spinnaker2
# (requires py-spinnaker2 library and SpiNNaker2 hardware access)
```

NIR captures each layer as a graph node -- convolution layers (weights, stride, padding), linear layers (weight matrices), LIF neurons (time constants, thresholds), pooling layers.

### Proven end-to-end pipeline

A 2025 paper demonstrated the complete pipeline:
1. Train SNN in snnTorch (Conv2D + LIF architecture)
2. Quantize weights to 8-bit (PTQ or QAT)
3. Export to NIR
4. Deploy on SpiNNaker2 via py-spinnaker2
5. Result: **94.13% accuracy on-chip** (vs 95.07% on GPU)

Source: [Efficient Deployment of SNNs on SpiNNaker2](https://arxiv.org/html/2504.06748v1)

### The big caveat: SpiNNaker1 vs SpiNNaker2

- **NIR -> SpiNNaker2**: Supported (via py-spinnaker2). But SpiNNaker2 access is restricted -- `py-spinnaker2` dependencies are in private repos.
- **NIR -> SpiNNaker1**: NOT directly supported. SpiNNaker1 uses sPyNNaker/PyNN, and there's no automated NIR-to-sPyNNaker converter.
- **Manual conversion to SpiNNaker1**: You'd need to manually translate trained weights into PyNN `FromListConnector` weight matrices. Doable but tedious.

---

## 3. SpiNNaker capabilities

### Supported neuron models

| Model | Type | Description |
|-------|------|-------------|
| `IF_curr_exp` | Standard | Current-based LIF, exponential synapses |
| `IF_cond_exp` | Standard | Conductance-based LIF, exponential synapses |
| `IF_curr_alpha` | Standard | Current-based LIF, alpha synapses |
| `IF_curr_delta` | Standard | Current-based LIF, delta synapses |
| `Izhikevich` | Standard | Current-based Izhikevich model |
| `Izhikevich_cond` | Extended | Conductance-based Izhikevich |
| `IFCurrExpCa2Adaptive` | Extended | Calcium-adaptive LIF |
| `IFCondExpStoc` | Extended | Stochastic threshold conductance-based |

Source: [sPyNNaker Models and Limitations](http://spinnakermanchester.github.io/spynnaker/6.0.0/SPyNNakerModelsAndLimitations.html)
