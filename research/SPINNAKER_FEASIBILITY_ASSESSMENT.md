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

### Key Differences Summarized

1. **snnTorch thinks in layers** (Linear, Conv2d, Leaky). **PyNN thinks in populations** (groups of neurons) and **projections** (synaptic connections).
2. **snnTorch trains via backpropagation** through surrogate gradients. **sPyNNaker typically does NOT train** -- it runs inference or STDP-based learning.
3. **snnTorch uses PyTorch tensors** on GPU. **sPyNNaker compiles to ARM machine code** on SpiNNaker chips.
4. **You cannot directly run snnTorch code on SpiNNaker.** They are fundamentally different systems.

---

## Question 2: snnTorch to SpiNNaker Conversion

### The NIR Pathway (Neuromorphic Intermediate Representation)

**YES, a conversion pathway exists -- via NIR.** This is the critical finding.

NIR (Neuromorphic Intermediate Representation) is a standardized graph-based format that bridges multiple SNN frameworks and hardware platforms. It currently connects:
- **Software**: snnTorch, Norse, Lava, Nengo, Rockpool, Sinabs, Spyx
- **Hardware**: Loihi 2 (via Lava), Speck (via Sinabs), **SpiNNaker2** (via py-spinnaker2), Xylo (via Rockpool)

**Source:** [Nature Communications paper on NIR](https://www.nature.com/articles/s41467-024-52259-9)

### How the Conversion Works

```python
# Step 1: Train in snnTorch (standard workflow)
import snntorch as snn
import torch

net = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(784, 500),
    snn.Leaky(beta=0.9, init_hidden=True),
    torch.nn.Linear(500, 10),
    snn.Leaky(beta=0.9, init_hidden=True, output=True)
)
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

### What NIR Captures

Each layer becomes a graph node:
- Convolution layers: weights, stride, padding
- Linear layers: weight matrices
- LIF neurons: time constants (tau), membrane resistance, voltage leak, thresholds
- Pooling layers: kernel size, stride, padding

### Proven End-to-End Pipeline (DVS Gesture Recognition)

A 2025 paper demonstrated the complete pipeline:
1. Train SNN in snnTorch (Conv2D + LIF architecture)
2. Quantize weights to 8-bit (PTQ or QAT)
3. Export to NIR
4. Deploy on SpiNNaker2 via py-spinnaker2
5. **Result: 94.13% accuracy on-chip** (vs 95.07% on GPU)

**Source:** [Efficient Deployment of SNNs on SpiNNaker2](https://arxiv.org/html/2504.06748v1)

### CRITICAL CAVEAT: SpiNNaker1 vs SpiNNaker2

- **NIR -> SpiNNaker2**: Supported (via py-spinnaker2). But SpiNNaker2 access is **restricted** -- `py-spinnaker2` dependencies are in private repositories.
- **NIR -> SpiNNaker1**: NOT directly supported. SpiNNaker1 uses sPyNNaker/PyNN, and there is no automated NIR-to-sPyNNaker converter.
- **Manual conversion to SpiNNaker1**: You would need to manually translate your trained weights into PyNN `FromListConnector` weight matrices. This is doable but tedious.

---

## Question 3: SpiNNaker Capabilities

### Supported Neuron Models

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

**Source:** [sPyNNaker Models and Limitations](http://spinnakermanchester.github.io/spynnaker/6.0.0/SPyNNakerModelsAndLimitations.html)

### Convolutional Support

- **SpiNNaker1 (sPyNNaker):** Supports `KernelConnector` and `ConvolutionConnector` for structured convolutional connectivity. Also supports digital retina input. However, this is NOT the same as PyTorch `nn.Conv2d` -- you define convolutions as connectivity patterns between populations.
- **SpiNNaker2 (py-spinnaker2):** Full convolutional layer support through NIR import. Conv2D layers are mapped directly.
- **Community code:** [SpikingConvNet](https://github.com/SvenGronauer/SpikingConvNet) provides infrastructure for arbitrarily deep spiking CNNs on SpiNNaker.

### Training Methods

| Method | SpiNNaker1 | SpiNNaker2 |
|--------|-----------|-----------|
| STDP (spike-timing dependent plasticity) | YES (native) | YES |
| Surrogate gradient backpropagation | NO (not on-chip) | Limited (e-prop demonstrated) |
| Pre-trained weight loading | YES (via FromListConnector) | YES (via NIR/py-spinnaker2) |
| Reinforcement learning (reward-modulated) | YES (3-factor STDP) | YES (spiking Q-networks demonstrated) |

**The standard approach is: Train off-chip (snnTorch/PyTorch) -> Load weights -> Run inference on SpiNNaker.**

### Maximum Network Size

- **Per core:** 256 neurons maximum
- **Per chip (SpiNNaker1):** 18 cores = ~4,608 neurons max
- **48-chip board (SpiNNaker1):** ~220,000 neurons
- **Full million-core machine:** ~500,000+ neurons with complex connectivity
- **SpiNNaker2 single chip:** 153 ARM cores, 19MB SRAM, 2GB DRAM
- **Weight precision:** 16-bit fixed-point (SpiNNaker1), 8-bit integer (SpiNNaker2 via quantization)

### Delay and Timing

- Delays: 1-144 timesteps (delays > 16 require automatic delay populations)
- Timestep: configurable (typically 1.0ms)
- Real-time: simulations run at biological real-time speed

---

## Question 4: SpiNNaker Access at Manchester

### SpiNNaker1 Access (EBRAINS -- Available NOW)

**This is the accessible option.** The full SpiNNaker1 machine (1 million cores) is hosted at the University of Manchester and accessible remotely via EBRAINS.

**How to get access:**
1. Register for a free EBRAINS account at [ebrains.eu](https://www.ebrains.eu/)
2. Email `neuromorphic@humanbrainproject.eu` with your EBRAINS username
3. They create a Collab for you with test quota and examples -- **free of charge**
4. Access via Jupyter notebook at `spinn-20.cs.man.ac.uk` or via the EBRAINS portal
5. Write PyNN scripts, submit via the job queue, results returned to you

**No physical lab access required.** Everything is remote.

**Source:** [EBRAINS Neuromorphic Getting Access](https://wiki.ebrains.eu/bin/view/Collabs/neuromorphic/Getting%20access/)

### SpiNNaker2 Access (Restricted)

- Single-chip boards exist but access is restricted
- `py-spinnaker2` software dependencies are in **private repositories**
- Access requires direct contact with TU Dresden / SpiNNcloud Systems
- sPyNNaker (the SpiNNaker1 software) does NOT run on SpiNNaker2
- No public student access program exists for SpiNNaker2

### Oliver Rhodes as Supervisor

Oliver Rhodes is a Lecturer in Bio-Inspired Computing at Manchester, part of the APT (Advanced Processor Technologies) group that built SpiNNaker. He:
- Co-authored the sPyNNaker paper
- Has supervised 4+ student projects (details not publicly listed)
- Works directly on SpiNNaker software
- Could potentially facilitate access to SpiNNaker hardware beyond EBRAINS quotas
- Published with Brian Ezinwoke on SNN applications (financial prediction)

**Having Oliver Rhodes as supervisor is a significant advantage for SpiNNaker access.**

---

## Question 5: Training vs Inference on SpiNNaker

