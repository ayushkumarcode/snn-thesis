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

### The Standard Workflow

```
[Train on GPU/CPU]          [Deploy on SpiNNaker]
snnTorch + PyTorch    --->   sPyNNaker (SpiNNaker1)
  |                          or py-spinnaker2 (SpiNNaker2)
  |-- surrogate gradients     |-- inference only
  |-- backpropagation          |-- pre-loaded weights
  |-- GPU acceleration         |-- real-time execution
  |-- batch training           |-- energy measurement
  v                            v
Trained weights (.pt)    Spike output + energy data
```

**SpiNNaker is primarily used for inference**, not training. The training happens on conventional hardware.

### Exception: On-Chip Learning

- **STDP:** SpiNNaker1 supports native STDP (unsupervised learning rule)
- **E-prop:** Demonstrated on both SpiNNaker1 and SpiNNaker2 (surrogate gradient variant)
- **Deep Rewiring (DEEP R):** Demonstrated on SpiNNaker2 prototype for on-chip training

But for your thesis, the workflow is:
1. Train in snnTorch on your laptop/GPU
2. Export weights (via NIR or manual extraction)
3. Load weights into SpiNNaker
4. Run inference on SpiNNaker
5. Measure real energy and latency
6. Compare with simulated NeuroBench estimates

### Is This Workflow Documented?

Yes. Two key papers document it end-to-end:
1. **DVS Gesture on SpiNNaker2 (2025):** [arxiv.org/html/2504.06748v1](https://arxiv.org/html/2504.06748v1)
2. **Spiking Q-Networks on SpiNNaker2 (2025):** [arxiv.org/html/2507.23562v1](https://arxiv.org/html/2507.23562v1)

---

## Question 6: Project-Specific Compatibility

### ESC-50 (Audio Classification)

| Aspect | SpiNNaker1 | SpiNNaker2 |
|--------|-----------|-----------|
| Conv layers for mel-spectrogram | Possible via KernelConnector but non-trivial | YES (via NIR, Conv2D supported) |
| Mel-spectrogram as input | Convert to spike trains first | Convert to spike trains first |
| Fully-connected alternative | YES (easier) | YES |
| Precedent | YES -- audio classification on SpiNNaker exists | Limited precedent |

**Audio on SpiNNaker has been demonstrated.** Dominguez-Morales et al. built a [multilayer SNN for audio classification on SpiNNaker](https://github.com/jpdominguez/Multilayer-SNN-for-audio-samples-classification-using-SpiNNaker) using LIF neurons and rate-based training. They achieved >85% accuracy on tone classification with noise robustness.

**For ESC-50 specifically:** You would likely need to:
1. Train your convolutional SNN in snnTorch as planned
2. For SpiNNaker1: possibly simplify to a fully-connected architecture (easier to port)
3. For SpiNNaker2: the full conv architecture could transfer via NIR

### ECG PTB-XL (1D Time Series)

| Aspect | Feasibility |
|--------|-------------|
| 1D convolutions | No direct 1D conv support in sPyNNaker -- would need to flatten to 2D or use FC layers |
| Time-series input | Convert to spike trains (rate or latency coding) |
| Real-time processing | SpiNNaker runs at biological real-time, suitable for ECG |
| Precedent | No ECG-specific SpiNNaker papers found |

**Assessment:** ECG on SpiNNaker is feasible but has no direct precedent. You would need to convert 1D convolutions to equivalent connectivity patterns or use fully-connected layers on SpiNNaker.

### Robot Reflexes (RL Policy Inference)

| Aspect | Feasibility |
|--------|-------------|
| Real-time inference | YES -- SpiNNaker's core strength |
| RL policy execution | YES -- Spiking Q-Networks demonstrated on SpiNNaker2 |
| Energy efficiency | 24-32x reduction vs GPU demonstrated |
| Precedent | Strong -- multiple RL + SpiNNaker papers |

**This is the BEST fit for SpiNNaker deployment.** The Spiking Q-Network paper demonstrated:
- CartPole and Acrobot environments
- snnTorch training -> SpiNNaker2 deployment
- 0.006 J per inference (CartPole), 0.333W average power
- 24-32x energy reduction vs GTX 1650

**Source:** [Hardware-Aware Fine-Tuning of Spiking Q-Networks on SpiNNaker2](https://arxiv.org/html/2507.23562v1)

---

## Question 7: Hybrid Workflow

### The Standard Approach (Confirmed)

The standard approach IS:
1. **Develop and train** in snnTorch/PyTorch on a regular computer
2. **Export** trained weights (via NIR or manual extraction)
3. **Deploy** on SpiNNaker for inference demonstration
4. **Measure** real hardware metrics (energy, latency)
5. **Compare** with software-simulated estimates

This is exactly what recent papers demonstrate. There is no common alternative workflow for classification/RL tasks.

### Alternative Approaches (Less Common)

- **Train directly on SpiNNaker with STDP:** Possible but limited to unsupervised learning
- **Co-simulation:** Run part on SpiNNaker, part on conventional hardware (exists for robotics)
- **ANN-to-SNN conversion then deploy:** Train ANN, convert to SNN (SNN Toolbox), then deploy

---

## Question 8: SpiNNaker2

### Key Differences from SpiNNaker1

| Feature | SpiNNaker1 | SpiNNaker2 |
|---------|-----------|-----------|
| Process node | 130nm CMOS | 22nm FDSOI |
| Cores per chip | 18 ARM968 | 153 ARM Cortex-M4F |
| On-chip memory | 128KB DTCM + 32KB ITCM per core | 128KB per PE + shared SRAM |
| External memory | 128MB SDRAM per chip | 2GB LPDDR4 per chip |
| DRAM | SDRAM | LPDDR4 |
| Total SRAM | ~2.3MB per chip | 19MB per chip |
| ML accelerators | None | MAC array, exp/log, RNG |
| Floating point | No | Yes (Cortex-M4F) |
| DVFS | No | Yes (down to 0.5V) |
| Power efficiency | Baseline | 10x improvement per watt |
| Software | sPyNNaker (mature, stable) | py-spinnaker2 (early, restricted) |
| PyNN support | Full (sPyNNaker) | PyNN-like API (less stable) |
| NIR support | NO | YES |
| Public access | YES (EBRAINS) | NO (restricted) |

### Is SpiNNaker2 Available to Students?

**Not generally.** Current status:
- Single-chip development boards exist
- Remote access can potentially be arranged by contacting TU Dresden
- The `py-spinnaker2` library has private dependencies
- Server boards (48 chips) are in commissioning
- sPyNNaker does NOT run on SpiNNaker2

**For your thesis: Plan for SpiNNaker1 via EBRAINS. If Oliver Rhodes can arrange SpiNNaker2 access, that would be a bonus.**

---

## Question 9: Previous Undergraduate Projects (Tyler Gaffey, Brian Ezinwoke)

### Brian Ezinwoke

- Co-authored a paper with Oliver Rhodes: "Predicting Price Movements in High-Frequency Financial Data with Spiking Neural Networks" (arXiv:2512.05868, December 2025)
- The paper used SNNs trained with STDP for financial time-series prediction
- **SpiNNaker usage: NOT confirmed** in the paper. The paper does not specify whether SpiNNaker hardware was used or if it was software-only simulation
- **Tools used:** Custom SNN with STDP (framework not specified), Bayesian Optimization for hyperparameter tuning

### Tyler Gaffey

- LinkedIn profile lists "Deep Neural Network Research" at Manchester (UK)
- No publicly accessible thesis or paper found
- No confirmed SpiNNaker usage

### What This Tells Us

- Manchester undergraduate SNN theses are **not publicly accessible** (behind `studentnet.cs.manchester.ac.uk` authentication)
- Brian Ezinwoke's work with Oliver Rhodes focused on financial SNN applications, possibly software-only
- It is likely that these students used **snnTorch or similar frameworks** rather than SpiNNaker hardware, given that:
  - SpiNNaker deployment adds significant complexity
  - Software-only SNN projects are the norm for undergraduate theses
  - The papers found do not mention SpiNNaker

**Recommendation:** Ask Oliver Rhodes directly about previous student SpiNNaker usage.

---

## Question 10: Time Cost of Adding SpiNNaker

### Estimated Time by Pathway

| Pathway | Time Estimate | Difficulty |
|---------|--------------|------------|
| SpiNNaker1 via EBRAINS (FC network, manual weight transfer) | 1-2 weeks | Moderate |
| SpiNNaker1 via EBRAINS (conv network, KernelConnector) | 2-4 weeks | High |
| SpiNNaker2 via NIR (if access is arranged) | 1-2 weeks | Moderate (tooling is newer) |
| SpiNNaker2 via NIR (with quantization) | 2-3 weeks | Moderate-High |

### Breakdown of Activities

1. **Account setup and learning sPyNNaker/PyNN:** 2-3 days
2. **Running tutorial examples on EBRAINS:** 1-2 days
3. **Converting trained weights to PyNN format:** 2-5 days (depending on architecture complexity)
4. **Debugging deployment issues:** 2-5 days (fixed-point conversion issues, spike loss, timing)
5. **Running inference experiments:** 1-2 days
6. **Collecting and analyzing energy/latency data:** 1-2 days
7. **Writing up results for thesis:** 2-3 days

### Risk Factors

- **Weight precision loss:** snnTorch uses 32-bit float; SpiNNaker1 uses 16-bit fixed-point. Accuracy degradation is common.
- **Architecture mismatch:** Some snnTorch layer types may not have sPyNNaker equivalents
- **EBRAINS queue times:** Shared resource, jobs may queue
- **Debugging remotely:** Limited visibility into on-chip execution
- **Documentation gaps:** sPyNNaker documentation is adequate but not comprehensive

---

## Question 11: What SpiNNaker Adds to the Thesis

### Measurable Metrics from Real Hardware

If you successfully deploy on SpiNNaker, you can report:

| Metric | How Measured | Value to Thesis |
|--------|-------------|----------------|
| **Real energy per inference (Joules)** | SpiNNaker power monitoring | HIGH -- real hardware measurement vs simulated estimate |
| **Real inference latency (ms)** | Wall-clock time on SpiNNaker | HIGH -- biological real-time demonstration |
| **Power draw (Watts)** | Hardware power measurement | MODERATE -- contextualizes energy |
| **Accuracy on hardware** | Compare with software accuracy | HIGH -- quantifies precision loss |
| **Throughput (inferences/sec)** | Batch inference timing | MODERATE |

### Comparison with Simulated NeuroBench Estimates

| Approach | What You Get | Limitations |
|----------|-------------|-------------|
| **NeuroBench (simulated)** | SynOps/J estimate, theoretical energy | Model-based, not real measurement |
| **SpiNNaker (real hardware)** | Actual energy, actual latency | Platform-specific, not generalizable |
| **Both (ideal)** | Validate NeuroBench estimates against real hardware | Best for thesis quality |

### Thesis Impact

Having a SpiNNaker deployment chapter would:
1. **Demonstrate practical neuromorphic deployment** (not just simulation)
2. **Validate energy efficiency claims** with real measurements
3. **Show cross-platform portability** (snnTorch -> NIR -> SpiNNaker)
4. **Differentiate from pure-software theses** (most undergrad SNN theses are software-only)
5. **Align with supervisor expertise** (Oliver Rhodes is a SpiNNaker developer)

The SpiNNaker2 RL paper reported: **SpiNNaker2 consumed 0.006J vs GPU's 0.19J for CartPole inference -- a 32x reduction.** Being able to report similar findings would significantly strengthen the thesis.

---

## Question 12: sPyNNaker Documentation and Tutorials

### Available Resources

| Resource | URL | Quality |
|----------|-----|---------|
| sPyNNaker GitHub | [github.com/SpiNNakerManchester/sPyNNaker](https://github.com/SpiNNakerManchester/sPyNNaker) | Active, well-maintained |
| PyNN Examples | [github.com/SpiNNakerManchester/PyNNExamples](https://github.com/SpiNNakerManchester/PyNNExamples) | Good variety of examples |
| sPyNNaker ReadTheDocs | [spynnaker.readthedocs.io](https://spynnaker.readthedocs.io/) | API reference, adequate |
| SpiNNaker Tutorial | [spinnaker-tutorial.readthedocs.io](https://spinnaker-tutorial.readthedocs.io/) | Setup and basic usage |
| Models & Limitations | [SPyNNakerModelsAndLimitations](http://spinnakermanchester.github.io/spynnaker/6.0.0/SPyNNakerModelsAndLimitations.html) | Critical reference for constraints |
| EBRAINS Guidebook | [electronicvisions.github.io/hbp-sp9-guidebook](https://electronicvisions.github.io/hbp-sp9-guidebook/mc/using_spiNNaker.html) | Platform-specific guide |
| Lab Manuals (PDF) | Various workshop PDFs | Hands-on exercises |
| sPyNNaker Paper | [Frontiers 2018](https://www.frontiersin.org/articles/10.3389/fnins.2018.00816/full) | Definitive reference |
| SpiNNaker Users Group | [Google Groups](https://groups.google.com/g/spinnakerusers) | Community support |
| NIR Documentation | [neuroir.org/docs](https://neuroir.org/docs/) | Cross-platform conversion |
| snnTorch export_nir | [snntorch.readthedocs.io/en/latest/snntorch.export_nir.html](https://snntorch.readthedocs.io/en/latest/snntorch.export_nir.html) | snnTorch-specific NIR export |

### Documentation Quality Assessment

- **Strengths:** Well-structured API docs, active GitHub, research papers provide depth, community Google Group for questions
- **Weaknesses:** Some documentation is for older versions (4.0.0, 5.0.0), limited "cookbook" style tutorials for common tasks, EBRAINS portal documentation can be confusing, py-spinnaker2 docs are sparse

### Quickstart Path

1. Register for EBRAINS account
2. Read the [Lab Manual PDF](https://spinnakermanchester.github.io/spynnaker/4.0.0/RunningPyNNSimulationsonSpiNNaker-LabManual.pdf)
3. Run the synfire chain example (code shown in Question 1 above)
4. Explore [PyNNExamples](https://github.com/SpiNNakerManchester/PyNNExamples) repository
5. Study the [Models & Limitations page](http://spinnakermanchester.github.io/spynnaker/6.0.0/SPyNNakerModelsAndLimitations.html)

---

## Recommended Strategy for Your Thesis

### Tier 1: Core Project (No SpiNNaker Required)

- Train SNNs in snnTorch for your chosen application(s)
- Use NeuroBench for simulated energy estimates
- Compare SNN vs ANN accuracy and efficiency
- This alone makes a complete, publishable thesis

### Tier 2: SpiNNaker1 via EBRAINS (2-3 weeks extra)

- Register for EBRAINS account early (do this NOW)
- Build a simplified FC version of your best model
- Port trained weights manually to sPyNNaker
- Run inference on SpiNNaker1
- Report real energy/latency measurements
- Compare with NeuroBench estimates

### Tier 3: SpiNNaker2 via Oliver Rhodes (1-2 weeks if access works)

- Ask Oliver Rhodes about SpiNNaker2 access
- Use NIR export from snnTorch
- Deploy via py-spinnaker2
- Report energy measurements (potentially 24-32x vs GPU)
- This would be the strongest possible thesis addition

### Timeline Recommendation

```
Month 1-6: Core SNN development in snnTorch (Tier 1)
Month 4: Register for EBRAINS account, start sPyNNaker tutorials
Month 5: Ask Oliver Rhodes about SpiNNaker2 access
Month 6-7: Attempt SpiNNaker deployment (Tier 2 or 3)
Month 7: Write up SpiNNaker results (or pivot to software-only if deployment fails)
```

### Go/No-Go Decision Points

- **Go for SpiNNaker** if: EBRAINS account works, tutorial examples run, weight conversion succeeds in initial tests
- **No-go** if: EBRAINS queue times are excessive, weight conversion causes >10% accuracy drop, or time is running short
