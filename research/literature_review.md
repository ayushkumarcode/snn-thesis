# Neuromorphic Computing & Spiking Neural Networks — Literature Review

> Initial research notes compiled from 3 key survey papers to help narrow down a thesis project direction.

---

## Paper 1: Spiking Neural Networks and Their Applications: A Review

**Source:** Yamazaki et al., Brain Sciences, 2022
**Link:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9313413/
**Purpose for us:** Understanding what people are *actually building* with SNNs — helps pick a use case.

### What are SNNs?

- Third generation of neural networks, inspired by biological neurons
- Neurons communicate via discrete **spikes** (binary events) rather than continuous values
- Information is encoded in **spike timing** and **spike frequency**, not just magnitude
- Event-driven: neurons only compute when they receive a spike, making them inherently energy-efficient

### Neuron Models (simplest to most complex)

| Model | What it does | Trade-off |
|-------|-------------|-----------|
| **LIF (Leaky Integrate-and-Fire)** | Accumulates input, leaks over time, fires when threshold is reached | Simple, fast, but biologically limited |
| **Izhikevich** | 2 equations that can reproduce 20+ biological firing patterns | Good balance of realism and speed |
| **AdEx (Adaptive Exponential)** | LIF + exponential spike initiation + adaptation current | Better spike pattern fidelity |
| **Hodgkin-Huxley** | Full ion-channel dynamics (Na+, K+) | Most realistic, most expensive to simulate |

**Takeaway:** LIF is the go-to for most practical projects. Izhikevich if you need more biological realism without the cost of Hodgkin-Huxley.

### How to Encode Data as Spikes

- **Rate coding:** Higher spike frequency = stronger signal. Simple but requires many timesteps.
- **Temporal coding:** Information in precise spike *timing*. More efficient (fewer spikes) but harder to work with.

### Training Methods

