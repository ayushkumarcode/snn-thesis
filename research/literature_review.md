# Neuromorphic Computing & Spiking Neural Networks -- Literature Review

Starting notes from reading 3 survey papers to figure out what direction to take for the thesis. Trying to narrow things down.

---

## Paper 1: Spiking Neural Networks and Their Applications: A Review

Yamazaki et al., Brain Sciences, 2022
https://pmc.ncbi.nlm.nih.gov/articles/PMC9313413/

Reading this mostly to understand what people are actually building with SNNs -- helps with picking a use case.

### What are SNNs?

- Third generation of neural networks, inspired by biological neurons
- Neurons communicate via discrete spikes (binary events) rather than continuous values
- Information is encoded in spike timing and spike frequency, not just magnitude
- Event-driven: neurons only compute when they receive a spike, making them inherently energy-efficient

### Neuron Models (simplest to most complex)

| Model | What it does | Trade-off |
|-------|-------------|-----------|
| **LIF (Leaky Integrate-and-Fire)** | Accumulates input, leaks over time, fires when threshold is reached | Simple, fast, but biologically limited |
| **Izhikevich** | 2 equations that can reproduce 20+ biological firing patterns | Good balance of realism and speed |
| **AdEx (Adaptive Exponential)** | LIF + exponential spike initiation + adaptation current | Better spike pattern fidelity |
| **Hodgkin-Huxley** | Full ion-channel dynamics (Na+, K+) | Most realistic, most expensive to simulate |
