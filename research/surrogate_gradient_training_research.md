# Surrogate Gradient Training for SNNs

looked into whether direct SNN training via surrogate gradients is actually practical for an undergrad thesis. short answer: yes, it's practical, well-documented, and the way to go.

key things i found:

- **choice of surrogate function shape barely matters.** Zenke and Vogels (2021) showed "remarkable robustness" to shape. what matters more is the **scale/slope parameter** and **activity regularization**.
- **fast sigmoid is the best default.** similar accuracy to arctangent but sparser spiking (more energy-efficient) and computationally cheaper (no `exp`).
- **snnTorch is the most accessible framework** for undergrads. 1.9k GitHub stars, Colab-ready tutorials, built on PyTorch.
- **training is moderately harder than ANNs** but manageable. main added complexity is the time loop and tuning neuron-specific hyperparams.
- **achievable accuracy on MNIST: 95-98%** with a basic conv SNN and minimal tuning, in minutes on free Colab GPU.
- **DVS128 Gesture with surrogate gradients gets 96-97%**, competitive with SOTA.

---

## What Are Surrogate Gradients and Why Do We Need Them

### The Problem

spiking neurons fire discrete binary spikes (0 or 1) when membrane potential exceeds threshold. modeled by the Heaviside step function:

```
S = H(U - U_threshold) = { 1 if U >= U_threshold, 0 otherwise }
```

derivative of Heaviside is the Dirac delta -- zero almost everywhere, infinite at threshold. so:
- **subthreshold:** gradient is exactly 0, no learning signal propagates
- **at threshold:** gradient is undefined/infinite, numerically useless
