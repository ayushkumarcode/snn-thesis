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
- small weight perturbations either do nothing or cause big discontinuous jumps

this is the "dead neuron problem" of SNNs. without a workaround, gradient-based training just doesn't work.

### The Solution

surrogate gradients use a two-pass strategy:
1. **Forward pass:** use true Heaviside. network produces real discrete spikes. spiking dynamics preserved.
2. **Backward pass:** substitute a smooth, differentiable approximation. gradients flow through the spiking nonlinearity.

formalized by Neftci, Mostafa, and Zenke (2019) in "Surrogate Gradient Learning in Spiking Neural Networks." now the standard method.

### Why It Works

the surrogate gradient doesn't need to be an exact gradient of anything. it just needs to provide a reasonable direction for weight updates. empirically works great. recent theoretical work by Gygax and Zenke (2025) explains why: for single neurons, surrogate gradients equal the gradient of expected output under a stochastic interpretation, though this breaks in deep networks where they act as "smoothed stochastic derivatives."

> Neftci, E.O., Mostafa, H., and Zenke, F. (2019). Surrogate Gradient Learning in Spiking Neural Networks. IEEE Signal Processing Magazine, 36, 51-63. https://arxiv.org/abs/1901.09948

---

## Common Surrogate Gradient Functions

| Function | Mathematical Form (Backward) | Default Params | Characteristics |
|----------|------------------------------|----------------|-----------------|
| **Fast Sigmoid** | 1 / (1 + k|U|)^2 | slope=25 | Cheapest compute, good sparsity |
| **Sigmoid** | k * exp(-kU) / (exp(-kU)+1)^2 | slope=25 | Smooth, classic |
| **ATan (Arctangent)** | 1 / (pi * (1 + (pi*U*alpha/2)^2)) | alpha=2.0 | snnTorch default, smooth tails |
| **Triangular** | Piecewise linear | threshold=1 | Simple but weaker gradient strength |
