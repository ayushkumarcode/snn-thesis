# Surrogate Gradient Training for SNNs: Comprehensive Research Report

**Research Date:** 2026-02-25
**Purpose:** Evaluate whether direct SNN training via surrogate gradients is practical for an undergraduate thesis
**Verdict:** YES -- surrogate gradient training is practical, well-documented, and the recommended approach for a thesis-level SNN project.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What Are Surrogate Gradients and Why Are They Needed](#2-what-are-surrogate-gradients-and-why-are-they-needed)
3. [Commonly Used Surrogate Gradient Functions](#3-commonly-used-surrogate-gradient-functions)
4. [Does the Choice of Surrogate Function Matter](#4-does-the-choice-of-surrogate-function-matter)
5. [Best Tutorials and Code Examples](#5-best-tutorials-and-code-examples)
6. [Difficulty Compared to Standard ANN Training](#6-difficulty-compared-to-standard-ann-training)
7. [Common Pitfalls and Debugging Challenges](#7-common-pitfalls-and-debugging-challenges)
8. [snnTorch Implementation Guide](#8-snntorch-implementation-guide)
9. [Comparative Studies of Surrogate Gradient Approaches](#9-comparative-studies-of-surrogate-gradient-approaches)
10. [Recent Advances 2024-2025](#10-recent-advances-2024-2025)
11. [Practical Recommendations for the Thesis](#11-practical-recommendations-for-the-thesis)
12. [Sources](#12-sources)

---

## 1. Executive Summary

Surrogate gradient training is the dominant method for directly training Spiking Neural Networks (SNNs) and is highly practical for an undergraduate thesis project. The core idea is simple: spiking neurons use a non-differentiable Heaviside step function to generate spikes, which breaks standard backpropagation. Surrogate gradients solve this by keeping the Heaviside function in the forward pass (preserving actual spiking behavior) while substituting a smooth, differentiable approximation during the backward pass to allow gradient flow.

**Key findings from this investigation:**

- **The choice of surrogate function shape has limited impact on accuracy.** The landmark study by Zenke and Vogels (2021) demonstrated "remarkable robustness" to surrogate shape. What matters more is the **scale/slope parameter** and **activity regularization**.
- **Fast sigmoid is the best default choice** for practical work. It yields similar accuracy to arctangent but produces sparser spiking activity (more energy-efficient) and is computationally cheaper (no `exp` evaluation).
- **snnTorch is the most accessible framework** for an undergraduate. It has 1.9k GitHub stars, extensive Colab-ready tutorials, and is built directly on PyTorch. All tutorials run in Google Colab with free GPU access.
- **Training difficulty is moderately higher than ANNs** but manageable. The main added complexity is the temporal dimension (iterating over time steps) and tuning neuron-specific hyperparameters (beta/decay, threshold, slope). The training loop itself is standard PyTorch.
- **Achievable accuracy on MNIST is 95-98%** with a basic convolutional SNN and minimal tuning, within a few minutes of training on a free Colab GPU. This is entirely within thesis scope.
- **DVS128 Gesture recognition with surrogate gradients achieves 96-97% accuracy** using standard architectures, which is competitive with the state of the art.

---

## 2. What Are Surrogate Gradients and Why Are They Needed

### The Core Problem

A spiking neuron fires a discrete binary spike (0 or 1) when its membrane potential exceeds a threshold. This is modeled by the Heaviside step function:

```
S = H(U - U_threshold) = { 1 if U >= U_threshold, 0 otherwise }
```

The derivative of the Heaviside function is the Dirac delta function, which is zero almost everywhere and infinite at the threshold. This means:

- **Subthreshold region:** Gradient is exactly 0 -- no learning signal propagates
- **At threshold:** Gradient is undefined/infinite -- numerically useless
- Small weight perturbations either produce no change in output spikes or cause large discontinuous changes

This is the "dead neuron problem" of SNNs. Without a workaround, gradient-based training is impossible.

### The Surrogate Gradient Solution

The surrogate gradient method uses a two-pass strategy:

1. **Forward pass:** Use the true Heaviside function. The network produces real discrete spikes. Spiking dynamics are preserved exactly.
2. **Backward pass:** Replace the Heaviside derivative with a smooth, continuous approximation (the "surrogate gradient"). This allows gradients to flow through the spiking nonlinearity.

This approach was formalized by Neftci, Mostafa, and Zenke (2019) in their seminal paper "Surrogate Gradient Learning in Spiking Neural Networks" and has since become the standard method for direct SNN training.

### Why It Works

The surrogate gradient does not need to be an exact gradient of anything. It serves as a "gradient-like" signal that provides a reasonable direction for weight updates. Empirically, this works extremely well across a wide range of tasks, and recent theoretical work by Gygax and Zenke (2025) has begun to explain why: for single neurons, surrogate gradients are equivalent to the gradient of the expected output under a stochastic interpretation, though this equivalence breaks in deep networks where they function as "smoothed stochastic derivatives."

### Reference

> Neftci, E.O., Mostafa, H., and Zenke, F. (2019). Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-based optimization to spiking neural networks. IEEE Signal Processing Magazine, 36, 51-63. https://arxiv.org/abs/1901.09948

---

