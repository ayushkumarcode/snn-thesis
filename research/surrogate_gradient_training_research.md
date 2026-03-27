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

## 3. Commonly Used Surrogate Gradient Functions

### Overview Table

| Function | Mathematical Form (Backward) | Default Params | Characteristics |
|----------|------------------------------|----------------|-----------------|
| **Fast Sigmoid** | 1 / (1 + k\|U\|)^2 | slope=25 | Computationally cheapest, good sparsity |
| **Sigmoid** | k * exp(-kU) / (exp(-kU)+1)^2 | slope=25 | Smooth, classic choice |
| **ATan (Arctangent)** | 1 / (pi * (1 + (pi*U*alpha/2)^2)) | alpha=2.0 | Default in snnTorch, smooth tails |
| **Triangular** | Piecewise linear | threshold=1 | Simple, but weaker gradient strength |
| **Straight Through Estimator** | 1 (constant) | none | Simplest possible, coarse approximation |
| **Spike Rate Escape** | k * exp(-beta\|U-1\|) | beta=1, slope=25 | Biologically inspired escape rate |

### Detailed Descriptions

**Fast Sigmoid (Recommended Default)**
- Uses the derivative of a "fast sigmoid" function as the surrogate
- Computationally efficient because it avoids `exp()` evaluation
- Produces lower firing rates (higher sparsity) compared to arctangent at similar accuracy levels
- The `slope` parameter controls sharpness: higher slope = closer to true Heaviside derivative but potentially less stable

**Sigmoid**
- Classic logistic sigmoid derivative
- Smooth and well-behaved
- Slightly more expensive than fast sigmoid due to exponential computation
- Good default for those familiar with sigmoid from standard deep learning

**ATan (Arctangent)**
- Default surrogate in snnTorch
- Smooth tails that provide gradients over a wider range of membrane potentials
- Slightly higher firing rates compared to fast sigmoid
- Well-studied theoretically

**Triangular**
- Piecewise linear function centered at threshold
- Simplest geometric interpretation
- Research suggests it "struggles significantly, performing poorly across datasets" due to weak gradient strength and learning instability
- Not recommended as primary choice

**Straight Through Estimator (STE)**
- Simplest possible approximation: gradient is always 1
- Very coarse but sometimes effective for simple tasks
- No tunable parameters
- Can be useful as a baseline comparison

### Visual Intuition

All surrogate gradients share the same basic shape: a bell-shaped or peaked curve centered at the firing threshold. The key differences are:
- **Width:** How far from threshold the gradient signal extends (controlled by slope/alpha)
- **Height:** Peak magnitude of the gradient
- **Tails:** How quickly the gradient decays away from threshold

---

## 4. Does the Choice of Surrogate Function Matter?

### Short Answer: Shape matters little; scale matters a lot.

### The Landmark Study

Zenke and Vogels (2021), "The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks," systematically investigated this question. Their key findings:

1. **Surrogate gradient learning is robust to different shapes of underlying surrogate derivatives.** Whether you use sigmoid, fast sigmoid, arctangent, or other smooth approximations, the network learns effectively.

2. **The scale (width/slope) of the surrogate derivative substantially affects learning performance.** Too narrow and gradients vanish (neurons far from threshold receive no learning signal). Too wide and gradients become noisy (updates are imprecise, potentially harmful).

3. **Activity regularization is critical.** When combined with appropriate firing rate regularization, spiking networks perform robust information processing even at the sparse activity limit.

### Practical Implications

| What Matters | What Does Not Matter Much |
|-------------|--------------------------|
| Slope/scale parameter of the surrogate | Exact mathematical form (sigmoid vs. atan vs. fast sigmoid) |
| Activity/firing rate regularization | Whether the surrogate is symmetric or not |
| Beta (membrane decay constant) | Minor differences in tail behavior |
| Membrane threshold setting | |
| Number of time steps | |

### The Fine-Tuning Study (2024)

A systematic study on fine-tuning surrogate gradient learning for hardware performance confirmed:
- Fast sigmoid consistently delivers the best results across datasets due to its sharp gradient slopes
- Fast sigmoid yields lower firing activity (higher sparsity) compared to arctangent at similar accuracy
- By cross-sweeping beta and threshold hyperparameters, practitioners achieved a 48% reduction in hardware inference latency with only 2.88% accuracy trade-off
- The optimal balance was found at beta=0.5 and threshold=1.5

### Recommendation for the Thesis

Start with **fast sigmoid (slope=25)** and **beta=0.5**. These are well-tested defaults. Only explore other surrogate functions if you specifically want to compare them as part of your thesis contribution.

---

## 5. Best Tutorials and Code Examples

### Tier 1: Essential (Start Here)

**1. snnTorch Tutorial Series (Google Colab Notebooks)**
- Tutorial 5: Training SNNs with snnTorch -- fully-connected SNN on MNIST
  - https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html
- Tutorial 6: Surrogate Gradient Descent in a Convolutional SNN -- ConvSNN on MNIST
  - https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html
  - Colab: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_6_CNN.ipynb
- Tutorial 7: Neuromorphic Datasets with Tonic + snnTorch
  - https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html

All tutorials are interactive Jupyter notebooks that run in Google Colab with free GPU. They include complete code, explanations, and visualizations.

**2. SpyTorch Tutorials (Friedemann Zenke)**
- GitHub: https://github.com/fzenke/spytorch
- Introductory video: https://youtu.be/xPYiAjceAqU
- Shows how to implement surrogate gradients from scratch in pure PyTorch
- Created by the co-author of the original surrogate gradient paper
- Four tutorial notebooks covering LIF neurons, surrogate gradients, and training

### Tier 2: Deeper Understanding

**3. "Building and Training SNNs From Scratch" (R Gaurav, January 2024)**
- https://r-gaurav.github.io/2024/01/04/Building-And-Training-Spiking-Neural-Networks-From-Scratch.html
- Builds everything from scratch in PyTorch (no snnTorch dependency)
- Excellent for understanding what happens under the hood
- Achieves >97% on MNIST within 10 epochs with 25 time steps
- Designed for absolute beginners with basic-to-intermediate PyTorch knowledge

**4. Eshraghian et al. "Training Spiking Neural Networks Using Lessons From Deep Learning" (2023)**
- https://arxiv.org/abs/2109.12894
- Published in Proceedings of the IEEE
- Comprehensive tutorial and review (39 pages)
- Covers encoding, learning challenges, spike timing, and online learning
- The paper behind snnTorch

### Tier 3: Reference Material

**5. snnTorch Documentation**
- Full API: https://snntorch.readthedocs.io/en/latest/
- Surrogate functions: https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html
- Loss functions: https://snntorch.readthedocs.io/en/latest/snntorch.functional.html

**6. "A Practical Tutorial on Spiking Neural Networks" (MDPI, 2025)**
- https://www.mdpi.com/2673-4117/6/11/304
- Comprehensive review with standardized experiments across neuron models
- Benchmarks on MNIST and CIFAR-10 with multiple frameworks

**7. Neftci, Mostafa, and Zenke (2019) -- The Original Paper**
- https://arxiv.org/abs/1901.09948
- Essential reading for the literature review section of the thesis

---

## 6. Difficulty Compared to Standard ANN Training

### What Stays the Same

| Aspect | SNN with Surrogate Gradients | Standard ANN |
|--------|------------------------------|--------------|
| Framework | PyTorch (via snnTorch) | PyTorch |
| Optimizer | Adam, SGD, etc. | Adam, SGD, etc. |
| Loss function | Cross-entropy (adapted) | Cross-entropy |
| Data loading | torch.utils.data.DataLoader | torch.utils.data.DataLoader |
| GPU acceleration | CUDA via PyTorch | CUDA via PyTorch |
| Gradient computation | loss.backward() | loss.backward() |
| Weight update | optimizer.step() | optimizer.step() |

### What Changes (Added Complexity)

| New Aspect | Description | Difficulty Level |
|-----------|-------------|-----------------|
| **Time dimension** | Must iterate over T time steps per input, unrolling the network temporally | Low-Medium |
| **State management** | Neurons maintain membrane potential across time steps; must reset between samples | Low (snnTorch handles this) |
| **Surrogate gradient selection** | Choose and configure surrogate function | Low (use defaults) |
| **Neuron hyperparameters** | Beta (decay), threshold, reset mechanism | Medium |
| **Loss function adaptation** | Use spike-count or rate-based loss instead of direct output | Low (snnTorch provides these) |
| **Memory overhead** | BPTT stores activations for all T time steps | Medium (limits batch size) |
| **Training time** | Roughly T times slower than equivalent ANN per epoch | Medium |
| **Debugging** | Cannot directly visualize "activations" -- must monitor spike rates and membrane potentials | Medium |

### Concrete Training Loop Comparison

**Standard ANN:**
```python
for data, targets in train_loader:
    output = model(data)
    loss = criterion(output, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**SNN with Surrogate Gradients (snnTorch):**
```python
for data, targets in train_loader:
    utils.reset(net)  # Reset neuron states
    spk_rec = []
    for step in range(num_steps):  # Iterate over time
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
    spk_rec = torch.stack(spk_rec)
    loss = loss_fn(spk_rec, targets)  # Spike-count based loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The difference is essentially: (1) a time loop, (2) state reset, and (3) stacking spike outputs. The rest is identical PyTorch.

### Honest Assessment of Difficulty

For someone who has trained a CNN in PyTorch:
- **Getting a basic SNN running:** 1-2 days following tutorials
- **Understanding what is happening:** 3-5 days of reading and experimentation
- **Tuning for good performance:** 1-2 weeks of experimentation
- **Understanding the theory well enough to write about it:** 1-2 weeks of reading

This is very manageable for a semester-long thesis project.

---

## 7. Common Pitfalls and Debugging Challenges

### Pitfall 1: Dead Neurons (Most Common)

**Problem:** Neurons never fire, or fire at every time step. Both prevent learning.

**Cause:** Membrane potential never reaches threshold (too much leak / too high threshold) or always exceeds threshold (too little leak / too low threshold).

**Solution:**
- Start with beta=0.5 (moderate decay) and threshold=1.0
- Monitor average firing rates per layer during training
- Add firing rate regularization loss to encourage moderate activity
- Use `snn.Leaky(beta=0.5, threshold=1.0)` as starting point

### Pitfall 2: Gradient Vanishing/Exploding

**Problem:** Gradients become extremely small or large during BPTT through many time steps.

**Cause:**
- Too many time steps combined with the surrogate gradient approximation
- The gradient explosion/vanishing problem in SNNs is more severe than in ANNs because of tanh-like surrogate function behavior
- Narrow surrogate gradient (high slope) causes gradient vanishing; wide surrogate (low slope) causes noisy, potentially harmful updates

**Solution:**
- Use fewer time steps (25-50 is usually sufficient for static images)
- Use gradient clipping: `torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)`
- Use moderate slope values (25 for fast_sigmoid is a good default)
- Consider batch normalization adapted for SNNs (tdBN or BNTT)

### Pitfall 3: Surrogate Gradient Scale Mismatch

**Problem:** Learning stalls or oscillates despite gradients flowing.

**Cause:** The slope parameter of the surrogate gradient is inappropriately scaled. Research shows that while shape does not matter much, scale has a major impact on learning performance.

**Solution:**
- Start with default slope values (25 for fast_sigmoid, 2.0 for atan)
- If learning stalls, try reducing slope (wider gradient, more neurons receive signal)
- If learning is noisy/unstable, try increasing slope (narrower, more precise gradient)

### Pitfall 4: Memory Overflow from BPTT

**Problem:** Out-of-memory errors, especially with many time steps or large batch sizes.

**Cause:** BPTT must store activations for all T time steps, multiplying memory requirements by T compared to a feedforward ANN.

**Solution:**
- Reduce batch size
- Reduce number of time steps
- Use `torch.cuda.amp` for mixed-precision training
- For very long sequences, consider truncated BPTT or online learning variants

### Pitfall 5: Loss of Sparsity

**Problem:** The trained SNN fires at very high rates, defeating the purpose of energy-efficient spiking computation.

**Cause:** Without explicit regularization, the surrogate gradient method can push the network toward high-firing-rate solutions that resemble continuous-valued ANNs.

**Solution:**
- Add spike rate regularization to the loss function
- Monitor average firing rates and penalize rates above a target (e.g., 10-20% of maximum)
- Use the homeostatic regularization approach from Zenke and Vogels (2021)
- Example in snnTorch: add `l1_rate = torch.mean(torch.abs(spk_rec))` to loss

### Pitfall 6: Forgetting to Reset States

**Problem:** Erratic training behavior, accuracy fluctuates wildly.

**Cause:** Membrane potentials from the previous sample leak into the current sample.

**Solution:**
- Always call `utils.reset(net)` before processing each new input sample
- With `init_hidden=True` in snnTorch, states are managed internally but still need reset

### Pitfall 7: Incorrect Time Step Configuration

**Problem:** Very poor accuracy despite correct architecture.

**Cause:** Too few time steps for the network to accumulate meaningful spike patterns, or too many time steps causing temporal dilution.

**Solution:**
- For static images (MNIST, CIFAR): 25-50 time steps is usually sufficient
- For neuromorphic data (DVS): match time steps to the temporal structure of the data
- Research shows accuracy often peaks at moderate time steps (T=4-8 for some tasks) and can decrease with excessive time steps

### Debugging Checklist

```
[ ] Are neurons firing? Check average spike rates per layer
[ ] Is the loss decreasing? Plot training loss over iterations
[ ] Are gradients flowing? Check gradient norms per layer
[ ] Are states being reset? Verify utils.reset() is called
[ ] Is the surrogate gradient slope reasonable? Try defaults first
[ ] Is beta/decay appropriate? Start with 0.5
[ ] Is the number of time steps reasonable? Try 25 for static images
[ ] Is batch size causing OOM? Reduce if needed
[ ] Is learning rate appropriate? 1e-3 to 1e-2 for Adam is typical
[ ] Are spike counts being used correctly in loss function?
```

---

## 8. snnTorch Implementation Guide

### Installation

```bash
pip install snntorch
```

That is it. snnTorch depends only on PyTorch and installs cleanly via pip.

### Minimal Working Example (MNIST Classification)

```python
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torchvision

# --- Hyperparameters ---
num_steps = 25        # Number of time steps
beta = 0.5            # Membrane decay factor
batch_size = 128
num_epochs = 1
lr = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Surrogate gradient ---
spike_grad = surrogate.fast_sigmoid(slope=25)

# --- Network ---
net = nn.Sequential(
    nn.Conv2d(1, 12, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Conv2d(12, 64, 5),
    nn.MaxPool2d(2),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
    nn.Flatten(),
    nn.Linear(64 * 4 * 4, 10),
    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
).to(device)

# --- Data ---
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((28, 28)),
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (1,)),
])
train_ds = torchvision.datasets.MNIST(root="./data", train=True,
                                       download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,
                                            shuffle=True, drop_last=True)

# --- Loss and optimizer ---
loss_fn = SF.ce_rate_loss()           # Cross-entropy on spike counts
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# --- Forward pass ---
def forward_pass(net, data, num_steps):
    spk_rec = []
    mem_rec = []
    utils.reset(net)                   # Reset neuron states

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)

