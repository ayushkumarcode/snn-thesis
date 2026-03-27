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
| **STE** | 1 (constant) | none | Simplest possible, coarse |
| **Spike Rate Escape** | k * exp(-beta|U-1|) | beta=1, slope=25 | Biologically inspired escape rate |

### Details

**Fast Sigmoid (my recommendation):**
efficient because no `exp()`. produces lower firing rates (higher sparsity) than atan at similar accuracy. slope controls sharpness: higher = closer to true Heaviside but potentially less stable.

**ATan (Arctangent):**
default in snnTorch. smooth tails give gradients over wider membrane potential range. slightly higher firing rates vs fast sigmoid. well-studied theoretically.

**Triangular:**
piecewise linear, simplest geometrically. research says it "struggles significantly, performing poorly across datasets" due to weak gradient strength. would avoid as primary choice.

**STE:**
simplest possible -- gradient is always 1. coarse but sometimes works for simple tasks. no tunable params.

all surrogates share the same basic shape: bell/peak centered at firing threshold. key differences are width (how far gradient extends), height (peak magnitude), and tail behavior (decay rate).

---

## Does the Choice of Surrogate Actually Matter?

### short answer: shape barely matters; scale matters a lot.

### The Landmark Study

Zenke and Vogels (2021), "The Remarkable Robustness of Surrogate Gradient Learning":

1. **learning is robust to different surrogate shapes.** sigmoid, fast sigmoid, atan -- all work.
2. **the scale (width/slope) substantially affects performance.** too narrow = gradients vanish (neurons far from threshold get no signal). too wide = noisy/imprecise updates.
3. **activity regularization is critical.** with proper firing rate regularization, networks work even at very sparse activity.

### What Matters vs What Doesn't

| What Matters | What Doesn't Matter Much |
|-------------|--------------------------|
| Slope/scale parameter | Exact mathematical form |
| Activity/firing rate regularization | Whether surrogate is symmetric or not |
| Beta (membrane decay) | Minor tail behavior differences |
| Threshold setting | |
| Number of time steps | |

### The Fine-Tuning Study (2024)

confirmed that fast sigmoid consistently delivers best results across datasets because of sharp gradient slopes. fast sigmoid also yields lower firing rates (higher sparsity) than atan at similar accuracy. by cross-sweeping beta and threshold, you can get 48% reduction in hardware inference latency with only 2.88% accuracy trade-off. optimal: beta=0.5, threshold=1.5.

### My Recommendation

start with **fast sigmoid (slope=25)** and **beta=0.5**. these are well-tested. only explore other surrogates if comparing them is part of your thesis contribution.

---

## Best Tutorials and Code

### Tier 1: Start Here

**1. snnTorch Tutorials (Colab Notebooks)**
- Tutorial 5: FC SNN on MNIST -- https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html
- Tutorial 6: ConvSNN on MNIST -- https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html
  - Colab: https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_6_CNN.ipynb
- Tutorial 7: Neuromorphic Datasets -- https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html

all interactive, all Colab-ready with free GPU.

**2. SpyTorch (Friedemann Zenke)**
- GitHub: https://github.com/fzenke/spytorch
- Video: https://youtu.be/xPYiAjceAqU
- implements surrogate gradients from scratch in pure PyTorch. by the co-author of the original paper.

### Tier 2: Deeper Understanding

**3. "Building and Training SNNs From Scratch" (R Gaurav, Jan 2024)**
- https://r-gaurav.github.io/2024/01/04/Building-And-Training-Spiking-Neural-Networks-From-Scratch.html
- builds everything from scratch in PyTorch. great for understanding what's happening under the hood. >97% on MNIST within 10 epochs with T=25.

**4. Eshraghian et al. "Training SNNs Using Lessons From Deep Learning" (2023)**
- https://arxiv.org/abs/2109.12894
- Proceedings of the IEEE, 39 pages. the paper behind snnTorch.

### Tier 3: Reference

- snnTorch API: https://snntorch.readthedocs.io/en/latest/
- Surrogate functions: https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html
- Loss functions: https://snntorch.readthedocs.io/en/latest/snntorch.functional.html
- "A Practical Tutorial on SNNs" (MDPI, 2025): https://www.mdpi.com/2673-4117/6/11/304
- Neftci, Mostafa, Zenke (2019) -- the original: https://arxiv.org/abs/1901.09948

---

## Difficulty vs Standard ANN Training

### What Stays the Same

| Aspect | SNN with Surrogate Gradients | Standard ANN |
|--------|------------------------------|--------------|
| Framework | PyTorch (via snnTorch) | PyTorch |
| Optimizer | Adam, SGD, etc. | Adam, SGD, etc. |
| Loss function | Cross-entropy (adapted) | Cross-entropy |
| Data loading | DataLoader | DataLoader |
| GPU acceleration | CUDA via PyTorch | CUDA via PyTorch |
| Gradient computation | loss.backward() | loss.backward() |
| Weight update | optimizer.step() | optimizer.step() |

### What's New

| New Aspect | Description | Difficulty |
|-----------|-------------|-----------|
| **Time dimension** | iterate over T timesteps per input | Low-Medium |
| **State management** | neurons maintain membrane potential; must reset between samples | Low (snnTorch handles it) |
| **Surrogate gradient selection** | choose and configure surrogate | Low (use defaults) |
| **Neuron hyperparams** | beta (decay), threshold, reset mechanism | Medium |
| **Loss adaptation** | spike-count or rate-based loss | Low (snnTorch provides these) |
| **Memory overhead** | BPTT stores activations for all T steps | Medium (limits batch size) |
| **Training time** | ~T times slower per epoch | Medium |
| **Debugging** | can't just look at "activations" -- need to monitor spike rates and membrane potentials | Medium |

### Training Loop Comparison

**ANN:**
```python
for data, targets in train_loader:
    output = model(data)
    loss = criterion(output, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**SNN:**
```python
for data, targets in train_loader:
    utils.reset(net)  # Reset neuron states
    spk_rec = []
    for step in range(num_steps):  # Time loop
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
    spk_rec = torch.stack(spk_rec)
    loss = loss_fn(spk_rec, targets)  # Spike-count loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

difference is essentially: (1) a time loop, (2) state reset, (3) stacking spike outputs. everything else is identical PyTorch.

### Honest Time Estimate

for someone who's trained a CNN in PyTorch:
- **getting a basic SNN running:** 1-2 days following tutorials
- **understanding what's happening:** 3-5 days of reading + experimentation
- **tuning for good performance:** 1-2 weeks
- **understanding theory well enough to write about it:** 1-2 weeks

totally manageable for a semester project.

---

## Common Pitfalls

### Dead Neurons (most common)

neurons never fire (or fire every step). both prevent learning. cause: membrane potential never reaches threshold (too much leak / threshold too high) or always exceeds it. fix: start with beta=0.5, threshold=1.0. monitor firing rates. add firing rate regularization.

### Gradient Vanishing/Exploding

gradients go tiny or huge during BPTT through many steps. worse in SNNs than ANNs because of tanh-like surrogate behavior. narrow surrogate (high slope) = vanishing; wide (low slope) = noisy. fix: fewer time steps (25-50 usually fine for images), gradient clipping, moderate slope values, consider SNN-adapted batch norm (tdBN or BNTT).

