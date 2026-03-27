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

### Surrogate Gradient Scale Mismatch

learning stalls or oscillates despite gradients flowing. cause: slope parameter scaled wrong. fix: start with defaults (25 for fast_sigmoid, 2.0 for atan). if stalling, reduce slope. if unstable, increase slope.

### Memory Overflow from BPTT

OOM errors with many time steps or large batches. BPTT stores activations for all T steps, multiplying memory by T. fix: smaller batch, fewer steps, mixed precision (`torch.cuda.amp`).

### Loss of Sparsity

trained SNN fires at very high rates, defeating the purpose. without regularization, surrogate gradients can push toward high-firing-rate solutions that resemble ANNs. fix: add spike rate regularization, monitor firing rates, target 10-20% max.

### Forgetting to Reset States

erratic training, accuracy fluctuates. cause: membrane potentials from previous sample leak in. fix: always call `utils.reset(net)` before each new input.

### Wrong Number of Time Steps

very poor accuracy despite correct architecture. too few = not enough time for meaningful spike patterns. too many = temporal dilution. fix: 25-50 for static images, match data temporal structure for neuromorphic data. accuracy often peaks at moderate T (4-8 for some tasks).

### Debugging Checklist

```
[ ] Are neurons firing? Check avg spike rates per layer
[ ] Is loss decreasing? Plot loss curve
[ ] Are gradients flowing? Check gradient norms per layer
[ ] States being reset? Verify utils.reset()
[ ] Slope reasonable? Try defaults first
[ ] Beta appropriate? Start with 0.5
[ ] Time steps reasonable? Try 25 for images
[ ] Batch size causing OOM? Reduce
[ ] Learning rate appropriate? 1e-3 to 1e-2 for Adam
[ ] Spike counts used correctly in loss?
```

---

## snnTorch Implementation Guide

### Installation

```bash
pip install snntorch
```

that's it. only depends on PyTorch.

### Minimal Working Example (MNIST)

```python
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torchvision

# Hyperparams
num_steps = 25
beta = 0.5
batch_size = 128
lr = 1e-2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Surrogate gradient
spike_grad = surrogate.fast_sigmoid(slope=25)

# Network
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

# Data
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((28, 28)),
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0,), (1,)),
])
train_ds = torchvision.datasets.MNIST("./data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

# Loss and optimizer
loss_fn = SF.ce_rate_loss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# Forward pass
def forward_pass(net, data, num_steps):
    spk_rec, mem_rec = [], []
    utils.reset(net)
    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
    return torch.stack(spk_rec), torch.stack(mem_rec)

# Training
for epoch in range(1):
    for i, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        net.train()
        spk_rec, mem_rec = forward_pass(net, data, num_steps)
        loss = loss_fn(spk_rec, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            acc = SF.accuracy_rate(spk_rec, targets)
            print(f"Epoch {epoch}, Iter {i}, Loss: {loss.item():.4f}, Acc: {acc*100:.2f}%")
```

### Key Design Decisions

**Surrogate selection:**
```python
spike_grad = surrogate.fast_sigmoid(slope=25)    # recommended
spike_grad = surrogate.atan(alpha=2.0)            # default if unspecified
spike_grad = surrogate.sigmoid(slope=25)          # classic
spike_grad = surrogate.custom_surrogate(my_fn)    # user-defined
```

**Loss functions:**
```python
loss_fn = SF.ce_rate_loss()       # cross-entropy on spike counts (rate coding)
loss_fn = SF.mse_count_loss()     # MSE on spike counts
loss_fn = SF.ce_temporal_loss()   # cross-entropy on first spike time
```

**Neuron models:**
```python
snn.Leaky(beta=0.5, spike_grad=spike_grad)           # LIF (most common)
snn.Synaptic(alpha=0.9, beta=0.85, spike_grad=spike_grad)  # 2nd-order
snn.RLeaky(beta=0.5, spike_grad=spike_grad)           # Recurrent LIF
```

### All Surrogates in snnTorch

| Function | Usage | Key Param |
|----------|-------|-----------|
| `surrogate.atan(alpha=2.0)` | Arctangent | alpha |
| `surrogate.fast_sigmoid(slope=25)` | Fast sigmoid | slope |
| `surrogate.sigmoid(slope=25)` | Sigmoid | slope |
| `surrogate.triangular(threshold=1)` | Triangular | threshold |
| `surrogate.straight_through_estimator()` | STE | none |
| `surrogate.spike_rate_escape(beta=1, slope=25)` | Escape rate | beta, slope |
| `surrogate.LSO(slope=0.1)` | Leaky spike operator | slope |
| `surrogate.SFS(slope=25, B=1)` | Sparse fast sigmoid | slope, B |
| `surrogate.SSO(mean=0, variance=0.2)` | Stochastic spike operator | mean, variance |
| `surrogate.heaviside()` | True derivative (not useful for training) | none |
| `surrogate.custom_surrogate(fn)` | User-defined | custom function |

### How Straightforward Is snnTorch?

pretty straightforward honestly:
- **Installation:** single pip install, no special deps
- **Learning curve:** if you know PyTorch CNNs, working SNN in under 1 hour via Tutorial 6
- **API:** neuron models (snn.Leaky etc.) drop in between standard nn.Linear/nn.Conv2d layers
- **Docs:** excellent. 7+ interactive tutorials, all Colab-ready. created by Jason Eshraghian at UCSC specifically for teaching
- **Community:** 1.9k GitHub stars, active issues/discussions
- **Undergrad adoption:** used in UCSC "Brain-Inspired Deep Learning" course
- **Limitation:** less performant than SpikingJelly with CuPy for large-scale stuff, but negligible at thesis scale

---

## Comparative Studies

### Zenke and Vogels (2021) -- "The Remarkable Robustness"

systematic variation of surrogate shape and scale. key results: performance largely insensitive to shape, scale significantly impacts performance, activity regularization enables sparse robust function, works across feedforward and recurrent architectures.

### Fine-Tuning Study (2024)

compared fast sigmoid and atan on SVHN. fast sigmoid yields lower firing rate with similar accuracy, 1.72x improvement in accelerator efficiency (FPS/W). optimal: beta=0.5, threshold=1.5. 48% reduction in hardware latency with only 2.88% accuracy loss.

### Deng et al. (2023) -- Surrogate Module Learning (ICML 2023)

analyzed gradient error accumulation in deep SNNs. vanilla SNN accuracy peaks at much shallower depth (L=5) vs ANN (L=13). vanilla SNN degrades dramatically past 13 layers. proposed approach gets 85.23% on CIFAR10-DVS (SOTA at the time).

### Learnable Surrogate Gradient (IJCAI 2023)

made the surrogate width learnable. fixed-width causes gradient vanishing in deep layers. learnable width modulation based on membrane potential distribution fixes this. competitive on CIFAR-10, CIFAR-100, DVS-CIFAR10.

### Summary

| Surrogate | Accuracy Impact | Sparsity | Compute Cost | Best For |
|-----------|----------------|----------|--------------|---------|
| Fast Sigmoid | Best overall | Highest | Lowest | Production, thesis default |
| Arctangent | Very good | Moderate | Low | Wider gradient coverage |
| Sigmoid | Good | Moderate | Medium | Familiarity |
| Triangular | Poor | Variable | Lowest | Avoid for classification |
| STE | Baseline | Low | Lowest | Simple baselines only |
| Learnable | Best (deep nets) | Adaptive | High | Research on deep SNNs |

---

## Recent Advances (2024-2025)

### Theoretical Understanding (2025)

Gygax and Zenke (2025) published in Neural Computation -- surrogate gradients equal true gradients of expected output for single neurons under stochastic interpretation, but this breaks in deep networks. surrogate derivative linked to "escape noise function" of stochastic neuron models. first rigorous theoretical framework for why surrogates work.

### Exact Gradient Methods (2025)

Klos and Memmesheimer (2025) in Physical Review Letters -- exact (not surrogate) gradient computation using "pseudospikes" for continuous gradient flow. potential future alternative but currently more complex.

### Stabilizing Deep SNN Training (2025)

MP-Init and TrSG (2025): membrane potential initialization and threshold-robust surrogate gradients for temporal covariate shift. specifically targets instability in deeper architectures.

### Parametric Surrogates (2024-2025)

Wang et al. (2025): parametric surrogate gradient strategy that iteratively finds optimal surrogate per layer. removes manual parameter selection. also adaptive gradient learning (IJCAI 2025) that auto-calibrates sharpness based on membrane potential distribution.

### SOTA Benchmarks

| Dataset | Architecture | Accuracy | Time Steps | Method |
|---------|-------------|----------|------------|--------|
| MNIST | ConvSNN | 98.1% | 2 | Sigma-delta neurons |
| CIFAR-10 | ResNet-19 | 96.43% | 4 | Direct training |
| CIFAR-100 | ResNet-19 | 81.86% | 4 | Direct training |
| ImageNet-1k | SGLFormer | 83.73% | 4 | Spiking Transformer |
| ImageNet-1k | QKFormer | >85% | 4 | Spiking Transformer |
| CIFAR10-DVS | VGG-SNN | 82.95% | - | LNM |
| DVS128 Gesture | Various | 96-97% | - | Direct training |

### Trends

1. learnable/adaptive surrogates replacing fixed ones in cutting-edge work
2. spiking transformers reaching competitive performance with ANN transformers
3. theoretical foundations finally being established (Gygax & Zenke 2025)
4. fewer time steps needed (T=2-4 getting strong results)
5. normalization techniques (BNTT, tdBN, MP-Init) crucial for depth
6. SNN-ANN accuracy gap rapidly closing, especially on ImageNet

---

## Practical Recommendations

### Default Configuration

```python
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5           # Membrane decay
threshold = 1.0      # Firing threshold
num_steps = 25       # For static images
optimizer = Adam(lr=1e-2 to 1e-3)
loss_fn = SF.ce_rate_loss()
```

### What to Cite

| Paper | Why |
|-------|-----|
| Neftci, Mostafa, Zenke (2019) | Original surrogate gradient formalization |
| Zenke and Vogels (2021) | Robustness to surrogate shape |
| Eshraghian et al. (2023) | Review, snnTorch paper |
| Gygax and Zenke (2025) | Latest theoretical understanding |
| Zhou et al. (2024) | Direct training review |

### Timeline

| Task | Time |
|------|------|
| Complete snnTorch tutorials 5-7 | 3-5 days |
| Get baseline SNN working on dataset | 1-2 weeks |
| Tune hyperparams | 1-2 weeks |
| Run comparison experiments | 1-2 weeks |
| Write up | 2-3 weeks |

very feasible within a semester.

---

## Sources

### Foundational
- Neftci et al. (2019). Surrogate Gradient Learning. https://arxiv.org/abs/1901.09948
- Zenke & Vogels (2021). Remarkable Robustness. https://direct.mit.edu/neco/article/33/4/899/97482
- Eshraghian et al. (2023). Training SNNs Using Lessons From Deep Learning. https://arxiv.org/abs/2109.12894

### Recent (2024-2025)
- Gygax & Zenke (2025). Theoretical Underpinnings. https://direct.mit.edu/neco/article/37/5/886/128506
- Klos & Memmesheimer (2025). Exact Gradient Descent. https://doi.org/10.1103/PhysRevLett.134.027301
- MP-Init and TrSG (2025). https://arxiv.org/abs/2511.08708
- Wang et al. (2025). Parametric Surrogate Gradient. https://www.sciencedirect.com/science/article/abs/pii/S092523122401960X
- Fine-Tuning Surrogate Gradient Learning (2024). https://arxiv.org/abs/2402.06211
- Zhou et al. (2024). Direct Training Review. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1383844/full
- Adaptive Gradient Learning (IJCAI 2025). https://www.ijcai.org/proceedings/2025/0464.pdf

### Comparative Studies
- Lian et al. (2023). Learnable Surrogate Gradient. IJCAI. https://www.ijcai.org/proceedings/2023/335
- Deng et al. (2023). Surrogate Module Learning. ICML. https://proceedings.mlr.press/v202/deng23d
- KLIF Neuron (2024). Neural Computation. https://direct.mit.edu/neco/article/36/12/2636/124535

### Tutorials and Code
- snnTorch Tutorial 5: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html
- snnTorch Tutorial 6: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html
- snnTorch Tutorial 7: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_7.html
- snnTorch Surrogate API: https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html
- SpyTorch: https://github.com/fzenke/spytorch
- R Gaurav blog (2024): https://r-gaurav.github.io/2024/01/04/Building-And-Training-Spiking-Neural-Networks-From-Scratch.html
- snnTorch GitHub: https://github.com/jeshraghian/snntorch
- Practical Tutorial (MDPI, 2025): https://www.mdpi.com/2673-4117/6/11/304

### Framework Comparison
- SNN Framework Benchmarks: https://open-neuromorphic.org/blog/spiking-neural-network-framework-benchmarking/
- SpikingJelly: https://github.com/fangwei123456/spikingjelly

### Normalization
- tdBN: https://cdn.aaai.org/ojs/17320/17320-13-20814-1-2-20210518.pdf
- Batch Norm for Low-Latency SNNs: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.773954/full
