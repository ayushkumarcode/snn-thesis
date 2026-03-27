# STDP as a Thesis Focus -- Deep Dive

alright so i went deep into whether STDP (Spike-Timing-Dependent Plasticity) would work as a thesis topic, specifically for unsupervised feature learning with a biological plausibility angle. here's what i found.

STDP is far from "old news" in 2026. there's been a real resurgence driven by three things: (1) the NeurIPS 2024 NCG paper showing STDP-based local learning can be competitive on CIFAR-10/100 when done right, (2) growing demand for on-device learning that can't use backprop because of its non-local nature, and (3) neuromorphic hardware (Loihi 2, SpiNNaker2, memristive chips) that natively implements STDP in silicon.

the practical reality though -- pure STDP on MNIST gets about 95% (Diehl and Cook, 2015). the current SOTA hybrid approach (unsupervised STDP features + supervised STDP classifier with NCG) reaches 98.92% on MNIST, 88.72% on Fashion-MNIST, and 66.41% on CIFAR-10 (NeurIPS 2024). these are decent numbers but still lag behind surrogate-gradient SNNs by 5-15 pp on harder datasets. but the thesis shouldn't be about "beating backprop" -- it should be about "what can local, biologically plausible learning actually achieve, and where does it have real advantages?"

for an undergrad thesis, the hybrid approach (STDP unsupervised feature extraction + simple supervised classifier) seems like the sweet spot. implementable in one semester using BindsNET or SpykeTorch, produces visually interpretable learned features, and has multiple experimental dimensions to explore. strongest novel angles for 2026 would be: STDP on event-camera/DVS data where temporal coding naturally matches the learning rule, three-factor learning rules (reward-modulated STDP) for RL, or STDP for continual learning where its local nature might resist catastrophic forgetting.

---

## 1. What Can STDP Actually Learn?

### 1.1 Core Mechanism

STDP adjusts synaptic weights based on relative timing of pre- and post-synaptic spikes:
- **Pre fires before post (causal):** synapse strengthened (Long-Term Potentiation, LTP)
- **Post fires before pre (anti-causal):** synapse weakened (Long-Term Depression, LTD)

this creates an unsupervised, Hebbian-like rule that extracts temporal correlations in input spike patterns without any labels or global error signal.

### 1.2 What STDP Learns Well

| Task Domain | What STDP Extracts | Quality | Evidence |
|---|---|---|---|
| **Edge/Gabor-like filters** | Oriented edge detectors from natural images | Excellent | Masquelier & Thorpe (2007), Kheradpisheh et al. (2018) |
| **Digit prototypes** | Template-like representations of handwritten digits | Very Good | Diehl & Cook (2015) -- 95% MNIST |
| **Object parts/prototypes** | Intermediate visual features in deep CSNN | Good | Kheradpisheh et al. (2018) -- 99.1% Caltech face/motorbike |
| **Temporal patterns** | Repeating spike sequences, coincidence detection | Excellent | Foundational STDP property |
| **Audio/speech features** | Spectrotemporal patterns in audio | Good | 93.3% Spoken-MNIST (2024) |
| **Event-camera features** | Motion-sensitive filters from DVS data | Good | Paredes-Valles et al., cuSNN |
| **Spatial navigation** | Place/grid cell representations | Good | SpiNNaker implementations |

### 1.3 Where STDP Struggles

- **Fine-grained classification on complex datasets:** CIFAR-10 accuracy caps around 66% with pure STDP vs 95%+ for surrogate gradient methods
- **Deep network training:** has difficulty propagating useful learning signals through many layers
- **Precise categorical boundaries:** without supervision, learned features cluster by visual similarity, not semantic category
- **Scalability to high-res images:** computational cost grows significantly, convergence slows

### 1.4 Key Insight

STDP is fundamentally a feature extraction mechanism, not a classifier. its strength is unsupervised representation learning -- discovering statistical structure in input data. classification should be handled separately. this is directly analogous to how unsupervised pre-training (autoencoders, contrastive learning) works in deep learning, which gives the thesis a clean conceptual framework.

---

## 2. Best Implementations Available

### 2.1 Framework Comparison

| Framework | Backend | STDP Support | GPU | Best For | Maturity | Active? |
|---|---|---|---|---|---|---|
| **BindsNET** | PyTorch | Extensive (pair, post-pre, MSTDP, MSTDPET) | Yes | ML-oriented STDP experiments | High | Moderate (last release ~2023) |
| **Brian2** | Code generation (C++/Cython) | Fully customizable (any equation) | No (CPU only) | Neuroscience-accurate simulations | Very High | Yes |
| **SpykeTorch** | PyTorch | STDP + R-STDP for convolutional SNNs | Yes | Deep convolutional STDP | Medium | Low (archived) |
