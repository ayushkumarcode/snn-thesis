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
| **ngc-learn** | JAX | Trace STDP, event STDP, R-STDP | Yes | Biologically plausible models | Medium | Yes (v3.0.1) |
| **SpikeNN** | CPU Python | S2-STDP, SSTDP, NCG architecture | No | NeurIPS 2024 NCG paper code | New | Yes |
| **Norse** | PyTorch | Limited (focus on surrogate gradients) | Yes | Modern deep SNN training | High | Yes |
| **SpikingJelly** | PyTorch/CuPy | Limited STDP (focus on surrogate gradients) | Yes | High-performance deep SNNs | Very High | Yes |
| **snnTorch** | PyTorch | Minimal STDP | Yes | Educational + surrogate gradients | High | Yes |
| **Lava** (Intel) | Custom | Three-factor learning, R-STDP | CPU | Loihi deployment | High | Yes |
| **Custom (from scratch)** | Python/NumPy | Whatever you build | No | Deep understanding | N/A | N/A |

### 2.2 What i'd Pick

**Primary: BindsNET**

- built on PyTorch, GPU acceleration works out of the box
- ships with a near-replication of Diehl & Cook 2015 (`eth_mnist.py`) that gets ~95% on MNIST
- supports multiple STDP variants: standard pair-based, post-pre only, reward-modulated (MSTDP, MSTDPET)
- well-documented with examples for unsupervised, supervised, and RL tasks
- `DiehlAndCook2015` network class provides a ready-made baseline
- runs in ~1 hour on Intel i7 for full MNIST training, faster on GPU
- repo: https://github.com/BindsNET/bindsnet
- paper: Hazan et al., "BindsNET: A Machine Learning-Oriented Spiking Neural Networks Library in Python," Frontiers in Neuroinformatics, 2018

**Secondary: SpykeTorch (for convolutional STDP)**

if the thesis goes deep convolutional:
- implements STDP and R-STDP for conv layers with at-most-one-spike-per-neuron constraint
- comes with reimplementation of Kheradpisheh et al. (2018)
- repo: https://github.com/miladmozafari/SpykeTorch

**For the NCG SOTA results: SpikeNN**

official NeurIPS 2024 code:
- repo: https://github.com/ggoupy/SpikeNN
- CPU-only, Python 3.8+
- implements S2-STDP, SSTDP, and NCG architecture

**For neuroscience accuracy: Brian2**

if the thesis emphasizes biological plausibility over ML performance:
- arbitrary differential equations for neuron and synapse models
- has a Diehl & Cook 2015 example: https://brian2.readthedocs.io/en/2.9.0/examples/frompapers.Diehl_Cook_2015.html
- CPU-only, slower for large networks but more biologically faithful

### 2.3 Quick-Start Path

1. install BindsNET: `pip install bindsnet`
2. run `eth_mnist.py` from examples -- replicates Diehl & Cook 2015
3. visualize learned weight filters (they'll look like digit templates)
4. gives you a working STDP baseline in under 2 hours

---

## 3. Results on MNIST with STDP

### 3.1 Benchmarks

| Method | Architecture | MNIST Acc. | Year | Learning Type | Reference |
|---|---|---|---|---|---|
| **Diehl & Cook** | 2-layer FC, lateral inhibition | 95.0% | 2015 | Unsupervised STDP + label assignment | Frontiers in Comp. Neuro. |
| **STDP-CSNN + SVM** | Conv STDP + SVM classifier | ~97.2% | 2018 | Unsupervised STDP features + supervised SVM | Kheradpisheh et al. |
| **SSTDP** | FC layers | 98.1% | 2021 | Supervised STDP (hybrid with backprop info) | Frontiers in Neuroscience |
| **S2-STDP** | STDP-CSNN + FC | ~97.7% | 2024 | Unsupervised STDP features + supervised S2-STDP classifier | Goupy et al. |
| **S2-STDP + NCG** | STDP-CSNN + NCG FC | **98.92%** | 2024 | Unsupervised STDP features + supervised S2-STDP + NCG | NeurIPS 2024 |
| **S2-STDP + NCG (SoftHebb)** | SoftHebb-CNN + NCG FC | **99.17%** | 2024 | Unsupervised Hebbian features + supervised S2-STDP + NCG | NeurIPS 2024 |
| **Deep STDP pre-train + fine-tune** | Deep Conv SNN | ~98.0% | 2018 | STDP pre-training + gradient fine-tuning | Frontiers in Neuroscience |

### 3.2 What I Notice

- **Pure unsupervised STDP** (zero labels during training) peaks at ~95% on MNIST with 6400 excitatory neurons
- **STDP features + supervised classifier** pushes to 97-99%, competitive with many ANN baselines
- **NCG paper (NeurIPS 2024) is the current SOTA** for STDP-based classification at 98.92% (STDP-CSNN features) or 99.17% (SoftHebb-CNN features)
- for comparison, surrogate-gradient SNNs get ~99.5% on MNIST, standard CNNs get ~99.7%

### 3.3 What 95% Actually Means

Diehl & Cook's 95% with pure unsupervised STDP is actually kind of amazing when you think about it because:
1. no labels used during training at all
2. the network self-organizes to represent different digit classes
3. labels are assigned post-hoc by seeing which neuron fires most for which digit
4. each excitatory neuron's weight pattern visually resembles a digit template
5. directly comparable to k-means clustering (~96%) or unsupervised autoencoders

---

## 4. Hybrid Approach: STDP Features + Supervised Classifier

### 4.1 The Standard Pipeline

this is the most practical approach for a thesis:

```
[Input Image] --> [Spike Encoding] --> [STDP Conv/FC Layers] --> [Learned Features] --> [Supervised Classifier] --> [Output]
     |                  |                      |                        |                      |
  Raw pixels     Rate/temporal         Unsupervised            Fixed feature            SVM, logistic
  or events       coding              weight learning           extraction             regression, or
                                      (no labels)              (forward pass)          supervised STDP
```

### 4.2 Specific Architectures from the Literature

**Architecture A: Kheradpisheh et al. (2018) -- Deep CSNN + SVM**
1. input images encoded via DoG + temporal coding (first-spike)
2. multiple conv layers trained layer-by-layer with STDP
3. pooling between conv layers
4. final feature map flattened
5. linear SVM trained on flattened features
6. results: 99.1% Caltech face/motorbike, 82.8% ETH-80, ~97% MNIST

**Architecture B: NeurIPS 2024 NCG Pipeline**
1. input images as Poisson spike trains
2. STDP-trained conv SNN (STDP-CSNN) extracts features unsupervised
3. features converted to first-spike times via temporal coding
4. supervised S2-STDP trains a FC classification SNN
5. NCG adds intra-class competition for diversity
6. results: 98.92% MNIST, 88.72% Fashion-MNIST, 66.41% CIFAR-10

**Architecture C: STDP Pre-training + Gradient Fine-tuning (Lee et al., 2018)**
1. deep spiking CNN with multiple conv layers
2. phase 1: layer-wise unsupervised STDP pre-training
3. phase 2: end-to-end supervised fine-tuning with spike-based gradient descent
4. ~2.5x faster convergence vs random initialization
5. ~98% MNIST

### 4.3 What i'd Actually Do for a Thesis

**Phase 1: Unsupervised Feature Learning**
- train a single conv STDP layer (or Diehl & Cook FC network) on training set without labels
- visualize learned weight filters (should look like meaningful features)
- ~1-2 hours to train on MNIST

**Phase 2: Feature Extraction**
- pass training and test images through the trained STDP network
- record spike responses of excitatory neurons as feature vectors
- each image becomes a vector of firing rates or first-spike times

**Phase 3: Supervised Classification**
- train a simple classifier (SVM, logistic regression, even k-NN) on the extracted features
- compare against: (a) raw pixel features, (b) random SNN features, (c) ANN-learned features

**Phase 4: Analysis**
- visualize what STDP neurons learned (weight matrices as images)
