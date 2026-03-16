# STDP (Spike-Timing-Dependent Plasticity) as a Thesis Focus: Deep Research Report

**Research Date:** 2026-02-25
**Scope:** Comprehensive investigation of STDP for unsupervised feature learning with biological plausibility -- feasibility, implementations, results, novel angles, and thesis framing.

---

## Executive Summary

STDP remains a vibrant and publishable research area in 2026, far from "old news." The field has experienced a significant resurgence driven by three converging forces: (1) the NeurIPS 2024 acceptance of the Neuronal Competition Groups (NCG) paper demonstrating that STDP-based local learning can achieve competitive results on CIFAR-10/100 when properly architected, (2) the growing demand for on-device, privacy-preserving learning that cannot use backpropagation due to its non-local nature, and (3) the emergence of neuromorphic hardware (Loihi 2, SpiNNaker2, memristive chips) that natively implements STDP in silicon. The biological plausibility narrative is compelling for a thesis: STDP is the dominant experimentally-observed synaptic learning rule in the brain, and framing a project as "bridging neuroscience and machine learning" gives strong narrative coherence.

The practical reality is nuanced. Pure STDP on MNIST achieves approximately 95% accuracy (Diehl and Cook, 2015), while the state-of-the-art hybrid approach (unsupervised STDP feature extraction + supervised STDP classifier with NCG) reaches 98.92% on MNIST, 88.72% on Fashion-MNIST, and 66.41% on CIFAR-10 using a STDP-trained convolutional feature extractor (NeurIPS 2024). These numbers are respectable but lag behind surrogate-gradient-trained SNNs by 5-15 percentage points on complex datasets. However, the thesis angle should not be "beat backpropagation" -- it should be "what can local, biologically plausible learning achieve, and where does it have fundamental advantages?"

For an undergraduate thesis, the hybrid approach (STDP unsupervised feature extraction + simple supervised classifier) is the sweet spot: it is implementable in one semester using BindsNET or SpykeTorch, produces visually interpretable learned features, and offers multiple dimensions for experimental investigation. The strongest novel angles for 2026 would be: (a) STDP on event-camera/DVS data where the temporal coding matches the learning rule naturally, (b) three-factor learning rules (reward-modulated STDP) for reinforcement learning tasks, or (c) STDP for continual/lifelong learning where its local nature provides natural resistance to catastrophic forgetting.

---

## 1. What Can STDP Actually Learn? What Tasks Is It Good At?

### 1.1 Core Mechanism

STDP is a biologically observed synaptic plasticity rule that adjusts synaptic weights based on the relative timing of pre- and post-synaptic spikes:
- **Pre fires before post (causal):** synapse is strengthened (Long-Term Potentiation, LTP)
- **Post fires before pre (anti-causal):** synapse is weakened (Long-Term Depression, LTD)

This creates an unsupervised, Hebbian-like learning rule that extracts temporal correlations in input spike patterns without any labels or global error signal.

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

### 1.3 What STDP Struggles With

- **Fine-grained classification on complex datasets:** CIFAR-10 accuracy caps around 66% with pure STDP pipelines vs. 95%+ for surrogate gradient methods
- **Deep network training:** STDP has difficulty propagating useful learning signals through many layers
- **Precise categorical boundaries:** Without supervision, learned features cluster by visual similarity, not semantic category
- **Scalability to high-resolution images:** Computational cost grows significantly; convergence slows

### 1.4 Key Insight for the Thesis

STDP is fundamentally a feature extraction mechanism, not a classifier. Its strength is unsupervised representation learning -- discovering the statistical structure of input data. The classification step should be handled by a separate (potentially supervised) mechanism. This is directly analogous to how unsupervised pre-training (autoencoders, contrastive learning) works in deep learning, giving the thesis a clean conceptual framework.

---

## 2. Best Implementations Available

### 2.1 Framework Comparison Table

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

### 2.2 Recommended Stack for This Thesis

**Primary recommendation: BindsNET**

Reasons:
- Built on PyTorch, so GPU acceleration works out of the box
- Ships with a near-replication of Diehl & Cook 2015 (`eth_mnist.py`) that achieves ~95% on MNIST
- Supports multiple STDP variants: standard pair-based, post-pre only, reward-modulated (MSTDP, MSTDPET)
- Well-documented with examples for unsupervised, supervised, and RL tasks
- The `DiehlAndCook2015` network class provides a ready-made baseline
- Running time: ~1 hour on Intel i7 CPU for full MNIST training; faster on GPU
- Repository: https://github.com/BindsNET/bindsnet
- Paper: Hazan et al., "BindsNET: A Machine Learning-Oriented Spiking Neural Networks Library in Python," Frontiers in Neuroinformatics, 2018

**Secondary recommendation: SpykeTorch (for convolutional STDP)**

If the thesis focuses on deep convolutional STDP feature extraction:
- Implements STDP and R-STDP for convolutional layers with at-most-one-spike-per-neuron constraint
- Comes with a reimplementation of Kheradpisheh et al. (2018) deep CSNN
- Repository: https://github.com/miladmozafari/SpykeTorch

**For the NCG/S2-STDP state-of-the-art results:**

SpikeNN is the official code from the NeurIPS 2024 paper:
- Repository: https://github.com/ggoupy/SpikeNN
- CPU-only, Python 3.8+
- Implements S2-STDP, SSTDP, and the NCG architecture

**For neuroscience accuracy: Brian2**

If the thesis emphasizes biological plausibility over ML performance:
- Allows arbitrary differential equations for neuron and synapse models
- Includes a Diehl & Cook 2015 example: https://brian2.readthedocs.io/en/2.9.0/examples/frompapers.Diehl_Cook_2015.html
- CPU-only; slower for large networks but more biologically faithful

### 2.3 Quick-Start Path

1. Install BindsNET: `pip install bindsnet`
2. Run `eth_mnist.py` from the examples folder -- this replicates Diehl & Cook 2015
3. Visualize learned weight filters (they will look like digit templates)
4. This gives a working STDP baseline in under 2 hours of setup

---

## 3. Typical Results on MNIST with STDP-Only Approaches

### 3.1 Benchmark Results Table

| Method | Architecture | MNIST Acc. | Year | Learning Type | Reference |
|---|---|---|---|---|---|
| **Diehl & Cook** | 2-layer FC, lateral inhibition | 95.0% | 2015 | Unsupervised STDP + label assignment | Frontiers in Computational Neuroscience |
| **STDP-CSNN + SVM** | Conv STDP + SVM classifier | ~97.2% | 2018 | Unsupervised STDP features + supervised SVM | Kheradpisheh et al. |
| **SSTDP** | FC layers | 98.1% | 2021 | Supervised STDP (hybrid with backprop info) | Frontiers in Neuroscience |
| **S2-STDP** | STDP-CSNN + FC | ~97.7% | 2024 | Unsupervised STDP features + supervised S2-STDP classifier | Goupy et al. |
| **S2-STDP + NCG** | STDP-CSNN + NCG FC | **98.92%** | 2024 | Unsupervised STDP features + supervised S2-STDP + NCG | NeurIPS 2024 |
| **S2-STDP + NCG (SoftHebb)** | SoftHebb-CNN + NCG FC | **99.17%** | 2024 | Unsupervised Hebbian features + supervised S2-STDP + NCG | NeurIPS 2024 |
| **Deep STDP pre-train + supervised fine-tune** | Deep Conv SNN | ~98.0% | 2018 | STDP pre-training + gradient fine-tuning | Frontiers in Neuroscience |

### 3.2 Key Observations

- **Pure unsupervised STDP** (no labels at all during training) peaks at ~95% on MNIST with 6400 excitatory neurons
- **STDP features + supervised classifier** pushes to 97-99%, competitive with many ANN baselines
- **The NCG paper (NeurIPS 2024) represents the current state of the art** for STDP-based classification, at 98.92% (STDP-CSNN features) or 99.17% (SoftHebb-CNN features)
- For comparison, surrogate-gradient SNNs achieve ~99.5% on MNIST, and standard CNNs achieve ~99.7%

### 3.3 What 95% Actually Means

Diehl & Cook's 95% with pure unsupervised STDP is remarkable because:
1. No labels are used during training at all
2. The network self-organizes to represent different digit classes
3. Labels are assigned post-hoc by seeing which neuron fires most for which digit
4. Each excitatory neuron's weight pattern visually resembles a digit template
5. This is directly comparable to k-means clustering (~96%) or unsupervised autoencoders

---

## 4. How to Combine STDP Feature Extraction + Supervised Classifier (Hybrid Approach)

### 4.1 The Standard Pipeline

This is the most practical and well-studied approach for a thesis:

```
[Input Image] --> [Spike Encoding] --> [STDP Conv/FC Layers] --> [Learned Features] --> [Supervised Classifier] --> [Output]
     |                  |                      |                        |                      |
  Raw pixels     Rate/temporal         Unsupervised            Fixed feature            SVM, logistic
  or events       coding              weight learning           extraction             regression, or
                                      (no labels)              (forward pass)          supervised STDP
```

### 4.2 Specific Hybrid Architectures from Literature

**Architecture A: Kheradpisheh et al. (2018) -- Deep CSNN + SVM**
1. Input images encoded via Difference-of-Gaussians + temporal coding (first-spike)
2. Multiple convolutional layers trained layer-by-layer with STDP
3. Pooling layers between convolutional layers
4. Final feature map extracted and flattened
5. Linear SVM classifier trained on the flattened features
6. Results: 99.1% Caltech face/motorbike, 82.8% ETH-80, ~97% MNIST

**Architecture B: NeurIPS 2024 NCG Pipeline**
1. Input images encoded as Poisson spike trains
2. STDP-trained convolutional SNN (STDP-CSNN) extracts features unsupervised
3. Features converted to first-spike times via temporal coding
4. Supervised S2-STDP trains a fully-connected classification SNN
5. Neuronal Competition Groups (NCG) add intra-class competition for diversity
6. Results: 98.92% MNIST, 88.72% Fashion-MNIST, 66.41% CIFAR-10

**Architecture C: STDP Pre-training + Gradient Fine-tuning (Lee et al., 2018)**
1. Deep spiking CNN with multiple convolutional layers
2. Phase 1: Layer-wise unsupervised STDP pre-training
3. Phase 2: End-to-end supervised fine-tuning with spike-based gradient descent
4. Result: ~2.5x faster convergence compared to random initialization
5. Results: ~98% MNIST with faster convergence

### 4.3 Recommended Thesis Approach

For an undergraduate thesis, **Architecture B (simplified)** is the ideal target:

**Phase 1: Unsupervised Feature Learning**
- Train a single convolutional STDP layer (or the Diehl & Cook FC network) on the training set without labels
- Visualize learned weight filters (these should look like meaningful features)
- This takes ~1-2 hours to train on MNIST

**Phase 2: Feature Extraction**
- Pass training and test images through the trained STDP network
- Record the spike responses of the excitatory neurons as feature vectors
- Each image becomes a vector of firing rates or first-spike times

**Phase 3: Supervised Classification**
- Train a simple classifier (SVM, logistic regression, or even k-NN) on the extracted features
- Compare against: (a) raw pixel features, (b) random SNN features, (c) ANN-learned features

**Phase 4: Analysis**
- Visualize what the STDP neurons learned (weight matrices as images)
- Analyze selectivity of individual neurons to specific classes
- Compare STDP feature quality to unsupervised ANN methods (PCA, autoencoders, k-means)
- Measure energy efficiency (spike counts, synaptic operations)

### 4.4 Practical Implementation Notes

Using BindsNET, the feature extraction step looks roughly like:

```python
# After training the Diehl & Cook network with STDP:
# 1. Set network to inference mode (disable learning)
network.learning = False

# 2. Present each image and record spikes
for image in dataset:
    network.run(inputs={"X": image}, time=350)  # 350ms per image
    spikes = network.monitors["Ae"].get("s")  # excitatory layer spikes
    feature_vector = spikes.sum(dim=0)  # firing rate encoding
    features.append(feature_vector)

# 3. Train SVM on extracted features
from sklearn.svm import SVC
clf = SVC().fit(train_features, train_labels)
accuracy = clf.score(test_features, test_labels)
```

---

## 5. Is STDP a Good Thesis Topic or Is It "Old News"?

### 5.1 Verdict: STDP Is Alive, Active, and Publishable

**Evidence that STDP is NOT old news:**

| Signal | Evidence | Year |
|---|---|---|
| **Top venue publication** | NCG with Supervised STDP accepted at **NeurIPS 2024** | 2024 |
| **Top venue publication** | Dendritic Localized Learning (STDP-adjacent) at **ICML 2025** | 2025 |
| **Comprehensive review** | Three-factor learning in SNNs review in **Patterns (Cell Press)** | Nov 2025 |
| **Comprehensive review** | Modulated STDP review in **Neurocomputing** | Feb 2025 |
| **Nature publication** | Unsupervised post-training learning with triplet STDP in **Scientific Reports** | May 2025 |
| **Nature publication** | TEXEL neuromorphic chip with on-chip STDP in **Nature Communications** | 2025 |
| **Hardware integration** | Intel Loihi 2 natively supports STDP and three-factor rules | Ongoing |
| **Active GitHub repos** | SpikeNN (NCG code), ngc-learn v3, BindsNET all maintained | 2024-2025 |
| **New frameworks** | Inferno (Sept 2024) -- new SNN framework with extensible plasticity | 2024 |

### 5.2 Why STDP Has Renewed Relevance in 2026

1. **Energy crisis in AI:** Training GPT-4-class models costs millions of dollars in electricity. STDP-based local learning on neuromorphic hardware consumes 3-5 orders of magnitude less energy per synaptic operation (20-50 pJ vs. microjoules on GPUs).

2. **On-device / edge learning:** Backpropagation requires storing full computation graphs and performing backward passes -- impossible on tiny edge devices. STDP is purely local: each synapse only needs information from its two connected neurons. This enables on-chip, online learning.

3. **Privacy-preserving AI:** STDP enables on-device learning without sending data to the cloud, which is increasingly important under GDPR and similar regulations.

4. **Neuromorphic hardware maturation:** Loihi 2, SpiNNaker2, BrainScaleS-2, and memristive chips all implement STDP natively. The hardware exists; now researchers need algorithms.

5. **Biological understanding:** Neuroscience is discovering increasingly complex STDP variants (dendritic STDP, voltage-dependent plasticity, heterosynaptic plasticity). Computational models of these are needed.

### 5.3 What Would Make It "Old News"

The thesis should NOT simply replicate Diehl & Cook (2015) on MNIST. That is indeed a 10-year-old result. The thesis needs a novel angle (see Section 6).

### 5.4 The Narrative Advantage

"Biologically plausible learning" is an excellent thesis narrative because:
- It connects to neuroscience (interdisciplinary appeal)
- It connects to energy-efficient AI (practical relevance)
- It connects to neuromorphic hardware (cutting-edge technology)
- It has a clear research question: "How well can the brain's learning rule work for machine learning tasks?"
- It produces visually compelling results (learned filters look like Gabor filters or digit templates)
- Examiners find the biological angle intellectually interesting

---

## 6. What Would Make an STDP Project Interesting in 2026?

### 6.1 Ranked Project Ideas (Best to Good)

#### Tier 1: Strongest Novel Angles

**Idea A: STDP on Event-Camera (DVS) Data -- "Learning the Way the Brain Sees"**
- **Why novel:** DVS cameras produce asynchronous spike-like events -- a natural match for STDP's temporal learning rule. Most DVS classification work uses surrogate gradients, not STDP. There is a gap.
- **What to do:** Train an STDP-based CSNN on N-MNIST or DVS128 Gesture using BindsNET or SpykeTorch. Compare STDP-learned features vs. random features vs. surrogate-gradient features.
- **Datasets:** N-MNIST (neuromorphic MNIST), DVS128 Gesture (11 gesture classes), CIFAR10-DVS
- **Expected results:** STDP should achieve 90-95% on N-MNIST, 80-90% on DVS128 Gesture with a good architecture
- **Why examiners will like it:** Natural fit between data modality and learning rule; tells a coherent biological story
- **Feasibility:** HIGH -- Tonic library handles data loading; BindsNET/SpykeTorch handle STDP

**Idea B: Three-Factor Learning (Reward-Modulated STDP) for RL Tasks**
- **Why novel:** Standard STDP is unsupervised. Adding a dopamine-like reward signal creates three-factor learning: pre-synaptic activity x post-synaptic activity x reward. This is how the brain is believed to do reinforcement learning. A 2025 review in Patterns calls this "a crucial extension of traditional STDP."
- **What to do:** Implement R-STDP using BindsNET's MSTDP or MSTDPET rules. Train an SNN agent on a simple RL task (e.g., CartPole, or a custom maze navigation). Compare R-STDP vs. standard RL (DQN).
- **Expected results:** R-STDP should solve simple tasks. The interesting analysis is energy efficiency and biological plausibility comparison.
- **Why examiners will like it:** Connects neuroscience (dopamine), ML (reinforcement learning), and neuromorphic computing in one project
- **Feasibility:** MEDIUM-HIGH -- BindsNET has R-STDP built in; RL environments via OpenAI Gym

**Idea C: STDP for Continual/Lifelong Learning -- "Learning Without Forgetting"**
- **Why novel:** Catastrophic forgetting is a major unsolved problem in deep learning. STDP's local nature means it only modifies synapses relevant to current inputs, potentially preserving old knowledge. A 2024 paper demonstrated wake-sleep learning in R-STDP networks to avoid catastrophic forgetting.
- **What to do:** Train an STDP network on MNIST digits 0-4, then on digits 5-9. Measure how much accuracy on 0-4 degrades. Compare against standard ANN training (which will catastrophically forget). Implement sleep/replay mechanisms.
- **Expected results:** STDP should show less forgetting than naive ANNs, but still some. The analysis is where the thesis value lies.
- **Why examiners will like it:** Addresses a hot problem (continual learning) from a biological angle
- **Feasibility:** MEDIUM -- requires careful experimental design

#### Tier 2: Strong Angles

**Idea D: Comparing STDP Variants on the Same Task**
- **Why interesting:** There are now many STDP variants (pair-based, triplet, voltage-dependent, R-STDP, symmetric STDP, S2-STDP). No comprehensive undergraduate-level comparison exists.
- **What to do:** Implement 3-4 STDP variants in BindsNET/Brian2. Train on MNIST and Fashion-MNIST. Compare: accuracy, convergence speed, energy (spike counts), feature quality, biological plausibility.
- **Feasibility:** HIGH -- mostly parameter/rule changes in existing code

**Idea E: STDP Feature Learning vs. Unsupervised ANN Methods**
- **Why interesting:** Direct comparison between STDP (brain's learning rule) and modern unsupervised methods (autoencoders, contrastive learning, k-means) as feature extractors.
- **What to do:** Extract features with STDP, PCA, autoencoder, SimCLR-tiny, then classify with the same SVM. Which features are best? Analyze feature quality, biological plausibility, compute cost.
- **Feasibility:** HIGH -- all methods have standard implementations

**Idea F: STDP + Audio/Speech Recognition**
- **Why interesting:** Audio is inherently temporal -- a natural fit for STDP. Recent work achieved 93.3% on Spoken-MNIST and 88.1% on Spiking Heidelberg Digits.
- **What to do:** Apply STDP feature extraction to audio spectrograms or direct spike-encoded audio. Compare with standard audio ML pipelines.
- **Feasibility:** MEDIUM -- audio spike encoding requires more setup

#### Tier 3: Solid But More Standard

**Idea G: Replicating and Extending Diehl & Cook (2015)**
- Replicate the 95% MNIST result in BindsNET, then extend with: (a) convolutional topology, (b) more neurons, (c) different datasets (Fashion-MNIST, EMNIST), (d) adding a supervised readout layer
- **Feasibility:** VERY HIGH -- most straightforward project

### 6.2 Specific Thesis Title Suggestions

- "Biologically Plausible Feature Learning: A Comparative Study of STDP Variants for Visual Pattern Recognition"
- "Learning Without Labels: STDP-Based Unsupervised Feature Extraction in Spiking Neural Networks"
- "From Spikes to Decisions: Hybrid STDP Feature Extraction with Supervised Classification in Spiking Neural Networks"
- "Brain-Inspired Learning for Event-Driven Vision: STDP on Dynamic Vision Sensor Data"
- "Three-Factor Learning in Spiking Neural Networks: Reward-Modulated STDP for Reinforcement Learning Tasks"
- "Can the Brain's Learning Rule Prevent Forgetting? STDP for Continual Learning in Spiking Neural Networks"

---

## 7. Novel STDP Variants from Recent Papers

### 7.1 Comprehensive Variant Table

| Variant | Key Innovation | Reference | Year | Maturity |
|---|---|---|---|---|
| **Standard Pair-based STDP** | Classical pre-post / post-pre timing rule | Bi & Poo (1998) | 1998 | Foundational |
| **Triplet STDP** | Considers triplets of spikes (pre-post-pre, post-pre-post) for richer dynamics | Pfister & Gerstner (2006) | 2006 | Mature |
| **R-STDP (Reward-Modulated)** | Multiplies STDP update by a global reward/dopamine signal | Izhikevich (2007), Fremaux & Gerstner (2016) | 2007/2016 | Mature |
| **Voltage-Dependent Plasticity (VDSP)** | Updates based on membrane potential rather than spike timing | Clopath et al. (2010) | 2010 | Moderate |
| **Symmetric STDP** | LTP for both pre-before-post and post-before-pre timing | Hao & Huang (2019) | 2019 | Moderate |
| **SSTDP (Supervised STDP)** | Bridges backpropagation and STDP using both spatial and temporal information | Mirsadeghi et al. (2021) | 2021 | Moderate |
| **S2-STDP (Stabilized Supervised)** | Dynamic target timestamps; alternates firing between target and non-target times | Goupy et al. (2024) | 2024 | Recent |
| **NCG (Neuronal Competition Groups)** | Architecture with intra-class WTA and two-compartment thresholds | Goupy et al. (NeurIPS 2024) | 2024 | State-of-art |
| **SADP (Spike Agreement Dependent Plasticity)** | Replaces pairwise timing with population-level agreement metrics | arXiv, Jan 2026 | 2026 | Cutting-edge |
| **Dendritic Localized Learning (DLL)** | Three-compartment neuron (soma, apical, basal) with local error computation | ICML 2025 | 2025 | Cutting-edge |
| **Meta-Learning R-STDP** | R-STDP with hippocampus/prefrontal-cortex-inspired meta-learning | Neurocomputing, Oct 2024 | 2024 | Recent |
| **Wake-Sleep R-STDP** | R-STDP during "day phase" + generative replay during "night phase" for continual learning | 2024 | 2024 | Recent |
| **Heterogeneous STDP** | Different STDP rules at different synapses in the same network | Advanced Materials, 2025 | 2025 | Emerging |
| **Forecast-based STDP** | Predictive coding version -- learns to predict future spikes | Nature Communications, 2023 | 2023 | Moderate |

### 7.2 Most Thesis-Relevant Novel Variants

**S2-STDP + NCG (NeurIPS 2024)** is the most impactful recent contribution. The key innovations:
1. S2-STDP dynamically computes target timestamps (Ttarget, Tnon-target) per sample based on average firing time
2. Target neurons learn to fire before the mean; non-target neurons after
3. NCG groups multiple neurons per class with intra-class competition
4. Two-compartment thresholds regulate competition
5. Code available: https://github.com/ggoupy/SpikeNN

**Three-Factor Learning Rules (2025 Review in Patterns/Cell Press)**:
The formal three-factor rule is: Delta_w = M * F(pre, post), where M is a neuromodulatory signal (reward, error, attention) and F is a Hebbian/STDP-like coincidence detector. Variants include:
- **R-max:** Maximal for pre-before-post coincidences, modulated by reward minus baseline
- **R-STDP:** Bi-phasic coincidence window, modulated by success signal
- **TD-STDP:** Modulated by temporal-difference error (for RL)
- **e-prop:** Eligibility propagation with eligibility traces (biologically plausible gradient approximation)

---

## 8. How Does STDP Scale to Harder Datasets Beyond MNIST?

### 8.1 Accuracy on Progressively Harder Datasets

| Dataset | Best STDP-Based Result | Method | Best SNN (any method) | Gap |
|---|---|---|---|---|
| **MNIST** | 98.92% | S2-STDP + NCG (STDP-CSNN) | ~99.5% (surrogate grad) | ~0.6 pp |
| **Fashion-MNIST** | 88.72% | S2-STDP + NCG (STDP-CSNN) | ~93%+ (surrogate grad) | ~4-5 pp |
| **CIFAR-10** | 66.41% | S2-STDP + NCG (STDP-CSNN) | ~95%+ (surrogate grad) | ~29 pp |
| **CIFAR-10** | 79.55% | S2-STDP + NCG (SoftHebb-CNN) | ~95%+ (surrogate grad) | ~16 pp |
| **CIFAR-100** | 35.90% | S2-STDP + NCG (STDP-CSNN) | ~78%+ (surrogate grad) | ~42 pp |
| **CIFAR-100** | 53.49% | S2-STDP + NCG (SoftHebb-CNN) | ~78%+ (surrogate grad) | ~25 pp |
| **N-MNIST** | ~93-95% | STDP-based MLP | ~99.5% (surrogate grad) | ~5-6 pp |
| **DVS128 Gesture** | ~90-92% | STDP-based methods | ~98.7% (modern SNN) | ~7-8 pp |
| **Caltech face/motorbike** | 99.1% | STDP-CSNN + SVM | N/A | N/A (binary) |
| **ETH-80** | 82.8% | STDP-CSNN + SVM | N/A | N/A |
| **Spoken-MNIST** | 93.3% | SOM-Associated-SNN with STDP | Higher with surrogate grad | TBD |
| **SHD (Spiking Heidelberg Digits)** | 88.1% | SOM-Associated-SNN with STDP | ~95%+ | ~7 pp |

### 8.2 The Scaling Problem -- Honest Assessment

The data tells a clear story: **STDP scales poorly to complex datasets when used as the only learning mechanism.** The gap between STDP-based and surrogate-gradient approaches widens dramatically as dataset complexity increases:
- MNIST: gap ~0.6 pp (negligible)
- Fashion-MNIST: gap ~4-5 pp (noticeable)
- CIFAR-10: gap ~16-29 pp (significant)
- CIFAR-100: gap ~25-42 pp (very large)

### 8.3 Why the Gap Exists

1. **No global error signal:** STDP only sees local spike timing; it cannot propagate errors backward through layers
2. **Feature hierarchy problem:** Deep hierarchical features require coordinated learning across layers, which pure STDP cannot achieve
3. **Curse of unsupervised learning:** Without labels, STDP learns visually salient features, not necessarily discriminative ones
4. **Convergence instability:** STDP can cause weight explosion or death in deep networks without careful homeostatic mechanisms

### 8.4 How to Frame This in the Thesis

Do NOT frame the thesis as "STDP vs. backpropagation" -- STDP will lose on accuracy. Instead frame it as:
- **"What can local, biologically plausible learning achieve?"** -- This is a legitimate scientific question
- **"STDP as an efficient feature extractor"** -- Compare the quality of STDP features to other unsupervised methods, not to supervised methods
- **"Energy-accuracy trade-off"** -- STDP may achieve lower accuracy but with orders of magnitude less energy. Quantify this trade-off.
- **"Biological plausibility"** -- Rate the biological plausibility of different approaches. STDP wins on this axis.
- **"Hybrid approaches"** -- Show that combining STDP features with a simple classifier bridges much of the gap

---

## 9. Network Map: Key Research Groups and People

| Person/Group | Affiliation | Contribution | Key Papers |
|---|---|---|---|
| **Peter Diehl & Matthew Cook** | ETH Zurich / INI | Foundational STDP-MNIST paper (2015) | Unsupervised learning of digit recognition using STDP |
| **Saeed Reza Kheradpisheh** | University of Tehran | Deep CSNN with STDP for object recognition | STDP-based spiking deep CNNs (2018) |
| **Milad Mozafari** | University of Tehran | SpykeTorch framework, R-STDP for categorization | SpykeTorch (2019), First-spike categorization (2018) |
| **Gaspard Goupy et al.** | University of Lille / Fox team | S2-STDP, PCN, NCG architecture | NeurIPS 2024 NCG paper |
| **Timothee Masquelier** | CNRS CerCo, Toulouse | Unsupervised visual feature learning with STDP | PLOS Comp Bio (2007), extensive STDP work |
| **Wulfram Gerstner** | EPFL | Three-factor learning theory, neuromodulated STDP | Fremaux & Gerstner (2016) review |
| **Alexander Ororbia** | RIT | ngc-learn framework, biologically plausible computing | ngc-learn documentation + papers |
| **Hananel Hazan & Daniel Saunders** | UMass Amherst / BINDS Lab | BindsNET framework | BindsNET paper (2018) |
| **Mike Davies** | Intel Labs | Loihi processor with on-chip STDP | Loihi papers |

---

## 10. Recommended Thesis Structure

For a thesis framed as "Biologically Plausible Feature Learning with STDP":

```
Chapter 1: Introduction
  - Motivation: energy crisis in AI, biological inspiration
  - Research question: "How effective is the brain's STDP learning rule
    for unsupervised feature extraction in visual recognition tasks?"
  - Contributions: implementation, comparison, analysis

Chapter 2: Background
  - Spiking neural networks (LIF neurons, spike coding)
  - STDP: biological evidence and computational models
  - Comparison with ANN learning rules
  - Neuromorphic hardware context

Chapter 3: Methodology
  - Network architecture (Diehl & Cook or convolutional STDP)
  - STDP learning rule implementation details
  - Hybrid pipeline: unsupervised features + supervised classifier
  - Datasets: MNIST, Fashion-MNIST, [optional: N-MNIST or audio]
  - Evaluation metrics: accuracy, spike count, feature quality

Chapter 4: Experiments and Results
  - Experiment 1: STDP feature learning on MNIST (baseline replication)
  - Experiment 2: STDP vs. unsupervised ANN feature extractors
  - Experiment 3: Hybrid STDP + classifier on Fashion-MNIST
  - Experiment 4: [Novel angle -- DVS data / continual learning / STDP variants]
  - Experiment 5: Energy efficiency analysis (synaptic operations)

Chapter 5: Analysis and Discussion
  - What did STDP learn? (weight visualization, selectivity analysis)
  - Where does STDP succeed and fail?
  - Biological plausibility assessment
  - Energy-accuracy trade-off

Chapter 6: Conclusion and Future Work
  - Summary of findings
  - Limitations
  - Future work: deeper networks, neuromorphic deployment, three-factor rules
```

---

## 11. Research Gaps and Confidence Assessment

### 11.1 Information Gaps

| Gap | Why It Exists | Impact |
|---|---|---|
| Exact S2-STDP + PCN accuracy numbers per dataset | Tables in Frontiers paper not fully extractable from web | Low -- NCG paper provides the state-of-art numbers |
| BindsNET release date and version | GitHub releases page not deeply inspected | Low -- framework works regardless |
| Brian2 STDP performance benchmarks vs BindsNET on same task | No direct comparison paper found | Medium -- would be useful for framework selection |
| STDP on ImageNet or large-scale datasets | Appears to not exist -- likely too computationally expensive | Low -- not relevant for undergraduate thesis |

### 11.2 Confidence Levels

| Finding | Confidence | Basis |
|---|---|---|
| STDP achieves ~95% on MNIST (Diehl & Cook) | VERY HIGH | Original paper, 1000+ citations, replicated many times |
| NCG achieves 98.92% on MNIST with STDP features | HIGH | NeurIPS 2024 paper, code available, results in paper |
| STDP scales poorly to CIFAR-10 (66% with STDP-CSNN) | HIGH | Multiple sources confirm, NCG paper provides numbers |
| BindsNET is the best framework for ML-oriented STDP | HIGH | Multiple comparison sources, most ML features |
| STDP is still an active research area in 2025-2026 | VERY HIGH | NeurIPS 2024, ICML 2025, multiple 2025 reviews |
| Three-factor learning is the most promising STDP extension | HIGH | 2025 Cell Press review, growing publication count |
| STDP on DVS data is an underexplored thesis angle | MEDIUM-HIGH | Limited STDP-specific DVS papers; most DVS work uses surrogate gradients |

---

## 12. Key References and Sources

### Foundational Papers
- [Diehl & Cook (2015) -- Unsupervised learning of digit recognition using STDP](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full)
- [Masquelier & Thorpe (2007) -- Unsupervised learning of visual features through STDP](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0030031)
- [Kheradpisheh et al. (2018) -- STDP-based spiking deep CNNs for object recognition](https://www.sciencedirect.com/science/article/abs/pii/S0893608017302903)

### State-of-the-Art (2024-2025)
- [Goupy et al. (NeurIPS 2024) -- Neuronal Competition Groups with Supervised STDP](https://arxiv.org/abs/2410.17066)
- [Goupy et al. (2024) -- Paired competing neurons improving STDP supervised local learning](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1401690/full)
- [Unsupervised post-training learning in SNNs (Scientific Reports, 2025)](https://www.nature.com/articles/s41598-025-01749-x)
- [Deep Unsupervised Learning Using STDP (arXiv, 2023)](https://arxiv.org/abs/2307.04054)

### Reviews and Surveys
- [Three-factor learning in SNNs review (Patterns/Cell Press, Nov 2025)](https://www.cell.com/patterns/fulltext/S2666-3899(25)00262-4)
- [Modulated STDP-based learning for SNN: A review (Neurocomputing, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0925231224019416)
- [Fremaux & Gerstner (2016) -- Neuromodulated STDP, and Theory of Three-Factor Learning Rules](https://pmc.ncbi.nlm.nih.gov/articles/PMC4717313/)

### Frameworks and Code
- [BindsNET -- ML-oriented SNN library](https://github.com/BindsNET/bindsnet)
- [SpykeTorch -- Convolutional SNN with STDP](https://github.com/miladmozafari/SpykeTorch)
- [SpikeNN -- NeurIPS 2024 NCG code](https://github.com/ggoupy/SpikeNN)
- [ngc-learn -- Biologically plausible models in JAX](https://ngc-learn.readthedocs.io/en/latest/)
- [Brian2 Diehl & Cook example](https://brian2.readthedocs.io/en/2.9.0/examples/frompapers.Diehl_Cook_2015.html)
- [Original Diehl & Cook code](https://github.com/peter-u-diehl/stdp-mnist)

### Novel Directions
- [Meta-learning in SNNs with R-STDP (Neurocomputing, Oct 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0925231224009445)
- [SADP -- Spike Agreement Dependent Plasticity (arXiv, Jan 2026)](https://arxiv.org/html/2601.08526v1)
- [Dendritic Localized Learning (ICML 2025)](https://arxiv.org/abs/2501.09976)
- [TEXEL neuromorphic chip with on-chip STDP (Nature Communications, 2025)](https://www.nature.com/articles/s41467-025-61576-6)
- [Wake-Sleep R-STDP for catastrophic forgetting avoidance](https://www.researchgate.net/publication/397358146_Wake-Sleep_Learning_in_R-STDP-Based_Spiking_Neural_Networks_to_Avoid_Catastrophic_Forgetting)

### Audio and Event-Based Vision
- [SOM-Associated-SNN for audio classification (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0925231225010884)
- [Unsupervised speech recognition through STDP in convolutional SNN (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0204596)
- [Hardware, Algorithms, and Applications of the Neuromorphic Vision Sensor (2025)](https://arxiv.org/html/2504.08588v1)

### Continual Learning
- [Brain-inspired algorithm mitigating catastrophic forgetting (Science Advances)](https://www.science.org/doi/10.1126/sciadv.adi2947)
- [Hybrid neural networks for continual learning (Nature Communications, 2025)](https://www.nature.com/articles/s41467-025-56405-9)

### Neuromorphic Hardware
- [Neuromorphic Computing 2025: Current SotA](https://humanunsupervised.com/papers/neuromorphic_landscape.html)
- [Open Neuromorphic Hardware Guide](https://open-neuromorphic.org/neuromorphic-computing/hardware/)

---

## 13. Bottom-Line Recommendation

**STDP is an excellent thesis topic for 2026.** It is not old news -- it had a NeurIPS 2024 paper, an ICML 2025 paper, and multiple Nature/Cell Press publications in 2025. The "biologically plausible learning" framing gives a strong narrative that distinguishes the thesis from the majority of SNN work (which uses surrogate gradients and has little biological connection).

**The recommended project for this student:**

1. **Replicate** Diehl & Cook (2015) in BindsNET as the baseline (~95% MNIST, 1-2 weeks)
2. **Extend** to Fashion-MNIST and visualize what STDP learns on harder data (1 week)
3. **Implement** the hybrid approach: STDP features + SVM/logistic regression classifier (1 week)
4. **Compare** STDP features vs. PCA/autoencoder features as an ablation study (1 week)
5. **Novel contribution** (pick one): DVS data, STDP variant comparison, R-STDP, or continual learning angle (3-4 weeks)
6. **Write up** with biological plausibility discussion and energy analysis (ongoing)

Total estimated development time: 8-10 weeks of focused work, well within a semester.

The strongest single thesis pitch: **"Biologically Plausible Feature Learning: Evaluating STDP as an Unsupervised Feature Extractor for Visual Recognition in Spiking Neural Networks"** -- with the hybrid approach as the main architecture and one novel angle (DVS data or continual learning or STDP variants comparison) as the original contribution.
