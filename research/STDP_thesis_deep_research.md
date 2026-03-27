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
- analyze selectivity of individual neurons to specific classes
- compare STDP feature quality to unsupervised ANN methods (PCA, autoencoders, k-means)
- measure energy efficiency (spike counts, synaptic operations)

### 4.4 Implementation Notes

using BindsNET, feature extraction looks roughly like:

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

## 5. Is STDP a Good Thesis Topic or Is It Old News?

### 5.1 it's alive and well

**evidence that STDP is NOT old news:**

| Signal | Evidence | Year |
|---|---|---|
| **Top venue publication** | NCG with Supervised STDP accepted at **NeurIPS 2024** | 2024 |
| **Top venue publication** | Dendritic Localized Learning (STDP-adjacent) at **ICML 2025** | 2025 |
| **Major review** | Three-factor learning in SNNs review in **Patterns (Cell Press)** | Nov 2025 |
| **Major review** | Modulated STDP review in **Neurocomputing** | Feb 2025 |
| **Nature publication** | Unsupervised post-training learning with triplet STDP in **Scientific Reports** | May 2025 |
| **Nature publication** | TEXEL neuromorphic chip with on-chip STDP in **Nature Communications** | 2025 |
| **Hardware integration** | Intel Loihi 2 natively supports STDP and three-factor rules | Ongoing |
| **Active GitHub repos** | SpikeNN (NCG code), ngc-learn v3, BindsNET all maintained | 2024-2025 |
| **New frameworks** | Inferno (Sept 2024) -- new SNN framework with extensible plasticity | 2024 |

### 5.2 Why STDP Has Renewed Relevance

1. **Energy crisis in AI:** training GPT-4-class models costs millions in electricity. STDP on neuromorphic hardware is 3-5 orders of magnitude less energy per synaptic operation (20-50 pJ vs microjoules on GPUs).

2. **On-device / edge learning:** backprop requires storing full computation graphs and backward passes -- impossible on tiny edge devices. STDP is purely local: each synapse only needs info from its two connected neurons.

3. **Privacy-preserving AI:** STDP enables on-device learning without sending data to the cloud, which matters increasingly under GDPR.

4. **Neuromorphic hardware maturation:** Loihi 2, SpiNNaker2, BrainScaleS-2, memristive chips all implement STDP natively. hardware exists, now researchers need algorithms.

5. **Biological understanding:** neuroscience is discovering increasingly complex STDP variants (dendritic STDP, voltage-dependent plasticity, heterosynaptic plasticity). computational models are needed.

### 5.3 What Would Make It Old News

the thesis should NOT just replicate Diehl & Cook (2015) on MNIST. that is indeed a 10-year-old result. needs a novel angle (see section 6).

### 5.4 The Narrative Advantage

"biologically plausible learning" is actually a great thesis narrative because:
- connects to neuroscience (interdisciplinary appeal)
- connects to energy-efficient AI (practical relevance)
- connects to neuromorphic hardware (cutting-edge tech)
- clear research question: "how well can the brain's learning rule work for ML tasks?"
- produces visually compelling results (learned filters look like Gabor filters or digit templates)
- examiners tend to find the biological angle intellectually interesting

---

## 6. What Would Make an STDP Project Interesting in 2026?

### 6.1 Project Ideas (ranked)

#### Tier 1: Strongest Novel Angles

**Idea A: STDP on Event-Camera (DVS) Data -- "Learning the Way the Brain Sees"**
- **Why novel:** DVS cameras produce async spike-like events -- natural match for STDP's temporal learning rule. most DVS classification uses surrogate gradients, not STDP. there's a gap.
- **What to do:** train STDP-based CSNN on N-MNIST or DVS128 Gesture using BindsNET or SpykeTorch. compare STDP-learned features vs random features vs surrogate-gradient features.
- **Datasets:** N-MNIST, DVS128 Gesture (11 classes), CIFAR10-DVS
- **Expected results:** 90-95% on N-MNIST, 80-90% on DVS128 Gesture
- **Why it works as a thesis:** natural fit between data modality and learning rule, tells a coherent biological story
- **Feasibility:** HIGH -- Tonic handles data loading, BindsNET/SpykeTorch handle STDP

**Idea B: Three-Factor Learning (Reward-Modulated STDP) for RL**
- **Why novel:** standard STDP is unsupervised. adding a dopamine-like reward signal creates three-factor learning: pre-synaptic x post-synaptic x reward. this is how the brain supposedly does RL. a 2025 Patterns review calls this "a crucial extension of traditional STDP."
- **What to do:** implement R-STDP using BindsNET's MSTDP or MSTDPET rules. train an SNN agent on a simple RL task (CartPole, custom maze). compare vs standard DQN.
- **Expected results:** R-STDP should solve simple tasks. the interesting bit is energy efficiency and bio-plausibility comparison.
- **Why it works:** connects neuroscience (dopamine), ML (RL), and neuromorphic computing in one project
- **Feasibility:** MEDIUM-HIGH -- BindsNET has R-STDP built in, RL envs via OpenAI Gym

**Idea C: STDP for Continual/Lifelong Learning -- "Learning Without Forgetting"**
- **Why novel:** catastrophic forgetting is a major unsolved problem. STDP's local nature means it only modifies synapses relevant to current inputs, potentially preserving old knowledge. a 2024 paper showed wake-sleep R-STDP networks can avoid forgetting.
- **What to do:** train STDP on digits 0-4, then 5-9. measure how much 0-4 accuracy degrades. compare vs standard ANN (which will catastrophically forget). implement sleep/replay.
- **Expected results:** STDP should show less forgetting than naive ANNs, but still some. the analysis is where thesis value lies.
- **Feasibility:** MEDIUM -- requires careful experimental design

#### Tier 2: Strong Angles

**Idea D: Comparing STDP Variants**
- many STDP variants now exist (pair-based, triplet, voltage-dependent, R-STDP, symmetric, S2-STDP). no undergrad-level comparison exists.
- implement 3-4 variants in BindsNET/Brian2. train on MNIST and Fashion-MNIST. compare accuracy, convergence, energy, feature quality, bio-plausibility.
- **Feasibility:** HIGH -- mostly parameter/rule changes in existing code

**Idea E: STDP Features vs Unsupervised ANN Methods**
- direct comparison: STDP (brain's rule) vs modern unsupervised methods (autoencoders, contrastive learning, k-means) as feature extractors.
- extract features with each, classify with same SVM. which features are best? analyze quality, bio-plausibility, compute cost.
- **Feasibility:** HIGH -- all methods have standard implementations

**Idea F: STDP + Audio/Speech Recognition**
- audio is inherently temporal -- natural fit for STDP. recent work got 93.3% on Spoken-MNIST and 88.1% on SHD.
- apply STDP feature extraction to audio spectrograms or spike-encoded audio. compare with standard audio ML pipelines.
- **Feasibility:** MEDIUM -- audio spike encoding needs more setup

#### Tier 3: Solid But More Standard

**Idea G: Replicating and Extending Diehl & Cook (2015)**
- replicate the 95% MNIST result in BindsNET, then extend with: conv topology, more neurons, different datasets (Fashion-MNIST, EMNIST), supervised readout
- **Feasibility:** VERY HIGH -- most straightforward project

### 6.2 Thesis Title Ideas

- "Biologically Plausible Feature Learning: A Comparative Study of STDP Variants for Visual Pattern Recognition"
- "Learning Without Labels: STDP-Based Unsupervised Feature Extraction in Spiking Neural Networks"
- "From Spikes to Decisions: Hybrid STDP Feature Extraction with Supervised Classification in Spiking Neural Networks"
- "Brain-Inspired Learning for Event-Driven Vision: STDP on Dynamic Vision Sensor Data"
- "Three-Factor Learning in Spiking Neural Networks: Reward-Modulated STDP for Reinforcement Learning Tasks"
- "Can the Brain's Learning Rule Prevent Forgetting? STDP for Continual Learning in Spiking Neural Networks"

---

## 7. STDP Variants from Recent Papers

### 7.1 Variant Table

| Variant | Key Innovation | Reference | Year | Maturity |
|---|---|---|---|---|
| **Standard Pair-based STDP** | Classical pre-post / post-pre timing rule | Bi & Poo (1998) | 1998 | Foundational |
| **Triplet STDP** | Considers triplets of spikes for richer dynamics | Pfister & Gerstner (2006) | 2006 | Mature |
| **R-STDP (Reward-Modulated)** | Multiplies STDP update by global reward/dopamine signal | Izhikevich (2007), Fremaux & Gerstner (2016) | 2007/2016 | Mature |
| **Voltage-Dependent Plasticity (VDSP)** | Updates based on membrane potential rather than spike timing | Clopath et al. (2010) | 2010 | Moderate |
| **Symmetric STDP** | LTP for both pre-before-post and post-before-pre timing | Hao & Huang (2019) | 2019 | Moderate |
| **SSTDP (Supervised)** | Bridges backprop and STDP using spatial + temporal info | Mirsadeghi et al. (2021) | 2021 | Moderate |
| **S2-STDP (Stabilized Supervised)** | Dynamic target timestamps; alternates firing between target and non-target times | Goupy et al. (2024) | 2024 | Recent |
| **NCG (Neuronal Competition Groups)** | Architecture with intra-class WTA and two-compartment thresholds | Goupy et al. (NeurIPS 2024) | 2024 | State-of-art |
| **SADP (Spike Agreement Dependent Plasticity)** | Replaces pairwise timing with population-level agreement metrics | arXiv, Jan 2026 | 2026 | Cutting-edge |
| **Dendritic Localized Learning (DLL)** | Three-compartment neuron with local error computation | ICML 2025 | 2025 | Cutting-edge |
| **Meta-Learning R-STDP** | R-STDP with hippocampus/PFC-inspired meta-learning | Neurocomputing, Oct 2024 | 2024 | Recent |
| **Wake-Sleep R-STDP** | R-STDP during "day" + generative replay during "night" for continual learning | 2024 | 2024 | Recent |
| **Heterogeneous STDP** | Different STDP rules at different synapses in the same network | Advanced Materials, 2025 | 2025 | Emerging |
| **Forecast-based STDP** | Predictive coding version -- learns to predict future spikes | Nature Communications, 2023 | 2023 | Moderate |

### 7.2 Most Relevant for a Thesis

**S2-STDP + NCG (NeurIPS 2024)** is the most impactful recent contribution:
1. S2-STDP dynamically computes target timestamps (Ttarget, Tnon-target) per sample based on average firing time
2. target neurons learn to fire before the mean; non-target neurons after
3. NCG groups multiple neurons per class with intra-class competition
4. two-compartment thresholds regulate competition
5. code: https://github.com/ggoupy/SpikeNN

**Three-Factor Learning Rules (2025 Patterns/Cell Press review)**:
formal three-factor rule: Delta_w = M * F(pre, post), where M is a neuromodulatory signal (reward, error, attention) and F is Hebbian/STDP-like. variants include:
- **R-max:** maximal for pre-before-post, modulated by reward minus baseline
- **R-STDP:** bi-phasic coincidence window, modulated by success signal
- **TD-STDP:** modulated by temporal-difference error (for RL)
- **e-prop:** eligibility propagation with eligibility traces (biologically plausible gradient approximation)

---

## 8. How Does STDP Scale Beyond MNIST?

### 8.1 Results on Harder Datasets

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
| **SHD** | 88.1% | SOM-Associated-SNN with STDP | ~95%+ | ~7 pp |

### 8.2 The Scaling Problem -- Being Honest

the numbers tell a pretty clear story: **STDP scales poorly to complex datasets as the sole learning mechanism.** the gap between STDP and surrogate-gradient approaches widens dramatically:
- MNIST: gap ~0.6 pp (negligible)
- Fashion-MNIST: gap ~4-5 pp (noticeable)
- CIFAR-10: gap ~16-29 pp (significant)
- CIFAR-100: gap ~25-42 pp (very large)

### 8.3 Why the Gap Exists

1. **No global error signal:** STDP only sees local spike timing; can't propagate errors backwards
2. **Feature hierarchy problem:** deep hierarchical features need coordinated learning across layers, which pure STDP can't do
3. **Curse of unsupervised learning:** without labels, STDP learns visually salient features, not necessarily discriminative ones
4. **Convergence instability:** can cause weight explosion or death in deep networks without careful homeostatic mechanisms

### 8.4 How to Frame This in a Thesis

do NOT frame it as "STDP vs backprop" -- STDP will lose on accuracy. instead:
- **"What can local, biologically plausible learning achieve?"** -- legitimate scientific question
- **"STDP as an efficient feature extractor"** -- compare STDP features to other unsupervised methods, not supervised ones
- **"Energy-accuracy trade-off"** -- STDP may get lower accuracy but with orders of magnitude less energy. quantify it.
- **"Biological plausibility"** -- rate the bio-plausibility of different approaches. STDP wins here.
- **"Hybrid approaches"** -- show that combining STDP features + simple classifier bridges much of the gap

---

## 9. Key Research Groups

| Person/Group | Affiliation | Contribution | Key Papers |
|---|---|---|---|
| **Peter Diehl & Matthew Cook** | ETH Zurich / INI | Foundational STDP-MNIST paper (2015) | Unsupervised learning of digit recognition using STDP |
| **Saeed Reza Kheradpisheh** | University of Tehran | Deep CSNN with STDP for object recognition | STDP-based spiking deep CNNs (2018) |
| **Milad Mozafari** | University of Tehran | SpykeTorch framework, R-STDP | SpykeTorch (2019), First-spike categorization (2018) |
| **Gaspard Goupy et al.** | University of Lille / Fox team | S2-STDP, PCN, NCG architecture | NeurIPS 2024 NCG paper |
| **Timothee Masquelier** | CNRS CerCo, Toulouse | Unsupervised visual feature learning with STDP | PLOS Comp Bio (2007) |
| **Wulfram Gerstner** | EPFL | Three-factor learning theory, neuromodulated STDP | Fremaux & Gerstner (2016) review |
| **Alexander Ororbia** | RIT | ngc-learn framework | ngc-learn docs + papers |
| **Hananel Hazan & Daniel Saunders** | UMass Amherst / BINDS Lab | BindsNET framework | BindsNET paper (2018) |
| **Mike Davies** | Intel Labs | Loihi processor with on-chip STDP | Loihi papers |

---

## 10. Thesis Structure (if i went with this)

for a thesis framed as "biologically plausible feature learning with STDP":

```
Chapter 1: Introduction
  - Motivation: energy crisis in AI, biological inspiration
  - Research question: "How effective is the brain's STDP learning rule
    for unsupervised feature extraction in visual recognition tasks?"
  - Contributions

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
  - Evaluation metrics

Chapter 4: Experiments and Results
  - Experiment 1: STDP feature learning on MNIST (baseline replication)
  - Experiment 2: STDP vs. unsupervised ANN feature extractors
  - Experiment 3: Hybrid STDP + classifier on Fashion-MNIST
  - Experiment 4: [Novel angle]
  - Experiment 5: Energy efficiency analysis

Chapter 5: Analysis and Discussion
  - What did STDP learn? (weight viz, selectivity analysis)
  - Where does STDP succeed and fail?
  - Biological plausibility assessment
  - Energy-accuracy trade-off

Chapter 6: Conclusion and Future Work
```

---

## 11. Gaps and Confidence

### 11.1 Info Gaps

| Gap | Why | Impact |
|---|---|---|
| Exact S2-STDP + PCN accuracy per dataset | Tables in Frontiers paper not fully extractable | Low -- NCG paper has SOTA numbers |
| BindsNET exact release date/version | Didn't deep dive GitHub releases | Low -- framework works regardless |
| Brian2 vs BindsNET performance benchmarks | No direct comparison paper found | Medium -- would help framework selection |
| STDP on ImageNet | Doesn't seem to exist -- probably too expensive | Low -- not relevant for undergrad thesis |

### 11.2 Confidence

| Finding | Confidence | Basis |
|---|---|---|
| STDP gets ~95% on MNIST (Diehl & Cook) | VERY HIGH | Original paper, 1000+ citations, replicated many times |
| NCG gets 98.92% on MNIST with STDP features | HIGH | NeurIPS 2024, code available |
| STDP scales poorly to CIFAR-10 (66% with STDP-CSNN) | HIGH | Multiple sources, NCG paper confirms |
| BindsNET is the best framework for ML STDP | HIGH | Multiple comparison sources |
| STDP is still active research in 2025-2026 | VERY HIGH | NeurIPS 2024, ICML 2025, multiple 2025 reviews |
| Three-factor learning is the most promising STDP extension | HIGH | 2025 Cell Press review, growing publications |
| STDP on DVS data is underexplored | MEDIUM-HIGH | Limited STDP-specific DVS papers, most DVS work uses surrogate gradients |

---

## 12. References

### Foundational
- [Diehl & Cook (2015) -- Unsupervised learning of digit recognition using STDP](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2015.00099/full)
- [Masquelier & Thorpe (2007) -- Unsupervised learning of visual features through STDP](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.0030031)
- [Kheradpisheh et al. (2018) -- STDP-based spiking deep CNNs](https://www.sciencedirect.com/science/article/abs/pii/S0893608017302903)
