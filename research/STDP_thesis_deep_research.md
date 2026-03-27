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
