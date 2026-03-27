# SNN vs ANN energy efficiency -- the full picture

Looked into this properly: energy comparisons, NeuroBench benchmarks, neuromorphic hardware measurements, AC vs MAC costs, spike sparsity thresholds, edge deployments. Consulted 40+ sources.

The energy efficiency narrative for SNNs is way more nuanced than people make it sound. **SNNs are NOT automatically more energy-efficient than ANNs** on conventional digital hardware. It depends on three things: (1) spike sparsity rates, which need to exceed ~92-93% for moderate time windows; (2) the hardware platform -- you need neuromorphic hardware to realize the theoretical gains; and (3) whether memory access and data movement costs are included. When you properly account for those, the bar for SNN energy superiority rises a lot.

The key threshold from Dampfhoffer et al. (2023) and Yan et al. (2024): SNNs need spike rates below roughly 6-8% (sparsity above 92-94%) at time window T=6 to beat quantized ANNs. Most real-world SNN implementations on vision tasks report spike rates of 20-40%, well above this. So the energy claim is questionable for many practical deployments on digital hardware, though neuromorphic hardware changes things significantly.

For my project specifically: my measured 74.16% activation sparsity (NeuroBench) translates to ~25.84% spike rate -- well above the 6-8% threshold for software-level energy superiority. But on neuromorphic hardware with native AC operations, the per-operation cost advantage (0.9 pJ/AC vs 4.6 pJ/MAC) gives a genuine 5.1x per-operation benefit. The honest framing: "SNNs achieve energy efficiency advantages on neuromorphic hardware through sparse AC operations, but require specialized hardware to realize these gains."

---

## 1. Energy comparisons: SNN vs ANN

### 1.1 The core papers

#### Dampfhoffer et al. (2023) -- "Are SNNs Really More Energy-Efficient Than ANNs?"
- IEEE TECI, Vol. 7, pp. 731+, 2023
- Key finding: SNNs with the IF model can compete with efficient ANNs when there is very high spike sparsity, between 0.15 and 1.38 spikes per synapse per inference.
- The main advantage of SNNs on digital hardware comes from **exploiting spike sparsity, NOT from replacing MAC with AC operations**.
- Many studies don't consider memory accesses, which are a huge fraction of energy consumption.
- For T=6 timesteps, SNNs need 92-93% sparsity to match optimized quantized ANNs.

#### Yan, Bai & Wong (2024) -- "Reconsidering the Energy Efficiency of SNNs"
- arXiv:2409.08290
- Establishes a fair baseline by mapping rate-encoded SNNs with T timesteps to functionally equivalent QNNs with ceil(log2(T+1)) bits.
- Sparsity thresholds by time window:

