# SNN energy efficiency: how to measure and report it in a thesis

i was wondering whether i could credibly include energy analysis in my thesis without access to neuromorphic hardware. turns out: yes, absolutely. almost every SNN paper claims energy efficiency, but few measure it properly. the core approach used by the vast majority of published papers (CVPR, ICLR, NeurIPS, ECCV) is a theoretical/analytical estimation based on counting synaptic operations and multiplying by known energy-per-operation constants from Horowitz's 2014 ISSCC reference. no hardware needed.

---

## 1. how researchers estimate SNN energy without neuromorphic hardware

there are three tiers, simplest to most complex:

### tier 1: operation-count based (what 90%+ of papers do -- use this)

1. count synaptic operations during inference (how many times a spike causes a weight accumulation)
2. multiply by energy-per-operation constants from known hardware characterizations
3. compare total energy between SNN and ANN on the same task

the key insight: in an ANN, every synapse does a Multiply-Accumulate (MAC) every forward pass. in an SNN, a synapse only does an Accumulate (AC) when it receives a spike. since spikes are sparse and binary, SNNs potentially use fewer and cheaper operations.

the formulas everyone uses:

```
E_ANN = FLOPs_ANN * E_MAC

E_SNN = SOP_SNN * E_AC + FLOPs_non_spiking * E_MAC

where:
  SOP (Synaptic Operations) = sum over all layers of:
    (spike_count_per_neuron * fan_out_connections) * T_timesteps

  E_MAC = 4.6 pJ   (32-bit float multiply-accumulate at 45nm)
  E_AC  = 0.9 pJ   (32-bit float accumulate/addition at 45nm)
```

these energy constants come from Horowitz's 2014 ISSCC keynote "Computing's Energy Problem (and what we can do about it)." most cited source for operation energy costs in the entire neural network efficiency literature.

Source: [Horowitz 2014, ISSCC](https://www.researchgate.net/publication/271463146_11_Computing's_energy_problem_and_what_we_can_do_about_it)

### tier 2: analytical model with memory access costs (more rigorous)

Lemaire et al. (2022) proposed a model accounting for three costs:

1. synaptic operations (AC for SNN, MAC for ANN)
2. memory accesses (reading weights, reading/writing membrane potentials)
3. addressing operations (indexing into memory)

each spike leads to: 2 reads (weights + current membrane potential) + 1 write (updated potential).

memory access energy at 45nm (Horowitz 2014):
- 8KB SRAM: ~10 pJ
- 1MB SRAM: ~100 pJ
- DRAM: ~2000 pJ

this is more accurate because memory access often dominates energy over arithmetic. but it needs more assumptions about memory hierarchy.

Source: [Lemaire et al., ICONIP 2022](https://arxiv.org/abs/2210.13107)

### tier 3: hardware simulation (most accurate, most complex)

tools like SANA-FE and SpikeSim simulate actual neuromorphic hardware behavior -- spike routing, network-on-chip communication, pipeline stages. most realistic but significant setup.

---

## 2. metrics to report

### primary metrics (you should report these)

| Metric | What it is | How to compute |
|--------|-----------|----------------|
| **Synaptic Operations (SOPs)** | Total spike-driven ops during inference | Sum: spike_count * fan_out * timesteps per layer |
| **Effective ACs** | AC ops excluding zero activations | Count only non-zero (spiking) activations |
| **Effective MACs** | MAC ops for non-spiking layers | Standard FLOP count for non-spiking layers |
| **Firing Rate / Spike Sparsity** | Avg fraction of neurons spiking per timestep | total_spikes / (total_neurons * timesteps) |
| **Energy per Inference** | Estimated joules for one forward pass | E = SOP * E_AC + FLOPs_nonspiking * E_MAC |

### secondary metrics (good to include)

| Metric | What it is | How to compute |
|--------|-----------|----------------|
| **Activation Sparsity** | Fraction of zero activations | 1 - firing_rate |
| **Connection Sparsity** | Fraction of zero weights (if pruned) | count_zero / total_weights |
| **Memory Footprint** | Model size in bytes | sum of parameter sizes |
| **Timesteps (T)** | Simulation steps | Hyperparameter |
| **Energy Ratio** | E_SNN / E_ANN | Ratio on same task |

### advanced metrics from recent literature

- **EMAC (Equivalent MAC):** normalizes all SNN ops into MAC-equivalents. hardware-agnostic. ([Source](https://arxiv.org/abs/2508.19654))
- **Bit Budget:** refines SOPs by weight bit-width and spike patterns. Shen et al. CVPR 2024. ([Source](https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Are_Conventional_SNNs_Really_Efficient_A_Perspective_from_Network_Quantization_CVPR_2024_paper.pdf))
- **BES / EAS Composite Metrics:** jointly account for normalized accuracy and SOP-based energy proxies.

---

## 3. is there a standard methodology?

### the de facto standard (what most papers do)

```
1. Train SNN and ANN on same task with same architecture
2. Count SOPs for SNN (spike counts * fan-out per layer)
3. Count FLOPs for ANN (standard)
4. Multiply: E_SNN = SOP * 0.9pJ, E_ANN = FLOPs * 4.6pJ
5. Report ratio: "X times more efficient"
```

example from Spike-driven Transformer V2 (ICLR 2024): ANN energy = FLOPs * E_MAC, SNN energy = FLOPs * E_AC * firing_rate per layer.

Source: [Spike-driven Transformer V2, ICLR 2024](https://arxiv.org/html/2404.03663v1)

### the NeuroBench standard (most rigorous)

NeuroBench is the first community-wide attempt at standardizing benchmarks. published in Nature Communications (2025). provides standardized metrics (Eff_MACs, Eff_ACs, Activation Sparsity), open-source Python framework, and multiple benchmark tasks.

Source: [NeuroBench, Nature Communications 2025](https://www.nature.com/articles/s41467-025-56739-4)

### known limitations and criticisms

several papers argue the standard approach is too optimistic:

1. **ignores memory access costs:** "Most SNN works only consider counting of additions to evaluate energy consumption, neglecting other overheads such as memory accesses and data movement." -- [Shen et al., 2024](https://arxiv.org/html/2409.08290v1)

2. **assumes ideal sparse hardware:** the 0.9 pJ/AC cost assumes neuromorphic hardware that skips zero-activation synapses. on GPUs/CPUs, SNNs are often slower and more energy-hungry than ANNs.

3. **ignores timestep overhead:** T timesteps means T forward passes, each with memory reads/writes.

4. **critical sparsity thresholds:** for VGG16 with T=6, sparsity must exceed 93%. with T>16, must exceed 97%. ([Source](https://arxiv.org/html/2409.08290v1))

### what i'd recommend for a thesis

report both the standard SOP-based estimate AND acknowledge limitations. this shows critical thinking. good approach:
1. report standard E_SNN = SOP * E_AC (tier 1)
2. maybe add Lemaire-style estimate with memory costs (tier 2)
3. include discussion noting these are theoretical estimates assuming ideal neuromorphic hardware
4. note that on conventional hardware, the SNN would likely be slower

---

## 4. tools and libraries for energy estimation

### NeuroBench (highest recommendation)

