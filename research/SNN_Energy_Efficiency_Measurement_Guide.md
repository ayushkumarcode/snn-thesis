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

- pip install neurobench
- computes Eff_MACs, Eff_ACs, Activation Sparsity, Connection Sparsity, Footprint
- published in Nature Communications, community standard, works with snnTorch
- difficulty: LOW

```python
from neurobench.models import SNNTorchModel
from neurobench.metrics.workload import (
    ActivationSparsity, SynapticOperations, ClassificationAccuracy
)
from neurobench.metrics.static import Footprint, ConnectionSparsity
from neurobench.benchmarks import Benchmark

model = SNNTorchModel(your_trained_net)
static_metrics = [Footprint, ConnectionSparsity]
workload_metrics = [ClassificationAccuracy, ActivationSparsity, SynapticOperations]
benchmark = Benchmark(model, test_loader, preprocessors, postprocessors,
                      [static_metrics, workload_metrics])
results = benchmark.run()
# Output includes: Effective_MACs, Effective_ACs, Dense, ActivationSparsity, etc.
```

### manual spike counting with snnTorch (simple, full control)

if you're already using snnTorch, you can count spikes during inference:

```python
import snntorch as snn
import torch

def count_spikes_per_layer(model, data_loader, num_steps):
    spike_counts = {}
    with torch.no_grad():
        for data, targets in data_loader:
            snn.functional.reset(model)
            for step in range(num_steps):
                spk_out, mem_out = model(data)
                for name, module in model.named_modules():
                    if isinstance(module, (snn.Leaky, snn.Synaptic)):
                        if name not in spike_counts:
                            spike_counts[name] = 0
                        spike_counts[name] += spk_out.sum().item()
    return spike_counts

def estimate_energy(spike_counts, layer_fan_outs, flops_ann,
                    E_AC=0.9e-12, E_MAC=4.6e-12):
    total_sops = sum(
        spike_counts[layer] * layer_fan_outs[layer]
        for layer in spike_counts
    )
    E_SNN = total_sops * E_AC
    E_ANN = flops_ann * E_MAC
    energy_ratio = E_ANN / E_SNN
    return {
        'total_sops': total_sops,
        'E_SNN_joules': E_SNN,
        'E_ANN_joules': E_ANN,
        'energy_ratio': energy_ratio,
    }
```

### advanced tools (for reference)

| Tool | Type | Difficulty | GitHub |
|------|------|-----------|--------|
| **SANA-FE** | Neuromorphic arch simulator | MEDIUM-HIGH | [SLAM-Lab/SANA-FE](https://github.com/SLAM-Lab/SANA-FE) |
| **SpikeSim** | CIM hardware evaluator | MEDIUM-HIGH | [Intelligent-Computing-Lab-Panda/SpikeSim](https://github.com/Intelligent-Computing-Lab-Panda/SpikeSim) |
| **SNN Toolbox** | ANN-to-SNN converter | MEDIUM | [NeuromorphicProcessorProject/snn_toolbox](https://github.com/NeuromorphicProcessorProject/snn_toolbox) |

---

## 5. how to make a fair comparison

this is maybe the most important part. a fair comparison requires:

### controlled variables

| Variable | Must be same? | Notes |
|----------|:-----:|-------|
| Task | Yes | Same dataset, same problem |
| Architecture | Yes (or justified) | Same layers, kernels, channels |
| Accuracy | Yes (within ~1-2%) | Energy at comparable accuracy |
| Data type | Yes | Same input resolution, encoding |
| Training protocol | Document fully | Epochs, LR, augmentation |

### step by step

1. train ANN baseline. record accuracy, FLOPs, param count.
2. train SNN with same architecture (replace ReLU with LIF). record accuracy, spike counts per layer per timestep.
3. ensure comparable accuracy (within 1-2%). if SNN is much lower, the comparison isn't fair.
4. compute energy:
   ```
   E_ANN = FLOPs * E_MAC  (4.6 pJ at 45nm)
   E_SNN = SOP * E_AC     (0.9 pJ at 45nm)
   ```
5. report in a table:

| Model | Accuracy | FLOPs/SOPs | Energy (mJ) | Ratio |
|-------|----------|-----------|-------------|-------|
| ANN (VGG-9) | 93.2% | 606M FLOPs | 2.79 | 1.0x |
| SNN (VGG-9, T=4) | 92.8% | 148M SOPs | 0.13 | 21.2x |
| SNN (VGG-9, T=8) | 93.1% | 312M SOPs | 0.28 | 10.0x |

6. include sparsity analysis: average firing rate per layer, energy vs timesteps, energy-accuracy tradeoff plot.

### things to watch out for

- **don't compare against a non-optimized ANN.** use a well-trained, reasonable baseline.
- **quantized ANNs change the picture.** 8-bit INT MACs cost ~0.2 pJ vs 4.6 pJ for FP32. Shen et al. (CVPR 2024) showed SNNs with T timesteps are equivalent to quantized ANNs with ceil(log2(T+1)) bits.
- **report firing rate honestly.** above ~50%, the energy advantage disappears. need average_spikes_per_neuron < 1 over inference for benefits.
- **state assumptions clearly:** "Energy estimates assume ideal neuromorphic hardware. On conventional GPU/CPU, the SNN would not achieve these savings."

---

## 6. can an undergrad do this without hardware?

### yes

here's why:
1. it's what the majority of published papers do. if it's good enough for CVPR/ICLR/NeurIPS peer review, it's good enough for a thesis.
2. NeuroBench makes it trivial -- pip install, wrap model, call benchmark.run().
3. you're estimating, not measuring. frame it correctly: "We *estimate* energy using the standard synaptic operations methodology."
4. critical analysis of the limitations adds value that many published papers lack.

### what would make it more credible
- use NeuroBench for standardized metrics
- report in clear tables with controlled variables
- include per-layer firing rate analysis
- show energy vs accuracy tradeoff (vary T, threshold)
- acknowledge limitations in a dedicated discussion section
- cite the critical papers (Kundu 2023, Shen 2024) that question SNN efficiency
- compare against a properly trained ANN (not a strawman)

### what NOT to do
- don't claim "our SNN is X times more efficient" without specifying assumptions
- don't ignore timestep overhead
- don't compare against an unoptimized ANN
- don't claim hardware-level savings from software simulation
- don't ignore memory access costs entirely (at least acknowledge them)

---

## the Horowitz 2014 energy table (45nm CMOS)

this is the single most referenced table in the SNN energy literature. you should cite it.

| Operation | Energy (pJ) | Notes |
|-----------|------------|-------|
| 8-bit Integer Add | 0.03 | |
| 32-bit Integer Add | 0.1 | E_AC for integer SNNs |
| 16-bit Float Add | 0.4 | |
| 32-bit Float Add | 0.9 | **E_AC for FP32 SNNs** |
| 8-bit Integer Multiply | 0.2 | |
| 32-bit Integer Multiply | 3.1 | |
| 16-bit Float Multiply | 1.1 | |
| 32-bit Float Multiply | 3.7 | |
| **32-bit FP MAC** | **4.6** | **E_MAC = 3.7 + 0.9** |
| 8KB SRAM Read (64-bit) | 10 | On-chip cache |
| 32KB SRAM Read (64-bit) | 20 | |
| 1MB SRAM Read (64-bit) | 100 | |
| DRAM Read (64-bit) | 1300-2600 | 100-200x SRAM |

a MAC (4.6 pJ) costs ~5.1x more than an AC (0.9 pJ) at FP32. but SRAM access (10-100 pJ) can cost more than either operation. that's why memory-aware analysis matters.

---

## suggested thesis structure for energy analysis section

```
4. Energy Efficiency Analysis

4.1 Methodology
    - Describe operation-counting approach
    - Cite Horowitz 2014, Lemaire et al. 2022
    - State assumptions (45nm, ideal sparse hardware, etc.)

4.2 Experimental Setup
    - Same architecture for ANN and SNN
    - Same dataset and preprocessing
    - Accuracy comparison table
    - Note: using NeuroBench framework

4.3 Results
    - Table: Model | Accuracy | FLOPs/SOPs | Energy Est. | Ratio
    - Per-layer firing rate bar chart
    - Energy vs timestep line plot
    - Energy vs accuracy Pareto plot

4.4 Discussion
    - Compare with published results
    - Acknowledge limitations of software estimation
    - Note: assumes neuromorphic hardware
    - On conventional hardware, SNN overhead exists
    - Reference critical papers (Kundu 2023, Shen 2024)
    - Where does your SNN's sparsity fall vs critical thresholds?

4.5 Summary
    - Key finding and consistency with literature
```

---

## papers to cite in energy analysis

### foundational (must cite)
1. **Horowitz 2014** -- energy per operation constants at 45nm. ISSCC.
2. **Lemaire et al. 2022** -- analytical SNN energy model. [arXiv:2210.13107](https://arxiv.org/abs/2210.13107)
3. **NeuroBench 2025** -- standardized benchmarking. [Nature Communications](https://www.nature.com/articles/s41467-025-56739-4)

### critical/nuanced (should cite for balance)
4. **Kundu et al. 2023** -- hardware perspective on SNN efficiency. [arXiv:2309.03388](https://arxiv.org/abs/2309.03388)
5. **Shen et al. 2024 (CVPR)** -- SNNs equivalent to quantized ANNs. [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Shen_Are_Conventional_SNNs_Really_Efficient_A_Perspective_from_Network_Quantization_CVPR_2024_paper.pdf)
6. **Dampfhoffer et al. 2023** -- hardware-aware study. [HAL archive](https://cea.hal.science/cea-03852141/file/Are_SNNs_Really_More_Energy_Efficient_Than_ANNs__An_In_Depth_Hardware_Aware_Study_versionacceptee.pdf)
7. **Li et al. 2024** -- sparsity thresholds (93-97%). [arXiv:2409.08290](https://arxiv.org/html/2409.08290v1)
8. **Lunghi et al. 2025** -- hardware-aware vs agnostic estimation. [arXiv:2508.19654](https://arxiv.org/abs/2508.19654)

### methodology examples
9. **Yao et al. 2024 (ICLR)** -- Spike-driven Transformer V2. clear per-layer methodology. [arXiv:2404.03663](https://arxiv.org/html/2404.03663v1)
10. **Kundu et al. 2021 (WACV)** -- Spike-Thrift. energy-aware training. [WACV 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Kundu_Spike-Thrift_Towards_Energy-Efficient_Deep_Spiking_Neural_Networks_by_Limiting_Spiking_WACV_2021_paper.pdf)

---

## 13 energy metrics taxonomy (Pereira et al. 2025)

| Metric | Accessible | High Fidelity | Actionable | Trend-Based |
|--------|:----------:|:-------------:|:----------:|:-----------:|
| Parameters | Yes | No | No | No |
| Effective Synaptic Ops | Yes | No | No | Yes |
| Membrane Updates | Yes | No | No | Yes |
| Activation Sparsity | Yes | No | Yes | Yes |
| Memory Footprint | Yes | No | No | No |
| Connection Sparsity | Yes | No | No | No |
| Memory Accesses | Yes | No | No | Yes |
| Training Time | Yes | No | No | Yes |
| Energy per Inference | No | Yes | No | No |
| Energy per Learning | No | Yes | No | No |
| Energy Area FoM | No | Yes | No | No |
| Peak Energy Consumption | No | Yes | No | No |
| Power Density | No | Yes | Yes | No |

there's a gap between accessible metrics (computable in software) and high-fidelity metrics (require hardware). a thesis will use accessible metrics -- that's standard practice. just acknowledge the gap.

Source: [Pereira et al. 2025](https://arxiv.org/html/2506.09599v1)

---

## quick reference

```
Do you have neuromorphic hardware?
  |
  NO (most likely)
