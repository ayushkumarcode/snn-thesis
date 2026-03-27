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

