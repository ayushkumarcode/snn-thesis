# SpiNNaker PhD Theses & Scope Comparison Report
*Generated: 5 March 2026*

## Executive Summary

The SpiNNaker project (Manchester, Steve Furber, since ~2006) has produced at least **13 identifiable PhD theses**. They mostly cover chip design, software infrastructure, biological brain simulation, and plasticity. **ZERO Manchester PhDs address audio classification.** Our ESC-50 project's scope — 7 encodings, transfer learning, adversarial robustness, energy analysis, hardware deployment — exceeds most individual SpiNNaker PhD theses, each of which typically addresses 1-2 of these dimensions.

---

## Manchester SpiNNaker PhD Theses

| # | Author | Title | Year | Topic |
|---|--------|-------|------|-------|
| 1 | Xin Jin | Parallel Simulation of Neural Networks on SpiNNaker | 2010 | Neural network simulation, STDP |
| 2 | Alexander Rast | Scalable Event-Driven Modelling Architectures | ~2011 | Event-driven processing, architecture design |
| 3 | Sergio Davies | Learning in Spiking Neural Networks | 2012 | STDP-TTS, SpikeServer, routing |
| 4 | Evangelos Stromatias | Scalability and Robustness of ANNs | 2016 | Deep Belief Networks on SpiNNaker, MNIST 95% |
| 5 | James Knight | Plasticity in Large-Scale Neuromorphic Models | 2016 | Largest plastic network (2×10⁴ neurons, 5.1×10⁷ synapses) |
| 6 | Jonathan Heathcote | Building and Operating Large-Scale SpiNNaker Machines | 2016 | Hardware infrastructure, routing |
| 7 | Qian Liu | Deep Spiking Neural Networks | ~2016-17 | Noisy Softplus, 99.07% MNIST |
| 8 | Andrew Mundy | Real Time Spaun on SpiNNaker | 2017 | 9000x speedup of Spaun brain model |
| 9 | Mantas Mikaitis | Arithmetic Accelerators for SpiNNaker 2 | 2020 | Stochastic rounding, fixed-point |
| 10 | Petrut Bogdan | Structural Plasticity on SpiNNaker | 2020 | Synaptic rewiring, unsupervised motion |
| 11 | Gabriel Fonseca Guerra | Stochastic Processes for Neuromorphic Hardware | 2020 | Constraint satisfaction on SpiNNaker/Loihi |
| 12 | Luca Peres | Parallelisation of Neural Processing | 2022 | First real-time Cortical Microcircuit |
| 13 | Meiling Ward | Modelling Neural Dynamics | ~2022-23 | Complex neuron models, SpiNNaker 2 prototype |

**Research focus breakdown:**
- Infrastructure/architecture: 4 theses
- Biological brain simulation: 4 theses
- Learning and plasticity: 3 theses
- Deep learning/classification: 2 theses
- **Audio classification: ZERO**

### Key Staff (not PhDs, but important):
- **Andrew Rowley** — Senior Research Software Engineer, led SpiNNTools/sPyNNaker
- **Oliver Rhodes** — Lecturer, basal ganglia models (PhD from Imperial)
- **Michael Hopkins** — Research Fellow, numerical precision
- **Luis Plana** — Hardware engineer, spiNNlink FPGA
- **Steve Temple** — Core chip architect (retired ~2017)

---

## SNN Masters Theses (Audio/Classification)

| Author | Topic | Institution | Year |
|--------|-------|-------------|------|
| Daniel Peterson | Bio-inspired learning for audio SNNs | U. Calgary | 2021 |
| Manon Dampfhoffer | Energy-efficient SNNs at edge (SpikGRU, keyword spotting) | U. Grenoble Alpes | 2023 (PhD) |
| Tim Krause | Rate vs temporal coding comparison | Ruhr-U. Bochum | ~2020 |
| Sven Gronauer | Deep spiking ConvNets on SpiNNaker | TU Munich | ~2018 |

**Critical gap confirmed**: No Masters or PhD thesis has deployed an SNN on ESC-50 (or any environmental sound dataset >10 classes) on neuromorphic hardware.

---

## SpiNNaker Applications Beyond Bio Simulation

### Classification/Deep Learning
- **MNIST** (Stromatias, 2015-16): 95%, 0.3W
- **N-MNIST LSM** (Seville, 2022): Liquid State Machine
- **Keyword Spotting** (Dresden, 2022): 91.12% Google Speech Commands on SpiNNaker 2
- **Radar Gesture** (Dresden, 2022): 35ms latency, 3.29μJ/frame on SpiNNaker 2
- **DVS Gesture** (2025): First benchmark on SpiNNaker 2
- **Audio (pure tones)** (Dominguez-Morales, 2016): >85% for tones 130-1397 Hz

### Robotics
- **iCub neurorobotics** (IIT Italy): Event-driven vision + SpiNNaker, 16ms saliency
- **Hippocampal navigation** (2023): Spike-based place cells for robot SLAM

### Other
- Constraint satisfaction, edge detection, cerebellar simulation (SpinnCer), basal ganglia RL

---

## Scope Comparison: Our Project vs PhD/Masters

| Dimension | Our Project | Typical SpiNNaker PhD | Typical SNN Masters |
|-----------|-------------|----------------------|---------------------|
| Dataset | ESC-50 (50 classes) | N/A or MNIST | MNIST or ESC-10 |
| Encodings | 7 | 1-2 | 1-3 |
| Hardware deployment | SpiNNaker (FC2 hybrid + 5-fold) | SpiNNaker (core focus) | Usually none |
| Transfer learning | PANNs+SNN (92.5%) | Not done | Rare |
| Adversarial robustness | FGSM + PGD at 7 epsilon values | Not done | Not done |
| Energy analysis | NeuroBench, AC/MAC | Sometimes | Rarely |
| Surrogate ablation | 8 surrogates | Not done | Not done |
| Continual learning | 5-task sequential + BWT | Not done | Not done |
| ANN baseline | Full matched architecture | Sometimes | Sometimes |
| 5-fold CV | All experiments | Varies | Rare for hardware |

### Assessment
**Our project is at Masters+ scope, approaching early-PhD scope.** Each of the 8-10 experimental threads could be a Masters thesis chapter. Typical SpiNNaker PhDs cover 1-2 dimensions deeply over 3-4 years.

---

## Steve Furber & SpiNNaker 2

- **Steve Furber**: Retired, now Professor Emeritus. Won 2022 Charles Stark Draper Prize.
- **SpiNNaker 1**: Operational but suffered cooling failure Easter 2025 (~80% capacity)
- **SpiNNaker 2**: Major upgrade — 22nm FDSOI, 152 ARM Cortex-M4F cores/chip, 64 MAC accelerators, 2GB DRAM/chip
- **TU Dresden** leads SpiNNaker 2 development (Christian Mayr)
- **SpiNNcloud Systems GmbH**: First commercially available neuromorphic supercomputer (May 2024)
- **Sandia National Labs**: Deployed SpiNNaker2 for AI/national security (June 2025, ~175M neurons)
- Claims **18x energy efficiency** over GPUs

### SpiNNaker 2 vs SpiNNaker 1

| Feature | SpiNNaker 1 | SpiNNaker 2 |
|---------|-------------|-------------|
| Process | 130nm CMOS | 22nm FDSOI |
| Cores/chip | 18 ARM968 | 152 ARM Cortex-M4F |
| SRAM/core | 96KB | 128KB + 19MB total |
| DRAM/chip | 128MB | 2GB |
| ML accelerator | None | 64 MAC units |
| Target scale | 1M cores | 10M cores |

---

## Key Sources
- Manchester Research Explorer (thesis repository)
- [Dominguez-Morales audio](https://link.springer.com/chapter/10.1007/978-3-319-44778-0_6)
- [E-prop keyword spotting SpiNNaker2](https://www.frontiersin.org/articles/10.3389/fnins.2022.1018006)
- [SpiNNaker2 at Sandia](https://www.hpcwire.com/off-the-wire/sandia-deploys-spinnaker2-neuromorphic-system-from-spinncloud/)
- [SpiNNaker overheating](https://www.theregister.com/2025/05/06/spinnaker_overheat/)
- [Larroza et al. 2025](https://arxiv.org/abs/2503.11206)
