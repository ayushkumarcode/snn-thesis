# SpiNNaker PhD Theses & Scope Comparison

looked into all the PhD and Masters theses that have come out of the SpiNNaker project at Manchester (Steve Furber's group, running since ~2006). found at least **13 identifiable PhD theses.** they mostly cover chip design, software infrastructure, biological brain simulation, and plasticity. **none of them do audio classification.** our ESC-50 project's scope -- 7 encodings, transfer learning, adversarial robustness, energy analysis, hardware deployment -- actually exceeds most individual SpiNNaker PhDs, which typically focus on 1-2 of these dimensions.

---

## Manchester SpiNNaker PhD Theses

| # | Author | Title | Year | Topic |
|---|--------|-------|------|-------|
| 1 | Xin Jin | Parallel Simulation of Neural Networks on SpiNNaker | 2010 | Neural network simulation, STDP |
| 2 | Alexander Rast | Scalable Event-Driven Modelling Architectures | ~2011 | Event-driven processing, architecture design |
| 3 | Sergio Davies | Learning in Spiking Neural Networks | 2012 | STDP-TTS, SpikeServer, routing |
| 4 | Evangelos Stromatias | Scalability and Robustness of ANNs | 2016 | Deep Belief Networks on SpiNNaker, MNIST 95% |
| 5 | James Knight | Plasticity in Large-Scale Neuromorphic Models | 2016 | Largest plastic network (2x10^4 neurons, 5.1x10^7 synapses) |
| 6 | Jonathan Heathcote | Building and Operating Large-Scale SpiNNaker Machines | 2016 | Hardware infrastructure, routing |
| 7 | Qian Liu | Deep Spiking Neural Networks | ~2016-17 | Noisy Softplus, 99.07% MNIST |
| 8 | Andrew Mundy | Real Time Spaun on SpiNNaker | 2017 | 9000x speedup of Spaun brain model |
| 9 | Mantas Mikaitis | Arithmetic Accelerators for SpiNNaker 2 | 2020 | Stochastic rounding, fixed-point |
| 10 | Petrut Bogdan | Structural Plasticity on SpiNNaker | 2020 | Synaptic rewiring, unsupervised motion |
| 11 | Gabriel Fonseca Guerra | Stochastic Processes for Neuromorphic Hardware | 2020 | Constraint satisfaction on SpiNNaker/Loihi |
| 12 | Luca Peres | Parallelisation of Neural Processing | 2022 | First real-time Cortical Microcircuit |
| 13 | Meiling Ward | Modelling Neural Dynamics | ~2022-23 | Complex neuron models, SpiNNaker 2 prototype |

research focus breakdown:
- infrastructure/architecture: 4 theses
- biological brain simulation: 4 theses
- learning and plasticity: 3 theses
- deep learning/classification: 2 theses
- **audio classification: ZERO**

### key staff (not PhDs but worth noting):
- **Andrew Rowley** -- Senior Research Software Engineer, led SpiNNTools/sPyNNaker
- **Oliver Rhodes** -- Lecturer, basal ganglia models (PhD from Imperial)
- **Michael Hopkins** -- Research Fellow, numerical precision
- **Luis Plana** -- Hardware engineer, spiNNlink FPGA
- **Steve Temple** -- Core chip architect (retired ~2017)

---

## SNN Masters Theses (Audio/Classification)

| Author | Topic | Institution | Year |
|--------|-------|-------------|------|
| Daniel Peterson | Bio-inspired learning for audio SNNs | U. Calgary | 2021 |
| Manon Dampfhoffer | Energy-efficient SNNs at edge (SpikGRU, keyword spotting) | U. Grenoble Alpes | 2023 (PhD) |
| Tim Krause | Rate vs temporal coding comparison | Ruhr-U. Bochum | ~2020 |
| Sven Gronauer | Deep spiking ConvNets on SpiNNaker | TU Munich | ~2018 |

confirmed: no Masters or PhD thesis has deployed an SNN on ESC-50 (or any environmental sound dataset >10 classes) on neuromorphic hardware.

---

## SpiNNaker Applications Beyond Bio Simulation

### classification/deep learning
