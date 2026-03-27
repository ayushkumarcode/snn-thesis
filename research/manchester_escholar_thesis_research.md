# Manchester eScholar / Research Explorer: Thesis Search Results
## Neuromorphic Computing and Adjacent Topics

**Date of Research:** 2026-02-24
**Source:** University of Manchester Research Explorer (research.manchester.ac.uk)
**Total Theses in Database:** ~14,367 student theses

---

## CRITICAL FINDING: Degree Level Distinction

**Manchester Research Explorer almost exclusively hosts PhD theses (and occasionally MPhil / MSc by Research).** No undergraduate (BSc) dissertations were found in any search. The university library guide confirms that the system catalogues "postgraduate research theses" -- meaning PhD, MPhil, and MSc by Research only. Undergraduate dissertations at Manchester are NOT deposited in this system.

This means:
- All theses below are **PhD-level** unless otherwise noted
- The 2 exceptions found are: 1 MPhil and 1 MSc by Research
- These are **NOT representative of undergraduate scope** -- they represent 3-4 years of full-time research
- For calibrating undergraduate (final year project) scope, these are dramatically more ambitious than what an undergrad would produce

---

## SECTION 1: NEUROMORPHIC COMPUTING / SpiNNaker THESES (Core Domain)

### 1.1 Modelling Neural Dynamics On Neuromorphic Hardware
| Field | Detail |
|-------|--------|
| **Author** | Mollie Ward |
| **Year** | 2024 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Oliver Rhodes (main), James Garside (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/modelling-neural-dynamics-on-neuromorphic-hardware/ |
| **Abstract** | Explores implementing complex, biologically accurate neuron models (Hodgkin-Huxley, two-compartment dendritic model) on SpiNNaker and SpiNNaker2 systems. Demonstrates excellent agreement with reference models. Shows that a two-compartment model can decrease overall energy consumption compared to LIF-based SNNs. |
| **What They Built** | Implementations of HH and two-compartment neuron models on SpiNNaker/SpiNNaker2; benchmarked against NEURON simulation environment |
| **Tools/Frameworks** | SpiNNaker, SpiNNaker2 prototype, NEURON simulation environment, SNNs |
| **Datasets** | None specific -- model benchmarking |
| **Scope** | Very high -- hardware-level neuroscience modeling across two generations of neuromorphic chip |
| **PDF Size** | 16.6 MB |

---

### 1.2 Deep Spiking Neural Networks
| Field | Detail |
|-------|--------|
| **Author** | Qian Liu |
| **Year** | 2018 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Steve Furber (main), David Lester (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/deep-spiking-neural-networks |
| **Abstract** | Proposes Noisy Softplus (NSP) activation function to model biologically-plausible spiking neurons. Introduces Parametric Activation Function (PAF) to map ANN values to physical SNN units. Achieves 99.07% accuracy on MNIST using deep spiking CNNs. Develops spike-based rate multiplication for online training with STDP. |
| **What They Built** | Novel activation functions, ANN-to-SNN conversion pipeline, spiking autoencoders and RBMs |
| **Tools/Frameworks** | Custom SNN framework, STDP, Spiking Autoencoders, Restricted Boltzmann Machines |
| **Datasets** | MNIST |
| **Scope** | Very high -- novel mathematical formulation + implementation + benchmarking |
| **PDF Size** | 15.1 MB |

---

### 1.3 Parallelisation of Neural Processing on Neuromorphic Hardware
| Field | Detail |
|-------|--------|
| **Author** | Luca Peres |
| **Year** | 2022 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Steve Furber (main), Oliver Rhodes (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/parallelisation-of-neural-processing-on-neuromorphic-hardware |
| **Abstract** | Investigates parallelisation strategies for real-time SNN simulations on SpiNNaker. Achieved the world's first real-time simulation of the Cortical Microcircuit model (20x better than previous). Developed partitioning approaches demonstrating up to 9x higher throughput. |
| **What They Built** | Novel parallelisation strategies, real-time Cortical Microcircuit simulation, partitioning algorithms |
| **Tools/Frameworks** | SpiNNaker |
| **Datasets** | Cortical Microcircuit benchmark model |
| **Scope** | Very high -- world-first real-time simulation achievement |
| **PDF Size** | 15.4 MB |

---

### 1.4 Learning in Spiking Neural Networks
