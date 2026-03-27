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
| Field | Detail |
|-------|--------|
| **Author** | Sergio Davies |
| **Year** | 2012 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Not listed |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/learning-in-spiking-neural-networks |
| **Abstract** | Proposes a novel learning rule based on spike-pair STDP that is less computationally expensive. Addresses SpiNNaker implementation, spike injection via Ethernet, and population-based routing. |
| **What They Built** | Novel STDP learning rule, SpiNNaker implementation, spike injection system, population-based routing |
| **Tools/Frameworks** | SpiNNaker, LIF/HH neuron models, STDP variants |
| **Datasets** | Not specified |
| **Scope** | Very high -- foundational work on learning mechanisms for SpiNNaker |
| **PDF Size** | 14.1 MB |

---

### 1.5 Stochastic Processes For Neuromorphic Hardware
| Field | Detail |
|-------|--------|
| **Author** | Gabriel Fonseca Guerra |
| **Year** | 2020 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Steve Furber (main), David Lester (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/stochastic-processes-for-neuromorphic-hardware |
| **Abstract** | Two contributions: (1) Solving constraint satisfaction problems using SNNs on SpiNNaker AND Loihi chips, achieving comparable performance to state-of-the-art; (2) Implementing ion-channel current models and realistic postsynaptic potentials on SpiNNaker. |
| **What They Built** | Constraint satisfaction solver on neuromorphic hardware; ion-channel current models on SpiNNaker |
| **Tools/Frameworks** | SpiNNaker, Intel Loihi |
| **Datasets** | Constraint satisfaction benchmarks |
| **Scope** | Very high -- cross-platform (SpiNNaker + Loihi) neuromorphic implementations |
| **PDF Size** | 22.5 MB |

---

### 1.6 Parallel Simulation of Neural Networks on SpiNNaker Universal Neuromorphic Hardware
| Field | Detail |
|-------|--------|
| **Author** | Xin Jin |
| **Year** | 2010 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/parallel-simulation-of-neural-networks-on-spinnaker-universal-neu/ |
| **Abstract** | Addresses computational speed challenges in ANN simulation. Proposes parallel processing on SpiNNaker for SNNs with STDP and parallel distributed processing with backpropagation. Demonstrates linear scalability. |
| **What They Built** | Parallel simulation framework for SNNs and MLPs on SpiNNaker |
| **Tools/Frameworks** | SpiNNaker, Izhikevich model, ARM processors |
| **Datasets** | Not specified |
| **Scope** | Very high -- foundational early SpiNNaker thesis |
| **PDF Size** | 6.57 MB |

---

### 1.7 Arithmetic Accelerators for a Digital Neuromorphic Processor
| Field | Detail |
|-------|--------|
| **Author** | Mantas Mikaitis |
| **Year** | 2020 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | David Lester (main), Steve Furber (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/arithmetic-accelerators-for-a-digital-neuromorphic-processor/ |
| **Abstract** | Investigates programmable accelerator for exponential and logarithm functions in SNN models within SpiNNaker2. Explores numerical accuracy of ODE solvers for Izhikevich neuron model. Investigates stochastic rounding methods. |
| **What They Built** | Hardware accelerator designs for SpiNNaker2, numerical analysis of neuron model solvers |
| **Tools/Frameworks** | SpiNNaker2, fixed-point/floating-point arithmetic |
| **Datasets** | Not specified |
| **Scope** | High -- chip-level hardware accelerator design |
| **PDF Size** | 2.8 MB |

---

### 1.8 Building and Operating Large-Scale SpiNNaker Machines
| Field | Detail |
|-------|--------|
| **Author** | Jonathan Heathcote |
| **Year** | 2016 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | James Garside (main), Steve Furber (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/building-and-operating-large-scale-spinnaker-machines |
| **Abstract** | Addresses practical challenges in scaling SpiNNaker to simulate up to 1 billion neurons. Contributions: physical layout scheme for hexagonal torus topologies, improved routing algorithms, placement and routing algorithms that tolerate network faults. Demonstrated on half-million core prototype. |
| **What They Built** | Physical layout schemes, routing algorithms, placement/routing algorithms for SpiNNaker at scale |
| **Tools/Frameworks** | SpiNNaker, simulated annealing |
| **Datasets** | Not specified |
| **Scope** | Very high -- supercomputer-scale engineering |
| **PDF Size** | 8.54 MB |

---

### 1.9 Structural Plasticity on SpiNNaker
| Field | Detail |
|-------|--------|
| **Author** | Petrut Bogdan |
| **Year** | 2019 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Steve Furber (main), David Lester (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/structural-plasticity-on-spinnaker |
| **Abstract** | Implements structural plasticity model on SpiNNaker that operates in real-time alongside STDP. Applications in topographic map generation, unsupervised handwritten digit classification, and motion detection. |
| **What They Built** | Structural plasticity implementation on SpiNNaker; topographic map, digit classification, motion detection demos |
| **Tools/Frameworks** | SpiNNaker, STDP |
| **Datasets** | Handwritten digits (likely MNIST) |
| **Scope** | Very high -- novel plasticity mechanism on neuromorphic hardware |
| **PDF Size** | 47.6 MB |

---

### 1.10 Real Time Spaun on SpiNNaker
| Field | Detail |
|-------|--------|
| **Author** | Andrew Mundy |
| **Year** | 2016 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | James Garside (main), Steve Furber (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/real-time-spaun-on-spinnaker-functional-brain-simulation-on-a-mas |
| **Abstract** | Achieves real-time execution of Spaun (2.5M neuron functional brain model) on SpiNNaker. 9000x speed-up over previously reported results. Only 5% of cores previously needed. Novel routing table optimization. |
| **What They Built** | Real-time Spaun brain model on SpiNNaker; memory/compute reduction; routing table optimization |
| **Tools/Frameworks** | SpiNNaker, Neural Engineering Framework (NEF), Spaun model |
| **Datasets** | Spaun model benchmarks |
| **Scope** | Extremely high -- 9000x speedup of complete brain model |
| **PDF Size** | 4.77 MB |

---

### 1.11 Biologically Inspired Neural Computation
| Field | Detail |
|-------|--------|
| **Author** | Adam Perrett |
| **Year** | 2022 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Steve Furber (main), Oliver Rhodes (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/biologically-inspired-neural-computation |
| **Abstract** | Contrasts biological learning with current ML. Three areas: (1) biologically-inspired visual attention on SpiNNaker; (2) e-prop learning algorithm implementation; (3) gradient-descent-free architecture using dendritic abstractions with neurogenesis, achieving comparable performance to Adam optimizer. |
| **What They Built** | Visual attention model on SpiNNaker, e-prop implementation, neurogenesis-based learning architecture |
| **Tools/Frameworks** | SpiNNaker, iCub robot platform |
| **Datasets** | General benchmarks |
| **Scope** | Very high -- three distinct research contributions |
| **PDF Size** | 20 MB |

