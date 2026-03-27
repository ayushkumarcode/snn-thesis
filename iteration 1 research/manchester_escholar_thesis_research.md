# Manchester eScholar / Research Explorer Thesis Search

searched the university of manchester research explorer (research.manchester.ac.uk) on 2026-02-24 to see what kind of SNN/neuromorphic theses have been done here before. the database has about 14,367 student theses.

---

## important thing i noticed right away

manchester research explorer basically only hosts PhD theses (and occasionally MPhil / MSc by Research). i couldn't find a single undergrad BSc dissertation in any search. the library guide confirms it's for "postgraduate research theses" only -- PhD, MPhil, MSc by Research.

so everything below is PhD-level unless noted. these are 3-4 years of full-time research, way beyond undergrad scope. but still useful to see what's been done and who supervises what.

---

## neuromorphic / SpiNNaker theses (the core stuff)

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

---

