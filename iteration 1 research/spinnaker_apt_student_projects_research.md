# SpiNNaker / APT Group Student Projects Research

looked into student projects (PhD, MSc, undergrad) from the APT (Advanced Processor Technologies) group at Manchester on 2026-02-24, focusing on SpiNNaker neuromorphic computing stuff.

found 14 PhD theses directly related to SpiNNaker/neuromorphic computing, 1 confirmed MSc dissertation with code on GitHub, and several external student projects (from TUM and other places) that used SpiNNaker hardware. unfortunately, undergrad/3rd-year projects from Manchester are not publicly accessible -- the project listing system (`studentnet.cs.manchester.ac.uk`) requires Manchester authentication. also the old APT website (`apt.cs.manchester.ac.uk`) has been mostly redirected to the main CS department page, so historical thesis listings are lost. the good news is all PhD theses listed below have full-text PDFs freely available through Manchester Research Explorer.

---

## 1. PhD theses from the SpiNNaker/APT group

### 1.1 Xin Jin -- Parallel Simulation of Neural Networks on SpiNNaker (2010)

| Field | Details |
|-------|---------|
| **Title** | Parallel Simulation of Neural Networks on SpiNNaker Universal Neuromorphic Hardware |
| **Author** | Xin Jin |
| **Year** | June 2010 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | Not listed (likely Steve Furber) |
| **Abstract** | Investigated efficient modelling schemes for SpiNNaker considering communication, processing, and storage constraints across spiking neural networks with STDP and parallel distributed processing models with backpropagation. Demonstrated feasibility and linear scalability. |
| **Keywords** | PDP, STDP, Backpropagation, MLP, Real-time, Parallel simulation, Izhikevich, Spiking neural network, SpiNNaker, ARM |
| **Tools** | SpiNNaker hardware, custom C on ARM968 |
| **PDF** | Available (6.57 MB) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/parallel-simulation-of-neural-networks-on-spinnaker-universal-neu/ |

### 1.2 M.M. Khan -- Configuring a Massively Parallel CMP System (2009)

| Field | Details |
|-------|---------|
| **Title** | Configuring a Massively Parallel CMP System for Real Time Neural Applications |
| **Author** | M.M. Khan |
| **Year** | 2009 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | Steve Furber (likely) |
| **Abstract** | Configuration and mapping of neural networks onto the SpiNNaker massively parallel chip multiprocessor system. |
| **PDF** | Available at https://apt.cs.manchester.ac.uk/ftp/pub/amulet/theses/mmkhan09_phd.pdf |

### 1.3 Alexander Rast -- Scalable Event-Driven Modelling Architectures (2011)

| Field | Details |
|-------|---------|
| **Title** | Scalable Event-Driven Modelling Architectures for Neuromimetic Hardware |
| **Author** | Alexander D. Rast |
| **Year** | January 2011 |
| **Degree** | PhD, School of Computer Science |
| **Supervisors** | Steve Furber (supervisor), James Garside (advisor) |
| **Abstract** | Developed a library of predesigned event-driven neural components for SpiNNaker. Addressed burstiness, scalability, and asynchronous event-driven models. |
| **PDF** | Available at https://apt.cs.manchester.ac.uk/ftp/pub/apt/theses/Rast11_phd.pdf |
| **URL** | https://www.escholar.manchester.ac.uk/uk-ac-man-scw:111900 |

### 1.4 Eustace Painkras -- A Chip Multiprocessor for a Large-scale Neural Simulator (2012)

| Field | Details |
|-------|---------|
| **Title** | A Chip Multiprocessor for a Large-scale Neural Simulator |
| **Author** | Eustace Painkras |
| **Year** | December 2012 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | Steve Furber (likely) |
| **Abstract** | Design of the SpiNNaker CMP chip -- many simple power-efficient ARM processors with small local memories, asynchronous networks-on-chip, and GALS architecture. Demonstrated successful neural simulation on 48-chip PCBs. |
| **URL** | https://www.escholar.manchester.ac.uk/uk-ac-man-scw:198344 |

### 1.5 Sergio Davies -- Learning in Spiking Neural Networks (2012)

| Field | Details |
|-------|---------|
| **Title** | Learning in Spiking Neural Networks |
| **Author** | Sergio Davies |
| **Year** | December 2012 |
| **Degree** | PhD, Department of Computer Science |
| **Abstract** | Novel learning rule based on spike-pair STDP algorithm. Developed SpikeServer tool for spike injection via Ethernet. Introduced population-based routing. Created STDP-TTS learning rule. |
| **Keywords** | TTS, STDP, Asynchronous software execution, Real-time software, Population-based routing, Neuromorphic hardware, SpiNNaker |
| **Tools** | SpiNNaker hardware, custom C, Python |
| **PDF** | Available (14.1 MB) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/learning-in-spiking-neural-networks |

### 1.6 Thomas Sharp -- Real-Time Million-Synapse Simulation of Cortical Tissue (2013)

| Field | Details |
|-------|---------|
| **Title** | Real-Time Million-Synapse Simulation of Cortical Tissue |
| **Author** | Thomas Sharp |
| **Year** | June 2013 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | Steve Furber (main), James Garside (co-supervisor) |
| **Abstract** | Demonstrated real-time simulation of rodent somatosensory cortex on SpiNNaker prototype. Model: 10^5 neurons, 7x10^7 synapses across 360 processors on 23 chips. Each chip draws just 1 watt. |
| **Tools** | SpiNNaker hardware |
| **PDF** | Available (21.7 MB) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/real-time-million-synapse-simulation-of-cortical-tissue |

### 1.7 Francesco Galluppi -- Information Representation on a Universal Neural Chip (2013)

| Field | Details |
|-------|---------|
| **Title** | Information Representation on a Universal Neural Chip |
| **Author** | Francesco Galluppi |
| **Year** | 2013 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | Steve Furber (likely) |
| **Abstract** | Modelling biologically plausible neural networks on SpiNNaker. Understanding mechanisms the brain uses to represent and elaborate information. Also developed hierarchical configuration systems. |
| **Note** | Galluppi first joined SpiNNaker in January 2009 for his MSc thesis, then returned April 2010 for PhD. |

### 1.8 Jonathan Heathcote -- Building and Operating Large-Scale SpiNNaker Machines (2016)

| Field | Details |
|-------|---------|
| **Title** | Building and Operating Large-Scale SpiNNaker Machines |
| **Author** | Jonathan Heathcote |
| **Year** | October 2016 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | James Garside (main), Steve Furber (co-supervisor) |
| **Abstract** | Physical layout scheme for hexagonal torus topologies minimizing cable length. Improved routing algorithms. Placement and routing algorithms minimizing congestion and tolerating network faults. Demonstrated on half-million core prototype. |
| **Keywords** | Fault tolerance, Graphs, Simulated annealing, Place and Route, Hexagonal Torus Topology, SpiNNaker |
| **Tools** | SpiNNaker hardware, Python (Rig library) |
| **PDF** | Available (8.54 MB) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/building-and-operating-large-scale-spinnaker-machines |
| **GitHub** | https://github.com/mossblaser/phd_thesis_experiments |

### 1.9 James Knight -- Plasticity in Large-scale Neuromorphic Models of the Neocortex (2016)

| Field | Details |
|-------|---------|
| **Title** | Plasticity in Large-scale Neuromorphic Models of the Neocortex |
| **Author** | James Knight |
| **Year** | November 2016 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | Steve Furber (main), David Lester (co-supervisor) |
| **Abstract** | New SpiNNaker synaptic plasticity implementation. Neocortically-inspired model with 2x10^4 neurons and 5.1x10^7 plastic synapses -- the largest plastic neural network ever simulated on neuromorphic hardware at that time. |
| **PDF** | Available (8.37 MB) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/plasticity-in-large-scale-neuromorphic-models-of-the-neocortex |

### 1.10 Andrew Mundy -- Real time Spaun on SpiNNaker (2016)

| Field | Details |
|-------|---------|
| **Title** | Real time Spaun on SpiNNaker -- Functional brain simulation on a massively-parallel computer architecture |
| **Author** | Andrew Mundy |
| **Year** | November 2016 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | James Garside (main), Steve Furber (co-supervisor) |
| **Abstract** | Three optimization techniques for simulating Spaun (2.5M neuron model): (1) reducing NEF memory/compute (1/20th cores needed); (2) additional cores to minimize network traffic; (3) novel logic minimization for routing tables. Achieved 9000x speed-up over prior results. |
| **PDF** | Available (4.77 MB) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/real-time-spaun-on-spinnaker-functional-brain-simulation-on-a-mas |

### 1.11 Qian Liu -- Deep Spiking Neural Networks (2018)

| Field | Details |
|-------|---------|
| **Title** | Deep Spiking Neural Networks |
| **Author** | Qian Liu |
| **Year** | January 2018 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | Steve Furber (main), David Lester (co-supervisor) |
| **Abstract** | Bridging the performance gap between SNNs and ANNs. Proposed "Noisy Softplus" activation function. Achieved 99.07% accuracy on MNIST with spiking convolutional networks. Spike-based rate multiplication for online training. |
| **PDF** | Available (15.1 MB) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/deep-spiking-neural-networks |

### 1.12 Petrut Bogdan -- Structural Plasticity on SpiNNaker (2019)

| Field | Details |
|-------|---------|
| **Title** | Structural Plasticity on SpiNNaker |
| **Author** | Petrut Bogdan |
| **Year** | September 2019 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | Steve Furber (main), David Lester (co-supervisor) |
| **Abstract** | Structural synaptic plasticity implementation on SpiNNaker. Combined with STDP for topographic map quality. Handwritten digit classification and motion detection. Simulations spanning 5+ hours with responses resembling Visual Cortex and Superior Colliculus. |
| **PDF** | Available (47.6 MB) |
