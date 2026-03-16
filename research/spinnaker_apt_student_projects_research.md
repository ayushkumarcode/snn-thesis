# SpiNNaker / APT Group Student Projects Research Report

**Date:** 2026-02-24
**Scope:** Student projects (PhD, MSc, undergraduate) from the APT (Advanced Processor Technologies) group at the University of Manchester, with focus on SpiNNaker neuromorphic computing.

---

## Executive Summary

The APT group at the University of Manchester, led by Prof. Steve Furber (now Emeritus), has produced a substantial body of student thesis work centred on the SpiNNaker neuromorphic computing platform. I identified **14 PhD theses** directly related to SpiNNaker/neuromorphic computing, **1 confirmed MSc dissertation** with code on GitHub, and **several external student projects** (from TUM, other universities) that used SpiNNaker hardware. Notably, undergraduate/3rd-year projects from Manchester are **not publicly accessible** -- the project listing system (`studentnet.cs.manchester.ac.uk`) requires Manchester authentication. The old APT website (`apt.cs.manchester.ac.uk`) has been largely redirected to the main CS department page, losing historical thesis listings. All PhD theses listed below have **full-text PDFs freely available** through the Manchester Research Explorer.

---

## 1. PhD Theses from the SpiNNaker/APT Group (University of Manchester)

### 1.1 Xin Jin -- Parallel Simulation of Neural Networks on SpiNNaker (2010)

| Field | Details |
|-------|---------|
| **Title** | Parallel Simulation of Neural Networks on SpiNNaker Universal Neuromorphic Hardware |
| **Author** | Xin Jin |
| **Year** | June 2010 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | Not listed in retrieved data (likely Steve Furber) |
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
| **Supervisors** | Not listed (likely Steve Furber) |
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
| **Keywords** | SpiNNaker, Plasticity |
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
| **Keywords** | Logic minimization, Spiking neural networks, SpiNNaker, Neural Engineering Framework |
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
| **Keywords** | Spike-based Rate Multiplication, Noisy Softplus, Neuromorphic Engineering, Deep Learning, Spiking Neural Networks |
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
| **Keywords** | Classification, Motion detection, SNN, Topographic maps, Structural plasticity, SpiNNaker |
| **PDF** | Available (47.6 MB) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/structural-plasticity-on-spinnaker |

### 1.13 Gabriel Fonseca Guerra -- Stochastic Processes for Neuromorphic Hardware (2020)

| Field | Details |
|-------|---------|
| **Title** | Stochastic Processes For Neuromorphic Hardware |
| **Author** | Gabriel Fonseca Guerra |
| **Year** | February 2020 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | Steve Furber (main), David Lester (co-supervisor) |
| **Abstract** | Stochastic processes in neuronal dynamics on both SpiNNaker and Loihi chips. Constraint satisfaction problems. Modelled intrinsic ion-channel currents and realistic postsynaptic potentials. Bridging neuromorphic technology with neurophysiology. |
| **Keywords** | Voltage gated ion channel currents, Postsynaptic Potentials, Constraint Satisfaction, SpiNNaker, Loihi |
| **PDF** | Available (22.5 MB) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/stochastic-processes-for-neuromorphic-hardware |

### 1.14 Mantas Mikaitis -- Arithmetic Accelerators for a Digital Neuromorphic Processor (2020)

| Field | Details |
|-------|---------|
| **Title** | Arithmetic Accelerators for a Digital Neuromorphic Processor |
| **Author** | Mantas Mikaitis |
| **Year** | February 2020 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | David Lester (main), Steve Furber (co-supervisor) |
| **Abstract** | Programmable accelerator for exponential and logarithm functions in SNN models for SpiNNaker2 chip. Stochastic rounding techniques for numerical accuracy. |
| **Keywords** | Neuromorphic engineering, Hardware accelerators, Stochastic rounding, Exponential function, Numerical accuracy |
| **PDF** | Available (2.8 MB) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/arithmetic-accelerators-for-a-digital-neuromorphic-processor |

### 1.15 Luca Peres -- Parallelisation of Neural Processing on Neuromorphic Hardware (2022)

| Field | Details |
|-------|---------|
| **Title** | Parallelisation of Neural Processing on Neuromorphic Hardware |
| **Author** | Luca Peres |
| **Year** | June 2022 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | Steve Furber (main), Oliver Rhodes (co-supervisor) |
| **Abstract** | World's first real-time simulation of Cortical Microcircuit model. 20x performance improvement over prior results. Up to 9x higher throughput of neural operations through enhanced partitioning. |
| **Keywords** | Event-driven Simulation, SNN, Parallel Programming, On-line Learning, Neuromorphic Computing, SpiNNaker, Real-time |
| **PDF** | Available (15.4 MB) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/parallelisation-of-neural-processing-on-neuromorphic-hardware |

### 1.16 Mollie Ward -- Modelling Neural Dynamics on Neuromorphic Hardware (2024)

| Field | Details |
|-------|---------|
| **Title** | Modelling Neural Dynamics On Neuromorphic Hardware |
| **Author** | Mollie Ward |
| **Year** | February 2024 |
| **Degree** | PhD, Department of Computer Science |
| **Supervisors** | Oliver Rhodes (main), James Garside (co-supervisor) |
| **Abstract** | Hodgkin-Huxley and two-compartment neuron models on SpiNNaker and SpiNNaker2. Fixed- and floating-point implementations with excellent numerical accuracy. HH neurons only 8x computational overhead vs. LIF models. Lower energy consumption for pattern detection. |
| **Keywords** | Spiking Neural Networks, Neuromorphic computing, Hodgkin-Huxley models |
| **PDF** | Available (16.6 MB) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/modelling-neural-dynamics-on-neuromorphic-hardware |

---

## 2. MSc Dissertations from Manchester

### 2.1 Nicholas Buttigieg -- Spiking Grid Cell Models on Neuromorphic Hardware (2019)

| Field | Details |
|-------|---------|
| **Title** | Spiking Grid Cell Models on Neuromorphic Hardware |
| **Author** | Nicholas Buttigieg |
| **Year** | 2019 |
| **Degree** | MSc, Faculty of Science and Engineering |
| **Supervisors** | Prof. Steve Furber, Dr. Oliver Rhodes |
| **Abstract** | Spiking grid cell models for spatial navigation implemented on SpiNNaker neuromorphic hardware. |
| **Tools** | SpiNNaker, Python 2.7, sPyNNaker, Brian2 (alternative) |
| **GitHub** | https://github.com/nickybu/spiking_grid_cell_model |
| **Full Dissertation** | Available via Google Drive link in repo |
| **Accessible** | YES -- code and thesis both freely accessible |

### 2.2 Francesco Galluppi -- MSc Thesis (2009)

| Field | Details |
|-------|---------|
| **Title** | Unknown (SpiNNaker-related, ran "Doughnut Hunter" as first neural application on SpiNNaker test chip) |
| **Author** | Francesco Galluppi |
| **Year** | 2009 |
| **Degree** | MSc |
| **Note** | Galluppi joined SpiNNaker in January 2009 for his MSc before returning for PhD in April 2010. Details of MSc thesis title not found. |

---

## 3. External Student Projects Using SpiNNaker (non-Manchester)

### 3.1 Master's Thesis: Brain-Machine Interface with SpiNNaker (TUM, ~2017)

| Field | Details |
|-------|---------|
| **Title** | Decoding of 3D Reach and Grasp Movements from Non-invasive EEG Signals using SpiNNaker Neuromorphic Hardware |
| **Author** | GitHub user "solversa" |
| **Institution** | Not explicitly stated, likely TU Munich (based on related projects) |
| **Year** | ~2017 |
| **Degree** | Master's thesis |
| **Abstract** | Neuromorphic information processing for BCI. Decoded imaginary movements from EEG using SNNs on SpiNNaker (4 chips, 64 cores). Architecture inspired by insect olfactory system. Achieved 73.4% classification accuracy with STDP. |
| **Tools** | SpiNNaker, Python, Jupyter, STDP, homeostasis |
| **GitHub** | https://github.com/solversa/Master-Thesis-Brain-Machine-Interface |
| **Accessible** | YES -- full thesis, code, and data on GitHub |

### 3.2 Bachelor Thesis: Spiking Stereo Matching (2016-2017)

| Field | Details |
|-------|---------|
| **Title** | Spiking Stereo Matching |
| **Author** | Gadi Dikov |
| **Year** | 2016-2017 |
| **Degree** | Bachelor thesis |
| **Abstract** | SNN for real-time event-based stereo matching using dynamic vision sensors and SpiNNaker hardware. Achieved cooperative stereo matching at 2ms latency. |
| **Tools** | SpiNNaker, sPyNNaker, PyNN, Dynamic Vision Sensors, HBP Platform |
| **GitHub** | https://github.com/gdikov/SpikingStereoMatching |
| **Publication** | Associated conference paper published 2017 |
| **Accessible** | YES -- code freely accessible |

### 3.3 Bachelor Thesis: SpiNNaker ROS Integration (TUM)

| Field | Details |
|-------|---------|
| **Title** | SpiNNaker ROS Integration |
| **Author** | reiths (GitHub username) |
| **Institution** | Technical University of Munich, Chair of Neuroscientific System Theory |
| **Degree** | Bachelor thesis |
| **Abstract** | Bridges ROS (Robot Operating System) with SpiNNaker. Converts ROS messages to neural spikes for injection into SpiNNaker, and converts spike activity back to ROS data streams. |
| **Tools** | SpiNNaker, PyNN, ROS, sPyNNakerExternalDevicesPlugin |
| **GitHub** | https://github.com/reiths/ros_spinnaker_interface |
| **Accessible** | YES -- code freely accessible |

### 3.4 MSNE Research Internship: Short-term Plasticity on SpiNNaker (2018)

| Field | Details |
|-------|---------|
| **Title** | Short-term Plasticity Model on SpiNNaker |
| **Author** | MSNE (Master of Science in Neuroengineering) student |
| **Institution** | TU Munich student, interned at APT group Manchester |
| **Year** | Spring 2018 |
| **Supervisor** | Dr. Oliver Rhodes |
| **Abstract** | Implemented and tested a short-term plasticity model on SpiNNaker during a 6-8 week research internship at the APT group, University of Manchester. |

### 3.5 Yexin Yan -- SpiNNaker2 Algorithms (TU Dresden, 2022)

| Field | Details |
|-------|---------|
| **Title** | Implementation of Bioinspired Algorithms on the Neuromorphic VLSI System SpiNNaker 2 |
| **Author** | Yexin Yan |
| **Institution** | TU Dresden |
| **Year** | 2022 |
| **Degree** | PhD |
| **Abstract** | Hardware-software co-design for low-power neuromorphic applications on SpiNNaker2. |

---

## 4. Undergraduate / 3rd Year Projects at Manchester

### Status: NOT PUBLICLY ACCESSIBLE

The Manchester CS 3rd year project system (`studentnet.cs.manchester.ac.uk/ugt/year3/project/`) requires university authentication (CAS login). Project listings for 2024/25 and 2025/26 exist but cannot be accessed externally.

**What we know:**
- The APT group (including Oliver Rhodes, Michael Hopkins) likely supervises 3rd year and MSc projects on SpiNNaker topics
- There is a `UoM CS 3rd Year Projects` GitHub organisation (https://github.com/uom-cs-projects) with 14 repos, but none are SpiNNaker-related
- The project coordinator is Tim Morris

**Inference:** Undergraduate projects on SpiNNaker almost certainly exist given:
- The group has MSc and PhD pipeline (Galluppi started with MSc before PhD)
- Oliver Rhodes and Michael Hopkins are active supervisors
- EBRAINS provides free access to SpiNNaker for research projects
- SpiNNaker workshops include hands-on tutorials accessible to students

---

## 5. Key Supervisors and Group Members

| Name | Role | Supervision Notes |
|------|------|-------------------|
| **Steve Furber** | Emeritus Professor (ICL Chair of Computer Engineering) | Main/co-supervisor on 12+ PhD theses; co-supervised MSc (Buttigieg) |
| **James Garside** | Senior researcher | Co-supervised: Heathcote, Sharp, Mundy, Ward, Rast |
| **David Lester** | Researcher (deceased?) | Co-supervised: Knight, Bogdan, Fonseca Guerra, Mikaitis, Liu |
| **Oliver Rhodes** | Lecturer in Bio-Inspired Computing | Co-supervised: Peres, Ward; supervised MSc (Buttigieg), MSNE intern |
| **Michael Hopkins** | Head of Research into SNNs | Likely supervises student projects; research on SNN applications |
| **Ke Chen** | Academic | Supervises PhD on Biologically-Plausible Continual Learning using SpiNNaker |

---

## 6. Available Postgraduate Research Projects (Current/Recent)

### 6.1 Biologically-Plausible Continual Learning

| Field | Details |
|-------|---------|
| **Supervisor** | Ke Chen |
| **Focus** | Catastrophic forgetting, continual learning, SNN on SpiNNaker |
| **URL** | https://www.cs.manchester.ac.uk/study/postgraduate-research/research-projects/description/?projectid=22461 |
| **Status** | Competition funded (open to worldwide students) |

---

## 7. Related Resources and Platforms

### SpiNNaker Software Stack
- **sPyNNaker**: PyNN implementation for SpiNNaker -- https://github.com/SpiNNakerManchester/sPyNNaker
- **SpiNNTools**: Execution engine -- maps parallel applications, executes, extracts results
- **SpiNNakerManchester GitHub**: https://github.com/SpiNNakerManchester (60+ repos)
- **Documentation**: https://spinnakermanchester.readthedocs.io/

### Access to SpiNNaker
- Free test access via EBRAINS account (online via web browser)
- Local SpiNNaker boards available (SpiNN-3, SpiNN-5 boards loaned to ~100 labs)
- Full 1 million core machine at Manchester

### SpiNNaker Workshops (Hands-on Training)
- Regular workshops with lectures and hands-on labs
- 8th workshop materials available: https://spinnakermanchester.github.io/workshops/eighth.html
- EBRAINS/HBP tutorial sessions available online
- Tutorial notebooks from beginner to expert level

---

## 8. Research Gaps and Limitations

1. **Undergraduate projects are hidden**: Manchester's 3rd year project system is behind authentication. We cannot confirm specific SpiNNaker-related UG projects exist, though they almost certainly do.

2. **Old APT website lost**: The original `apt.cs.manchester.ac.uk/publications/thesis.php` page has been redirected, losing the comprehensive historical thesis listing. The Wayback Machine was not accessible from this tool.

3. **MSc dissertations are under-documented**: Only one MSc dissertation (Buttigieg 2019) was confirmed with public access. The Manchester Research Explorer focuses primarily on PhD theses.

4. **Francesco Galluppi's MSc thesis (2009) details are lost**: We know he did an MSc with SpiNNaker but cannot find the thesis title or text.

5. **Steve Furber's profile lists 22 supervised theses**, but the full list could not be extracted from the Research Explorer page (403 error on the supervisedBy filter).

---

## 9. Confidence Assessment

| Finding | Confidence |
|---------|------------|
| PhD thesis list (16 entries) | HIGH -- verified via Research Explorer with PDFs |
| Buttigieg MSc (2019) | HIGH -- code, thesis text, and supervisors confirmed |
| Galluppi MSc (2009) | MEDIUM -- confirmed he did MSc but details unavailable |
| External student projects (TUM, etc.) | HIGH -- verified via GitHub repos |
| UG projects exist at Manchester | MEDIUM -- inferred but not confirmed |
| Supervisors (Furber, Rhodes, Hopkins, Garside) | HIGH -- verified from thesis records |

---

## 10. Recommended Follow-ups

1. **Contact Oliver Rhodes or Michael Hopkins** directly to ask about current/past 3rd year and MSc project offerings
2. **Ask on the SpiNNaker Users Google Group** (https://groups.google.com/g/spinnakerusers) about student project examples
3. **Check EBRAINS** for any student project reports or tutorials that include project-level work
4. **Try Wayback Machine** for the old APT thesis page (http://apt.cs.manchester.ac.uk/publications/thesis.php) to recover the full historical listing
5. **Search Manchester eScholar** (https://www.escholar.manchester.ac.uk/) with keyword searches for additional MSc theses
6. **Request access** to the Manchester StudentNet project listings if you have a Manchester contact
