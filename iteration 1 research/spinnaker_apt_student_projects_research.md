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
