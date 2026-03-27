# SpiNNaker / APT Group Student Projects

Looking into student projects (PhD, MSc, undergrad) from the APT (Advanced Processor Technologies) group at Manchester, which is where SpiNNaker comes from.

Found 14 PhD theses directly related to SpiNNaker/neuromorphic computing, 1 confirmed MSc dissertation with code on GitHub, and several external student projects from TUM and other universities that used SpiNNaker hardware. Unfortunately, undergrad/3rd-year projects from Manchester aren't publicly accessible -- the project listing system (`studentnet.cs.manchester.ac.uk`) requires Manchester authentication. The old APT website (`apt.cs.manchester.ac.uk`) has been redirected to the main CS department page, so the historical thesis listings are gone. All PhD theses listed below have full-text PDFs freely available through Manchester Research Explorer.

---

## PhD Theses from the SpiNNaker/APT Group

### Xin Jin -- Parallel Simulation of Neural Networks on SpiNNaker (2010)
- PhD, Computer Science
- Investigated efficient modelling for SpiNNaker considering communication, processing, and storage constraints. Covered both SNNs with STDP and parallel distributed processing with backpropagation. Showed feasibility and linear scalability.
- Keywords: PDP, STDP, Backpropagation, MLP, Real-time, Izhikevich, SpiNNaker, ARM
- PDF: 6.57 MB at https://research.manchester.ac.uk/en/studentTheses/parallel-simulation-of-neural-networks-on-spinnaker-universal-neu/

### M.M. Khan -- Configuring a Massively Parallel CMP System (2009)
- PhD, Computer Science, supervised by Steve Furber (likely)
- Configuration and mapping of neural networks onto the SpiNNaker chip multiprocessor system.
- PDF: https://apt.cs.manchester.ac.uk/ftp/pub/amulet/theses/mmkhan09_phd.pdf

### Alexander Rast -- Scalable Event-Driven Modelling Architectures (2011)
- PhD, supervised by Steve Furber (supervisor), James Garside (advisor)
- Developed a library of predesigned event-driven neural components for SpiNNaker. Dealt with burstiness, scalability, and asynchronous event-driven models.
- PDF: https://apt.cs.manchester.ac.uk/ftp/pub/apt/theses/Rast11_phd.pdf

### Eustace Painkras -- A Chip Multiprocessor for a Large-scale Neural Simulator (2012)
- PhD, Computer Science
- Design of the actual SpiNNaker CMP chip -- many simple power-efficient ARM processors with small local memories, asynchronous networks-on-chip, and GALS architecture. Demonstrated successful neural simulation on 48-chip PCBs.
- https://www.escholar.manchester.ac.uk/uk-ac-man-scw:198344

### Sergio Davies -- Learning in Spiking Neural Networks (2012)
- PhD, Computer Science
- Novel learning rule based on spike-pair STDP algorithm. Developed SpikeServer tool for spike injection via Ethernet. Introduced population-based routing. Created STDP-TTS learning rule.
- PDF: 14.1 MB at https://research.manchester.ac.uk/en/studentTheses/learning-in-spiking-neural-networks

### Thomas Sharp -- Real-Time Million-Synapse Simulation of Cortical Tissue (2013)
- PhD, supervised by Steve Furber (main), James Garside (co)
- Real-time simulation of rodent somatosensory cortex on SpiNNaker prototype. Model: 10^5 neurons, 7x10^7 synapses across 360 processors on 23 chips. Each chip draws just 1 watt.
- PDF: 21.7 MB at https://research.manchester.ac.uk/en/studentTheses/real-time-million-synapse-simulation-of-cortical-tissue

### Francesco Galluppi -- Information Representation on a Universal Neural Chip (2013)
- PhD, Computer Science
- Modelling biologically plausible neural networks on SpiNNaker. Understanding how the brain represents and elaborates information. Also did hierarchical configuration systems.
- Note: Galluppi first joined SpiNNaker in January 2009 for his MSc, then came back April 2010 for PhD.

### Jonathan Heathcote -- Building and Operating Large-Scale SpiNNaker Machines (2016)
- PhD, supervised by James Garside (main), Steve Furber (co)
- Physical layout scheme for hexagonal torus topologies minimizing cable length. Improved routing algorithms. Placement and routing that tolerates network faults. Demonstrated on half-million core prototype.
- GitHub: https://github.com/mossblaser/phd_thesis_experiments
- PDF: 8.54 MB at https://research.manchester.ac.uk/en/studentTheses/building-and-operating-large-scale-spinnaker-machines

### James Knight -- Plasticity in Large-scale Neuromorphic Models of the Neocortex (2016)
- PhD, supervised by Steve Furber (main), David Lester (co)
- New SpiNNaker synaptic plasticity implementation. Neocortically-inspired model with 20K neurons and 51M plastic synapses -- largest plastic neural network ever simulated on neuromorphic hardware at that time.
- PDF: 8.37 MB at https://research.manchester.ac.uk/en/studentTheses/plasticity-in-large-scale-neuromorphic-models-of-the-neocortex
