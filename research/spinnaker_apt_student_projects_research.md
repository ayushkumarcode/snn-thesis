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

### Andrew Mundy -- Real time Spaun on SpiNNaker (2016)
- PhD, supervised by James Garside (main), Steve Furber (co)
- Three optimization techniques for simulating Spaun (2.5M neuron model): (1) reducing NEF memory/compute (only 1/20th the cores needed); (2) additional cores to minimize network traffic; (3) novel logic minimization for routing tables. 9000x speed-up over prior results.
- PDF: 4.77 MB at https://research.manchester.ac.uk/en/studentTheses/real-time-spaun-on-spinnaker-functional-brain-simulation-on-a-mas

### Qian Liu -- Deep Spiking Neural Networks (2018)
- PhD, supervised by Steve Furber (main), David Lester (co)
- Bridging the SNN-ANN performance gap. Proposed "Noisy Softplus" activation function. 99.07% on MNIST with spiking convnets. Spike-based rate multiplication for online training.
- PDF: 15.1 MB at https://research.manchester.ac.uk/en/studentTheses/deep-spiking-neural-networks

### Petrut Bogdan -- Structural Plasticity on SpiNNaker (2019)
- PhD, supervised by Steve Furber (main), David Lester (co)
- Structural synaptic plasticity on SpiNNaker, combined with STDP. Handwritten digit classification and motion detection. Simulations running 5+ hours with responses resembling Visual Cortex and Superior Colliculus.
- PDF: 47.6 MB at https://research.manchester.ac.uk/en/studentTheses/structural-plasticity-on-spinnaker

### Gabriel Fonseca Guerra -- Stochastic Processes for Neuromorphic Hardware (2020)
- PhD, supervised by Steve Furber (main), David Lester (co)
- Stochastic processes on both SpiNNaker and Loihi chips. Constraint satisfaction problems. Modelled intrinsic ion-channel currents and realistic postsynaptic potentials.
- PDF: 22.5 MB at https://research.manchester.ac.uk/en/studentTheses/stochastic-processes-for-neuromorphic-hardware

### Mantas Mikaitis -- Arithmetic Accelerators for a Digital Neuromorphic Processor (2020)
- PhD, supervised by David Lester (main), Steve Furber (co)
- Programmable accelerator for exp/log functions in SNN models for SpiNNaker2. Stochastic rounding techniques for numerical accuracy.
- PDF: 2.8 MB at https://research.manchester.ac.uk/en/studentTheses/arithmetic-accelerators-for-a-digital-neuromorphic-processor

### Luca Peres -- Parallelisation of Neural Processing on Neuromorphic Hardware (2022)
- PhD, supervised by Steve Furber (main), Oliver Rhodes (co)
- World's first real-time Cortical Microcircuit simulation. 20x over prior results. Up to 9x higher throughput through enhanced partitioning.
- PDF: 15.4 MB at https://research.manchester.ac.uk/en/studentTheses/parallelisation-of-neural-processing-on-neuromorphic-hardware

### Mollie Ward -- Modelling Neural Dynamics on Neuromorphic Hardware (2024)
- PhD, supervised by Oliver Rhodes (main), James Garside (co)
- Hodgkin-Huxley and two-compartment neuron models on SpiNNaker and SpiNNaker2. Fixed- and floating-point implementations with excellent accuracy. HH neurons only 8x computational overhead vs LIF. Lower energy consumption for pattern detection with more complex models, which is interesting.
- PDF: 16.6 MB at https://research.manchester.ac.uk/en/studentTheses/modelling-neural-dynamics-on-neuromorphic-hardware

---

## MSc Dissertations from Manchester

### Nicholas Buttigieg -- Spiking Grid Cell Models on Neuromorphic Hardware (2019)
- MSc, Faculty of Science and Engineering
- Supervised by Steve Furber and Oliver Rhodes
- Spiking grid cell models for spatial navigation implemented on SpiNNaker
- Uses SpiNNaker, Python 2.7, sPyNNaker, Brian2
- GitHub: https://github.com/nickybu/spiking_grid_cell_model
- Full dissertation available via Google Drive link in repo
- This is the only MSc dissertation i could confirm with public access

### Francesco Galluppi -- MSc Thesis (2009)
- He ran "Doughnut Hunter" as the first neural application on the SpiNNaker test chip
- Joined SpiNNaker January 2009 for MSc, then came back for PhD April 2010
- Couldn't find the actual thesis title or text anywhere

---

## External Student Projects Using SpiNNaker (not from Manchester)

### Brain-Machine Interface with SpiNNaker (TUM, ~2017)
- Master's thesis by GitHub user "solversa"
- Decoded imaginary 3D reach/grasp movements from EEG using SNNs on SpiNNaker (4 chips, 64 cores). Architecture inspired by insect olfactory system. 73.4% accuracy with STDP.
- GitHub: https://github.com/solversa/Master-Thesis-Brain-Machine-Interface
- Full thesis, code, and data all on GitHub

### Spiking Stereo Matching (Bachelor Thesis, 2016-2017)
- Gadi Dikov
- SNN for real-time event-based stereo matching with DVS + SpiNNaker. 2ms latency.
- Published as conference paper (Biomimetic and Biohybrid Systems, 2017)
- GitHub: https://github.com/gdikov/SpikingStereoMatching

### SpiNNaker ROS Integration (TUM Bachelor Thesis)
- By reiths (GitHub username), TU Munich, Chair of Neuroscientific System Theory
- Bridges ROS with SpiNNaker. Converts ROS messages to neural spikes for injection into SpiNNaker, and converts spike activity back to ROS data streams.
- GitHub: https://github.com/reiths/ros_spinnaker_interface

### Short-term Plasticity on SpiNNaker (MSNE Research Internship, 2018)
- MSNE (Master of Science in Neuroengineering) student from TU Munich
- Interned at APT group Manchester, Spring 2018
- Supervised by Oliver Rhodes
- Implemented and tested a short-term plasticity model on SpiNNaker during a 6-8 week internship

### Yexin Yan -- SpiNNaker2 Algorithms (TU Dresden, 2022)
- PhD thesis
- Hardware-software co-design for low-power neuromorphic applications on SpiNNaker2

---

## Undergraduate / 3rd Year Projects at Manchester

### Status: can't see them

The Manchester CS 3rd year project system (`studentnet.cs.manchester.ac.uk/ugt/year3/project/`) requires university authentication (CAS login). Project listings for 2024/25 and 2025/26 exist but can't be accessed from outside.

What i do know:
- The APT group (Oliver Rhodes, Michael Hopkins) likely supervises 3rd year and MSc projects on SpiNNaker topics
- There's a `UoM CS 3rd Year Projects` GitHub org (https://github.com/uom-cs-projects) with 14 repos, but none are SpiNNaker-related
- Project coordinator is Tim Morris

Undergrad projects on SpiNNaker almost certainly exist given:
- The group has an MSc and PhD pipeline (Galluppi started with MSc before PhD)
- Oliver Rhodes and Michael Hopkins are active supervisors
- EBRAINS provides free SpiNNaker access for research projects
- SpiNNaker workshops include hands-on tutorials accessible to students

---

## Key Supervisors and Group Members

| Name | Role | Notes |
|------|------|-------|
| **Steve Furber** | Emeritus Professor (ICL Chair of Computer Engineering) | Main/co-supervisor on 12+ PhD theses; co-supervised MSc (Buttigieg) |
| **James Garside** | Senior researcher | Co-supervised: Heathcote, Sharp, Mundy, Ward, Rast |
| **David Lester** | Researcher (deceased?) | Co-supervised: Knight, Bogdan, Fonseca Guerra, Mikaitis, Liu |
| **Oliver Rhodes** | Lecturer in Bio-Inspired Computing | Co-supervised: Peres, Ward; supervised MSc (Buttigieg), MSNE intern |
| **Michael Hopkins** | Head of Research into SNNs | Likely supervises student projects; research on SNN applications |
| **Ke Chen** | Academic | Supervises PhD on Biologically-Plausible Continual Learning using SpiNNaker |

---

## Current/Recent Postgrad Research Projects

### Biologically-Plausible Continual Learning
- Supervisor: Ke Chen
- Focus: catastrophic forgetting, continual learning, SNN on SpiNNaker
- https://www.cs.manchester.ac.uk/study/postgraduate-research/research-projects/description/?projectid=22461
- Competition funded, open to worldwide students

---

## SpiNNaker Resources

### Software Stack
- **sPyNNaker**: PyNN implementation for SpiNNaker -- https://github.com/SpiNNakerManchester/sPyNNaker
- **SpiNNTools**: execution engine -- maps parallel applications, executes, extracts results
- **SpiNNakerManchester GitHub**: https://github.com/SpiNNakerManchester (60+ repos)
- **Documentation**: https://spinnakermanchester.readthedocs.io/

### Access
- Free test access via EBRAINS account (online via web browser)
- Local SpiNNaker boards available (SpiNN-3, SpiNN-5 boards loaned to ~100 labs)
- Full 1 million core machine at Manchester

### Workshops
- Regular workshops with lectures and hands-on labs
- 8th workshop materials: https://spinnakermanchester.github.io/workshops/eighth.html
- EBRAINS/HBP tutorial sessions available online
- Tutorial notebooks from beginner to expert level

---

## Things i couldn't find or verify

1. **Undergrad projects are behind auth** -- can't confirm specific SpiNNaker UG projects exist, though they almost certainly do
2. **Old APT website lost** -- the original `apt.cs.manchester.ac.uk/publications/thesis.php` page got redirected, losing the full historical thesis listing
3. **MSc dissertations are under-documented** -- only one (Buttigieg 2019) confirmed with public access
4. **Galluppi's MSc thesis (2009) details are lost** -- know he did an MSc with SpiNNaker but can't find title or text
5. **Furber's profile lists 22 supervised theses** but couldn't extract the full list (403 error on the supervisedBy filter)

---

## Where to look next

1. Contact Oliver Rhodes or Michael Hopkins about current/past 3rd year and MSc project offerings
2. Ask on the SpiNNaker Users Google Group (https://groups.google.com/g/spinnakerusers) about student project examples
3. Check EBRAINS for student project reports or tutorials
4. Try Wayback Machine for the old APT thesis page to recover the historical listing
5. Search Manchester eScholar (https://www.escholar.manchester.ac.uk/) for additional MSc theses
6. Get a Manchester contact to check the StudentNet project listings
