# Manchester eScholar / Research Explorer: Thesis Search

Searched through the University of Manchester Research Explorer (research.manchester.ac.uk), which has about 14,367 student theses.

## Important thing to note first

**Manchester Research Explorer almost exclusively hosts PhD theses (occasionally MPhil / MSc by Research).** I couldn't find any undergraduate (BSc) dissertations. The university library guide confirms it catalogues "postgraduate research theses" -- PhD, MPhil, and MSc by Research only. Undergrad dissertations aren't deposited here.

So everything below is PhD-level unless noted otherwise. Found 2 exceptions: 1 MPhil and 1 MSc by Research. These represent 3-4 years of full-time research, way beyond what an undergrad would do. Just useful for seeing what exists in the space.

---

## Neuromorphic Computing / SpiNNaker Theses

### Modelling Neural Dynamics On Neuromorphic Hardware
- **Mollie Ward**, 2024, PhD, Computer Science
- Supervised by Oliver Rhodes (main), James Garside (co)
- https://research.manchester.ac.uk/en/studentTheses/modelling-neural-dynamics-on-neuromorphic-hardware/
- Explores implementing Hodgkin-Huxley and two-compartment dendritic models on SpiNNaker and SpiNNaker2. Shows excellent agreement with reference models. Interesting finding: two-compartment model can actually decrease overall energy consumption compared to LIF-based SNNs.
- 16.6 MB PDF

### Deep Spiking Neural Networks
- **Qian Liu**, 2018, PhD, Computer Science
- Supervised by Steve Furber (main), David Lester (co)
- https://research.manchester.ac.uk/en/studentTheses/deep-spiking-neural-networks
- Proposes Noisy Softplus (NSP) activation function for biologically-plausible spiking neurons. Introduces Parametric Activation Function (PAF) for mapping ANN values to physical SNN units. Gets 99.07% on MNIST using deep spiking CNNs. Also develops spike-based rate multiplication for online training with STDP.
- Novel activation functions, ANN-to-SNN conversion pipeline, spiking autoencoders and RBMs
- 15.1 MB PDF

### Parallelisation of Neural Processing on Neuromorphic Hardware
- **Luca Peres**, 2022, PhD, Computer Science
- Supervised by Steve Furber (main), Oliver Rhodes (co)
- https://research.manchester.ac.uk/en/studentTheses/parallelisation-of-neural-processing-on-neuromorphic-hardware
- Did the world's first real-time simulation of the Cortical Microcircuit model (20x better than previous). Developed partitioning approaches with up to 9x higher throughput. Pretty impressive stuff.
- 15.4 MB PDF

### Learning in Spiking Neural Networks
- **Sergio Davies**, 2012, PhD, Computer Science
- https://research.manchester.ac.uk/en/studentTheses/learning-in-spiking-neural-networks
- Proposes a novel STDP learning rule that's less computationally expensive. Addresses SpiNNaker implementation, spike injection via Ethernet, and population-based routing.
- Foundational early SpiNNaker thesis
- 14.1 MB PDF

### Stochastic Processes For Neuromorphic Hardware
- **Gabriel Fonseca Guerra**, 2020, PhD, Computer Science
- Supervised by Steve Furber (main), David Lester (co)
- https://research.manchester.ac.uk/en/studentTheses/stochastic-processes-for-neuromorphic-hardware
- Two contributions: (1) Solving constraint satisfaction problems using SNNs on both SpiNNaker AND Loihi, matching state-of-the-art performance; (2) Implementing ion-channel current models and realistic postsynaptic potentials on SpiNNaker.
- One of the few theses that uses both SpiNNaker and Intel Loihi
- 22.5 MB PDF

### Parallel Simulation of Neural Networks on SpiNNaker
- **Xin Jin**, 2010, PhD, Computer Science
- https://research.manchester.ac.uk/en/studentTheses/parallel-simulation-of-neural-networks-on-spinnaker-universal-neu/
- Addresses computational speed challenges in ANN simulation. Proposes parallel processing on SpiNNaker for SNNs with STDP and parallel distributed processing with backprop. Shows linear scalability.
- Early foundational SpiNNaker thesis
