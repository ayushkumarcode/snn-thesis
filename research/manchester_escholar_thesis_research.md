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
- 6.57 MB PDF

### Arithmetic Accelerators for a Digital Neuromorphic Processor
- **Mantas Mikaitis**, 2020, PhD, Computer Science
- Supervised by David Lester (main), Steve Furber (co)
- https://research.manchester.ac.uk/en/studentTheses/arithmetic-accelerators-for-a-digital-neuromorphic-processor/
- Investigates programmable accelerator for exponential and logarithm functions in SNN models within SpiNNaker2. Explores numerical accuracy of ODE solvers for Izhikevich neuron model. Looks into stochastic rounding methods.
- Chip-level hardware accelerator design stuff
- 2.8 MB PDF

### Building and Operating Large-Scale SpiNNaker Machines
- **Jonathan Heathcote**, 2016, PhD, Computer Science
- Supervised by James Garside (main), Steve Furber (co)
- https://research.manchester.ac.uk/en/studentTheses/building-and-operating-large-scale-spinnaker-machines
- Scaling SpiNNaker to simulate up to 1 billion neurons. Physical layout scheme for hexagonal torus topologies, improved routing algorithms, placement and routing algorithms that tolerate network faults. Demonstrated on half-million core prototype.
- Supercomputer-scale engineering
- 8.54 MB PDF

### Structural Plasticity on SpiNNaker
- **Petrut Bogdan**, 2019, PhD, Computer Science
- Supervised by Steve Furber (main), David Lester (co)
- https://research.manchester.ac.uk/en/studentTheses/structural-plasticity-on-spinnaker
- Implements structural plasticity model on SpiNNaker that runs real-time alongside STDP. Applications in topographic map generation, unsupervised handwritten digit classification, and motion detection.
- 47.6 MB PDF

### Real Time Spaun on SpiNNaker
- **Andrew Mundy**, 2016, PhD, Computer Science
- Supervised by James Garside (main), Steve Furber (co)
- https://research.manchester.ac.uk/en/studentTheses/real-time-spaun-on-spinnaker-functional-brain-simulation-on-a-mas
- Achieved real-time execution of Spaun (2.5M neuron functional brain model) on SpiNNaker. 9000x speed-up over previous results. Only needed 5% of the cores previously required. Also did novel routing table optimization.
- That's a wild speedup
- 4.77 MB PDF

### Biologically Inspired Neural Computation
- **Adam Perrett**, 2022, PhD, Computer Science
- Supervised by Steve Furber (main), Oliver Rhodes (co)
- https://research.manchester.ac.uk/en/studentTheses/biologically-inspired-neural-computation
- Three things: (1) biologically-inspired visual attention on SpiNNaker; (2) e-prop learning algorithm implementation; (3) gradient-descent-free architecture using dendritic abstractions with neurogenesis, getting comparable performance to Adam optimizer.
- Uses iCub robot platform
- 20 MB PDF

### Scalability and Robustness of Artificial Neural Networks
- **Evangelos Stromatias**, 2016, PhD, Computer Science
- Supervised by Steve Furber (main), James Garside (co)
- https://research.manchester.ac.uk/en/studentTheses/scalability-and-robustness-of-artificial-neural-networks
- Examines power consumption and communication latencies on SpiNNaker running large-scale SNNs. Develops power estimation model. Characterizes impact of hardware bit precision, noise, and weight variation on spiking DBNs for handwritten digit recognition. Shows spiking DBNs work on limited-precision hardware without drastic performance loss.
- 25.4 MB PDF

### Plasticity in Large-Scale Neuromorphic Models of the Neocortex
- **James Knight**, 2016, PhD, Computer Science
- Supervised by Steve Furber (main), David Lester (co)
- https://research.manchester.ac.uk/en/studentTheses/plasticity-in-large-scale-neuromorphic-models-of-the-neocortex
- Created the largest plastic neural network ever simulated on neuromorphic hardware: 20,000 neurons and 51 million plastic synapses. Developed neocortically-inspired temporal sequence learning model.
- World-record simulation
- 8.37 MB PDF

---

## Adjacent Hardware / FPGA Theses

### Memristive Crossbar Arrays for Machine Learning Systems
- **Manu Vijayagopalan Nair**, 2015, **MPhil** (not PhD), EEE
- Supervised by Piotr Dudek (main), Hujun Yin (co)
- https://research.manchester.ac.uk/en/studentTheses/memristive-crossbar-arrays-for-machine-learning-systems/
- Specialized computing systems diverging from Von-Neumann architectures. Presents Unregulated Step Descent (USD) algorithm for training memristive crossbar arrays. References TrueNorth, SpiNNaker, Neurogrid.
- **One of only 2 non-PhD theses found**
- 4.34 MB PDF

### Efficient Execution of CNNs on Low Powered Heterogeneous Systems
- **Crefeda Rodrigues**, 2020, PhD, Computer Science
- Supervised by Graham Riley (main), Mikel Lujan (co)
- https://research.manchester.ac.uk/en/studentTheses/efficient-execution-of-convolutional-neural-networks-on-low-power
- SyNERGY framework for evaluating DL models on execution time and energy metrics on mobile platforms. NNTaskSim for task-parallel neural network computation exploration. Energy-efficient ML on edge devices.
- Tested on Jetson TX1, Snapdragon 820

### Modular FPGA Systems with Dynamic Workloads
- **Anuj Vaishnav**, 2020, PhD, Computer Science
- Supervised by Dirk Koch (main), James Garside (co)
- https://research.manchester.ac.uk/en/studentTheses/modular-fpga-systems-with-support-for-dynamic-workloads-and-virtu/
- FPGA OS, OpenCL scheduling across CPU/FPGA, resource elasticity, live migration across FPGA clusters.
- 7.42 MB PDF

### FPGA Virtualisation on Heterogeneous Computing Systems
- **Khoa Pham**, 2020, PhD, Computer Science
