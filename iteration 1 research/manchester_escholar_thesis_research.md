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

---

### 1.12 Scalability and Robustness of Artificial Neural Networks
| Field | Detail |
|-------|--------|
| **Author** | Evangelos Stromatias |
| **Year** | 2016 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Steve Furber (main), James Garside (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/scalability-and-robustness-of-artificial-neural-networks |
| **Abstract** | Examines power consumption and communication latencies on SpiNNaker running large-scale SNNs. Develops power estimation model. Characterizes impact of hardware bit precision, noise, and weight variation on spiking DBNs for handwritten digit recognition. Shows spiking DBNs work on limited-precision hardware without drastic performance loss. |
| **What They Built** | Power estimation model for SpiNNaker; spiking DBN robustness analysis |
| **Tools/Frameworks** | SpiNNaker, Deep Belief Networks |
| **Datasets** | Handwritten digits (likely MNIST) |
| **Scope** | High -- power modeling + robustness characterization |
| **PDF Size** | 25.4 MB |

---

### 1.13 Plasticity in Large-Scale Neuromorphic Models of the Neocortex
| Field | Detail |
|-------|--------|
| **Author** | James Knight |
| **Year** | 2016 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Steve Furber (main), David Lester (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/plasticity-in-large-scale-neuromorphic-models-of-the-neocortex |
| **Abstract** | Created the largest plastic neural network ever simulated on neuromorphic hardware: 20,000 neurons and 51 million plastic synapses. Developed neocortically-inspired temporal sequence learning model. |
| **What They Built** | Record-breaking plastic SNN on SpiNNaker; temporal sequence learning model |
| **Tools/Frameworks** | SpiNNaker, ARM processors |
| **Datasets** | Temporal sequence benchmarks |
| **Scope** | Very high -- world-record simulation |
| **PDF Size** | 8.37 MB |

---

## SECTION 2: ADJACENT HARDWARE / FPGA THESES

### 2.1 Memristive Crossbar Arrays for Machine Learning Systems
| Field | Detail |
|-------|--------|
| **Author** | Manu Vijayagopalan Nair |
| **Year** | 2015 |
| **Degree** | **MPhil (Master of Philosophy)** |
| **Department** | Electrical & Electronic Engineering |
| **Supervisors** | Piotr Dudek (main), Hujun Yin (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/memristive-crossbar-arrays-for-machine-learning-systems/ |
| **Abstract** | Examines specialized computing systems diverging from Von-Neumann architectures. Presents Unregulated Step Descent (USD) algorithm for training memristive crossbar arrays. References TrueNorth, SpiNNaker, Neurogrid. |
| **What They Built** | Novel training algorithm (USD) for memristive hardware; simulation studies |
| **Tools/Frameworks** | Memristive crossbar arrays, logistic regression, MLPs, RBMs |
| **Datasets** | Random data, MNIST |
| **Scope** | Moderate (MPhil) -- algorithm development + simulation |
| **PDF Size** | 4.34 MB |
| **NOTE** | **This is one of only 2 non-PhD theses found -- MPhil level** |

---

### 2.2 Efficient Execution of CNNs on Low Powered Heterogeneous Systems
| Field | Detail |
|-------|--------|
| **Author** | Crefeda Rodrigues |
| **Year** | 2020 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Graham Riley (main), Mikel Lujan (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/efficient-execution-of-convolutional-neural-networks-on-low-power |
| **Abstract** | Presents SyNERGY framework for evaluating DL models on execution time and energy metrics on mobile platforms. Presents NNTaskSim for task-parallel neural network computation exploration. Addresses energy-efficient ML on edge devices. |
| **What They Built** | SyNERGY framework; NNTaskSim simulator; predictive energy models |
| **Tools/Frameworks** | SyNERGY, NNTaskSim, Jetson TX1, Snapdragon 820 |
| **Datasets** | DL benchmark models |
| **Scope** | High -- two novel tools/frameworks |
| **PDF Size** | Not listed |

---

### 2.3 Modular FPGA Systems with Support for Dynamic Workloads and Virtualisation
| Field | Detail |
|-------|--------|
| **Author** | Anuj Vaishnav |
| **Year** | 2020 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Dirk Koch (main), James Garside (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/modular-fpga-systems-with-support-for-dynamic-workloads-and-virtu/ |
| **Abstract** | Demonstrates modular FPGA systems with dynamic resource adjustment. Contributions: modular dev flow for FPGA OS, OpenCL scheduling across CPU/FPGA, resource elasticity concept, live migration across FPGA clusters. |
| **What They Built** | FPGA operating system framework; OpenCL scheduler; live migration system |
| **Tools/Frameworks** | OpenCL, FPGAs |
| **Datasets** | N/A |
| **Scope** | Very high -- systems-level FPGA research |
| **PDF Size** | 7.42 MB |

---

### 2.4 FPGA Virtualisation on Heterogeneous Computing Systems
| Field | Detail |
|-------|--------|
| **Author** | Khoa Pham |
| **Year** | 2020 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Dirk Koch (main), James Garside (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/fpga-virtualisation-on-heterogeneous-computing-systems-model-tool/ |
| **Abstract** | Proposes fully FPGA-virtualised computing model for heterogeneous CPU+FPGA systems. Advances partial reconfiguration techniques for embedded, edge, and cloud computing. |
| **What They Built** | FPGA virtualisation model; partial reconfiguration methodology |
| **Tools/Frameworks** | FPGAs, partial reconfiguration |
| **Datasets** | N/A |
| **Scope** | High -- novel computing model |
| **PDF Size** | 16.1 MB |

---

### 2.5 Harnessing Reconfigurable Hardware to Design Heterogeneous Systems
| Field | Detail |
|-------|--------|
| **Author** | Konstantinos Iordanou |
| **Year** | 2023 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Mikel Lujan (main), Christos-Efthymios Kotselidis (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/harnessing-reconfigurable-hardware-to-design-heterogeneous-system |
| **Abstract** | Presents AutoTiC (Auto Tiny Classifiers): evolutionary algorithms to generate prediction circuits for tabular data. 10-75x less area/power in ASIC, 3-11x fewer FPGA resources vs ML baselines. Also microarchitectural SoC simulation framework. |
| **What They Built** | AutoTiC classifier generator; SoC simulation framework |
| **Tools/Frameworks** | Evolutionary algorithms, RTL simulation, FPGA, ASIC |
| **Datasets** | Tabular datasets |
| **Scope** | High -- novel approach to ML on hardware |
| **PDF Size** | 16.9 MB |

---

### 2.6 Dynamic CPU ISA Customizations through FPGA Interlays
| Field | Detail |
|-------|--------|
| **Author** | Jose Garcia Ordaz |
| **Year** | 2018 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Dirk Koch (main), James Garside (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/dynamic-cpu-isa-customizations-through-fpga-interlays |
| **Abstract** | Proposes embedding reconfigurable FPGA fabric ("Interlay") into CPUs for dynamic instruction set customization at runtime. |
| **What They Built** | FPGA-in-CPU architecture concept; ISA customization framework |
| **Tools/Frameworks** | FPGA, SIMD, soft-processor |
| **Datasets** | N/A |
| **Scope** | High -- novel architectural concept |
| **PDF Size** | 7.59 MB |

---

## SECTION 3: DEEP LEARNING / COMPUTER VISION THESES (Calibration)

### 3.1 Unsupervised Image Feature Learning for Convolutional Neural Networks
| Field | Detail |
|-------|--------|
| **Author** | Richard Hankins |
| **Year** | 2019 |
| **Degree** | PhD |
| **Department** | Electrical & Electronic Engineering |
| **Supervisors** | Hujun Yin (main), Piotr Dudek (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/unsupervised-image-feature-learning-for-convolutional-neural-netw/ |
| **Abstract** | Uses self-organising maps (SOM) for unsupervised feature learning in CNNs. Develops SOMNet architecture, novel feature aggregation methods, and SOM-based pre-training for deep learning. |
| **What They Built** | SOMNet architecture; SOM-based pre-training pipeline |
| **Tools/Frameworks** | SOMs, PCANet, CNNs |
| **Datasets** | Multiple image/video classification datasets |
| **Scope** | High |
| **PDF Size** | 7.31 MB |

---

### 3.2 Deep Learning for Semantic Feature Extraction in Aerial Imagery
| Field | Detail |
|-------|--------|
| **Author** | Ananya Gupta |
| **Year** | 2020 |
| **Degree** | PhD |
| **Department** | Electrical & Electronic Engineering |
| **Supervisors** | Hujun Yin (main), Simon Watson (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/deep-learning-for-semantic-feature-extraction-in-aerial-imagery/ |
| **Abstract** | Semantic segmentation of aerial images, road mapping in disaster areas, multitemporal image registration, tree identification in LiDAR point clouds (~90% accuracy). |
| **What They Built** | Multiple DL pipelines for aerial/satellite imagery analysis |
| **Tools/Frameworks** | PointNet++, 3D CNNs, voxel networks, ImageNet |
| **Datasets** | Palu Indonesia disaster dataset, ISPRS benchmark |
| **Scope** | High -- multi-modal remote sensing |
| **PDF Size** | 20.6 MB |

---

### 3.3 Bayesian Deep Learning for Pulsar Classification
| Field | Detail |
|-------|--------|
| **Author** | Alexandra Bonta |
| **Year** | 2022 |
| **Degree** | **MSc by Research (Master of Science by Research)** |
| **Department** | Physics & Astronomy |
| **Supervisors** | Anna Scaife (main), Albert Zijlstra (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/bayesian-deep-learning-for-pulsar-classification |
| **Abstract** | Develops "BonNet" classifier for automated pulsar classification with uncertainty quantification. 99% accuracy on HTRU1, 97% on HTRU2. |
| **What They Built** | BonNet neural network classifier with Bayesian uncertainty |
| **Tools/Frameworks** | Neural networks with dropout (Bayesian approximation) |
| **Datasets** | HTRU1, HTRU2 (High Time Resolution Universe survey) |
| **Scope** | Moderate (MSc level) -- single classifier with clear benchmarks |
| **PDF Size** | 7.93 MB |
| **NOTE** | **This is one of only 2 non-PhD theses found -- MSc by Research level. Most relevant for calibrating masters-level scope.** |

---

### 3.4 Object Detection with Few-Shot Learning, Vision-Language Knowledge and Vision Transformers
| Field | Detail |
|-------|--------|
| **Author** | Mengyuan Ma |
| **Year** | 2025 |
| **Degree** | PhD |
| **Department** | Electrical & Electronic Engineering |
| **Supervisors** | Hujun Yin (main), Krikor Ozanyan (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/object-detection-with-few-shot-learning-vision-language-knowledge/ |
| **Abstract** | Three contributions: SCENet (scene context + person identity), DINO-FocalNet (64.8% AP on low-quality images), EPNet/KDNet (few-shot detection with knowledge distillation). |
| **What They Built** | Multiple novel object detection architectures |
| **Tools/Frameworks** | DINO, FocalNet, CutMix, Vision Transformers |
| **Datasets** | MS COCO, PASCAL VOC |
| **Scope** | Very high -- three novel architectures |
| **PDF Size** | 14.8 MB |

---

## SECTION 4: COMPUTATIONAL NEUROSCIENCE / COGNITIVE SCIENCE

### 4.1 Bayesian Mechanisms in Spatial Cognition
| Field | Detail |
|-------|--------|
| **Author** | Tamas Madl |
| **Year** | 2016 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Ke Chen (main), Daniela Montaldi (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/bayesian-mechanisms-in-spatial-cognition-towards-real-world-capab |
| **Abstract** | Develops cognitively plausible spatial memory models. Bayesian localization and error correction. Neural recording evidence from rats. Extends LIDA cognitive architecture with probabilistic models. Robotic simulations. |
| **What They Built** | Extended LIDA cognitive architecture; Bayesian spatial models; robotic validation |
| **Tools/Frameworks** | LIDA cognitive architecture |
| **Datasets** | Rat neural recordings, human behavior data, VR environments |
| **Scope** | Very high -- interdisciplinary neuroscience + robotics |
| **PDF Size** | 19.3 MB |

---

## SECTION 5: ML FOR ELECTROPHYSIOLOGY / BIOSIGNAL PROCESSING

### 5.1 Use of Machine Learning to Analyse Auditory Evoked Electrophysiological Data
| Field | Detail |
|-------|--------|
| **Author** | Vicki Kennedy |
| **Year** | 2024 |
| **Degree** | Clinical Science Doctorate (ClinSciD) |
| **Department** | Psychology Communication and Human Neuroscience |
| **Supervisors** | Chris Plack (main) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/use-of-machine-learning-to-analyse-auditory-evoked-electrophysiol |
| **Abstract** | ML for auditory brainstem response (ABR) analysis. Wave V latency estimation (R2 = 0.8925), tinnitus classification (80% sensitivity, 81.82% specificity), noise exposure prediction (poor). |
| **What They Built** | ML classifiers for ABR analysis; tinnitus detection system |
| **Tools/Frameworks** | SVM, Gaussian process regression, ANNs, wavelet feature extraction |
| **Datasets** | ABR waveforms from tinnitus/non-tinnitus subjects |
| **Scope** | Moderate -- applied ML to clinical data |
| **PDF Size** | 2.59 MB |

---

## SECTION 6: EDGE / IoT / EMBEDDED

### 6.1 Sensing and Image Processing Methods for IoT Applications
| Field | Detail |
|-------|--------|
| **Author** | Yu Li |
| **Year** | 2023 |
| **Degree** | **MPhil (Master of Philosophy)** |
| **Department** | Electrical & Electronic Engineering |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/sensing-and-image-processing-methods-for-iot-applications |
| **Abstract** | Monitoring industrial production processes, baked potato quality assessment, oven anomaly detection. Uses HSV, LBP, OCSVM. |
| **What They Built** | IoT monitoring pipeline with image processing + sensor analysis |
| **Tools/Frameworks** | HSV, LBP, wavelet packets, OCSVM |
| **Datasets** | Camera images, thermocouple data, accelerometer data |
| **Scope** | Moderate (MPhil) -- applied image processing and ML |
| **PDF Size** | 4.79 MB |
| **NOTE** | **MPhil level -- somewhat closer to masters scope** |

---

## SECTION 7: REINFORCEMENT LEARNING / ROBOTICS (Adjacent)

### 7.1 Simulation Architectures for Reinforcement Learning applied to Robotics
| Field | Detail |
|-------|--------|
| **Author** | Diego Ferigo |
| **Year** | 2023 |
| **Degree** | PhD |
| **Department** | Computer Science |
| **Supervisors** | Angelo Cangelosi (main), Daniele Pucci (co) |
| **URL** | https://research.manchester.ac.uk/en/studentTheses/simulation-architectures-for-reinforcement-learning-applied-to-ro |
| **Abstract** | Software architecture for RL environments in robotics. Physics engine optimized for GPUs/TPUs using JAX. Push-recovery controller for iCub humanoid robot. |
| **What They Built** | RL simulation framework; JAX-based physics engine; iCub push-recovery controller |
| **Tools/Frameworks** | Gazebo Sim, JAX, iCub platform |
| **Datasets** | N/A (simulation) |
| **Scope** | High |
| **PDF Size** | 3.36 MB |

---

## SUMMARY TABLE: ALL THESES FOUND

| # | Title (abbreviated) | Author | Year | Degree | Department | Core Topic |
|---|---------------------|--------|------|--------|------------|------------|
| 1 | Modelling Neural Dynamics on Neuromorphic HW | Mollie Ward | 2024 | PhD | CS | Neuromorphic/SpiNNaker |
| 2 | Deep Spiking Neural Networks | Qian Liu | 2018 | PhD | CS | SNN/Deep Learning |
| 3 | Parallelisation of Neural Processing on Neuromorphic HW | Luca Peres | 2022 | PhD | CS | SpiNNaker/Parallel |
| 4 | Learning in Spiking Neural Networks | Sergio Davies | 2012 | PhD | CS | SNN/STDP |
| 5 | Stochastic Processes for Neuromorphic HW | Gabriel Fonseca Guerra | 2020 | PhD | CS | SpiNNaker/Loihi |
| 6 | Parallel Simulation of NNs on SpiNNaker | Xin Jin | 2010 | PhD | CS | SpiNNaker |
| 7 | Arithmetic Accelerators for Neuromorphic Processor | Mantas Mikaitis | 2020 | PhD | CS | SpiNNaker2/HW |
| 8 | Building Large-Scale SpiNNaker Machines | Jonathan Heathcote | 2016 | PhD | CS | SpiNNaker/Systems |
| 9 | Structural Plasticity on SpiNNaker | Petrut Bogdan | 2019 | PhD | CS | SpiNNaker/Plasticity |
| 10 | Real Time Spaun on SpiNNaker | Andrew Mundy | 2016 | PhD | CS | SpiNNaker/Brain Model |
| 11 | Biologically Inspired Neural Computation | Adam Perrett | 2022 | PhD | CS | SpiNNaker/Bio-inspired |
| 12 | Scalability and Robustness of ANNs | Evangelos Stromatias | 2016 | PhD | CS | SpiNNaker/Power |
| 13 | Plasticity in Large-Scale Neuromorphic Models | James Knight | 2016 | PhD | CS | SpiNNaker/Plasticity |
| 14 | Memristive Crossbar Arrays for ML | Manu V. Nair | 2015 | **MPhil** | EEE | Memristors/ML |
| 15 | Efficient Execution of CNNs on Low Power | Crefeda Rodrigues | 2020 | PhD | CS | Edge AI/Energy |
| 16 | Modular FPGA Systems | Anuj Vaishnav | 2020 | PhD | CS | FPGA |
| 17 | FPGA Virtualisation | Khoa Pham | 2020 | PhD | CS | FPGA |
| 18 | Reconfigurable HW for Heterogeneous Systems | Konstantinos Iordanou | 2023 | PhD | CS | FPGA/ML |
| 19 | Dynamic CPU ISA through FPGA Interlays | Jose Garcia Ordaz | 2018 | PhD | CS | FPGA/CPU |
| 20 | Unsupervised Image Feature Learning for CNNs | Richard Hankins | 2019 | PhD | EEE | Deep Learning/Vision |
| 21 | DL for Semantic Feature Extraction in Aerial Imagery | Ananya Gupta | 2020 | PhD | EEE | Deep Learning/Remote Sensing |
| 22 | Bayesian DL for Pulsar Classification | Alexandra Bonta | 2022 | **MSc Res** | Physics | Bayesian DL |
| 23 | Object Detection: Few-Shot, Vision-Language, ViT | Mengyuan Ma | 2025 | PhD | EEE | Computer Vision |
| 24 | Bayesian Mechanisms in Spatial Cognition | Tamas Madl | 2016 | PhD | CS | Computational Neuroscience |
| 25 | ML for Auditory Electrophysiological Data | Vicki Kennedy | 2024 | ClinSciD | Psychology | ML/Biosignal |
| 26 | Sensing and Image Processing for IoT | Yu Li | 2023 | **MPhil** | EEE | IoT/Image Processing |
| 27 | Simulation for RL in Robotics | Diego Ferigo | 2023 | PhD | CS | RL/Robotics |

---

## KEY OBSERVATIONS AND ANALYSIS

### 1. Degree Level Distribution
- **PhD theses:** 24 out of 27 (89%)
- **MPhil theses:** 2 (Memristive Crossbar Arrays; Sensing for IoT)
- **MSc by Research:** 1 (Bayesian DL for Pulsar Classification)
- **BSc / Undergraduate:** 0 found
- **Clinical Doctorate:** 1 (ML for Auditory Data)

**Manchester Research Explorer does NOT host undergraduate dissertations.** This is confirmed by the library guide which states the system is for "postgraduate research theses."

### 2. Supervisor Clustering
The neuromorphic theses are heavily concentrated around:
- **Steve Furber** (main or co-supervisor on 10+ theses) -- creator of SpiNNaker, former ICL/ARM architect
- **James Garside** (co-supervisor on many) -- SpiNNaker team
- **David Lester** (co-supervisor on several) -- SpiNNaker team
- **Oliver Rhodes** (newer supervisor, 2020s theses) -- SpiNNaker team
- **Dirk Koch** (FPGA theses) -- reconfigurable computing
- **Piotr Dudek** (vision/EEE theses) -- pixel processor arrays
- **Hujun Yin** (DL/vision theses) -- pattern recognition

### 3. Technology Stack Patterns
- **SpiNNaker** is the dominant platform across neuromorphic theses (appears in 13/13 core neuromorphic theses)
- **Intel Loihi** appears in 1 thesis (Fonseca Guerra 2020) as a comparison platform
- **MNIST** is the most common benchmark dataset
- **Python/PyNN/sPyNNaker** implied but rarely explicitly named (these are the standard SpiNNaker software tools)
- **NEURON simulator** used as reference for biological accuracy
- **No TensorFlow/PyTorch** in the neuromorphic theses (these use SpiNNaker-specific toolchains)

### 4. Temporal Distribution
- 2010: 1 thesis (early SpiNNaker work)
- 2012: 1
- 2015: 1
- 2016: 4 (peak year for first-generation SpiNNaker theses)
- 2018: 2
- 2019: 2
- 2020: 6 (peak overall year)
- 2022: 3
- 2023: 3
- 2024: 2
- 2025: 1

### 5. Scope Calibration for Undergrad Context

**These PhD theses are NOT representative of undergraduate scope.** A typical PhD thesis here represents:
- 3-4 years of full-time research
- Multiple novel contributions (often 3+ papers worth)
- World-first achievements (real-time Cortical Microcircuit, largest plastic SNN, 9000x speedup)
- Deep hardware expertise (chip-level design, FPGA virtualisation)

**The closest to undergraduate/masters scope would be:**
- The MPhil theses (Nair 2015, Li 2023) -- roughly 1-2 years, more focused scope
- The MSc by Research (Bonta 2022) -- ~1 year, single well-scoped contribution (one classifier, two datasets, clear metrics)

**For an undergraduate final-year project, a realistic scope would be approximately 1/4 to 1/6 of these PhD theses** -- e.g., implementing ONE existing SNN model on SpiNNaker and benchmarking it, or training ONE classifier on ONE dataset.

### 6. Research Gaps / What Was NOT Found
- No theses on **EEG classification with SNNs** (despite this being a natural combination)
- No theses on **gesture recognition** with neuromorphic hardware
- No theses on **event cameras / DVS** as student theses (publications exist but no theses)
- No theses on **TinyML / embedded ML** on microcontrollers (the "edge" work is at FPGA/mobile SoC level)
- No theses on **neuromorphic + BCI** integration
- No theses combining **SpiNNaker with real-world sensory data** beyond vision models
- Very few theses on **SpiNNaker2** (only Ward 2024 and Mikaitis 2020 touch on it)

---

## RECOMMENDED FOLLOW-UPS

1. **Manchester undergraduate dissertations** are NOT in the Research Explorer. They may be in:
   - Individual department archives (contact CS department directly)
   - Manchester's internal Blackboard/Canvas system (not public)
   - Supervisor portfolios (ask supervisors directly)

2. **For comparable undergraduate scope**, search instead:
   - Other UK universities' final year project repositories (e.g., Imperial, Edinburgh)
   - GitHub repositories tagged with "final year project" + neuromorphic
   - Conference papers from undergraduate research programs

3. **Manchester eScholar** (escholar.manchester.ac.uk) is a DIFFERENT system focused on learning resources, not theses. Do not confuse with Research Explorer.

4. **EThOS** (British Library) may have additional Manchester theses not in the Research Explorer.
