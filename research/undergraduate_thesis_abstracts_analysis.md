# Analysis of Undergraduate/Bachelor's Thesis Abstracts in SNN, Neuromorphic Computing, and ML

## How Do They State Objectives, Frame Contributions, and Structure Research Questions?

**Date:** 2026-02-25
**Scope:** 14 thesis-level works analyzed (bachelor's theses, honours theses, BEng final year projects, undergraduate honours papers)

---

## 1. Executive Summary

After analyzing 14 undergraduate-level thesis abstracts and project descriptions across SNN, neuromorphic computing, and machine learning, clear patterns emerge in how students frame their work. The dominant framing is **"we built/implemented X and evaluated/tested it on Y"** -- an engineering-oriented approach where the student constructs something (a network, a system, a calibration routine, a hardware configuration) and then measures how well it works. Very few undergraduate theses claim to "discover" something novel in the scientific sense. Instead, contributions are framed as: demonstrations that something *works* on a specific platform, empirical comparisons between approaches, or adaptations of existing techniques to new domains/hardware. Research questions, when stated explicitly, tend to take the form "Can approach A achieve task B on platform/dataset C?" rather than open-ended exploratory questions.

---

## 2. The 14 Theses Analyzed

### Thesis 1: Musical Pattern Recognition in Spiking Neural Networks
- **Author:** Matthew Rahtz
- **Type:** BEng Final Year Project
- **Institution:** Unknown UK university
- **Year:** ~2015
- **Source:** [GitHub README](https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks) and report at http://amid.fish/beng_project_report.pdf

**How objectives are stated:**
The project implements "a layer of spiking neurons which can differentiate between individual notes in a series of simple monophonic test audio sequences." The architecture derives from Peter Diehl's model for "unsupervised learning of digit recognition using spike-timing-dependent plasticity."

**What is claimed as the output/contribution:**
A working implementation of the first layer of an SNN for audio pattern recognition. The author candidly notes: "only a small portion of what was originally intended was actually achieved."

**Framing:** "We built X" -- specifically an implementation of an existing architecture adapted to a new domain (audio instead of digits). The honest acknowledgment of limited scope is notable and characteristic of BEng projects.

---

### Thesis 2: Spiking Neural Networks: A Biologically Informed Approach to Classification
- **Author:** Unknown (supervised by Erik Meijering)
- **Type:** Bachelor Honours Thesis
- **Institution:** University of New South Wales (UNSW), Australia
- **Year:** August 2022
- **Source:** [Publication page](https://imagescience.org/meijering/publications/1233/)

**How objectives are stated:**
Three explicitly stated aims:
1. "Compare spiking neural network performance against conventional artificial networks on classification problems"
2. "Explore new mechanisms for structural plasticity in artificial spiking networks inspired by the biological process of neurogenesis"
3. "Evaluate implications for understanding biological signal processing and AI's future direction"

**What is claimed as the output/contribution:**
"Empirical benchmarking data" comparing perceptrons vs. LIF spiking neurons, and the introduction of "an artificial neurogenesis mechanism" for modifying spiking network structure.

**Framing:** "We compared A vs B" combined with "we explored a new mechanism X." This is one of the more ambitious undergraduate theses, combining benchmarking with a novel (for the student) architectural contribution. The research question is implied: "How do spiking networks perform relative to traditional perceptron-based networks?" and "Can artificial neurogenesis improve spiking network classification performance?"

---

### Thesis 3: Binaural Sound Localization on Neuromorphic Hardware
- **Author:** Laura Kriener
- **Type:** Bachelor's Thesis (Bachelorarbeit)
- **Institution:** University of Heidelberg, Kirchhoff Institute for Physics (KIP)
- **Year:** 2014
- **Source:** [KIP Details](http://www.kip.uni-heidelberg.de/Veroeffentlichungen/details.php?id=3106)

**How objectives are stated:**
"The primary goal was demonstrating that a multi-frequency Jeffress model could operate effectively on neuromorphic hardware after addressing physical constraints of the chip and unexpected signal interactions."

**What is claimed as the output/contribution:**
- Developing compensation methods for hardware inhomogeneities and limited signal bandwidth
- Identifying and investigating "a previously unknown interaction between input signals that impaired ITD detection"
- Modifying the network architecture to reduce signal interaction effects
- Successfully demonstrating ITD detection on the Spikey neuromorphic microchip

**Framing:** "We built X and demonstrated it works on hardware Y, discovering problem Z along the way." This is a hardware-focused thesis where the contribution is getting a known algorithm (Jeffress model) to work on physical neuromorphic hardware despite its imperfections. The unexpected discovery of a signal interaction problem is presented as a bonus finding.

---

### Thesis 4: Firing States of Recurrent Leaky Integrate-and-Fire Networks
- **Author:** Agnes Korcsak-Gorzo
- **Type:** Bachelor's Thesis (Bachelorarbeit)
- **Institution:** University of Heidelberg, KIP
- **Year:** 2015
- **Source:** [KIP Details](http://www.kip.uni-heidelberg.de/Veroeffentlichungen/details.php?id=3155&lang=en)

**How objectives are stated:**
The research aimed to "examine firing patterns in current-based leaky integrate-and-fire networks, with particular focus on biologically plausible Asynchronous Irregular (AI) states used to model spontaneous activity in cortical regions."

**What is claimed as the output/contribution:**
- Developed current-based networks using PyNN across various configurations
- Using cross-correlation measures and interspike interval analysis, "three distinct firing modes were characterized"
- A mean-field approach predicted population firing rates showing "good agreement" between theoretical predictions and simulations
- Results enable output "suitable as input for probabilistic inference models"

**Framing:** "We investigated phenomenon X and characterized Y." This is more scientifically oriented than most undergraduate theses, characterizing network behavior across parameter space and validating theoretical predictions against simulation. The contribution is understanding rather than a built artifact.

---

### Thesis 5: Accelerated Classification in Hierarchical Neural Networks on Neuromorphic Hardware
- **Author:** Carola Fischer
- **Type:** Bachelor's Thesis (Bachelorarbeit)
- **Institution:** University of Heidelberg, KIP
- **Year:** 2017
- **Source:** [KIP Details](http://www.kip.uni-heidelberg.de/Veroeffentlichungen/details.php?id=3533)

**How objectives are stated:**
The thesis aimed to "implement two interconnected layers of a feedforward network on the Spikey neuromorphic chip for classifying MNIST digits."

**What is claimed as the output/contribution:**
- "Characterized synaptic connections between on-chip neurons to enable configurable network connectivity"
- "Systematically evaluated all neurons across both chip halves to maximize available computational resources"
- "Successfully demonstrated classification of an MNIST subset on-chip using an improved software framework"

**Framing:** "We implemented X on hardware Y and demonstrated classification of Z." Classic engineering thesis: take an existing algorithm (Boltzmann machines for classification), put it on specific hardware (Spikey chip), solve the practical problems, demonstrate it works.

---

### Thesis 6: Towards Spike-based Expectation Maximization in a Closed-Loop Setup on an Accelerated Neuromorphic Substrate
- **Author:** Felix Schneider
- **Type:** Bachelor's Thesis (Bachelorarbeit)
- **Institution:** University of Heidelberg, KIP
- **Year:** June 2018
- **Source:** [KIP Publications](https://www.kip.uni-heidelberg.de/Veroeffentlichungen/download.php/6229/temp/3814.pdf)

**How objectives are stated:**
Framed through a motivation that "learning experiments are generally time-consuming and computationally expensive on conventional computing machines, but can be efficiently emulated using application-specific circuits on neuromorphic hardware like the BrainScaleS system." The aim is implementing Spike-based Expectation Maximization (SEM) -- "where a population of neurons tries to find the hidden cause of spike patterns" -- in a closed-loop setup on BrainScaleS.

**What is claimed as the output/contribution:**
Implementation of the SEM algorithm on the accelerated BrainScaleS neuromorphic substrate in a closed-loop configuration.

**Framing:** "We implemented algorithm X on platform Y." The "Towards" in the title signals that this is progress toward a goal rather than a completed system, which is a common and honest framing for bachelor's work.

---

### Thesis 7: Neuromorphic Network-on-Chip Architecture for SNNs
- **Authors:** Team project (multiple students)
- **Type:** Undergraduate Final Year Project (4YP)
- **Institution:** University of Peradeniya, Sri Lanka
- **Year:** ~2022-2023
- **Source:** [Project page](https://cepdnaclk.github.io/e17-4yp-Neuromorphic-NoC-Architecture-for-SNNs/)

**How objectives are stated:**
"The core goal is to design and implement a Network-on-Chip (NoC) architecture based on the RISC-V instruction set architecture (ISA) which allows for hardware-level processing of spiking neural networks, and the implementation of the design on an FPGA."

**What is claimed as the output/contribution:**
- Customized RISC-V processing nodes with network interfaces
- 2D mesh Network-on-Chip with routing framework
- Specialized neuron bank hardware for offloading calculations
- Event-driven messaging mechanism for spike simulation
- The design bridges "programming flexibility and platform maturity" by leveraging open-source RISC-V

**Framing:** "We designed and built X." A pure engineering/design project: design a hardware architecture, implement it on FPGA, demonstrate it works. The contribution is the artifact itself and the demonstration that RISC-V can be augmented for SNN simulation.

---

### Thesis 8: Simple Spiking Neural Network with STDP (University Osnabruck Term Project)
- **Author:** C. Wolff et al.
- **Type:** University lecture term project/paper
- **Institution:** University of Osnabruck, Germany
- **Year:** ~2022
- **Source:** [GitHub](https://github.com/cowolff/Simple-Spiking-Neural-Network-STDP)

**How objectives are stated:**
The team sought to "obtain a better understanding of SNNs" by comparing their performance in image classification against traditional fully-connected artificial neural networks using the MNIST dataset.

**What is claimed as the output/contribution:**
- SNNs achieved "pretty good classification performance after only one epoch of training"
- Performance plateaued quickly without significant improvement beyond initial training
- Classical ANNs with dense layers "substantially outperformed SNNs within a few epochs"
- SNNs showed diminishing returns with increased neuron counts

**Framing:** "We compared A vs B to understand X." The primary contribution is the empirical comparison and the understanding gained from it, not the implementation itself. The research question is: how do SNNs compare to ANNs on image classification?

---

### Thesis 9: Spiking Neural Networks for Image Classification
- **Authors:** Osaze Shears, Ahmad Hossein Yazdani
- **Type:** Advanced Machine Learning Course Project (graduate-level course, but relevant pattern)
- **Institution:** Virginia Tech
- **Year:** November 2020
- **Source:** [Project website](https://oshears.github.io/adv-ml-2020-snn-project/)

**How objectives are stated:**
"The group reimplements tests in the BindsNET framework using different neural models, encoding methods, and training techniques to study how these factors affect the SNN model accuracy." Specific objectives include: "measuring their accuracy in classifying the MNIST and CIFAR10 benchmarks, comparing each of the networks' memory cost for storing weights, and comparing cost of performing computations with each network."

**What is claimed as the output/contribution:**
Empirical analysis of multiple SNN configurations, providing insights into which design choices optimize accuracy.

**Framing:** "We reimplemented and evaluated X to study the effect of factors Y and Z." Systematic comparison and evaluation, not novel creation.

---

### Thesis 10: Learning in Biologically Plausible Neural Networks
- **Author:** Draco (Yunlong) Xu
- **Type:** Undergraduate Honours Thesis
- **Institution:** University of Rochester, Department of Mathematics
- **Year:** 2023
- **Source:** [PDF](https://www.sas.rochester.edu/mth/undergraduate/honorspaperspdfs/d_xu23.pdf)

**How objectives are stated:**
The thesis "presents a thorough review of learning processes in biologically plausible neural networks, with an emphasis on spiking neural networks." It demonstrates "the implementation and training of CDNNs (Constrained Deep Neural Networks) and introduces a novel learning method for RSNNs (Reduced Spiking Neural Networks)." Additionally, the work "proposes an innovative approach to compare Spiking Neural Networks and Constrained Deep Neural Networks."

**What is claimed as the output/contribution:**
- A review of biologically plausible learning
- Implementation and training of CDNNs
- A novel learning method for RSNNs
- A new comparison methodology between SNNs and CDNNs

**Framing:** "We reviewed X, implemented Y, and proposed a novel method Z." This is one of the more ambitious undergraduate theses, combining literature review with implementation and a claim of novelty. The framing uses stronger language ("novel," "innovative") than most.

---

### Thesis 11: Evaluation of Convolutional Neural Network Performance Using Synthetic Data
- **Author:** Jeonghyun Son
- **Type:** Bachelor Thesis
- **Institution:** HAW Hamburg (University of Applied Sciences Hamburg)
- **Year:** October 2019
- **Source:** [PDF](https://reposit.haw-hamburg.de/bitstream/20.500.12738/9168/1/Bachelorthesis_JeonghyunSon.pdf)

**How objectives are stated:**
"One of the limitations of supervised learning in deep learning algorithm is to gather and label a large set of data. In this document, the approach to solve this limitation is presented by using synthetic data."

**What is claimed as the output/contribution:**
- Created a 3D traffic scene with bicycles using THREE.js to generate synthetic training data
- Trained a CNN on synthetic data for image classification
- "At the end, the performance of convolutional neural network model is evaluated on real image dataset"

**Framing:** "We built X to address limitation Y, and evaluated the performance on Z." The abstract starts by identifying a problem (limited training data), proposes a solution (synthetic data), implements it, and evaluates it. Classic problem-solution-evaluation structure.

---

### Thesis 12: A Deep Learning Prediction Model for Object Classification
- **Author:** Nordin Sahla
- **Type:** Bachelor Thesis (Mechanical Engineering)
