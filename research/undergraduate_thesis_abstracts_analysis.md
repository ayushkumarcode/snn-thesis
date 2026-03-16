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
- **Institution:** TU Delft
- **Year:** ~2020
- **Source:** [TU Delft Repository](https://repository.tudelft.nl/islandora/object/uuid:f7667cb4-70d4-4b82-ac1b-75df476655cd)

**How objectives are stated:**
"The aim of this research is to investigate whether a usable relation exist between object features such as size or shape, and barcode location, that can be used to robustly identify objects in a bin."

**What is claimed as the output/contribution:**
"A deep convolutional neural network (CNN) is built in MATLAB and trained on a labeled dataset of thousand product images from various perspectives." Results: "training set accuracy reaches 100%, a maximum validation accuracy of only 45% is achieved." Honest conclusion: "A larger dataset is required to reduce overfitting."

**Framing:** "We investigated whether X and built Y to test it." The framing is exploratory/investigative: can we do this? The honest reporting of underwhelming results (45% validation accuracy) is characteristic of undergraduate work and is presented without apology.

---

### Thesis 13: Artificial Neural Networks and Deep Learning: Possibilities and Limits
- **Author:** Seila Laakso
- **Type:** Bachelor Thesis
- **Institution:** Oulu University of Applied Sciences, Finland
- **Year:** Autumn 2022
- **Source:** [Theseus repository](https://www.theseus.fi/handle/10024/779806)

**How objectives are stated:**
The thesis "addresses artificial intelligence, artificial neural networks and deep learning" and explores "how artificial neural networks function and how they are structured." It also "covers practical applications of artificial neural network technology and how it is used in different fields of industry."

**What is claimed as the output/contribution:**
- Literature review and explanation of ANN/DL fundamentals
- A practical project: "ProGAN DaliA is a progressive generative adversarial neural network, which creates new artwork from a dataset of art pieces"
- Identifying that "unpredictability and its Blackbox like operation principle" remain critical challenges

**Framing:** "We reviewed the field and built a demonstration project." This is a literature-review-heavy thesis with a practical component attached as an annex. The contribution is educational/explanatory rather than research-oriented.

---

### Thesis 14: Exploring the Chemical Universe with Spiking Neural Networks
- **Author:** Philipp Kuppers
- **Type:** Bachelor's Thesis (Computing Science)
- **Institution:** Radboud University, Netherlands
- **Year:** 2024
- **Source:** [PDF](https://www.cs.ru.nl/bachelors-theses/2024/Philipp_K%C3%BCppers___1073738___Exploring_the_Chemical_Universe_with_Spiking_Neural_Networks.pdf)

**How objectives are stated:**
The thesis explores whether SNNs can be applied to molecular property prediction, specifically drug discovery. The problem is framed as: "ANNs are only able to consider molecules in the range of billions when finding hit candidates due to their high requirements of compute resources, energy and time." The research transforms the problem "into a binary classification task, dividing the molecules into active and inactive molecules."

**What is claimed as the output/contribution:**
Application of SNNs (trained using surrogate gradients) to molecular property prediction as binary classification. The thesis explores whether SNNs can match or improve upon traditional ANN approaches for chemical screening.

**Framing:** "We explored whether approach A can be applied to domain B." The "Exploring" in the title signals this is an investigative study rather than a definitive solution.

---

## 3. Pattern Analysis: How Do Undergraduate Theses State Objectives?

### 3.1 Common Phrasing for Objectives

The most frequently used phrasings for stating objectives fall into these categories:

**Direct aim statements:**
- "The aim of this research is to investigate whether..."  (TU Delft, Thesis 12)
- "The primary goal was demonstrating that..."  (Heidelberg KIP, Thesis 3)
- "The core goal is to design and implement..."  (Peradeniya, Thesis 7)
- "The thesis aimed to implement..."  (Heidelberg KIP, Thesis 5)

**Problem-motivated framing:**
- "One of the limitations of X is Y. In this document, the approach to solve this limitation is presented by..."  (HAW Hamburg, Thesis 11)
- "Learning experiments are generally time-consuming... but can be efficiently emulated using..."  (Heidelberg KIP, Thesis 6)

**Understanding-oriented framing:**
- "The team sought to obtain a better understanding of X by comparing..."  (Osnabruck, Thesis 8)
- "The research aimed to examine X, with particular focus on Y"  (Heidelberg KIP, Thesis 4)

**Exploration framing:**
- "This thesis presents a thorough review of X... and introduces a novel method for Y"  (Rochester, Thesis 10)
- Uses "Exploring" or "Towards" in the title to signal investigative nature  (Radboud Thesis 14, Heidelberg Thesis 6)

### 3.2 What Do They Claim as Output/Contribution?

| Contribution Type | Count | Examples |
|---|---|---|
| "We implemented X on platform Y and demonstrated it works" | 5 | Theses 1, 3, 5, 6, 7 |
| "We compared/evaluated A vs B and report results" | 4 | Theses 2, 8, 9, 11 |
| "We investigated/explored whether X can do Y" | 3 | Theses 12, 14, 4 |
| "We reviewed the field and built a demo" | 1 | Thesis 13 |
| "We proposed/introduced a novel method" | 1 | Thesis 10 |

The overwhelming pattern is **implementation + evaluation** rather than **discovery**. Students build something and test it. The contribution is the working system and the empirical results, not a theoretical advance.

### 3.3 How Do They Frame Their Work?

**Framework 1: "We built X and evaluated it" (Most Common)**
This is the dominant pattern. The thesis implements a system or algorithm and tests it on a benchmark or hardware platform.
- "We implement X on the Spikey neuromorphic chip and demonstrate classification of MNIST"
- "We implement binaural sound localization on neuromorphic hardware using the Jeffress model"
- "A deep convolutional neural network is built in MATLAB and trained on..."

**Framework 2: "We compared A vs B"**
The thesis systematically compares approaches, architectures, or configurations.
- "Compare spiking neural network performance against conventional artificial networks on classification problems"
- "The team sought to obtain a better understanding of SNNs by comparing their performance... against traditional fully-connected artificial neural networks"

**Framework 3: "We investigated/explored whether X"**
The thesis asks a question and seeks evidence.
- "Investigate whether a usable relation exists between object features and barcode location"
- "Exploring" whether SNNs can be applied to molecular property prediction

**Framework 4: "We discovered/found Y" (Rare)**
Very few undergraduate theses frame their contribution as a discovery. The closest examples:
- "Identifying and investigating a previously unknown interaction between input signals that impaired ITD detection" (Thesis 3, but this was a secondary finding, not the primary aim)
- "Three distinct firing modes were characterized" (Thesis 4, a characterization study)

### 3.4 Typical Research Question Structure

Research questions in these theses, when stated explicitly, follow predictable patterns:

1. **Can X achieve Y?**
   - "Can a multi-frequency Jeffress model operate effectively on neuromorphic hardware?"
   - "Can SNNs be applied to molecular property prediction?"

2. **How does X compare to Y on task Z?**
   - "How do spiking networks perform relative to traditional perceptron-based networks?"
   - "How do these factors [neural models, encoding methods] affect the SNN model accuracy?"

3. **What happens when we do X?**
   - "What are the firing patterns in current-based LIF networks under different configurations?"

4. **Can we solve problem X using approach Y?**
   - "Can we solve the limited training data problem by using synthetic data?"

Most research questions are **closed** (yes/no answerable) rather than open-ended. They ask "can we?" or "does it work?" rather than "why?" or "how does the underlying mechanism function?"

---

## 4. Heidelberg KIP Bachelor's Theses: A Special Case

The Heidelberg KIP theses deserve separate analysis because they represent a specific and prolific tradition of undergraduate research on neuromorphic hardware. From the complete listing (2010-2025), there are 70+ bachelor's theses, almost all focused on the BrainScaleS/Spikey neuromorphic hardware platforms.

**Common title patterns from KIP theses:**
- "Towards X on BrainScaleS" (signals progress, not completion)
- "Characterization of X on neuromorphic hardware" (systematic measurement)
- "Calibration of X on Y" (getting hardware to work correctly)
- "Implementation of X for neuromorphic hardware" (building software/hardware)
- "Development of X for the BrainScaleS system" (engineering contribution)
- "Testing X for neuromorphic hardware" (verification and validation)

**Key observations:**
- KIP theses are overwhelmingly **hardware-focused engineering work**: calibrating chips, characterizing circuits, implementing software frameworks, testing components
- The framing is almost always "I built/implemented/characterized X on hardware platform Y"
- Titles using "Towards" acknowledge incomplete progress, which is honest and accepted
- Scientific investigation (e.g., "Investigating Competitive Dynamics," "Firing States of Recurrent Networks") is less common but does occur
- Many theses contribute directly to a larger research infrastructure rather than standalone results

**Full listing of selected KIP bachelor thesis titles (for reference):**
- "Real-time Image Classification on Analog Neuromorphic Hardware" (Ebert, 2021)
- "Multi-Single-Chip Training of Spiking Neural Networks with BrainScaleS-2" (Straub, 2023)
- "Gradient-Based Training of Multi-Compartmental Neuron Models on BrainScales-2" (Janz, 2025)
- "Event-based Learning of Synaptic Delays and Arbitrary Topologies" (Fischer, 2025)
- "Structural Plasticity for Feature Selection in Auditory Stimuli on Neuromorphic Hardware" (Kreft, 2019)
- "Solving Map Coloring Problems on Analog Neuromorphic Hardware" (Steidel, 2018)
- "Spike-based Expectation Maximization on the HICANN-DLSv2 Neuromorphic Chip" (Spilger, 2018)
- "Boltzmann Sampling with Neuromorphic Hardware" (Stockel, 2015)
- "Analysis of the Liquid Computing Paradigm on a Neuromorphic Hardware System" (Probst, 2011)

Source: http://www.kip.uni-heidelberg.de/vision/publications/mscbsc/

---

## 5. Key Takeaways for Your Own Thesis

### 5.1 Safe and Expected Framing for an Undergraduate Thesis

Based on these 14 examples, the following framing would be entirely standard and expected:

**"This thesis implements [algorithm/approach X] for [task Y] using [framework/platform Z], and evaluates its performance on [dataset/benchmark W]."**

Variations:
- "This thesis investigates the feasibility of using [X] for [Y] by implementing and evaluating [Z]."
- "This thesis compares [approach A] and [approach B] for [task C] on [dataset D], analyzing [metrics E and F]."
- "This work presents an implementation of [X] on [platform Y] and characterizes its performance with respect to [Z]."

### 5.2 What You Do NOT Need to Claim

- You do NOT need to claim a novel scientific discovery
- You do NOT need to beat state-of-the-art benchmarks
- You do NOT need to propose a new algorithm or architecture
- You DO need to show you built something, tested it systematically, and can interpret the results
- Honest reporting of negative or underwhelming results is perfectly acceptable (see Thesis 12: 45% validation accuracy)

### 5.3 The "Towards" Escape Hatch

Many Heidelberg KIP theses use "Towards" in their titles, which is a signal that the work represents progress toward a larger goal rather than a completed system. This is a perfectly legitimate framing for undergraduate work. Examples:
- "Towards Spike-based Expectation Maximization in a Closed-Loop Setup..."
- "Towards Balanced Random Networks on the BrainScaleS I System"
- "Towards Fast Iterative Learning on the BrainScaleS Neuromorphic Hardware System"

### 5.4 Acceptable Contribution Types (Ranked by Ambition)

1. **Implementation + Demonstration:** "I implemented X and showed it works" (lowest bar, very common)
2. **Implementation + Evaluation:** "I implemented X, evaluated it on Y, and report the results" (most common)
3. **Comparison:** "I compared approaches A, B, C on task X and analyzed trade-offs" (common)
4. **Adaptation:** "I adapted known technique X to new domain/platform Y" (common in KIP)
5. **Characterization:** "I systematically characterized behavior X across parameters Y" (KIP-style)
6. **Novel method + Evaluation:** "I proposed a new approach X and evaluated it" (less common, more ambitious)

### 5.5 Abstract Structure Template (Derived from Analysis)

The typical undergraduate thesis abstract follows this structure:

```
[1-2 sentences: Context/Motivation]
  - "Spiking neural networks offer energy-efficient computation but..."
  - "Neuromorphic hardware enables X but faces challenges Y..."

[1 sentence: Gap/Problem]
  - "However, limited research has applied X to domain Y"
  - "It remains unclear whether approach X can achieve Y"

[1-2 sentences: What this thesis does]
  - "This thesis implements/investigates/compares..."
  - "In this work, we present/develop/evaluate..."

[1-2 sentences: Method]
  - "We use framework X to implement Y on platform Z"
  - "We train a network using STDP on the MNIST dataset"

[1-2 sentences: Key results]
  - "Results show that X achieves Y% accuracy on Z"
  - "We demonstrate successful classification on neuromorphic hardware"

[Optional: 1 sentence on implications]
  - "These results suggest that X is a viable approach for Y"
```

---

## 6. Sources

- [Musical Pattern Recognition in SNNs - GitHub](https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks)
- [BEng Report](http://amid.fish/beng_project_report.pdf)
- [UNSW Honours Thesis - SNN Classification](https://imagescience.org/meijering/publications/1233/)
- [Heidelberg KIP Thesis List](http://www.kip.uni-heidelberg.de/vision/publications/mscbsc/)
- [KIP - Kriener Thesis Details](http://www.kip.uni-heidelberg.de/Veroeffentlichungen/details.php?id=3106)
- [KIP - Korcsak-Gorzo Thesis Details](http://www.kip.uni-heidelberg.de/Veroeffentlichungen/details.php?id=3155&lang=en)
- [KIP - Fischer Thesis Details](http://www.kip.uni-heidelberg.de/Veroeffentlichungen/details.php?id=3533)
- [Peradeniya NoC Architecture Project](https://cepdnaclk.github.io/e17-4yp-Neuromorphic-NoC-Architecture-for-SNNs/)
- [Osnabruck SNN-STDP Project](https://github.com/cowolff/Simple-Spiking-Neural-Network-STDP)
- [Virginia Tech SNN Image Classification](https://oshears.github.io/adv-ml-2020-snn-project/)
- [Rochester Honours Thesis - Xu](https://www.sas.rochester.edu/mth/undergraduate/honorspaperspdfs/d_xu23.pdf)
- [HAW Hamburg - Son Thesis](https://reposit.haw-hamburg.de/bitstream/20.500.12738/9168/1/Bachelorthesis_JeonghyunSon.pdf)
- [TU Delft - Sahla Thesis](https://repository.tudelft.nl/islandora/object/uuid:f7667cb4-70d4-4b82-ac1b-75df476655cd)
- [Oulu UAS - Laakso Thesis](https://www.theseus.fi/handle/10024/779806)
- [Radboud - Kuppers Thesis](https://www.cs.ru.nl/bachelors-theses/2024/Philipp_K%C3%BCppers___1073738___Exploring_the_Chemical_Universe_with_Spiking_Neural_Networks.pdf)
- [BrainScaleS Publications](https://brainscales.kip.uni-heidelberg.de/jss/Publications)
- [IIT Guwahati SNN Implementation](https://github.com/Shikhargupta/Spiking-Neural-Network)
- [Jegp Master's Thesis](https://github.com/Jegp/thesis)

---

## 7. Research Gaps and Confidence Assessment

**High confidence findings:**
- The dominant framing is "implemented + evaluated" (based on 14 direct examples)
- Undergraduate theses rarely claim scientific discovery (based on all examples)
- "Towards" is an accepted title prefix for incomplete work (based on multiple KIP examples)
- Honest reporting of limited results is normal and accepted (based on Theses 1, 12)

**Medium confidence findings:**
- Research questions tend to be closed-form "can we?" questions (inferred from abstracts, but many theses don't state explicit research questions)
- The UNSW thesis and Rochester thesis represent the more ambitious end of undergraduate work

**Gaps:**
- Could not access the full PDF text of the amid.fish BEng report (file too large for web fetch)
- Could not access most Heidelberg KIP thesis PDFs (compressed format not parseable)
- The Brno VUT thesis PDF was not parseable
- The Radboud thesis PDF was not parseable
- Limited access to actual thesis introductions (most analysis is based on abstracts and project descriptions)

**Recommended follow-ups:**
- Download the amid.fish PDF directly and read the abstract/introduction sections
- Access Heidelberg KIP theses through university library if possible
- Search ProQuest Dissertations and Theses for additional examples
- Look at ETH Zurich and TU Munich bachelor thesis repositories for additional ML/SNN theses
