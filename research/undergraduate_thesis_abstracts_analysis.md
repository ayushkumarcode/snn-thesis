# How Do Undergrad Theses Actually Frame Their Work?

i went through 14 undergraduate-level thesis abstracts (bachelor's, honours, BEng final year projects) in SNN, neuromorphic computing, and ML to figure out how students actually state objectives and frame contributions.

the big pattern: the dominant framing is **"we built/implemented X and evaluated/tested it on Y"** -- basically an engineering approach where you build something and measure how well it works. almost nobody at the undergrad level claims to "discover" something novel in a scientific sense. instead, contributions are things like: demonstrating something works on a specific platform, empirical comparisons between approaches, or adapting existing techniques to new domains/hardware. research questions, when stated at all, tend to be "Can approach A achieve task B on dataset C?" rather than big open-ended questions.

---

## The 14 Theses

### Thesis 1: Musical Pattern Recognition in Spiking Neural Networks
- **Author:** Matthew Rahtz
- **Type:** BEng Final Year Project
- **Institution:** Unknown UK university
- **Year:** ~2015
- **Source:** [GitHub](https://github.com/mrahtz/musical-pattern-recognition-in-spiking-neural-networks) and report at http://amid.fish/beng_project_report.pdf

**Objectives:** implements "a layer of spiking neurons which can differentiate between individual notes in a series of simple monophonic test audio sequences." architecture comes from Diehl's STDP digit recognition model.

**Contribution:** a working implementation of the first layer of an SNN for audio pattern recognition. the author honestly notes: "only a small portion of what was originally intended was actually achieved." -- i appreciate this kind of honesty.

**Framing:** "we built X" -- an implementation of an existing architecture adapted to a new domain (audio instead of digits).

---

### Thesis 2: Spiking Neural Networks: A Biologically Informed Approach to Classification
- **Author:** Unknown (supervised by Erik Meijering)
- **Type:** Bachelor Honours Thesis
- **Institution:** UNSW, Australia
- **Year:** August 2022
- **Source:** [Publication page](https://imagescience.org/meijering/publications/1233/)

**Objectives (explicitly stated):**
1. "Compare spiking neural network performance against conventional artificial networks on classification problems"
2. "Explore new mechanisms for structural plasticity in artificial spiking networks inspired by the biological process of neurogenesis"
3. "Evaluate implications for understanding biological signal processing and AI's future direction"

**Contribution:** empirical benchmarking data comparing perceptrons vs LIF spiking neurons, plus introduction of "an artificial neurogenesis mechanism."

**Framing:** "we compared A vs B" combined with "we explored new mechanism X." one of the more ambitious undergrad theses, combining benchmarking with a novel architectural contribution. implied research question: how do spiking networks perform vs traditional perceptrons, and can artificial neurogenesis help?

---

### Thesis 3: Binaural Sound Localization on Neuromorphic Hardware
- **Author:** Laura Kriener
- **Type:** Bachelorarbeit
- **Institution:** University of Heidelberg, KIP
- **Year:** 2014
- **Source:** [KIP](http://www.kip.uni-heidelberg.de/Veroeffentlichungen/details.php?id=3106)

**Objectives:** "demonstrating that a multi-frequency Jeffress model could operate effectively on neuromorphic hardware after addressing physical constraints of the chip and unexpected signal interactions."

**Contribution:**
- compensation methods for hardware inhomogeneities and limited signal bandwidth
- identifying "a previously unknown interaction between input signals that impaired ITD detection"
- modifying architecture to reduce signal interactions
- successful ITD detection on Spikey chip

**Framing:** "we built X and demonstrated it works on hardware Y, discovering problem Z along the way." hardware-focused -- contribution is getting a known algorithm to work on physical neuromorphic hardware despite its quirks. the unexpected discovery of a signal interaction was a bonus.

---

### Thesis 4: Firing States of Recurrent Leaky Integrate-and-Fire Networks
- **Author:** Agnes Korcsak-Gorzo
- **Type:** Bachelorarbeit
- **Institution:** University of Heidelberg, KIP
- **Year:** 2015
- **Source:** [KIP](http://www.kip.uni-heidelberg.de/Veroeffentlichungen/details.php?id=3155&lang=en)

**Objectives:** "examine firing patterns in current-based leaky integrate-and-fire networks, with particular focus on biologically plausible Asynchronous Irregular states used to model spontaneous activity in cortical regions."

**Contribution:**
- developed current-based networks using PyNN across various configs
- "three distinct firing modes were characterized" using cross-correlation and ISI analysis
- mean-field approach predicted population firing rates with "good agreement" between theory and simulation
- results enable output "suitable as input for probabilistic inference models"

**Framing:** "we investigated phenomenon X and characterized Y." more scientifically oriented than most undergrad theses -- characterizing behavior across parameter space and validating theory against simulation. the contribution is understanding, not just an artifact.

---

### Thesis 5: Accelerated Classification in Hierarchical Neural Networks on Neuromorphic Hardware
- **Author:** Carola Fischer
- **Type:** Bachelorarbeit
- **Institution:** University of Heidelberg, KIP
- **Year:** 2017
- **Source:** [KIP](http://www.kip.uni-heidelberg.de/Veroeffentlichungen/details.php?id=3533)

**Objectives:** "implement two interconnected layers of a feedforward network on the Spikey neuromorphic chip for classifying MNIST digits."

**Contribution:**
- "characterized synaptic connections between on-chip neurons"
- "systematically evaluated all neurons across both chip halves"
- "successfully demonstrated classification of an MNIST subset on-chip"

**Framing:** "we implemented X on hardware Y and demonstrated classification of Z." classic engineering thesis: take existing algorithm (Boltzmann machines), put on specific hardware (Spikey chip), solve the practical problems, demo it works.

---

### Thesis 6: Towards Spike-based Expectation Maximization on BrainScaleS
- **Author:** Felix Schneider
- **Type:** Bachelorarbeit
- **Institution:** University of Heidelberg, KIP
- **Year:** June 2018
- **Source:** [KIP](https://www.kip.uni-heidelberg.de/Veroeffentlichungen/download.php/6229/temp/3814.pdf)

**Objectives:** implementing SEM (Spike-based Expectation Maximization) -- "where a population of neurons tries to find the hidden cause of spike patterns" -- in a closed-loop setup on BrainScaleS.

**Contribution:** implementation of the SEM algorithm on BrainScaleS in a closed-loop configuration.

**Framing:** "we implemented algorithm X on platform Y." the "Towards" in the title signals progress toward a goal rather than a completed system -- common and honest framing for bachelor's work.

---

### Thesis 7: Neuromorphic Network-on-Chip Architecture for SNNs
- **Authors:** Team project
- **Type:** 4YP
- **Institution:** University of Peradeniya, Sri Lanka
- **Year:** ~2022-2023
- **Source:** [Project page](https://cepdnaclk.github.io/e17-4yp-Neuromorphic-NoC-Architecture-for-SNNs/)

**Objectives:** "design and implement a Network-on-Chip architecture based on RISC-V ISA which allows for hardware-level processing of spiking neural networks, and implement it on an FPGA."

**Contribution:**
- customized RISC-V processing nodes with network interfaces
- 2D mesh NoC with routing framework
- specialized neuron bank hardware
- event-driven messaging for spike simulation

**Framing:** "we designed and built X." pure engineering project. the contribution is the artifact itself and showing RISC-V can be augmented for SNN simulation.

---

### Thesis 8: Simple Spiking Neural Network with STDP (Osnabruck)
- **Author:** C. Wolff et al.
- **Type:** University lecture term project
- **Institution:** University of Osnabruck
- **Year:** ~2022
- **Source:** [GitHub](https://github.com/cowolff/Simple-Spiking-Neural-Network-STDP)

**Objectives:** "obtain a better understanding of SNNs" by comparing performance against traditional ANNs on MNIST.

**Contribution:**
- SNNs got "pretty good classification performance after only one epoch"
- performance plateaued quickly after that
- classical ANNs "substantially outperformed SNNs within a few epochs"
- SNNs showed diminishing returns with increased neuron counts

**Framing:** "we compared A vs B to understand X." primary contribution is the empirical comparison and understanding, not the implementation itself.

---

### Thesis 9: Spiking Neural Networks for Image Classification
- **Authors:** Osaze Shears, Ahmad Hossein Yazdani
- **Type:** Advanced ML Course Project (grad course, but useful pattern)
- **Institution:** Virginia Tech
- **Year:** November 2020
- **Source:** [Project website](https://oshears.github.io/adv-ml-2020-snn-project/)

**Objectives:** "reimplements tests in the BindsNET framework using different neural models, encoding methods, and training techniques to study how these factors affect SNN model accuracy."

**Contribution:** empirical analysis of multiple SNN configurations.

**Framing:** "we reimplemented and evaluated X to study factors Y and Z."

---

### Thesis 10: Learning in Biologically Plausible Neural Networks
- **Author:** Draco (Yunlong) Xu
- **Type:** Undergraduate Honours Thesis
- **Institution:** University of Rochester, Dept of Mathematics
- **Year:** 2023
- **Source:** [PDF](https://www.sas.rochester.edu/mth/undergraduate/honorspaperspdfs/d_xu23.pdf)

**Objectives:** "presents a thorough review of learning processes in biologically plausible neural networks" and "introduces a novel learning method for RSNNs" plus "proposes an innovative approach to compare SNNs and CDNNs."

**Contribution:**
- review of biologically plausible learning
- implementation and training of CDNNs
- novel learning method for RSNNs
- new comparison methodology between SNNs and CDNNs

**Framing:** "we reviewed X, implemented Y, and proposed novel method Z." one of the more ambitious undergrad theses -- uses stronger language ("novel," "innovative") than most.

---

### Thesis 11: Evaluation of CNN Performance Using Synthetic Data
- **Author:** Jeonghyun Son
- **Type:** Bachelor Thesis
- **Institution:** HAW Hamburg
- **Year:** October 2019
- **Source:** [PDF](https://reposit.haw-hamburg.de/bitstream/20.500.12738/9168/1/Bachelorthesis_JeonghyunSon.pdf)

**Objectives:** "One of the limitations of supervised learning in deep learning algorithm is to gather and label a large set of data. In this document, the approach to solve this limitation is presented by using synthetic data."

**Contribution:**
- created 3D traffic scene with bicycles using THREE.js for synthetic training data
- trained CNN on synthetic data
- evaluated on real images

**Framing:** "we built X to address limitation Y, evaluated on Z." classic problem-solution-evaluation.

---

### Thesis 12: A Deep Learning Prediction Model for Object Classification
- **Author:** Nordin Sahla
- **Type:** Bachelor Thesis (Mechanical Engineering)
- **Institution:** TU Delft
- **Year:** ~2020
- **Source:** [TU Delft Repository](https://repository.tudelft.nl/islandora/object/uuid:f7667cb4-70d4-4b82-ac1b-75df476655cd)

**Objectives:** "investigate whether a usable relation exists between object features such as size or shape, and barcode location, that can be used to robustly identify objects in a bin."

**Contribution:** built a CNN in MATLAB, trained on a thousand product images. results: "training set accuracy reaches 100%, a maximum validation accuracy of only 45% is achieved." honest conclusion: "A larger dataset is required to reduce overfitting."

**Framing:** "we investigated whether X and built Y to test it." reporting of underwhelming results without apology -- this is totally fine for undergrad work.

---

### Thesis 13: Artificial Neural Networks and Deep Learning: Possibilities and Limits
- **Author:** Seila Laakso
- **Type:** Bachelor Thesis
- **Institution:** Oulu UAS, Finland
- **Year:** Autumn 2022
- **Source:** [Theseus](https://www.theseus.fi/handle/10024/779806)

**Objectives:** "addresses artificial intelligence, artificial neural networks and deep learning" and covers "practical applications."

**Contribution:**
- lit review of ANN/DL fundamentals
- practical project: "ProGAN DaliA" -- a ProGAN that creates new artwork from a dataset
- identified "unpredictability and its Blackbox like operation principle" as challenges

**Framing:** "we reviewed the field and built a demo." literature-review-heavy with a practical component attached.

---

### Thesis 14: Exploring the Chemical Universe with Spiking Neural Networks
- **Author:** Philipp Kuppers
- **Type:** Bachelor's Thesis (Computing Science)
- **Institution:** Radboud University, Netherlands
- **Year:** 2024
- **Source:** [PDF](https://www.cs.ru.nl/bachelors-theses/2024/Philipp_K%C3%BCppers___1073738___Exploring_the_Chemical_Universe_with_Spiking_Neural_Networks.pdf)

**Objectives:** can SNNs be applied to molecular property prediction for drug discovery? frame: "ANNs are only able to consider molecules in the range of billions due to high compute/energy/time requirements." transforms this into binary classification (active vs inactive molecules).

**Contribution:** application of surrogate-gradient-trained SNNs to molecular property prediction as binary classification.

**Framing:** "we explored whether approach A can be applied to domain B." the "Exploring" signals an investigative study.

---

## Pattern Analysis

### How Objectives Get Stated

most common phrasings fall into these buckets:

**Direct aim:**
- "The aim of this research is to investigate whether..." (TU Delft)
- "The primary goal was demonstrating that..." (Heidelberg KIP)
- "The core goal is to design and implement..." (Peradeniya)

**Problem-motivated:**
- "One of the limitations of X is Y. In this document, the approach to solve this limitation is presented by..." (HAW Hamburg)

**Understanding-oriented:**
- "The team sought to obtain a better understanding of X by comparing..." (Osnabruck)

**Exploratory:**
- uses "Exploring" or "Towards" in the title (Radboud, Heidelberg)

### What Gets Claimed as Contribution

| Contribution Type | Count | Examples |
|---|---|---|
| "We implemented X on platform Y and showed it works" | 5 | Theses 1, 3, 5, 6, 7 |
| "We compared/evaluated A vs B and report results" | 4 | Theses 2, 8, 9, 11 |
| "We investigated/explored whether X can do Y" | 3 | Theses 12, 14, 4 |
| "We reviewed the field and built a demo" | 1 | Thesis 13 |
| "We proposed a novel method" | 1 | Thesis 10 |

overwhelmingly **implementation + evaluation** rather than **discovery**. students build something and test it. the contribution is the working system and results, not a theoretical advance.

### Framing Patterns

**Framework 1: "We built X and evaluated it" (most common)**
- "we implement X on the Spikey chip and demonstrate MNIST classification"
- "a CNN is built in MATLAB and trained on..."

**Framework 2: "We compared A vs B"**
- "compare SNN performance against conventional ANNs on classification problems"

**Framework 3: "We investigated/explored whether X"**
- "investigate whether a usable relation exists between object features and barcode location"

**Framework 4: "We discovered/found Y" (very rare)**
- only secondary findings, never the primary aim. e.g. "identifying a previously unknown interaction" (thesis 3, but that was an accidental bonus finding)

### Research Question Structure

when stated explicitly, they follow predictable patterns:

1. **Can X achieve Y?** -- "can a Jeffress model work on neuromorphic hardware?"
2. **How does X compare to Y?** -- "how do spiking networks perform vs traditional perceptrons?"
3. **What happens when we do X?** -- "what are the firing patterns under different configurations?"
4. **Can we solve X using Y?** -- "can synthetic data solve the limited training data problem?"

most questions are **closed** (yes/no answerable) rather than open-ended. "can we?" or "does it work?" rather than "why?" or "how does the mechanism function?"

---

## Heidelberg KIP Bachelor's Theses: Worth Noting Separately

the Heidelberg KIP theses are a special case -- 70+ bachelor's theses mostly on BrainScaleS/Spikey hardware.

**Common title patterns:**
- "Towards X on BrainScaleS" (progress, not completion)
- "Characterization of X on neuromorphic hardware" (measurement)
- "Calibration of X on Y" (getting hardware working)
- "Implementation of X for neuromorphic hardware" (building stuff)
- "Development of X for BrainScaleS" (engineering)
- "Testing X for neuromorphic hardware" (verification)

these are overwhelmingly **hardware-focused engineering**: calibrating chips, characterizing circuits, implementing software, testing components. framing is almost always "I built/implemented/characterized X on hardware Y." scientific investigation (e.g., "Investigating Competitive Dynamics") is less common but does happen. many contribute to a larger research infrastructure rather than standalone results.

**Selected titles for reference:**
- "Real-time Image Classification on Analog Neuromorphic Hardware" (Ebert, 2021)
- "Multi-Single-Chip Training of SNNs with BrainScaleS-2" (Straub, 2023)
- "Gradient-Based Training of Multi-Compartmental Neuron Models on BrainScaleS-2" (Janz, 2025)
- "Structural Plasticity for Feature Selection in Auditory Stimuli on Neuromorphic Hardware" (Kreft, 2019)
- "Solving Map Coloring Problems on Analog Neuromorphic Hardware" (Steidel, 2018)

Source: http://www.kip.uni-heidelberg.de/vision/publications/mscbsc/

---
