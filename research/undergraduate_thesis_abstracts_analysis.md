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

