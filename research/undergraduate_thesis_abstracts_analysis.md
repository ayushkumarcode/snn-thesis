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
