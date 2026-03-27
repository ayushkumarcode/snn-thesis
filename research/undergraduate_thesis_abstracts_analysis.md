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
