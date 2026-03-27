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

