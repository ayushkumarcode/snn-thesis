# SNN Applications in AGI, Robotics, and Unconventional Domains: Comprehensive Research Report

**Date:** 2026-02-27
**Purpose:** Investigate SNN applications in AGI-related, robotics-related, and unconventional domains not yet considered. Assess feasibility for a University of Manchester undergraduate thesis (~28 days, snnTorch, macOS, no physical robot).

---

## EXECUTIVE SUMMARY

This report investigates 25+ SNN application domains across three categories: AGI-related concepts, robotics applications, and unconventional uses. After exhaustive searching across arxiv, IEEE, Nature, Frontiers, GitHub, and multiple academic databases, the key findings are:

**AGI-Related:** The Kyambadde thesis is a Philosophy PhD with no SNN content whatsoever. SNN + continual learning is the most viable AGI-adjacent direction, with a rich literature and clear undergraduate-feasible scope (Split-MNIST with EWC adaptation). SNN + meta-learning exists but is too technically demanding. Cognitive architectures (SPAUN, BrainCog) are fascinating but scope-inappropriate for 28 days.

**Robotics:** Beyond reflexes, the most promising directions are (1) SNN for obstacle avoidance in simulation (CartPole/LunarLander), (2) SNN for robotic arm target-reaching in MuJoCo, and (3) SNN for optical flow estimation from event camera data. All can be done in simulation-only on macOS. Tactile sensing, SLAM, and swarm robotics are interesting but require either specialized hardware data or excessive implementation complexity.

**Unconventional:** SNN for cybersecurity/intrusion detection is the strongest unconventional option -- tabular data, public datasets (NSL-KDD, CICIDS2017), clear SNN efficiency narrative, and published results showing 99%+ accuracy. SNN for ECG classification is another strong contender. Game AI and autonomous driving are established fields with limited novelty. Drug discovery and weather prediction have minimal SNN presence.

**Top 3 NEW candidates from this research (not previously considered):**
1. SNN for Continual Learning (Split-MNIST, catastrophic forgetting mitigation)
2. SNN for Robotic Arm Control in MuJoCo (RL-based, simulation-only)
3. SNN for Image Denoising (novel, few papers, competitive results)

---

## PART 1: AGI-RELATED SNN RESEARCH

---

### 1.1 The Kyambadde Thesis: What It Actually Is

**Full Title:** "Replicating cognitive self-sufficiency to realize computational self-sufficiency in Artificial General Intelligence models"
**Author:** Ivan Serwano Kyambadde
**Institution:** University of York, Department of **Philosophy** (NOT Computer Science)
**Supervisor:** Tom Stoneham (philosopher)
**Completion:** November 2023
**Repository:** [White Rose eTheses](https://etheses.whiterose.ac.uk/id/eprint/36019/)

**VERDICT: This thesis is NOT about SNNs, not about neural networks, and not about technical AI at all.**

It is a philosophy thesis arguing that "qualitative states" (consciousness, subjective experience) are integral to achieving AGI. The thesis takes a philosophical/epistemological approach to the question of whether purely computational models can replicate the cognitive self-sufficiency that biological agents possess. The supervisor is a philosopher, the department is Philosophy, and the methodology is philosophical argumentation, not implementation.

**Relevance to your project: ZERO.** This is a red herring. Do not cite it, reference it, or base any technical work on it. The "AGI" in the title is misleading -- it addresses the philosophical question of whether AGI is possible, not how to build it.
