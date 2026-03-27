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

---

### 1.2 Papers Combining SNNs with AGI Concepts

There are no papers that directly propose "SNN as a path to AGI" in any rigorous technical sense. However, there is a significant literature connecting SNNs to AGI-adjacent cognitive capabilities:

| AGI Concept | SNN Papers | Key Works | Maturity |
|---|---|---|---|
| Continual Learning | 20+ papers | NACA (Science Advances 2024), CH-HNN (Nature Comms 2025), CLP-SNN on Loihi 2 (2024) | **ACTIVE, GROWING** |
| Meta-Learning | 5-10 papers | Meta-SpikePropamine (PMC 2023), SNN-MAML (arxiv 2022) | Moderate |
| Predictive Coding / World Models | 10-15 papers | SNN-PC (2024), CSDP (PMC 2024), PC-SNN (ResearchGate 2022) | Moderate |
| Self-Supervised Learning | 5-10 papers | Contrastive Predictive Coding + SNN (arxiv 2025), Active Efficient Coding (Frontiers 2024) | Emerging |
| Cognitive Architectures | 5-10 systems | SPAUN/Nengo, BrainCog, NeuCube | Established but niche |
| Reinforcement Learning (AGI component) | 20+ papers | DSQN (2022), PopSAN (2021), SpikeGym (2024) | Active |

**Key insight:** The SNN community does not frame their work as "AGI research." Instead, they frame it as "brain-inspired computing" or "neuromorphic intelligence." The connection to AGI is implicit, through capabilities like continual learning, few-shot adaptation, and energy-efficient inference that are considered prerequisites for general intelligence.

Sources:
- [Brain-inspired neuromodulation algorithm for catastrophic forgetting, Science Advances 2024](https://www.science.org/doi/10.1126/sciadv.adi2947)
- [Hybrid neural networks for continual learning, Nature Communications 2025](https://www.nature.com/articles/s41467-025-56405-9)
- [Meta-SpikePropamine, PMC 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10213417/)
- [Meta-learning SNNs with Surrogate Gradient Descent, arxiv 2022](https://arxiv.org/abs/2201.10777)
- [BrainCog cognitive intelligence engine](https://github.com/braincog-x/brain-cog)

---

### 1.3 SNN for Continual Learning -- Detailed Assessment

**The idea:** The brain doesn't forget old tasks when learning new ones (catastrophic forgetting). Can SNNs do this better than ANNs? This is arguably the most AGI-relevant capability that SNNs might naturally possess.

**Literature volume: 20+ papers, VERY active in 2024-2025**

**Key papers:**
| Paper | Year | Venue | Key Finding |
|---|---|---|---|
| NACA: Neuromodulation-assisted credit assignment | 2024 | Science Advances | Brain-inspired SNN mitigates forgetting at low computational cost |
| Hybrid corticohippocampal neural network (CH-HNN) | 2025 | Nature Communications | ANN+SNN hybrid significantly reduces forgetting in task- and class-incremental learning |
| CLP-SNN on Intel Loihi 2 | 2024 | arxiv | Real-time continual learning with neurogenesis and metaplasticity |
| Energy-Aware Spike Budgeting | 2026 | arxiv | Framework for continual SNN learning with energy constraints |
| Sleep prevents catastrophic forgetting in SNNs | 2023 | NSF Public Access | Biologically-inspired sleep-based consolidation |
| Mitigating forgetting through threshold modulation | 2024 | OpenReview (NeurIPS workshop) | Threshold-based approach to protect old knowledge |
| NIMBLE: Continual multi-label learning | 2024 | ScienceDirect | Task-agnostic, memory-free SNN framework |
| Investigating continuous learning in SNNs | 2023 | arxiv | Direct comparison on Split-MNIST and Permuted-MNIST |

**Biological motivation:** This is a genuinely strong argument. Biological neurons do not catastrophically forget -- synaptic consolidation, neurogenesis, and sleep-based replay all contribute to stable long-term memory. SNNs, being closer to biological neurons, may inherit some of these properties or be more amenable to biologically-inspired solutions.

**What an undergraduate thesis would look like:**

```
Title: "Mitigating Catastrophic Forgetting in Spiking Neural Networks:
        A Comparative Study of Continual Learning Strategies"

Components:
1. SNN baseline trained on MNIST (snnTorch, standard tutorial approach)
2. Split-MNIST benchmark: Train on digits 0-1, then 2-3, then 4-5, etc.
3. Measure forgetting: accuracy on task 1 after training on task 2, 3, etc.
4. Implement 2-3 mitigation strategies:
   a. Elastic Weight Consolidation (EWC) adapted for SNN
   b. Experience Replay (store and replay old spike trains)
   c. Threshold Modulation (freeze/protect important neurons)
5. Compare SNN vs ANN forgetting behavior
6. Energy efficiency comparison (spike count analysis)

Datasets: MNIST, Fashion-MNIST (standard, well-understood)
Framework: snnTorch + PyTorch
Hardware: CPU/MPS on macOS (no GPU needed for MNIST-scale)
```

**Feasibility for 28 days: MODERATE-HIGH**
- Week 1: MNIST SNN baseline (Tutorial 5 from snnTorch gets you there)
- Week 2: Split-MNIST setup + measure baseline forgetting
- Week 3: Implement EWC and one other strategy
- Week 4: Experiments, analysis, comparison with ANN, write-up

**Risks:**
- EWC adaptation to SNNs requires understanding Fisher Information Matrix calculation for spiking models
- The existing paper "Investigating Continuous Learning in SNNs" (2023) already does some of this
- Need to find a novel angle (e.g., specific to snnTorch, or comparing more strategies, or adding energy analysis)

**macOS compatible: YES** (MNIST is tiny, CPU is fine, snnTorch runs natively)

**VERDICT: VIABLE and interesting. The "SNN + continual learning" space is active enough to have good references but not so saturated that novelty is impossible. The key risk is that it has been done before -- you need a clear novel angle.**

---

### 1.4 SNN for Meta-Learning / Learning to Learn -- Detailed Assessment

**The idea:** Can SNNs learn how to learn new tasks quickly (few-shot learning)? This is directly relevant to AGI because general intelligence requires rapid adaptation.

**Literature volume: 5-10 papers**

**Key papers:**
| Paper | Year | Key Finding |
|---|---|---|
| Meta-SpikePropamine | 2023 | Three-factor learning rules trained via meta-learning for online learning |
| SNN-MAML (Meta-learning SNNs with surrogate gradients) | 2022 | MAML works with SNNs, matches ANN performance on event-based tasks |
| Robust Spike-Based Continual Meta-Learning | 2022 | Combined continual and meta-learning with minimum error entropy |

**What it would involve:**
- Implementing MAML (Model-Agnostic Meta-Learning) with snnTorch
- Training SNN to rapidly adapt to new classification tasks (e.g., Omniglot, mini-ImageNet)
- Comparing SNN-MAML vs ANN-MAML in few-shot scenarios

**Feasibility for 28 days: LOW**
- MAML itself is complex (second-order gradients, inner/outer loops)
- Combining MAML with SNN surrogate gradients adds another layer of complexity
- No existing snnTorch tutorial or implementation for meta-learning
- Would need to build from scratch based on the arxiv papers
- Debugging meta-learning + SNN is extremely time-consuming

**VERDICT: TOO ADVANCED for an undergraduate in 28 days. Fascinating research direction but implementation complexity is prohibitive. If continual learning is the interest, stick with 1.3 above.**
