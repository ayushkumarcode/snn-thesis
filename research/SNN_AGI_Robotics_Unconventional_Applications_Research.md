# SNN applications in AGI, robotics, and unconventional domains

i investigated 25+ SNN application domains across three categories: AGI-related concepts, robotics applications, and unconventional uses. Searched across arxiv, IEEE, Nature, Frontiers, GitHub, and multiple academic databases.

**AGI-Related:** The Kyambadde thesis turned out to be a Philosophy PhD with zero SNN content. SNN + continual learning is the most viable AGI-adjacent direction, with good literature and clear undergrad-feasible scope (Split-MNIST with EWC adaptation). SNN + meta-learning exists but is too technically demanding. Cognitive architectures (SPAUN, BrainCog) are fascinating but way too much scope for a few weeks.

**Robotics:** The most promising directions are (1) SNN for obstacle avoidance in simulation (CartPole/LunarLander), (2) SNN for robotic arm target-reaching in MuJoCo, and (3) SNN for optical flow estimation from event camera data. All can be done simulation-only on macOS. Tactile sensing, SLAM, and swarm robotics are interesting but need either specialized hardware data or way too much implementation.

**Unconventional:** SNN for cybersecurity/intrusion detection is the strongest unconventional option -- tabular data, public datasets (NSL-KDD, CICIDS2017), clear SNN efficiency narrative, and published results showing 99%+ accuracy. SNN for ECG classification is another strong contender. Game AI and autonomous driving are established fields with limited novelty. Drug discovery and weather prediction have minimal SNN presence.

**Top 3 new candidates from this research (not previously considered):**
1. SNN for continual learning (Split-MNIST, catastrophic forgetting mitigation)
2. SNN for robotic arm control in MuJoCo (RL-based, simulation-only)
3. SNN for image denoising (novel, few papers, competitive results)

---

## Part 1: AGI-related SNN research

---

### 1.1 The Kyambadde thesis: what it actually is

**Full title:** "Replicating cognitive self-sufficiency to realize computational self-sufficiency in Artificial General Intelligence models"
**Author:** Ivan Serwano Kyambadde
**Institution:** University of York, Department of **Philosophy** (NOT Computer Science)
**Supervisor:** Tom Stoneham (philosopher)
**Completion:** November 2023
**Repository:** [White Rose eTheses](https://etheses.whiterose.ac.uk/id/eprint/36019/)

This thesis is NOT about SNNs, not about neural networks, and not about technical AI at all. It's a philosophy thesis arguing that "qualitative states" (consciousness, subjective experience) are integral to achieving AGI. The supervisor is a philosopher, the department is Philosophy, and the methodology is philosophical argumentation, not implementation.

Relevance to my project: zero. Don't cite it, don't reference it, don't base any technical work on it.

---

### 1.2 Papers combining SNNs with AGI concepts

There aren't any papers that directly propose "SNN as a path to AGI" in any rigorous technical sense. But there's a significant literature connecting SNNs to AGI-adjacent cognitive capabilities:

| AGI Concept | SNN Papers | Key Works | Maturity |
|---|---|---|---|
| Continual Learning | 20+ papers | NACA (Science Advances 2024), CH-HNN (Nature Comms 2025), CLP-SNN on Loihi 2 (2024) | **ACTIVE, GROWING** |
| Meta-Learning | 5-10 papers | Meta-SpikePropamine (PMC 2023), SNN-MAML (arxiv 2022) | Moderate |
| Predictive Coding / World Models | 10-15 papers | SNN-PC (2024), CSDP (PMC 2024), PC-SNN (ResearchGate 2022) | Moderate |
| Self-Supervised Learning | 5-10 papers | Contrastive Predictive Coding + SNN (arxiv 2025), Active Efficient Coding (Frontiers 2024) | Emerging |
| Cognitive Architectures | 5-10 systems | SPAUN/Nengo, BrainCog, NeuCube | Established but niche |
| Reinforcement Learning (AGI component) | 20+ papers | DSQN (2022), PopSAN (2021), SpikeGym (2024) | Active |

The SNN community doesn't frame their work as "AGI research." Instead they use "brain-inspired computing" or "neuromorphic intelligence." The connection to AGI is implicit, through capabilities like continual learning, few-shot adaptation, and energy-efficient inference.

Sources:
- [Brain-inspired neuromodulation algorithm for catastrophic forgetting, Science Advances 2024](https://www.science.org/doi/10.1126/sciadv.adi2947)
- [Hybrid neural networks for continual learning, Nature Communications 2025](https://www.nature.com/articles/s41467-025-56405-9)
- [Meta-SpikePropamine, PMC 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10213417/)
- [Meta-learning SNNs with Surrogate Gradient Descent, arxiv 2022](https://arxiv.org/abs/2201.10777)
- [BrainCog cognitive intelligence engine](https://github.com/braincog-x/brain-cog)

---

### 1.3 SNN for continual learning -- detailed assessment

The idea: the brain doesn't forget old tasks when learning new ones (catastrophic forgetting). Can SNNs do this better than ANNs? This is arguably the most AGI-relevant capability that SNNs might naturally have.

**Literature: 20+ papers, very active in 2024-2025**

Key papers:
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

The biological motivation is genuinely strong. Biological neurons don't catastrophically forget -- synaptic consolidation, neurogenesis, and sleep-based replay all contribute to stable long-term memory. SNNs, being closer to biological neurons, may inherit some of these properties.

What an undergrad thesis would look like:
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

**Feasibility for ~4 weeks: MODERATE-HIGH**
- Week 1: MNIST SNN baseline (Tutorial 5 from snnTorch gets you there)
- Week 2: Split-MNIST setup + measure baseline forgetting
- Week 3: Implement EWC and one other strategy
- Week 4: Experiments, analysis, comparison with ANN, write-up

Risks:
- EWC adaptation to SNNs requires understanding Fisher Information Matrix calculation for spiking models
- The existing paper "Investigating Continuous Learning in SNNs" (2023) already does some of this
- Need to find a novel angle

macOS compatible: YES (MNIST is tiny, CPU is fine, snnTorch runs natively)

**Verdict: viable and interesting. The space is active enough for good references but not so saturated that novelty is impossible. The key risk is that it's been done before -- need a clear novel angle.**

---

### 1.4 SNN for meta-learning / learning to learn

The idea: can SNNs learn how to learn new tasks quickly (few-shot learning)? Directly relevant to AGI because general intelligence requires rapid adaptation.

**Literature: 5-10 papers**

Key papers:
| Paper | Year | Key Finding |
|---|---|---|
| Meta-SpikePropamine | 2023 | Three-factor learning rules trained via meta-learning for online learning |
| SNN-MAML (Meta-learning SNNs with surrogate gradients) | 2022 | MAML works with SNNs, matches ANN performance on event-based tasks |
| Robust Spike-Based Continual Meta-Learning | 2022 | Combined continual and meta-learning with minimum error entropy |

What it would involve:
- Implementing MAML (Model-Agnostic Meta-Learning) with snnTorch
- Training SNN to rapidly adapt to new classification tasks (e.g., Omniglot, mini-ImageNet)
- Comparing SNN-MAML vs ANN-MAML in few-shot scenarios

**Feasibility for ~4 weeks: LOW**
- MAML itself is complex (second-order gradients, inner/outer loops)
- Combining MAML with SNN surrogate gradients adds another layer
- No existing snnTorch tutorial or implementation for meta-learning
- Would need to build from scratch based on arxiv papers
- Debugging meta-learning + SNN is extremely time-consuming

**Verdict: too advanced for an undergrad in a few weeks. If continual learning is the interest, stick with 1.3 above.**

---

### 1.5 SNN-based cognitive architectures

**SPAUN (Semantic Pointer Architecture Unified Network):**
- 2.5 million spiking neurons, 20 brain areas, 8 cognitive tasks
- Built with Nengo framework
- Processes visual input, produces handwritten output via simulated arm
- Includes working memory, reinforcement learning, fluid reasoning
- Published: Eliasmith et al. (2012), Science
- Source: [Nengo SPA Documentation](https://www.nengo.ai/nengo-spa/user-guide/spa-intro.html)

**BrainCog:**
- Open-source SNN-based cognitive intelligence engine
- Implements perception, decision-making, motor control, reasoning, social cognition
- Available on [PyPI](https://pypi.org/project/braincog/) and [GitHub](https://github.com/braincog-x/brain-cog)
- Python/PyTorch-based, includes 18+ functional SNN algorithms
- Brain simulations at multiple scales (drosophila to human)

**Nengo:**
- General-purpose neural simulator
- Can run on CPU (no GPU required)
- Source: [Nengo Documentation](https://www.nengo.ai/)

What an undergrad thesis could do:
- Replicate a single cognitive function from SPAUN (e.g., working memory task) using snnTorch
- Build a simple decision-making circuit inspired by BrainCog's BDM-SNN model
- Compare performance with equivalent ANN architecture

**Feasibility for ~4 weeks: LOW**
- Understanding cognitive architectures requires significant neuroscience background
- SPAUN is massive (2.5M neurons) -- even replicating a subset is complex
- BrainCog is more accessible but still requires understanding brain region interactions
- Nengo has its own learning curve separate from snnTorch

**Verdict: not recommended. These are PhD-scale projects.**

---

### 1.6 SNN for predictive coding / world models

The idea: the brain constantly predicts its sensory inputs and learns from prediction errors. Can SNNs implement this?

**Literature: 10-15 papers, growing in 2024-2025**

Key papers:
- [Predictive coding with spiking neurons and feedforward gist signaling, Frontiers 2024](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1338280/full)
- [Integration of Contrastive Predictive Coding and SNNs, arxiv 2025](https://arxiv.org/html/2506.09194)
- [PC-SNN: Supervised Learning with Predictive Coding in SNNs](https://www.researchgate.net/publication/365821328)
- [Energy optimization induces predictive-coding properties in multi-compartment SNN, PLOS Comp Bio](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013112)

**Feasibility for ~4 weeks: LOW**
- Predictive coding requires multi-layer hierarchical architectures with top-down feedback
- Not supported by snnTorch's standard tutorials
- Theoretical background is substantial (Bayesian brain hypothesis, free energy principle)
- Implementation from scratch is a research-level challenge

**Verdict: not recommended. Intellectually fascinating but too theoretical and complex for the timeline.**

---

### 1.7 Part 1 summary

| AGI Concept | Papers | Feasible in ~4 weeks? | macOS? | snnTorch? | Novel? | Recommendation |
|---|---|---|---|---|---|---|
| Continual Learning | 20+ | **YES** (moderate) | Yes | Yes | Moderate | **Recommended** |
| Meta-Learning | 5-10 | No | Yes | No native support | High | Too complex |
| Cognitive Architectures | 5-10 systems | No | Nengo yes | No (use Nengo) | N/A (replication) | PhD-scale |
| Predictive Coding | 10-15 | No | Yes | No native support | Moderate | Too theoretical |
| Self-Supervised Learning | 5-10 | Possibly | Yes | Partially | High | Risky |
| World Models | <5 | No | Yes | No | Very High | Too early-stage |

---

## Part 2: Robotics SNN applications (beyond reflexes)

---

### 2.1 SNN for robotic grasping / manipulation

**Literature: 5-10 papers**

Key works:
- [Towards Grasping with SNNs for Anthropomorphic Robot Hands (ResearchGate 2017)](https://www.researchgate.net/publication/320580642)
- [Touch and slippage detection with SNNs (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0952197624011114)
- SNN classification of sEMG signals triggers finger reflexes on robotic hand

Simulation-only possible? Partially. MuJoCo can simulate grasping, but encoding tactile feedback into spikes requires custom work.

**Feasibility: LOW** -- too many moving parts. Grasping needs 3D physics + contact dynamics. No existing open-source SNN-grasping pipeline.

---

### 2.2 SNN for SLAM

**Literature: 5-10 papers**

Key works:
- [SNN on neuromorphic hardware for energy-efficient SLAM (IEEE 2019)](https://ieeexplore.ieee.org/document/8967864/)
- [Semantic Spiking Neural SLAM using Nengo (GitHub 2023)](https://github.com/nsdumont/Semantic-Spiking-Neural-SLAM-2023)
- [Exploiting semantic information in spiking neural SLAM (Frontiers 2023)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1190515/full)

Key finding: SNN SLAM on Loihi consumes 100x less energy than GMapping on CPU with comparable accuracy.

**Feasibility: LOW-MODERATE** -- SLAM itself is complex, and the existing implementation uses Nengo not snnTorch. Framework mismatch is a deal-breaker.

---

### 2.3 SNN for obstacle avoidance

**Literature: 10-20 papers**

Key works:
- [Directly-trained SNNs for DRL: Energy-efficient obstacle avoidance (ScienceDirect 2023)](https://www.sciencedirect.com/science/article/pii/S0925231223010081)
- [SNN for UAV Obstacle Avoidance targeting Neuromorphic Processors (IEEE 2019)](https://ieeexplore.ieee.org/document/8867860/)
- [Artificial vs Spiking NNs for RL in UAV obstacle avoidance (ACM 2022)](https://dl.acm.org/doi/abs/10.1145/3528416.3530865)

Simulation-only: YES. CartPole, LunarLander, and simple 2D navigation all work.
macOS: YES. Gymnasium (OpenAI Gym) runs fine on macOS.

**Feasibility: MODERATE-HIGH** (if scoped to simple environments)

What the thesis would look like:
```
Title: "SNN-Based Deep Reinforcement Learning for Obstacle Avoidance:
        An Energy-Efficiency Analysis"

Components:
1. Standard DQN agent on CartPole/LunarLander (ANN baseline)
2. Replace DQN network with Spiking DQN (LIF neurons, surrogate gradient)
3. Train both, compare performance (reward curves)
4. Compare energy efficiency (spike count, operations)
5. Extend to simple 2D navigation if time permits

Framework: snnTorch + Gymnasium + PyTorch
```

Risks: SNN+RL training is finicky (hyperparameter sensitivity). May not converge as well as ANN baseline. Limited novelty if just doing CartPole (many papers already).

**Verdict: feasible and well-scoped. The CartPole/LunarLander framing keeps scope manageable. Novelty is low unless you add an angle (e.g., energy analysis, or a novel encoding scheme).**

---

### 2.4 SNN for drone control

**Literature: 10-15 papers**

Simulation-only: yes, but drone simulators (AirSim, Gazebo) are heavyweight. AirSim is deprecated. Gazebo is Linux-only.

**Feasibility: LOW** -- drone dynamics are 6-DOF, much more complex than CartPole. Sim setup alone could take a week. If interested in RL+SNN, use CartPole/LunarLander instead.

---

### 2.5 SNN for tactile sensing (neuromorphic touch)

**Literature: 15-20 papers, active field**

Key works:
- [Event-Driven Tactile Sensing with Dense Spiking GNNs (IEEE 2025)](https://ieeexplore.ieee.org/document/10884798/)
- [Recent advances in spike-based neural coding for tactile perception (Nature 2025)](https://www.nature.com/articles/s41378-025-01074-3)
- [TactileSGNet (ResearchGate 2020)](https://www.researchgate.net/publication/342588043)

Simulation-only: not really in a meaningful way. Tactile sensing requires physical sensor data. Some public datasets exist (EvTouch, iCub tactile data) but they're niche.

**Feasibility: LOW-MODERATE** -- data dependency is the main risk.

---

### 2.6 SNN for swarm robotics

**Literature: 5-10 papers**

**Feasibility: LOW** -- multi-agent systems add exponential complexity. SNN + multi-agent RL is bleeding-edge. No existing frameworks combining snnTorch with multi-agent environments. Debugging multi-agent systems is extremely time-consuming. This is PhD-level.

---

### 2.7 SNN for prosthetic / exoskeleton control

**Literature: 10-15 papers**

Key finding: SNN reduces power consumption by 1-2 orders of magnitude vs CNN/LSTM for EMG gesture recognition, with comparable or better accuracy under electrode shift conditions.

Simulation-only: yes, using public EMG datasets. No physical prosthetic needed.

**Feasibility: MODERATE** -- public datasets exist (Ninapro, Myo Armband, UCI EMG). snnTorch can process temporal EMG signals. Well-scoped: classify 5-10 hand gestures from EMG. But this overlaps with existing SNN for wearable sensor data research.

**Verdict: feasible but may overlap with HAR/wearable domain. The EMG-specific angle (prosthetic control) adds a distinct narrative.**

---

### 2.8 SNN for optical flow estimation (event camera)

**Literature: 10-15 papers**

Key finding: SNNs achieve 48.3x parameter reduction and 10.2x energy reduction vs ANNs with ~10% lower error for optical flow from event cameras.

Simulation-only: yes, using public event camera datasets (MVSEC, DSEC-Flow).

**Feasibility: MODERATE** -- public datasets and GitHub code exist, but event camera data processing has a learning curve. snnTorch may not directly support required architectures (U-Net with skip connections).

**Verdict: interesting but risky for the timeline.**

---

### 2.9 SNN for robotic arm control (RL in MuJoCo)

**Literature: 5-10 papers, growing rapidly in 2024**

Key finding: a 3-DoF robotic arm target-reaching task was implemented using SNN + SpyTorch + MuJoCo Reacher-v4, extended from 2D to 3D.

Simulation-only: YES. MuJoCo is pure simulation.
macOS: YES. MuJoCo 2.1+ works on macOS natively.

**Feasibility: MODERATE**

```
Title: "Energy-Efficient Robotic Arm Control Using Spiking Neural Networks
        in Simulated Environments"

