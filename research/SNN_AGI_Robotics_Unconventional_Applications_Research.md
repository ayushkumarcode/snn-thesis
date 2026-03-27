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

---

### 1.5 SNN-Based Cognitive Architectures -- Detailed Assessment

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
- Supports building large-scale brain models
- Python-based, well-documented
- Can run on CPU (no GPU required)
- Source: [Nengo Documentation](https://www.nengo.ai/)

**What an undergraduate thesis could do:**
- Replicate a single cognitive function from SPAUN (e.g., working memory task) using snnTorch
- Build a simple decision-making circuit inspired by BrainCog's BDM-SNN model
- Compare performance with equivalent ANN architecture

**Feasibility for 28 days: LOW**
- Understanding cognitive architectures requires significant neuroscience background
- SPAUN is a massive system (2.5M neurons) -- even replicating a subset is complex
- BrainCog is more accessible but still requires understanding brain region interactions
- The Nengo framework has a steep learning curve separate from snnTorch
- Scope is very hard to define -- what counts as "success"?

**VERDICT: NOT RECOMMENDED for 28 days. These are PhD-scale projects. The underlying neuroscience knowledge requirement alone would consume most of the available time. However, BrainCog's modular design means a very focused sub-project (e.g., "replicating the BDM-SNN decision-making model") could potentially work if you accept a purely replication-based thesis (novelty level (a), which is uncommon at Manchester).**

---

### 1.6 SNN for Predictive Coding / World Models

**The idea:** The brain constantly predicts its sensory inputs and learns from prediction errors. Can SNNs implement this "predictive coding" framework?

**Literature: 10-15 papers, growing in 2024-2025**

Key papers:
- [Predictive coding with spiking neurons and feedforward gist signaling, Frontiers 2024](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1338280/full)
- [Integration of Contrastive Predictive Coding and SNNs, arxiv 2025](https://arxiv.org/html/2506.09194)
- [PC-SNN: Supervised Learning with Predictive Coding in SNNs](https://www.researchgate.net/publication/365821328)
- [Energy optimization induces predictive-coding properties in multi-compartment SNN, PLOS Comp Bio](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013112)

**Feasibility for 28 days: LOW**
- Predictive coding requires multi-layer hierarchical architectures with top-down feedback
- Not supported by snnTorch's standard tutorials
- Theoretical background is substantial (Bayesian brain hypothesis, free energy principle)
- Implementation from scratch is a research-level challenge

**VERDICT: NOT RECOMMENDED. Intellectually fascinating but too theoretical and complex for the timeline.**

---

### 1.7 Part 1 Summary Table

| AGI Concept | Papers | 28-Day Feasible? | macOS? | snnTorch? | Novel? | Recommendation |
|---|---|---|---|---|---|---|
| Continual Learning | 20+ | **YES** (moderate) | Yes | Yes | Moderate | **RECOMMENDED** |
| Meta-Learning | 5-10 | No | Yes | No native support | High | Too complex |
| Cognitive Architectures | 5-10 systems | No | Nengo yes | No (use Nengo) | N/A (replication) | PhD-scale |
| Predictive Coding | 10-15 | No | Yes | No native support | Moderate | Too theoretical |
| Self-Supervised Learning | 5-10 | Possibly | Yes | Partially | High | Risky |
| World Models | <5 | No | Yes | No | Very High | Too early-stage |

---

## PART 2: ROBOTICS SNN APPLICATIONS (BEYOND REFLEXES)

---

### 2.1 SNN for Robotic Grasping / Manipulation

**Literature: 5-10 papers**

Key works:
- [Towards Grasping with SNNs for Anthropomorphic Robot Hands (ResearchGate 2017)](https://www.researchgate.net/publication/320580642)
- [Touch and slippage detection with SNNs (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0952197624011114)
- SNN classification of sEMG signals triggers finger reflexes on robotic hand

**Simulation-only possible?** Partially. MuJoCo can simulate grasping, but encoding tactile feedback into spikes requires custom work.

**macOS compatible?** MuJoCo works on macOS. snnTorch works on macOS. Integration is the challenge.

**Feasibility for 28 days: LOW**
- Grasping requires 3D physics simulation + contact dynamics
- SNN integration with simulation environment is non-trivial
- Most existing work uses real neuromorphic tactile sensors
- No existing open-source SNN-grasping pipeline

**System components:** MuJoCo/PyBullet + snnTorch + custom environment wrapper + spike encoding for joint angles/contacts

**VERDICT: NOT RECOMMENDED. Too many moving parts for 28 days.**

---

### 2.2 SNN for SLAM (Simultaneous Localization and Mapping)

**Literature: 5-10 papers**

Key works:
- [SNN on neuromorphic hardware for energy-efficient SLAM (IEEE 2019)](https://ieeexplore.ieee.org/document/8967864/)
- [Semantic Spiking Neural SLAM using Nengo (GitHub 2023)](https://github.com/nsdumont/Semantic-Spiking-Neural-SLAM-2023)
- [Exploiting semantic information in spiking neural SLAM (Frontiers 2023)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1190515/full)

**Key finding:** SNN SLAM on Loihi consumes 100x less energy than GMapping on CPU with comparable accuracy.

**Simulation-only possible?** Yes, using 2D grid worlds or simple environments. The Nengo-based implementation uses simulated environments.

**macOS compatible?** Nengo runs on macOS. However, this uses Nengo, NOT snnTorch.

**Feasibility for 28 days: LOW-MODERATE**
- SLAM itself is complex (particle filters, occupancy grids, loop closure)
- The existing Nengo implementation provides a starting point but requires Nengo expertise
- Not in snnTorch -- would need to use Nengo or build from scratch
- A simplified 1D SLAM demo might be feasible

**System components:** Nengo + simulated 1D/2D environment + place cells + head direction cells

**VERDICT: NOT RECOMMENDED unless you switch to Nengo. The scope is too large and framework mismatch with snnTorch is a deal-breaker.**

---

### 2.3 SNN for Obstacle Avoidance

**Literature: 10-20 papers**

Key works:
- [Directly-trained SNNs for DRL: Energy-efficient obstacle avoidance (ScienceDirect 2023)](https://www.sciencedirect.com/science/article/pii/S0925231223010081)
- [SNN for UAV Obstacle Avoidance targeting Neuromorphic Processors (IEEE 2019)](https://ieeexplore.ieee.org/document/8867860/)
- [Artificial vs Spiking NNs for RL in UAV obstacle avoidance (ACM 2022)](https://dl.acm.org/doi/abs/10.1145/3528416.3530865)
- [Nature-inspired collision avoidance for drone swarm with reward-modulated SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC9676561/)

**Simulation-only possible?** YES. CartPole, LunarLander, and simple 2D navigation environments all work.

**macOS compatible?** YES. Gymnasium (OpenAI Gym) works on macOS. snnTorch works on macOS.

**Feasibility for 28 days: MODERATE-HIGH** (if scoped to simple environments)

**What the thesis would look like:**
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
Datasets: CartPole-v1, LunarLander-v2 environments
Hardware: CPU/MPS on macOS
```

**Risks:**
- SNN+RL training is notoriously finicky (hyperparameter sensitivity)
- May not converge as well as ANN baseline
- Limited novelty if just doing CartPole (many papers already)

**VERDICT: FEASIBLE and well-scoped. The CartPole/LunarLander framing keeps scope manageable. Novelty is low unless you add an angle (e.g., specific energy analysis, or a novel encoding scheme).**

---

### 2.4 SNN for Drone Control

**Literature: 10-15 papers**

Key works:
- [SNN-based PID controller for UAV (RSS 2020)](https://www.roboticsproceedings.org/rss16/p074.pdf)
- [Nature-inspired self-organizing collision avoidance for drone swarm (PMC 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9676561/)
- [Parameter optimization in SNN for UAV obstacle avoidance (arxiv 2019)](https://arxiv.org/abs/1910.07960)

**Simulation-only possible?** Yes, but drone simulators (AirSim, Gazebo) are heavyweight.

**macOS compatible?** AirSim is deprecated. Gazebo is Linux-only. PyBullet can simulate basic drone physics on macOS but is complex.

**Feasibility for 28 days: LOW**
- Drone dynamics are 6-DOF (position + orientation) -- much more complex than CartPole
- Simulation setup alone could take a week
- SNN integration adds another layer of complexity

**VERDICT: NOT RECOMMENDED. Too complex for the timeline. If interested in RL+SNN, use CartPole/LunarLander instead.**

---

### 2.5 SNN for Tactile Sensing (Neuromorphic Touch)

**Literature: 15-20 papers, active field**

Key works:
- [Event-Driven Tactile Sensing with Dense Spiking GNNs (IEEE 2025)](https://ieeexplore.ieee.org/document/10884798/)
- [Recent advances in spike-based neural coding for tactile perception (Nature 2025)](https://www.nature.com/articles/s41378-025-01074-3)
- [TactileSGNet: Spiking GNN for event-based tactile recognition (ResearchGate 2020)](https://www.researchgate.net/publication/342588043)
- [Neuromorphic Tactile Perception System for Texture Recognition (Springer 2024)](https://link.springer.com/chapter/10.1007/978-981-96-0789-1_13)

**Simulation-only possible?** NO in a meaningful way. Tactile sensing requires physical sensor data. Some public datasets exist (EvTouch, iCub tactile data) but they are niche.

**macOS compatible?** Processing existing datasets: yes. Generating new data: no (requires hardware).

**Feasibility for 28 days: LOW-MODERATE**
- If using an existing public dataset, could be feasible
- But finding and preparing neuromorphic tactile datasets is a challenge in itself
- Graph Neural Network variants (TactileSGNet) add complexity

**VERDICT: NOT RECOMMENDED unless a good public dataset can be found. The data dependency is the main risk.**

---

### 2.6 SNN for Swarm Robotics

**Literature: 5-10 papers**

Key works:
- [SNNs as Controllers for Emergent Swarm Agents (arxiv 2024)](https://arxiv.org/abs/2410.16175)
- [Nature-inspired collision avoidance for drone swarm with reward-modulated SNN](https://pmc.ncbi.nlm.nih.gov/articles/PMC9676561/)

**Simulation-only possible?** Yes. Simple 2D multi-agent simulation environments exist.

**macOS compatible?** Yes, for simple simulations.

**Feasibility for 28 days: LOW**
- Multi-agent systems add exponential complexity
- SNN + multi-agent RL is bleeding-edge research
- No existing frameworks combining snnTorch with multi-agent environments
- Debugging multi-agent systems is extremely time-consuming

**VERDICT: NOT RECOMMENDED. Multi-agent complexity makes this a PhD-level project.**

---

### 2.7 SNN for Prosthetic / Exoskeleton Control

**Literature: 10-15 papers**

Key works:
- [Feasibility of SNN in myoelectric control systems (Frontiers 2023)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1174760/full)
- [SNN classifies sEMG signals and triggers finger reflexes on robotic hand (ScienceDirect 2020)](https://www.sciencedirect.com/science/article/pii/S0921889020304061)
- [SNN Approach for Classifying Hand Movement (WSEAS 2025)](https://wseas.com/journals/bab/2025/a325108-254.pdf)

**Key finding:** SNN reduces power consumption by 1-2 orders of magnitude vs CNN/LSTM for EMG gesture recognition, with comparable or better accuracy under electrode shift conditions.

**Simulation-only possible?** Yes, using public EMG datasets. No physical prosthetic needed.

**macOS compatible?** Yes. EMG datasets are standard numerical data.

**Feasibility for 28 days: MODERATE**
- Public datasets exist (Ninapro, Myo Armband datasets, UCI EMG)
- snnTorch can process temporal EMG signals
- Well-scoped: classify 5-10 hand gestures from EMG
- Clear energy efficiency narrative for wearable deployment
- **BUT:** This overlaps heavily with existing "SNN for wearable sensor data" research that's already been considered in your novel domains assessment

**VERDICT: FEASIBLE but may overlap with HAR/wearable domain already assessed. The EMG-specific angle (prosthetic control) adds a distinct narrative.**

---

### 2.8 SNN for Optical Flow Estimation (Event Camera)

**Literature: 10-15 papers**

Key works:
- [Optical flow estimation from event cameras and SNNs (Frontiers 2023)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1160034/full)
- [Adaptive-SpikeNet: Event-based optical flow with learnable SNN dynamics (arxiv 2022)](https://arxiv.org/abs/2209.11741)
- [ST-FlowNet: Efficient SNN for event-based optical flow (arxiv 2025)](https://arxiv.org/html/2503.10195v1)
- [GitHub implementation: OF_EV_SNN](https://github.com/j-cuadrado/of_ev_snn)

**Key finding:** SNNs achieve 48.3x parameter reduction and 10.2x energy reduction vs ANNs with ~10% lower error for optical flow from event cameras.

**Simulation-only possible?** Yes, using public event camera datasets (MVSEC, DSEC-Flow).

**macOS compatible?** Yes. Processing event data is CPU-feasible.

**Feasibility for 28 days: MODERATE**
- Public datasets and GitHub code exist
- U-Net-like SNN architecture is well-documented
- However, event camera data processing has a learning curve
- snnTorch may not directly support the required architectures (U-Net with skip connections)

**VERDICT: INTERESTING but risky. The existing GitHub code could be a starting point, but adapting it to snnTorch and writing a thesis around it in 28 days is tight.**

---

### 2.9 SNN for Robotic Arm Control (RL in MuJoCo)

**Literature: 5-10 papers, growing rapidly in 2024**

Key works:
- [Exploring SNNs for Deep RL in Robotic Tasks (Nature Scientific Reports 2024)](https://www.nature.com/articles/s41598-024-77779-8)
- [Designing SNN-Based RL for 3D Robotic Arm Applications (MDPI 2025)](https://www.mdpi.com/2079-9292/14/3/578)
- [SNNs for Continuous Control via End-to-End Model-Based Learning (arxiv 2025)](https://arxiv.org/html/2509.05356v2)

**Key finding:** A 3-DoF robotic arm target-reaching task was implemented using SNN + SpyTorch + MuJoCo Reacher-v4 environment, extended from 2D to 3D.

**Simulation-only possible?** YES. MuJoCo is a pure simulation environment.

**macOS compatible?** YES. MuJoCo 2.1+ works on macOS natively. Gymnasium provides Reacher-v4 environment.

**Feasibility for 28 days: MODERATE**

**What the thesis would look like:**
```
Title: "Energy-Efficient Robotic Arm Control Using Spiking Neural Networks
        in Simulated Environments"

Components:
1. MuJoCo Reacher-v2/v4 environment setup (Gymnasium)
2. ANN-based policy network (PPO or DQN) as baseline
3. Replace policy network with SNN (LIF neurons, snnTorch)
4. Train both, compare:
   - Task performance (distance to target, success rate)
   - Training efficiency (episodes to convergence)
   - Energy proxy (spike count, synaptic operations)
5. Analysis of SNN temporal dynamics in control

Framework: snnTorch + Gymnasium + MuJoCo
Hardware: macOS CPU (MuJoCo is CPU-efficient)
```

**Risks:**
- SNN + RL training instability
- MuJoCo environment setup can have compatibility quirks
- Hyperparameter tuning for spiking RL is time-consuming
- The 2025 MDPI paper already does something very similar

**VERDICT: FEASIBLE if carefully scoped. The MuJoCo Reacher environment is well-documented and simple enough for 28 days. Novelty angle: using snnTorch specifically (most papers use SpyTorch or custom implementations).**

---

### 2.10 Part 2 Summary Table

| Robotics Application | Papers | Simulation-Only? | macOS? | 28-Day Feasible? | Components |
|---|---|---|---|---|---|
| **Obstacle Avoidance (CartPole/LunarLander)** | 10-20 | **YES** | **YES** | **MODERATE-HIGH** | snnTorch + Gymnasium |
| **Robotic Arm Control (MuJoCo)** | 5-10 | **YES** | **YES** | **MODERATE** | snnTorch + Gymnasium + MuJoCo |
| Optical Flow (Event Camera) | 10-15 | YES (datasets) | YES | MODERATE | snnTorch + MVSEC/DSEC data |
| Prosthetic/EMG Control | 10-15 | YES (datasets) | YES | MODERATE | snnTorch + Ninapro/UCI EMG |
| Tactile Sensing | 15-20 | Partially | YES | LOW-MODERATE | snnTorch + tactile dataset |
| SLAM | 5-10 | YES | YES (Nengo) | LOW-MODERATE | Nengo (not snnTorch) |
| Drone Control | 10-15 | Partially | Difficult | LOW | Complex sim setup |
| Grasping/Manipulation | 5-10 | Partially | YES | LOW | MuJoCo + custom env |
| Swarm Robotics | 5-10 | YES | YES | LOW | Multi-agent complexity |

---

## PART 3: UNCONVENTIONAL SNN APPLICATIONS

---

### 3.1 SNN for Game AI (Atari, Board Games)

**Literature: 10-15 papers**
