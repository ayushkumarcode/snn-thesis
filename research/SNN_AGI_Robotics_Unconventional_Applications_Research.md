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

Key works:
- [Deep Spiking Q-Network (DSQN) outperforms DQN on 17 Atari games (arxiv 2022)](https://arxiv.org/abs/2201.09754)
- [Multi-compartment neuron for deep distributional RL (arxiv 2023)](https://arxiv.org/html/2301.07275)
- [Improved robustness of RL policies upon conversion to SNN, Atari Breakout (ScienceDirect 2019)](https://www.sciencedirect.com/science/article/abs/pii/S0893608019302266)

**Assessment:**
- Papers: 10-15 (moderate)
- Simulation-only: YES
- macOS: YES (Gymnasium + Atari ROMs)
- 28-day feasibility: **MODERATE** (CartPole is feasible; full Atari is hard)
- Components: snnTorch + Gymnasium + Atari Learning Environment

**Key finding:** DSQN represents Q-values via membrane voltage of non-spiking output neurons, outperforming standard DQN on most Atari games while being more robust to input noise.

**Novel angle:** Most papers use custom SNN frameworks. Implementing spiking DQN in snnTorch specifically, with energy analysis, could be novel.

**VERDICT: MODERATE candidate. CartPole/LunarLander scope is feasible. Full Atari is too ambitious. Overlaps heavily with the obstacle avoidance RL direction (2.3).**

---

### 3.2 SNN for Cybersecurity / Intrusion Detection

**Literature: 10-20 papers, actively growing**

Key works:
- [Efficient intrusion detection based on convolutional SNN (Nature Scientific Reports 2024)](https://www.nature.com/articles/s41598-024-57691-x)
- [Energy-aware protocol-aware transformer-spiking hybrid (Nature Scientific Reports 2026)](https://www.nature.com/articles/s41598-026-37367-4)
- [Intrusion Detection for 5G SDN with binarized deep spiking capsule network (MDPI 2024)](https://www.mdpi.com/1999-5903/16/10/359)
- [One-Class Anomaly Detection for Cyber-Security on ICS (ResearchGate 2017)](https://www.researchgate.net/publication/318822219)
- [Event-Driven IDS using SNNs for Edge and IoT Security (ResearchGate 2025)](https://www.researchgate.net/publication/395850158)

**Assessment:**
- Papers: 10-20 (growing rapidly)
- Simulation-only: YES (all dataset-based)
- macOS: YES
- 28-day feasibility: **HIGH**
- Components: snnTorch + NSL-KDD/CICIDS2017 dataset + standard classification pipeline

**Published results:** SNN models achieve 99.0% accuracy on NSL-KDD, 99.53% on CICIDS2017, 96.80% on UNSW-NB15. These are competitive with ANNs.

**What the thesis would look like:**
```
Title: "Energy-Efficient Intrusion Detection Using Spiking Neural Networks:
        A Comparative Analysis on Standard Network Security Benchmarks"

Components:
1. Preprocess NSL-KDD/CICIDS2017 (feature engineering, normalization)
2. Encode tabular features into spike trains (rate coding, latency coding)
3. Train SNN classifier (fully-connected or convolutional)
4. ANN baseline (MLP, CNN) for comparison
5. Energy analysis (spike counts, operations comparison)
6. Per-attack-type analysis (DDoS, probe, U2R, R2L)

Framework: snnTorch + PyTorch + scikit-learn (preprocessing)
Datasets: NSL-KDD (available on Kaggle), CICIDS2017
Hardware: CPU on macOS (tabular data, small models)
```

**Key advantages:**
- Tabular data is straightforward to preprocess
- Datasets are public and well-documented
- Clear "energy efficiency for edge deployment" narrative
- Well-established baselines for comparison
- **Already assessed in your research as "SNN for IDS Cybersecurity" -- this overlaps with existing research file**

**VERDICT: STRONG candidate. But check your existing SNN_IDS_Cybersecurity_Research_Report.md -- this may already be fully covered.**

---

### 3.3 SNN for Financial Trading

**Literature: 10-15 papers**

Key works:
- [SNN for predictive modelling of financial time series + online news (Nature Scientific Reports 2023)](https://www.nature.com/articles/s41598-023-42605-0)
- [Financial Time Series Prediction Using SNNs (PLOS ONE 2014)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0103656)
- [High-Frequency Trading with SNNs (Imperial College 2021)](https://www.doc.ic.ac.uk/~wl/papers/21/wilm21kg.pdf)
- [ICS-SNN for Financial Time Series Forecasting (MDPI 2025)](https://www.mdpi.com/1999-4893/18/5/262)

**Assessment:**
- Papers: 10-15
- Simulation-only: YES (historical data)
- macOS: YES
- 28-day feasibility: **MODERATE**
- Components: snnTorch + financial data API + preprocessing pipeline

**NOTE:** Brian Ezinwoke (Manchester, 2025) already did this. His thesis on SNN for HFT price spike prediction scored well. Doing something similar risks looking derivative unless you find a clearly distinct angle.

**VERDICT: NOT RECOMMENDED. Already done by a Manchester student with the same supervisor. Find a different domain.**

---

### 3.4 SNN for Autonomous Driving Perception

**Literature: 20+ papers, active field**

Key works:
- [Autonomous Driving with Spiking Neural Networks (NeurIPS 2024)](https://arxiv.org/abs/2405.19687)
- [Spiking Neural Networks for Autonomous Driving: A Review (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S0952197624015732)
- [SAD: End-to-end autonomous driving with SNNs (GitHub)](https://github.com/ridgerchu/SAD)

**Assessment:**
- Papers: 20+ (well-established)
- Simulation-only: YES (nuScenes dataset, CARLA simulator)
- macOS: CARLA is Linux/Windows only. Dataset processing works on macOS.
- 28-day feasibility: **LOW**
- Components: Heavy compute (multi-view camera processing, BEV generation)

**VERDICT: NOT RECOMMENDED. Too established (low novelty), too computationally heavy, CARLA doesn't run on macOS. The NeurIPS 2024 paper sets a bar that's impossible for an undergraduate to approach.**

---

### 3.5 SNN for Weather/Time Series Prediction

**Literature: 5-10 papers**

Key works:
- [Efficient Time-Series Forecasting with SNNs (arxiv 2024)](https://arxiv.org/abs/2402.01533)
- [Forecasting Weather Signals Using Polychronous SNN (Springer 2015)](https://link.springer.com/chapter/10.1007/978-3-319-22180-9_12)
- [Methodology for univariate time-series forecasting with SNN (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S0893608024000959)
- [TS-LIF: Temporal Segment Spiking Neuron for Time Series Forecasting (OpenReview)](https://openreview.net/forum?id=rDe9yQQYKt)

**Assessment:**
- Papers: 5-10 (emerging field)
- Simulation-only: YES (public datasets)
- macOS: YES
- 28-day feasibility: **MODERATE-HIGH**
- Components: snnTorch + weather/time series dataset + temporal encoding

**What the thesis would look like:**
```
Title: "Spiking Neural Networks for Energy-Efficient Time Series Forecasting"

Components:
1. Select benchmark time series datasets (ETTh1, Weather, Exchange-Rate)
2. Encode time series data into spike trains
3. Train SNN-based forecasting model
4. Compare with LSTM, Transformer baselines
5. Energy efficiency analysis

Framework: snnTorch + PyTorch
Datasets: ETTh1 (Electricity Transformer Temperature), Weather dataset
Hardware: CPU on macOS
```

**Novel angle:** Most SNN time series work (5-10 papers) doesn't use snnTorch. The field is young enough that a snnTorch-based implementation with systematic benchmarking could contribute.

**VERDICT: INTERESTING emerging field. Feasible scope, moderate novelty. But overlaps with your existing Time_Series_Forecasting research file.**

---

### 3.6 SNN for Drug Discovery / Molecular Property Prediction

**Literature: 1-2 papers**

Key work:
- [Screening P450 Enzyme Bioactivity by SNNs (Springer 2025)](https://link.springer.com/chapter/10.1007/978-3-031-90714-2_20)

**Assessment:**
- Papers: 1-2 (EXTREMELY sparse)
- Simulation-only: YES
- macOS: YES
- 28-day feasibility: **LOW**
- Domain knowledge requirement: HIGH (biochemistry, molecular fingerprints)

**VERDICT: NOT RECOMMENDED. Minimal SNN precedent, high domain knowledge barrier. Graph Neural Networks dominate this space, and for good reason -- molecular structures are naturally graphs, not temporal sequences.**

---

### 3.7 SNN for ECG / Medical Signal Classification

**Literature: 15-20 papers**

Key works:
- [Accurate ECG Classification Based on SNN and Attentional Mechanism (MDPI 2022)](https://www.mdpi.com/2079-9292/11/12/1889)
- [ECG Classification with LIF Neurons (Sensors 2024)](https://www.mdpi.com/1424-8220/24/11/3426)
- [Neuromorphic implementation of ECG anomaly detection (arxiv 2022)](https://arxiv.org/pdf/2209.01266)
- [Review on SNN-based ECG classification for low-power environments (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362428/)

**Assessment:**
- Papers: 15-20 (established subfield)
- Simulation-only: YES (MIT-BIH dataset)
- macOS: YES
- 28-day feasibility: **HIGH**
- Components: snnTorch + MIT-BIH Arrhythmia Database + 1D Conv SNN

**NOTE:** You already have SNN_ECG_Classification_Research_Report.md in your research directory. This is already covered.

**VERDICT: Already researched. See existing report.**

---

### 3.8 SNN for NLP / Text Classification

**Literature: 15-25 papers**

Key works:
- [SpikeGPT: 216M parameter SNN language model (arxiv 2023)](https://arxiv.org/html/2302.13939v5)
- [SNNLP: Energy-Efficient NLP with SNNs (arxiv 2024)](https://arxiv.org/abs/2401.17911)
- [Spiking ConvNNs for Text Classification (arxiv 2024)](https://arxiv.org/html/2406.19230v1)
- [Neuromorphic Sentiment Analysis (PMC 2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10536645/)

**Assessment:**
- Papers: 15-25 (growing)
- Simulation-only: YES
- macOS: YES
- 28-day feasibility: **MODERATE**

**NOTE:** Already covered in snn_nlp_text_processing_research.md.

**VERDICT: Already researched. See existing report.**

---

### 3.9 SNN for Image Denoising / Restoration (NEW -- NOT PREVIOUSLY COVERED)

**Literature: 3-5 papers (VERY SPARSE)**

Key works:
- [SPIDEN: Deep SNN for efficient image denoising (Frontiers 2023)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1224457/full)
- [Neural information coding for efficient spike-based image denoising (arxiv 2023)](https://arxiv.org/abs/2305.11898)
- [SNNSIR: Simple SNN for Stereo Image Restoration (arxiv 2025)](https://arxiv.org/pdf/2508.12271)

**Key finding:** SPIDEN achieves 30.18 dB PSNR on Set12 (only 0.25 dB below equivalent DCNN) with 20% energy reduction. This is surprisingly competitive.

**Assessment:**
- Papers: 3-5 (HIGHLY novel)
- Simulation-only: YES (standard image datasets)
- macOS: YES
- 28-day feasibility: **MODERATE-HIGH**
- Components: snnTorch + image datasets (Set12, BSD68, Urban100) + U-Net-like SNN architecture

**What the thesis would look like:**
```
Title: "Energy-Efficient Image Denoising with Spiking Neural Networks"

Components:
1. Implement DnCNN (standard denoising CNN) as ANN baseline
2. Convert/train equivalent SNN denoising network using snnTorch
3. Test on standard benchmarks (Set12, BSD68) with various noise levels
4. Compare PSNR/SSIM quality metrics
5. Compare energy metrics (operations, activations)
6. Explore different spike encoding strategies for images

Framework: snnTorch + PyTorch + torchvision
Datasets: Set12, BSD68, Urban100 (standard denoising benchmarks)
Hardware: CPU/MPS on macOS (images are small for denoising benchmarks)
```

**Why this is interesting:**
- Only 3-5 papers in total -- VERY high novelty
- Image processing is a mainstream ML task, so the comparison framework is well-understood
- PSNR/SSIM are standard, well-defined metrics
- The "SNN processes images" angle is novel (most SNN work is on temporal data)
- Energy efficiency story is clear and compelling
- Can start from the SPIDEN architecture as a reference

**Risks:**
- U-Net architectures with skip connections may be tricky in snnTorch
- Image denoising is not a "natural" SNN task (images aren't temporal)
- SPIDEN uses custom architecture, not directly snnTorch

**VERDICT: EXCELLENT novel candidate. 3-5 papers means genuine contribution potential. The image denoising benchmarks are mature and well-understood, reducing implementation risk. The "can SNNs do non-temporal tasks efficiently?" question is genuinely interesting.**

---

### 3.10 SNN for EEG / Brain-Computer Interface

**Literature: 15-20 papers**

Key works:
- [SNN for EEG signal analysis with snnTorch (Frontiers 2025)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1652274/pdf)
- [Advancing EEG stress detection using SNN and CSNN (Nature Scientific Reports 2025)](https://www.nature.com/articles/s41598-025-10270-0)
- [SNN with Adaptive Graph Convolution and LSTM for EEG-Based BCI (IEEE 2023)](https://ieeexplore.ieee.org/document/10049464/)

**NOTE:** Already covered in SNN_EEG_BCI_Research_Report.md.

**VERDICT: Already researched. See existing report.**

---

### 3.11 SNN for Privacy / Federated Learning (NEW -- NOT PREVIOUSLY COVERED)

**Literature: 3-5 papers (VERY SPARSE, ALL 2023-2025)**

Key works:
- [Privacy in Federated Learning with SNNs (arxiv 2025)](https://arxiv.org/abs/2511.21181)
- [Sensitivity of Firing Rate-Based Federated SNNs to Differential Privacy (arxiv 2026)](https://arxiv.org/abs/2602.12009v1)
- [Privacy-Preserving Federated Neuromorphic Learning via Spiking Neuron Models (MDPI 2023)](https://www.mdpi.com/2079-9292/12/18/3984)

**Key finding:** SNNs' discrete spike events and surrogate gradient approximations may provide inherent privacy advantages -- the "noise" from surrogate gradients could naturally obscure gradient information, making gradient inversion attacks harder.

**Assessment:**
- Papers: 3-5 (EXTREMELY novel)
- Simulation-only: YES
- macOS: YES
- 28-day feasibility: **LOW-MODERATE**
- Components: snnTorch + federated learning framework (Flower/PySyft) + MNIST/CIFAR

**Why this is interesting:**
- Cutting-edge intersection of two hot fields (federated learning + neuromorphic)
- The arxiv 2026 paper is literally weeks old
- Novel question: "Do SNNs naturally resist gradient inversion attacks?"
- Clear practical motivation (privacy-preserving edge AI)

**Why it's risky:**
- Federated learning frameworks add significant complexity
- Need to understand gradient inversion attacks (non-trivial)
- Very few references to build on
- May be too theoretical for 28 days

**VERDICT: FASCINATING but risky. If the student is interested in security/privacy, this is a bleeding-edge direction. The 28-day timeline makes it tight. Would need to simplify significantly (e.g., just compare gradient leakage between SNN and ANN without full federated infrastructure).**

---

### 3.12 SNN for Recommendation Systems (NEW -- NOT PREVIOUSLY COVERED)

**Literature: 1-2 papers**

Key work:
- [Cost-Effective On-Device Sequential Recommendation with SNNs (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/0398.pdf)

**Assessment:**
- Papers: 1-2 (EXTREMELY sparse)
- Very new (IJCAI 2025)
- Feasibility: MODERATE (if focusing on simple sequential recommendation)
- The energy-efficient on-device angle is compelling

**VERDICT: TOO SPARSE to build on. Only 1-2 papers means almost no reference framework. Interesting future direction but not practical for 28 days.**

---

### 3.13 Part 3 Summary Table

| Application | Papers | Novel? | macOS? | 28-Day Feasible? | Already Covered? |
|---|---|---|---|---|---|
| Game AI (CartPole) | 10-15 | Low | YES | Moderate-High | Partially (overlaps RL) |
| **Cybersecurity/IDS** | **10-20** | **Moderate** | **YES** | **HIGH** | **YES (existing report)** |
| Financial Trading | 10-15 | Low | YES | Moderate | YES (Ezinwoke did it) |
| Autonomous Driving | 20+ | Very Low | No (CARLA) | Low | No |
| Weather/Time Series | 5-10 | Moderate | YES | Moderate-High | YES (existing report) |
| Drug Discovery | 1-2 | Very High | YES | Low | No |
| **ECG Classification** | **15-20** | **Low** | **YES** | **HIGH** | **YES (existing report)** |
| NLP/Text | 15-25 | Low | YES | Moderate | YES (existing report) |
| **Image Denoising** | **3-5** | **VERY HIGH** | **YES** | **Moderate-High** | **NO -- NEW** |
| EEG/BCI | 15-20 | Low | YES | High | YES (existing report) |
| **Privacy/Federated Learning** | **3-5** | **VERY HIGH** | **YES** | **Low-Moderate** | **NO -- NEW** |
| Recommendation Systems | 1-2 | Very High | YES | Low-Moderate | NO -- NEW |

---

## PART 4: CROSS-CUTTING ANALYSIS AND RECOMMENDATIONS

---

### Truly NEW Directions Not Previously Covered

Based on cross-referencing with all existing research files in `/Users/kumar/Documents/University/Year3/thesisproject/research/`, these are the genuinely new findings from this research:

| Rank | Direction | Novelty | Feasibility | Risk |
|---|---|---|---|---|
| 1 | **SNN for Continual Learning (Split-MNIST)** | Moderate-High | Moderate-High | Moderate (needs novel angle) |
| 2 | **SNN for Image Denoising** | Very High | Moderate-High | Moderate (architecture complexity) |
| 3 | **SNN for Robotic Arm Control (MuJoCo)** | Moderate | Moderate | Moderate (RL instability) |
| 4 | **SNN for Privacy/Federated Learning** | Very High | Low-Moderate | High (complexity) |
| 5 | **SNN for Obstacle Avoidance RL** | Low-Moderate | Moderate-High | Moderate (well-trodden) |

---

### Detailed Feasibility Breakdown: Top 3 New Candidates

#### Candidate A: SNN for Continual Learning

**Week-by-week plan:**
- Week 1: snnTorch MNIST baseline (Tutorial 5), implement Split-MNIST data loader
- Week 2: Measure baseline catastrophic forgetting, implement EWC for SNN
- Week 3: Implement experience replay for SNN, run experiments
- Week 4: ANN comparison, energy analysis, write-up

**Novel angle options:**
1. "EWC adapted specifically for spiking neuron dynamics" (Fisher Information over membrane potentials)
2. "Comparing spike-timing-based vs rate-based encoding for continual learning"
3. "Threshold modulation as a biologically-inspired continual learning mechanism"

**Biggest risk:** The 2023 paper "Investigating Continuous Learning in SNNs" already covers Split-MNIST with SNNs. Must differentiate clearly.

**snnTorch compatibility:** FULL -- standard classification pipeline with multiple training phases.

#### Candidate B: SNN for Image Denoising

**Week-by-week plan:**
- Week 1: Implement DnCNN ANN baseline, prepare Set12/BSD68 data
- Week 2: Design and implement SNN denoising network in snnTorch
- Week 3: Train with different noise levels (sigma=15,25,50), evaluate PSNR/SSIM
- Week 4: Energy analysis, ablation studies, write-up

**Novel angle:** Being one of the first snnTorch-based image denoising implementations. The SPIDEN paper uses custom code, not snnTorch.

**Biggest risk:** The SNN architecture for denoising may require skip connections and multi-scale features that are non-trivial in snnTorch. May need to simplify to a shallower architecture.

**snnTorch compatibility:** MODERATE -- snnTorch supports Conv2d layers but U-Net skip connections need custom work.

#### Candidate C: SNN for Robotic Arm Control in MuJoCo

**Week-by-week plan:**
- Week 1: Set up MuJoCo Reacher-v4 in Gymnasium, implement ANN-DQN baseline
- Week 2: Replace policy network with SNN, adapt training loop
- Week 3: Hyperparameter tuning, convergence analysis
- Week 4: Energy comparison, analysis, write-up

**Novel angle:** snnTorch-based implementation (most papers use SpyTorch or custom). Could also compare different neuron models (LIF vs Synaptic vs Alpha).

**Biggest risk:** SNN+RL convergence is notoriously difficult. May spend all of Week 3 on hyperparameter tuning without getting good results.

**snnTorch compatibility:** MODERATE -- need to integrate snnTorch neurons into RL training loop (no built-in RL support).

---

### What NOT to Pursue (and Why)

| Direction | Why Not |
|---|---|
| SNN for AGI (philosophical) | Not a technical thesis. Kyambadde's approach is philosophy. |
| SNN Meta-Learning | MAML + SNN is PhD-level complexity |
| SNN Cognitive Architecture | PhD-scale, massive scope, neuroscience prerequisite |
| SNN for Drones | Simulator setup alone takes a week+ on macOS |
| SNN for Swarm Robotics | Multi-agent complexity, bleeding-edge research |
| SNN for Autonomous Driving | Too established, needs NVIDIA GPU, CARLA Linux-only |
| SNN for Drug Discovery | 1-2 papers, biochemistry domain knowledge needed |
| SNN for Financial Trading | Ezinwoke already did it at Manchester with same supervisor |
| SNN for SLAM | Requires Nengo (not snnTorch), SLAM is inherently complex |

---

### Confidence Assessment

| Finding | Confidence |
|---|---|
| Kyambadde thesis is philosophy, not CS/SNN | **HIGH** (confirmed via White Rose repository metadata) |
| SNN + continual learning is an active, viable field | **VERY HIGH** (20+ papers, including Science Advances and Nature Comms) |
| SNN + meta-learning is too hard for 28 days | **HIGH** (no snnTorch support, MAML complexity) |
| SNN + image denoising has very few papers | **HIGH** (found only 3-5 across all databases) |
| MuJoCo works on macOS | **HIGH** (confirmed in MuJoCo documentation) |
| snnTorch runs on macOS CPU | **VERY HIGH** (PyTorch-based, no CUDA required) |
| SNN+RL training is finicky | **VERY HIGH** (multiple papers explicitly note hyperparameter sensitivity) |

---

### Research Gaps Identified

1. **No snnTorch-specific tutorial for continual learning** -- this is a gap that a thesis could fill
2. **No snnTorch implementation of image denoising** -- SPIDEN uses custom code
3. **No snnTorch+MuJoCo integration guide** -- existing papers use SpyTorch or custom SNN
4. **No systematic comparison of EWC variants in spiking vs non-spiking networks** -- theoretical interest
5. **No Manchester undergraduate thesis on continual learning with SNNs** -- all existing Manchester SNN theses focus on classification or financial prediction

---

## SOURCES

### Part 1: AGI-Related
- [Kyambadde PhD Thesis, White Rose Repository](https://etheses.whiterose.ac.uk/id/eprint/36019/)
- [NACA Algorithm, Science Advances 2024](https://www.science.org/doi/10.1126/sciadv.adi2947)
- [CH-HNN, Nature Communications 2025](https://www.nature.com/articles/s41467-025-56405-9)
- [CLP-SNN on Loihi 2, arxiv 2024](https://arxiv.org/html/2511.01553v1)
- [Energy-Aware Spike Budgeting, arxiv 2026](https://arxiv.org/html/2602.12236)
- [Meta-SpikePropamine, PMC 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10213417/)
- [SNN-MAML, arxiv 2022](https://arxiv.org/abs/2201.10777)
- [BrainCog, arxiv/GitHub](https://github.com/braincog-x/brain-cog)
- [SPAUN/Nengo Documentation](https://www.nengo.ai/nengo-spa/user-guide/spa-intro.html)
- [Predictive Coding with SNNs, Frontiers 2024](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1338280/full)
- [Contrastive Predictive Coding + SNN, arxiv 2025](https://arxiv.org/html/2506.09194)

### Part 2: Robotics
- [SNN for Deep RL in Robotic Tasks, Nature Scientific Reports 2024](https://www.nature.com/articles/s41598-024-77779-8)
