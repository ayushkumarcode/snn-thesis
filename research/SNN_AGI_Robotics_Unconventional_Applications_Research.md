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
