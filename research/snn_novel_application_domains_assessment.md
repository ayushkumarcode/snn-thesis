# SNN Novel Application Domains: What's Actually Underexplored?

went through a bunch of potential application domains for SNNs to figure out what's genuinely underexplored vs what's already been done to death. searched arxiv, Google Scholar, IEEE, Springer, and GitHub pretty thoroughly.

the tl;dr: **music generation**, **astronomy transient detection**, and **drug discovery** have the fewest papers and are the most genuinely novel. **wearable sensor data**, **radar/sonar**, and **industrial anomaly detection** are moderately explored with clear SNN advantages. **NLP/sentiment**, **game playing/RL**, and **financial fraud** have growing but established literature. **environmental monitoring** is in a middle ground with a handful of pioneering papers.

for an undergrad thesis balancing novelty, feasibility, and natural SNN advantage, i'd say the best candidates are: **(1) music generation**, **(2) environmental monitoring**, **(3) wearable sensor data**, and **(4) industrial IoT anomaly detection**.

---

## Domain-by-Domain Breakdown

---

### 1. SNN for Music Generation / Audio Synthesis

**papers that exist: about 5-8 total**

| Paper | Year | Key Contribution |
|-------|------|------------------|
| Stylistic Composition of Melodies (NeuCube) | 2021 | First SNN melody composition using STDP and sequential memory |
| Musical Pattern Recognition in SNNs | ~2018 | First-layer note differentiation in monophonic sequences |
| SpiNNaker Audio Classification | ~2019 | 3-layer LIF for pure tones on SpiNNaker |
| Mode-conditioned music learning | 2024 | Tonality-aware SNN for musical mode/key |
| MuSpike: Benchmark for Symbolic Music Generation | 2025 | First comprehensive benchmark; 5 SNN architectures across 5 datasets |
| Spiking Vocos | 2025 | Spiking vocoder for audio synthesis |

**why SNNs make sense here:** music is inherently temporal and spike-like (note onsets, rhythmic patterns). MIDI events are discrete, event-driven data. biological auditory processing uses spike-timing codes. STDP mirrors associative musical memory. energy efficiency matters for real-time embedded music.
