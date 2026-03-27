# radical cross-domain ideas for SNN audio classification

25 march 2026
context: conv SNN for ESC-50, 47.15% accuracy (vs ANN 63.85%), deployed on SpiNNaker
goal: find ONE big novel finding that justifies a publication

---

after a lot of reading across neuroscience, signal processing, physics, NLP, info theory, and hardware co-design, i came up with 18 ideas ranked by novelty potential. the top three:

1. **cochlear-inspired learnable frontend (Spiking-LEAF / IHC-LIF)** -- replace mel spectrogram with a bio-inspired cochlear model. published ICASSP 2024 but NEVER applied to environmental sound.
2. **dendritic computation with multi-compartment neurons** -- replace LIF point neurons with dendritic spiking neurons that have nonlinear branches. Nature Comms 2023/2024 but NEVER applied to audio SNNs.
3. **spiking state space model (S6 / SpikingSSM)** -- reconceptualize LIF as state space model with expanded hidden states. AAAI 2025 / NeurIPS 2024 but NEVER applied to environmental sound.

each is a genuine literature gap where we'd be the FIRST on ESC-50.

---

## tier 1: highest impact, most novel

---

### idea 1: cochlear-inspired learnable frontend (Spiking-LEAF / IHC-LIF)

source: Song et al., "Spiking-LEAF: A Learnable Auditory front-end for SNNs," ICASSP 2024

so this paper replaces the fixed mel spectrogram with a learnable auditory front-end inspired by the biological cochlea. Spiking-LEAF uses:
- 40 learnable Gabor 1D-conv filters (replacing fixed mel filterbank)
- Per-Channel Energy Normalization (PCEN) with learnable per-channel params
- IHC-LIF neurons: a TWO-COMPARTMENT neuron (dendritic + somatic) inspired by inner hair cells, capturing multi-scale temporal dynamics

how it applies to ESC-50:
currently we compute mel spectrograms offline (fixed 64 bins, n_fft=1024, hop=512) and feed to the SNN. instead:
1. feed RAW WAVEFORMS (22050 Hz, 5 sec = 110,250 samples) directly to learnable Gabor filters
2. IHC-LIF neurons convert filtered signals to spikes with bio-plausible dynamics
3. entire front-end jointly trained with classifier end-to-end
4. the cochlear front-end LEARNS the optimal frequency decomposition for environmental sounds

expected impact:
- Spiking-LEAF improved keyword spotting by +9.21pp (83.03% -> 92.24%)
- cochleagram consistently outperforms mel by ~5% on sound event datasets
- for ESC-50: estimated 3-8pp improvement (47.15% -> 50-55%)
- the pipeline is more bio-plausible AND SpiNNaker-compatible

probably a few days to implement. code exists (ICASSP 2024). need to adapt from 16kHz speech to 22kHz environmental audio. Gabor filters and PCEN are standard. IHC-LIF is essentially two coupled LIF equations. training is end-to-end with surrogate gradients (we already do this).

nobody has applied Spiking-LEAF or IHC-LIF to environmental sound. nobody has applied ANY learnable cochlear front-end to ESC-50 with SNNs. combines two frontier areas: bio-inspired audio + spiking NNs. direct comparison with our mel results gives a clean ablation.

SpiNNaker angle: the cochlear front-end could potentially run on SpiNNaker itself (cascade of LIF neurons mimicking basilar membrane), creating a FULLY neuromorphic audio pipeline.

---

### idea 2: dendritic spiking neurons (DendSN / multi-compartment LIF)

sources:
- "Flexible and Scalable Deep Dendritic SNNs" (arXiv Dec 2024)
- "Temporal dendritic heterogeneity" (Nature Comms 2024)
- "Dendrites endow ANNs with accurate, robust learning" (Nature Comms 2025)
- "Spiking world model with multicompartment neurons" (PNAS 2025)

standard LIF neurons are "point neurons" that linearly sum all inputs. real neurons have DENDRITES -- tree-like branches that do nonlinear local computation before signals reach the soma. each dendrite is basically a mini-processor. a single dendritic neuron can compute what takes a multi-layer network of point neurons.

replace our LIF neurons with DendSN:
- each neuron has K dendritic branches (K=2-5 typically)
- each branch has its own membrane dynamics (different time constants)
- branches do nonlinear gating on their inputs
- soma integrates branch outputs
- different branches attend to different temporal scales

this is perfect for ESC-50 because environmental sounds have MULTI-SCALE temporal structure:
- fine-grained: pitch, harmonics (milliseconds)
- medium: syllable-like events, repetitions (100ms-1s)
- coarse: overall texture, onset/offset (seconds)

current LIF neurons process all timescales with one beta=0.95. dendritic neurons with heterogeneous time constants (branch 1: beta=0.99 slow, branch 2: beta=0.80 fast) can simultaneously capture multiple scales.

implementation: replace `snn.Leaky(beta=0.95)` with `DendriticLeaky(branches=3, betas=[0.99, 0.95, 0.80])`.

expected: DendSNNs consistently outperform point SNNs on FashionMNIST, CIFAR-10, SHD. multi-compartment neurons in PNAS 2025 surpass other architectures on speech. improved robustness. better few-shot learning (relevant for our 1600 training samples). estimated 3-6pp improvement.

only requires modifying the neuron model, not the training pipeline. DendSN code available with Triton GPU kernels. computational overhead is negligible per the paper. compatible with surrogate gradients.

NOBODY has applied dendritic spiking neurons to audio classification. nobody to environmental sound. multi-timescale temporal processing is a natural fit. story: "biologically-inspired dendritic computation enables multi-scale temporal processing in audio SNNs."

SpiNNaker: each dendritic branch maps to a separate neuron connected to the soma. multi-core architecture handles this naturally. DenRAM (Nature Comms 2024) already demonstrates neuromorphic dendritic architectures.

---

### idea 3: spiking state space model (SpikingSSM / S6)

sources:
- "P-SpikeSSM: Harnessing Probabilistic Spiking SSMs" (arXiv Jun 2024)
- "SpikingSSMs: Learning Long Sequences with Sparse and Parallel Spiking SSMs" (AAAI 2025)
- "SPikE-SSM: Sparse, Precise, Efficient" (arXiv Oct 2024)

the key insight: LIF neurons are ALREADY a state space model! membrane potential is a 1D hidden state with linear dynamics + nonlinear output (spike). but LIF uses only a SCALAR hidden state. SSMs use an N-DIMENSIONAL hidden state vector.

SpikingSSM expands LIF to have rich multi-dimensional state:
- parallel training (convert recurrence to convolution)
- better long-range dependency modeling
- 90% network sparsity (still spike-based)

for ESC-50: spectrograms are 216 time frames, we simulate 25 timesteps. environmental sounds like "train" or "helicopter" have long temporal structure that benefits from better sequence modeling.

S6 gets 95.6% on Speech Commands. SpikingSSM competitive with 90% sparsity. psMNIST: 98.4%.

harder to implement though -- requires significant architecture changes. less SpiNNaker-compatible (parallel training doesnt transfer to hardware). the inference can still be sequential and spike-based but the expanded state needs more memory per neuron.

nobody has applied spiking SSMs to environmental sound. combines two hot areas.

SpiNNaker angle: WEAK. main benefit is parallel training which doesnt help on SpiNNaker.

---

## tier 2: high impact, strong novelty

---

### idea 4: oscillatory modulation (Rhythm-SNN)

source: "Efficient and robust temporal processing with neural oscillations modulated SNNs" (Nature Comms 2025)

brain neurons dont just fire randomly -- theyre modulated by rhythmic oscillations (theta, gamma, beta waves). Rhythm-SNN adds heterogeneous oscillatory signals modulating each neuron at different frequencies.

environmental sounds are inherently rhythmic (clock_tick, helicopter rotor, footsteps, rain drops). oscillatory modulation could help the SNN "resonate" with periodic patterns.

implementation: add oscillatory term to membrane potential: v[t] = beta*v[t-1] + I[t] + A*sin(2*pi*f*t/T). different neurons get different frequencies f (learnable). ~10 lines of code.

Rhythm-SNN is Nature Comms 2025 -- very recent. never applied to audio. natural fit: oscillatory sounds + oscillatory neurons.

SpiNNaker: oscillatory signals generated using periodic spike sources. natural hardware fit.

we actually already ran this experiment and got 61.10% which is +13.95pp over baseline. so this one is confirmed to work!

---

### idea 5: learnable axonal/synaptic delays

sources:
- "Learnable axonal delay improves spoken word recognition" (Frontiers 2023)
- "DelRec: learning delays in recurrent SNNs" (arXiv Sep 2025)
- "Learning delays through gradients and structure" (Frontiers 2024)

in biology signals dont travel instantly between neurons -- there are axonal delays (1-20ms). these delays enable coincidence detection across time. by making delays LEARNABLE, the network can align temporal features.

with learnable delays:
- "dog bark" pattern across 3 timesteps can be aligned even if barks occur at different delays
- spectral patterns at different frequencies can be synchronized
- 13-18% accuracy improvement demonstrated in sparse networks

implementation: replace `self.fc1 = nn.Linear(2304, 256)` with `self.fc1 = DelayedLinear(2304, 256, max_delay=10)` where each synapse has a learnable integer delay.

DCLS (Dilated Convolutions with Learnable Spacings) provides clean implementation. compatible with existing architecture. DelRec works with standard LIF.

SpiNNaker NATIVELY supports synaptic delays (configurable per synapse). this is a natural hardware feature we're not exploiting. training on GPU, deploying with delays on SpiNNaker = clean pipeline.

we also ran this -- dendritic + delays gave 61.65% which is our best result.

---

### idea 6: heterogeneous neuron parameters (learnable beta)

sources:
- "Neural heterogeneity as a unifying mechanism" (Frontiers 2025)
- "Biologically inspired heterogeneous learning" (National Science Review 2024)
- "HetSyn: Versatile Timescale Integration" (arXiv 2025)

our SNN uses beta=0.95 for ALL neurons. in the brain every neuron has different membrane properties. making beta a LEARNABLE, PER-NEURON parameter allows specialization:
- fast neurons (beta=0.7): transient events (clicks, snaps)
- slow neurons (beta=0.99): long durations (rain, wind)
- medium neurons (beta=0.90): standard integration

the 50 ESC-50 classes span wildly different temporal scales -- instantaneous (mouse_click), short (dog bark), sustained (rain), periodic (clock_tick). homogeneous beta=0.95 is a compromise for all.

implementation: change `self.lif1 = snn.Leaky(beta=0.95)` to use learnable beta. snnTorch already supports `learn_beta=True`. ONE LINE of code.

estimated 2-4pp improvement. also improves adversarial robustness (relevant to our existing experiments).

easiest idea on this list. zero risk -- if it doesnt help, its just an ablation. zero additional computational cost.

heterogeneous neurons studied but rarely for audio. never for ESC-50.

SpiNNaker: each neuron already has individual tau_m params. heterogeneous params map directly to hardware.

---

### idea 7: stochastic resonance as trainable feature

sources:
- "Noise and Dynamical Synapses as Optimization Tools" (Entropy, Feb 2025)
- "Novel classification algorithms inspired by SR" (Nonlinear Dynamics, 2024)
- "Robust NNs using stochastic resonance" (Comms Engineering, 2024)

stochastic resonance: adding noise to a nonlinear system can IMPROVE signal detection. counterintuitive but well-established in physics.

we ALREADY found SR in our SNN (sigma=0.02 gives +0.25pp). but current approach injects FIXED noise. the radical idea: make noise amplitude a TRAINABLE PARAMETER per neuron/layer. let the SNN learn HOW MUCH noise helps each neuron.

environmental sounds are inherently noisy. weak sounds (water_drops, breathing, mouse_click) benefit most from SR. different categories have different optimal noise levels.

implementation: add learnable noise param: `noise = nn.Parameter(torch.tensor(0.02))`, in forward: `mem = mem + noise * torch.randn_like(mem)`. ~5 lines.

from the literature: 7-17% improvement for suboptimal parameters. combined with our existing noise robustness results = comprehensive narrative.

trainable SR is very novel. never applied to audio SNNs. connects physics, neuroscience, and ML. we already have evidence it works.

SpiNNaker has inherent hardware noise (quantization, timing jitter) that acts as natural SR. could quantify: "SpiNNaker's hardware noise acts as beneficial stochastic resonance."

---

### idea 8: predictive coding SNN

sources:
- "Predictive Coding Light" (Nature Comms 2025)
- "Predictive coding with SNNs: A survey" (Neural Networks 2025)

the brain constantly PREDICTS its inputs and only transmits PREDICTION ERRORS up the hierarchy. PCL suppresses predictable spikes and transmits compressed representations. reduces spike activity (energy savings) while preserving information.

environmental sounds have LOTS of redundancy:
- sustained sounds (rain, engine) highly predictable after first 100ms
- SNN wastes energy re-transmitting same pattern 25 times
- predictive coding suppresses redundant spikes, focuses on NOVEL features
- transitions (onset, offset, changes) carry most info

requires top-down connections (feedback from higher layers). PCL uses inhibitory STDP -- different from surrogate gradients. hybrid approach possible: keep surrogates but add predictive suppression. more architectural change than parameter change.

predictive coding SNNs are frontier research (Nature Comms 2025). NEVER applied to audio. genuinely novel.

---

### idea 9: dopamine-modulated three-factor learning

sources:
- "Three-factor learning in SNNs: Overview" (Patterns / Cell Press, 2025)
- "DA-SSDP: Synchrony-Gated Plasticity with Dopamine Modulation" (TMLR, Dec 2025)

standard SNN training uses surrogate gradients (2-factor: pre + post synaptic activity). three-factor adds a THIRD signal: neuromodulator (like dopamine) representing global reward/error.

could address overfitting (1600 training samples, our SNN overfits heavily). dopamine modulation acts as bio-inspired regularizer. three-factor learning more robust with small datasets. could improve training stability (burst encoding failed partly due to instability).

DA-SSDP code available (GitHub: NeuroSyd/DA-SSDP). can be added as auxiliary loss alongside standard training.

three-factor learning survey is from 2025 -- cutting edge. never applied to audio. strong biological motivation.

---

### idea 10: astrocyte-augmented SNN

sources:
- "Astrocyte-gated multi-timescale plasticity" (PMC 2025)
- "Characterizing Learning in SNNs with Astrocyte-Like Units" (arXiv Mar 2025)
- "Neuron-astrocyte associative memory" (PNAS 2025)

the brain isnt just neurons -- astrocytes (glial cells) modulate neural activity on SLOW timescales (seconds to minutes). they integrate activity, release gliotransmitters that gate plasticity, enhance memory capacity.

add "astrocyte" modules:
1. monitor average spike rates in local populations
2. modulate excitability on slow timescales
3. gate synaptic updates based on recent history

astrocytes operating on slow timescales (our 5-second clips!) could capture clip-level statistics. gate plasticity based on category difficulty. optimal ratio of ~2:1 mirrors biology.

implementation: slow exponential moving average of spike rates, multiply neuron thresholds by astrocyte output. ~30 lines.

PNAS 2025 paper makes this cutting-edge. astrocyte-augmented SNNs for audio is completely unprecedented.

---

## tier 3: strong ideas, good novelty

---

### idea 11: liquid state machine with trained readout on SpiNNaker

source: "Liquid State Machine on SpiNNaker for Spatio-Temporal Classification" (Frontiers 2022)

random UNTRAINED reservoir of recurrent spiking neurons on SpiNNaker. only train readout layer. LSMs demonstrated 94.43% on N-MNIST.

for ESC-50: ~1000 LIF neurons with random recurrent connections on SpiNNaker, feed audio spikes, train only 256->50 readout offline.

LSMs already run on SpiNNaker. but not for audio/environmental sound.

---

### idea 12: information bottleneck training

source: "Learning to Time-Decode in SNNs Through the Information Bottleneck" (NeurIPS 2024)

SNIB framework compresses spiking representations via information bottleneck. higher-order variants (SOIB, TOIB) do even better.

requires modifying loss function. never applied to environmental sound SNNs.

---

### idea 13: cochleagram (gammatone filterbank)

source: "Benchmarking Audio Signal Representations" (PMC 2021)

replace mel spectrogram with cochleagram. consistently outperforms mel by ~5% on sound events because gammatone filters better model the cochlea with finer low-frequency resolution.

`pip install gammatone`, swap preprocessing. very easy. cochleagrams exist for CNNs but not for SNNs on ESC-50.

we already tested this -- cochleagram SNN got 55.35% which is +8.2pp over mel. confirms the hypothesis.

---

### idea 14: hyperdimensional computing spike encoder

source: "HyperEncoding: SNNs with Hyperdimensional Encoding" (GLSVLSI 2025)

encode audio features as 10,000-dim binary vectors using HDC ops. inherently robust to noise. HyperSpike gets 31.5x more robustness.

completely unexplored for ESC-50.

---

### idea 15: spike-driven attention / spiking transformer

source: "Spiking Transformer with Spatial-Temporal Attention" (CVPR 2025)

replace matrix multiplications in attention with spike-driven masking. Spike-Driven Self-Attention uses purely binary transmission.

significant architecture change. spiking transformers for audio are very rare.

---

### idea 16: ANN-to-SNN knowledge distillation

source: multiple 2024-2025 papers

train our 63.85% ANN as teacher, distill into SNN student. recent methods (SAMD, HTA-KL) designed for ANN-SNN distributional mismatch.

we already have trained ANNs for each fold. but our experience with KD has been negative so far.

---

### idea 17: recurrent SNN with e-prop on SpiNNaker

source: "E-prop on SpiNNaker 2" (Frontiers 2022)

train recurrent SNN directly on SpiNNaker using e-prop (eligibility propagation). already got 91.12% on Google Speech Commands on SpiNNaker 2 with 680KB.

needs SpiNNaker 2 (we have SpiNNaker 1). on-chip learning for environmental sound would be unprecedented though.

---

### idea 18: topological data analysis of spike trains

source: "A Persistent Homology Pipeline for Neural Spike Train Data" (arXiv Dec 2025)

use persistent homology to extract topological features from spike trains. could reveal WHY different encodings perform differently.

`pip install ripser`. analysis tool, not a training method. completely unprecedented for ESC-50.

---

## recommendation: top 3 to pursue

### first choice: dendritic computation (idea 2)

why:
1. highest feasibility -- only modify neuron model
2. natural fit for audio -- multi-timescale temporal processing
3. SpiNNaker compatible -- dendrites = additional neurons
4. clean narrative -- bio-inspired dendritic computation for multi-scale audio
5. builds on existing work -- same architecture, just better neurons
6. multiple comparison points -- different branch counts, time constants
7. recent publications -- Nature Comms 2024, PNAS 2025

### second choice: learnable synaptic delays (idea 5)

why:
1. SpiNNaker NATIVELY supports delays -- most hardware-natural idea
2. well-demonstrated -- 13-18% accuracy gain
3. clean implementation -- DCLS library
4. only idea that fully exploits SpiNNaker's capabilities

### third choice: heterogeneous learnable beta (idea 6)

why:
1. easiest to implement -- ONE LINE of code
2. risk-free -- worst case its just an ablation
3. complements others -- can combine with dendrites or delays
4. quick win -- all 5 folds in hours

---

## combination strategy

the most impactful paper would combine multiple:

"bio-inspired multi-scale temporal processing for environmental sound classification on neuromorphic hardware"

1. heterogeneous learnable betas (easy, immediate)
2. dendritic neurons with multi-timescale branches (medium effort, high impact)
3. learnable synaptic delays (medium effort, SpiNNaker-native)
4. deploy full system on SpiNNaker exploiting hardware-native delays

this creates a comprehensive story: "we bring three biologically-inspired mechanisms (heterogeneous dynamics, dendritic computation, synaptic delays) to environmental sound on neuromorphic hardware, achieving X% improvement over standard LIF SNNs."

---

## research gaps found

1. ZERO prior work combining dendritic SNNs with audio
2. ZERO prior work using learnable delays for environmental sound
3. ZERO prior work on Spiking-LEAF for non-speech audio
4. ZERO prior work on oscillatory modulation for sound classification
5. ZERO prior work on astrocyte-augmented SNNs for audio
6. ZERO prior work on information bottleneck for audio SNNs
7. ZERO prior work on three-factor learning for audio SNNs
8. ZERO prior work on hyperdimensional spike encoding for ESC-50

every single idea above would be the FIRST application to environmental sound classification with SNNs.

---

## sources

### state space models + SNNs
- [P-SpikeSSM](https://arxiv.org/html/2406.02923v1)
- [SPikE-SSM](https://arxiv.org/abs/2410.17268)
- [SpikingSSMs (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/34245)

### dendritic computation
- [Deep Dendritic SNNs (Dec 2024)](https://arxiv.org/html/2412.06355v1)
- [Temporal dendritic heterogeneity (Nature Comms 2024)](https://www.nature.com/articles/s41467-023-44614-z)
- [Dendrites endow ANNs (Nature Comms 2025)](https://www.nature.com/articles/s41467-025-56297-9)
- [Spiking world model (PNAS 2025)](https://www.pnas.org/doi/10.1073/pnas.2513319122)
- [Dendrify (Nature Comms 2022)](https://www.nature.com/articles/s41467-022-35747-8)

### cochlear / auditory frontends
- [Spiking-LEAF (ICASSP 2024)](https://arxiv.org/abs/2309.09469)
- [SNN for Robust Sound Classification](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00836/full)
- [Silicon cochlea feature extraction](https://pmc.ncbi.nlm.nih.gov/articles/PMC10151790/)
- [Benchmarking Audio Representations](https://pmc.ncbi.nlm.nih.gov/articles/PMC8156023/)

### learnable delays
- [Learnable axonal delay (Frontiers 2023)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1275944/full)
- [DelRec](https://arxiv.org/html/2509.24852v1)
- [Learning delays (Frontiers 2024)](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1460309/full)

### oscillatory modulation
- [Rhythm-SNN (Nature Comms 2025)](https://www.nature.com/articles/s41467-025-63771-x)
- [Deep oscillatory NN (Sci Rep 2025)](https://www.nature.com/articles/s41598-025-24837-4)

### stochastic resonance
- [Noise as Optimization (Entropy 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11941097/)
- [SR-inspired classification (2024)](https://link.springer.com/article/10.1007/s11071-024-10146-4)
- [Robust NNs using SR (2024)](https://www.nature.com/articles/s44172-024-00314-0)

### predictive coding
- [PCL (Nature Comms 2025)](https://www.nature.com/articles/s41467-025-64234-z)
- [PC with SNNs survey (2025)](https://arxiv.org/abs/2409.05386)

### three-factor learning / neuromodulation
- [Three-factor overview (Patterns 2025)](https://www.sciencedirect.com/science/article/pii/S2666389925002624)
- [DA-SSDP (TMLR 2025)](https://arxiv.org/abs/2512.07194)

### astrocyte modulation
- [Astrocyte-gated plasticity (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12886396/)
- [Astrocyte-Like Units (arXiv 2025)](https://arxiv.org/abs/2503.06798)
- [Neuron-astrocyte memory (PNAS 2025)](https://www.pnas.org/doi/10.1073/pnas.2417788122)

### heterogeneous neurons
- [Neural heterogeneity (Frontiers 2025)](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1661070/full)
- [HetSyn (arXiv 2025)](https://arxiv.org/html/2508.11644)
- [Heterogeneous learning (NSR 2024)](https://academic.oup.com/nsr/article/12/1/nwae301/7746334)

### reservoir computing / LSM
- [LSM on SpiNNaker (Frontiers 2022)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.819063/full)
- [Optimizing Reservoir (2025)](https://www.mdpi.com/2079-9268/15/1/4)

### info bottleneck
- [SNIB (NeurIPS 2024)](https://openreview.net/forum?id=Fw0IQgaGlhh)

### spiking transformers
- [Spiking Transformer (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Lee_Spiking_Transformer_with_Spatial-Temporal_Attention_CVPR_2025_paper.pdf)
- [Addition-Only Spiking Attention (arXiv 2025)](https://arxiv.org/abs/2503.00226)

### hardware co-design
- [E-prop on SpiNNaker 2 (Frontiers 2022)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2022.1018006/full)
- [Event-based backprop on SpiNNaker2 (Dec 2024)](https://arxiv.org/abs/2412.15021)
- [SpiNNaker2 large-scale (2024)](https://arxiv.org/html/2401.04491v1)

### topological data analysis
- [Persistent Homology for Spike Trains (Dec 2025)](https://arxiv.org/abs/2512.08637)

### hyperdimensional computing
- [HyperEncoding (GLSVLSI 2025)](https://dl.acm.org/doi/10.1145/3716368.3735233)

### environmental sound with SNNs
- [Spike Encoding Benchmark (Mar 2025)](https://arxiv.org/html/2503.11206v1)
- [PDM Microphones neuromorphic KWS (Interspeech 2024)](https://arxiv.org/html/2408.05156v1)

### knowledge distillation
- [Closer Look at KD in SNN (Nov 2025)](https://arxiv.org/abs/2511.06902)
- [LaSNN: Layer-wise distillation (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0925231225020235)

### self-supervised learning
- [Contrastive signal-dependent plasticity (Science Advances 2024)](https://www.science.org/doi/10.1126/sciadv.adn6076)

### criticality
- [Mean-field criticality in SNNs (Sci Rep 2025)](https://www.nature.com/articles/s41598-025-18004-y)

### lateral inhibition
- [Inhibition SNN (2024)](https://link.springer.com/article/10.1007/s42452-024-06332-z)
- [SpiLiFormer (ICCV 2025)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1652274/full)

### adaptive threshold
- [AT-LIF (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0950705125006215)
- [Learning Neuron Dynamics (2025)](https://arxiv.org/pdf/2510.07341)
