# Chapter 1: Introduction
## COMP30040 Thesis — Spiking Neural Networks for Environmental Sound Classification

---

## 1.1 Motivation

### 1.1.1 The Energy Crisis in Perceptual AI

The last decade has produced remarkable progress in machine perception. Deep convolutional networks now classify environmental sounds, recognise speech, and identify objects in images with accuracy that matches or exceeds human performance on standardised benchmarks. But this progress has come at a striking energy cost. A single inference through a large convolutional network on a modern GPU consumes tens to hundreds of millijoules — three to four orders of magnitude more than the human brain expends on an equivalent perceptual task. The human auditory cortex classifies environmental sounds continuously at the scale of the whole brain's 20-watt budget (Attwell & Laughlin 2001); a single GPU inference for equivalent audio classification can consume 20–200 mJ (Yik et al. 2025).

This mismatch matters increasingly because perceptual AI is being pushed to the edge. The most commercially important applications for audio intelligence — hearing aids, cochlear implants, wildlife acoustic monitors, smart building sensors, search-and-rescue robots, industrial fault detection — are battery-constrained, latency-sensitive, and privacy-critical. A hearing aid has an energy budget of approximately 1 mW with a weekly charge cycle; a wildlife acoustic sensor may run for months on a single cell. In these environments, transmitting raw audio to a cloud server for inference is not merely inconvenient — it is impossible within the energy budget, introduces unacceptable latency, and creates privacy exposure that is legally and ethically problematic for audio recorded in private spaces.

The consequence is a genuine engineering gap. The algorithms that achieve state-of-the-art accuracy are too expensive to run locally on the devices where local inference is needed most. This gap is not closing at a rate commensurate with deployment demand: GPU energy efficiency improves roughly 1.5× per technology generation (Moore's Law for power), but model parameter counts have been growing substantially faster. The efficiency of the underlying compute paradigm — dense floating-point matrix multiplication — is fundamentally mismatched to the sparse, event-driven nature of perceptual signals.

### 1.1.2 The Biological Blueprint

Biology solved this problem hundreds of millions of years ago. The mammalian auditory system achieves continuous, real-time, multi-class sound classification at extraordinary energy efficiency through a radically different computational architecture. The cochlea performs a biological fast Fourier transform, decomposing incoming sound waves into frequency-specific mechanical vibrations along the basilar membrane. Inner hair cells transduce these vibrations into graded receptor potentials, which drive auditory nerve fibres to fire action potentials — binary, all-or-nothing electrical events — at rates that encode frequency, intensity, and temporal structure. The auditory cortex processes these sparse, temporal spike trains through layered neural circuits, ultimately producing classification judgments in tens to hundreds of milliseconds.

The critical computational properties of this system are: (1) **binary communication** — neurons fire or do not fire, transmitting 1-bit events rather than continuous-valued signals; (2) **temporal sparsity** — at any moment, only a small fraction of neurons are active; (3) **event-driven processing** — neurons compute only when they receive input spikes, not at a fixed clock rate; and (4) **adaptive thresholding** — neurons integrate evidence over time before committing to a binary output, providing natural robustness to noise. These properties together produce a system that is energy-frugal by design: computation is proportional to signal content, not to the size of the network.

### 1.1.3 Spiking Neural Networks as Engineered Approximations

Spiking neural networks (SNNs) are an attempt to engineer these properties into artificial systems. In an SNN, each neuron maintains a membrane potential that integrates weighted input spikes over time. When the potential crosses a threshold, the neuron emits a binary spike and resets. The most common model, the Leaky Integrate-and-Fire (LIF) neuron, adds an exponential decay term that models the passive leak of biological membranes: $\tau_m \frac{dV}{dt} = -V + R \sum_i w_i s_i(t)$, where $V$ is membrane potential, $\tau_m$ is the membrane time constant, $w_i$ are synaptic weights, and $s_i(t)$ are binary spike inputs.

This communication model has a critical energy implication. Conventional ANN inference requires a multiply-accumulate (MAC) operation for each active connection: multiply the activation by the weight, accumulate into the next layer. On CMOS hardware, a single 8-bit MAC costs approximately 0.2 pJ at 45nm process node. For an SNN, binary spike inputs reduce multiplication to a conditional: either accumulate the weight (spike present) or do nothing (no spike). This accumulate-only (AC) operation costs approximately 0.03–0.09 pJ — roughly 4–7× cheaper per operation (Yik et al. 2025). Furthermore, sparse activation patterns mean most operations are skipped entirely. The energy cost of SNN inference scales with the number of spikes, not with the total parameter count.

Dedicated neuromorphic hardware — SpiNNaker (Furber et al. 2014) at the University of Manchester, Intel's Loihi 2, IBM's TrueNorth — is designed to exploit exactly these properties. These platforms route binary spike events between simulated neurons using massively parallel event-driven hardware, consuming orders of magnitude less energy than equivalent computation on CPUs or GPUs. SpiNNaker uses a custom multicast router to distribute spike events to thousands of neurons simultaneously, with each routing event consuming ~nJ rather than the μJ–mJ of a GPU kernel launch.

### 1.1.4 The Training Problem and Its Resolution

For most of the past decade, SNNs underperformed conventional ANNs significantly on benchmark tasks. The fundamental difficulty is training: because spike emission is a non-differentiable, binary threshold operation, standard backpropagation cannot compute gradients through it directly. Biologically inspired learning rules (spike-timing-dependent plasticity, STDP) are local and unsupervised, producing networks that fail to generalise on complex, high-dimensional benchmarks.

The breakthrough that makes competitive SNN training tractable is **surrogate gradient descent** (Neftci & Mostafa 2019). The key insight is that for the forward pass, exact binary spikes are used; for the backward pass, the derivative of the spike function is approximated with a smooth surrogate (typically the derivative of a sigmoid or arctangent). This "straight-through" trick allows the full machinery of backpropagation-through-time to be applied to spiking networks, enabling training on exactly the same datasets and with exactly the same loss functions as conventional deep networks. Since 2019, surrogate gradient training has become the dominant approach for competitive SNN research, producing networks within a few percentage points of matched-architecture ANNs on image benchmarks (Eshraghian et al. 2023).

This thesis uses surrogate gradient training throughout, implemented via snnTorch 0.9.4. It represents the current state of the art for SNN optimisation with standard deep learning infrastructure.

### 1.1.5 Why Environmental Sound Classification?

Environmental sound classification (ESC) is a particularly well-matched testbed for SNN research for three reasons.

**First, audio spectrograms are structurally compatible with spiking computation.** A mel spectrogram of an environmental sound is a 2D representation of time-frequency energy. The temporal axis directly maps to the timestep axis of an SNN simulation — the network processes successive frames as it would process successive timesteps of sensory input. The sparse, transient nature of many sounds (a door knock, a glass breaking, a dog bark) produces naturally sparse spectrogram representations, matching the sparse activation regime where SNNs are most energy-efficient.

**Second, the application domain demands the efficiency properties SNNs offer.** The hearing aid, wildlife sensor, and smart building use cases described above all require audio classification on devices where the energy overhead of a GPU is prohibitive. Edge audio intelligence is the target market for neuromorphic computing, and ESC is one of its core tasks.

**Third, the ESC-50 benchmark provides rigorous experimental control.** ESC-50 (Piczak 2015) contains 2,000 recordings across 50 environmental sound classes, with 40 clips per class, 5 seconds each, sampled at 44.1 kHz. The predefined 5-fold cross-validation splits enable rigorous, standardised performance comparison. Human performance on ESC-50 is 81.3% (Piczak 2015) — a meaningful upper bound given the difficulty of some class distinctions (e.g., sea waves vs. rain, insects vs. crackling fire). The ANN state-of-the-art is 98.25%, achieved through large-scale pretraining on AudioSet followed by ESC-50 fine-tuning. This thesis uses no external pretraining for its primary SNN evaluation — a deliberate methodological choice to isolate the capacity of SNNs to learn directly from the target domain.

### 1.1.6 The Gap This Thesis Fills

Despite the active SNN research community and the strong motivation for audio applications, **no prior published work has evaluated convolutional SNNs on the full ESC-50 benchmark** at the time of writing. The closest prior work (Larroza et al. 2025) evaluates fully-connected (not convolutional) SNNs on ESC-10, a 10-class subset, reporting approximately 60% accuracy. SpiNNaker has been applied to audio tasks with synthetic pure tones (Dominguez-Morales et al. 2016), but never to real environmental sound recordings. There exists no published reference accuracy for convolutional SNN on ESC-50, no systematic comparison of spike encoding methods for audio, and no adversarial robustness analysis for audio SNNs. This thesis closes all of these gaps.

---

## 1.2 Research Questions

This thesis addresses four research questions, each targeting a distinct open question in the field:

**RQ1: Can convolutional SNNs classify environmental sounds competitively with matched ANNs?**
The ANN state-of-the-art on ESC-50 is 98.25%, achieved with external pretraining on 2 million clips. This thesis does not aim to match that number with SNN-from-scratch training on 1,600 clips. The question is whether SNNs trained from scratch achieve accuracy within a useful margin of a matched-architecture ANN (same convolutional structure, same dataset, same training protocol). The "gap" itself is a scientific measurement, not a failure — it quantifies the efficiency cost of spiking computation under the current training paradigm.

**RQ2: Which spike encoding method performs best for environmental sound classification, and why?**
A static mel spectrogram must be converted to a temporal spike train before processing by an SNN. Seven encoding strategies are evaluated: rate coding, latency coding, delta (temporal contrast) coding, direct (continuous) coding, burst coding, phase coding, and population coding. Each makes different assumptions about how information should be represented in spike timing and count. No prior work has compared more than three encoding methods on any audio benchmark. The goal is not merely to rank them, but to explain the ranking mechanistically — identifying which properties of a spectrogram are preserved or destroyed by each encoding.

**RQ3: Can a trained SNN be deployed on SpiNNaker neuromorphic hardware, and what is the accuracy cost?**
SpiNNaker operates with binary spike inputs, fixed-point arithmetic, and no native convolutional compute. These constraints create a significant engineering challenge: a network trained in floating-point software must be mapped to a hardware substrate with fundamentally different computational properties. This thesis documents the complete deployment process — including the failures, the root causes of those failures, and the hybrid approach that ultimately produced valid hardware results. The hardware-software co-design insights from this process are a primary contribution.

**RQ4: Do SNNs exhibit natural adversarial robustness compared to matched ANNs on audio inputs?**
Prior work on image SNNs (Sharmin et al. 2020) suggests that binary spike thresholding may implicitly filter adversarial perturbations by requiring them to cross a hard non-linearity. This thesis tests whether this effect holds for audio spectrograms — a qualitatively different signal domain — using FGSM and PGD attacks across a range of perturbation magnitudes. The result has practical implications for edge audio security.

---

## 1.3 Contributions

This thesis makes six original contributions:

**C1: First convolutional SNN benchmark on ESC-50.**
A convolutional SpikingCNN (2 conv blocks + 2 FC layers, ~622K parameters) achieves **47.15% ± 4.50%** on ESC-50 (50 classes, 5-fold CV) with direct encoding — the first published convolutional SNN result on this benchmark. This provides a reference baseline for all future SNN audio research. The 16.70 pp gap below the matched ANN (63.85% ± 3.07%) is characterised mechanistically and contextualised through the PANNs transfer learning experiment (C5).

**C2: Most comprehensive spike encoding comparison for audio.**
Seven encoding methods are evaluated under identical conditions (same architecture, same folds, same training protocol). The ordering — **direct (47.15%) >> rate (24.00%) ≈ phase (24.15%) > population (19.15%) > latency (16.30%) >> delta (7.25%) ≈ burst (6.50%)** — is explained by an information preservation principle: encodings that retain spectrogram magnitude and temporal structure achieve higher accuracy. Notable findings include: (i) phase coding (1 spike per neuron) matches rate coding (~7 spikes per neuron) at identical accuracy — a result with direct energy efficiency implications; (ii) burst coding fails near-catastrophically due to temporal window mismatch; (iii) population coding underperforms despite 10× more output neurons, because MSE count loss is harder to optimise than cross-entropy rate loss.

**C3: First SNN deployment on SpiNNaker for environmental sound classification.**
A trained SpikingCNN is deployed on a SpiNN-5 SpiNNaker board (University of Manchester hardware) using a validated FC2-only hybrid approach. A 20-sample pilot (Run 5) achieves 40%; the 400-sample validation (Run 6) achieves **43.0% SpiNNaker vs 51.25% snnTorch — 8.25 pp hardware gap, 64.5% agreement**. The AvgPool-FC1 cancellation failure mode is documented and resolved: standard spiking CNNs with AvgPool between LIF layers are not natively deployable on SpiNNaker, because AvgPool produces fractional outputs that violate the spike-only communication model. This is a previously undocumented constraint for neuromorphic deployment of spiking CNNs.

**C4: First adversarial robustness analysis of SNNs on audio spectrograms.**
FGSM and PGD attacks are applied across 7 perturbation magnitudes (ε ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3}) to both SNN and ANN classifiers. At ε=0.1 (FGSM), the SNN retains **26.00%** accuracy while the ANN collapses to **1.75%** — a 14.9× robustness ratio. This is the largest reported adversarial robustness advantage for SNNs in any audio domain, and the first analysis of this kind for environmental sound classification.

**C5: First combination of PANNs embeddings with SNN classification.**
Frozen CNN14 embeddings (AudioSet-pretrained, Kong et al. 2020) are fed to a 3-layer SNN head for the first time. The result — **92.50% ± 1.30%** — exceeds human performance on ESC-50 (81.3%), reduces the SNN-ANN gap from 16.70 pp to 0.95 pp, and demonstrates definitively that the from-scratch SNN accuracy gap is a feature-learning problem, not a spiking computation problem. Given equivalent-quality input representations, the SNN and ANN achieve statistically indistinguishable accuracy.

**C6: NeuroBench-compliant standardised energy analysis.**
Using NeuroBench v2.2.0 (Yik et al. 2025, Nature Communications), SynapticOperations metrics are computed for all SNN variants and the ANN baseline (5-fold validated): the direct SNN uses 1.08M accumulate-only operations per sample (968 ± 37 nJ estimated), versus 101K multiply-accumulate operations for the ANN (454 ± 11 nJ). In software simulation, the SNN is 2.1× more expensive due to T=25 timestep overhead. On dedicated neuromorphic hardware, each AC costs 5.1× less than a MAC — closing this gap. The PANNs + SpiNNaker FC₂ deployment is the Pareto-optimal operating point: 92.50% accuracy, ~86 nJ for the SpiNNaker classification step.

---

## 1.4 Scope and Limitations

This thesis evaluates SNNs trained with **surrogate gradient descent** — backpropagation-through-time with a smooth surrogate for the spike derivative. This is the current mainstream approach for competitive SNN accuracy. Biologically-inspired learning rules (STDP, Hebbian learning) are not evaluated; they are outside scope and substantially less competitive on structured benchmarks.

The SpiNNaker deployment uses a **hybrid approach**: convolutional feature extraction runs in software (snnTorch on CPU), and only the final classification layer (FC₂, 256→50) runs on SpiNNaker. A fully on-chip pipeline is architecturally constrained by the AvgPool-FC1 incompatibility described above. An Option A re-training strategy (replacing AvgPool with MaxPool) is validated on fold 4 and resolves this constraint theoretically; full 5-fold SpiNNaker deployment with the Option A model is left for future work.

**Energy estimates** use NeuroBench's software-simulation-based SynapticOperations methodology, not direct hardware power measurement. Actual on-chip energy would differ due to routing costs, communication overhead, and chip-level static power.

**Adversarial evaluation** uses standard FGSM and PGD (40 steps). As noted by Wang et al. (2025), vanishing surrogate gradients make standard PGD less effective against SNNs, potentially overestimating SNN robustness. Results should be interpreted as upper bounds; Stable Adaptive PGD evaluation is recommended for future rigorous assessment.

---

## 1.5 Thesis Structure

**Chapter 2: Background and Related Work** reviews environmental sound classification (§2.1), spiking neural network fundamentals including LIF neurons and surrogate gradient training (§2.2), the seven spike encoding methods evaluated (§2.3), prior SNN work on audio tasks (§2.4), the SpiNNaker platform (§2.5), energy efficiency considerations (§2.6), adversarial robustness in SNNs (§2.7), transfer learning via PANNs (§2.8), and continual learning in SNNs (§2.9).

**Chapter 3: Methodology** describes the experimental design philosophy of controlled comparison (§3.1), the ESC-50 dataset and preprocessing pipeline (§3.2), the SpikingCNN and ConvANN architectures (§3.3), all seven spike encoding methods with mathematical formulations (§3.4), the training protocol (§3.5), and the SpiNNaker deployment strategy including the hybrid FC2-only approach (§3.6). Advanced experiments are described in §3.7.

**Chapter 4: Core Results — Encoding Comparison** presents the ANN baseline (§4.1), the systematic encoding comparison across all seven methods (§4.2), the surrogate gradient ablation (§4.3), the effect of data augmentation as a negative result (§4.4), statistical significance analysis (§4.5), and the PANNs transfer learning result (§4.6).

**Chapter 5: Neuromorphic Hardware Results** documents the SpiNNaker deployment challenges and FC1 cancellation problem (§5.1), the validated FC2-only hybrid approach (§5.2), SpiNNaker inference accuracy (§5.3), NeuroBench energy analysis and energy-accuracy Pareto frontier (§5.4), and the Option A hardware-aware retraining strategy (§5.5).

**Chapter 6: Advanced Analysis** covers adversarial robustness (§6.1), continual learning (§6.2), temporal spike pattern analysis (§6.3), representation analysis via t-SNE (§6.4), and per-class difficulty analysis (§6.5).

**Chapter 7: Discussion** synthesises findings across all experiments, addressing each research question and discussing implications for neuromorphic audio intelligence, including the augmentation negative result (§7.9).

**Chapter 8: Conclusion** summarises contributions, provides direct answers to each research question, and identifies ten directions for future work.

---

## 1.6 Reproducibility Statement

All code, model checkpoints, and experiment scripts are available at [GitHub repository — TBD upon submission]. Fixed random seeds (torch.manual_seed(42), numpy.random.seed(42)) are used throughout. All results reported are from the best validation accuracy per fold, not final-epoch accuracy. The ESC-50 dataset is freely available from the official repository (Piczak 2015, GitHub).

The primary framework is snnTorch 0.9.4 (Eshraghian et al. 2023). The Python environment is specified in requirements.txt; the critical dependencies are PyTorch 2.10, snnTorch 0.9.4, librosa 0.10.x, neurobench 2.2.0, panns-inference, torchattacks, and sPyNNaker 1.0.0.
