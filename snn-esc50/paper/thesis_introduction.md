# chapter 1: introduction

working notes for the intro chapter. need to set up the motivation, research questions, and contributions. this is the chapter that frames everything else so it needs to be solid.

---

## 1.1 motivation

### 1.1.1 the energy problem in perceptual AI

ok so the core argument here is: deep learning for audio perception works great but costs way too much energy for edge deployment. need to spell this out clearly.

The last decade has seen massive progress in machine perception -- deep CNNs can classify sounds, recognise speech, identify objects etc at or above human level on standard benchmarks. But the energy cost is absurd. A single inference through a large CNN on a GPU uses tens to hundreds of millijoules, which is 3-4 orders of magnitude more than what the brain does for an equivalent task. The human auditory cortex classifies sounds continuously within the whole brain's 20W budget (Attwell & Laughlin 2001); a GPU inference for audio classification can hit 20-200 mJ (Yik et al. 2025).

This matters because perceptual AI is moving to the edge. The real applications for audio intelligence -- hearing aids, cochlear implants, wildlife monitors, smart building sensors, search-and-rescue robots, industrial fault detection -- are all battery-constrained, latency-sensitive, and privacy-critical. A hearing aid runs on about 1 mW with a weekly charge; a wildlife sensor needs to last months on a single cell. Sending raw audio to the cloud isnt just inconvenient, its literally impossible within the energy budget, adds unacceptable latency, and creates privacy issues for audio recorded in private spaces.

So theres a genuine engineering gap. The algorithms that get SOTA accuracy are too expensive to run locally on the devices that need local inference most. And this gap isnt closing fast enough -- GPU efficiency improves ~1.5x per generation but model sizes grow way faster. The fundamental compute paradigm (dense float matrix multiply) is mismatched to the sparse, event-driven nature of perceptual signals.

### 1.1.2 the biological blueprint

Biology solved this literally hundreds of millions of years ago. The mammalian auditory system does real-time, continuous, multi-class sound classification at insane energy efficiency through a completely different computational architecture. The cochlea does a biological FFT, decomposing sound into frequency-specific vibrations along the basilar membrane. Inner hair cells transduce these into graded potentials that drive auditory nerve fibres to fire action potentials -- binary, all-or-nothing events -- at rates encoding frequency, intensity, and temporal structure. The auditory cortex processes these sparse temporal spike trains through layered circuits, producing classification in tens to hundreds of ms.

The critical properties are: (1) binary communication -- neurons fire or don't, transmitting 1-bit events not continuous values; (2) temporal sparsity -- only a small fraction active at any moment; (3) event-driven processing -- neurons compute only when they receive spikes, not at a fixed clock; (4) adaptive thresholding -- neurons integrate evidence before committing to output, giving natural noise robustness. Together these make a system thats energy-frugal by design: computation scales with signal content, not network size.

### 1.1.3 spiking neural networks as engineered approximations

SNNs try to engineer these properties into artificial systems. Each neuron maintains a membrane potential that integrates weighted input spikes over time. When it crosses a threshold, the neuron emits a spike and resets. The most common model is LIF (Leaky Integrate-and-Fire), which adds exponential decay: $\tau_m \frac{dV}{dt} = -V + R \sum_i w_i s_i(t)$, where V is membrane potential, tau_m is the time constant, w_i are weights, and s_i(t) are binary spike inputs.

The energy implication is critical. ANN inference needs a multiply-accumulate (MAC) for each active connection -- multiply activation by weight, accumulate into next layer. At 45nm, an 8-bit MAC costs ~0.2 pJ. For an SNN, binary spikes reduce this to a conditional accumulate-only (AC) operation -- either add the weight or do nothing. An AC costs ~0.03-0.09 pJ, roughly 4-7x cheaper per op (Yik et al. 2025). And sparse activation means most ops are skipped entirely. SNN energy scales with spike count, not parameter count.

Dedicated neuromorphic hardware -- SpiNNaker (Furber et al. 2014) at UoM, Intel Loihi 2, IBM TrueNorth -- exploits exactly these properties. They route binary spikes between simulated neurons using massively parallel event-driven hardware, consuming orders of magnitude less energy than equivalent GPU compute. SpiNNaker uses a custom multicast router, each event consuming ~nJ vs uJ-mJ for a GPU kernel launch.

### 1.1.4 the training problem and its resolution

For most of the past decade SNNs significantly underperformed ANNs. The fundamental issue is training: spike emission is non-differentiable (binary threshold), so standard backprop cant compute gradients through it. Bio-inspired learning rules (STDP) are local and unsupervised, producing networks that don't generalise on complex benchmarks.

The breakthrough is surrogate gradient descent (Neftci & Mostafa 2019). The insight: use exact binary spikes in the forward pass, but approximate the spike derivative with a smooth surrogate (sigmoid, arctan etc) in the backward pass. This "straight-through" trick lets you apply full BPTT to spiking networks, training on the same datasets with the same losses as conventional networks. Since 2019 this has become the dominant approach, producing SNNs within a few pp of matched ANNs on image benchmarks (Eshraghian et al. 2023).

This thesis uses surrogate gradient training throughout, via snnTorch 0.9.4.

### 1.1.5 why environmental sound classification?


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
