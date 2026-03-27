# chapter 2: background and related work

notes for the lit review chapter. need to cover ESC-50, SNN basics, encoding methods, prior audio SNN work, SpiNNaker, energy, adversarial, PANNs, continual learning. its a lot of ground to cover.

---

## 2.1 environmental sound classification

ESC is a core task in computational auditory scene analysis -- applications in smart cities (noise monitoring), wildlife conservation, assistive tech, robotics. Unlike speech recognition, ESC has to deal with the full diversity of real-world acoustic events without structured phonological patterns.

**ESC-50** (Piczak, 2015) is the de facto standard benchmark: 2000 recordings, 50 classes, 5-fold predefined CV, human performance 81.3%. ANN SOTA is 98.25%, mostly through CNN architectures on AudioSet-pretrained features (Gong et al. 2021 AST; Kong et al. 2020 PANNs).

Standard input representation is the log-mel spectrogram: audio -> STFT -> mel filterbanks (mimics cochlear spacing) -> dB scale. This works well across CNN, Transformer, and recurrent architectures.

---

## 2.2 spiking neural networks

### 2.2.1 LIF neurons

The Leaky Integrate-and-Fire neuron is the standard unit in modern SNNs:
$$\tau_m \frac{dU(t)}{dt} = -U(t) + RI(t)$$

where U(t) is membrane potential, tau_m is time constant, I(t) is input current. When U reaches threshold theta, neuron spikes and resets. Discrete form (snnTorch):
$$U[t] = \beta U[t-1] + I[t] - S[t-1]\vartheta$$
$$S[t] = \mathbf{1}[U[t] \geq \vartheta]$$

where beta = exp(-dt/tau_m) is decay rate (0.95 in this work).

### 2.2.2 surrogate gradient training

The spike function S[t] = 1[U[t] >= theta] is discontinuous, making backprop impossible (dS/dU = 0 almost everywhere). Surrogate gradient descent (Neftci et al. 2019; Zenke & Vogels 2021) substitutes a smooth function in the backward pass:

$$\frac{\partial \mathcal{L}}{\partial U} \approx \frac{\partial \mathcal{L}}{\partial S} \cdot \sigma'(U - \vartheta)$$

Common surrogates: fast sigmoid (1/(1+|x|)^2, slope=25), ATan, triangular, STE. Surrogate choice affects training stability and sparsity; this work ablates all 8 surrogates in snnTorch 0.9.4.

**snnTorch** (Eshraghian et al. 2023) is the primary framework -- PyTorch-compatible LIF, Leaky, Recurrent, Synaptic neuron types, spike generation, population coding and loss utilities.

---

## 2.3 spike encoding methods

Spike encoding converts static inputs (images, spectrograms) into temporal spike patterns. The encoding method is a critical design choice -- determines information density, energy cost, and task suitability.

**Rate coding** (de Ruyter van Steveninck et al. 1997): spike probability proportional to intensity. Maximises info per spike but stochastic, needs many timesteps. Biologically plausible.

**Latency/time-to-first-spike** (Thorpe et al. 1996): higher intensity = earlier spike. Efficient (1 spike/neuron) but noise-sensitive.

**Delta/temporal contrast** (Lichtsteiner et al. 2008): spikes on positive intensity changes. Directly implements DVS output. Poor for static inputs -- this is going to be a problem for us.

**Direct/continuous** (Rathi et al. 2020; Yin et al. 2021): continuous values fed directly to LIF neurons which do their own implicit coding. Best accuracy in practice; not truly sparse input but network activation is sparse.

**Burst coding** (Izhikevich 2004): neurons fire N spikes in rapid succession, N encodes intensity. Observed in auditory and visual cortex biologically.

**Phase coding** (O'Keefe & Recce 1993; Montemurro et al. 2008): spike timing relative to a global oscillation cycle encodes intensity. Theta-phase precession in hippocampus is the canonical example.

**Population coding** (Georgopoulos et al. 1986): each class represented by a population of neurons rather than one. Output-side population codes let multiple neurons vote per class, potentially reducing noise sensitivity. Implemented via snnTorch's SF.mse_count_loss(population_code=True).

No prior work has systematically compared these seven on ESC-50 or any standard audio benchmark. Closest is Larroza et al. 2025 who do 3 methods (rate, latency, direct) on ESC-10 only.

---

## 2.4 SNNs for audio classification

### 2.4.1 prior work summary

| Paper | Benchmark | Model | Encoding | Accuracy | Hardware |
|-------|-----------|-------|----------|----------|----------|
| Larroza et al. 2025 (arXiv:2503.11206) | ESC-10 | FC only | Rate, direct, latency | ~60% | None |
| Dominguez-Morales et al. 2016 (ICANN) | Pure tones | FC | Rate | ~90% | SpiNNaker |
| Dong et al. 2018 | TIMIT (speech) | CSNN | Rate | 66% | Simulation |

**Key gap:** no conv SNNs on full ESC-50, no SNN deployment for environmental sound on SpiNNaker.

### 2.4.2 closest prior work

**Larroza et al. (2025)** -- most directly relevant. SNN with FC layers on ESC-10, ~60% with direct encoding. Limitations: ESC-10 is much easier (10 vs 50 classes), FC only (no conv), no hardware deployment. We extend to full ESC-50 with conv architecture.

**Dominguez-Morales et al. (2016)** deployed SNNs on SpiNNaker for audio but used pure synthetic tones and silicon cochlea input. Task difficulty isnt comparable to ESC-50 at all.

---

## 2.5 SpiNNaker neuromorphic platform

**SpiNNaker** (Furber et al. 2014) -- massively parallel neuromorphic platform from UoM. Each chip has 18 ARM968 processors connected by custom packet-switched network. Communication is entirely spike-driven: each event is a 4-byte packet with neuron address, propagated asynchronously. AC-only communication (no multiply) is the source of energy efficiency.

**sPyNNaker** provides PyNN-compatible Python interface. Supported models: IF_curr_exp, IF_cond_exp, Izhikevich, HH. Membrane dynamics simulated in fixed-point on ARM cores.

Deployment challenges:
1. All weights and membrane potentials must be fixed-point representable
2. Input must be binary spikes (SpikeSourceArray or SpikeSourcePoisson)
3. Max network size limited by available ARM cores (~1000 neurons/chip)
4. Timing synchronous at 1ms resolution

For this work, FC2-only hybrid (256->50) fits comfortably on SpiNNaker.

---

## 2.6 energy efficiency: SNNs vs ANNs

The energy argument is nuanced and depends on the execution platform.

**On neuromorphic hardware** (Loihi, SpiNNaker, TrueNorth): AC-only communication. Bio neurons ~20 fJ/spike; CMOS ~0.9 pJ/AC at 45nm (Yik et al. 2025). MACs cost ~4.6 pJ at 45nm. The 5.1x ratio means SNNs are cheaper *if* spike rate is low enough.

**In software simulation** (GPU/CPU): SNNs run T timesteps while ANNs run once. Each timestep has same matrix ops, but SNN may benefit from sparsity. However T overhead typically dominates; software SNNs are generally less efficient (Dampfhoffer et al. 2023 show need <6.4% spike rate to beat quantized ANNs on CPU).

**NeuroBench** (Yik et al. 2025, Nature Comms 16:1589) provides standardised framework -- SynapticOperations metrics: Effective_ACs, Effective_MACs, Dense. This work uses NeuroBench v2.2.0 for all energy comparisons.

---

## 2.7 adversarial robustness in SNNs
Adversarial examples (Goodfellow et al. 2015) are inputs crafted to maximise model loss: $x' = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L})$ (FGSM). For SNNs, adversarial robustness is complex:

**Sharmin et al. (ECCV 2020, arXiv:2003.10399)** showed that SNNs with Poisson encoding exhibit higher adversarial accuracy in black-box scenarios, attributing this to the stochasticity of rate coding.

**Gradient masking in SNNs**: The spike threshold creates a hard non-linearity. Small perturbations to the input may not cross the threshold, leaving the spike pattern unchanged. This creates gradient masking — the gradient ∂S/∂x through the spike function is zero almost everywhere, making gradient-based attacks weaker.

**Important caveat (Wang et al. 2025, arXiv:2512.22522)**: Standard PGD (Projected Gradient Descent) underestimates SNN robustness because vanishing surrogate gradients make the attack weaker than it should be. Stable Adaptive PGD (SA-PGD) should be used for reliable evaluation. This work uses standard PGD (40 steps) with the caveat that SNN robustness numbers may be slightly inflated.

---

## 2.8 Transfer Learning for Audio (PANNs)

**PANNs** (Pretrained Audio Neural Networks, Kong et al. 2020, IEEE TASLP) are large CNN models pretrained on AudioSet (Google, 1.8M 10s clips, 527 class tags). CNN14 achieves 43.1% mAP on AudioSet and provides 2048-dimensional embeddings that transfer extremely well to downstream tasks: ESC-50 accuracy of 94.7% with CNN14 fine-tuning.

This work uses CNN14 embeddings as fixed features and trains only a lightweight SNN classification head, demonstrating that the SNN-ANN accuracy gap closes from 16.7 pp to <1 pp when rich pretrained features are available. This is the first combination of PANNs with an SNN classifier.

---

## 2.9 Continual Learning in SNNs

Catastrophic forgetting (McCloskey & Cohen 1989) — the tendency of neural networks to forget previously learned tasks when trained on new ones — is a major challenge for lifelong learning systems. SNNs have been proposed as more biologically plausible candidates for continual learning due to their sparse, local activation patterns.

**Golden et al. (2022, PLoS Computational Biology)** showed that offline consolidation (analogous to sleep) prevents catastrophic forgetting in SNNs by replaying patterns during quiescent periods.

**Zhang et al. (2023, Science Advances)** introduced NACA (Neuron Activation Consolidation Algorithm) for online continual learning in SNNs.

This work evaluates catastrophic forgetting in the simplest setting: sequential fine-tuning on 5 ESC-50 super-categories without any replay or regularisation mechanism. The SNN is compared directly to an identical ANN under the same sequential training protocol.
